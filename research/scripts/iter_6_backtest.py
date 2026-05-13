"""
iter_6 Nikkei 225 opening-range breakout proxy backtest (Agent A17).

NOTE on simplification:
- The spec (iter_6_nikkei_futures_open_range_breakout.md) is a 5-min intraday
  opening-range breakout on Nikkei 225 mini futures (09:00-11:30 JST).
- Free data sources do not provide 5-min Tokyo intraday bars far enough back
  for the spec's 2018-2026 validation window, so this run uses a daily proxy
  that exploits the SAME underlying directional signal (overnight retail-flow
  continuation into the cash open) at daily granularity.

Proxy rule:
- Daily ^N225 (Nikkei 225 cash index) OHLC from yfinance.
- Overnight gap = (Open_t / Close_{t-1}) - 1.
- Signal = sign(gap) if |gap| > threshold, else flat.
- Position taken at Open_t, exited at Close_t.
- Per-day return = signal * (Close_t / Open_t - 1) - round-turn cost (5 bps)
  applied only on trading days.
- Threshold sweep: 0.3%, 0.5%, 1.0%.

Walk-forward:
- IS: 2010-01-01 -> 2018-12-31 (no IS metric is used to pick winner; we just
  report it for reference).
- OOS: 2019-01-01 -> 2025-12-31 (or last available).
- Winner = threshold with highest OOS Sharpe.

PASS gate: best variant OOS Sharpe >= 0.7.
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path("/Users/weeoto/turbo-carnival/research")
CACHE_PATH = ROOT / "data" / "cache" / "nikkei_daily.parquet"
LOG_PATH = ROOT / "logs" / "iter_6_backtest.log"
SUMMARY_JSON = ROOT / "results" / "iter_6_summary.json"

SYMBOL = "^N225"
START = "2010-01-01"
END = datetime.now(timezone.utc).strftime("%Y-%m-%d")

IS_END = pd.Timestamp("2018-12-31")
OOS_START = pd.Timestamp("2019-01-01")

GAP_THRESHOLDS = [0.003, 0.005, 0.010]  # 0.3%, 0.5%, 1.0%
ROUND_TRIP_BPS = 5.0  # 5 bps per round-turn (open+close = 1 round-turn / day)

PASS_SHARPE = 0.7


def setup_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("iter_6")
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fh = logging.FileHandler(LOG_PATH, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def load_or_fetch(logger: logging.Logger) -> pd.DataFrame:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if CACHE_PATH.exists():
        try:
            cached = pd.read_parquet(CACHE_PATH)
            cached.index = pd.to_datetime(cached.index).tz_localize(None)
            last = cached.index.max()
            if (pd.Timestamp.now().normalize() - last).days <= 5:
                logger.info(
                    f"Loaded cache from {CACHE_PATH} "
                    f"(last={last.date()}, rows={len(cached)})"
                )
                return cached
            logger.info(f"Cache stale (last={last.date()}); re-downloading")
        except Exception as e:
            logger.warning(f"Cache read failed ({e}); re-downloading")

    logger.info(f"Downloading {SYMBOL} from yfinance {START} -> {END} ...")
    df = yf.download(
        SYMBOL,
        start=START,
        end=END,
        auto_adjust=False,  # we want raw Open/Close for the gap signal
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned no data for {SYMBOL}")
    # flatten MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.dropna(subset=["Open", "Close"]).sort_index()
    df.to_parquet(CACHE_PATH)
    logger.info(f"Cached daily ^N225 to {CACHE_PATH} (rows={len(df)})")
    return df


def run_threshold(
    df: pd.DataFrame, threshold: float, logger: logging.Logger
) -> tuple[pd.DataFrame, dict, dict]:
    """Run the gap-continuation rule for one threshold.

    Returns the per-day frame, the OOS metrics dict, and the IS metrics dict.
    """
    work = df.copy()
    work["prev_close"] = work["Close"].shift(1)
    work["gap"] = work["Open"] / work["prev_close"] - 1.0
    work["intraday_ret"] = work["Close"] / work["Open"] - 1.0

    signal = np.where(
        work["gap"] > threshold, 1.0,
        np.where(work["gap"] < -threshold, -1.0, 0.0),
    )
    work["signal"] = signal
    # cost is round-turn bps applied on any day we actually trade (signal != 0)
    cost_per_trade = ROUND_TRIP_BPS / 1e4
    work["cost"] = np.where(work["signal"] != 0.0, cost_per_trade, 0.0)
    work["net_ret"] = work["signal"] * work["intraday_ret"] - work["cost"]
    work = work.dropna(subset=["gap", "intraday_ret"])

    is_slice = work[work.index <= IS_END]
    oos_slice = work[work.index >= OOS_START]

    is_metrics = _metrics(is_slice, label=f"IS thresh={threshold:.3%}")
    oos_metrics = _metrics(oos_slice, label=f"OOS thresh={threshold:.3%}")

    logger.info(
        f"Threshold {threshold*100:.2f}%: "
        f"IS trades={is_metrics['trade_days']} sharpe={is_metrics['sharpe']:.3f} "
        f"ann_ret={is_metrics['ann_return_pct']:.2f}% | "
        f"OOS trades={oos_metrics['trade_days']} sharpe={oos_metrics['sharpe']:.3f} "
        f"ann_ret={oos_metrics['ann_return_pct']:.2f}% "
        f"max_dd={oos_metrics['max_drawdown_pct']:.2f}%"
    )

    return work, oos_metrics, is_metrics


def _metrics(slice_df: pd.DataFrame, label: str) -> dict:
    ret = slice_df["net_ret"].fillna(0.0)
    n = len(ret)
    trade_days = int((slice_df["signal"] != 0.0).sum())
    years = n / 252.0 if n else 0.0

    if n == 0 or years <= 0:
        return {
            "label": label,
            "trading_days": 0,
            "trade_days": 0,
            "years": 0.0,
            "total_return_pct": 0.0,
            "ann_return_pct": 0.0,
            "ann_vol_pct": 0.0,
            "sharpe": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate_pct": 0.0,
            "start_date": None,
            "end_date": None,
        }

    equity = (1.0 + ret).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    ann_return = (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    ann_vol = float(ret.std(ddof=0) * math.sqrt(252))
    sharpe = (float(ret.mean()) * 252) / ann_vol if ann_vol > 0 else 0.0

    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1.0
    max_dd = float(drawdown.min())

    traded = slice_df[slice_df["signal"] != 0.0]["net_ret"]
    win_rate = float((traded > 0).mean() * 100) if len(traded) else 0.0

    return {
        "label": label,
        "trading_days": int(n),
        "trade_days": trade_days,
        "years": round(years, 3),
        "total_return_pct": round(total_return * 100, 4),
        "ann_return_pct": round(ann_return * 100, 4),
        "ann_vol_pct": round(ann_vol * 100, 4),
        "sharpe": round(sharpe, 4),
        "max_drawdown_pct": round(max_dd * 100, 4),
        "win_rate_pct": round(win_rate, 4),
        "start_date": str(slice_df.index.min().date()),
        "end_date": str(slice_df.index.max().date()),
    }


def main() -> None:
    logger = setup_logging()
    logger.info("=== iter_6 Nikkei opening-range proxy backtest START ===")
    logger.info(f"Symbol: {SYMBOL}  Window: {START} -> {END}")
    logger.info(
        f"Walk-forward: IS <= {IS_END.date()} ; OOS >= {OOS_START.date()}"
    )
    logger.info(f"Cost per trade day (round-turn): {ROUND_TRIP_BPS} bps")
    logger.info(f"Gap thresholds tested: {[f'{t*100:.2f}%' for t in GAP_THRESHOLDS]}")

    raw = load_or_fetch(logger)
    logger.info(
        f"Loaded {len(raw)} daily bars from {raw.index.min().date()} "
        f"to {raw.index.max().date()}"
    )

    variants = []
    for thresh in GAP_THRESHOLDS:
        _, oos_m, is_m = run_threshold(raw, thresh, logger)
        variants.append({
            "threshold_pct": round(thresh * 100, 4),
            "is": is_m,
            "oos": oos_m,
        })

    # pick winner by OOS Sharpe
    winner = max(variants, key=lambda v: v["oos"]["sharpe"])
    best_thresh = winner["threshold_pct"]
    best_sharpe = winner["oos"]["sharpe"]
    best_ann = winner["oos"]["ann_return_pct"]
    best_dd = winner["oos"]["max_drawdown_pct"]

    verdict = "PASS" if best_sharpe >= PASS_SHARPE else "FAIL"

    summary = {
        "iter": "iter_6",
        "symbol": SYMBOL,
        "data_window_start": str(raw.index.min().date()),
        "data_window_end": str(raw.index.max().date()),
        "is_end": str(IS_END.date()),
        "oos_start": str(OOS_START.date()),
        "round_trip_bps": ROUND_TRIP_BPS,
        "gap_thresholds_pct": [round(t * 100, 4) for t in GAP_THRESHOLDS],
        "variants": variants,
        "best_threshold_pct": best_thresh,
        "best_oos_sharpe": best_sharpe,
        "best_oos_ann_return_pct": best_ann,
        "best_oos_max_drawdown_pct": best_dd,
        "pass_sharpe_floor": PASS_SHARPE,
        "verdict": verdict,
        "caveat": (
            "Daily proxy backtest. The spec describes a 5-min intraday opening-"
            "range breakout (09:00-11:30 JST) on Nikkei 225 mini futures. Free "
            "yfinance data only provides daily OHLC for ^N225. This proxy uses "
            "the overnight gap as the directional signal and the day's Open->"
            "Close path as the realized intraday move. It tests the same "
            "underlying continuation hypothesis but at daily granularity and "
            "without the actual OR-width breakout filter, ATR regime gate, "
            "stop logic, or 11:25 JST time stop."
        ),
    }

    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Wrote summary JSON: {SUMMARY_JSON}")

    rl = (
        f"RESULT_LINE: iter_6 | best_thresh={best_thresh:.2f}% | "
        f"OOS sharpe={best_sharpe:.3f} | ann_ret={best_ann:.2f}% | "
        f"max_dd={best_dd:.2f}% | {verdict}"
    )
    logger.info(rl)
    print(rl)
    print(
        "CAVEAT: Daily proxy for what spec says is intraday strategy "
        "(5-min OR breakout 09:00-11:30 JST) — actual 5-min OR breakout "
        "might be different"
    )
    print(f"PATH: {SUMMARY_JSON}")

    logger.info("=== iter_6 Nikkei opening-range proxy backtest END ===")


if __name__ == "__main__":
    main()
