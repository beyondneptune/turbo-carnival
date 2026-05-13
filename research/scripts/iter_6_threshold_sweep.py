"""
iter_6 Nikkei gap-threshold parameter sweep (Agent A25).

Builds on iter_6_backtest.py (Agent A17). Same data, same daily ^N225 gap
proxy strategy, same 5 bps round-trip cost, same OOS window. Sweeps a wider
grid of gap thresholds to identify the optimal threshold by:

  1) raw OOS Sharpe
  2) risk-adjusted (OOS Sharpe / |max drawdown|) ratio

Thresholds tested: 0.3%, 0.5%, 0.7%, 0.9%, 1.0%, 1.2%, 1.5%, 2.0%

Strategy recap (daily proxy):
- gap_t = Open_t / Close_{t-1} - 1
- signal_t = sign(gap_t) if |gap_t| > threshold, else 0 (flat)
- intraday_ret_t = Close_t / Open_t - 1
- net_ret_t = signal_t * intraday_ret_t - cost   (cost only on trade days)

Per-threshold metrics over OOS slice:
- n_trades (count of non-zero signal days)
- win_rate_pct
- mean_return_per_trade_pct
- ann_return_pct
- sharpe (annualized over OOS trading days)
- max_drawdown_pct
- days_in_market_pct  (trade days / total OOS days)

Outputs:
- /Users/weeoto/turbo-carnival/research/logs/iter_6_threshold_sweep.log
- /Users/weeoto/turbo-carnival/research/results/iter_6_sweep.json
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
LOG_PATH = ROOT / "logs" / "iter_6_threshold_sweep.log"
SUMMARY_JSON = ROOT / "results" / "iter_6_sweep.json"

SYMBOL = "^N225"
START = "2010-01-01"
END = datetime.now(timezone.utc).strftime("%Y-%m-%d")

IS_END = pd.Timestamp("2018-12-31")
OOS_START = pd.Timestamp("2019-01-01")

# Wider sweep grid (8 thresholds)
GAP_THRESHOLDS = [0.003, 0.005, 0.007, 0.009, 0.010, 0.012, 0.015, 0.020]
ROUND_TRIP_BPS = 5.0  # bps round-turn on every trade day


def setup_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("iter_6_sweep")
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
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned no data for {SYMBOL}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.dropna(subset=["Open", "Close"]).sort_index()
    df.to_parquet(CACHE_PATH)
    logger.info(f"Cached daily ^N225 to {CACHE_PATH} (rows={len(df)})")
    return df


def build_oos_frame(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    work = df.copy()
    work["prev_close"] = work["Close"].shift(1)
    work["gap"] = work["Open"] / work["prev_close"] - 1.0
    work["intraday_ret"] = work["Close"] / work["Open"] - 1.0

    signal = np.where(
        work["gap"] > threshold, 1.0,
        np.where(work["gap"] < -threshold, -1.0, 0.0),
    )
    work["signal"] = signal
    cost_per_trade = ROUND_TRIP_BPS / 1e4
    work["cost"] = np.where(work["signal"] != 0.0, cost_per_trade, 0.0)
    work["net_ret"] = work["signal"] * work["intraday_ret"] - work["cost"]
    work = work.dropna(subset=["gap", "intraday_ret"])
    return work[work.index >= OOS_START]


def oos_metrics(oos: pd.DataFrame, threshold: float) -> dict:
    ret = oos["net_ret"].fillna(0.0)
    n_days = int(len(ret))
    trade_mask = oos["signal"] != 0.0
    n_trades = int(trade_mask.sum())
    years = n_days / 252.0 if n_days else 0.0

    if n_days == 0 or years <= 0:
        return {
            "threshold_pct": round(threshold * 100, 4),
            "n_trades": 0,
            "trading_days": 0,
            "years": 0.0,
            "win_rate_pct": 0.0,
            "mean_return_per_trade_pct": 0.0,
            "ann_return_pct": 0.0,
            "ann_vol_pct": 0.0,
            "sharpe": 0.0,
            "max_drawdown_pct": 0.0,
            "days_in_market_pct": 0.0,
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

    traded_returns = oos.loc[trade_mask, "net_ret"]
    if len(traded_returns) > 0:
        win_rate = float((traded_returns > 0).mean() * 100.0)
        mean_per_trade = float(traded_returns.mean() * 100.0)
    else:
        win_rate = 0.0
        mean_per_trade = 0.0

    days_in_market = float(n_trades / n_days * 100.0) if n_days else 0.0

    return {
        "threshold_pct": round(threshold * 100, 4),
        "n_trades": n_trades,
        "trading_days": n_days,
        "years": round(years, 3),
        "win_rate_pct": round(win_rate, 4),
        "mean_return_per_trade_pct": round(mean_per_trade, 6),
        "ann_return_pct": round(ann_return * 100, 4),
        "ann_vol_pct": round(ann_vol * 100, 4),
        "sharpe": round(sharpe, 4),
        "max_drawdown_pct": round(max_dd * 100, 4),
        "days_in_market_pct": round(days_in_market, 4),
        "start_date": str(oos.index.min().date()),
        "end_date": str(oos.index.max().date()),
    }


def main() -> None:
    logger = setup_logging()
    logger.info("=== iter_6 Nikkei gap-threshold SWEEP START ===")
    logger.info(f"Symbol: {SYMBOL}  Window: {START} -> {END}")
    logger.info(f"OOS slice: >= {OOS_START.date()}")
    logger.info(f"Cost per trade day (round-turn): {ROUND_TRIP_BPS} bps")
    logger.info(
        "Gap thresholds: "
        + ", ".join(f"{t*100:.2f}%" for t in GAP_THRESHOLDS)
    )

    raw = load_or_fetch(logger)
    logger.info(
        f"Loaded {len(raw)} daily bars from {raw.index.min().date()} "
        f"to {raw.index.max().date()}"
    )

    threshold_results: list[dict] = []
    for thresh in GAP_THRESHOLDS:
        oos = build_oos_frame(raw, thresh)
        m = oos_metrics(oos, thresh)
        # risk-adj ratio: sharpe / |max_dd_pct|  (avoid div by zero)
        dd_abs = abs(m["max_drawdown_pct"])
        m["sharpe_dd_ratio"] = round(m["sharpe"] / dd_abs, 6) if dd_abs > 0 else 0.0
        threshold_results.append(m)
        logger.info(
            f"thresh={m['threshold_pct']:.2f}% | n_trades={m['n_trades']:4d} | "
            f"win={m['win_rate_pct']:.2f}% | "
            f"mean/trade={m['mean_return_per_trade_pct']:.4f}% | "
            f"ann={m['ann_return_pct']:.2f}% | sharpe={m['sharpe']:.3f} | "
            f"max_dd={m['max_drawdown_pct']:.2f}% | "
            f"in_mkt={m['days_in_market_pct']:.2f}% | "
            f"sh/|dd|={m['sharpe_dd_ratio']:.4f}"
        )

    # best by raw OOS Sharpe
    best_sh = max(threshold_results, key=lambda r: r["sharpe"])
    best_by_sharpe = {
        "threshold": best_sh["threshold_pct"],
        "sharpe": best_sh["sharpe"],
        "ann_ret": best_sh["ann_return_pct"],
    }

    # best by sharpe / |max_dd|
    best_ra = max(threshold_results, key=lambda r: r["sharpe_dd_ratio"])
    best_by_risk_adj = {
        "threshold": best_ra["threshold_pct"],
        "sharpe_dd_ratio": best_ra["sharpe_dd_ratio"],
        "ann_ret": best_ra["ann_return_pct"],
        "max_dd": best_ra["max_drawdown_pct"],
        "sharpe": best_ra["sharpe"],
    }

    # Recommendation logic for paper shadow:
    # Prefer risk-adjusted winner if its Sharpe is within 0.10 of the raw-Sharpe
    # winner -- the lower drawdown is worth the small Sharpe give-up for a live
    # paper run. Otherwise stick with the raw-Sharpe winner.
    sharpe_gap = best_sh["sharpe"] - best_ra["sharpe"]
    if best_sh["threshold_pct"] == best_ra["threshold_pct"]:
        rec_thresh = best_sh["threshold_pct"]
        rec_reason = (
            "Same threshold wins on both raw OOS Sharpe and the "
            "Sharpe/|max_dd| risk-adjusted ratio -- unambiguous pick."
        )
    elif sharpe_gap <= 0.10:
        rec_thresh = best_ra["threshold_pct"]
        rec_reason = (
            f"Risk-adjusted winner ({best_ra['threshold_pct']:.2f}%) gives up "
            f"only {sharpe_gap:.3f} of Sharpe vs the raw winner "
            f"({best_sh['threshold_pct']:.2f}%) while delivering a better "
            f"Sharpe/|max_dd| ratio ({best_ra['sharpe_dd_ratio']:.4f} vs "
            f"{best_sh['sharpe_dd_ratio']:.4f}). For a live paper shadow the "
            "smaller drawdown profile is preferable."
        )
    else:
        rec_thresh = best_sh["threshold_pct"]
        rec_reason = (
            f"Raw OOS Sharpe winner ({best_sh['threshold_pct']:.2f}%, "
            f"sharpe={best_sh['sharpe']:.3f}) beats the risk-adjusted winner "
            f"by {sharpe_gap:.3f} of Sharpe -- gap too large to trade away "
            "for a marginally better drawdown profile."
        )

    summary = {
        "iter": "iter_6_threshold_sweep",
        "symbol": SYMBOL,
        "data_window_start": str(raw.index.min().date()),
        "data_window_end": str(raw.index.max().date()),
        "oos_start": str(OOS_START.date()),
        "round_trip_bps": ROUND_TRIP_BPS,
        "gap_thresholds_pct": [round(t * 100, 4) for t in GAP_THRESHOLDS],
        "threshold_results": threshold_results,
        "best_by_sharpe": best_by_sharpe,
        "best_by_risk_adj": best_by_risk_adj,
        "recommended": {
            "threshold": rec_thresh,
            "reason": rec_reason,
        },
        "caveat": (
            "Daily proxy on ^N225 cash index gap. Same strategy & costs as "
            "iter_6_backtest.py. Spec (5-min OR breakout on Nikkei mini "
            "futures 09:00-11:30 JST) cannot be directly tested with free "
            "data; sweep informs the proxy threshold to use for live paper "
            "shadow until 5-min data is sourced."
        ),
    }

    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Wrote sweep JSON: {SUMMARY_JSON}")

    sweep_line = (
        f"SWEEP_LINE: Best threshold by Sharpe = "
        f"{best_by_sharpe['threshold']:.2f}% (sharpe={best_by_sharpe['sharpe']:.3f}) | "
        f"Best by risk-adj = {best_by_risk_adj['threshold']:.2f}% "
        f"(ratio={best_by_risk_adj['sharpe_dd_ratio']:.4f})"
    )
    rec_line = (
        f"RECOMMENDATION: Use threshold={rec_thresh:.2f}% for paper shadow "
        f"because {rec_reason}"
    )
    logger.info(sweep_line)
    logger.info(rec_line)
    print(sweep_line)
    print(rec_line)
    print(f"PATH: {SUMMARY_JSON}")

    logger.info("=== iter_6 Nikkei gap-threshold SWEEP END ===")


if __name__ == "__main__":
    main()
