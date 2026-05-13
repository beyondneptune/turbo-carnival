"""
iter_5 VIX term-structure SVXY rotation backtest (Agent A4 re-verification).

Strategy summary (from spec):
- Daily signal: R = VIX / VIX3M (fallback VXV) at close.
- If R < 0.92 -> 25% NAV SVXY + 75% SGOV.
- Else -> 100% SGOV.
- Circuit breakers (force flat): VIX > 30 intraday OR VIX 1-day change > +50%.
- Cost: 1 bp per rotation event (per leg of the SVXY weight change).
- Period: 2018-01-01 (post -0.5x leverage change) to today.
- PASS if Sharpe >= 0.5 AND ann_return >= 8%.
- REPRODUCES_PRIOR if ann_return in [12%, 22%] AND Sharpe in [0.4, 0.9].

This script is intentionally simple and audit-friendly. It is the script
that is supposed to either back the live deploy tomorrow, or break it.
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
CACHE_PATH = ROOT / "data" / "cache" / "vix_svxy_daily.parquet"
LOG_PATH = ROOT / "logs" / "iter_5_backtest.log"
DAILY_PORTFOLIO_CSV = ROOT / "results" / "iter_5_daily_portfolio.csv"
SUMMARY_JSON = ROOT / "results" / "iter_5_summary.json"

START = "2018-01-01"
END = datetime.now(timezone.utc).strftime("%Y-%m-%d")  # today

RATIO_THRESHOLD = 0.92
SVXY_WEIGHT_ON = 0.25
SGOV_WEIGHT_ON = 1.0 - SVXY_WEIGHT_ON
COST_PER_ROTATION_BPS = 1.0  # 1 bp per rotation, applied to traded notional
CIRCUIT_VIX_LEVEL = 30.0
CIRCUIT_VIX_1D_CHG = 0.50  # +50% day-over-day in VIX spot

TICKERS = {
    "VIX": "^VIX",
    "VIX3M": "^VIX3M",
    "VXV": "^VXV",  # fallback for VIX3M
    "SVXY": "SVXY",
    "VXX": "VXX",
    "SGOV": "SGOV",
}


def setup_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("iter_5")
    logger.setLevel(logging.INFO)
    # clear any prior handlers (so re-runs don't double-log)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fh = logging.FileHandler(LOG_PATH, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _download_one(symbol: str, logger: logging.Logger) -> pd.Series:
    logger.info(f"Downloading {symbol} from yfinance ...")
    df = yf.download(
        symbol,
        start=START,
        end=END,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        logger.warning(f"yfinance returned empty for {symbol}")
        return pd.Series(dtype=float, name=symbol)
    # yfinance may return MultiIndex columns when threads >1; we set threads=False
    if isinstance(df.columns, pd.MultiIndex):
        # take the Close level
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
    else:
        close = df["Close"]
    close.name = symbol
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.dropna()


def load_or_fetch(logger: logging.Logger) -> pd.DataFrame:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if CACHE_PATH.exists():
        try:
            cached = pd.read_parquet(CACHE_PATH)
            cached.index = pd.to_datetime(cached.index).tz_localize(None)
            # only re-use cache if it covers up to within 5 days of today
            last = cached.index.max()
            if (pd.Timestamp.now().normalize() - last).days <= 5:
                logger.info(f"Loaded cache from {CACHE_PATH} (last={last.date()}, rows={len(cached)})")
                return cached
            else:
                logger.info(f"Cache stale (last={last.date()}); re-downloading")
        except Exception as e:
            logger.warning(f"Cache read failed ({e}); re-downloading")

    series = {}
    for key, sym in TICKERS.items():
        s = _download_one(sym, logger)
        if not s.empty:
            series[key] = s
        else:
            logger.warning(f"{key} ({sym}) returned no data")

    df = pd.concat(series, axis=1)
    df.columns = list(series.keys())
    df = df.sort_index()
    df.to_parquet(CACHE_PATH)
    logger.info(f"Cached merged frame to {CACHE_PATH} (rows={len(df)})")
    return df


def build_panel(df: pd.DataFrame, logger: logging.Logger) -> tuple[pd.DataFrame, str]:
    """Return aligned trading-day panel and which 3M series we used."""
    cols_present = [c for c in ["VIX", "VIX3M", "VXV", "SVXY", "VXX", "SGOV"] if c in df.columns]
    logger.info(f"Columns present in raw frame: {cols_present}")

    # pick VIX3M; fall back to VXV
    if "VIX3M" in df.columns and df["VIX3M"].notna().sum() > 250:
        vix3m_col = "VIX3M"
    elif "VXV" in df.columns and df["VXV"].notna().sum() > 250:
        vix3m_col = "VXV"
        logger.warning("^VIX3M unavailable or sparse; falling back to ^VXV for 3M VIX")
    else:
        raise RuntimeError("Neither ^VIX3M nor ^VXV available with enough data")

    needed = {"VIX": df["VIX"], "VIX3M_eff": df[vix3m_col], "SVXY": df["SVXY"], "SGOV": df.get("SGOV")}
    panel = pd.concat(needed, axis=1)
    panel.columns = ["VIX", "VIX3M_eff", "SVXY", "SGOV"]

    # SGOV only began trading 2020-05; for pre-SGOV days, fall back to a flat 0% cash return.
    # We will compute daily returns and substitute 0 where SGOV is NaN.
    panel = panel.sort_index()

    # drop weekend / holiday rows where VIX is NaN (CBOE closed)
    panel = panel[panel["VIX"].notna() & panel["VIX3M_eff"].notna() & panel["SVXY"].notna()]
    logger.info(f"Aligned panel rows={len(panel)} from {panel.index.min().date()} to {panel.index.max().date()}")
    logger.info(f"Using 3M series: {vix3m_col}")
    return panel, vix3m_col


def run_backtest(panel: pd.DataFrame, logger: logging.Logger) -> tuple[pd.DataFrame, dict]:
    """Backtest the rotation. All weights are decided at close on day t using
    information available at close on day t (ratio computed from that close).
    Returns from day t to day t+1 are then applied to the day-t target weights.
    This is the most lenient (and most common) close-to-close framing.
    """
    df = panel.copy()
    df["SVXY_ret"] = df["SVXY"].pct_change()
    if df["SGOV"].notna().any():
        df["SGOV_ret"] = df["SGOV"].pct_change()
    else:
        df["SGOV_ret"] = 0.0
    df["SGOV_ret"] = df["SGOV_ret"].fillna(0.0)

    df["ratio"] = df["VIX"] / df["VIX3M_eff"]
    df["vix_chg"] = df["VIX"].pct_change()

    # raw signal: ratio < threshold means go on SVXY
    contango = df["ratio"] < RATIO_THRESHOLD

    # circuit breaker: VIX > 30 OR VIX 1-day change > +50%
    cb_level = df["VIX"] > CIRCUIT_VIX_LEVEL
    cb_jump = df["vix_chg"] > CIRCUIT_VIX_1D_CHG
    cb_any = cb_level | cb_jump
    df["circuit_breaker"] = cb_any

    target_svxy = np.where(contango & ~cb_any, SVXY_WEIGHT_ON, 0.0)
    df["target_svxy_w"] = target_svxy
    df["target_sgov_w"] = 1.0 - df["target_svxy_w"]

    # The position used to earn day t's market return is the target set at close of day t-1.
    df["pos_svxy_w"] = df["target_svxy_w"].shift(1).fillna(0.0)
    df["pos_sgov_w"] = df["target_sgov_w"].shift(1).fillna(1.0)

    # rotation cost: 1 bp on traded SVXY notional change (per leg). Applied on the day the
    # new target takes effect (i.e. the day after the signal is read, which is when the
    # broker would fill the rotation in the MOC scenario).
    weight_chg = df["pos_svxy_w"].diff().abs().fillna(df["pos_svxy_w"].abs())
    df["rotation_cost"] = weight_chg * (COST_PER_ROTATION_BPS / 1e4)

    # daily portfolio gross return
    df["gross_ret"] = df["pos_svxy_w"] * df["SVXY_ret"].fillna(0.0) + df["pos_sgov_w"] * df["SGOV_ret"].fillna(0.0)
    df["net_ret"] = df["gross_ret"] - df["rotation_cost"]

    # portfolio value
    df["portfolio"] = (1.0 + df["net_ret"].fillna(0.0)).cumprod()

    # rotation event: any change in target_svxy_w from prior day (post-shift this is pos_svxy_w diff)
    df["is_rotation"] = (df["pos_svxy_w"].diff().fillna(0.0).abs() > 1e-9).astype(int)

    # metrics
    daily_ret = df["net_ret"].dropna()
    n = len(daily_ret)
    years = n / 252.0
    total_return = df["portfolio"].iloc[-1] / df["portfolio"].iloc[0] - 1.0 if len(df) else 0.0
    ann_return = (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    ann_vol = daily_ret.std(ddof=0) * math.sqrt(252) if n else 0.0
    sharpe = (daily_ret.mean() * 252) / ann_vol if ann_vol > 0 else 0.0

    rolling_max = df["portfolio"].cummax()
    drawdown = df["portfolio"] / rolling_max - 1.0
    max_dd = drawdown.min()

    rotations = int(df["is_rotation"].sum())
    rotations_per_year = rotations / years if years > 0 else 0.0
    cb_events = int(df["circuit_breaker"].sum())

    # PASS / REPRODUCES_PRIOR verdict
    passes = (sharpe >= 0.5) and (ann_return >= 0.08)
    reproduces_prior = (0.12 <= ann_return <= 0.22) and (0.4 <= sharpe <= 0.9)

    if passes and reproduces_prior:
        verdict = "PASS_AND_REPRODUCES_PRIOR"
    elif passes:
        verdict = "PASS"
    elif reproduces_prior:
        verdict = "REPRODUCES_PRIOR"
    else:
        verdict = "FAIL"

    summary = {
        "start_date": str(df.index.min().date()),
        "end_date": str(df.index.max().date()),
        "years": round(years, 3),
        "trading_days": int(n),
        "total_return_pct": round(total_return * 100, 4),
        "ann_return_pct": round(ann_return * 100, 4),
        "ann_vol_pct": round(ann_vol * 100, 4),
        "sharpe": round(sharpe, 4),
        "max_drawdown_pct": round(max_dd * 100, 4),
        "num_rotations": rotations,
        "num_rotations_per_year": round(rotations_per_year, 3),
        "num_circuit_breaker_events": cb_events,
        "ratio_threshold": RATIO_THRESHOLD,
        "svxy_weight_on": SVXY_WEIGHT_ON,
        "rotation_cost_bps": COST_PER_ROTATION_BPS,
        "circuit_vix_level": CIRCUIT_VIX_LEVEL,
        "circuit_vix_1d_chg": CIRCUIT_VIX_1D_CHG,
        "verdict": verdict,
        "prior_user_ann_return_pct": 17.1,
        "prior_user_sharpe": 0.66,
    }

    return df, summary


def main() -> None:
    logger = setup_logging()
    logger.info("=== iter_5 VIX term-structure backtest START ===")
    logger.info(f"Window: {START} to {END}")
    logger.info(f"Threshold: ratio < {RATIO_THRESHOLD} -> {int(SVXY_WEIGHT_ON*100)}% SVXY, else 100% SGOV")
    logger.info(f"Circuit breakers: VIX > {CIRCUIT_VIX_LEVEL} OR VIX 1d > +{int(CIRCUIT_VIX_1D_CHG*100)}%")
    logger.info(f"Rotation cost: {COST_PER_ROTATION_BPS} bp per traded notional")

    raw = load_or_fetch(logger)
    panel, vix3m_col = build_panel(raw, logger)

    bt, summary = run_backtest(panel, logger)
    summary["vix3m_source"] = vix3m_col

    # save outputs
    DAILY_PORTFOLIO_CSV.parent.mkdir(parents=True, exist_ok=True)
    out = bt[[
        "VIX", "VIX3M_eff", "ratio", "vix_chg", "circuit_breaker",
        "SVXY", "SGOV", "SVXY_ret", "SGOV_ret",
        "target_svxy_w", "pos_svxy_w", "is_rotation",
        "rotation_cost", "gross_ret", "net_ret", "portfolio",
    ]]
    out.to_csv(DAILY_PORTFOLIO_CSV, index_label="date")
    logger.info(f"Wrote daily portfolio CSV: {DAILY_PORTFOLIO_CSV} ({len(out)} rows)")

    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Wrote summary JSON: {SUMMARY_JSON}")

    # console-friendly result line
    rl = (
        f"RESULT_LINE: iter_5 | ann_ret={summary['ann_return_pct']:.2f}% | "
        f"sharpe={summary['sharpe']:.2f} | max_dd={summary['max_drawdown_pct']:.2f}% | "
        f"rotations/yr={summary['num_rotations_per_year']:.2f} | {summary['verdict']}"
    )
    logger.info(rl)
    print(rl)

    prior_ann = summary["prior_user_ann_return_pct"]
    prior_sharpe = summary["prior_user_sharpe"]
    ann_within = abs(summary["ann_return_pct"] - prior_ann) <= 0.25 * abs(prior_ann)
    sharpe_within = abs(summary["sharpe"] - prior_sharpe) <= 0.25 * abs(prior_sharpe)
    if ann_within and sharpe_within:
        comp = "REPRODUCES (within +/-25%)"
    else:
        comp = "DIVERGES significantly (outside +/-25%)"
    cmp_line = (
        f"COMPARISON_TO_PRIOR: prior=+17.1%/Sharpe 0.66; this run="
        f"{summary['ann_return_pct']:.2f}%/Sharpe {summary['sharpe']:.2f} -> {comp}"
    )
    logger.info(cmp_line)
    print(cmp_line)
    print(f"PATH: {SUMMARY_JSON}")

    logger.info("=== iter_5 VIX term-structure backtest END ===")


if __name__ == "__main__":
    main()
