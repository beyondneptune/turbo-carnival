"""
iter_7 — DAX Time-Series Momentum (TSMOM) Backtest

Spec: research/candidates/iter_7_dax_futures_tsmom.md
- Data: ^GDAXI daily (proxy for FDAX futures, since the index tracks futures closely
  on a return basis; futures carry no dividend leak the way the price index doesn't either,
  so total-return divergence is small).
- Signal: sign(1m) + sign(3m) + sign(12m), combined ∈ {-3,-1,+1,+3}
  - Long when combined >= +1
  - Short when combined <= -1
  - Flat when combined == 0
- Vol target 10% ann; size = target / realized_60d (computed BEFORE entering the bar,
  so we use a 1-day-lagged sigma to avoid look-ahead).
- Rebalance: Friday close; position constant Monday open through next Friday close.
- Cost: 1 bp per rebalance (turnover-based; only the *change* in position incurs cost).
- IS: 2000-2014, OOS: 2015-2025. PASS if OOS Sharpe >= 0.5.
"""

from __future__ import annotations
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ---------- paths ----------
ROOT = Path("/Users/weeoto/turbo-carnival/research")
CACHE = ROOT / "data" / "cache" / "dax_daily.parquet"
LOG = ROOT / "logs" / "iter_7_backtest.log"
PNL_CSV = ROOT / "results" / "iter_7_daily_pnl.csv"
SUMMARY_JSON = ROOT / "results" / "iter_7_summary.json"

# ---------- logging ----------
LOG.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG, mode="w"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("iter_7")


# ---------- data ----------
def load_dax() -> pd.DataFrame:
    """Pull ^GDAXI daily, cached as parquet."""
    if CACHE.exists():
        df = pd.read_parquet(CACHE)
        log.info("Loaded cache: %s rows=%d range=%s..%s",
                 CACHE.name, len(df), df.index[0].date(), df.index[-1].date())
        return df
    log.info("Downloading ^GDAXI from yfinance 2000-01-01 .. today")
    df = yf.download(
        "^GDAXI",
        start="2000-01-01",
        end=None,
        progress=False,
        auto_adjust=False,
    )
    if df is None or df.empty:
        raise RuntimeError("yfinance returned no data for ^GDAXI")
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance >=0.2 returns MultiIndex even for one ticker
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
    df.index = pd.to_datetime(df.index)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE)
    log.info("Cached %d rows -> %s", len(df), CACHE)
    return df


# ---------- backtest ----------
def run_backtest(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    px = df["Close"].astype(float).copy()
    log_ret = np.log(px / px.shift(1))

    # Three look-back sub-signals: sign of cumulative log return over h business days.
    # NOTE: r_{t,h} uses returns from t-h to t; signal is computed AT bar t close,
    # then applied to next-bar return (no look-ahead).
    r_1m = np.log(px / px.shift(21))
    r_3m = np.log(px / px.shift(63))
    r_12m = np.log(px / px.shift(252))

    s1 = np.sign(r_1m)
    s3 = np.sign(r_3m)
    s12 = np.sign(r_12m)
    combined = (s1 + s3 + s12)  # ∈ {-3,-1,+1,+3} (0 essentially never)

    # Discrete position direction from combined
    direction = pd.Series(0.0, index=px.index)
    direction[combined >= 1] = 1.0
    direction[combined <= -1] = -1.0
    # flat when combined == 0 (i.e. 1+1-1-1 type ties — impossible with 3 signs summing
    # to odd values, but keep the guard)

    # Realized 60d annualized vol, lagged by 1 day so today's sizing uses *yesterday's*
    # estimate — strictly no look-ahead.
    sigma_60 = log_ret.rolling(60, min_periods=40).std() * np.sqrt(252)
    sigma_60_lag = sigma_60.shift(1)

    target_vol = 0.10
    raw_size = target_vol / sigma_60_lag.replace(0, np.nan)
    # Cap leverage at 3.0x to avoid an unrealistic 10x size when sigma collapses.
    # This is a standard guardrail in TSMOM live deployment (see AQR Risk-Premia papers);
    # without it, sigma can dive to ~5% in 2017 and inflate size > 2x ambient.
    raw_size = raw_size.clip(upper=3.0)

    # Weekly rebalance on Friday close: position constant from Friday close → next Friday close.
    # Build target_pos at Friday timestamps, forward-fill to daily.
    target_pos_daily = direction * raw_size

    # Identify Fridays (weekday == 4). For non-trading Fridays, fall back to the last
    # trading day of that week.
    weekday = px.index.weekday
    is_friday = pd.Series(weekday, index=px.index) == 4
    # also include the last trading day of each ISO week as the "rebalance day"
    iso_year = px.index.isocalendar().year
    iso_week = px.index.isocalendar().week
    week_key = iso_year.astype(str) + "_" + iso_week.astype(str)
    last_in_week = pd.Series(week_key.values, index=px.index).groupby(week_key.values).cumcount(
        ascending=False
    ) == 0
    rebal_mask = (is_friday.values | last_in_week.values)

    position = pd.Series(np.nan, index=px.index)
    position[rebal_mask] = target_pos_daily[rebal_mask]
    position = position.ffill().fillna(0.0)
    # Shift by 1: a Friday-close rebalance only earns return from Monday onwards.
    position_applied = position.shift(1).fillna(0.0)

    # Strategy daily return = position * underlying simple return
    simple_ret = px.pct_change().fillna(0.0)
    gross_pnl = position_applied * simple_ret

    # Costs: 1 bp on the *change* in position (turnover), per rebalance.
    cost_bp = 0.0001
    turnover = position_applied.diff().abs().fillna(0.0)
    cost = turnover * cost_bp
    net_pnl = gross_pnl - cost

    # Audit
    n_long = int((position_applied > 0).sum())
    n_short = int((position_applied < 0).sum())
    n_flat = int((position_applied == 0).sum())

    out = pd.DataFrame({
        "close": px,
        "ret": simple_ret,
        "s1": s1,
        "s3": s3,
        "s12": s12,
        "combined": combined,
        "direction": direction,
        "sigma_60_lag": sigma_60_lag,
        "raw_size": raw_size,
        "position": position_applied,
        "turnover": turnover,
        "gross_pnl": gross_pnl,
        "cost": cost,
        "net_pnl": net_pnl,
    })

    counters = {"n_long": n_long, "n_short": n_short, "n_flat": n_flat}
    return out, counters


# ---------- stats ----------
def perf_stats(net_ret: pd.Series) -> dict:
    """Annualized return, vol, Sharpe, max drawdown."""
    net_ret = net_ret.dropna()
    if net_ret.empty:
        return dict(years=0.0, ann_return_pct=0.0, ann_vol_pct=0.0, sharpe=0.0,
                    max_drawdown_pct=0.0)
    n = len(net_ret)
    years = n / 252.0
    ann_ret = net_ret.mean() * 252
    ann_vol = net_ret.std(ddof=0) * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    # max drawdown on cumulative log returns of the simple-return stream
    cum = (1.0 + net_ret).cumprod()
    peak = cum.cummax()
    dd = cum / peak - 1.0
    max_dd = float(dd.min())
    return dict(
        years=float(years),
        ann_return_pct=float(ann_ret * 100),
        ann_vol_pct=float(ann_vol * 100),
        sharpe=float(sharpe),
        max_drawdown_pct=float(max_dd * 100),
    )


def main() -> int:
    log.info("=== iter_7 DAX TSMOM backtest ===")
    df = load_dax()
    bt, counters = run_backtest(df)

    PNL_CSV.parent.mkdir(parents=True, exist_ok=True)
    bt.to_csv(PNL_CSV, index_label="date")
    log.info("Wrote daily P&L: %s (rows=%d)", PNL_CSV, len(bt))

    # IS / OOS split
    is_mask = (bt.index >= "2000-01-01") & (bt.index <= "2014-12-31")
    oos_mask = (bt.index >= "2015-01-01") & (bt.index <= "2025-12-31")

    is_stats = perf_stats(bt.loc[is_mask, "net_pnl"])
    oos_stats = perf_stats(bt.loc[oos_mask, "net_pnl"])

    log.info("IS  (2000-2014): %s", is_stats)
    log.info("OOS (2015-2025): %s", oos_stats)
    log.info("Position counts: long=%d short=%d flat=%d",
             counters["n_long"], counters["n_short"], counters["n_flat"])

    # OOS-only position counts (for the summary)
    pos_oos = bt.loc[oos_mask, "position"]
    n_long_oos = int((pos_oos > 0).sum())
    n_short_oos = int((pos_oos < 0).sum())
    n_flat_oos = int((pos_oos == 0).sum())

    verdict = "PASS" if oos_stats["sharpe"] >= 0.5 else "FAIL"

    summary = {
        "instrument": "^GDAXI (DAX index, proxy for FDAX futures returns)",
        "period_full": f"{bt.index[0].date()}..{bt.index[-1].date()}",
        "is_window": "2000-2014",
        "oos_window": "2015-2025",
        "years_oos": oos_stats["years"],
        "ann_return_pct_oos": oos_stats["ann_return_pct"],
        "ann_vol_pct": oos_stats["ann_vol_pct"],
        "sharpe_oos": oos_stats["sharpe"],
        "max_drawdown_pct_oos": oos_stats["max_drawdown_pct"],
        "num_long_periods": n_long_oos,
        "num_short_periods": n_short_oos,
        "num_flat_periods": n_flat_oos,
        "ann_return_pct_is": is_stats["ann_return_pct"],
        "sharpe_is": is_stats["sharpe"],
        "max_drawdown_pct_is": is_stats["max_drawdown_pct"],
        "cost_bp_per_rebalance": 1.0,
        "vol_target_ann_pct": 10.0,
        "max_leverage_cap": 3.0,
        "verdict": verdict,
        "pass_threshold_sharpe_oos": 0.5,
    }

    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    log.info("Wrote summary: %s", SUMMARY_JSON)
    log.info("VERDICT: %s (OOS Sharpe=%.3f vs threshold 0.50)",
             verdict, oos_stats["sharpe"])

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
