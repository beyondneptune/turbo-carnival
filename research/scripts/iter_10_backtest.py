"""iter_10 backtest: Gold/Silver ratio (GLD/SLV) z-score mean reversion.

Strategy (per /Users/weeoto/turbo-carnival/research/candidates/iter_10_gold_silver_ratio_reversion.md):
  ratio_t = GLD_t / SLV_t
  rolling 252-day mean and std using strictly past data (shift by 1)
  z_t = (ratio_t - mu_t) / sigma_t
  Entry SHORT ratio (long SLV, short GLD) when z > +2.0 (and flat)
  Entry LONG ratio  (long GLD, short SLV) when z < -2.0 (and flat)
  Exit:
    |z| < 0.5  -> convergence
    |z| > 3.0  -> stop-out (note: spec text mentions 3.5; the task brief specifies 3.0; using brief)
    holding > 90 calendar days -> time stop
  Cost: 2 bp per side per leg = 8 bp per round-trip pair-trade
  Sizing: $50k per leg, $100k gross, dollar-neutral

P&L per round-trip is reported in bps of total notional ($100k).
Walk-forward: 2006-2016 IS, 2016-present OOS. PASS/FAIL gate is computed on OOS.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path("/Users/weeoto/turbo-carnival/research")
CACHE_PATH = ROOT / "data" / "cache" / "gold_silver_daily.parquet"
LOG_PATH = ROOT / "logs" / "iter_10_backtest.log"
TRADES_CSV = ROOT / "results" / "iter_10_trades.csv"
SUMMARY_JSON = ROOT / "results" / "iter_10_summary.json"

TICKERS = ["GLD", "SLV"]
START = "2006-01-01"
ROLL = 252
ENTRY_Z = 2.0
EXIT_Z = 0.5
STOP_Z = 3.0  # per task brief
HOLD_MAX_DAYS = 90
COST_BPS_PER_LEG = 2.0  # bp per side per leg
NOTIONAL_PER_LEG = 50_000.0  # $50k each leg
GROSS_NOTIONAL = 2 * NOTIONAL_PER_LEG  # $100k total

OOS_START = pd.Timestamp("2016-01-01")
IS_START = pd.Timestamp(START)
IS_END = pd.Timestamp("2015-12-31")


def setup_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(LOG_PATH, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("iter_10")


def fetch_prices(log: logging.Logger) -> pd.DataFrame:
    if CACHE_PATH.exists():
        df = pd.read_parquet(CACHE_PATH)
        # Refresh only if older than 1 day or doesn't extend to recent
        last = df.index.max()
        if last >= pd.Timestamp.today().normalize() - pd.Timedelta(days=4):
            log.info(f"Loaded cache {CACHE_PATH} rows={len(df)} last={last.date()}")
            return df
        log.info(f"Cache stale (last={last.date()}); refetching")

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Downloading yfinance: {TICKERS} from {START}")
    raw = yf.download(
        TICKERS,
        start=START,
        end=None,
        auto_adjust=True,  # adjusted closes
        progress=False,
        threads=True,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        # auto_adjust=True gives 'Close' as adjusted
        close = raw["Close"].copy()
    else:
        close = raw[["Close"]].copy()
        close.columns = TICKERS

    close = close.dropna(how="any")
    close.columns = [c.upper() for c in close.columns]
    close = close[TICKERS]
    close.index = pd.to_datetime(close.index)
    close.to_parquet(CACHE_PATH)
    log.info(f"Saved cache {CACHE_PATH} rows={len(close)} {close.index.min().date()}..{close.index.max().date()}")
    return close


def build_signals(prices: pd.DataFrame, log: logging.Logger) -> pd.DataFrame:
    df = prices.copy()
    df["ratio"] = df["GLD"] / df["SLV"]
    # Strictly past-data rolling (shift by 1 so today's mu/sigma use up to yesterday)
    past_ratio = df["ratio"].shift(1)
    df["mu"] = past_ratio.rolling(ROLL).mean()
    df["sigma"] = past_ratio.rolling(ROLL).std(ddof=1)
    df["z"] = (df["ratio"] - df["mu"]) / df["sigma"]
    n_valid = df["z"].notna().sum()
    log.info(f"Signals built: {n_valid} valid z-score rows (first valid={df['z'].first_valid_index()})")
    return df


def run_backtest(df: pd.DataFrame, log: logging.Logger) -> pd.DataFrame:
    """Simulate trades with daily granularity. Returns trades DataFrame."""
    trades: list[dict] = []
    state = "flat"  # 'flat', 'long_ratio', 'short_ratio'
    entry_idx = None
    entry_date = None
    entry_ratio = None
    entry_gld = None
    entry_slv = None
    entry_z = None
    # Direction: +1 = long ratio (long GLD, short SLV), -1 = short ratio (long SLV, short GLD)
    direction = 0
    # Pre-sizing: we hold a fixed dollar notional per leg at entry; share counts fixed for the trade.
    gld_shares = 0.0
    slv_shares = 0.0
    # Daily peak equity for drawdown tracking on the trade pnl curve (computed at the end)

    rows = df.dropna(subset=["z"]).copy()

    def close_trade(exit_date, exit_idx, exit_ratio, exit_gld, exit_slv, exit_z, reason):
        nonlocal state, entry_idx, entry_date, entry_ratio, entry_gld, entry_slv, entry_z, direction, gld_shares, slv_shares
        # P&L per leg in dollars (long-side positive, short-side negative on price-up)
        if direction == +1:  # long ratio: long GLD, short SLV
            gld_pnl = gld_shares * (exit_gld - entry_gld)
            slv_pnl = -slv_shares * (exit_slv - entry_slv)
        else:  # direction == -1: short ratio: long SLV, short GLD
            gld_pnl = -gld_shares * (exit_gld - entry_gld)
            slv_pnl = slv_shares * (exit_slv - entry_slv)
        gross_pnl = gld_pnl + slv_pnl
        # Costs: 2 bp per side per leg = 4 bp per leg round-trip = 8 bp on pair-trade total notional
        # Total cost = NOTIONAL_PER_LEG * 4 bp on each leg = 2 legs * NOTIONAL_PER_LEG * 4e-4
        cost_dollars = 2 * NOTIONAL_PER_LEG * (2 * COST_BPS_PER_LEG / 10_000.0)
        net_pnl = gross_pnl - cost_dollars
        bps = net_pnl / GROSS_NOTIONAL * 10_000.0  # bp of $100k notional
        holding_days = (exit_date - entry_date).days
        trade = {
            "entry_date": entry_date,
            "exit_date": exit_date,
            "direction": "long_ratio" if direction == +1 else "short_ratio",
            "entry_ratio": entry_ratio,
            "exit_ratio": exit_ratio,
            "entry_z": entry_z,
            "exit_z": exit_z,
            "entry_gld": entry_gld,
            "exit_gld": exit_gld,
            "entry_slv": entry_slv,
            "exit_slv": exit_slv,
            "gld_shares": gld_shares,
            "slv_shares": slv_shares,
            "gross_pnl": gross_pnl,
            "cost_dollars": cost_dollars,
            "net_pnl": net_pnl,
            "bps_of_notional": bps,
            "holding_days": holding_days,
            "exit_reason": reason,
            "period": "is" if exit_date < OOS_START else "oos",
        }
        trades.append(trade)
        state = "flat"
        entry_idx = None
        entry_date = None
        entry_ratio = None
        entry_gld = None
        entry_slv = None
        entry_z = None
        direction = 0
        gld_shares = 0.0
        slv_shares = 0.0

    for ts, row in rows.iterrows():
        z = row["z"]
        gld = row["GLD"]
        slv = row["SLV"]
        ratio = row["ratio"]

        if state == "flat":
            # Entry signals (act at close of signal day at that day's prices)
            if z > ENTRY_Z:
                # SHORT ratio: long SLV, short GLD, dollar-neutral
                direction = -1
                state = "short_ratio"
                entry_idx = ts
                entry_date = ts
                entry_ratio = ratio
                entry_gld = gld
                entry_slv = slv
                entry_z = z
                gld_shares = NOTIONAL_PER_LEG / gld
                slv_shares = NOTIONAL_PER_LEG / slv
            elif z < -ENTRY_Z:
                # LONG ratio: long GLD, short SLV
                direction = +1
                state = "long_ratio"
                entry_idx = ts
                entry_date = ts
                entry_ratio = ratio
                entry_gld = gld
                entry_slv = slv
                entry_z = z
                gld_shares = NOTIONAL_PER_LEG / gld
                slv_shares = NOTIONAL_PER_LEG / slv
        else:
            # In trade: check exits in priority order
            held_days = (ts - entry_date).days
            if abs(z) > STOP_Z:
                close_trade(ts, ts, ratio, gld, slv, z, "stop_out")
            elif abs(z) < EXIT_Z:
                close_trade(ts, ts, ratio, gld, slv, z, "convergence")
            elif held_days > HOLD_MAX_DAYS:
                close_trade(ts, ts, ratio, gld, slv, z, "time_stop")

    # Force-close any open trade at end of data (so we record it)
    if state != "flat":
        last_ts = rows.index[-1]
        last = rows.iloc[-1]
        close_trade(last_ts, last_ts, last["ratio"], last["GLD"], last["SLV"], last["z"], "end_of_data")

    trades_df = pd.DataFrame(trades)
    log.info(f"Total trades: {len(trades_df)}")
    if len(trades_df):
        log.info(f"  IS trades:  {(trades_df['period']=='is').sum()}")
        log.info(f"  OOS trades: {(trades_df['period']=='oos').sum()}")
    return trades_df


def max_drawdown_pct(equity: pd.Series) -> float:
    """Max drawdown of an equity curve, returned as a positive percent of peak."""
    if equity.empty:
        return 0.0
    running_peak = equity.cummax()
    dd = (equity - running_peak) / running_peak
    return float(-dd.min() * 100.0)


def summarize(trades_df: pd.DataFrame, log: logging.Logger) -> dict:
    oos = trades_df[trades_df["period"] == "oos"].copy()
    is_ = trades_df[trades_df["period"] == "is"].copy()

    summary: dict = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "spec": "iter_10_gold_silver_ratio_reversion",
        "params": {
            "tickers": TICKERS,
            "start": START,
            "rolling_window_days": ROLL,
            "entry_z": ENTRY_Z,
            "exit_z": EXIT_Z,
            "stop_z": STOP_Z,
            "hold_max_days": HOLD_MAX_DAYS,
            "cost_bps_per_leg_per_side": COST_BPS_PER_LEG,
            "notional_per_leg_usd": NOTIONAL_PER_LEG,
            "gross_notional_usd": GROSS_NOTIONAL,
            "oos_start": str(OOS_START.date()),
        },
        "total_trades_is": int(len(is_)),
        "total_trades_oos": int(len(oos)),
        "win_rate": None,
        "mean_bps_per_trade": None,
        "median_holding_days": None,
        "max_drawdown_pct": None,
        "verdict": "FAIL",
        "verdict_reason": "",
    }

    if len(oos) == 0:
        summary["verdict_reason"] = "no_oos_trades"
        return summary

    bps = oos["bps_of_notional"]
    win_rate = float((bps > 0).mean() * 100.0)
    mean_bps = float(bps.mean())
    median_bps = float(bps.median())
    median_hold = float(oos["holding_days"].median())
    mean_hold = float(oos["holding_days"].mean())

    # Build a simple per-trade equity curve (OOS only) for max DD.
    eq = (oos.sort_values("exit_date")["net_pnl"].cumsum() + GROSS_NOTIONAL).reset_index(drop=True)
    eq.index = pd.RangeIndex(len(eq))
    mdd = max_drawdown_pct(eq) if len(eq) else 0.0

    summary.update(
        {
            "win_rate": round(win_rate, 2),
            "mean_bps_per_trade": round(mean_bps, 2),
            "median_bps_per_trade": round(median_bps, 2),
            "median_holding_days": round(median_hold, 2),
            "mean_holding_days": round(mean_hold, 2),
            "max_drawdown_pct": round(mdd, 2),
            "exit_reason_counts_oos": oos["exit_reason"].value_counts().to_dict(),
            "direction_counts_oos": oos["direction"].value_counts().to_dict(),
        }
    )

    pass_gate = (mean_bps >= 200.0) and (win_rate >= 50.0)
    summary["verdict"] = "PASS" if pass_gate else "FAIL"
    summary["verdict_reason"] = (
        f"mean_bps={mean_bps:.1f} (req>=200), win_rate={win_rate:.1f}% (req>=50)"
    )

    log.info(f"OOS summary: trades={len(oos)} mean_bps={mean_bps:.2f} median_bps={median_bps:.2f} "
             f"win_rate={win_rate:.2f}% median_hold={median_hold:.1f}d max_dd={mdd:.2f}% "
             f"verdict={summary['verdict']}")
    return summary


def main() -> int:
    log = setup_logging()
    log.info("=== iter_10 backtest start ===")
    prices = fetch_prices(log)
    df = build_signals(prices, log)
    trades_df = run_backtest(df, log)
    TRADES_CSV.parent.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(TRADES_CSV, index=False)
    log.info(f"Wrote {TRADES_CSV}")

    summary = summarize(trades_df, log)
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, default=str))
    log.info(f"Wrote {SUMMARY_JSON}")
    log.info("=== iter_10 backtest done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
