"""Phase 3 backtest for MFTv1 — Opening Range Breakout on liquid US ETFs.

Tests 324 variants of ORB across SPY/QQQ/IWM/DIA/MDY 5-min bars 2017-2025,
with full multi-test correction (White RC + PBO + DSR + Harvey-Liu-Zhu t > 3)
and comparison against SPY buy-and-hold (the only meaningful benchmark for
intraday strategies).

Run:
    .venv/bin/python -m research.mft_v1.backtest --quick  # 12 variants
    .venv/bin/python -m research.mft_v1.backtest          # full 324
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from research.candidate_v1.backtest import (
    deflated_sharpe, whites_reality_check_bootstrap,
    pbo_combinatorial, metrics,
)

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "cache" / "mft_v1"
RESULTS_DIR = ROOT / "research" / "mft_v1" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSE = ["SPY", "QQQ", "IWM", "DIA", "MDY"]
TRAIN_END = pd.Timestamp("2022-12-31").tz_localize("UTC")
OOS_START = pd.Timestamp("2023-01-01").tz_localize("UTC")

COST_PER_SIDE_BPS = 5.0  # round-trip ~ 10 bps for liquid ETFs
SLIPPAGE_BPS_PER_SIDE = 1.0  # added on top of spread, for retail at-market

VARIANT_GRID = {
    "OR_DURATION_MIN": [15, 30, 60],
    "STOP_BUFFER_BPS": [0, 5, 10],
    "TARGET_MULT": [1.0, 1.5, 2.0],
    "POSITION_SIZE_PCT": [0.10, 0.20],
    "ALLOW_SHORT": [True, False],
    "MIN_RANGE_WIDTH_BPS": [25, 50, 100],
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("mft_v1.backtest")


@dataclass
class ORBVariant:
    or_dur: int
    stop_bps: int
    target_mult: float
    size: float
    allow_short: bool
    min_range_bps: int

    def key(self) -> str:
        return (f"D{self.or_dur}_S{self.stop_bps}_T{self.target_mult}"
                f"_P{self.size}_X{'BOTH' if self.allow_short else 'LONG'}"
                f"_W{self.min_range_bps}")


def all_variants() -> list[ORBVariant]:
    return [ORBVariant(*c) for c in product(*VARIANT_GRID.values())]


def load_bars(symbol: str) -> pd.DataFrame:
    df = pd.read_parquet(CACHE_DIR / f"{symbol}.parquet")
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    return df.sort_values("ts_utc").reset_index(drop=True)


def simulate_orb_one_ticker(bars: pd.DataFrame, v: ORBVariant,
                            symbol: str) -> pd.DataFrame:
    """Simulate ORB on one ticker. Returns per-day P&L records."""
    bars = bars.copy()
    bars["session_date"] = pd.to_datetime(bars["ts_utc"]).dt.tz_convert("America/New_York").dt.date
    bars["bar_time_et"] = pd.to_datetime(bars["ts_utc"]).dt.tz_convert("America/New_York").dt.strftime("%H:%M")
    records = []

    for session, day_bars in bars.groupby("session_date"):
        day_bars = day_bars.reset_index(drop=True)
        if len(day_bars) < 20:
            continue  # skip partial sessions

        # Define OR window
        or_end_mins = 30 + v.or_dur if v.or_dur < 60 else 90  # 09:30 + dur
        or_end_time = f"{9 + or_end_mins // 60:02d}:{(30 + or_end_mins) % 60:02d}"
        # Actually simpler: use first N bars at 5-min intervals
        n_or_bars = v.or_dur // 5
        or_bars = day_bars.iloc[:n_or_bars]
        if len(or_bars) < n_or_bars:
            continue
        or_high = or_bars["high"].max()
        or_low = or_bars["low"].min()
        or_width = or_high - or_low
        or_mid = (or_high + or_low) / 2

        # Skip narrow-range days
        width_bps = (or_width / or_mid) * 1e4
        if width_bps < v.min_range_bps:
            continue

        # Trade window: after OR bars, before 15:55 ET
        trade_bars = day_bars.iloc[n_or_bars:]
        if len(trade_bars) < 5:
            continue

        position = 0  # 0 flat, +1 long, -1 short
        entry_price = 0.0
        entry_idx = -1
        stop_price = 0.0
        target_price = 0.0
        pnl_pct = 0.0

        buf = v.stop_bps / 1e4

        for i, bar in trade_bars.iterrows():
            time_et = bar["bar_time_et"]
            # Force flatten at 15:55
            if time_et >= "15:55" and position != 0:
                exit_price = bar["close"]
                pnl_pct += position * (exit_price / entry_price - 1)
                # Cost on exit
                pnl_pct -= (COST_PER_SIDE_BPS + SLIPPAGE_BPS_PER_SIDE) / 1e4
                position = 0
                break

            # Manage existing position
            if position == 1:
                if bar["low"] <= stop_price:
                    pnl_pct += stop_price / entry_price - 1
                    pnl_pct -= (COST_PER_SIDE_BPS + SLIPPAGE_BPS_PER_SIDE) / 1e4
                    position = 0
                elif bar["high"] >= target_price:
                    pnl_pct += target_price / entry_price - 1
                    pnl_pct -= (COST_PER_SIDE_BPS + SLIPPAGE_BPS_PER_SIDE) / 1e4
                    position = 0
            elif position == -1:
                if bar["high"] >= stop_price:
                    pnl_pct += 1 - stop_price / entry_price
                    pnl_pct -= (COST_PER_SIDE_BPS + SLIPPAGE_BPS_PER_SIDE) / 1e4
                    position = 0
                elif bar["low"] <= target_price:
                    pnl_pct += 1 - target_price / entry_price
                    pnl_pct -= (COST_PER_SIDE_BPS + SLIPPAGE_BPS_PER_SIDE) / 1e4
                    position = 0

            # Look for new entry (only if flat)
            if position == 0:
                # Long breakout
                if bar["close"] > or_high * (1 + buf):
                    entry_price = bar["close"]
                    position = 1
                    entry_idx = i
                    stop_price = or_low * (1 - buf)
                    target_price = entry_price + or_width * v.target_mult
                    pnl_pct -= (COST_PER_SIDE_BPS + SLIPPAGE_BPS_PER_SIDE) / 1e4
                elif v.allow_short and bar["close"] < or_low * (1 - buf):
                    entry_price = bar["close"]
                    position = -1
                    entry_idx = i
                    stop_price = or_high * (1 + buf)
                    target_price = entry_price - or_width * v.target_mult
                    pnl_pct -= (COST_PER_SIDE_BPS + SLIPPAGE_BPS_PER_SIDE) / 1e4

        records.append({
            "session_date": session,
            "symbol": symbol,
            "or_high": or_high, "or_low": or_low, "or_width_bps": width_bps,
            "pnl_pct": pnl_pct,
        })

    return pd.DataFrame(records)


def simulate_orb_portfolio(all_bars: dict, v: ORBVariant) -> pd.Series:
    """Aggregate per-ETF daily P&L into a portfolio daily return series."""
    per_etf_dfs = []
    for sym, bars in all_bars.items():
        df = simulate_orb_one_ticker(bars, v, sym)
        if df.empty:
            continue
        df["weighted_pnl"] = df["pnl_pct"] * v.size
        per_etf_dfs.append(df[["session_date", "symbol", "weighted_pnl"]])
    if not per_etf_dfs:
        return pd.Series(dtype=float)
    long_df = pd.concat(per_etf_dfs)
    daily = long_df.groupby("session_date")["weighted_pnl"].sum()
    daily.index = pd.to_datetime(daily.index)
    return daily


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--variants", default="all")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--block", type=int, default=10)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args(argv)

    log.info("loading 5-min bars for %d ETFs...", len(UNIVERSE))
    all_bars = {}
    for sym in UNIVERSE:
        try:
            all_bars[sym] = load_bars(sym)
            log.info("%s: %d bars", sym, len(all_bars[sym]))
        except FileNotFoundError:
            log.error("%s: cache missing — run ingest.py first", sym)
            return 1

    # Split train / OOS by session date
    sample_bars = all_bars[UNIVERSE[0]]
    sample_bars["ts_utc"] = pd.to_datetime(sample_bars["ts_utc"], utc=True)
    train_end_ts = pd.Timestamp("2022-12-31", tz="UTC")
    oos_start_ts = pd.Timestamp("2023-01-01", tz="UTC")

    train_bars = {s: b[b["ts_utc"] <= train_end_ts] for s, b in all_bars.items()}
    oos_bars = {s: b[b["ts_utc"] >= oos_start_ts] for s, b in all_bars.items()}
    log.info("train bars per ETF: %d; OOS bars per ETF: %d",
             len(train_bars[UNIVERSE[0]]), len(oos_bars[UNIVERSE[0]]))

    # Benchmarks (SPY buy-hold = daily returns)
    spy_bars = all_bars["SPY"]
    spy_bars["session_date"] = pd.to_datetime(spy_bars["ts_utc"]).dt.tz_convert("America/New_York").dt.date
    spy_daily_close = spy_bars.groupby("session_date")["close"].last()
    spy_daily_ret = spy_daily_close.pct_change().fillna(0)
    spy_daily_ret.index = pd.to_datetime(spy_daily_ret.index)
    spy_train = spy_daily_ret[spy_daily_ret.index <= TRAIN_END.tz_localize(None)]
    spy_oos = spy_daily_ret[spy_daily_ret.index >= OOS_START.tz_localize(None)]
    spy_oos_m = metrics(spy_oos, "SPY_oos")
    log.info("SPY OOS Sharpe: %.3f", spy_oos_m["sharpe"])

    variants = all_variants()
    if args.quick:
        variants = variants[:12]
        args.bootstrap = 200
    elif args.variants != "all":
        variants = variants[:int(args.variants)]
    log.info("running %d variants", len(variants))

    t0 = time.time()
    train_ret_cols, oos_ret_cols = {}, {}
    per_variant_meta = []
    for i, v in enumerate(variants):
        if i % 10 == 0:
            log.info("variant %d/%d  (%.1fs)", i, len(variants), time.time() - t0)
        try:
            tr = simulate_orb_portfolio(train_bars, v)
            oo = simulate_orb_portfolio(oos_bars, v)
        except Exception as e:
            log.warning("variant %s failed: %s", v.key(), e); continue
        train_ret_cols[v.key()] = tr
        oos_ret_cols[v.key()] = oo
        per_variant_meta.append({
            "variant": v.key(),
            **{f"train_{k}": val for k, val in metrics(tr, "train").items() if k != "label"},
            **{f"oos_{k}": val for k, val in metrics(oo, "oos").items() if k != "label"},
        })

    elapsed = time.time() - t0
    log.info("variant sweep done in %.1fs", elapsed)

    if not train_ret_cols:
        log.error("no variants produced returns")
        return 1

    train_ret_df = pd.DataFrame(train_ret_cols).fillna(0)
    oos_ret_df = pd.DataFrame(oos_ret_cols).fillna(0)
    train_sharpes = train_ret_df.apply(lambda r: metrics(r, "")["sharpe"])
    best_variant = train_sharpes.idxmax()
    best_train_sharpe = train_sharpes.max()
    log.info("best variant: %s (train Sharpe %.3f)", best_variant, best_train_sharpe)

    log.info("White RC bootstrap (B=%d)...", args.bootstrap)
    rc = whites_reality_check_bootstrap(train_ret_df, B=args.bootstrap, block=args.block)
    log.info("White RC p-value: %.4f", rc["p_RC"])

    log.info("PBO (16 partitions)...")
    pbo = pbo_combinatorial(train_ret_df, n_partitions=16)
    log.info("PBO: %.3f", pbo["PBO"])

    dsr = deflated_sharpe(best_train_sharpe, n_trials=len(variants), T=len(train_ret_df))
    log.info("DSR: %.4f", dsr)

    best_train = train_ret_df[best_variant]
    best_oos = oos_ret_df[best_variant]
    t_stat = (best_train.mean() / (best_train.std() / math.sqrt(len(best_train)))) if best_train.std() > 0 else 0.0
    best_oos_m = metrics(best_oos, "oos")

    summary = {
        "ran_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "candidate": "mft_v1_orb",
        "n_variants": len(variants),
        "elapsed_sec": elapsed,
        "best_variant": best_variant,
        "best_train_sharpe": float(best_train_sharpe),
        "best_t_stat": float(t_stat),
        "white_RC_p": rc["p_RC"],
        "PBO": pbo["PBO"],
        "DSR": dsr,
        "benchmark_SPY_oos": spy_oos_m,
        "best_train_metrics": metrics(best_train, "train"),
        "best_oos_metrics": best_oos_m,
        "vs_SPY_sharpe_diff_oos": float(best_oos_m["sharpe"] - spy_oos_m["sharpe"]),
        "hurdles": {
            "white_RC_p_lt_0.05": rc["p_RC"] < 0.05,
            "PBO_lt_0.50": pbo["PBO"] < 0.50,
            "DSR_gt_0.80": dsr > 0.80,
            "t_stat_gt_3.0": t_stat > 3.0,
            "OOS_sharpe_gt_1.5": best_oos_m["sharpe"] > 1.5,
            "OOS_sharpe_beats_SPY_by_0.20": (best_oos_m["sharpe"] - spy_oos_m["sharpe"]) >= 0.20,
            "OOS_max_DD_lt_15pct": abs(best_oos_m["max_dd"]) < 0.15,
            "OOS_calmar_gt_1.5": best_oos_m["calmar"] > 1.5,
            "shrinkage_lt_50pct": abs(best_train_sharpe - best_oos_m["sharpe"]) / max(abs(best_train_sharpe), 1e-9) < 0.5,
        },
    }
    summary["all_hurdles_pass"] = all(summary["hurdles"].values())

    out = RESULTS_DIR / f"backtest_summary_{int(time.time())}.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    pd.DataFrame(per_variant_meta).to_csv(RESULTS_DIR / "variant_metrics.csv", index=False)
    train_ret_df.to_parquet(RESULTS_DIR / "train_returns.parquet")
    oos_ret_df.to_parquet(RESULTS_DIR / "oos_returns.parquet")

    print("\n=== MFTv1 ORB BACKTEST SUMMARY ===")
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary["all_hurdles_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
