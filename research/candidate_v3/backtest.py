"""Phase 3 backtest for candidate_v3 — HRP-weighted diversified ETF basket.

Tests whether HRP beats equal-weight (1/N) benchmark on the same universe
over a walk-forward, applying multiple-testing correction across the variant
grid + DSR + PBO.

Run:
    .venv/bin/python -m research.candidate_v3.backtest --quick   # 6 variants
    .venv/bin/python -m research.candidate_v3.backtest           # full 72
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
from research.candidate_v3.hrp import hrp_weights, equal_weight

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "cache" / "candidate_v3"
RESULTS_DIR = ROOT / "research" / "candidate_v3" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM",
            "TLT", "IEF", "LQD", "HYG", "GLD", "DBC", "VNQ"]
TRAIN_END = pd.Timestamp("2022-12-31")
OOS_START = pd.Timestamp("2023-01-01")

COST_PER_SIDE_BPS = 5.0

VARIANT_GRID = {
    "LOOKBACK_DAYS": [126, 252, 504],
    "REBALANCE_FREQ": ["M", "Q"],
    "WEIGHT_FLOOR": [0.0, 0.02],
    "WEIGHT_CAP": [0.20, 0.30, 0.50],
    "LINKAGE_METHOD": ["single", "average"],
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("candidate_v3.backtest")


@dataclass
class HRPVariant:
    lookback: int
    rebal_freq: str
    weight_floor: float
    weight_cap: float
    linkage: str

    def key(self) -> str:
        return (f"L{self.lookback}_R{self.rebal_freq}_F{self.weight_floor}"
                f"_C{self.weight_cap}_M{self.linkage}")


def all_variants() -> list[HRPVariant]:
    return [HRPVariant(*c) for c in product(*VARIANT_GRID.values())]


def load_prices() -> pd.DataFrame:
    frames = []
    for sym in UNIVERSE:
        df = pd.read_parquet(CACHE_DIR / f"{sym}.parquet")[["date", "close"]]
        df = df.rename(columns={"close": sym}).set_index("date")
        frames.append(df)
    wide = pd.concat(frames, axis=1).sort_index()
    wide.index = pd.to_datetime(wide.index)
    wide = wide.ffill(limit=2).dropna()
    return wide


def rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """Last trading day of each rebalance period."""
    if freq == "M":
        marker = index.to_series().groupby([index.year, index.month]).last()
    elif freq == "Q":
        marker = index.to_series().groupby([index.year, index.quarter]).last()
    else:
        raise ValueError(freq)
    return pd.DatetimeIndex(marker.values)


def simulate_hrp_walk_forward(prices: pd.DataFrame, v: HRPVariant,
                              cost_bps: float = COST_PER_SIDE_BPS) -> pd.Series:
    """Walk-forward: at each rebalance date, refit HRP weights on past
    `lookback` days, hold to next rebalance date. Apply cost on turnover."""
    returns = prices.pct_change().dropna()
    rebal = rebalance_dates(returns.index, v.rebal_freq)
    rebal = rebal[rebal >= returns.index[0] + pd.Timedelta(days=v.lookback)]
    if len(rebal) < 4:
        return pd.Series(0.0, index=returns.index)

    weights_ts = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    prev_w = pd.Series(0.0, index=returns.columns)
    turnover_per_rebal: list[float] = []

    for rd in rebal:
        window_end = returns.index[returns.index <= rd][-1]
        window_start_idx = returns.index.get_loc(window_end) - v.lookback
        if window_start_idx < 0:
            continue
        window = returns.iloc[window_start_idx:returns.index.get_loc(window_end) + 1]
        try:
            w = hrp_weights(window, linkage_method=v.linkage,
                            weight_floor=v.weight_floor, weight_cap=v.weight_cap)
        except Exception:
            continue
        # Apply weights from rd to next rebal date
        next_idx = returns.index.get_loc(window_end) + 1
        end_idx = len(returns)
        if rd != rebal[-1]:
            next_rd = rebal[list(rebal).index(rd) + 1]
            end_idx = returns.index.get_loc(returns.index[returns.index <= next_rd][-1])
        for col in w.index:
            weights_ts.iloc[next_idx:end_idx, weights_ts.columns.get_loc(col)] = w[col]
        # Turnover cost = sum(|w_new - w_old|) × cost_bps
        turnover = (w - prev_w).abs().sum()
        turnover_per_rebal.append(float(turnover))
        prev_w = w

    # Daily portfolio return = sum(w_i × r_i)
    port_ret = (weights_ts * returns).sum(axis=1)
    # Apply turnover costs on rebalance days
    cost_series = pd.Series(0.0, index=returns.index)
    for rd, turn in zip(rebal, turnover_per_rebal):
        # Cost charged the day AFTER the rebalance (signal is on rd close)
        candidates = returns.index[returns.index > rd]
        if len(candidates):
            cost_series.loc[candidates[0]] = turn * (cost_bps / 1e4) * 2  # both sides
    port_ret_net = port_ret - cost_series
    port_ret_net.attrs["mean_turnover"] = float(np.mean(turnover_per_rebal)) if turnover_per_rebal else 0.0
    port_ret_net.attrs["n_rebal"] = len(rebal)
    return port_ret_net


def simulate_equal_weight(prices: pd.DataFrame, rebal_freq: str = "M",
                          cost_bps: float = COST_PER_SIDE_BPS) -> pd.Series:
    """The 1/N benchmark, rebalanced at the same cadence as HRP variant."""
    returns = prices.pct_change().dropna()
    n = len(returns.columns)
    rebal = rebalance_dates(returns.index, rebal_freq)
    rebal = rebal[rebal >= returns.index[0]]

    weights_ts = pd.DataFrame(1.0 / n, index=returns.index, columns=returns.columns)
    prev_w = pd.Series(1.0 / n, index=returns.columns)
    turnover = []
    for rd in rebal:
        w = pd.Series(1.0 / n, index=returns.columns)
        turnover.append(float((w - prev_w).abs().sum()))
        prev_w = w
    port = (weights_ts * returns).sum(axis=1)
    # Each rebalance costs `turnover × cost`
    cost_series = pd.Series(0.0, index=returns.index)
    for rd, t in zip(rebal, turnover):
        candidates = returns.index[returns.index > rd]
        if len(candidates):
            cost_series.loc[candidates[0]] = t * (cost_bps / 1e4) * 2
    return port - cost_series


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--variants", default="all")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--block", type=int, default=10)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args(argv)

    log.info("loading prices (v3 diversified ETF, %d symbols)...", len(UNIVERSE))
    prices = load_prices()
    train = prices[prices.index <= TRAIN_END]
    oos = prices[prices.index >= OOS_START]
    log.info("train: %s..%s (%d days)  oos: %s..%s (%d days)",
             train.index.min().date(), train.index.max().date(), len(train),
             oos.index.min().date(), oos.index.max().date(), len(oos))

    variants = all_variants()
    if args.quick:
        variants = variants[:6]; args.bootstrap = 200
    elif args.variants != "all":
        variants = variants[:int(args.variants)]
    log.info("running %d HRP variants", len(variants))

    # Benchmark: 1/N monthly
    log.info("computing 1/N benchmark...")
    ew_train = simulate_equal_weight(train, "M")
    ew_oos = simulate_equal_weight(oos, "M")
    ew_train_m = metrics(ew_train, "ew_train")
    ew_oos_m = metrics(ew_oos, "ew_oos")
    log.info("1/N train Sharpe: %.3f  OOS Sharpe: %.3f", ew_train_m["sharpe"], ew_oos_m["sharpe"])

    t0 = time.time()
    train_ret_cols, oos_ret_cols = {}, {}
    per_variant_meta = []
    for i, v in enumerate(variants):
        if i % 6 == 0:
            log.info("variant %d/%d  (%.1fs)", i, len(variants), time.time() - t0)
        try:
            tr = simulate_hrp_walk_forward(train, v)
            oo = simulate_hrp_walk_forward(oos, v)
        except Exception as e:
            log.warning("variant %s failed: %s", v.key(), e); continue
        train_ret_cols[v.key()] = tr
        oos_ret_cols[v.key()] = oo
        per_variant_meta.append({
            "variant": v.key(),
            "mean_turnover": tr.attrs.get("mean_turnover", 0.0),
            "n_rebal_train": tr.attrs.get("n_rebal", 0),
            **{f"train_{k}": val for k, val in metrics(tr, "train").items() if k != "label"},
            **{f"oos_{k}": val for k, val in metrics(oo, "oos").items() if k != "label"},
        })

    elapsed = time.time() - t0
    log.info("variant sweep done in %.1fs", elapsed)

    train_ret_df = pd.DataFrame(train_ret_cols).fillna(0)
    oos_ret_df = pd.DataFrame(oos_ret_cols).fillna(0)
    train_sharpes = train_ret_df.apply(lambda r: metrics(r, "")["sharpe"])
    best_variant = train_sharpes.idxmax()
    best_train_sharpe = train_sharpes.max()
    log.info("best HRP variant: %s (train Sharpe %.3f)", best_variant, best_train_sharpe)

    log.info("White RC bootstrap...")
    rc = whites_reality_check_bootstrap(train_ret_df, B=args.bootstrap, block=args.block)
    log.info("White RC p-value: %.4f", rc["p_RC"])

    log.info("PBO...")
    pbo = pbo_combinatorial(train_ret_df, n_partitions=16)
    log.info("PBO: %.3f", pbo["PBO"])

    dsr = deflated_sharpe(best_train_sharpe, n_trials=len(variants), T=len(train_ret_df))
    log.info("DSR (fixed formula): %.4f", dsr)

    best_train = train_ret_df[best_variant]
    best_oos = oos_ret_df[best_variant]
    t_stat = (best_train.mean() / (best_train.std() / math.sqrt(len(best_train)))) if best_train.std() > 0 else 0.0
    oos_sharpe = metrics(best_oos, "")["sharpe"]
    oos_max_dd = metrics(best_oos, "")["max_dd"]

    # HRP-vs-1/N comparison (the core test for this candidate)
    hrp_vs_ew_sharpe = oos_sharpe - ew_oos_m["sharpe"]
    hrp_vs_ew_dd_cut = (abs(ew_oos_m["max_dd"]) - abs(oos_max_dd)) / max(abs(ew_oos_m["max_dd"]), 1e-9)

    summary = {
        "ran_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "candidate": "v3_hrp_diversified",
        "n_variants": len(variants),
        "elapsed_sec": elapsed,
        "best_variant": best_variant,
        "best_train_sharpe": float(best_train_sharpe),
        "best_t_stat": float(t_stat),
        "white_RC_p": rc["p_RC"],
        "PBO": pbo["PBO"],
        "DSR": dsr,
        "benchmark_1N_oos": ew_oos_m,
        "best_hrp_oos": metrics(best_oos, "oos"),
        "hrp_vs_1N_sharpe_diff_oos": float(hrp_vs_ew_sharpe),
        "hrp_vs_1N_dd_cut_pct_oos": float(hrp_vs_ew_dd_cut * 100),
        "hurdles": {
            "OOS_sharpe_diff_gte_0.15": hrp_vs_ew_sharpe >= 0.15,
            "OOS_dd_cut_gte_20pct": hrp_vs_ew_dd_cut >= 0.20,
            "OOS_sharpe_gt_0.5": oos_sharpe > 0.5,
            "white_RC_p_lt_0.05": rc["p_RC"] < 0.05,
            "DSR_gt_0.80": dsr > 0.80,
            "shrinkage_lt_50pct": abs(best_train_sharpe - oos_sharpe) / max(abs(best_train_sharpe), 1e-9) < 0.5,
        },
    }
    summary["all_hurdles_pass"] = all(summary["hurdles"].values())

    out = RESULTS_DIR / f"backtest_summary_{int(time.time())}.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    pd.DataFrame(per_variant_meta).to_csv(RESULTS_DIR / "variant_metrics.csv", index=False)
    train_ret_df.to_parquet(RESULTS_DIR / "train_returns.parquet")
    oos_ret_df.to_parquet(RESULTS_DIR / "oos_returns.parquet")

    print("\n=== CANDIDATE V3 (HRP) BACKTEST SUMMARY ===")
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary["all_hurdles_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
