"""Backtest engine for candidate_v2 — sector ETF cointegration pairs.

Reuses 100% of the v1 backtest engine (including the fixed DSR formula);
only loads data from the v2 cache and uses the 9-ETF sector universe.

Run:
    .venv/bin/python -m research.candidate_v2.backtest --quick   # 30-variant dev
    .venv/bin/python -m research.candidate_v2.backtest           # full 810
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd

import research.candidate_v1.backtest as _v1bt
from research.candidate_v1.backtest import (
    VariantParams, simulate_portfolio as _v1_simulate_portfolio,
    metrics, deflated_sharpe, whites_reality_check_bootstrap,
    pbo_combinatorial, all_variants, VARIANT_GRID,
    PER_PAIR_NAV_FRAC, COST_PER_SIDE_BPS,
)

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "cache" / "candidate_v2"
RESULTS_DIR = ROOT / "research" / "candidate_v2" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSE = ["XLE", "XLF", "XLK", "XLY", "XLP", "XLI", "XLB", "XLV", "XLU"]
TRAIN_END = pd.Timestamp("2022-12-31")
OOS_START = pd.Timestamp("2023-01-01")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("candidate_v2.backtest")


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


def simulate_portfolio_v2(prices: pd.DataFrame, p: VariantParams) -> pd.Series:
    """Same as v1 but uses the v2 universe."""
    pair_returns = []
    n_fills = 0
    for a, b in combinations(UNIVERSE, 2):
        pdr = _v1bt.simulate_pair(prices[a], prices[b], p)
        pair_returns.append(pdr["ret_net"].rename(f"{a}_{b}"))
        n_fills += pdr.attrs.get("n_fills", 0)
    pair_df = pd.concat(pair_returns, axis=1)
    portfolio = pair_df.sum(axis=1) * PER_PAIR_NAV_FRAC
    portfolio.attrs["n_fills"] = n_fills
    return portfolio


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--variants", default="all")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--block", type=int, default=10)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args(argv)

    log.info("loading prices (v2 sector ETF universe, %d symbols)...", len(UNIVERSE))
    prices = load_prices()
    train = prices[prices.index <= TRAIN_END]
    oos = prices[prices.index >= OOS_START]
    log.info("train: %s..%s (%d days)  oos: %s..%s (%d days)",
             train.index.min().date(), train.index.max().date(), len(train),
             oos.index.min().date(), oos.index.max().date(), len(oos))

    variants = all_variants()
    if args.quick:
        variants = variants[:30]; args.bootstrap = 200
    elif args.variants != "all":
        variants = variants[:int(args.variants)]
    n_pairs = len(list(combinations(UNIVERSE, 2)))
    log.info("running %d variants × %d pairs = %d pair-backtests",
             len(variants), n_pairs, len(variants) * n_pairs)

    t0 = time.time()
    train_ret_cols, oos_ret_cols = {}, {}
    per_variant_meta = []
    for i, v in enumerate(variants):
        if i % 20 == 0:
            log.info("variant %d/%d  (%.1fs elapsed)", i, len(variants), time.time() - t0)
        try:
            tr = simulate_portfolio_v2(train, v)
            oo = simulate_portfolio_v2(oos, v)
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

    train_ret_df = pd.DataFrame(train_ret_cols).fillna(0)
    oos_ret_df = pd.DataFrame(oos_ret_cols).fillna(0)
    train_sharpes = train_ret_df.apply(lambda r: metrics(r, "")["sharpe"])
    best_variant = train_sharpes.idxmax()
    best_train_sharpe = train_sharpes.max()
    log.info("best train Sharpe variant: %s  (%.3f)", best_variant, best_train_sharpe)

    log.info("White's RC bootstrap (B=%d, block=%d)...", args.bootstrap, args.block)
    rc = whites_reality_check_bootstrap(train_ret_df, B=args.bootstrap, block=args.block)
    log.info("White RC p-value: %.4f", rc["p_RC"])

    log.info("PBO (16 partitions)...")
    pbo = pbo_combinatorial(train_ret_df, n_partitions=16)
    log.info("PBO: %.3f", pbo["PBO"])

    dsr = deflated_sharpe(best_train_sharpe, n_trials=len(variants), T=len(train_ret_df))
    log.info("Deflated Sharpe Ratio (fixed formula): %.4f", dsr)

    best_ret = train_ret_df[best_variant]
    t_stat = (best_ret.mean() / (best_ret.std() / math.sqrt(len(best_ret)))) if best_ret.std() > 0 else 0.0
    log.info("best variant t-stat: %.3f (Harvey-Liu-Zhu hurdle: 3.0)", t_stat)

    summary = {
        "ran_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "candidate": "v2_sector_etf_cointegration",
        "n_variants": len(variants),
        "n_pairs": n_pairs,
        "elapsed_sec": elapsed,
        "best_variant": best_variant,
        "best_train_sharpe": float(best_train_sharpe),
        "best_t_stat": float(t_stat),
        "white_RC_p": rc["p_RC"],
        "PBO": pbo["PBO"],
        "DSR": dsr,
        "hurdles": {
            "white_RC_p_lt_0.05": rc["p_RC"] < 0.05,
            "PBO_lt_0.50": pbo["PBO"] < 0.50,
            "DSR_gt_0.95": dsr > 0.95,
            "t_stat_gt_3.0": t_stat > 3.0,
            "OOS_sharpe_gt_1.0": metrics(oos_ret_df[best_variant], "")["sharpe"] > 1.0,
        },
        "best_train_metrics": metrics(train_ret_df[best_variant], "train"),
        "best_oos_metrics": metrics(oos_ret_df[best_variant], "oos"),
    }
    summary["all_hurdles_pass"] = all(summary["hurdles"].values())

    out_path = RESULTS_DIR / f"backtest_summary_{int(time.time())}.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    pd.DataFrame(per_variant_meta).to_csv(RESULTS_DIR / "variant_metrics.csv", index=False)
    train_ret_df.to_parquet(RESULTS_DIR / "train_returns.parquet")
    oos_ret_df.to_parquet(RESULTS_DIR / "oos_returns.parquet")

    print("\n=== CANDIDATE V2 BACKTEST SUMMARY ===")
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary["all_hurdles_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
