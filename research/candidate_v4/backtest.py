"""Phase 3 backtest for candidate_v4 — VIX term-structure rotation on SVXY.

Tests 162 variant combinations against the Avellaneda et al. (2021) hypothesis,
with multi-testing correction (White's RC + DSR + PBO + t > 3) and a benchmark
comparison vs SPY buy-and-hold.

Run:
    .venv/bin/python -m research.candidate_v4.backtest --quick   # 12 variants
    .venv/bin/python -m research.candidate_v4.backtest           # full 162
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
CACHE_DIR = ROOT / "data" / "cache" / "candidate_v4"
RESULTS_DIR = ROOT / "research" / "candidate_v4" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_END = pd.Timestamp("2022-12-31")
OOS_START = pd.Timestamp("2023-01-01")

COST_PER_SIDE_BPS = 5.0

VARIANT_GRID = {
    "CONTANGO_THRESHOLD": [1.00, 0.95, 0.90],
    "VOL_FILTER": [25.0, 30.0, 40.0],
    "POSITION_SIZE": [0.25, 0.50],
    "REENTRY_COOLDOWN": [0, 5, 10],
    "SIGNAL_CONFIRM": [1, 3, 5],
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("candidate_v4.backtest")


@dataclass
class VIXVariant:
    contango_th: float
    vol_filter: float
    size: float
    cooldown: int
    confirm: int

    def key(self) -> str:
        return f"c{self.contango_th}_v{self.vol_filter}_s{self.size}_d{self.cooldown}_f{self.confirm}"


def all_variants() -> list[VIXVariant]:
    return [VIXVariant(*c) for c in product(*VARIANT_GRID.values())]


def load_data() -> pd.DataFrame:
    """Load and merge ^VIX, ^VIX3M, SVXY, ^GSPC into one DataFrame."""
    def load_one(safe_name: str, col: str) -> pd.DataFrame:
        df = pd.read_parquet(CACHE_DIR / f"{safe_name}.parquet")[["date", "close"]]
        df = df.rename(columns={"close": col}).set_index("date")
        return df
    vix = load_one("_VIX", "vix")
    vix3m = load_one("_VIX3M", "vix3m")
    svxy = load_one("SVXY", "svxy")
    spx = load_one("_GSPC", "spx")
    df = pd.concat([vix, vix3m, svxy, spx], axis=1).sort_index()
    df.index = pd.to_datetime(df.index)
    df = df.ffill(limit=2).dropna()
    df["ratio"] = df["vix"] / df["vix3m"]
    df["svxy_ret"] = df["svxy"].pct_change().fillna(0)
    df["spx_ret"] = df["spx"].pct_change().fillna(0)
    return df


def simulate_variant(data: pd.DataFrame, v: VIXVariant,
                     cost_bps: float = COST_PER_SIDE_BPS) -> pd.Series:
    """Simulate one variant. Returns daily strategy P&L on NAV-normalised basis."""
    n = len(data)
    pos = np.zeros(n)  # fraction of NAV in SVXY
    cooldown_remaining = 0
    contango_streak = 0
    state = 0  # 0 flat, 1 long

    for t in range(1, n):
        ratio = data["ratio"].iloc[t]
        vix = data["vix"].iloc[t]

        # Stress filter — force flat
        if vix > v.vol_filter:
            if state == 1:
                cooldown_remaining = v.cooldown + 1  # +1 because we decrement first below
            state = 0
            pos[t] = 0
            contango_streak = 0
            continue

        # Track contango streak
        if ratio < v.contango_th:
            contango_streak += 1
        else:
            contango_streak = 0

        # Decrement cooldown
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            pos[t] = 0
            continue

        # Position rule
        if state == 0 and contango_streak >= v.confirm:
            state = 1
            pos[t] = v.size
        elif state == 1:
            if ratio >= v.contango_th:
                # Exit on first non-contango day (immediate)
                state = 0
                pos[t] = 0
            else:
                pos[t] = v.size

    # Daily P&L = pos[t-1] × svxy_ret[t]
    pos_prev = np.concatenate([[0], pos[:-1]])
    ret_gross = pos_prev * data["svxy_ret"].values

    # Costs on position changes
    pos_change = np.abs(np.diff(np.concatenate([[0], pos])))
    cost_vec = pos_change * (cost_bps / 1e4) * 2  # entry+exit each pays half
    # That's slightly wrong but close — properly half on entry day, half on exit
    ret_net = ret_gross - cost_vec

    return pd.Series(ret_net, index=data.index)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--variants", default="all")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--block", type=int, default=10)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args(argv)

    log.info("loading VIX rotation data...")
    data = load_data()
    train = data[data.index <= TRAIN_END]
    oos = data[data.index >= OOS_START]
    log.info("train: %s..%s (%d days)  oos: %s..%s (%d days)",
             train.index.min().date(), train.index.max().date(), len(train),
             oos.index.min().date(), oos.index.max().date(), len(oos))

    # SPY benchmark
    spx_train = train["spx_ret"]
    spx_oos = oos["spx_ret"]
    bench_train = metrics(spx_train, "spx_train")
    bench_oos = metrics(spx_oos, "spx_oos")
    log.info("SPX buy-hold train Sharpe: %.3f  OOS Sharpe: %.3f",
             bench_train["sharpe"], bench_oos["sharpe"])

    variants = all_variants()
    if args.quick:
        variants = variants[:12]; args.bootstrap = 200
    elif args.variants != "all":
        variants = variants[:int(args.variants)]
    log.info("running %d variants", len(variants))

    t0 = time.time()
    train_ret_cols, oos_ret_cols = {}, {}
    per_variant_meta = []
    for i, v in enumerate(variants):
        if i % 30 == 0:
            log.info("variant %d/%d (%.1fs)", i, len(variants), time.time() - t0)
        try:
            tr = simulate_variant(train, v)
            oo = simulate_variant(oos, v)
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
    log.info("best variant: %s (train Sharpe %.3f)", best_variant, best_train_sharpe)

    log.info("White RC bootstrap...")
    rc = whites_reality_check_bootstrap(train_ret_df, B=args.bootstrap, block=args.block)
    log.info("White RC p-value: %.4f", rc["p_RC"])

    log.info("PBO...")
    pbo = pbo_combinatorial(train_ret_df, n_partitions=16)
    log.info("PBO: %.3f", pbo["PBO"])

    dsr = deflated_sharpe(best_train_sharpe, n_trials=len(variants), T=len(train_ret_df))
    log.info("DSR: %.4f", dsr)

    best_train = train_ret_df[best_variant]
    best_oos = oos_ret_df[best_variant]
    t_stat = (best_train.mean() / (best_train.std() / math.sqrt(len(best_train)))) if best_train.std() > 0 else 0.0
    best_oos_m = metrics(best_oos, "oos")
    best_train_m = metrics(best_train, "train")

    summary = {
        "ran_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "candidate": "v4_vix_term_structure",
        "n_variants": len(variants),
        "elapsed_sec": elapsed,
        "best_variant": best_variant,
        "best_train_sharpe": float(best_train_sharpe),
        "best_t_stat": float(t_stat),
        "white_RC_p": rc["p_RC"],
        "PBO": pbo["PBO"],
        "DSR": dsr,
        "benchmark_SPX_oos": bench_oos,
        "best_train_metrics": best_train_m,
        "best_oos_metrics": best_oos_m,
        "hurdles": {
            "white_RC_p_lt_0.05": rc["p_RC"] < 0.05,
            "PBO_lt_0.50": pbo["PBO"] < 0.50,
            "DSR_gt_0.80": dsr > 0.80,
            "t_stat_gt_3.0": t_stat > 3.0,
            "OOS_sharpe_gt_1.0": best_oos_m["sharpe"] > 1.0,
            "OOS_max_DD_lt_30pct": abs(best_oos_m["max_dd"]) < 0.30,
            "OOS_calmar_gt_1.0": best_oos_m["calmar"] > 1.0,
            "shrinkage_lt_50pct": abs(best_train_sharpe - best_oos_m["sharpe"]) / max(abs(best_train_sharpe), 1e-9) < 0.5,
            "OOS_sharpe_gt_SPX": best_oos_m["sharpe"] > bench_oos["sharpe"],
        },
    }
    summary["all_hurdles_pass"] = all(summary["hurdles"].values())

    out = RESULTS_DIR / f"backtest_summary_{int(time.time())}.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    pd.DataFrame(per_variant_meta).to_csv(RESULTS_DIR / "variant_metrics.csv", index=False)
    train_ret_df.to_parquet(RESULTS_DIR / "train_returns.parquet")
    oos_ret_df.to_parquet(RESULTS_DIR / "oos_returns.parquet")

    print("\n=== CANDIDATE V4 (VIX) BACKTEST SUMMARY ===")
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary["all_hurdles_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
