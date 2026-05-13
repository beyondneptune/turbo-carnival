"""Walk-forward OOS validation for candidate_v1.

Per the preregistration §11:
- 5 purged k-fold folds within the train set (2015-2022)
- 22-day purge gap between train and test slices
- 2000-iteration stationary bootstrap CI for OOS Sharpe (10-day blocks)
- Per-fold metrics + aggregate + bootstrap CI lower bound

Gates (§9):
- OOS DSR > 1.0
- 95% bootstrap CI lower bound > 0
- Sharpe shrinkage (train -> walk-forward) < 50%

Run:
    .venv/bin/python -m research.candidate_v1.walk_forward
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.candidate_v1.backtest import (
    UNIVERSE, TRAIN_END, OOS_START, VariantParams,
    load_prices, simulate_portfolio, metrics, deflated_sharpe,
    PER_PAIR_NAV_FRAC, COST_PER_SIDE_BPS,
)

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "research" / "candidate_v1" / "results"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("candidate_v1.walk_forward")


def load_best_variant() -> VariantParams:
    """Read best variant from backtest summary JSON."""
    summaries = sorted(RESULTS_DIR.glob("backtest_summary_*.json"))
    if not summaries:
        raise SystemExit("no backtest_summary_*.json found — run backtest.py first")
    latest = json.loads(summaries[-1].read_text())
    key = latest["best_variant"]
    # Parse "w20_e1.5_s4.0_x0.0_p0.1_h60" back into VariantParams
    parts = key.split("_")
    parsed = {p[0]: p[1:] for p in parts}
    return VariantParams(
        window=int(parsed["w"]),
        entry=float(parsed["e"]),
        stop=float(parsed["s"]),
        exit=float(parsed["x"]),
        coint_p=float(parsed["p"]),
        max_hold=int(parsed["h"]),
    )


def purged_kfold_splits(dates: pd.DatetimeIndex, n_splits: int = 5,
                       purge_days: int = 22) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Generate purged k-fold splits — for each fold the test slice gets
    a `purge_days` gap on either side excluded from the train slice."""
    n = len(dates)
    fold_size = n // n_splits
    splits = []
    for i in range(n_splits):
        test_start = i * fold_size
        test_end = test_start + fold_size if i < n_splits - 1 else n
        purge_start = max(0, test_start - purge_days)
        purge_end = min(n, test_end + purge_days)
        train_idx = np.concatenate([
            np.arange(0, purge_start),
            np.arange(purge_end, n),
        ])
        test_idx = np.arange(test_start, test_end)
        splits.append((dates[train_idx], dates[test_idx]))
    return splits


def stationary_bootstrap_sharpe_ci(
    returns: np.ndarray, B: int = 2000, block: int = 10, seed: int = 42,
) -> dict:
    """Politis-Romano stationary bootstrap 95% CI for Sharpe."""
    rng = np.random.default_rng(seed)
    T = len(returns)
    if T < 30:
        return {"sharpe": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "B": B}
    sharpes = np.empty(B)
    n_blocks = T // block + 1
    for b in range(B):
        starts = rng.integers(0, T, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) % T for s in starts])[:T]
        r_b = returns[idx]
        if r_b.std() > 0:
            sharpes[b] = (r_b.mean() / r_b.std()) * np.sqrt(252)
        else:
            sharpes[b] = 0.0
    obs_sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0.0
    ci_lo, ci_hi = np.percentile(sharpes, [2.5, 97.5])
    return {
        "sharpe": float(obs_sharpe),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "B": int(B),
        "block": int(block),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--purge", type=int, default=22)
    args = p.parse_args(argv)

    log.info("loading best variant from prior backtest...")
    best = load_best_variant()
    log.info("best variant: %s", best.key())

    prices = load_prices()
    train_prices = prices[prices.index <= TRAIN_END]
    oos_prices = prices[prices.index >= OOS_START]

    # 1. Re-simulate the best variant on full train set
    log.info("re-simulating best variant on train set...")
    train_ret = simulate_portfolio(train_prices, best)
    train_metrics = metrics(train_ret, "train")
    log.info("train: Sharpe=%.3f  ret=%.2f%%  DD=%.2f%%",
             train_metrics["sharpe"], train_metrics["tot_ret"] * 100, train_metrics["max_dd"] * 100)

    # 2. Walk-forward purged k-fold on the train set itself
    log.info("running purged %d-fold walk-forward (purge=%d days)...", args.k, args.purge)
    splits = purged_kfold_splits(train_prices.index, n_splits=args.k, purge_days=args.purge)
    fold_metrics = []
    fold_returns_all = []
    for i, (train_dates, test_dates) in enumerate(splits):
        # Refit/predict isn't really meaningful for this rules-based strategy —
        # the "refit" is just rolling regression which is path-internal. So
        # walk-forward = compute returns on test_dates only.
        test_prices = train_prices.loc[test_dates]
        if len(test_prices) < best.window + 10:
            log.warning("fold %d too short, skipping", i)
            continue
        fold_ret = simulate_portfolio(test_prices, best)
        fold_metrics.append({"fold": i, **{k: v for k, v in metrics(fold_ret, f"fold_{i}").items() if k != "label"}})
        fold_returns_all.append(fold_ret)
        log.info("fold %d (%s..%s): Sharpe=%.3f  ret=%.2f%%",
                 i, test_dates.min().date(), test_dates.max().date(),
                 fold_metrics[-1]["sharpe"], fold_metrics[-1]["tot_ret"] * 100)

    wf_returns = pd.concat(fold_returns_all)
    wf_metrics_all = metrics(wf_returns, "walk_forward_aggregate")
    log.info("walk-forward aggregate Sharpe: %.3f", wf_metrics_all["sharpe"])

    # 3. OOS evaluation on 2023-2025
    log.info("running OOS evaluation (2023-01-01 onwards)...")
    oos_ret = simulate_portfolio(oos_prices, best)
    oos_metrics = metrics(oos_ret, "oos")
    log.info("OOS Sharpe: %.3f  ret=%.2f%%  DD=%.2f%%",
             oos_metrics["sharpe"], oos_metrics["tot_ret"] * 100, oos_metrics["max_dd"] * 100)

    # 4. Bootstrap CI on OOS Sharpe
    log.info("running stationary bootstrap CI (B=%d, block=10)...", args.bootstrap)
    ci = stationary_bootstrap_sharpe_ci(oos_ret.dropna().values, B=args.bootstrap)
    log.info("OOS Sharpe %.3f  95%% CI [%.3f, %.3f]",
             ci["sharpe"], ci["ci_lo"], ci["ci_hi"])

    # 5. Shrinkage check
    shrink = (train_metrics["sharpe"] - oos_metrics["sharpe"]) / max(abs(train_metrics["sharpe"]), 1e-9)
    log.info("Sharpe shrinkage train->OOS: %.1f%%", shrink * 100)

    # 6. DSR on OOS
    summaries = sorted(RESULTS_DIR.glob("backtest_summary_*.json"))
    n_trials = json.loads(summaries[-1].read_text())["n_variants"]
    dsr_oos = deflated_sharpe(oos_metrics["sharpe"], n_trials=n_trials, T=len(oos_ret))
    log.info("OOS DSR: %.4f", dsr_oos)

    hurdles = {
        "OOS_DSR_gt_1.0": dsr_oos > 1.0,
        "OOS_sharpe_CI_lo_gt_0": ci["ci_lo"] > 0,
        "shrinkage_lt_50pct": abs(shrink) < 0.50,
        "OOS_net_sharpe_gt_1": oos_metrics["sharpe"] > 1.0,
    }

    summary = {
        "ran_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "best_variant": best.key(),
        "train": train_metrics,
        "walk_forward_aggregate": wf_metrics_all,
        "fold_metrics": fold_metrics,
        "oos": oos_metrics,
        "oos_bootstrap_CI": ci,
        "shrinkage_pct": float(shrink * 100),
        "oos_DSR": float(dsr_oos),
        "hurdles": hurdles,
        "all_hurdles_pass": all(hurdles.values()),
    }
    out_path = RESULTS_DIR / f"walk_forward_summary_{int(time.time())}.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print("\n=== WALK-FORWARD SUMMARY ===")
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nresults saved to {out_path}")
    return 0 if summary["all_hurdles_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
