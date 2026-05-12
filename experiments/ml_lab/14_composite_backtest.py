"""Build 5 composite gate models from the top pass-gate performers
and backtest each against yesterday's 162 entries.

Composites tested:
  C1: any-of-3 surgical    (016 ∪ 017 ∪ 018)
  C2: majority-of-5 surgical (016, 017, 018, 074, 121 — block if ≥3 agree)
  C3: surgical UNION + 102  (any low-rate + 102 itself)
  C4: top-5 weighted soft vote (probability average weighted by historical Δ)
  C5: surgical intersection (016 ∩ 017 ∩ 018 — only block if ALL agree)

All probabilities come from the per_entry parquet already produced by
13_pass_gate_yesterday_backtest.py — so no retraining needed.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/weeoto/turbo-carnival")
RUNS = REPO / "experiments" / "ml_lab" / "runs" / "pass_gate_backtest"
OUT = REPO / "experiments" / "ml_lab" / "runs" / "composite_backtest"
OUT.mkdir(parents=True, exist_ok=True)

BLOCK_THRESHOLD = 0.5

# Top performers from yesterday's backtest
SURGICAL = ["016_local_outlier_factor", "017_one_class_svm", "018_elliptic_envelope",
             "074_nn_lbfgs_solver", "121_poly_kernel_ridge"]
AGGRESSIVE = ["102_rbf_sampler_logreg", "103_nystroem_logreg", "101_interaction_poly_logreg"]
ALL_TOP = SURGICAL + AGGRESSIVE


def main():
    per_entry = pd.read_parquet(RUNS / "per_entry.parquet")
    summary = pd.read_parquet(RUNS / "summary.parquet")
    print(f"Loaded {len(per_entry):,} per-entry rows from {per_entry['model'].nunique()} models")

    # Pivot to (entry_idx × model) matrix of probabilities
    # Use the first model's row order as the entry index basis
    base_model = per_entry["model"].iloc[0]
    base = per_entry[per_entry["model"] == base_model].reset_index(drop=True)
    n = len(base)
    print(f"Base: {n} entries from {base_model}")

    proba_by_model = {}
    for m in ALL_TOP:
        sub = per_entry[per_entry["model"] == m]
        if len(sub) != n:
            print(f"  WARN: {m} has {len(sub)} rows, expected {n} — skipping")
            continue
        proba_by_model[m] = sub.reset_index(drop=True)["model_proba"].values

    print(f"Loaded probabilities for {len(proba_by_model)} models")

    # Block decisions per model
    blocked_by_model = {m: p > BLOCK_THRESHOLD for m, p in proba_by_model.items()}

    pnl = base["pnl"].values
    actual_pnl = float(pnl.sum())
    print(f"Actual bar-close P&L = ${actual_pnl:+,.2f}")
    print()

    # Historical Δ per model (for weighted vote)
    deltas = dict(zip(summary["model"], summary["pnl_delta"]))

    composites = {}

    # C1: ANY-OF-3 surgical (016 ∪ 017 ∪ 018)
    c1 = np.zeros(n, dtype=bool)
    for m in ["016_local_outlier_factor", "017_one_class_svm", "018_elliptic_envelope"]:
        if m in blocked_by_model:
            c1 |= blocked_by_model[m]
    composites["C1_any_of_3_surgical"] = c1

    # C2: MAJORITY-OF-5 surgical
    surg_5 = [m for m in SURGICAL if m in blocked_by_model]
    votes = np.sum([blocked_by_model[m].astype(int) for m in surg_5], axis=0)
    composites["C2_majority_of_5_surgical"] = votes >= 3

    # C3: surgical UNION + 102_rbf_sampler_logreg
    c3 = c1.copy()
    if "102_rbf_sampler_logreg" in blocked_by_model:
        c3 |= blocked_by_model["102_rbf_sampler_logreg"]
    composites["C3_surgical_union_plus_102"] = c3

    # C4: weighted soft vote on top-8 (probabilities weighted by historical Δ)
    soft_p = np.zeros(n)
    weight_sum = 0
    for m in ALL_TOP:
        if m not in proba_by_model: continue
        d = float(deltas.get(m, 0))
        if d <= 0: continue
        soft_p += d * proba_by_model[m]
        weight_sum += d
    if weight_sum > 0:
        soft_p /= weight_sum
    composites["C4_soft_vote_delta_weighted"] = soft_p > BLOCK_THRESHOLD

    # C5: surgical INTERSECTION (must agree)
    c5 = np.ones(n, dtype=bool)
    for m in ["016_local_outlier_factor", "017_one_class_svm", "018_elliptic_envelope"]:
        if m in blocked_by_model:
            c5 &= blocked_by_model[m]
    composites["C5_intersection_of_3_surgical"] = c5

    # Evaluate
    rows = []
    print(f"{'Composite':40s}  {'Blocked':>8s}  {'Rate':>6s}  {'BW/BL':>8s}  {'Δ$':>10s}  {'CF P&L':>10s}")
    print("-" * 92)
    for name, blocked in composites.items():
        kept = ~blocked
        cf = float(pnl[kept].sum())
        bw = int((pnl[blocked] > 0).sum())
        bl = int((pnl[blocked] < 0).sum())
        delta = cf - actual_pnl
        rate = 100 * blocked.sum() / n
        rows.append({
            "composite": name,
            "n_blocked": int(blocked.sum()),
            "block_rate_pct": rate,
            "blocked_winners": bw, "blocked_losers": bl,
            "actual_pnl": actual_pnl,
            "counterfactual_pnl": cf,
            "pnl_delta": delta,
        })
        print(f"{name:40s}  {int(blocked.sum()):>8d}  {rate:>5.1f}%  {bw:>3d}/{bl:>3d}  "
              f"${delta:>+9,.2f}  ${cf:>+9,.2f}")
    print()

    # Per-entry decisions for the winning composite (for HTML viewer)
    summary_df = pd.DataFrame(rows).sort_values("pnl_delta", ascending=False).reset_index(drop=True)
    summary_df.to_parquet(OUT / "summary.parquet", index=False)
    winner = summary_df.iloc[0]["composite"]
    winner_mask = composites[winner]

    # Per-entry log for winner (for HTML)
    winner_rows = []
    for i in range(n):
        winner_rows.append({
            "composite": winner,
            "lane": base.iloc[i]["lane"], "symbol": base.iloc[i]["symbol"],
            "side_dir": base.iloc[i]["side_dir"], "entry_ts": base.iloc[i]["entry_ts"],
            "exit_ts": base.iloc[i]["exit_ts"], "exit_action": base.iloc[i]["exit_action"],
            "qty": base.iloc[i]["qty"], "pnl": base.iloc[i]["pnl"],
            "blocked": bool(winner_mask[i]),
            "votes_surgical": int(votes[i]) if i < len(votes) else 0,
            "rbf_proba": float(proba_by_model.get("102_rbf_sampler_logreg", np.zeros(n))[i]),
        })
    pd.DataFrame(winner_rows).to_parquet(OUT / "winner_per_entry.parquet", index=False)
    print(f"WINNER: {winner} — Δ ${summary_df.iloc[0]['pnl_delta']:+,.2f}")
    print(f"\nwrote {OUT / 'summary.parquet'}")
    print(f"wrote {OUT / 'winner_per_entry.parquet'}")


if __name__ == "__main__":
    main()
