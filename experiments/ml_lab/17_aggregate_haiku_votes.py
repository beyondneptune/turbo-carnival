"""Aggregate the 100 Haiku per-entry yes/no votes and compute counterfactual P&L.

Rule: BLOCK an entry if MORE THAN 50% of Haikus voted NO (i.e., < 50 said YES out of 100).

Outputs:
  experiments/ml_lab/runs/haiku_voting/summary.json
  experiments/ml_lab/runs/haiku_voting/per_entry.parquet
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/weeoto/turbo-carnival")
VOTE_DIR = REPO / "experiments" / "ml_lab" / "haiku_vote_runs"
META_PATH = Path("/tmp/haiku_voting_meta.json")
OUT = REPO / "experiments" / "ml_lab" / "runs" / "haiku_voting"
OUT.mkdir(parents=True, exist_ok=True)

# Load entry metadata
meta = json.loads(META_PATH.read_text())
n_entries = len(meta)
print(f"loaded {n_entries} entry metadata records")

# Load all 100 votes
vote_files = sorted(VOTE_DIR.glob("agent_*.json"))
print(f"found {len(vote_files)} vote files")

vote_matrix = np.zeros((len(vote_files), n_entries), dtype=np.int8)  # 1=YES, 0=NO
errors_per_file = {}
for k, f in enumerate(vote_files):
    try:
        votes = json.loads(f.read_text())
        for v in votes:
            i = int(v["i"])
            if 0 <= i < n_entries:
                vote_matrix[k, i] = 1 if v["vote"] == "YES" else 0
    except Exception as e:
        errors_per_file[f.name] = str(e)[:100]

if errors_per_file:
    print(f"\n  errors in {len(errors_per_file)} files (treating missing votes as 0):")
    for k, v in list(errors_per_file.items())[:5]:
        print(f"    {k}: {v}")

# Per-entry: count YES votes (out of 100)
yes_counts = vote_matrix.sum(axis=0)
print(f"\nYES votes per entry: min={yes_counts.min()}, max={yes_counts.max()}, "
      f"median={int(np.median(yes_counts))}")
print(f"Mean YES rate per entry: {yes_counts.mean():.1f} / 100")

# Block rule: BLOCK if YES votes < 50 (more than half said NO)
blocked_mask = yes_counts < 50
n_blocked = int(blocked_mask.sum())
print(f"\nblocked: {n_blocked} / {n_entries} entries ({100*n_blocked/n_entries:.1f}%)")

# P&L computation
pnls = np.array([m["pnl"] for m in meta])
actual_pnl = float(pnls.sum())
counterfactual_pnl = float(pnls[~blocked_mask].sum())
delta = counterfactual_pnl - actual_pnl
blocked_winners = int((pnls[blocked_mask] > 0).sum())
blocked_losers = int((pnls[blocked_mask] < 0).sum())
blocked_pnl_avoided = float(pnls[blocked_mask].sum())

print(f"\n=== RESULTS ===")
print(f"Actual P&L (bar-close):       ${actual_pnl:+,.2f}")
print(f"Counterfactual P&L:           ${counterfactual_pnl:+,.2f}")
print(f"Δ vs actual:                  ${delta:+,.2f}")
print(f"Blocked winners / losers:     {blocked_winners} / {blocked_losers}")
print(f"P&L of blocked entries:       ${blocked_pnl_avoided:+,.2f}")

# Vote distribution per entry
print(f"\nVote distribution:")
buckets = [(0, 25), (25, 50), (50, 75), (75, 100), (100, 101)]
for lo, hi in buckets:
    n = int(((yes_counts >= lo) & (yes_counts < hi)).sum())
    print(f"  {lo:3d}-{hi-1:3d} YES votes (out of 100): {n} entries")

# Per-entry log
per_entry = []
for i, m in enumerate(meta):
    per_entry.append({
        **{k: v for k, v in m.items() if k != "skip"},
        "yes_votes": int(yes_counts[i]),
        "no_votes": 100 - int(yes_counts[i]),
        "blocked": bool(blocked_mask[i]),
    })
df = pd.DataFrame(per_entry)
df.to_parquet(OUT / "per_entry.parquet", index=False)

summary = {
    "n_agents": len(vote_files),
    "n_entries": n_entries,
    "n_blocked": n_blocked,
    "block_rate_pct": 100.0 * n_blocked / n_entries,
    "blocked_winners": blocked_winners,
    "blocked_losers": blocked_losers,
    "actual_pnl": actual_pnl,
    "counterfactual_pnl": counterfactual_pnl,
    "pnl_delta": delta,
    "blocked_pnl_avoided": blocked_pnl_avoided,
    "mean_yes_rate_per_entry": float(yes_counts.mean()),
    "comparison": {
        "best_single_ml_model": {"name": "102_rbf_sampler_logreg", "delta": 1549.27},
        "best_composite": {"name": "C3_surgical_union_plus_102", "delta": 2029.53},
        "100_haiku_vote": {"delta": delta},
    },
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(f"\nwrote {OUT}/summary.json + per_entry.parquet")
print(f"\n=== COMPARISON ===")
print(f"  Best single ML model:    +$1,549")
print(f"  Best ML composite (C3):  +$2,030")
print(f"  100-Haiku majority vote: ${delta:+,.2f}")
