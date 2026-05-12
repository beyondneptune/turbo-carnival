"""Offline RL with Conservative Q-Learning (CQL) — scaffold.

Offline RL trains a policy from a static log of (state, action, reward, next_state)
tuples without further environment interaction. Perfect fit when:
  - Real exploration is expensive (we don't want bad trades during training)
  - We have a growing log of decisions (from the bandit shadow mode, eventually)

CQL (Kumar et al. 2020): standard Q-learning + a regularizer that penalizes
out-of-distribution actions. This prevents the policy from preferring actions
the offline log hasn't seen enough of.

This is a SCAFFOLD, not a working trainer. To actually train:
  1. Wait 30+ days of shadow bandit decisions (gives ~10k tuples)
  2. Install: pip install torch d3rlpy
  3. Fill in the TODOs below

References:
  - Kumar et al. 2020 — Conservative Q-Learning for Offline RL
  - d3rlpy library: https://github.com/takuseno/d3rlpy

When to actually run this:
  - After accumulating 10k+ settled bandit decisions
  - As a weekend batch job (training takes ~1-4 hours)
  - Output policy gets evaluated in 1-week shadow mode before promotion
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

REPO = Path("/Users/weeoto/turbo-carnival")
BANDIT_DB = REPO / "data" / "bandit_shadow.sqlite3"
OUT = REPO / "experiments" / "ml_lab" / "runs" / "offline_rl_cql"
OUT.mkdir(parents=True, exist_ok=True)


def load_replay_buffer() -> dict:
    """Pull (state, action, reward) tuples from bandit shadow DB."""
    if not BANDIT_DB.exists():
        return {"n": 0, "states": [], "actions": [], "rewards": []}
    conn = sqlite3.connect(str(BANDIT_DB))
    rows = conn.execute(
        "SELECT features, chosen_action, realized_pnl_usd "
        "FROM shadow_decisions WHERE settled = 1"
    ).fetchall()
    conn.close()
    states, actions, rewards = [], [], []
    for feats_json, action, pnl in rows:
        if pnl is None:
            continue
        f = json.loads(feats_json)
        states.append([
            f.get("ret1", 0), f.get("ret3", 0), f.get("ret10", 0),
            f.get("range_pct", 0), f.get("volume_z", 0),
            f.get("ret1_z", 0), f.get("vwap_dist", 0),
            f.get("tod_sin", 0), f.get("tod_cos", 0), f.get("vov", 1.0),
        ])
        actions.append(action)
        rewards.append(pnl)
    return {"n": len(states), "states": states, "actions": actions, "rewards": rewards}


def train_cql(replay: dict, *, n_epochs: int = 100):
    """Train CQL policy. Requires d3rlpy + torch.

    TODO:
      1. pip install d3rlpy torch
      2. Convert replay to d3rlpy.dataset.MDPDataset
      3. Configure CQL: d3rlpy.algos.DiscreteCQL(alpha=1.0)
      4. Fit, save model to OUT / "cql_policy.pt"
      5. Save training curves
    """
    if replay["n"] < 1000:
        print(f"  Only {replay['n']} tuples. Need 1000+ to train. Skipping.")
        return None

    print(f"  Would train CQL on {replay['n']} (state, action, reward) tuples.")
    print(f"  Implementation pending — see TODOs in train_cql().")

    # Placeholder: dump replay buffer for offline inspection
    (OUT / "replay_dump.json").write_text(json.dumps({
        "n": replay["n"],
        "reward_mean": sum(replay["rewards"]) / len(replay["rewards"]) if replay["rewards"] else 0,
        "reward_std": _std(replay["rewards"]),
        "action_distribution": _action_counts(replay["actions"]),
    }, indent=2))
    print(f"  Replay stats → {OUT / 'replay_dump.json'}")
    return None


def _std(values):
    if not values:
        return 0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5


def _action_counts(actions):
    out = {}
    for a in actions:
        out[a] = out.get(a, 0) + 1
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    print("=" * 60)
    print("Offline RL (CQL) — Scaffold")
    print("=" * 60)
    print()
    print("Loading replay buffer from bandit shadow DB...")
    replay = load_replay_buffer()
    print(f"  {replay['n']} settled (state, action, reward) tuples")
    print()

    if replay["n"] == 0:
        print("Empty replay buffer. Run experiments/ml_lab/24_linucb_bandit_shadow.py")
        print("and wire it into the host's entry path first.")
        return

    if replay["n"] < 10_000:
        print(f"WARNING: only {replay['n']} tuples. CQL paper recommends 10k+ for "
              f"useful policy. Current data is research-only.")

    train_cql(replay, n_epochs=args.epochs)
    print()
    print("Done. Implementation TODO: see train_cql() docstring.")


if __name__ == "__main__":
    main()
