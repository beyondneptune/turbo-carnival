"""LinUCB contextual bandit — shadow mode.

Wraps the existing composite C3 gate as a SAFETY PRIOR. The bandit decides
a size multiplier ∈ {0.0, 0.5, 1.0, 1.5, 2.0} given the same features the
C3 gate sees. Logs every decision + realized reward to a sqlite database
for 30+ days, then evaluates whether the bandit's policy beats C3's
binary block/keep baseline.

Shadow mode = bandit predicts, logs, but does NOT execute trades. We
compare the bandit's "would-have" P&L against the C3 baseline's actual
P&L. If bandit beats baseline by >5% lift after 60 days, promote to live
at 10% capital allocation.

Algorithm: LinUCB (Li, Chu, Langford, Schapire 2010)
  - Linear payoff model: Q(state, action) = state^T · theta_action
  - Confidence bound: UCB(state, action) = Q + alpha · sqrt(state^T · A_inv · state)
  - Pick action with highest UCB; observe reward; update A and b
  - Pure exploitation as alpha → 0, pure exploration as alpha → ∞

Wiring:
  - In host loop, call `bandit_log_shadow(context, c3_block, c3_kept_size, realized_pnl)`
    AFTER each entry's exit completes.
  - Bandit is read-only during shadow mode (no trade execution side-effects).

Reward function (the user's spec):
  - reward = realized_pnl_dollars   (positive on profit, NEGATIVE on loss)
  - normalized to [-1, 1] by dividing by base_notional × max expected % swing

After 30 days run:
    python experiments/ml_lab/24_linucb_bandit_shadow.py --evaluate
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path("/Users/weeoto/turbo-carnival")
DB_PATH = REPO / "data" / "bandit_shadow.sqlite3"

# Action space: size multipliers applied on top of C3-gated decision
ACTIONS = [0.0, 0.5, 1.0, 1.5, 2.0]
N_FEATS = 10  # same features as C3 gate

# Exploration constant — higher = more exploration, lower = more exploitation
ALPHA = 1.0


class LinUCB:
    """Per-action LinUCB. One A matrix + b vector per action."""

    def __init__(self, n_actions: int, n_feats: int, alpha: float = ALPHA):
        self.n_actions = n_actions
        self.n_feats = n_feats
        self.alpha = alpha
        # A[a] = identity to start (ridge-regularized)
        self.A = [np.eye(n_feats) for _ in range(n_actions)]
        self.b = [np.zeros(n_feats) for _ in range(n_actions)]

    def predict(self, x: np.ndarray) -> tuple[int, np.ndarray]:
        """Return (best_action_idx, ucb_array).

        ucb[a] = theta_a^T x + alpha * sqrt(x^T A_a^{-1} x)
        """
        ucbs = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            mean_reward = float(theta @ x)
            exploration_bonus = self.alpha * np.sqrt(float(x @ A_inv @ x))
            ucbs[a] = mean_reward + exploration_bonus
        return int(np.argmax(ucbs)), ucbs

    def update(self, x: np.ndarray, action: int, reward: float) -> None:
        """Online update for one observation."""
        self.A[action] += np.outer(x, x)
        self.b[action] += reward * x

    def save(self, path: Path) -> None:
        payload = {
            "A": [a.tolist() for a in self.A],
            "b": [b.tolist() for b in self.b],
            "alpha": self.alpha,
        }
        path.write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: Path, n_feats: int) -> "LinUCB":
        if not path.exists():
            return cls(len(ACTIONS), n_feats)
        d = json.loads(path.read_text())
        bandit = cls(len(d["A"]), n_feats, alpha=d.get("alpha", ALPHA))
        bandit.A = [np.array(a) for a in d["A"]]
        bandit.b = [np.array(b) for b in d["b"]]
        return bandit


def _init_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS shadow_decisions (
            id INTEGER PRIMARY KEY,
            ts_utc TEXT, lane TEXT, symbol TEXT,
            c3_blocked INTEGER,   -- did baseline C3 block?
            chosen_action INTEGER,  -- bandit's pick
            chosen_size_mult REAL,
            features TEXT,          -- JSON of feature dict
            ucbs TEXT,              -- JSON of UCB per action
            realized_pnl_usd REAL,  -- backfilled on exit
            baseline_pnl_usd REAL,  -- what C3 baseline would have made
            settled INTEGER DEFAULT 0
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_settled ON shadow_decisions(settled)")
    conn.commit()
    return conn


def bandit_log_shadow(
    *, lane: str, symbol: str, features: dict, c3_blocked: bool, base_notional: float,
) -> tuple[int, float]:
    """Called from host BEFORE entry (in shadow mode).

    Returns (chosen_action_idx, chosen_size_multiplier). Host should NOT
    use the chosen multiplier in live trading during shadow phase — this
    is just for logging the would-have decision.
    """
    bandit = LinUCB.load(DB_PATH.with_suffix(".weights.json"), N_FEATS)
    x = np.array([
        features.get("ret1", 0), features.get("ret3", 0), features.get("ret10", 0),
        features.get("range_pct", 0), features.get("volume_z", 0),
        features.get("ret1_z", 0), features.get("vwap_dist", 0),
        features.get("tod_sin", 0), features.get("tod_cos", 0),
        features.get("vov", 1.0),
    ])
    action_idx, ucbs = bandit.predict(x)
    size_mult = ACTIONS[action_idx]

    conn = _init_db()
    conn.execute(
        "INSERT INTO shadow_decisions(ts_utc, lane, symbol, c3_blocked, chosen_action, "
        "chosen_size_mult, features, ucbs, realized_pnl_usd, baseline_pnl_usd, settled) "
        "VALUES(?,?,?,?,?,?,?,?, NULL, NULL, 0)",
        (datetime.now(timezone.utc).isoformat(), lane, symbol,
         1 if c3_blocked else 0, action_idx, size_mult,
         json.dumps(features), json.dumps(ucbs.tolist())),
    )
    conn.commit()
    conn.close()
    return action_idx, size_mult


def bandit_settle_shadow(
    *, decision_id: int, realized_pnl_c3_actual: float, base_notional: float,
) -> None:
    """Backfill realized P&L after the entry exits.

    `realized_pnl_c3_actual` = the actual realized P&L from the C3-gated
    entry at full size. The bandit's "would-have" P&L is this × the chosen
    multiplier.
    """
    conn = _init_db()
    row = conn.execute(
        "SELECT chosen_size_mult, c3_blocked FROM shadow_decisions WHERE id = ?",
        (decision_id,),
    ).fetchone()
    if row is None:
        conn.close()
        return
    chosen_mult, c3_blocked = row
    # Bandit's would-have P&L: scaled by chosen size, but only if C3 didn't block
    bandit_pnl = realized_pnl_c3_actual * chosen_mult if not c3_blocked else 0.0
    # Normalized reward to [-1, 1]
    reward = max(-1, min(1, bandit_pnl / max(1, base_notional * 0.02)))
    conn.execute(
        "UPDATE shadow_decisions SET realized_pnl_usd=?, baseline_pnl_usd=?, settled=1 "
        "WHERE id=?",
        (bandit_pnl, realized_pnl_c3_actual, decision_id),
    )
    conn.commit()
    # Update bandit weights
    feats_row = conn.execute(
        "SELECT features, chosen_action FROM shadow_decisions WHERE id=?", (decision_id,)
    ).fetchone()
    if feats_row:
        features, action_idx = feats_row
        bandit = LinUCB.load(DB_PATH.with_suffix(".weights.json"), N_FEATS)
        f = json.loads(features)
        x = np.array([
            f.get("ret1", 0), f.get("ret3", 0), f.get("ret10", 0),
            f.get("range_pct", 0), f.get("volume_z", 0),
            f.get("ret1_z", 0), f.get("vwap_dist", 0),
            f.get("tod_sin", 0), f.get("tod_cos", 0), f.get("vov", 1.0),
        ])
        bandit.update(x, action_idx, reward)
        bandit.save(DB_PATH.with_suffix(".weights.json"))
    conn.close()


def evaluate() -> None:
    """After 30+ days of shadow mode, compare bandit vs baseline."""
    conn = _init_db()
    rows = conn.execute(
        "SELECT chosen_action, chosen_size_mult, realized_pnl_usd, baseline_pnl_usd "
        "FROM shadow_decisions WHERE settled = 1"
    ).fetchall()
    if not rows:
        print("No settled shadow decisions yet. Run for 30+ days first.")
        return

    bandit_pnl = sum(r[2] for r in rows if r[2] is not None)
    baseline_pnl = sum(r[3] for r in rows if r[3] is not None)
    n = len(rows)
    import collections
    action_counts = collections.Counter(r[0] for r in rows)
    print(f"=== LinUCB Shadow Evaluation (n={n} decisions) ===")
    print(f"  Baseline C3 P&L:    ${baseline_pnl:+,.2f}")
    print(f"  Bandit shadow P&L:  ${bandit_pnl:+,.2f}")
    if baseline_pnl != 0:
        lift = (bandit_pnl - baseline_pnl) / abs(baseline_pnl)
        print(f"  Lift vs baseline:   {lift*100:+.1f}%")
        print(f"  Promote to live?    {'YES (lift > 5%)' if lift > 0.05 else 'NO (need more data or no edge)'}")
    print()
    print(f"  Action distribution:")
    for a, count in sorted(action_counts.items()):
        print(f"    action {a} (size×{ACTIONS[a]}): {count} ({count/n*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", action="store_true",
                          help="Print bandit vs baseline comparison")
    args = parser.parse_args()
    if args.evaluate:
        evaluate()
    else:
        _init_db()
        print(f"Bandit shadow DB initialized at {DB_PATH}")
        print(f"Wire bandit_log_shadow() into host's entry path, bandit_settle_shadow() into exits")
        print(f"Run with --evaluate after 30+ days to compare to baseline")
