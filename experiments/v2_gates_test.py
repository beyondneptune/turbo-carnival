"""Test the new VIX-ROC gates against the existing council winner.

Builds spec variants of the winner with:
  - original gate (vrp_positive) for baseline
  - calm_market gate (vrp + vix calm)
  - vix_calm_only gate (just vix)
  - ultra_calm gate (everything)

At three risk levels: 1%, 2%, 5%
At two NAV sizes: $10k, $100k

Then applies realistic friction + sizing and reports the monthly return delta.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from experiments.aggressive_search import apply_aggressive_sizing


WINNER_BASE = {
    "entry_mode": "daily_ladder",
    "target_delta": 0.12,
    "target_dte": 30,
    "wing_width": 5.0,
    "profit_take": 0.30,
    "stop_loss_credit_mult": 2.0,
    "close_dte": 10,
    "min_credit": 0.20,
}


def make_specs() -> list[dict]:
    specs = []
    for gate in ("vrp_positive", "calm_market", "vix_calm_only", "ultra_calm"):
        s = dict(WINNER_BASE)
        s["name"] = f"winner_{gate}"
        s["gate"] = gate
        specs.append(s)
    return specs


def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = REPO_ROOT / "experiments" / "runs" / f"v2_gates_{stamp}"
    out.mkdir(parents=True, exist_ok=True)

    specs = make_specs()
    spec_path = out / "specs.json"
    spec_path.write_text(json.dumps({"strategies": specs}, indent=2))

    grid_dir = out / "grid"
    grid_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        ".venv/bin/python", "experiments/run_spy_core_strategy_grid.py",
        "--spec-json", str(spec_path),
        "--output-dir", str(grid_dir),
    ]
    print("Running grid:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        print("STDERR:", proc.stderr[-800:])
        return 1

    trades = pd.read_csv(grid_dir / "trades.csv")
    rows = []
    sizing_configs = [
        ("$10k_1pct", 10_000.0, 1.0, 15.0),
        ("$10k_2pct", 10_000.0, 2.0, 25.0),
        ("$10k_5pct", 10_000.0, 5.0, 50.0),
        ("$100k_1pct", 100_000.0, 1.0, 15.0),
        ("$100k_2pct", 100_000.0, 2.0, 25.0),
        ("$100k_5pct", 100_000.0, 5.0, 50.0),
    ]

    for spec_name in trades["strategy"].unique():
        w = trades[trades["strategy"] == spec_name].copy()
        for sizing_label, nav, risk, conc in sizing_configs:
            r = apply_aggressive_sizing(w, nav, risk, conc)
            if r["trades_placed"] == 0:
                continue
            rows.append({
                "strategy": spec_name, "sizing": sizing_label,
                "nav": nav, "risk_pct": risk,
                **{k: v for k, v in r.items() if k != "history"},
            })

    df = pd.DataFrame(rows)
    df.to_csv(out / "v2_results.csv", index=False)

    # Per-gate trade counts (how many trades each gate kept)
    print("\n=== Trade count per gate (the more selective, the better the gate) ===")
    counts = trades.groupby("strategy").size()
    for s, n in counts.items():
        print(f"  {s}: {n} trades")

    # Show comparison table
    print("\n=== V2 GATE COMPARISON (Realistic friction) ===")
    print(df[["strategy", "sizing", "monthly_pct", "cagr_pct",
             "max_dd_pct", "trades_placed", "win_rate_pct"]].to_string(index=False))

    # Pick the best (highest monthly subject to max DD constraint)
    safe = df[df["max_dd_pct"] > -25.0]
    print("\n=== Best gates with max DD better than -25% ===")
    if not safe.empty:
        top = safe.sort_values("monthly_pct", ascending=False).head(8)
        print(top[["strategy", "sizing", "monthly_pct", "cagr_pct",
                 "max_dd_pct", "trades_placed", "win_rate_pct"]].to_string(index=False))

    # Compare original vs improved
    baseline = df[df["strategy"] == "winner_vrp_positive"].copy()
    print("\n=== Baseline (vrp_positive only) ===")
    print(baseline[["sizing", "monthly_pct", "max_dd_pct", "trades_placed"]].to_string(index=False))

    print(f"\nOutput: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
