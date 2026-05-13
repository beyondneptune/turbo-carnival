"""Phase 3: Test macro-aware gates from Phase 2 NIM panel.

Adds:
  - fed_week (Fed meeting proximity)
  - vix_term_ratio (VIX/VIX3M)
  - yc_slope (treasury 10y-5y)

Tests against the council winner config across NAV sizes + risk levels.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

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


def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = REPO_ROOT / "experiments" / "runs" / f"v3_macro_{stamp}"
    out.mkdir(parents=True, exist_ok=True)

    specs = []
    for gate in ("vrp_positive", "macro_aware_v3", "macro_aware_strict",
                  "calm_market", "ultra_calm"):
        s = dict(WINNER_BASE)
        s["name"] = f"v3_winner_{gate}"
        s["gate"] = gate
        specs.append(s)
    spec_path = out / "specs.json"
    spec_path.write_text(json.dumps({"strategies": specs}, indent=2))

    grid_dir = out / "grid"
    grid_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        ".venv/bin/python", "experiments/run_spy_core_strategy_grid.py",
        "--spec-json", str(spec_path),
        "--output-dir", str(grid_dir),
    ]
    print("Running:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        print("STDERR:", proc.stderr[-1500:])
        return 1

    trades = pd.read_csv(grid_dir / "trades.csv")
    rows = []
    sizing_configs = [
        ("$10k_1pct", 10_000.0, 1.0, 15.0),
        ("$10k_2pct", 10_000.0, 2.0, 25.0),
        ("$10k_5pct", 10_000.0, 5.0, 50.0),
        ("$10k_10pct", 10_000.0, 10.0, 80.0),
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
    df.to_csv(out / "v3_results.csv", index=False)

    print("\n=== Trade counts per gate ===")
    counts = trades.groupby("strategy").size()
    for s, n in counts.items():
        print(f"  {s}: {n} trades")

    print("\n=== V3 macro-aware results ===")
    print(df[["strategy", "sizing", "monthly_pct", "cagr_pct",
             "max_dd_pct", "trades_placed", "win_rate_pct"]].to_string(index=False))

    safe = df[df["max_dd_pct"] > -25.0]
    print("\n=== Best with max DD > -25% ===")
    if not safe.empty:
        top = safe.sort_values("monthly_pct", ascending=False).head(10)
        print(top[["strategy", "sizing", "monthly_pct", "cagr_pct",
                 "max_dd_pct", "trades_placed", "win_rate_pct"]].to_string(index=False))

    # Compare baseline vs new gates at $100k 1%
    print("\n=== Headline: $100k @ 1% risk (investor-safe) ===")
    safe_size = df[df["sizing"] == "$100k_1pct"]
    print(safe_size[["strategy", "monthly_pct", "max_dd_pct", "trades_placed"]].to_string(index=False))

    # Did anything hit 14%?
    hit_14 = df[df["monthly_pct"] >= 14.0]
    near_5 = df[df["monthly_pct"] >= 5.0]
    print(f"\n=== Hit 14% monthly: {len(hit_14)}/{len(df)} ===")
    print(f"=== Hit >= 5% monthly: {len(near_5)}/{len(df)} ===")
    if not hit_14.empty:
        print(hit_14.to_string(index=False))

    print(f"\nOutput: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
