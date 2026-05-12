"""Run every experiment in experiments/lab/<NNN_name>/run.py sequentially.

Writes a unified leaderboard at experiments/lab/LEADERBOARD.md.

Usage: .venv/bin/python experiments/lab/run_all.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = "/Users/weeoto/turbo-carnival/.venv/bin/python"


def main() -> int:
    exps = sorted([p for p in ROOT.iterdir() if p.is_dir() and p.name[0].isdigit()])
    print(f"running {len(exps)} experiments…\n")
    timings = []
    for exp in exps:
        run = exp / "run.py"
        if not run.exists():
            print(f"  skip {exp.name} — no run.py")
            continue
        t0 = time.time()
        try:
            r = subprocess.run([PY, str(run)], capture_output=True, text=True, timeout=600)
            if r.returncode != 0:
                print(f"  [{exp.name}] FAILED  stderr: {r.stderr[:200]}")
                (exp / "results.json").write_text(json.dumps({
                    "experiment": exp.name, "error": r.stderr[:1000],
                    "passes_overfit_gate": False, "test_auc": 0.5,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }, indent=2))
            else:
                # Run script already prints its summary
                for line in r.stdout.splitlines():
                    if line.startswith("["):
                        print(f"  {line}")
        except subprocess.TimeoutExpired:
            print(f"  [{exp.name}] TIMEOUT (>10min)")
            (exp / "results.json").write_text(json.dumps({
                "experiment": exp.name, "error": "timeout",
                "passes_overfit_gate": False, "test_auc": 0.5,
            }, indent=2))
        dt = time.time() - t0
        timings.append((exp.name, dt))
        print(f"    ({dt:.1f}s)")

    print(f"\nTotal time: {sum(t for _, t in timings):.1f}s")

    # Build leaderboard
    rows = []
    for exp in exps:
        rj = exp / "results.json"
        if not rj.exists():
            continue
        try:
            obj = json.loads(rj.read_text())
            rows.append(obj)
        except Exception:
            continue

    # Two sections: classifier/anomaly (sortable by test_auc) and strategy (sortable by net_pnl)
    clf = [r for r in rows if "test_auc" in r and r.get("test_auc", 0) > 0 and "net_pnl_8bps" not in r]
    strat = [r for r in rows if "net_pnl_8bps" in r and r.get("n_trades", 0) > 0]
    failed = [r for r in rows if r.get("test_auc", 0) == 0 and "net_pnl_8bps" not in r]

    clf.sort(key=lambda r: -float(r.get("test_auc", 0)))
    strat.sort(key=lambda r: -float(r.get("net_pnl_8bps", 0)))

    md = ["# ML Lab Leaderboard",
           "",
           f"Generated {datetime.now(timezone.utc).isoformat()}. "
           f"Each experiment lives in `experiments/lab/<NNN_name>/run.py` and writes its own "
           f"`results.json`. Same train/test split for all classifier/anomaly experiments "
           f"(70/30 day-aware). Strategy experiments report net P&L at 8 bps slippage.",
           "",
           "## Eval gate (anti-overfit)",
           "Classifier passes if `test_auc > 0.55` AND `|train_test_gap| < 0.05`.",
           "Strategy passes if `net_pnl_8bps > 0` AND `n_trades >= 5`.",
           "",
           f"## Classifier / Anomaly leaderboard ({len(clf)} models)",
           "",
           "| Rank | Experiment | Architecture | Test AUC | Train-test gap | Pass? |",
           "|---:|---|---|---:|---:|:---:|"]
    for i, r in enumerate(clf, 1):
        name = r["experiment"]
        desc = r.get("description", "")[:60]
        auc = float(r.get("test_auc", 0))
        gap = float(r.get("train_test_gap", 0))
        gate = "PASS" if r.get("passes_overfit_gate") else "fail"
        md.append(f"| {i} | `{name}` | {desc} | {auc:.3f} | {gap:+.3f} | **{gate}** |")
    md.append("")
    md.append(f"## Strategy leaderboard ({len(strat)} variants)")
    md.append("")
    md.append("| Rank | Experiment | Architecture | # Trades | Net P&L 8bps | Win rate | Pass? |")
    md.append("|---:|---|---|---:|---:|---:|:---:|")
    for i, r in enumerate(strat, 1):
        name = r["experiment"]
        desc = r.get("description", "")[:60]
        n = int(r.get("n_trades", 0))
        pnl = float(r.get("net_pnl_8bps", 0))
        wr = float(r.get("win_rate", 0)) * 100
        gate = "PASS" if r.get("passes_overfit_gate") else "fail"
        md.append(f"| {i} | `{name}` | {desc} | {n} | ${pnl:+,.0f} | {wr:.1f}% | **{gate}** |")
    if failed:
        md.append("")
        md.append("## Failed / errored experiments")
        md.append("")
        for r in failed:
            md.append(f"- `{r['experiment']}` — {r.get('error', 'unknown error')[:200]}")

    (ROOT / "LEADERBOARD.md").write_text("\n".join(md) + "\n")
    print(f"\nwrote {ROOT / 'LEADERBOARD.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
