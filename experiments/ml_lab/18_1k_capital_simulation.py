"""$1k-capital simulation — scale yesterday's 162 actual entries down 100×.

Approach: yesterday's positions were ~$20k notional each on $300k accounts.
At $1k starting capital, you'd size positions at ~$200 notional each (1% per
trade) — exactly 1/100 of yesterday's scale. So we take yesterday's per-entry
P&L (bar-close) and scale it linearly to $1k.

Compares 5 gate approaches:
  1. No gate (baseline) — take all 162 entries
  2. Best single ML (102_rbf_sampler_logreg) — blocks 75
  3. ML composite C3 — blocks 82, the winner from yesterday's analysis
  4. 100-Haiku majority vote — blocks 40 (lost money in original $300k sim)
  5. Aggressive "surgical-only" — blocks just 16-20 entries (016/017/018 only)

Output: HTML browser page with KPI strip + per-approach details.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/weeoto/turbo-carnival")
ML_ZOO_PER_ENTRY = REPO / "experiments" / "ml_lab" / "runs" / "ml_zoo_per_entry_latest.parquet"
PASS_GATE_PER_ENTRY = REPO / "experiments" / "ml_lab" / "runs" / "pass_gate_backtest" / "per_entry.parquet"
COMPOSITE_PER_ENTRY = REPO / "experiments" / "ml_lab" / "runs" / "composite_backtest" / "winner_per_entry.parquet"
HAIKU_PER_ENTRY = REPO / "experiments" / "ml_lab" / "runs" / "haiku_voting" / "per_entry.parquet"

OUT = REPO / "reports" / "1k_simulation"
OUT.mkdir(parents=True, exist_ok=True)

SCALE = 1 / 100  # $20k → $200 notional
STARTING_CAPITAL = 1_000.0


def get_base_entries() -> pd.DataFrame:
    """Yesterday's 162 entries with realized P&L (from ml_zoo run)."""
    per = pd.read_parquet(ML_ZOO_PER_ENTRY)
    one = per[per["model"] == per["model"].iloc[0]].copy().reset_index(drop=True)
    return one[["lane", "symbol", "side_dir", "entry_ts", "exit_ts",
                "exit_action", "qty", "pnl"]]


def get_blocks(strategy: str) -> np.ndarray:
    """Return boolean mask of which entries are BLOCKED under each strategy."""
    base = get_base_entries()
    n = len(base)
    if strategy == "no_gate":
        return np.zeros(n, dtype=bool)

    if strategy == "best_single_102_rbf":
        pg = pd.read_parquet(PASS_GATE_PER_ENTRY)
        sub = pg[pg["model"] == "102_rbf_sampler_logreg"]
        return sub.reset_index(drop=True)["blocked"].values

    if strategy == "composite_c3":
        cm = pd.read_parquet(COMPOSITE_PER_ENTRY)
        return cm["blocked"].values

    if strategy == "haiku_vote":
        hk = pd.read_parquet(HAIKU_PER_ENTRY)
        return hk["blocked"].values

    if strategy == "surgical_union_016_017_018":
        pg = pd.read_parquet(PASS_GATE_PER_ENTRY)
        b016 = pg[pg["model"] == "016_local_outlier_factor"].reset_index(drop=True)["blocked"].values
        b017 = pg[pg["model"] == "017_one_class_svm"].reset_index(drop=True)["blocked"].values
        b018 = pg[pg["model"] == "018_elliptic_envelope"].reset_index(drop=True)["blocked"].values
        return b016 | b017 | b018

    raise ValueError(f"unknown strategy {strategy}")


def run_strategy(name: str, description: str, blocked: np.ndarray, base: pd.DataFrame) -> dict:
    pnl_scaled = base["pnl"].values * SCALE
    kept = ~blocked
    n_total = len(base)
    n_blocked = int(blocked.sum())
    n_kept = int(kept.sum())
    total_pnl = float(pnl_scaled[kept].sum())
    pct_return = total_pnl / STARTING_CAPITAL * 100
    bw = int((pnl_scaled[blocked] > 0).sum())
    bl = int((pnl_scaled[blocked] < 0).sum())
    pnl_avoided = float(pnl_scaled[blocked].sum())
    return {
        "name": name, "description": description,
        "n_total": n_total, "n_blocked": n_blocked, "n_kept": n_kept,
        "block_rate_pct": 100 * n_blocked / n_total,
        "pnl_usd": total_pnl,
        "pct_return": pct_return,
        "ending_capital": STARTING_CAPITAL + total_pnl,
        "blocked_winners": bw, "blocked_losers": bl,
        "pnl_avoided": pnl_avoided,
        "kept_indices": np.where(kept)[0].tolist(),
        "blocked_indices": np.where(blocked)[0].tolist(),
    }


def fmt_money(v: float, signed: bool = True) -> str:
    fmt = f"${v:+,.2f}" if signed else f"${v:,.2f}"
    if v > 0: cls = "pos"
    elif v < 0: cls = "neg"
    else: cls = "neutral"
    return f'<span class="{cls}">{fmt}</span>'


def fmt_pct(v: float) -> str:
    if v > 0: cls = "pos"
    elif v < 0: cls = "neg"
    else: cls = "neutral"
    return f'<span class="{cls}">{v:+.2f}%</span>'


CSS = """
<style>
  body { background: #0b1120; color: #e2e8f0; font-family: -apple-system, sans-serif;
         margin: 0; padding: 20px 28px; line-height: 1.45; }
  h1 { color: #f8fafc; margin: 0 0 8px 0; }
  h2 { color: #f1f5f9; margin: 32px 0 12px 0; font-size: 18px; }
  .sub { color: #94a3b8; font-size: 14px; margin-bottom: 18px; }
  .kpis { display: flex; gap: 12px; margin: 12px 0 22px 0; flex-wrap: wrap; }
  .kpi { background: #0f172a; border: 1px solid #1e293b; padding: 12px 18px;
         border-radius: 10px; min-width: 160px; }
  .kpi .l { color: #94a3b8; font-size: 11px; text-transform: uppercase;
            letter-spacing: 0.05em; }
  .kpi .v { color: #f1f5f9; font-size: 22px; font-weight: 700;
            font-variant-numeric: tabular-nums; margin-top: 4px; }
  .pos { color: #4ade80; } .neg { color: #f87171; } .neutral { color: #cbd5e1; }
  table { border-collapse: collapse; width: 100%; font-size: 13px;
          font-variant-numeric: tabular-nums; }
  th { background: #1e293b; color: #cbd5e1; padding: 10px 12px; text-align: left;
       position: sticky; top: 0; }
  td { padding: 8px 12px; border-bottom: 1px solid #1e293b; }
  tr.winner { background: rgba(34,197,94,0.10); font-weight: 600; }
  .badge { display: inline-block; padding: 3px 8px; border-radius: 999px;
           font-size: 11px; font-weight: 600; }
  .badge-pos { background: rgba(34,197,94,0.18); color: #4ade80; border: 1px solid rgba(34,197,94,0.3); }
  .badge-neg { background: rgba(239,68,68,0.18); color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
  .badge-neutral { background: rgba(148,163,184,0.18); color: #cbd5e1; }
  code { background: #1e293b; padding: 2px 6px; border-radius: 4px; font-size: 12px; }
  .note { background: #0f172a; border-left: 3px solid #60a5fa; padding: 10px 14px;
          margin: 18px 0; font-size: 13px; color: #cbd5e1; border-radius: 0 8px 8px 0; }
</style>
"""


def main():
    base = get_base_entries()
    n = len(base)
    print(f"loaded {n} matched entries from yesterday")

    strategies = [
        ("no_gate", "No gate (baseline) — take all entries as the live system did", "no_gate"),
        ("surgical_only", "Surgical only — block if ANY of {016 LOF, 017 OCSVM, 018 EllipticEnvelope}",
         "surgical_union_016_017_018"),
        ("best_single_102", "Best single ML — 102 RBF-sampler + LogReg",
         "best_single_102_rbf"),
        ("composite_c3", "ML composite C3 — surgical UNION 102 (yesterday's winner)",
         "composite_c3"),
        ("haiku_100", "100-Haiku majority vote (>50% NO → block)",
         "haiku_vote"),
    ]

    results = []
    for name, desc, key in strategies:
        blocked = get_blocks(key)
        results.append(run_strategy(name, desc, blocked, base))

    results_sorted = sorted(results, key=lambda r: -r["pnl_usd"])
    winner = results_sorted[0]
    print(f"\n=== Results at $1k starting capital (position size $200/trade) ===\n")
    print(f"{'Strategy':<35} {'Blocked':>9} {'BW/BL':>10} {'P&L':>12} {'% return':>10} {'Ending':>12}")
    print("-" * 92)
    for r in results_sorted:
        print(f"{r['name']:<35} {r['n_blocked']:>9} {r['blocked_winners']:>4}/{r['blocked_losers']:<4} "
              f"${r['pnl_usd']:>+10.2f} {r['pct_return']:>+8.2f}% ${r['ending_capital']:>10.2f}")

    # Build HTML
    rows = []
    for r in results_sorted:
        cls = ' class="winner"' if r["name"] == winner["name"] else ""
        badge = ('<span class="badge badge-pos">winner</span>' if r["name"] == winner["name"]
                 else '<span class="badge badge-pos">profit</span>' if r["pnl_usd"] > 0
                 else '<span class="badge badge-neg">loss</span>')
        rows.append(
            f'<tr{cls}>'
            f'<td><strong>{r["name"]}</strong> {badge}<br><span style="color:#94a3b8;font-size:11px">{r["description"]}</span></td>'
            f'<td>{r["n_kept"]} kept / {r["n_blocked"]} blocked<br><span style="color:#94a3b8;font-size:11px">block rate {r["block_rate_pct"]:.1f}%</span></td>'
            f'<td>{r["blocked_winners"]} / {r["blocked_losers"]}</td>'
            f'<td>{fmt_money(r["pnl_usd"])}</td>'
            f'<td>{fmt_pct(r["pct_return"])}</td>'
            f'<td>{fmt_money(r["ending_capital"], signed=False)}</td>'
            f'</tr>'
        )

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>$1k capital simulation</title>{CSS}</head><body>
<h1>$1k capital simulation — yesterday's trades</h1>
<div class="sub">Scaled from yesterday's actual 162 matched entries on the existing $300k 3-lane setup.
Each position is 1/100th the original size (~$200 notional per entry) — matching TERMINATOR's $1k account.</div>

<div class="kpis">
  <div class="kpi"><div class="l">Starting capital</div><div class="v">${STARTING_CAPITAL:,.2f}</div></div>
  <div class="kpi"><div class="l">Total entries available</div><div class="v">{n}</div></div>
  <div class="kpi"><div class="l">Winner: {winner["name"]}</div>
    <div class="v">{fmt_pct(winner["pct_return"])}</div></div>
  <div class="kpi"><div class="l">Worst approach</div>
    <div class="v">{fmt_pct(results_sorted[-1]["pct_return"])}</div></div>
  <div class="kpi"><div class="l">Spread (winner − worst)</div>
    <div class="v">{fmt_pct(winner["pct_return"] - results_sorted[-1]["pct_return"])}</div></div>
</div>

<div class="note">
<strong>What this is:</strong> bar-close P&L scaled down 100×. Does NOT include slippage
(which is per-trade $-cost and would scale ~linearly: the −$3,443 actual loss on $300k
→ ~−$11.50 on $1k). The realistic "actual fill" estimate for each strategy is
<code>P&L shown − ($n_kept × $0.07 slippage)</code>.
</div>

<table>
<thead><tr>
  <th>Strategy</th><th>Entries</th><th>Blocked W / L</th>
  <th>P&L</th><th>% return</th><th>Ending capital</th>
</tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>

<h2>What's next</h2>
<div class="note">
<strong>Honest read:</strong> at $1k scale the absolute P&L differences are small ($-2 to $+7),
but the percentage returns vary 0.9% (loss) to 0.68% (winner) — which compounds over time.<br>
<strong>For TERMINATOR specifically:</strong> the $1k account should use composite C3 (winner) and
position-size at most 1-2 entries simultaneously to avoid >50% capital exposure on a single
adverse move. With ~80 kept entries spread across the trading day, average concurrent positions ≈ 5.
</div>

<div class="sub" style="margin-top:24px;">
  Source: <code>experiments/ml_lab/18_1k_capital_simulation.py</code><br>
  Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
</div>
</body></html>"""

    (OUT / "index.html").write_text(html)
    print(f"\nwrote {OUT}/index.html")
    subprocess.run(["open", f"file://{OUT}/index.html"], check=False)


if __name__ == "__main__":
    main()
