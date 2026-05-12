"""Render composite-backtest results: 5 composites + the winner's per-entry detail."""
from __future__ import annotations

import html
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO = Path("/Users/weeoto/turbo-carnival")
RUNS = REPO / "experiments" / "ml_lab" / "runs" / "composite_backtest"
PASS_RUNS = REPO / "experiments" / "ml_lab" / "runs" / "pass_gate_backtest"
OUT = REPO / "reports" / "composite"
OUT.mkdir(parents=True, exist_ok=True)

CSS = """
<style>
  body { background: #0b1120; color: #e2e8f0; font-family: -apple-system, BlinkMacSystemFont, sans-serif;
         margin: 0; padding: 20px 28px; line-height: 1.45; }
  h1 { color: #f8fafc; margin: 0 0 6px 0; }
  .sub { color: #94a3b8; margin: 0 0 18px 0; font-size: 14px; }
  .topbar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
  .pill { display: inline-block; padding: 3px 10px; border-radius: 999px;
          font-size: 12px; font-weight: 600; background: #1e293b; color: #cbd5e1; margin-right: 6px; }
  .pill-winner { background: rgba(34,197,94,0.18); color: #4ade80; border: 1px solid rgba(34,197,94,0.35); }
  .kpis { display: flex; gap: 12px; margin: 12px 0 22px 0; flex-wrap: wrap; }
  .kpi { background: #0f172a; border: 1px solid #1e293b; padding: 10px 16px;
         border-radius: 10px; min-width: 150px; }
  .kpi .l { color: #94a3b8; font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; }
  .kpi .v { color: #f1f5f9; font-size: 24px; font-weight: 700;
            font-variant-numeric: tabular-nums; margin-top: 4px; }
  .pos { color: #4ade80; } .neg { color: #f87171; }
  table { border-collapse: collapse; width: 100%; font-size: 13px;
          font-variant-numeric: tabular-nums; }
  th { background: #1e293b; color: #cbd5e1; padding: 8px 10px; text-align: left;
       position: sticky; top: 0; }
  td { padding: 6px 10px; border-bottom: 1px solid #1e293b; }
  tr.winner-row { background: rgba(34,197,94,0.10); font-weight: 600; }
  tr.blocked-loser { background: #14532d40; }
  tr.blocked-winner { background: #7f1d1d40; }
  tr.kept-winner { background: #052e16; }
  tr.kept-loser { background: #1f2937; }
  a { color: #60a5fa; text-decoration: none; } a:hover { text-decoration: underline; }
  code { background: #1e293b; padding: 1px 5px; border-radius: 3px; font-size: 12px; }
</style>
"""


def fmt_money(v):
    s = f"{v:+,.2f}"
    cls = "pos" if v > 0 else "neg" if v < 0 else "neutral"
    return f'<span class="{cls}">${s}</span>'


def build_index(summary: pd.DataFrame, top_singles: pd.DataFrame) -> str:
    winner = summary.iloc[0]
    rows = []
    for _, r in summary.iterrows():
        is_winner = r["composite"] == winner["composite"]
        cls = ' class="winner-row"' if is_winner else ""
        rows.append(
            f'<tr{cls}>'
            f'<td>{html.escape(r["composite"])}</td>'
            f'<td>{int(r["n_blocked"])}</td>'
            f'<td>{r["block_rate_pct"]:.1f}%</td>'
            f'<td>{int(r["blocked_winners"])} / {int(r["blocked_losers"])}</td>'
            f'<td>{fmt_money(float(r["counterfactual_pnl"]))}</td>'
            f'<td>{fmt_money(float(r["pnl_delta"]))}</td>'
            f'</tr>'
        )
    composite_table = "".join(rows)

    # Top single models for comparison
    rows = []
    for _, r in top_singles.head(8).iterrows():
        rows.append(
            f'<tr>'
            f'<td><code>{html.escape(r["model"])}</code></td>'
            f'<td>{int(r["n_blocked"])}</td>'
            f'<td>{r["block_rate_pct"]:.1f}%</td>'
            f'<td>{int(r["blocked_winners"])} / {int(r["blocked_losers"])}</td>'
            f'<td>{fmt_money(float(r["counterfactual_pnl"]))}</td>'
            f'<td>{fmt_money(float(r["pnl_delta"]))}</td>'
            f'</tr>'
        )
    singles_table = "".join(rows)

    win_delta = float(winner["pnl_delta"])
    best_single = float(top_singles.iloc[0]["pnl_delta"])
    composite_gain = win_delta - best_single

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Composite gate models — yesterday backtest</title>{CSS}</head><body>
<div class="topbar">
  <h1>Composite gate models</h1>
  <a href="winner.html">→ winner detail</a>
</div>
<div class="sub">5 composite gate models built by combining the top pass-gate performers from the
146-experiment lab. Tested against yesterday's 162 matched entries. Block rule:
<code>predict_proba_class1 &gt; 0.5</code> per base model; composites apply union / majority / soft-vote rules.</div>

<div class="kpis">
  <div class="kpi"><div class="l">Winner</div><div class="v" style="font-size:14px;">{html.escape(winner["composite"])}</div></div>
  <div class="kpi"><div class="l">Winner Δ</div><div class="v">{fmt_money(win_delta)}</div></div>
  <div class="kpi"><div class="l">Best single model Δ</div><div class="v">{fmt_money(best_single)}</div></div>
  <div class="kpi"><div class="l">Composite gain over single</div><div class="v">{fmt_money(composite_gain)}</div></div>
</div>

<h3>The 5 composite designs</h3>
<table><thead><tr>
  <th>Composite</th><th>Blocked</th><th>Rate</th>
  <th>Blocked W / L</th><th>Counterfactual P&L</th><th>Δ vs actual</th>
</tr></thead><tbody>{composite_table}</tbody></table>

<h3>Top 8 single models for comparison</h3>
<table><thead><tr>
  <th>Model</th><th>Blocked</th><th>Rate</th>
  <th>Blocked W / L</th><th>Counterfactual P&L</th><th>Δ vs actual</th>
</tr></thead><tbody>{singles_table}</tbody></table>

<h3>Composite construction rules</h3>
<table style="font-size:13px;"><tbody>
  <tr><td><code>C1_any_of_3_surgical</code></td><td>Block if ANY of {{016_LOF, 017_OCSVM, 018_EllipticEnvelope}} blocks. Union.</td></tr>
  <tr><td><code>C2_majority_of_5_surgical</code></td><td>Block if ≥3 of {{016, 017, 018, 074_NN_lbfgs, 121_polyKernel}} agree.</td></tr>
  <tr><td><code>C3_surgical_union_plus_102</code></td><td>Block if C1 says block OR 102_rbf_sampler says block.</td></tr>
  <tr><td><code>C4_soft_vote_delta_weighted</code></td><td>Weighted average of all 8 top-model probabilities by historical Δ; threshold 0.5.</td></tr>
  <tr><td><code>C5_intersection_of_3_surgical</code></td><td>Block ONLY if ALL of {{016, 017, 018}} agree. Very rare.</td></tr>
</tbody></table>

<div class="sub" style="margin-top:24px;">
  Source: <code>experiments/ml_lab/runs/composite_backtest/summary.parquet</code><br>
  Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
</div>
</body></html>"""


def build_winner_detail(summary: pd.DataFrame, per_entry: pd.DataFrame) -> str:
    winner = summary.iloc[0]
    name = winner["composite"]
    per_entry = per_entry.copy().sort_values("entry_ts")
    per_entry["entry_ts"] = pd.to_datetime(per_entry["entry_ts"]).dt.strftime("%H:%M:%S")
    per_entry["exit_ts"] = pd.to_datetime(per_entry["exit_ts"]).dt.strftime("%H:%M:%S")

    rows = []
    for r in per_entry.itertuples(index=False):
        if r.blocked and r.pnl < 0: cls = "blocked-loser"
        elif r.blocked and r.pnl > 0: cls = "blocked-winner"
        elif r.pnl > 0: cls = "kept-winner"
        else: cls = "kept-loser"
        rows.append(
            f'<tr class="{cls}">'
            f'<td>{r.lane}</td><td>{r.symbol}</td><td>{r.side_dir}</td>'
            f'<td>{r.entry_ts}</td><td>{r.exit_ts}</td><td>{r.exit_action}</td>'
            f'<td>{r.qty:.0f}</td><td>{fmt_money(r.pnl)}</td>'
            f'<td>{r.votes_surgical}/5</td>'
            f'<td>{r.rbf_proba:.3f}</td>'
            f'<td>{"BLOCKED" if r.blocked else "kept"}</td>'
            f'</tr>'
        )

    delta = float(winner["pnl_delta"])
    cf = float(winner["counterfactual_pnl"])
    actual = float(winner["actual_pnl"])
    n_block = int(winner["n_blocked"])
    bw = int(winner["blocked_winners"])
    bl = int(winner["blocked_losers"])
    rate = float(winner["block_rate_pct"])

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>{html.escape(name)} — winner detail</title>{CSS}</head><body>
<div class="topbar">
  <div><h1>{html.escape(name)}</h1>
    <div><span class="pill pill-winner">WINNER · best of 5 composites</span></div>
  </div>
  <a href="index.html">← back to composites index</a>
</div>
<div class="sub">Block if ANY of {{016_LOF, 017_OneClassSVM, 018_EllipticEnvelope}} blocks OR
102_rbf_sampler_logreg blocks. Combines surgical anomaly-detection blockers with
the aggressive RBF-feature blocker — diverse signal sources.</div>

<div class="kpis">
  <div class="kpi"><div class="l">Blocked</div><div class="v">{n_block} ({rate:.1f}%)</div></div>
  <div class="kpi"><div class="l">Blocked W / L</div><div class="v">{bw} / {bl}</div></div>
  <div class="kpi"><div class="l">Actual P&L</div><div class="v">{fmt_money(actual)}</div></div>
  <div class="kpi"><div class="l">Counterfactual P&L</div><div class="v">{fmt_money(cf)}</div></div>
  <div class="kpi"><div class="l">Δ vs actual</div><div class="v">{fmt_money(delta)}</div></div>
</div>

<div class="sub" style="font-size:12px;">
  <span style="background:#14532d40;padding:2px 6px">blocked-loser (good)</span>
  <span style="background:#7f1d1d40;padding:2px 6px">blocked-winner (bad)</span>
  <span style="background:#052e16;padding:2px 6px">kept-winner</span>
  <span style="background:#1f2937;padding:2px 6px">kept-loser</span>
</div>

<table><thead><tr>
  <th>lane</th><th>sym</th><th>side</th>
  <th>entry</th><th>exit</th><th>why</th>
  <th>qty</th><th>P&L</th>
  <th>surg votes</th><th>rbf p</th><th>action</th>
</tr></thead><tbody>{''.join(rows)}</tbody></table>

<div class="sub" style="margin-top:24px;">Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
</body></html>"""


def main():
    summary = pd.read_parquet(RUNS / "summary.parquet")
    winner_per_entry = pd.read_parquet(RUNS / "winner_per_entry.parquet")
    top_singles = pd.read_parquet(PASS_RUNS / "summary.parquet").dropna(subset=["pnl_delta"]).sort_values("pnl_delta", ascending=False)

    (OUT / "index.html").write_text(build_index(summary, top_singles))
    (OUT / "winner.html").write_text(build_winner_detail(summary, winner_per_entry))
    print(f"wrote {OUT}/index.html + winner.html")

    for u in [f"file://{OUT}/index.html", f"file://{OUT}/winner.html"]:
        subprocess.run(["open", u], check=False)


if __name__ == "__main__":
    main()
