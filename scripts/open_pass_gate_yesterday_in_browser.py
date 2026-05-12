"""Render per-model HTML pages for the pass-gate yesterday backtest + open in browser.

Reads experiments/ml_lab/runs/pass_gate_backtest/{summary,per_entry}.parquet.
Writes reports/pass_gate_yesterday/<model>.html + index.html, opens all tabs.
"""
from __future__ import annotations

import html
import json
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO = Path("/Users/weeoto/turbo-carnival")
RUNS = REPO / "experiments" / "ml_lab" / "runs" / "pass_gate_backtest"
OUT = REPO / "reports" / "pass_gate_yesterday"
OUT.mkdir(parents=True, exist_ok=True)

CSS = """
<style>
  body { background: #0b1120; color: #e2e8f0; font-family: -apple-system, BlinkMacSystemFont, sans-serif;
         margin: 0; padding: 20px 28px; }
  h1 { color: #f8fafc; margin-bottom: 4px; }
  .sub { color: #94a3b8; margin-bottom: 18px; font-size: 14px; }
  .topbar { display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 12px; }
  .pill { display: inline-block; padding: 3px 10px; border-radius: 999px;
          font-size: 12px; font-weight: 600; background: #1e293b; color: #cbd5e1; }
  .pill-good { background: rgba(34,197,94,0.18); color: #4ade80;
               border: 1px solid rgba(34,197,94,0.35); }
  .pill-bad { background: rgba(239,68,68,0.18); color: #f87171;
              border: 1px solid rgba(239,68,68,0.35); }
  .kpis { display: flex; gap: 12px; margin: 12px 0 22px 0; flex-wrap: wrap; }
  .kpi { background: #0f172a; border: 1px solid #1e293b; padding: 10px 16px;
         border-radius: 10px; min-width: 130px; }
  .kpi .l { color: #94a3b8; font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; }
  .kpi .v { color: #f1f5f9; font-size: 22px; font-weight: 700;
            font-variant-numeric: tabular-nums; margin-top: 4px; }
  .pos { color: #4ade80; } .neg { color: #f87171; }
  table { border-collapse: collapse; width: 100%; font-size: 13px;
          font-variant-numeric: tabular-nums; }
  th { background: #1e293b; color: #cbd5e1; padding: 8px 10px; text-align: left;
       position: sticky; top: 0; }
  td { padding: 6px 10px; border-bottom: 1px solid #1e293b; }
  tr.blocked-loser { background: #14532d40; }
  tr.blocked-winner { background: #7f1d1d40; }
  tr.kept-winner { background: #052e16; }
  tr.kept-loser { background: #1f2937; }
  a { color: #60a5fa; text-decoration: none; } a:hover { text-decoration: underline; }
  .legend { color: #cbd5e1; font-size: 12px; margin: 14px 0; }
  .legend span { padding: 2px 8px; border-radius: 4px; margin-right: 8px; }
</style>
"""


def fmt_money(v):
    if pd.isna(v): return '<span style="color:#64748b">n/a</span>'
    cls = "pos" if v > 0 else "neg" if v < 0 else "neutral"
    s = f"{v:+,.2f}"
    return f'<span class="{cls}">${s}</span>'


def build_model_page(model: str, summary_row: pd.Series, sub: pd.DataFrame) -> str:
    delta = float(summary_row.get("pnl_delta", 0))
    cf = float(summary_row.get("counterfactual_pnl", 0))
    actual = float(summary_row.get("actual_pnl", 0))
    n_block = int(summary_row.get("n_blocked", 0))
    n_kept = int(summary_row.get("n_kept", 0))
    bw = int(summary_row.get("blocked_winners", 0))
    bl = int(summary_row.get("blocked_losers", 0))
    rate = float(summary_row.get("block_rate_pct", 0))

    sub = sub.copy().sort_values("entry_ts")
    sub["entry_ts"] = pd.to_datetime(sub["entry_ts"]).dt.strftime("%H:%M:%S")
    sub["exit_ts"] = pd.to_datetime(sub["exit_ts"]).dt.strftime("%H:%M:%S")

    rows = []
    for r in sub.itertuples(index=False):
        if r.blocked and r.pnl < 0: cls = "blocked-loser"
        elif r.blocked and r.pnl > 0: cls = "blocked-winner"
        elif r.pnl > 0: cls = "kept-winner"
        else: cls = "kept-loser"
        rows.append(
            f'<tr class="{cls}">'
            f'<td>{r.lane}</td><td>{r.symbol}</td><td>{r.side_dir}</td>'
            f'<td>{r.entry_ts}</td><td>{r.exit_ts}</td>'
            f'<td>{r.exit_action}</td><td>{r.qty:.0f}</td>'
            f'<td>{fmt_money(r.pnl)}</td>'
            f'<td>{r.model_proba:.3f}</td>'
            f'<td>{"BLOCKED" if r.blocked else "kept"}</td>'
            f'</tr>'
        )

    badge = ('<span class="pill pill-good">net positive Δ</span>' if delta > 0
              else '<span class="pill pill-bad">net negative Δ</span>' if delta < 0
              else '<span class="pill">no impact</span>')

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>{model} — yesterday backtest</title>{CSS}</head><body>
<div class="topbar">
  <div><h1>{model}</h1><div>{badge}</div></div>
  <a href="index.html">← back to all</a>
</div>
<div class="sub">Counterfactual: what if this pass-gate model had blocked entries
  whose <code>predict_proba_class1 &gt; 0.5</code> on yesterday's 162 matched entries.
  Realized P&L computed from minute-bar closes (NOT actual Alpaca fills).</div>

<div class="kpis">
  <div class="kpi"><div class="l">Blocked / kept</div><div class="v">{n_block} / {n_kept}</div></div>
  <div class="kpi"><div class="l">Block rate</div><div class="v">{rate:.1f}%</div></div>
  <div class="kpi"><div class="l">Blocked winners / losers</div><div class="v">{bw} / {bl}</div></div>
  <div class="kpi"><div class="l">Actual P&L (bar-close)</div><div class="v">{fmt_money(actual)}</div></div>
  <div class="kpi"><div class="l">Counterfactual P&L</div><div class="v">{fmt_money(cf)}</div></div>
  <div class="kpi"><div class="l">Δ vs actual</div><div class="v">{fmt_money(delta)}</div></div>
</div>

<div class="legend">Legend:
  <span style="background:#14532d40">blocked-loser (good)</span>
  <span style="background:#7f1d1d40">blocked-winner (bad)</span>
  <span style="background:#052e16">kept-winner</span>
  <span style="background:#1f2937">kept-loser</span>
</div>

<table><thead><tr>
  <th>lane</th><th>sym</th><th>side</th>
  <th>entry</th><th>exit</th><th>why</th>
  <th>qty</th><th>P&L</th><th>proba</th><th>action</th>
</tr></thead><tbody>{''.join(rows)}</tbody></table>

<div class="sub" style="margin-top:24px;">Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
</body></html>"""


def main() -> int:
    summary = pd.read_parquet(RUNS / "summary.parquet")
    per_entry = pd.read_parquet(RUNS / "per_entry.parquet")

    # Filter to clean runs
    summary = summary[summary.get("error", pd.Series([None]*len(summary))).isna() if "error" in summary.columns else slice(None)]
    summary = summary.dropna(subset=["pnl_delta"]).reset_index(drop=True)
    print(f"Generating pages for {len(summary)} pass-gate models")

    paths = {}
    for _, row in summary.iterrows():
        m = row["model"]
        sub = per_entry[per_entry["model"] == m]
        if sub.empty: continue
        p = OUT / f"{m}.html"
        p.write_text(build_model_page(m, row, sub))
        paths[m] = p

    summary_sorted = summary.sort_values("pnl_delta", ascending=False).reset_index(drop=True)

    # Index page
    rows = []
    for _, r in summary_sorted.iterrows():
        m = r["model"]
        delta = float(r.get("pnl_delta", 0))
        cf = float(r.get("counterfactual_pnl", 0))
        rate = float(r.get("block_rate_pct", 0))
        bw = int(r.get("blocked_winners", 0)); bl = int(r.get("blocked_losers", 0))
        rows.append(
            f'<tr>'
            f'<td><a href="{m}.html">{m}</a></td>'
            f'<td>{int(r.get("n_blocked", 0))}</td>'
            f'<td>{rate:.1f}%</td>'
            f'<td>{bw} / {bl}</td>'
            f'<td>{fmt_money(cf)}</td>'
            f'<td>{fmt_money(delta)}</td>'
            f'</tr>'
        )
    pages_js = ",".join(f'"{m}.html"' for m in summary_sorted["model"])

    idx_html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Pass-gate models — yesterday backtest</title>{CSS}
<script>function openAll() {{ const p = [{pages_js}]; p.forEach(x => window.open(x, "_blank")); }}</script>
</head><body>
<div class="topbar">
  <h1>Pass-gate models · yesterday's 162 entries · counterfactual P&L</h1>
  <button onclick="openAll()" style="background:#2563eb;color:white;border:0;padding:8px 14px;
    border-radius:6px;cursor:pointer">Open all {len(summary)} in new tabs</button>
</div>
<div class="sub">
  Each row = one PASS-gate model trained on the full 21-day cache, applied to yesterday's 162 matched entries.
  BLOCK rule: <code>predict_proba_class1 &gt; 0.5</code>. Sorted by Δ vs actual (best = saved most).
  <br>Actual P&L on bar-close basis = +$496. Real fills with slippage were −$3,443.
</div>
<table><thead><tr>
  <th>Model</th><th>Blocked</th><th>Rate</th>
  <th>Blocked W / L</th><th>Counterfactual P&L</th><th>Δ vs actual</th>
</tr></thead><tbody>{''.join(rows)}</tbody></table>
<div class="sub" style="margin-top:24px;">Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
</body></html>"""
    idx = OUT / "index.html"
    idx.write_text(idx_html)
    print(f"wrote {len(paths)} pages + index → {OUT}")

    urls = ["file://" + str(idx)] + [
        "file://" + str(paths[m]) for m in summary_sorted["model"] if m in paths
    ]
    print(f"opening {len(urls)} browser tabs…")
    for u in urls:
        try:
            subprocess.run(["open", u], check=False)
        except FileNotFoundError:
            import webbrowser; webbrowser.open(u, new=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
