"""Generate one HTML page per PASS-gate architecture + an index hub, open all in browser.

Each page shows:
  - The architecture's full run.py source code (syntax-styled)
  - Its results.json metrics in a KPI strip
  - Provenance: which commit, which Haiku agent / programmatic / manual

Reads experiments/lab/<NNN_name>/results.json + run.py. Opens via macOS `open`.
"""
from __future__ import annotations

import html
import json
import subprocess
from datetime import datetime
from pathlib import Path

REPO = Path("/Users/weeoto/turbo-carnival")
LAB = REPO / "experiments" / "lab"
OUT = REPO / "reports" / "pass_gate"
OUT.mkdir(parents=True, exist_ok=True)

CSS = """
<style>
  body { background: #0b1120; color: #e2e8f0; font-family: -apple-system, BlinkMacSystemFont, sans-serif;
         margin: 0; padding: 20px 28px; line-height: 1.45; }
  h1 { color: #f8fafc; margin-bottom: 6px; }
  .sub { color: #94a3b8; margin-bottom: 18px; font-size: 14px; }
  .topbar { display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 12px; }
  .pill { display: inline-block; padding: 3px 10px; border-radius: 999px;
          font-size: 12px; font-weight: 600; background: #1e293b; color: #cbd5e1;
          margin-right: 6px; }
  .pill-pass { background: rgba(34,197,94,0.18); color: #4ade80;
               border: 1px solid rgba(34,197,94,0.35); }
  .kpis { display: flex; gap: 12px; margin: 12px 0 22px 0; flex-wrap: wrap; }
  .kpi { background: #0f172a; border: 1px solid #1e293b; padding: 10px 16px;
         border-radius: 10px; min-width: 140px; }
  .kpi .l { color: #94a3b8; font-size: 11px; text-transform: uppercase;
            letter-spacing: 0.04em; }
  .kpi .v { color: #f1f5f9; font-size: 22px; font-weight: 700;
            font-variant-numeric: tabular-nums; margin-top: 4px; }
  pre { background: #020617; border: 1px solid #1e293b; border-radius: 8px;
        padding: 14px; overflow-x: auto; font-family: ui-monospace, Menlo, monospace;
        font-size: 13px; color: #e2e8f0; }
  table { border-collapse: collapse; width: 100%; font-size: 13px;
          font-variant-numeric: tabular-nums; }
  th { background: #1e293b; color: #cbd5e1; padding: 8px 10px; text-align: left;
       position: sticky; top: 0; }
  td { padding: 6px 10px; border-bottom: 1px solid #1e293b; }
  tr:hover { background: #111827; }
  a { color: #60a5fa; text-decoration: none; }
  a:hover { text-decoration: underline; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 4px;
           font-size: 11px; margin-right: 4px; }
  .badge-manual { background: #1e293b; color: #cbd5e1; }
  .badge-sweep { background: #1e3a8a40; color: #93c5fd; }
  .badge-haiku { background: #581c8740; color: #c4b5fd; }
  .badge-custom { background: #14532d40; color: #4ade80; }
</style>
"""


def provenance(nnn: int) -> str:
    """Map NNN to its source category."""
    if 1 <= nnn <= 30:
        return "manual"
    if 31 <= nnn <= 103:
        return "sweep"
    if 104 <= nnn <= 146:
        return "haiku"
    return "unknown"


def fmt_money(v: float, signed: bool = True) -> str:
    fmt = f"{v:+.4f}" if signed else f"{v:.4f}"
    cls = "pos" if v > 0 else "neg" if v < 0 else ""
    return f'<span style="color:{"#4ade80" if v>0 else "#f87171" if v<0 else "#cbd5e1"}">{fmt}</span>'


def build_experiment_page(exp_dir: Path, results: dict) -> str:
    nnn = int(exp_dir.name.split("_")[0])
    src_kind = provenance(nnn)
    badge_class = {"manual": "badge-manual", "sweep": "badge-sweep",
                    "haiku": "badge-haiku"}.get(src_kind, "badge-manual")
    badge_label = {"manual": "manual (001-030)", "sweep": "programmatic sweep (031-103)",
                    "haiku": "Haiku-agent (104-146)"}.get(src_kind, src_kind)

    run_py = exp_dir / "run.py"
    code = run_py.read_text() if run_py.exists() else "(no run.py found)"
    code_html = html.escape(code)

    auc = results.get("test_auc", 0)
    gap = results.get("train_test_gap", 0)
    brier = results.get("test_brier", 0)
    prec1 = results.get("precision_at_top1pct", 0)
    description = results.get("description", "")
    extra = results.get("extra", {})

    extra_html = ""
    if extra:
        rows = []
        for k, v in extra.items():
            v_str = json.dumps(v) if isinstance(v, (list, dict)) else str(v)
            if len(v_str) > 300:
                v_str = v_str[:300] + "…"
            rows.append(f"<tr><td>{html.escape(k)}</td><td>{html.escape(v_str)}</td></tr>")
        extra_html = f"""
        <h3>Extra fields</h3>
        <table><thead><tr><th>key</th><th>value</th></tr></thead><tbody>
        {''.join(rows)}
        </tbody></table>
        """

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>{exp_dir.name} — PASS</title>{CSS}</head><body>
<div class="topbar">
  <div>
    <h1>{exp_dir.name}</h1>
    <div class="sub">
      <span class="pill pill-pass">PASS overfit gate</span>
      <span class="badge {badge_class}">{badge_label}</span>
    </div>
  </div>
  <a href="index.html">← back to all 64</a>
</div>
<div class="sub">{html.escape(description)}</div>

<div class="kpis">
  <div class="kpi"><div class="l">Test AUC</div><div class="v">{auc:.4f}</div></div>
  <div class="kpi"><div class="l">Train AUC</div><div class="v">{results.get('train_auc', 0):.4f}</div></div>
  <div class="kpi"><div class="l">Train-test gap</div><div class="v">{fmt_money(gap)}</div></div>
  <div class="kpi"><div class="l">Test Brier</div><div class="v">{brier:.4f}</div></div>
  <div class="kpi"><div class="l">Precision @ top 1%</div><div class="v">{prec1:.3f}</div></div>
</div>

<h3>run.py</h3>
<pre>{code_html}</pre>

{extra_html}

<div class="sub" style="margin-top:24px;">
  Source: <code>experiments/lab/{exp_dir.name}/</code> ·
  Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
</div>
</body></html>"""


def main() -> int:
    # Collect all PASS-gate experiments
    passes = []
    for d in sorted(LAB.iterdir()):
        if not d.is_dir() or not d.name[0].isdigit():
            continue
        rj = d / "results.json"
        if not rj.exists():
            continue
        try:
            r = json.loads(rj.read_text())
        except Exception:
            continue
        if r.get("passes_overfit_gate"):
            passes.append((d, r))

    print(f"Found {len(passes)} pass-gate experiments")

    # Sort by test AUC desc for the index
    passes_sorted = sorted(passes, key=lambda x: -float(x[1].get("test_auc", 0)))

    # Write each page
    paths = {}
    for d, r in passes:
        p = OUT / f"{d.name}.html"
        p.write_text(build_experiment_page(d, r))
        paths[d.name] = p

    # Build index page
    rows = []
    for d, r in passes_sorted:
        nnn = int(d.name.split("_")[0])
        src = provenance(nnn)
        badge_class = {"manual": "badge-manual", "sweep": "badge-sweep",
                        "haiku": "badge-haiku"}.get(src, "badge-manual")
        auc = float(r.get("test_auc", 0))
        gap = float(r.get("train_test_gap", 0))
        prec1 = float(r.get("precision_at_top1pct", 0))
        desc = (r.get("description") or "")[:70]
        rows.append(
            f'<tr>'
            f'<td><a href="{paths[d.name].name}">{d.name}</a></td>'
            f'<td><span class="badge {badge_class}">{src}</span></td>'
            f'<td>{html.escape(desc)}</td>'
            f'<td>{auc:.3f}</td><td>{fmt_money(gap)}</td><td>{prec1:.3f}</td>'
            f'</tr>'
        )

    pages_js = ",".join(f'"{paths[d.name].name}"' for d, _ in passes_sorted)
    index_html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>PASS-gate architectures — 64 total</title>{CSS}
<script>
function openAll() {{
  const pages = [{pages_js}];
  pages.forEach(p => window.open(p, "_blank"));
}}
</script>
</head><body>
<div class="topbar">
  <h1>PASS-gate architectures · {len(passes)} total</h1>
  <button onclick="openAll()" style="background:#2563eb;color:white;border:0;
    padding:8px 14px;border-radius:6px;cursor:pointer;font-size:14px">
    Open all {len(passes)} in new tabs
  </button>
</div>
<div class="sub">
  Filtered from the 146-architecture lab. A model PASSES if test_auc &gt; 0.55 AND
  |train_test_gap| &lt; 0.05. Sorted by test AUC descending.
  Sources: <span class="badge badge-manual">manual</span> hand-built (001-030),
  <span class="badge badge-sweep">sweep</span> programmatic (031-103),
  <span class="badge badge-haiku">haiku</span> Haiku-agent (104-146).
</div>

<table>
<thead><tr>
  <th>Experiment</th><th>Source</th><th>Architecture</th>
  <th>Test AUC</th><th>Gap</th><th>Precision @ top 1%</th>
</tr></thead>
<tbody>
{''.join(rows)}
</tbody></table>

<div class="sub" style="margin-top:24px;">
  Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ·
  Source: <code>experiments/lab/&lt;NNN_name&gt;/results.json</code>
</div>
</body></html>"""
    idx = OUT / "index.html"
    idx.write_text(index_html)
    print(f"wrote {len(paths)} pages + index → {OUT}")

    # Open index first, then each individual page
    urls = ["file://" + str(idx)] + [
        "file://" + str(paths[d.name]) for d, _ in passes_sorted
    ]
    print(f"opening {len(urls)} browser tabs (index + {len(paths)} pages)…")
    for u in urls:
        try:
            subprocess.run(["open", u], check=False)
        except FileNotFoundError:
            import webbrowser
            webbrowser.open(u, new=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
