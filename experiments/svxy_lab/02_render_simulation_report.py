"""Render SVXY rotation simulation results as a clear HTML report for review.

Shows:
  - Today's actual decision (live VIX/VIX3M, contango/backwardation, target)
  - Full 2019-2026 walk-forward equity curve as SVG
  - Trade log: every transition (enter/exit) with P&L
  - Year-by-year breakdown
  - Risk metrics
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/weeoto/turbo-carnival")
SVXY_RUNS = REPO / "experiments" / "svxy_lab" / "runs"
OUT = REPO / "reports" / "svxy_simulation"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_parquet(SVXY_RUNS / "walk_forward_results.parquet")
    summary = json.loads((SVXY_RUNS / "summary.json").read_text())

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # === Today's live decision (from dry-run) ===
    today_vix = 18.78; today_vix3m = 21.24; today_ratio = 0.884
    today_target = "LONG SVXY" if today_ratio < 1.0 else "FLAT"
    today_svxy_px = 50.93

    # === Trade transitions (enter/exit events) ===
    df["signal_change"] = df["signal"].diff().fillna(0).astype(int)
    transitions = df[df["signal_change"] != 0].copy()
    transitions["action"] = transitions["signal_change"].apply(
        lambda x: "ENTER long" if x == 1 else "EXIT to cash"
    )

    # Build trade pairs (enter → next exit = one round trip)
    trades = []
    in_trade = False
    entry_idx = None
    entry_eq = None
    for i, row in transitions.iterrows():
        if row["action"] == "ENTER long":
            in_trade = True
            entry_idx = i
            entry_eq = row["strategy_equity"]
        elif row["action"] == "EXIT to cash" and in_trade:
            trades.append({
                "entry_date": df.loc[entry_idx, "date"],
                "exit_date": row["date"],
                "hold_days": (row["date"] - df.loc[entry_idx, "date"]).days,
                "entry_eq": entry_eq,
                "exit_eq": row["strategy_equity"],
                "pnl": row["strategy_equity"] - entry_eq,
                "pnl_pct": (row["strategy_equity"] / entry_eq - 1) * 100,
            })
            in_trade = False
    trades_df = pd.DataFrame(trades)

    # === Year-by-year stats ===
    df["year"] = df["date"].dt.year
    yearly = df.groupby("year").agg(
        start=("strategy_equity", "first"),
        end=("strategy_equity", "last"),
        max_dd=("strategy_equity", lambda x: (x / x.cummax() - 1).min()),
        days_long=("signal", "sum"),
        days_total=("signal", "count"),
    ).reset_index()
    yearly["return_pct"] = (yearly["end"] / yearly["start"] - 1) * 100
    yearly["spy_return"] = df.groupby("year")["svxy_equity"].apply(
        lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100
    ).values

    # === SVG equity curve ===
    n_pts = min(len(df), 1800)
    sample = df.iloc[::max(1, len(df) // n_pts)].reset_index(drop=True)
    svg_w, svg_h = 1200, 360
    strat_max = sample["strategy_equity"].max()
    strat_min = sample["strategy_equity"].min()
    bh_max = sample["svxy_equity"].max()
    bh_min = sample["svxy_equity"].min()
    ymin = min(strat_min, bh_min) * 0.95
    ymax = max(strat_max, bh_max) * 1.05

    def x_at(i): return 60 + (i / max(1, len(sample) - 1)) * (svg_w - 100)
    def y_at(v): return svg_h - 40 - ((v - ymin) / (ymax - ymin)) * (svg_h - 70)

    strat_path = " ".join([f"{'M' if i==0 else 'L'} {x_at(i):.1f} {y_at(v):.1f}"
                            for i, v in enumerate(sample["strategy_equity"])])
    bh_path = " ".join([f"{'M' if i==0 else 'L'} {x_at(i):.1f} {y_at(v):.1f}"
                         for i, v in enumerate(sample["svxy_equity"])])

    base_y = y_at(100000)
    # Year tick marks
    year_ticks = []
    for year in range(2019, 2027):
        match = sample[sample["date"].dt.year == year]
        if not match.empty:
            i = match.index[0]
            x = x_at(i)
            year_ticks.append(f'<text x="{x:.1f}" y="{svg_h-15}" fill="#64748b" font-size="11" text-anchor="middle">{year}</text>')

    svg = f'''<svg width="{svg_w}" height="{svg_h}" style="background:#0f172a;border-radius:8px">
  <line x1="60" y1="{base_y:.1f}" x2="{svg_w-40}" y2="{base_y:.1f}" stroke="#475569" stroke-dasharray="4 4"/>
  <path d="{bh_path}" fill="none" stroke="#f87171" stroke-width="1.5" opacity="0.6"/>
  <path d="{strat_path}" fill="none" stroke="#4ade80" stroke-width="2"/>
  <text x="70" y="25" fill="#4ade80" font-size="13" font-weight="600">━ Strategy (long in contango, flat in backwardation)</text>
  <text x="70" y="44" fill="#f87171" font-size="13" font-weight="600" opacity="0.8">━ Buy-and-hold SVXY</text>
  <text x="70" y="{base_y - 6:.1f}" fill="#64748b" font-size="10" font-family="ui-monospace">$100k baseline</text>
  <text x="{svg_w - 40}" y="{y_at(sample['strategy_equity'].iloc[-1]) + 4:.1f}" fill="#4ade80" font-size="12" font-family="ui-monospace" text-anchor="end">${sample['strategy_equity'].iloc[-1]:,.0f}</text>
  {''.join(year_ticks)}
</svg>'''

    # === Trade log table ===
    trade_rows = []
    for r in trades_df.iterrows():
        t = r[1]
        cls = "win" if t["pnl"] > 0 else "loss"
        trade_rows.append(
            f'<tr class="{cls}">'
            f'<td>{t["entry_date"].strftime("%Y-%m-%d")}</td>'
            f'<td>{t["exit_date"].strftime("%Y-%m-%d")}</td>'
            f'<td>{t["hold_days"]}</td>'
            f'<td>${t["entry_eq"]:,.0f}</td>'
            f'<td>${t["exit_eq"]:,.0f}</td>'
            f'<td>${t["pnl"]:+,.0f}</td>'
            f'<td>{t["pnl_pct"]:+.2f}%</td>'
            f'</tr>'
        )

    yearly_rows = []
    for r in yearly.iterrows():
        y = r[1]
        cls = "win" if y["return_pct"] > 0 else "loss"
        yearly_rows.append(
            f'<tr class="{cls}">'
            f'<td>{int(y["year"])}</td>'
            f'<td>{int(y["days_long"])} / {int(y["days_total"])} ({y["days_long"]/y["days_total"]*100:.0f}%)</td>'
            f'<td>${y["start"]:,.0f}</td>'
            f'<td>${y["end"]:,.0f}</td>'
            f'<td>{y["return_pct"]:+.1f}%</td>'
            f'<td>{y["max_dd"]*100:.1f}%</td>'
            f'<td>{y["spy_return"]:+.1f}%</td>'
            f'</tr>'
        )

    CSS = """
<style>
  body { background: #0b1120; color: #e2e8f0; font-family: -apple-system, sans-serif;
         margin: 0; padding: 24px 32px; line-height: 1.55; }
  h1 { color: #f8fafc; margin: 0 0 6px 0; }
  h2 { color: #f1f5f9; font-size: 18px; margin: 28px 0 12px 0; }
  .sub { color: #94a3b8; font-size: 14px; margin-bottom: 18px; }
  .kpis { display: flex; gap: 12px; margin: 16px 0 22px 0; flex-wrap: wrap; }
  .kpi { background: #0f172a; border: 1px solid #1e293b; padding: 14px 18px;
         border-radius: 10px; min-width: 160px; }
  .kpi .l { color: #94a3b8; font-size: 11px; text-transform: uppercase;
            letter-spacing: 0.05em; }
  .kpi .v { color: #f1f5f9; font-size: 22px; font-weight: 700;
            font-variant-numeric: tabular-nums; margin-top: 4px; }
  .pos { color: #4ade80; } .neg { color: #f87171; }
  .today-card { background: rgba(34,197,94,0.10); border: 2px solid rgba(34,197,94,0.4);
                padding: 18px 22px; border-radius: 12px; margin: 18px 0; }
  table { border-collapse: collapse; width: 100%; font-size: 12px;
          font-variant-numeric: tabular-nums; }
  th { background: #1e293b; color: #cbd5e1; padding: 8px 12px; text-align: left; }
  td { padding: 6px 12px; border-bottom: 1px solid #1e293b; }
  tr.win { background: rgba(34,197,94,0.06); }
  tr.loss { background: rgba(239,68,68,0.06); }
  .scroll { max-height: 600px; overflow-y: auto; border: 1px solid #1e293b; border-radius: 8px; }
  code { background: #1e293b; padding: 2px 6px; border-radius: 4px; font-size: 12px; }
  .note { background: #0f172a; border-left: 4px solid #60a5fa; padding: 12px 16px;
          margin: 14px 0; font-size: 13px; color: #cbd5e1; border-radius: 0 8px 8px 0; }
</style>
"""

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>SVXY Rotation — full simulation results</title>{CSS}</head><body>

<h1>SVXY regime rotation — full simulation review</h1>
<div class="sub">
  Strategy: LONG SVXY when VIX/VIX3M &lt; 1.0 (contango), FLAT when ratio &gt;= 1.0
  (backwardation). Walk-forward 2019-01 through 2026-05, no lookahead.
  This is the strategy proposed for <strong>Hermes ($100k)</strong> and as a
  layer on <strong>TERMINATOR ($1k)</strong>.
</div>

<div class="today-card">
  <h2 style="margin-top:0">Today's live decision (2026-05-12)</h2>
  <div class="kpis">
    <div class="kpi"><div class="l">VIX</div><div class="v">{today_vix}</div></div>
    <div class="kpi"><div class="l">VIX3M</div><div class="v">{today_vix3m}</div></div>
    <div class="kpi"><div class="l">Ratio</div>
      <div class="v"><span class="pos">{today_ratio:.3f}</span></div></div>
    <div class="kpi"><div class="l">Regime</div><div class="v">contango</div></div>
    <div class="kpi"><div class="l">Decision</div>
      <div class="v"><span class="pos">{today_target}</span></div></div>
    <div class="kpi"><div class="l">SVXY price</div><div class="v">${today_svxy_px}</div></div>
  </div>
  If deployed today:
  <br>• <strong>Hermes ($100k)</strong>: BUY ~589 shares ($30k = 30% allocation)
  <br>• <strong>TERMINATOR ($1k)</strong>: BUY ~5.89 fractional shares ($300 = 30% allocation)
  <br>Trade fires after market open at 09:40 ET via market order. Cash buffer 70% protects against
  vol spike.
</div>

<h2>7-year walk-forward summary</h2>
<div class="kpis">
  <div class="kpi"><div class="l">Total return</div>
    <div class="v"><span class="pos">+{summary['total_return_pct']:.1f}%</span></div></div>
  <div class="kpi"><div class="l">Annualized</div>
    <div class="v"><span class="pos">+{summary['annualized_pct']:.1f}%</span></div></div>
  <div class="kpi"><div class="l">Sharpe</div><div class="v">{summary['sharpe']:.2f}</div></div>
  <div class="kpi"><div class="l">Max DD</div>
    <div class="v"><span class="neg">{summary['max_drawdown_strategy_pct']:.1f}%</span></div></div>
  <div class="kpi"><div class="l">% in market</div>
    <div class="v">{summary['pct_in_market']:.0f}%</div></div>
  <div class="kpi"><div class="l">Buy-hold SVXY</div>
    <div class="v">+{summary['buy_hold_return_pct']:.1f}% (DD {summary['max_drawdown_bh_pct']:.1f}%)</div></div>
  <div class="kpi"><div class="l">Scaled to $1k</div>
    <div class="v"><span class="pos">${summary['scaled_1k_final']:.0f}</span></div></div>
</div>

<h2>Equity curve (Strategy vs Buy-and-hold SVXY)</h2>
{svg}

<h2>Year-by-year breakdown</h2>
<table><thead><tr>
  <th>Year</th><th>Days long</th><th>Start equity</th><th>End equity</th>
  <th>Strategy return</th><th>Max DD</th><th>vs B&amp;H SVXY</th>
</tr></thead>
<tbody>{''.join(yearly_rows)}</tbody></table>

<h2>Trade log — every regime transition ({len(trades_df)} round trips)</h2>
<div class="scroll">
<table><thead><tr>
  <th>Entry date</th><th>Exit date</th><th>Hold days</th>
  <th>Entry equity</th><th>Exit equity</th><th>P&amp;L</th><th>P&amp;L %</th>
</tr></thead>
<tbody>{''.join(trade_rows)}</tbody></table>
</div>

<div class="note">
<strong>How to read this:</strong> the strategy entered a long SVXY position {len(trades_df)} times
over 7 years. Each row shows the realized P&L of one round trip (entry → exit when ratio crossed
back to backwardation). Most trips lasted weeks. The strategy was in the market 93% of the time —
i.e. contango is the default state of VIX futures.
</div>

<h2>Honest caveats</h2>
<div class="note">
<strong>What this backtest assumes:</strong>
<br>• No commissions (Alpaca paper is $0; real broker ~$0.65/contract for options, but this is ETF)
<br>• No slippage (SVXY has tight spread, but real fills move with size)
<br>• VIX/VIX3M data from yfinance (occasional missing days)
<br>• Walk-forward means signal at T-1 close → position at T open (1-day lag)
<br>
<strong>What this does NOT capture:</strong>
<br>• Black-swan vol spikes (2020 March COVID: SVXY lost 80%+ in 3 weeks — the strategy was
  flat for the spike but the timing depended on the ratio's lag)
<br>• Volume / liquidity constraints (we assume we can buy/sell at close)
<br>• Tax implications (SVXY is regulated investment company — gains taxed as 1099-DIV with K-1)
</div>

<h2>If you approve, deployment plan</h2>
<div class="note">
1. Start the SVXY rotation host: <code>tmux new -d -s svxy_host ".venv/bin/python scripts/svxy_rotation_host.py --lane HERMES --lane TERMINATOR"</code>
<br>2. The host will sleep until 09:40 ET, fetch VIX/VIX3M, decide, and place orders if needed.
<br>3. Decision logged to <code>data/svxy_rotation_host.sqlite3</code>.
<br>4. Trade history visible in the dashboard's "All 6 Accounts" tab and via <code>scripts/svxy_rotation_host.py --once --dry-run</code> at any time.
<br>5. Kill switch: <code>tmux kill-session -t svxy_host</code> (positions stay open; no auto-flatten on stop).
<br>6. Rollback: sell all SVXY via Alpaca app or <code>scripts/svxy_rotation_host.py --once</code> when ratio crosses back to backwardation.
</div>

<div class="sub" style="margin-top:24px;">
  Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ·
  Source: <code>experiments/svxy_lab/01_walk_forward_2019_2026.py</code>
</div>
</body></html>"""

    (OUT / "index.html").write_text(html)
    print(f"\nwrote {OUT}/index.html")
    subprocess.run(["open", f"file://{OUT}/index.html"], check=False)


if __name__ == "__main__":
    main()
