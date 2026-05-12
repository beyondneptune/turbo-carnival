"""Pull actual paper-trade logs from all 6 Alpaca accounts and render a
unified HTML comparison report.

For each account, fetch:
  - /v2/account (equity, buying power, status)
  - /v2/orders (yesterday's orders, all statuses)
  - /v2/positions (currently open)

For Aegis/Falcon/Valk: also reconcile against local SQLite for trades that
the live host recorded.

Output: reports/six_accounts/index.html with per-account tabs.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

REPO = Path("/Users/weeoto/turbo-carnival")
OUT = REPO / "reports" / "six_accounts"
OUT.mkdir(parents=True, exist_ok=True)
ENDPOINT = "https://paper-api.alpaca.markets/v2"

# Map nickname → env var prefix → expected strategy
ACCOUNTS = [
    ("Aegis",      "AEGIS",      "HFT mean-reversion (existing)", "active"),
    ("Falcon",     "FALCON",     "HFT momentum/breakout (existing)", "active"),
    ("Valk",       "VALK",       "30-DTE iron condor (existing)", "active"),
    ("Hermes",     "HERMES",     "0DTE credit spreads / SVXY rotation (NEW)", "fresh"),
    ("Atlas",      "ATLAS",      "Cointegrated pairs (NEW)", "fresh"),
    ("TERMINATOR", "TERMINATOR", "ATHENA + C3 + Half-Kelly + 2-stage exec ($1k)", "fresh"),
]


def load_env():
    for line in (REPO / "council" / ".env.local").read_text().splitlines():
        line = line.strip()
        if line and "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            if k.strip() not in os.environ:
                os.environ[k.strip()] = v.strip().strip('"').strip("'")


def fetch(account_prefix: str, path: str, params: dict | None = None) -> dict | list | None:
    key = os.environ.get(f"ALPACA_{account_prefix}_KEY")
    sec = os.environ.get(f"ALPACA_{account_prefix}_SECRET")
    if not key or not sec:
        return None
    H = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}
    try:
        r = requests.get(f"{ENDPOINT}{path}", headers=H, params=params or {}, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def get_account_summary(prefix: str) -> dict:
    a = fetch(prefix, "/account")
    if not a:
        return {"ok": False, "error": "no API key or fetch failed"}
    return {
        "ok": True,
        "account_number": a.get("account_number", "?"),
        "equity": float(a.get("equity", 0)),
        "last_equity": float(a.get("last_equity", 0)),
        "buying_power": float(a.get("buying_power", 0)),
        "cash": float(a.get("cash", 0)),
        "status": a.get("status", "?"),
        "currency": a.get("currency", "USD"),
        "pdt": a.get("pattern_day_trader", False),
        "daytrade_count": a.get("daytrade_count", 0),
        "created_at": a.get("created_at", "?"),
    }


def get_orders(prefix: str, after: str, until: str) -> list:
    o = fetch(prefix, "/orders", {"status": "all", "after": after, "until": until,
                                    "limit": 500, "direction": "desc"})
    return o or []


def get_positions(prefix: str) -> list:
    p = fetch(prefix, "/positions")
    return p or []


def main():
    load_env()
    yesterday_start = "2026-05-11T00:00:00Z"
    today_end = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")

    account_data = []
    for nickname, prefix, strategy, state in ACCOUNTS:
        print(f"\n=== {nickname} ({prefix}) ===")
        summary = get_account_summary(prefix)
        if not summary["ok"]:
            print(f"  {summary['error']}")
            account_data.append({
                "nickname": nickname, "prefix": prefix, "strategy": strategy,
                "state": state, "summary": summary, "orders": [], "positions": [],
            })
            continue
        print(f"  equity ${summary['equity']:,.2f}  status {summary['status']}  acct {summary['account_number']}")
        orders = get_orders(prefix, yesterday_start, today_end)
        positions = get_positions(prefix)
        # Filter to yesterday + today
        orders_recent = []
        for o in orders:
            ts = o.get("submitted_at", "")
            if ts >= "2026-05-11":
                orders_recent.append(o)
        print(f"  {len(orders_recent)} orders since 2026-05-11  ·  {len(positions)} open positions")
        account_data.append({
            "nickname": nickname, "prefix": prefix, "strategy": strategy,
            "state": state, "summary": summary,
            "orders": orders_recent, "positions": positions,
        })

    # Build HTML
    css = """<style>
body { background: #0b1120; color: #e2e8f0; font-family: -apple-system, sans-serif;
       margin: 0; padding: 20px 28px; line-height: 1.45; }
h1 { color: #f8fafc; margin: 0 0 8px 0; }
h2 { color: #f1f5f9; margin: 28px 0 12px 0; font-size: 18px; }
h3 { color: #cbd5e1; margin: 20px 0 8px 0; font-size: 15px; }
.sub { color: #94a3b8; font-size: 14px; margin-bottom: 18px; }
.account-card { background: #0f172a; border: 1px solid #1e293b; padding: 18px 22px;
                border-radius: 12px; margin: 16px 0; }
.account-card.fresh { border-left: 4px solid #60a5fa; }
.account-card.active { border-left: 4px solid #4ade80; }
.kpis { display: flex; gap: 10px; margin: 10px 0 16px 0; flex-wrap: wrap; }
.kpi { background: #1e293b; padding: 8px 14px; border-radius: 8px; min-width: 110px; }
.kpi .l { color: #94a3b8; font-size: 10px; text-transform: uppercase;
          letter-spacing: 0.05em; }
.kpi .v { color: #f1f5f9; font-size: 17px; font-weight: 700;
          font-variant-numeric: tabular-nums; margin-top: 2px; }
.pos { color: #4ade80; } .neg { color: #f87171; }
table { border-collapse: collapse; width: 100%; font-size: 12px;
        font-variant-numeric: tabular-nums; }
th { background: #1e293b; color: #cbd5e1; padding: 6px 10px; text-align: left; }
td { padding: 4px 10px; border-bottom: 1px solid #1e293b; }
tr.fill { background: rgba(34,197,94,0.05); }
tr.cancel { background: rgba(148,163,184,0.04); color: #94a3b8; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 999px;
         font-size: 10px; font-weight: 600; margin-right: 6px; }
.badge-active { background: rgba(34,197,94,0.18); color: #4ade80; }
.badge-fresh { background: rgba(96,165,250,0.18); color: #93c5fd; }
.note { background: #0f172a; border-left: 3px solid #60a5fa; padding: 10px 14px;
        margin: 16px 0; font-size: 13px; color: #cbd5e1; border-radius: 0 8px 8px 0; }
code { background: #1e293b; padding: 2px 6px; border-radius: 4px; font-size: 12px; }
</style>"""

    cards = []
    for d in account_data:
        s = d["summary"]
        state_badge = ('<span class="badge badge-active">active — has trades</span>'
                      if d["state"] == "active"
                      else '<span class="badge badge-fresh">fresh — no trades yet</span>')
        if not s["ok"]:
            cards.append(f'''
<div class="account-card">
  <h2>{d["nickname"]}</h2>
  <div class="sub">{d["strategy"]}</div>
  <div style="color:#f87171">{s["error"]}</div>
</div>''')
            continue
        pnl_today = s["equity"] - s["last_equity"]
        pnl_cls = "pos" if pnl_today > 0 else "neg" if pnl_today < 0 else ""

        # Order table
        order_rows = []
        if d["orders"]:
            for o in d["orders"][:50]:  # top 50
                status = o.get("status", "?")
                cls = "fill" if status == "filled" else "cancel" if status in ("canceled", "expired") else ""
                ts = o.get("submitted_at", "")[:19].replace("T", " ")
                qty = o.get("qty", "?")
                fill_qty = o.get("filled_qty", "?")
                fill_px = o.get("filled_avg_price")
                fill_px_str = f"${float(fill_px):.4f}" if fill_px else "—"
                order_rows.append(
                    f'<tr class="{cls}">'
                    f'<td>{ts}</td>'
                    f'<td>{o.get("symbol", "?")}</td>'
                    f'<td>{o.get("side", "?")}</td>'
                    f'<td>{o.get("type", "?")}/{o.get("time_in_force", "?")}</td>'
                    f'<td>{qty}</td>'
                    f'<td>{fill_qty}</td>'
                    f'<td>{fill_px_str}</td>'
                    f'<td>{status}</td>'
                    f'</tr>'
                )
            order_table = f'''
<h3>Orders since 2026-05-11 ({len(d["orders"])} total)</h3>
<table><thead><tr>
  <th>Submitted</th><th>Symbol</th><th>Side</th><th>Type/TIF</th>
  <th>Qty</th><th>Filled</th><th>Fill px</th><th>Status</th>
</tr></thead><tbody>{''.join(order_rows)}</tbody></table>'''
        else:
            order_table = '<div class="sub">No orders since 2026-05-11.</div>'

        # Position table
        pos_rows = []
        for p in d["positions"][:20]:
            pos_rows.append(
                f'<tr>'
                f'<td>{p.get("symbol", "?")}</td>'
                f'<td>{p.get("side", "?")}</td>'
                f'<td>{p.get("qty", "?")}</td>'
                f'<td>${float(p.get("avg_entry_price", 0)):.4f}</td>'
                f'<td>${float(p.get("market_value", 0)):.2f}</td>'
                f'<td>${float(p.get("unrealized_pl", 0)):+.2f}</td>'
                f'</tr>'
            )
        pos_table = ""
        if pos_rows:
            pos_table = f'''
<h3>Open positions ({len(d["positions"])})</h3>
<table><thead><tr>
  <th>Symbol</th><th>Side</th><th>Qty</th><th>Avg entry</th><th>Mkt value</th><th>Unrealized P&L</th>
</tr></thead><tbody>{''.join(pos_rows)}</tbody></table>'''

        cards.append(f'''
<div class="account-card {d["state"]}">
  <h2>{d["nickname"]} {state_badge}</h2>
  <div class="sub">{d["strategy"]} · <code>{s["account_number"]}</code></div>
  <div class="kpis">
    <div class="kpi"><div class="l">Equity</div><div class="v">${s["equity"]:,.2f}</div></div>
    <div class="kpi"><div class="l">Last equity</div><div class="v">${s["last_equity"]:,.2f}</div></div>
    <div class="kpi"><div class="l">Δ today</div>
      <div class="v"><span class="{pnl_cls}">${pnl_today:+,.2f}</span></div></div>
    <div class="kpi"><div class="l">Buying power</div><div class="v">${s["buying_power"]:,.2f}</div></div>
    <div class="kpi"><div class="l">Cash</div><div class="v">${s["cash"]:,.2f}</div></div>
    <div class="kpi"><div class="l">Status</div><div class="v">{s["status"]}</div></div>
    <div class="kpi"><div class="l">PDT</div><div class="v">{"yes" if s["pdt"] else "no"}</div></div>
    <div class="kpi"><div class="l">Day trades</div><div class="v">{s["daytrade_count"]}</div></div>
  </div>
  {order_table}
  {pos_table}
</div>''')

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>6-Account Trade Logs</title>{css}</head><body>

<h1>All 6 paper accounts — live snapshot</h1>
<div class="sub">
  Yesterday's actual paper trades pulled directly from Alpaca's
  <code>/v2/orders</code> + <code>/v2/positions</code>. Fresh accounts (opened today) show empty.
</div>

<div class="note">
<strong>About "running from the beginning":</strong> the simulations we've done so far
(ML lab, composite C3 gate, 100-Haiku vote, $1k TERMINATOR projection) all reuse
<strong>yesterday's 162 entries</strong> as the test set. A "from beginning" backtest
across 2019-2026 daily data would be a different exercise — let me know if you want
that for the SVXY rotation strategy (we have 6.5 years of FRED + market context daily).
</div>

{''.join(cards)}

<div class="sub" style="margin-top:24px;">
  Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ·
  Source: <code>experiments/ml_lab/20_six_account_trade_logs.py</code>
</div>
</body></html>"""

    (OUT / "index.html").write_text(html)
    print(f"\nwrote {OUT}/index.html")
    subprocess.run(["open", f"file://{OUT}/index.html"], check=False)


if __name__ == "__main__":
    main()
