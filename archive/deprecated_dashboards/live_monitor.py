"""
Live trading monitor — focused dashboard for deployed strategies only.
Shows: account NAVs + last 30 log lines per tmux session.
Auto-refreshes every 5 seconds.
Designed to be exposed via cloudflared tunnel.
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse


# ----------------------------- config -----------------------------------

ROOT = Path("/Users/weeoto/turbo-carnival")
ENV_FILE = ROOT / "council" / ".env.local"

# Strategy sessions to show, in display order
SESSIONS = [
    ("svxy_sweet",         "AEGIS",  "SVXY VIX rotation"),
    ("adr_alpaca",         "AEGIS",  "US-HK ADR arbitrage"),
    ("nikkei_etf",         "AEGIS",  "Nikkei gap continuation (EWJ/DXJ)"),
    ("kraken_btc_funding", "KRAKEN", "BTC perp funding carry"),
    ("hft_engine",         "HERMES", "Rust HFT lead-lag (LIVE PAPER)"),
]

# Accounts to query for NAV
ALPACA_ACCOUNTS = [
    ("AEGIS",  "ALPACA_AEGIS_KEY",  "ALPACA_AEGIS_SECRET"),
    ("HERMES", "ALPACA_HERMES_KEY", "ALPACA_HERMES_SECRET"),
]


# --------------------------- helpers ------------------------------------

def _load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    if not ENV_FILE.exists():
        return env
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def _tmux_capture(session: str, lines: int = 30) -> str:
    try:
        out = subprocess.run(
            ["tmux", "capture-pane", "-t", session, "-p"],
            capture_output=True, text=True, timeout=4,
        )
        if out.returncode != 0:
            return f"[session not running]\n{out.stderr.strip()}"
        text = out.stdout
        lns = text.splitlines()
        return "\n".join(lns[-lines:])
    except subprocess.TimeoutExpired:
        return "[tmux timeout]"
    except Exception as exc:
        return f"[error: {exc}]"


def _tmux_running(session: str) -> bool:
    try:
        out = subprocess.run(
            ["tmux", "has-session", "-t", session],
            capture_output=True, timeout=2,
        )
        return out.returncode == 0
    except Exception:
        return False


def _alpaca_nav(key: str, secret: str) -> dict[str, Any]:
    try:
        r = requests.get(
            "https://paper-api.alpaca.markets/v2/account",
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
            timeout=5,
        )
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}"}
        a = r.json()
        return {
            "account_number": a.get("account_number"),
            "equity":          float(a.get("equity", 0)),
            "cash":            float(a.get("cash", 0)),
            "buying_power":    float(a.get("buying_power", 0)),
            "status":          a.get("status"),
        }
    except Exception as exc:
        return {"error": str(exc)}


def _recent_orders(key: str, secret: str, limit: int = 8) -> list[dict[str, Any]]:
    try:
        r = requests.get(
            "https://paper-api.alpaca.markets/v2/orders",
            params={"status": "all", "limit": limit},
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
            timeout=5,
        )
        if r.status_code != 200:
            return []
        out = []
        for o in r.json():
            out.append({
                "submitted": o.get("submitted_at", "")[:19],
                "side":      o.get("side"),
                "qty":       o.get("qty"),
                "symbol":    o.get("symbol"),
                "type":      o.get("order_type"),
                "status":    o.get("status"),
            })
        return out
    except Exception:
        return []


# --------------------------- API ----------------------------------------

app = FastAPI()
ENV = _load_env()


@app.get("/api/state")
def api_state() -> JSONResponse:
    sessions_data = []
    for sess, account, desc in SESSIONS:
        sessions_data.append({
            "name":    sess,
            "account": account,
            "desc":    desc,
            "running": _tmux_running(sess),
            "log":     _tmux_capture(sess, lines=30),
        })

    accounts_data = []
    for label, key_var, secret_var in ALPACA_ACCOUNTS:
        key = ENV.get(key_var, "")
        secret = ENV.get(secret_var, "")
        if not key or not secret:
            accounts_data.append({"label": label, "error": "no creds"})
            continue
        info = _alpaca_nav(key, secret)
        info["label"] = label
        info["orders"] = _recent_orders(key, secret, limit=8)
        accounts_data.append(info)

    return JSONResponse({
        "ts":       datetime.now(timezone.utc).isoformat(),
        "sessions": sessions_data,
        "accounts": accounts_data,
    })


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML


HTML = r"""
<!doctype html>
<html><head>
<meta charset="utf-8"><title>turbo-carnival · live monitor</title>
<style>
  :root { color-scheme: dark; }
  body { background:#0b0c10; color:#e6edf3; font-family:-apple-system,BlinkMacSystemFont,'SF Pro Text',monospace; margin:0; padding:14px; font-size:13px; }
  h1 { font-size:18px; margin:0 0 4px; font-weight:600; }
  .meta { color:#7d8590; font-size:12px; margin-bottom:14px; }
  .accounts { display:grid; grid-template-columns:repeat(auto-fit,minmax(380px,1fr)); gap:10px; margin-bottom:14px; }
  .acc { background:#0d1117; border:1px solid #30363d; border-radius:8px; padding:10px 12px; }
  .acc h2 { font-size:14px; margin:0 0 4px; }
  .acc .row { display:flex; justify-content:space-between; font-size:12px; color:#7d8590; }
  .acc .row b { color:#e6edf3; font-variant-numeric:tabular-nums; }
  .orders { margin-top:8px; font-size:11px; }
  .orders div { color:#7d8590; padding:1px 0; font-variant-numeric:tabular-nums; }
  .orders .filled { color:#3fb950; }
  .orders .canceled { color:#f85149; }
  .session { background:#0d1117; border:1px solid #30363d; border-radius:8px; padding:10px 12px; margin-bottom:8px; }
  .session h3 { font-size:14px; margin:0 0 4px; display:flex; justify-content:space-between; }
  .session .desc { color:#7d8590; font-size:11px; margin-bottom:6px; }
  .session pre { background:#010409; border:1px solid #21262d; border-radius:6px; padding:8px; font-size:11px; line-height:1.4; max-height:260px; overflow-y:auto; margin:0; white-space:pre-wrap; word-break:break-all; }
  .dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; }
  .dot.green { background:#3fb950; }
  .dot.red   { background:#f85149; }
  .badge { background:#1f6feb22; border:1px solid #1f6feb; color:#79c0ff; font-size:10px; padding:1px 6px; border-radius:10px; margin-left:6px; }
</style>
</head><body>
<h1>turbo-carnival · live trading monitor</h1>
<div class="meta" id="ts">loading…</div>

<div id="accounts" class="accounts"></div>
<div id="sessions"></div>

<script>
async function tick() {
  try {
    const r = await fetch('/api/state', {cache:'no-store'});
    const d = await r.json();
    document.getElementById('ts').textContent = 'updated ' + d.ts + ' UTC';

    const accBox = document.getElementById('accounts');
    accBox.innerHTML = '';
    for (const a of d.accounts) {
      const el = document.createElement('div'); el.className='acc';
      if (a.error) {
        el.innerHTML = `<h2>${a.label}</h2><div class="row"><span>error</span><b>${a.error}</b></div>`;
      } else {
        const orders = (a.orders||[]).slice(0,5).map(o =>
          `<div class="${o.status}">${o.submitted}  ${o.side.padEnd(5)} ${String(o.qty).padStart(6)} ${o.symbol.padEnd(6)} ${o.status}</div>`
        ).join('');
        el.innerHTML = `
          <h2>${a.label} <span class="badge">${a.status||''}</span></h2>
          <div class="row"><span>equity</span><b>$${a.equity.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2})}</b></div>
          <div class="row"><span>cash</span><b>$${a.cash.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2})}</b></div>
          <div class="row"><span>buying power</span><b>$${a.buying_power.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2})}</b></div>
          <div class="row"><span>acct</span><b>${a.account_number||''}</b></div>
          <div class="orders">${orders||'<div>(no recent orders)</div>'}</div>
        `;
      }
      accBox.appendChild(el);
    }

    const sessBox = document.getElementById('sessions');
    sessBox.innerHTML = '';
    for (const s of d.sessions) {
      const el = document.createElement('div'); el.className='session';
      const dot = s.running ? '<span class="dot green"></span>' : '<span class="dot red"></span>';
      el.innerHTML = `
        <h3>${dot}${s.name} <span style="color:#7d8590;font-weight:400">→ ${s.account}</span></h3>
        <div class="desc">${s.desc}</div>
        <pre>${(s.log||'').replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}</pre>
      `;
      sessBox.appendChild(el);
    }
  } catch (e) {
    document.getElementById('ts').textContent = 'fetch error: ' + e;
  }
}
tick();
setInterval(tick, 5000);
</script>
</body></html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8504, log_level="warning")
