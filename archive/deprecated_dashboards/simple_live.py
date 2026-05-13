"""Simple live monitor — Streamlit version.

Minimal, vertical, clean. Shows account NAVs, strategy status, and log tails.
Auto-refreshes every 5 seconds.

Run:
    streamlit run dashboard/simple_live.py --server.port 8504 \\
        --server.address 0.0.0.0 --server.enableCORS false \\
        --server.enableXsrfProtection false
"""
from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh


ROOT = Path("/Users/weeoto/turbo-carnival")
ENV_FILE = ROOT / "council" / ".env.local"


SESSIONS = [
    ("svxy_sweet",         "AEGIS",  "SVXY VIX rotation"),
    ("adr_alpaca",         "AEGIS",  "US-HK ADR arbitrage"),
    ("nikkei_etf",         "AEGIS",  "Nikkei gap continuation"),
    ("kraken_btc_funding", "KRAKEN", "BTC perp funding carry"),
    ("hft_engine",         "HERMES", "Rust HFT lead-lag"),
]


ALPACA_ACCOUNTS = [
    ("AEGIS",  "ALPACA_AEGIS_KEY",  "ALPACA_AEGIS_SECRET"),
    ("HERMES", "ALPACA_HERMES_KEY", "ALPACA_HERMES_SECRET"),
]


@st.cache_data(ttl=5)
def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    if not ENV_FILE.exists():
        return env
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


@st.cache_data(ttl=3)
def tmux_capture(session: str, lines: int = 25) -> tuple[bool, str]:
    try:
        out = subprocess.run(
            ["tmux", "capture-pane", "-t", session, "-p"],
            capture_output=True, text=True, timeout=3,
        )
        if out.returncode != 0:
            return False, "(session not running)"
        text = out.stdout.rstrip()
        lns = text.splitlines()
        return True, "\n".join(lns[-lines:])
    except Exception as exc:
        return False, f"(error: {exc})"


@st.cache_data(ttl=5)
def alpaca_account(key: str, secret: str) -> dict:
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
            "equity":       float(a.get("equity", 0)),
            "cash":         float(a.get("cash", 0)),
            "buying_power": float(a.get("buying_power", 0)),
            "status":       a.get("status"),
        }
    except Exception as exc:
        return {"error": str(exc)}


@st.cache_data(ttl=5)
def alpaca_positions(key: str, secret: str) -> list[dict]:
    try:
        r = requests.get(
            "https://paper-api.alpaca.markets/v2/positions",
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
            timeout=5,
        )
        if r.status_code != 200:
            return []
        return r.json()
    except Exception:
        return []


@st.cache_data(ttl=5)
def alpaca_orders(key: str, secret: str, limit: int = 8) -> list[dict]:
    try:
        r = requests.get(
            "https://paper-api.alpaca.markets/v2/orders",
            params={"status": "all", "limit": limit},
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
            timeout=5,
        )
        if r.status_code != 200:
            return []
        return r.json()
    except Exception:
        return []


# ----- page ------------------------------------------------------------

st.set_page_config(
    page_title="turbo-carnival live",
    page_icon="📈",
    layout="centered",
)

st_autorefresh(interval=5000, key="auto")

st.title("turbo-carnival · live")
st.caption(f"updated {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC · auto-refresh 5s")

env = load_env()

# --- accounts ---
st.header("Accounts")

for label, key_var, secret_var in ALPACA_ACCOUNTS:
    key = env.get(key_var, "")
    secret = env.get(secret_var, "")
    st.subheader(label)
    if not key or not secret:
        st.error("no credentials")
        continue

    info = alpaca_account(key, secret)
    if "error" in info:
        st.error(info["error"])
        continue

    c1, c2, c3 = st.columns(3)
    c1.metric("Equity", f"${info['equity']:,.2f}")
    c2.metric("Cash", f"${info['cash']:,.2f}")
    c3.metric("Buying Power", f"${info['buying_power']:,.2f}")

    positions = alpaca_positions(key, secret)
    if positions:
        pos_rows = []
        for p in positions:
            pos_rows.append({
                "symbol": p["symbol"],
                "side":   p["side"],
                "qty":    p["qty"],
                "entry":  f"${float(p['avg_entry_price']):,.2f}",
                "mkt val": f"${float(p['market_value']):,.2f}",
                "unrealized": f"${float(p['unrealized_pl']):+,.2f}",
            })
        st.write("**Open positions**")
        st.dataframe(pos_rows, use_container_width=True, hide_index=True)

    orders = alpaca_orders(key, secret, limit=6)
    if orders:
        ord_rows = []
        for o in orders:
            ord_rows.append({
                "time":   o.get("submitted_at", "")[11:19],
                "side":   o.get("side"),
                "qty":    o.get("qty"),
                "symbol": o.get("symbol"),
                "type":   o.get("order_type"),
                "status": o.get("status"),
            })
        st.write("**Recent orders**")
        st.dataframe(ord_rows, use_container_width=True, hide_index=True)
    st.divider()

# --- strategies ---
st.header("Strategies")

for sess, account, desc in SESSIONS:
    running, log = tmux_capture(sess, lines=25)
    dot = "🟢" if running else "🔴"
    st.subheader(f"{dot} {sess}")
    st.caption(f"{account} · {desc}")
    st.code(log, language="log")
