"""Portfolio monitoring dashboard — OBSERVATION ONLY (no kill-switches, no controls).

Single-page Streamlit app. Read-only against each strategy's sqlite ledger and
the host scripts' nav_state.json files. Designed to be viewed from a phone on
the same WiFi. User is in the UK so all wall-clock times are rendered in BST
alongside the relevant market timezone.

Launch (LAN-accessible on port 8503):
    streamlit run research/live/portfolio_dashboard.py \\
        --server.port 8503 --server.address 0.0.0.0

Strategies tracked:
    1. iter_3 BTC Funding Carry  (Kraken Futures Demo)
    2. SVXY Sweet Spot           (Alpaca AEGIS paper)
    3. iter_18 ADR Arbitrage     (Alpaca AEGIS paper, virtual HK leg)
    4. iter_6 Nikkei Gap ETF     (Alpaca AEGIS paper)

Each strategy is queried defensively: if the sqlite file or expected tables are
missing/empty the panel shows the absence explicitly rather than crashing.

Auto-refresh: 30s. Layout: wide, large fonts, mobile-friendly.
"""
from __future__ import annotations

import json
import math
import socket
import sqlite3
import subprocess
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
LONDON_TZ = ZoneInfo("Europe/London")
NY_TZ = ZoneInfo("America/New_York")
TOKYO_TZ = ZoneInfo("Asia/Tokyo")
UTC_TZ = ZoneInfo("UTC")

ROOT = Path("/Users/weeoto/turbo-carnival")
LIVE_DIR = ROOT / "research" / "live"

# DB file paths (match the actual filenames on disk)
BTC_DB = LIVE_DIR / "kraken_btc_funding_decisions.sqlite3"
SVXY_DB = LIVE_DIR / "svxy_decisions.sqlite3"
ADR_DB = LIVE_DIR / "adr_alpaca_decisions.sqlite3"
NIKKEI_DB = LIVE_DIR / "nikkei_etf_decisions.sqlite3"

# JSON nav state files written each cycle by the host scripts
BTC_NAV_JSON = LIVE_DIR / "kraken_btc_funding_nav_state.json"
ADR_NAV_JSON = LIVE_DIR / "adr_alpaca_nav_state.json"

REFRESH_SECS = 30

# Portfolio deployment date — the "running total since deployment" anchor.
DEPLOYMENT_DATE = date(2026, 5, 13)

# Strategy spec (drives panel ordering and labelling).
STRATEGIES: list[dict[str, Any]] = [
    {
        "key": "btc_funding",
        "name": "iter_3 BTC Funding Carry",
        "asset": "BTC perp + virtual spot (Kraken Futures Demo)",
        "db": BTC_DB,
        "nav_json": BTC_NAV_JSON,
        "broker_tz": UTC_TZ,
        "broker_tz_label": "UTC",
        "sod_anchor": "midnight_utc",
    },
    {
        "key": "svxy",
        "name": "SVXY Sweet Spot",
        "asset": "SVXY (Alpaca AEGIS paper)",
        "db": SVXY_DB,
        "nav_json": None,                  # equity comes from decisions table
        "broker_tz": NY_TZ,
        "broker_tz_label": "ET",
        "sod_anchor": "us_open_et",        # 09:30 ET
    },
    {
        "key": "adr",
        "name": "iter_18 ADR Arbitrage",
        "asset": "US ADR vs virtual HK (Alpaca AEGIS paper)",
        "db": ADR_DB,
        "nav_json": ADR_NAV_JSON,
        "broker_tz": NY_TZ,
        "broker_tz_label": "ET",
        "sod_anchor": "us_open_et",
    },
    {
        "key": "nikkei",
        "name": "iter_6 Nikkei Gap ETF",
        "asset": "EWJ / DXJ Nikkei ETF (Alpaca AEGIS paper)",
        "db": NIKKEI_DB,
        "nav_json": None,
        "broker_tz": TOKYO_TZ,
        "broker_tz_label": "JST",
        "sod_anchor": "jp_open_jst",       # 09:00 JST
    },
]


# ---------------------------------------------------------------------------
# Page config + CSS (wide, large fonts, dark, mobile-friendly)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Portfolio Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 0.8rem; padding-bottom: 1.5rem; max-width: 1600px; }
      html, body, [class*="css"] { font-size: 17px; }
      h1, h2, h3, h4 { color: #f1f5f9; }
      .clock-bar {
        display: flex; flex-wrap: wrap; gap: 0.6rem; align-items: center;
        background: #0b1220; border: 1px solid #1f2937; border-radius: 14px;
        padding: 0.7rem 1.0rem; margin-bottom: 0.6rem;
      }
      .clock-item { display: flex; flex-direction: column; min-width: 140px; }
      .clock-label { color: #94a3b8; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; }
      .clock-value { color: #f8fafc; font-size: 1.55rem; font-weight: 700; font-variant-numeric: tabular-nums; }
      .kpi-bar { display: flex; gap: 0.6rem; flex-wrap: wrap; margin: 0.3rem 0 0.6rem; }
      .kpi {
        background: #111827; border: 1px solid #1f2937; border-radius: 14px;
        padding: 0.7rem 1.0rem; min-width: 180px; flex: 1 1 200px;
      }
      .kpi-label { color: #94a3b8; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.04em; }
      .kpi-value { color: #f8fafc; font-size: 1.7rem; font-weight: 700; font-variant-numeric: tabular-nums; }
      .kpi-sub   { color: #9ca3af; font-size: 0.85rem; }
      .pos { color: #4ade80; }
      .neg { color: #f87171; }
      .strat-card {
        background: #0f172a; border: 1px solid #1e293b; border-radius: 16px;
        padding: 0.9rem 1.1rem; margin-bottom: 0.7rem;
      }
      .strat-title { font-size: 1.25rem; font-weight: 700; color: #e2e8f0; }
      .strat-sub   { font-size: 0.88rem; color: #94a3b8; margin-bottom: 0.2rem; }
      .stat-row {
        display: flex; justify-content: space-between; margin: 0.18rem 0;
        font-size: 1.0rem;
      }
      .stat-row .k { color: #94a3b8; }
      .stat-row .v { color: #e5e7eb; font-variant-numeric: tabular-nums; font-weight: 600; }
      .signal-box {
        background: #1e293b; border-left: 4px solid #38bdf8; padding: 0.5rem 0.8rem;
        border-radius: 8px; font-size: 0.98rem; color: #e2e8f0; margin: 0.3rem 0;
      }
      .lan-pill {
        display: inline-block; padding: 0.22rem 0.65rem; border-radius: 999px;
        background: #1f2937; color: #f3f4f6;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.90rem; margin-right: 0.35rem;
      }
      /* Mobile: collapse to single column, shrink padding, no horizontal scroll */
      @media (max-width: 768px) {
        .block-container { padding-left: 0.4rem; padding-right: 0.4rem; }
        html, body, [class*="css"] { font-size: 16px; }
        .clock-bar { flex-direction: column; align-items: flex-start; }
        .clock-item { min-width: 0; }
        .clock-value { font-size: 1.35rem; }
        .kpi { min-width: 0; flex: 1 1 100%; }
        .kpi-value { font-size: 1.4rem; }
        .strat-card { padding: 0.7rem 0.8rem; }
        [data-testid="column"] { width: 100% !important; flex: 1 1 100% !important; }
      }
      /* Avoid horizontal scroll on any dataframe */
      [data-testid="stDataFrame"] { overflow-x: auto; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)
def detect_lan_ip() -> str:
    candidates: list[str] = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            candidates.append(s.getsockname()[0])
        finally:
            s.close()
    except Exception:
        pass
    for iface in ("en0", "en1", "en2"):
        try:
            out = subprocess.run(
                ["ipconfig", "getifaddr", iface],
                capture_output=True, text=True, timeout=1.5,
            )
            ip = out.stdout.strip()
            if ip:
                candidates.append(ip)
        except Exception:
            pass
    for ip in candidates:
        if ip.startswith(("192.168.", "10.")) or any(
            ip.startswith(f"172.{n}.") for n in range(16, 32)
        ):
            return ip
    for ip in candidates:
        if ip and not ip.startswith("127."):
            return ip
    return "127.0.0.1"


def connect_readonly(path: Path) -> Optional[sqlite3.Connection]:
    if not path.exists():
        return None
    try:
        return sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    except sqlite3.Error:
        return None


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    try:
        cur = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        )
        return cur.fetchone() is not None
    except sqlite3.Error:
        return False


def safe_sql(path: Path, query: str, params: tuple[Any, ...] = ()) -> pd.DataFrame:
    conn = connect_readonly(path)
    if conn is None:
        return pd.DataFrame()
    try:
        return pd.read_sql_query(query, conn, params=params)
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()


def load_json_state(path: Optional[Path]) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def fmt_money(x: float | int | None, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if signed:
        sign = "+" if x > 0 else ""
        return f"{sign}${x:,.2f}"
    return f"${x:,.2f}"


def fmt_pct(x: float | None, places: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    sign = "+" if x > 0 else ""
    return f"{sign}{x:.{places}f}%"


def pnl_class(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)) or x == 0:
        return ""
    return "pos" if x > 0 else "neg"


def parse_utc(ts_raw: Any) -> Optional[datetime]:
    """Parse a timestamp string/value as UTC. Returns None if unparseable."""
    if ts_raw is None or (isinstance(ts_raw, float) and math.isnan(ts_raw)):
        return None
    try:
        ts = pd.to_datetime(ts_raw, utc=True)
        if pd.isna(ts):
            return None
        return ts.to_pydatetime()
    except Exception:
        return None


def to_bst(ts_utc: Optional[datetime]) -> str:
    if ts_utc is None:
        return "—"
    return ts_utc.astimezone(LONDON_TZ).strftime("%H:%M:%S")


def to_local(ts_utc: Optional[datetime], tz: ZoneInfo, fmt: str = "%H:%M:%S") -> str:
    if ts_utc is None:
        return "—"
    return ts_utc.astimezone(tz).strftime(fmt)


def start_of_day_utc(anchor: str, broker_tz: ZoneInfo, now_utc: datetime) -> datetime:
    """Compute the most-recent "start-of-day" boundary in UTC.

    - midnight_utc: 00:00 UTC today
    - us_open_et:   most-recent 09:30 ET (rolls back a day if not yet reached)
    - jp_open_jst:  most-recent 09:00 JST (rolls back a day if not yet reached)
    """
    if anchor == "midnight_utc":
        sod_local = now_utc.astimezone(UTC_TZ).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return sod_local
    if anchor == "us_open_et":
        local_now = now_utc.astimezone(broker_tz)
        candidate = local_now.replace(hour=9, minute=30, second=0, microsecond=0)
        if local_now < candidate:
            candidate -= timedelta(days=1)
        return candidate.astimezone(UTC_TZ)
    if anchor == "jp_open_jst":
        local_now = now_utc.astimezone(broker_tz)
        candidate = local_now.replace(hour=9, minute=0, second=0, microsecond=0)
        if local_now < candidate:
            candidate -= timedelta(days=1)
        return candidate.astimezone(UTC_TZ)
    # Fallback: midnight UTC
    return now_utc.astimezone(UTC_TZ).replace(hour=0, minute=0, second=0, microsecond=0)


# ---------------------------------------------------------------------------
# Strategy data loaders
# ---------------------------------------------------------------------------
def load_btc_funding(now_utc: datetime, cfg: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "key": cfg["key"], "name": cfg["name"], "asset": cfg["asset"],
        "db_exists": cfg["db"].exists(),
        "sod_utc": start_of_day_utc(cfg["sod_anchor"], cfg["broker_tz"], now_utc),
    }
    if not cfg["db"].exists():
        return out

    # NAV from json state (host writes this every cycle).
    nav_state = load_json_state(cfg["nav_json"])
    out["current_nav"] = float(nav_state.get("nav_usd") or 0.0)
    out["peak_nav"] = float(nav_state.get("peak_nav_usd") or out["current_nav"])
    out["cumulative_funding"] = float(nav_state.get("cumulative_funding_usd") or 0.0)
    out["cumulative_fees"] = float(nav_state.get("cumulative_fees_usd") or 0.0)

    # Start-of-day NAV: latest decisions row with ts_utc < sod_utc.
    sod_str = out["sod_utc"].strftime("%Y-%m-%dT%H:%M:%S")
    sod_nav_df = safe_sql(
        cfg["db"],
        "SELECT nav_usd FROM decisions WHERE ts_utc < ? "
        "AND nav_usd IS NOT NULL ORDER BY id DESC LIMIT 1",
        (sod_str,),
    )
    if not sod_nav_df.empty:
        out["sod_nav"] = float(sod_nav_df.iloc[0]["nav_usd"])
    else:
        # No earlier row — fall back to earliest seen NAV
        first_df = safe_sql(
            cfg["db"],
            "SELECT nav_usd FROM decisions WHERE nav_usd IS NOT NULL "
            "ORDER BY id ASC LIMIT 1",
        )
        out["sod_nav"] = (
            float(first_df.iloc[0]["nav_usd"]) if not first_df.empty
            else out["current_nav"]
        )

    # Latest signal / cycle for context.
    latest = safe_sql(
        cfg["db"],
        "SELECT ts_utc, cycle_type, signal, funding_rate, roll_apr, n_open, nav_usd "
        "FROM decisions ORDER BY id DESC LIMIT 1",
    )
    if not latest.empty:
        row = latest.iloc[0]
        out["last_ts_utc"] = parse_utc(row.get("ts_utc"))
        out["signal_text"] = (
            f"{row.get('cycle_type', '?')} cycle → signal={row.get('signal') or '—'} · "
            f"funding={(row.get('funding_rate') or 0.0)*100:.3f}% · "
            f"roll_apr={(row.get('roll_apr') or 0.0)*100:.2f}% · n_open={row.get('n_open') or 0}"
        )

    # Open positions (perp short + virtual spot long pair).
    open_pos = safe_sql(
        cfg["db"],
        "SELECT id, opened_ts_utc, qty_btc, perp_entry_px, spot_entry_px, "
        "funding_collected_usd, status FROM positions WHERE status='open' "
        "ORDER BY id DESC",
    )
    out["open_positions"] = open_pos

    # Closed positions → trade log (entry/exit time, qty, prices, P&L).
    closed = safe_sql(
        cfg["db"],
        "SELECT id, opened_ts_utc, closed_ts_utc, qty_btc, "
        "perp_entry_px, spot_entry_px, perp_exit_px, spot_exit_px, "
        "funding_collected_usd, exit_reason FROM positions "
        "WHERE status='closed' AND closed_ts_utc IS NOT NULL "
        "ORDER BY closed_ts_utc DESC LIMIT 10",
    )
    if not closed.empty:
        # Net P&L per closed position (delta-neutral perp short + spot long).
        # spot leg pnl = (spot_exit - spot_entry) * qty
        # perp leg pnl = (perp_entry - perp_exit) * qty   (short)
        # net = spot_pnl + perp_pnl + funding_collected
        closed["spot_pnl"] = (
            closed["spot_exit_px"] - closed["spot_entry_px"]
        ) * closed["qty_btc"]
        closed["perp_pnl"] = (
            closed["perp_entry_px"] - closed["perp_exit_px"]
        ) * closed["qty_btc"]
        closed["net_pnl_usd"] = (
            closed["spot_pnl"].fillna(0.0)
            + closed["perp_pnl"].fillna(0.0)
            + closed["funding_collected_usd"].fillna(0.0)
        )
    out["closed_positions"] = closed

    # Realized P&L today (sum of net_pnl for closes >= sod_utc).
    realized_today = 0.0
    if not closed.empty:
        closed_today = closed[closed["closed_ts_utc"] >= sod_str]
        if not closed_today.empty:
            realized_today = float(closed_today["net_pnl_usd"].sum())
    # Plus any funding collected today (approx from daily_summary).
    realized_today += float(
        safe_sql(
            cfg["db"],
            "SELECT COALESCE(SUM(net_pnl_usd),0) AS s FROM daily_summary "
            "WHERE day_utc = ?",
            (out["sod_utc"].strftime("%Y-%m-%d"),),
        ).iloc[0]["s"] if not safe_sql(
            cfg["db"],
            "SELECT COALESCE(SUM(net_pnl_usd),0) AS s FROM daily_summary "
            "WHERE day_utc = ?",
            (out["sod_utc"].strftime("%Y-%m-%d"),),
        ).empty else 0.0
    )
    out["realized_today"] = realized_today

    return out


def load_svxy(now_utc: datetime, cfg: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "key": cfg["key"], "name": cfg["name"], "asset": cfg["asset"],
        "db_exists": cfg["db"].exists(),
        "sod_utc": start_of_day_utc(cfg["sod_anchor"], cfg["broker_tz"], now_utc),
    }
    if not cfg["db"].exists():
        return out

    latest = safe_sql(
        cfg["db"],
        "SELECT * FROM decisions ORDER BY id DESC LIMIT 1",
    )
    if not latest.empty:
        row = latest.iloc[0]
        out["last_ts_utc"] = parse_utc(row.get("ts_utc"))
        out["current_nav"] = float(row.get("equity") or 0.0)
        out["peak_nav"] = float(row.get("equity_peak") or out["current_nav"])
        out["cash"] = float(row.get("cash") or 0.0)
        svxy_qty = float(row.get("svxy_qty_held") or 0.0)
        svxy_px = float(row.get("svxy_price") or 0.0)
        out["target"] = str(row.get("target") or "?")
        ratio = row.get("ratio")
        out["signal_text"] = (
            f"VIX/VIX3M = {float(ratio):.3f} → "
            f"{'LONG SVXY' if out['target']=='long' else 'FLAT (cash/SGOV)'}"
            if ratio is not None and not pd.isna(ratio) else
            "Awaiting first decision cycle"
        )

        open_positions = []
        if svxy_qty > 0:
            open_positions.append({
                "symbol": "SVXY", "qty": svxy_qty, "avg_price": svxy_px,
                "side": "long",
            })
        sgov_qty = float(row.get("sgov_qty_held") or 0.0)
        sgov_px = float(row.get("sgov_price") or 0.0)
        if sgov_qty > 0:
            open_positions.append({
                "symbol": "SGOV", "qty": sgov_qty, "avg_price": sgov_px,
                "side": "long",
            })
        out["open_positions"] = pd.DataFrame(open_positions) if open_positions else pd.DataFrame()

    # Start-of-day equity.
    sod_str = out["sod_utc"].strftime("%Y-%m-%dT%H:%M:%S")
    sod_df = safe_sql(
        cfg["db"],
        "SELECT equity FROM decisions WHERE ts_utc < ? AND equity IS NOT NULL "
        "ORDER BY id DESC LIMIT 1",
        (sod_str,),
    )
    if not sod_df.empty:
        out["sod_nav"] = float(sod_df.iloc[0]["equity"])
    else:
        first_df = safe_sql(
            cfg["db"],
            "SELECT equity FROM decisions WHERE equity IS NOT NULL "
            "ORDER BY id ASC LIMIT 1",
        )
        out["sod_nav"] = (
            float(first_df.iloc[0]["equity"]) if not first_df.empty
            else out.get("current_nav", 0.0)
        )

    # Trade log: rows where order_filled_qty > 0 (i.e. an actual fill happened).
    trades = safe_sql(
        cfg["db"],
        "SELECT ts_utc, order_action, order_filled_qty, order_filled_avg, "
        "svxy_price, equity FROM decisions "
        "WHERE order_filled_qty IS NOT NULL AND order_filled_qty > 0 "
        "ORDER BY id DESC LIMIT 10",
    )
    out["trade_log"] = trades

    # Today's realized P&L (proxy: equity delta from SOD).
    out["realized_today"] = (out.get("current_nav", 0.0) - out.get("sod_nav", 0.0))
    return out


def load_adr(now_utc: datetime, cfg: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "key": cfg["key"], "name": cfg["name"], "asset": cfg["asset"],
        "db_exists": cfg["db"].exists(),
        "sod_utc": start_of_day_utc(cfg["sod_anchor"], cfg["broker_tz"], now_utc),
    }
    if not cfg["db"].exists():
        return out

    # NAV from json state (cached by the host).
    nav_state = load_json_state(cfg["nav_json"])
    out["current_nav"] = float(nav_state.get("nav_usd") or 0.0)

    # Backstop: latest eod_summary row if json state empty.
    if out["current_nav"] <= 0:
        eod = safe_sql(
            cfg["db"],
            "SELECT nav_usd FROM eod_summary ORDER BY id DESC LIMIT 1",
        )
        if not eod.empty:
            out["current_nav"] = float(eod.iloc[0]["nav_usd"] or 0.0)

    # Latest decision context.
    latest = safe_sql(
        cfg["db"],
        "SELECT ts_utc, pair, dev_bps, action, direction, us_qty, hk_qty, reason "
        "FROM decisions ORDER BY id DESC LIMIT 1",
    )
    if not latest.empty:
        row = latest.iloc[0]
        out["last_ts_utc"] = parse_utc(row.get("ts_utc"))
        dev_bps = row.get("dev_bps")
        out["signal_text"] = (
            f"{row.get('pair', '?')} dev={float(dev_bps) if dev_bps is not None else 0:.1f} bps → "
            f"{(row.get('action') or 'HOLD').upper()}"
            f"{(' (' + str(row.get('reason')) + ')') if row.get('reason') else ''}"
        )

    # Start-of-day NAV from eod_summary.
    sod_str = out["sod_utc"].strftime("%Y-%m-%dT%H:%M:%S")
    sod_df = safe_sql(
        cfg["db"],
        "SELECT nav_usd FROM eod_summary WHERE ts_utc < ? AND nav_usd IS NOT NULL "
        "ORDER BY id DESC LIMIT 1",
        (sod_str,),
    )
    if not sod_df.empty:
        out["sod_nav"] = float(sod_df.iloc[0]["nav_usd"])
    else:
        first_df = safe_sql(
            cfg["db"],
            "SELECT nav_usd FROM eod_summary WHERE nav_usd IS NOT NULL "
            "ORDER BY id ASC LIMIT 1",
        )
        out["sod_nav"] = (
            float(first_df.iloc[0]["nav_usd"]) if not first_df.empty
            else out["current_nav"]
        )

    # Realized P&L today from latest eod_summary if it's in today's window.
    eod_today = safe_sql(
        cfg["db"],
        "SELECT real_us_pnl, sim_hk_pnl, total_combined_pnl FROM eod_summary "
        "WHERE ts_utc >= ? ORDER BY id DESC LIMIT 1",
        (sod_str,),
    )
    if not eod_today.empty:
        out["realized_today"] = float(eod_today.iloc[0]["total_combined_pnl"] or 0.0)
    else:
        out["realized_today"] = 0.0

    # Open positions snapshot (latest per symbol where qty > 0).
    snap = safe_sql(
        cfg["db"],
        "SELECT symbol, is_simulated, qty, avg_cost, market_value, unreal_pnl, ts_utc "
        "FROM positions_snapshot WHERE id IN ("
        "    SELECT MAX(id) FROM positions_snapshot GROUP BY symbol"
        ") ORDER BY symbol",
    )
    if not snap.empty:
        snap = snap[snap["qty"].fillna(0).abs() > 1e-9]
    out["open_positions"] = snap

    # Trade log (orders table, both real US and virtual HK).
    trades = safe_sql(
        cfg["db"],
        "SELECT ts_utc, pair, leg, symbol, side, qty, fill_price, status "
        "FROM orders ORDER BY id DESC LIMIT 10",
    )
    out["trade_log"] = trades
    return out


def load_nikkei(now_utc: datetime, cfg: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "key": cfg["key"], "name": cfg["name"], "asset": cfg["asset"],
        "db_exists": cfg["db"].exists(),
        "sod_utc": start_of_day_utc(cfg["sod_anchor"], cfg["broker_tz"], now_utc),
    }
    if not cfg["db"].exists():
        return out

    # NAV from latest eod_summaries row.
    eod = safe_sql(
        cfg["db"],
        "SELECT session_date, account_nav_usd, pnl_usd FROM eod_summaries "
        "ORDER BY session_date DESC LIMIT 1",
    )
    if not eod.empty:
        out["current_nav"] = float(eod.iloc[0]["account_nav_usd"] or 0.0)

    # Latest evaluation.
    eval_df = safe_sql(
        cfg["db"],
        "SELECT ts_utc, etf, gap_pct, threshold_pct, signal_dir, took_trade, reason "
        "FROM evaluations ORDER BY rowid DESC LIMIT 1",
    )
    if not eval_df.empty:
        row = eval_df.iloc[0]
        out["last_ts_utc"] = parse_utc(row.get("ts_utc"))
        out["signal_text"] = (
            f"{row.get('etf', '?')} gap={float(row.get('gap_pct') or 0.0):.2f}% "
            f"(threshold {float(row.get('threshold_pct') or 0.0):.2f}%) → "
            f"signal_dir={row.get('signal_dir')} took_trade={bool(row.get('took_trade'))} "
            f"reason={row.get('reason') or '—'}"
        )

    # SOD NAV (previous session's close).
    sod_str = out["sod_utc"].strftime("%Y-%m-%d")
    sod_df = safe_sql(
        cfg["db"],
        "SELECT account_nav_usd FROM eod_summaries WHERE session_date < ? "
        "ORDER BY session_date DESC LIMIT 1",
        (sod_str,),
    )
    if not sod_df.empty:
        out["sod_nav"] = float(sod_df.iloc[0]["account_nav_usd"] or 0.0)
    else:
        out["sod_nav"] = out.get("current_nav", 0.0)

    # Today's realized P&L from today's eod row.
    today_df = safe_sql(
        cfg["db"],
        "SELECT pnl_usd FROM eod_summaries WHERE session_date = ?",
        (sod_str,),
    )
    out["realized_today"] = (
        float(today_df.iloc[0]["pnl_usd"] or 0.0) if not today_df.empty else 0.0
    )

    # Open positions: derive from fills not yet exited. The schema doesn't have an
    # explicit "open positions" table — we surface "intraday fills with no
    # closing fill yet" as a proxy.
    out["open_positions"] = safe_sql(
        cfg["db"],
        "SELECT ts_utc, symbol, action, qty, fill_price FROM fills "
        "WHERE session_date >= ? ORDER BY ts_utc DESC",
        (sod_str,),
    )

    # Trade log: last 10 fills.
    trades = safe_sql(
        cfg["db"],
        "SELECT ts_utc, session_date, symbol, action, qty, fill_price, commission "
        "FROM fills ORDER BY ts_utc DESC LIMIT 10",
    )
    out["trade_log"] = trades
    return out


def load_all(now_utc: datetime) -> dict[str, dict[str, Any]]:
    return {
        "btc_funding": load_btc_funding(now_utc, STRATEGIES[0]),
        "svxy":        load_svxy(now_utc, STRATEGIES[1]),
        "adr":         load_adr(now_utc, STRATEGIES[2]),
        "nikkei":      load_nikkei(now_utc, STRATEGIES[3]),
    }


# ---------------------------------------------------------------------------
# Cumulative P&L since deployment (running total)
# ---------------------------------------------------------------------------
def cumulative_pnl_since_deployment(data: dict[str, dict[str, Any]]) -> float:
    """Sum of (current_nav − initial_nav_seen_in_db) across all 4 strategies.

    The initial-NAV proxy is the earliest non-null NAV in each strategy's
    history (which corresponds roughly to deployment day, 2026-05-13).
    """
    total = 0.0
    # BTC funding
    init = safe_sql(
        BTC_DB,
        "SELECT nav_usd FROM decisions WHERE nav_usd IS NOT NULL "
        "ORDER BY id ASC LIMIT 1",
    )
    if not init.empty:
        total += data["btc_funding"].get("current_nav", 0.0) - float(init.iloc[0]["nav_usd"])
    # SVXY
    init = safe_sql(
        SVXY_DB,
        "SELECT equity FROM decisions WHERE equity IS NOT NULL "
        "ORDER BY id ASC LIMIT 1",
    )
    if not init.empty:
        total += data["svxy"].get("current_nav", 0.0) - float(init.iloc[0]["equity"])
    # ADR
    init = safe_sql(
        ADR_DB,
        "SELECT nav_usd FROM eod_summary WHERE nav_usd IS NOT NULL "
        "ORDER BY id ASC LIMIT 1",
    )
    if not init.empty:
        total += data["adr"].get("current_nav", 0.0) - float(init.iloc[0]["nav_usd"])
    # Nikkei
    init = safe_sql(
        NIKKEI_DB,
        "SELECT account_nav_usd FROM eod_summaries WHERE account_nav_usd IS NOT NULL "
        "ORDER BY session_date ASC LIMIT 1",
    )
    if not init.empty:
        total += data["nikkei"].get("current_nav", 0.0) - float(init.iloc[0]["account_nav_usd"])
    return total


# ---------------------------------------------------------------------------
# Top header — wall-clock + portfolio KPIs
# ---------------------------------------------------------------------------
lan_ip = detect_lan_ip()
st.markdown("### Portfolio Dashboard — Observation Only")
st.markdown(
    f'<span class="lan-pill">http://{lan_ip}:8503/</span>'
    f'<span class="lan-pill">{socket.gethostname()}</span>'
    f'<span class="lan-pill">refresh: {REFRESH_SECS}s</span>',
    unsafe_allow_html=True,
)


@st.fragment(run_every=REFRESH_SECS)
def header_clock_and_kpis() -> None:
    now_utc = datetime.now(UTC_TZ)
    bst = now_utc.astimezone(LONDON_TZ)
    et = now_utc.astimezone(NY_TZ)
    jst = now_utc.astimezone(TOKYO_TZ)

    # London BST is the user's "home" clock; ET is the active US session clock.
    london_tz_label = bst.strftime("%Z")  # "BST" in summer, "GMT" in winter
    et_tz_label = et.strftime("%Z")       # "EDT" or "EST"
    jst_tz_label = jst.strftime("%Z")

    st.markdown(
        f'<div class="clock-bar">'
        f'<div class="clock-item"><div class="clock-label">London ({london_tz_label})</div>'
        f'<div class="clock-value">{bst.strftime("%H:%M:%S")}</div></div>'
        f'<div class="clock-item"><div class="clock-label">New York ({et_tz_label})</div>'
        f'<div class="clock-value">{et.strftime("%H:%M:%S")}</div></div>'
        f'<div class="clock-item"><div class="clock-label">Tokyo ({jst_tz_label})</div>'
        f'<div class="clock-value">{jst.strftime("%H:%M:%S")}</div></div>'
        f'<div class="clock-item"><div class="clock-label">Date (London)</div>'
        f'<div class="clock-value">{bst.strftime("%a %d %b")}</div></div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    # KPIs
    data = load_all(now_utc)
    total_nav = sum(d.get("current_nav", 0.0) for d in data.values())
    total_sod = sum(d.get("sod_nav", 0.0) for d in data.values())
    total_daily_delta = total_nav - total_sod
    total_realized_today = sum(d.get("realized_today", 0.0) for d in data.values())
    cum_pnl = cumulative_pnl_since_deployment(data)

    # Trades today: count fills/orders with ts in last 24h window.
    sod_utc_global = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    sod_str = sod_utc_global.strftime("%Y-%m-%dT%H:%M:%S")
    trades_today = 0
    for path, query, ts_col in [
        (BTC_DB, "SELECT COUNT(*) AS n FROM orders WHERE ts_utc >= ?", "ts_utc"),
        (SVXY_DB,
         "SELECT COUNT(*) AS n FROM decisions WHERE ts_utc >= ? "
         "AND order_filled_qty IS NOT NULL AND order_filled_qty > 0", "ts_utc"),
        (ADR_DB, "SELECT COUNT(*) AS n FROM orders WHERE ts_utc >= ?", "ts_utc"),
        (NIKKEI_DB, "SELECT COUNT(*) AS n FROM fills WHERE ts_utc >= ?", "ts_utc"),
    ]:
        df = safe_sql(path, query, (sod_str,))
        if not df.empty:
            trades_today += int(df.iloc[0]["n"] or 0)

    daily_cls = pnl_class(total_daily_delta)
    cum_cls = pnl_class(cum_pnl)

    days_since_deploy = (date.today() - DEPLOYMENT_DATE).days

    st.markdown(
        f'<div class="kpi-bar">'
        f'<div class="kpi"><div class="kpi-label">Total portfolio value</div>'
        f'<div class="kpi-value">${total_nav:,.0f}</div>'
        f'<div class="kpi-sub">across 4 strategies</div></div>'
        f'<div class="kpi"><div class="kpi-label">Today\'s P&L</div>'
        f'<div class="kpi-value {daily_cls}">{fmt_money(total_daily_delta, signed=True)}</div>'
        f'<div class="kpi-sub">SOD-anchored NAV delta</div></div>'
        f'<div class="kpi"><div class="kpi-label">Cumulative P&L since {DEPLOYMENT_DATE.isoformat()}</div>'
        f'<div class="kpi-value {cum_cls}">{fmt_money(cum_pnl, signed=True)}</div>'
        f'<div class="kpi-sub">day {days_since_deploy} of run</div></div>'
        f'<div class="kpi"><div class="kpi-label">Trades today</div>'
        f'<div class="kpi-value">{trades_today}</div>'
        f'<div class="kpi-sub">orders + fills across all strats</div></div>'
        f"</div>",
        unsafe_allow_html=True,
    )


header_clock_and_kpis()


# ---------------------------------------------------------------------------
# Per-strategy panels
# ---------------------------------------------------------------------------
def _localize_ts_column(df: pd.DataFrame, src_col: str = "ts_utc") -> pd.DataFrame:
    """Add a 'time_bst' column (HH:MM:SS BST) derived from src_col (UTC ISO)."""
    if df is None or df.empty or src_col not in df.columns:
        return df
    out = df.copy()
    parsed = pd.to_datetime(out[src_col], utc=True, errors="coerce")
    out["time_bst"] = parsed.dt.tz_convert(LONDON_TZ).dt.strftime("%Y-%m-%d %H:%M:%S")
    return out


def _render_open_positions(strat: dict[str, Any]) -> None:
    pos = strat.get("open_positions")
    if pos is None:
        st.caption("No open-position data.")
        return
    if isinstance(pos, pd.DataFrame):
        if pos.empty:
            st.caption("No open positions.")
            return
        # Localize any timestamp column.
        show = _localize_ts_column(pos)
        # Drop the original raw UTC ts column when we have a BST version.
        if "time_bst" in show.columns:
            show = show.drop(columns=[c for c in ("ts_utc", "opened_ts_utc") if c in show.columns])
        st.dataframe(
            show, hide_index=True, width="stretch",
            height=min(40 * max(1, len(show)) + 38, 220),
        )


def _render_trade_log(strat: dict[str, Any]) -> None:
    log = strat.get("trade_log")
    if not isinstance(log, pd.DataFrame) or log.empty:
        st.caption("No trades yet.")
        return
    show = log.copy()
    # Build a unified BST time column.
    ts_col = next((c for c in ("ts_utc", "closed_ts_utc") if c in show.columns), None)
    if ts_col:
        show = _localize_ts_column(show, src_col=ts_col)
        # Move time_bst to the front for readability.
        cols = ["time_bst"] + [c for c in show.columns if c not in ("time_bst", ts_col)]
        show = show[cols]
    # Format any pnl columns as $.
    for c in show.columns:
        if "pnl" in c.lower():
            show[c] = show[c].apply(
                lambda v: fmt_money(v, signed=True) if pd.notna(v) else "—"
            )
        elif c.endswith("_price") or c == "fill_price" or "_px" in c:
            show[c] = show[c].apply(
                lambda v: f"{float(v):,.4f}" if pd.notna(v) else "—"
            )
    st.dataframe(
        show, hide_index=True, width="stretch",
        height=min(40 * len(show) + 38, 360),
    )


def _render_closed_positions_log(strat: dict[str, Any]) -> None:
    closed = strat.get("closed_positions")
    if not isinstance(closed, pd.DataFrame) or closed.empty:
        st.caption("No closed positions yet.")
        return
    show = closed.copy()
    show["opened_bst"] = pd.to_datetime(show["opened_ts_utc"], utc=True, errors="coerce") \
        .dt.tz_convert(LONDON_TZ).dt.strftime("%Y-%m-%d %H:%M:%S")
    show["closed_bst"] = pd.to_datetime(show["closed_ts_utc"], utc=True, errors="coerce") \
        .dt.tz_convert(LONDON_TZ).dt.strftime("%Y-%m-%d %H:%M:%S")
    keep = [
        "opened_bst", "closed_bst", "qty_btc",
        "perp_entry_px", "perp_exit_px",
        "spot_entry_px", "spot_exit_px",
        "funding_collected_usd", "net_pnl_usd", "exit_reason",
    ]
    show = show[[c for c in keep if c in show.columns]]
    for c in ("perp_entry_px", "perp_exit_px", "spot_entry_px", "spot_exit_px"):
        if c in show.columns:
            show[c] = show[c].apply(
                lambda v: f"${float(v):,.2f}" if pd.notna(v) else "—"
            )
    for c in ("funding_collected_usd", "net_pnl_usd"):
        if c in show.columns:
            show[c] = show[c].apply(
                lambda v: fmt_money(v, signed=True) if pd.notna(v) else "—"
            )
    st.dataframe(
        show, hide_index=True, width="stretch",
        height=min(40 * len(show) + 38, 360),
    )


def _render_strategy_panel(cfg: dict[str, Any], strat: dict[str, Any]) -> None:
    daily_delta = strat.get("current_nav", 0.0) - strat.get("sod_nav", 0.0)
    daily_cls = pnl_class(daily_delta)
    realized_cls = pnl_class(strat.get("realized_today"))

    ts_label = (
        f"last update: {to_bst(strat.get('last_ts_utc'))} BST · "
        f"{to_local(strat.get('last_ts_utc'), cfg['broker_tz'])} {cfg['broker_tz_label']}"
        if strat.get("last_ts_utc") else "no host updates yet"
    )

    sod_dt = strat.get("sod_utc")
    sod_label = (
        f"SOD anchor: {to_bst(sod_dt)} BST · "
        f"{to_local(sod_dt, cfg['broker_tz'])} {cfg['broker_tz_label']}"
        if sod_dt else "—"
    )

    head_html = (
        f'<div class="strat-card">'
        f'<div class="strat-title">{cfg["name"]}</div>'
        f'<div class="strat-sub">{cfg["asset"]} · db: <code>{cfg["db"].name}</code></div>'
        f'<div class="strat-sub">{ts_label}</div>'
        f'<div class="strat-sub">{sod_label}</div>'
        f'</div>'
    )
    st.markdown(head_html, unsafe_allow_html=True)

    # 3-column layout on desktop (signal/stats, NAV/delta, today's P&L).
    col_sig, col_nav, col_pnl = st.columns([3, 2, 2], gap="medium")

    with col_sig:
        sig = strat.get("signal_text") or "no signal yet — host idle"
        st.markdown(
            f'<div class="signal-box"><b>Latest signal:</b> {sig}</div>',
            unsafe_allow_html=True,
        )

    with col_nav:
        st.markdown(
            f'<div class="stat-row"><span class="k">Current account value</span>'
            f'<span class="v">{fmt_money(strat.get("current_nav"))}</span></div>'
            f'<div class="stat-row"><span class="k">Start-of-day value</span>'
            f'<span class="v">{fmt_money(strat.get("sod_nav"))}</span></div>'
            f'<div class="stat-row"><span class="k">Δ vs start-of-day</span>'
            f'<span class="v {daily_cls}">{fmt_money(daily_delta, signed=True)}</span></div>',
            unsafe_allow_html=True,
        )

    with col_pnl:
        st.markdown(
            f'<div class="stat-row"><span class="k">Realized P&L today</span>'
            f'<span class="v {realized_cls}">{fmt_money(strat.get("realized_today"), signed=True)}</span></div>',
            unsafe_allow_html=True,
        )
        if cfg["key"] == "btc_funding":
            st.markdown(
                f'<div class="stat-row"><span class="k">Cumulative funding</span>'
                f'<span class="v">{fmt_money(strat.get("cumulative_funding"))}</span></div>'
                f'<div class="stat-row"><span class="k">Cumulative fees</span>'
                f'<span class="v">{fmt_money(strat.get("cumulative_fees"))}</span></div>',
                unsafe_allow_html=True,
            )

    # Open positions and trade log.
    st.markdown("**Open positions**")
    _render_open_positions(strat)

    # BTC has a separate closed-positions ledger with per-position P&L.
    if cfg["key"] == "btc_funding":
        st.markdown("**Closed positions (last 10)**")
        _render_closed_positions_log(strat)
    st.markdown("**Trade log — last 10 (times in BST)**")
    _render_trade_log(strat)


@st.fragment(run_every=REFRESH_SECS)
def strategy_panels() -> None:
    now_utc = datetime.now(UTC_TZ)
    data = load_all(now_utc)
    st.markdown("## Strategies")
    for cfg in STRATEGIES:
        _render_strategy_panel(cfg, data.get(cfg["key"], {}))


strategy_panels()


# ---------------------------------------------------------------------------
# Bottom summary — combined exposure + running cumulative since deployment
# ---------------------------------------------------------------------------
@st.fragment(run_every=REFRESH_SECS)
def portfolio_summary() -> None:
    now_utc = datetime.now(UTC_TZ)
    data = load_all(now_utc)

    total_nav = sum(d.get("current_nav", 0.0) for d in data.values())
    total_sod = sum(d.get("sod_nav", 0.0) for d in data.values())
    total_daily_delta = total_nav - total_sod
    total_realized = sum(d.get("realized_today", 0.0) for d in data.values())
    cum_pnl = cumulative_pnl_since_deployment(data)

    # Open exposure = sum of |market_value| (or |qty*price|) across all open positions.
    # Best-effort: pull from positions_snapshot (ADR) + open BTC perp + SVXY position.
    exposure = 0.0

    # BTC: qty_btc * perp_entry_px (current position value).
    btc_pos = data.get("btc_funding", {}).get("open_positions")
    if isinstance(btc_pos, pd.DataFrame) and not btc_pos.empty:
        exposure += float(
            (btc_pos["qty_btc"].fillna(0.0) * btc_pos["perp_entry_px"].fillna(0.0)).abs().sum()
        )
    # ADR: positions_snapshot market_value.
    adr_pos = data.get("adr", {}).get("open_positions")
    if isinstance(adr_pos, pd.DataFrame) and not adr_pos.empty and "market_value" in adr_pos.columns:
        exposure += float(adr_pos["market_value"].fillna(0.0).abs().sum())
    # SVXY: qty * avg_price.
    svxy_pos = data.get("svxy", {}).get("open_positions")
    if isinstance(svxy_pos, pd.DataFrame) and not svxy_pos.empty:
        exposure += float(
            (svxy_pos["qty"].fillna(0.0) * svxy_pos["avg_price"].fillna(0.0)).abs().sum()
        )
    # Nikkei: fills today qty * fill_price.
    nik_pos = data.get("nikkei", {}).get("open_positions")
    if isinstance(nik_pos, pd.DataFrame) and not nik_pos.empty and {"qty", "fill_price"} <= set(nik_pos.columns):
        exposure += float(
            (nik_pos["qty"].fillna(0.0) * nik_pos["fill_price"].fillna(0.0)).abs().sum()
        )

    daily_cls = pnl_class(total_daily_delta)
    realized_cls = pnl_class(total_realized)
    cum_cls = pnl_class(cum_pnl)

    st.markdown("## Portfolio combined")
    st.markdown(
        f'<div class="kpi-bar">'
        f'<div class="kpi"><div class="kpi-label">Combined Δ today (SOD)</div>'
        f'<div class="kpi-value {daily_cls}">{fmt_money(total_daily_delta, signed=True)}</div>'
        f'<div class="kpi-sub">total NAV delta vs SOD</div></div>'
        f'<div class="kpi"><div class="kpi-label">Combined realized today</div>'
        f'<div class="kpi-value {realized_cls}">{fmt_money(total_realized, signed=True)}</div>'
        f'<div class="kpi-sub">closed P&L per host</div></div>'
        f'<div class="kpi"><div class="kpi-label">Open gross exposure</div>'
        f'<div class="kpi-value">{fmt_money(exposure)}</div>'
        f'<div class="kpi-sub">|notional| across legs</div></div>'
        f'<div class="kpi"><div class="kpi-label">Cumulative since deploy</div>'
        f'<div class="kpi-value {cum_cls}">{fmt_money(cum_pnl, signed=True)}</div>'
        f'<div class="kpi-sub">anchor: {DEPLOYMENT_DATE.isoformat()}</div></div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    # Per-strategy summary table.
    rows = []
    for cfg in STRATEGIES:
        d = data.get(cfg["key"], {})
        rows.append({
            "strategy": cfg["name"],
            "asset": cfg["asset"].split(" (")[0],
            "current_nav": d.get("current_nav", 0.0),
            "sod_nav": d.get("sod_nav", 0.0),
            "delta_today": d.get("current_nav", 0.0) - d.get("sod_nav", 0.0),
            "realized_today": d.get("realized_today", 0.0),
        })
    sum_df = pd.DataFrame(rows)
    st.dataframe(
        sum_df.style.format({
            "current_nav": "${:,.2f}",
            "sod_nav": "${:,.2f}",
            "delta_today": "${:+,.2f}",
            "realized_today": "${:+,.2f}",
        }),
        hide_index=True, width="stretch",
    )


portfolio_summary()


st.caption(
    f"Read-only dashboard. NAV pulled from each strategy's sqlite ledger and "
    f"nav_state.json. Times rendered in London BST plus the relevant market "
    f"timezone. Auto-refresh every {REFRESH_SECS}s. Deployment anchor: "
    f"{DEPLOYMENT_DATE.isoformat()}."
)
