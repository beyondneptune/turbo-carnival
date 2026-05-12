"""Mac mini LAN-accessible system + trading dashboard (port 8502).

Read-only. Uses st.fragment so only live sections re-run on a timer; the page
itself never reloads. Each tab's auto-refresh is independent.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

NY_TZ = ZoneInfo("America/New_York")
LONDON_TZ = ZoneInfo("Europe/London")

import pandas as pd
import psutil
import requests
import streamlit as st

# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_LOCAL_PATH = REPO_ROOT / "council" / ".env.local"
LOG_DIR = REPO_ROOT / "data" / "logs"
NEWS_DIR = REPO_ROOT / "data" / "news"

KNOWN_LOGS = [
    "alpaca_max_hft_host.tmux.log",
    "alpaca_three_lane_monitor.tmux.log",
    "twitter_scraper.log",
    "stocktwits_scraper.log",
    "yahoo_news_scraper.log",
    "news_scraper.log",
    "system_dashboard.log",
]

LANES = ("Aegis", "Falcon", "Valk")
LANE_DESC = {
    "Aegis": "HFT mean-reversion",
    "Falcon": "HFT breakout",
    "Valk": "SPY iron condor",
}

st.set_page_config(
    page_title="Mac Mini Trading Dashboard",
    page_icon="🛰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.0rem; padding-bottom: 1.5rem; max-width: 1500px; }
      h1, h2, h3, h4 { color: #f1f5f9; }
      .kpi-bar {
        display: flex; gap: 0.5rem; flex-wrap: wrap; margin: 0.4rem 0 0.6rem 0;
      }
      .kpi {
        background: #111827; border: 1px solid #1f2937; border-radius: 12px;
        padding: 0.55rem 0.85rem; min-width: 130px;
      }
      .kpi-label { color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em; }
      .kpi-value { color: #f8fafc; font-size: 1.35rem; font-weight: 700; font-variant-numeric: tabular-nums; }
      .kpi-sub   { color: #9ca3af; font-size: 0.78rem; }
      .pos { color: #4ade80; }
      .neg { color: #f87171; }
      .neutral { color: #cbd5e1; }
      .lane-card {
        background: #0f172a; border: 1px solid #1e293b; border-radius: 14px;
        padding: 0.9rem 1.0rem; height: 100%;
      }
      .lane-title { font-size: 1.1rem; font-weight: 700; color: #e2e8f0; }
      .lane-subtitle { font-size: 0.78rem; color: #94a3b8; margin-bottom: 0.4rem; }
      .pill {
        display: inline-block; padding: 0.18rem 0.55rem; border-radius: 999px;
        font-size: 0.72rem; font-weight: 600; letter-spacing: 0.03em;
      }
      .pill-green { background: rgba(34,197,94,0.18); color: #4ade80; border: 1px solid rgba(34,197,94,0.35); }
      .pill-red   { background: rgba(239,68,68,0.18); color: #f87171; border: 1px solid rgba(239,68,68,0.35); }
      .pill-gray  { background: rgba(148,163,184,0.18); color: #cbd5e1; border: 1px solid rgba(148,163,184,0.35); }
      .stat-row { display: flex; justify-content: space-between; margin: 0.18rem 0; font-size: 0.92rem; }
      .stat-row .k { color: #94a3b8; }
      .stat-row .v { color: #e5e7eb; font-variant-numeric: tabular-nums; font-weight: 600; }
      .pos-table { font-size: 0.85rem; }
      .lan-pill {
        display: inline-block; padding: 0.22rem 0.65rem; border-radius: 999px;
        background: #1f2937; color: #f3f4f6; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.85rem; margin-right: 0.35rem;
      }
      div[data-testid="stMetric"] { background: transparent; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers (cached / fast)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=60)
def load_env_local() -> dict[str, str]:
    env: dict[str, str] = {}
    if not ENV_LOCAL_PATH.exists():
        return env
    for raw in ENV_LOCAL_PATH.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


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

    def _is_lan(ip: str) -> bool:
        return (
            ip.startswith("192.168.")
            or ip.startswith("10.")
            or any(ip.startswith(f"172.{n}.") for n in range(16, 32))
        )

    for ip in candidates:
        if _is_lan(ip):
            return ip
    for ip in candidates:
        if ip and not ip.startswith("127."):
            return ip
    return "127.0.0.1"


def human_bytes(n: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    n = float(n)
    while n >= 1024 and i < len(units) - 1:
        n /= 1024.0
        i += 1
    return f"{n:.1f} {units[i]}"


def human_uptime(seconds: float) -> str:
    seconds = int(seconds)
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if d:
        return f"{d}d {h}h {m}m"
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def market_status() -> tuple[bool, str, datetime]:
    """Check US regular market hours using America/New_York TZ explicitly."""
    ny = datetime.now(NY_TZ)
    if ny.weekday() >= 5:
        return False, f"Weekend • ET {ny.strftime('%H:%M')}", ny
    mod = ny.hour * 60 + ny.minute
    if 9 * 60 + 30 <= mod < 16 * 60:
        return True, f"Open • ET {ny.strftime('%H:%M')}", ny
    return False, f"Closed • ET {ny.strftime('%H:%M')}", ny


def safe_run(args: list[str], timeout: float = 4.0) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr
    except Exception as exc:
        return -1, "", f"{type(exc).__name__}: {exc}"


def read_tail(path: Path, n: int = 30) -> str:
    try:
        size = path.stat().st_size
        if size <= 0:
            return ""
        block = min(size, 256 * 1024)
        with path.open("rb") as f:
            if size >= block:
                f.seek(-block, os.SEEK_END)
            data = f.read()
        text = data.decode("utf-8", errors="replace")
        return "\n".join(text.splitlines()[-n:])
    except Exception as exc:
        return f"(read error: {exc})"


def alpaca_lane_creds(env: dict[str, str], lane: str) -> tuple[str | None, str | None, str]:
    key = env.get(f"ALPACA_{lane.upper()}_KEY")
    sec = env.get(f"ALPACA_{lane.upper()}_SECRET")
    endpoint = env.get("ALPACA_ENDPOINT", "https://paper-api.alpaca.markets/v2")
    return key, sec, endpoint


def alpaca_fetch(endpoint: str, key: str, secret: str, path: str) -> Any:
    resp = requests.get(
        endpoint.rstrip("/") + path,
        headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
        timeout=6,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_all_lanes(env: dict[str, str]) -> dict[str, dict[str, Any]]:
    """Pull account + positions for all lanes, one network call set per refresh."""
    out: dict[str, dict[str, Any]] = {}
    for lane in LANES:
        key, sec, ep = alpaca_lane_creds(env, lane)
        if not key or not sec:
            out[lane] = {"error": "credentials missing", "ok": False}
            continue
        try:
            acct = alpaca_fetch(ep, key, sec, "/account")
            pos = alpaca_fetch(ep, key, sec, "/positions") or []
            out[lane] = {"ok": True, "account": acct, "positions": pos}
        except requests.HTTPError as exc:
            code = exc.response.status_code if exc.response else "?"
            out[lane] = {"ok": False, "error": f"HTTP {code}"}
        except Exception as exc:
            out[lane] = {"ok": False, "error": str(exc)}
    return out


def render_lane_card(lane: str, data: dict[str, Any]) -> None:
    if not data.get("ok"):
        st.markdown(
            f'<div class="lane-card"><div class="lane-title">{lane}</div>'
            f'<div class="lane-subtitle">{LANE_DESC.get(lane, "")}</div>'
            f'<div class="pill pill-red">ERROR: {data.get("error", "?")}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
        return

    acct = data["account"]
    pos = data["positions"]
    eq = float(acct.get("equity") or 0.0)
    last_eq = float(acct.get("last_equity") or 0.0)
    pnl = eq - last_eq
    cash = float(acct.get("cash") or 0.0)
    bp = float(acct.get("buying_power") or 0.0)
    status = acct.get("status", "?")
    blocked = acct.get("trading_blocked", False)

    pnl_cls = "pos" if pnl >= 0 else "neg"
    pnl_sign = "+" if pnl >= 0 else ""
    pill_cls = "pill-green" if status == "ACTIVE" and not blocked else "pill-red"

    rows_html = "".join(
        [
            f'<div class="stat-row"><span class="k">Equity</span>'
            f'<span class="v">${eq:,.2f}</span></div>',
            f'<div class="stat-row"><span class="k">P&L today</span>'
            f'<span class="v {pnl_cls}">{pnl_sign}${pnl:,.2f}</span></div>',
            f'<div class="stat-row"><span class="k">Cash</span>'
            f'<span class="v">${cash:,.2f}</span></div>',
            f'<div class="stat-row"><span class="k">Buying power</span>'
            f'<span class="v">${bp:,.2f}</span></div>',
            f'<div class="stat-row"><span class="k">Positions</span>'
            f'<span class="v">{len(pos)}</span></div>',
        ]
    )

    st.markdown(
        f'<div class="lane-card">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
        f'<div><div class="lane-title">{lane}</div>'
        f'<div class="lane-subtitle">{LANE_DESC.get(lane, "")}</div></div>'
        f'<span class="pill {pill_cls}">{status}</span></div>'
        f"{rows_html}</div>",
        unsafe_allow_html=True,
    )

    if pos:
        rows = []
        for p in pos:
            try:
                rows.append({
                    "symbol": p.get("symbol"),
                    "qty": float(p.get("qty") or 0),
                    "side": p.get("side"),
                    "avg": float(p.get("avg_entry_price") or 0),
                    "uPL$": float(p.get("unrealized_pl") or 0),
                    "uPL%": float(p.get("unrealized_plpc") or 0) * 100.0,
                })
            except (TypeError, ValueError):
                continue
        df = pd.DataFrame(rows)
        styled = df.style.format({
            "avg": "${:,.2f}",
            "uPL$": "${:+,.2f}",
            "uPL%": "{:+.2f}%",
            "qty": "{:g}",
        }).map(
            lambda v: "color:#4ade80;font-weight:600;" if v > 0 else ("color:#f87171;font-weight:600;" if v < 0 else ""),
            subset=["uPL$", "uPL%"],
        )
        st.dataframe(styled, hide_index=True, width="stretch", height=min(38 * (len(rows) + 1) + 6, 200))


# ---------------------------------------------------------------------------
# Header — static
# ---------------------------------------------------------------------------
lan_ip = detect_lan_ip()
hdr_l, hdr_r = st.columns([3, 2])
with hdr_l:
    st.markdown("### 🛰 Mac Mini Trading Dashboard")
    st.markdown(
        f'<span class="lan-pill">http://{lan_ip}:8502/</span>'
        f'<span class="lan-pill">{socket.gethostname()}</span>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# KPI bar — fragment, refreshes every 15s
# ---------------------------------------------------------------------------
@st.fragment(run_every=15)
def kpi_bar() -> None:
    env = load_env_local()
    lanes = fetch_all_lanes(env)
    total_eq = 0.0
    total_pnl = 0.0
    total_pos = 0
    ok_lanes = 0
    for d in lanes.values():
        if not d.get("ok"):
            continue
        ok_lanes += 1
        acct = d["account"]
        total_eq += float(acct.get("equity") or 0.0)
        total_pnl += float(acct.get("equity") or 0.0) - float(acct.get("last_equity") or 0.0)
        total_pos += len(d["positions"])

    is_open, mk_label, ny = market_status()
    london = datetime.now(LONDON_TZ)
    pnl_cls = "pos" if total_pnl >= 0 else "neg"
    pnl_sign = "+" if total_pnl >= 0 else ""
    mk_pill = "pill-green" if is_open else "pill-gray"

    st.markdown(
        f'<div class="kpi-bar">'
        f'<div class="kpi"><div class="kpi-label">Total equity</div>'
        f'<div class="kpi-value">${total_eq:,.0f}</div>'
        f'<div class="kpi-sub">{ok_lanes}/{len(LANES)} lanes</div></div>'
        f'<div class="kpi"><div class="kpi-label">P&L today</div>'
        f'<div class="kpi-value {pnl_cls}">{pnl_sign}${total_pnl:,.0f}</div>'
        f'<div class="kpi-sub">across {ok_lanes} lanes</div></div>'
        f'<div class="kpi"><div class="kpi-label">Open positions</div>'
        f'<div class="kpi-value">{total_pos}</div>'
        f'<div class="kpi-sub">live broker count</div></div>'
        f'<div class="kpi"><div class="kpi-label">Market (ET)</div>'
        f'<div class="kpi-value">{ny.strftime("%H:%M")}</div>'
        f'<div class="kpi-sub"><span class="pill {mk_pill}">{mk_label}</span></div></div>'
        f'<div class="kpi"><div class="kpi-label">London (BST/GMT)</div>'
        f'<div class="kpi-value">{london.strftime("%H:%M:%S")}</div>'
        f'<div class="kpi-sub">{london.strftime("%a %b %d")} • mkt closes 21:00 BST</div></div>'
        f"</div>",
        unsafe_allow_html=True,
    )


kpi_bar()

tab_trading, tab_mlzoo, tab_system, tab_power, tab_tmux, tab_news = st.tabs(
    ["Live Trading", "ML Zoo (today)", "System", "Mac Power", "Tmux & Logs", "News Feed"],
)


# =========================================================================
# Tab — Live Trading (fragment, 15s)
# =========================================================================
with tab_trading:
    @st.fragment(run_every=15)
    def trading_fragment() -> None:
        env = load_env_local()
        if not env:
            st.error(f"Could not load credentials from {ENV_LOCAL_PATH}.")
            return
        lanes = fetch_all_lanes(env)
        cols = st.columns(len(LANES), gap="medium")
        for idx, lane in enumerate(LANES):
            with cols[idx]:
                render_lane_card(lane, lanes.get(lane, {"ok": False, "error": "no data"}))
        st.caption(f"Refreshed {datetime.now().strftime('%H:%M:%S')} • auto-refresh 15s")

    trading_fragment()


# =========================================================================
# Tab — ML Zoo (parallel-model backtest on today's trades)
# =========================================================================
ML_ZOO_DIR = REPO_ROOT / "experiments" / "ml_lab" / "runs"
ML_SUM_FILE = ML_ZOO_DIR / "ml_zoo_summary_latest.parquet"
ML_PER_FILE = ML_ZOO_DIR / "ml_zoo_per_entry_latest.parquet"

with tab_mlzoo:
    st.markdown("### ML Zoo — counterfactual P&L on today's entries")
    st.caption(
        "18 ML models trained on bars from days 1-20 (2026-04-13 → 2026-05-08), "
        "evaluated on today's 162 matched entries. "
        "Counterfactual P&L = realised P&L of entries the model *would have allowed*. "
        "Block decision: model's predicted P(next-5min |ret| > 30 bps) > 0.5. "
        "Realised P&L computed from minute-bar closes; ignores fills/slippage so it's offset from the live −$3,443."
    )
    if not ML_SUM_FILE.exists():
        st.warning(
            "No ML zoo results yet. Run:\n\n"
            "`.venv/bin/python experiments/ml_lab/04_ml_zoo_today_backtest.py`"
        )
    else:
        summary = pd.read_parquet(ML_SUM_FILE)
        per_entry = pd.read_parquet(ML_PER_FILE)

        actual_pnl = float(summary["actual_pnl"].iloc[0]) if "actual_pnl" in summary.columns else 0.0
        n_entries = int(summary["n_entries_seen"].iloc[0]) if "n_entries_seen" in summary.columns else 0
        n_winners = int((per_entry[per_entry["model"] == summary["model"].iloc[0]]["pnl"] > 0).sum())
        n_losers = int((per_entry[per_entry["model"] == summary["model"].iloc[0]]["pnl"] < 0).sum())

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Matched entries today", f"{n_entries}")
        k2.metric("Realised winners / losers", f"{n_winners} / {n_losers}")
        k3.metric("Actual P&L (bar-close basis)", f"${actual_pnl:+,.2f}")
        best_row = summary.iloc[0]
        k4.metric(
            "Best model counterfactual",
            f"${best_row['counterfactual_pnl']:+,.2f}",
            delta=f"${best_row['pnl_delta']:+,.2f} vs actual",
        )

        st.markdown("#### Model leaderboard — ranked by P&L delta")
        st.caption("Positive delta = the model would have saved money by blocking entries.")
        show = summary.copy()
        for c in ("actual_pnl", "counterfactual_pnl", "pnl_delta", "blocked_pnl_avoided"):
            if c in show.columns:
                show[c] = show[c].round(2)
        if "block_rate_pct" in show.columns:
            show["block_rate_pct"] = show["block_rate_pct"].round(1)
        if "avg_block_proba" in show.columns:
            show["avg_block_proba"] = show["avg_block_proba"].round(3)
        st.dataframe(show, use_container_width=True, hide_index=True)

        st.markdown("#### Per-entry decisions — pick a model to inspect")
        model_choice = st.selectbox(
            "Model", summary["model"].tolist(),
            index=0, key="ml_zoo_model_pick",
        )
        sub = per_entry[per_entry["model"] == model_choice].copy()
        sub["pnl"] = sub["pnl"].round(2)
        sub["entry_px"] = sub["entry_px"].round(4)
        sub["exit_px"] = sub["exit_px"].round(4)
        sub["model_proba"] = sub["model_proba"].round(3)
        sub["hold_min"] = sub["hold_min"].round(1)
        sub["entry_ts"] = pd.to_datetime(sub["entry_ts"]).dt.strftime("%H:%M:%S")
        sub["exit_ts"] = pd.to_datetime(sub["exit_ts"]).dt.strftime("%H:%M:%S")
        sub = sub[
            ["lane", "symbol", "side_dir", "entry_ts", "entry_px",
             "exit_ts", "exit_px", "exit_action", "qty", "hold_min",
             "pnl", "model_proba", "blocked"]
        ]
        # Colour-by-state filter
        f1, f2, f3 = st.columns(3)
        with f1:
            view = st.radio(
                "Filter", ["All", "Blocked only", "Kept only"],
                horizontal=True, key="ml_zoo_filter",
            )
        with f2:
            lane_filter = st.multiselect(
                "Lane", sorted(sub["lane"].unique()),
                default=sorted(sub["lane"].unique()), key="ml_zoo_lane",
            )
        with f3:
            sym_filter = st.multiselect(
                "Symbol", sorted(sub["symbol"].unique()),
                default=sorted(sub["symbol"].unique()), key="ml_zoo_sym",
            )
        if view == "Blocked only":
            sub = sub[sub["blocked"]]
        elif view == "Kept only":
            sub = sub[~sub["blocked"]]
        sub = sub[sub["lane"].isin(lane_filter) & sub["symbol"].isin(sym_filter)]

        def _row_style(row):
            if row["blocked"] and row["pnl"] < 0:
                return ["background-color: #14532d40"] * len(row)  # blocked a loser — good
            if row["blocked"] and row["pnl"] > 0:
                return ["background-color: #7f1d1d40"] * len(row)  # blocked a winner — bad
            if not row["blocked"] and row["pnl"] < 0:
                return ["background-color: #1f2937"] * len(row)
            return ["background-color: #052e16"] * len(row)

        st.dataframe(
            sub.style.apply(_row_style, axis=1),
            use_container_width=True,
            hide_index=True,
            height=520,
        )

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Rows shown", len(sub))
        if not sub.empty:
            b2.metric("Σ P&L shown", f"${sub['pnl'].sum():+,.2f}")
            b3.metric("Win rate shown", f"{(sub['pnl'] > 0).mean() * 100:.1f}%")
            b4.metric("Avg hold (min)", f"{sub['hold_min'].mean():.1f}")

        st.markdown(
            "Legend: <span style='background-color:#14532d40;padding:2px 6px'>green</span> "
            "blocked-loser (good) · "
            "<span style='background-color:#7f1d1d40;padding:2px 6px'>red</span> "
            "blocked-winner (bad) · "
            "<span style='background-color:#052e16;padding:2px 6px'>dark-green</span> "
            "kept-winner · "
            "<span style='background-color:#1f2937;padding:2px 6px'>grey</span> "
            "kept-loser.",
            unsafe_allow_html=True,
        )


# =========================================================================
# Tab — System (fragment, 5s)
# =========================================================================
with tab_system:
    @st.fragment(run_every=5)
    def system_fragment() -> None:
        # CPU
        per_core = psutil.cpu_percent(interval=None, percpu=True)
        overall = sum(per_core) / max(len(per_core), 1)
        vm = psutil.virtual_memory()
        try:
            load1, load5, load15 = os.getloadavg()
        except OSError:
            load1 = load5 = load15 = 0.0
        try:
            du = psutil.disk_usage("/")
        except Exception:
            du = None
        boot_ts = psutil.boot_time()
        up_secs = time.time() - boot_ts

        # Top row stats
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CPU %", f"{overall:.1f}")
        c2.metric("RAM %", f"{vm.percent:.1f}", f"{human_bytes(vm.used)} / {human_bytes(vm.total)}")
        c3.metric("Load 1m / 5m", f"{load1:.2f} / {load5:.2f}")
        c4.metric("Uptime", human_uptime(up_secs))

        if per_core:
            core_df = pd.DataFrame({
                "core": [f"#{i}" for i in range(len(per_core))],
                "cpu_pct": per_core,
            })
            st.bar_chart(core_df.set_index("core"), height=200, width="stretch")

        d1, d2 = st.columns(2)
        if du:
            d1.metric("Disk used", human_bytes(du.used), f"{du.percent:.1f}% of {human_bytes(du.total)}")
            d1.caption(f"Free: {human_bytes(du.free)}")
        n = psutil.net_io_counters()
        d2.metric("Net total out", human_bytes(n.bytes_sent))
        d2.metric("Net total in", human_bytes(n.bytes_recv))

        st.subheader("Top processes by CPU")
        rows = []
        for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
            try:
                info = p.info
                cpu = info.get("cpu_percent") or 0.0
                if cpu <= 0:
                    continue
                rows.append({
                    "pid": info["pid"],
                    "name": (info.get("name") or "")[:40],
                    "cpu_%": round(cpu, 1),
                    "mem_%": round(info.get("memory_percent") or 0.0, 2),
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if rows:
            df = pd.DataFrame(rows).sort_values(["cpu_%", "mem_%"], ascending=False).head(12).reset_index(drop=True)
            st.dataframe(df, width="stretch", hide_index=True, height=440)
        else:
            st.caption("No CPU-active processes captured (counters still priming).")
        st.caption(f"Refreshed {datetime.now().strftime('%H:%M:%S')} • auto-refresh 5s")

    system_fragment()


# =========================================================================
# Tab — Mac Power & Thermals (fragment, 30s — these calls are slow)
# =========================================================================
with tab_power:
    @st.fragment(run_every=30)
    def power_fragment() -> None:
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Battery / Power")
            rc, out, err = safe_run(["pmset", "-g", "batt"])
            if rc == 0 and "Battery" in out:
                st.code(out.strip())
            elif rc == 0:
                st.caption("No internal battery (desktop Mac mini).")
                st.code(out.strip() or "(empty)")
            else:
                st.warning(f"pmset -g batt: {err.strip() or rc}")

            rc, out, err = safe_run(["pmset", "-g", "ps"])
            if rc == 0:
                st.code(out.strip() or "(empty)")

        with col_b:
            st.subheader("Memory pressure")
            rc, out, err = safe_run(["top", "-l", "1", "-n", "0"], timeout=6)
            if rc == 0:
                keep = []
                for line in out.splitlines():
                    if not line.strip():
                        if keep:
                            break
                        continue
                    keep.append(line)
                    if len(keep) >= 10:
                        break
                st.code("\n".join(keep) or out[:600])
            else:
                st.warning(f"top: {err.strip() or rc}")

            rc, out, err = safe_run(["vm_stat"])
            if rc == 0:
                st.code(out.strip())

        st.caption(f"Refreshed {datetime.now().strftime('%H:%M:%S')} • auto-refresh 30s")

    power_fragment()


# =========================================================================
# Tab — Tmux & Logs (fragment, 10s)
# =========================================================================
with tab_tmux:
    @st.fragment(run_every=10)
    def tmux_fragment() -> None:
        st.subheader("Tmux sessions")
        rc, out, err = safe_run(["tmux", "ls"])
        if rc == 0:
            st.code(out.strip() or "(no sessions)")
        else:
            msg = err.strip() or out.strip()
            st.info("No tmux server running." if "no server" in msg.lower() or not msg else msg)

        st.subheader("Live log tails")
        for name in KNOWN_LOGS:
            path = LOG_DIR / name
            if not path.exists():
                continue
            try:
                stat = path.stat()
                mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%H:%M:%S")
                age = human_uptime(time.time() - stat.st_mtime)
                with st.expander(f"{name} • {human_bytes(stat.st_size)} • {age} ago ({mtime})", expanded=False):
                    st.code(read_tail(path, n=30) or "(empty)")
            except Exception as exc:
                st.warning(f"{name}: {exc}")
        st.caption(f"Refreshed {datetime.now().strftime('%H:%M:%S')} • auto-refresh 10s")

    tmux_fragment()


# =========================================================================
# Tab — News (fragment, 30s)
# =========================================================================
with tab_news:
    @st.fragment(run_every=30)
    def news_fragment() -> None:
        today = datetime.now().strftime("%Y-%m-%d")
        if not NEWS_DIR.exists():
            st.info("data/news/ not present yet.")
            return
        files = sorted(NEWS_DIR.glob(f"*/{today}/*.jsonl"))
        if not files:
            st.info(f"No news files for {today} yet. Scrapers may still be warming up.")
            return

        items: list[dict[str, Any]] = []
        per_source_counts: dict[str, int] = {}
        for f in files:
            source = f.relative_to(NEWS_DIR).parts[0]
            per_source_counts.setdefault(source, 0)
            try:
                with f.open("r", encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        obj.setdefault("source", source)
                        items.append(obj)
                        per_source_counts[source] += 1
            except Exception as exc:
                st.warning(f"{f.name}: {exc}")

        if not items:
            st.info("Files exist but no parseable items yet.")
            return

        # Source breakdown
        src_cols = st.columns(len(per_source_counts) or 1)
        for i, (src, n) in enumerate(sorted(per_source_counts.items())):
            with src_cols[i]:
                st.metric(src, n)

        def _ts(obj: dict[str, Any]) -> float:
            for k in ("timestamp", "ts", "time", "created_at", "datetime", "published_at", "providerPublishTime"):
                v = obj.get(k)
                if isinstance(v, (int, float)):
                    return float(v)
                if isinstance(v, str):
                    try:
                        return datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()
                    except ValueError:
                        try:
                            return float(v)
                        except ValueError:
                            continue
            return 0.0

        items.sort(key=_ts, reverse=True)
        items = items[:60]
        rows = []
        for obj in items:
            ts = _ts(obj)
            t_str = datetime.fromtimestamp(ts).strftime("%H:%M:%S") if ts else "?"
            rows.append({
                "time": t_str,
                "src": obj.get("source", ""),
                "ticker": obj.get("ticker") or obj.get("symbol") or "",
                "text": (obj.get("headline") or obj.get("title") or obj.get("text") or obj.get("body") or "")[:200],
                "sent": obj.get("sentiment") or "",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=600)
        st.caption(f"Refreshed {datetime.now().strftime('%H:%M:%S')} • auto-refresh 30s")

    news_fragment()
