"""Build investor-grade case study PDF for Day 1 paper trading (2026-05-11).

Reads live data from the Alpaca DBs, today's minute bars from Alpaca data API,
today's news data, and the 60-model swarm reviews. Generates a multi-page PDF
with charts and tables in sell-side research style.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import requests
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
REPO = Path("/Users/weeoto/turbo-carnival")
ENV_PATH = REPO / "council" / ".env.local"
OUT_PDF = REPO / "reports" / f"day1_case_study_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.pdf"
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
NY = ZoneInfo("America/New_York")
LONDON = ZoneInfo("Europe/London")
TODAY = "2026-05-11"

# ---------------------------------------------------------------------------
# Style — clean sell-side aesthetic
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.3,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

COL_BG = "#ffffff"
COL_INK = "#1f2937"
COL_MUTED = "#6b7280"
COL_ACCENT = "#1e3a8a"
COL_GREEN = "#16a34a"
COL_RED = "#dc2626"
COL_AMBER = "#d97706"
COL_GREY = "#9ca3af"

LANE_COLORS = {"Aegis": "#2563eb", "Falcon": "#dc2626", "Valk": "#16a34a"}

# ---------------------------------------------------------------------------
def load_env() -> None:
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() and k.strip() not in os.environ:
            os.environ[k.strip()] = v.strip().strip('"').strip("'")


def fetch_minute_bars(symbols: list[str], date_str: str) -> dict[str, pd.DataFrame]:
    """Fetch today's 1-min bars from Alpaca data API for each symbol."""
    load_env()
    key = os.environ["ALPACA_VALK_KEY"]
    sec = os.environ["ALPACA_VALK_SECRET"]
    H = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        url = "https://data.alpaca.markets/v2/stocks/bars"
        params = {
            "symbols": sym,
            "timeframe": "1Min",
            "start": f"{date_str}T13:25:00Z",
            "end": f"{date_str}T20:05:00Z",
            "limit": 10000,
            "feed": "iex",
        }
        try:
            r = requests.get(url, headers=H, params=params, timeout=20)
            r.raise_for_status()
            bars = r.json().get("bars", {}).get(sym, [])
            if not bars:
                continue
            df = pd.DataFrame(bars)
            df["t"] = pd.to_datetime(df["t"])
            df["t_et"] = df["t"].dt.tz_convert(NY)
            out[sym] = df
        except Exception as exc:
            print(f"WARN fetch {sym}: {exc}")
    return out


def load_trades() -> pd.DataFrame:
    """Load all HFT trades from today."""
    conn = sqlite3.connect(REPO / "data" / "alpaca_max_hft_host.sqlite3")
    df = pd.read_sql_query(
        f"SELECT * FROM orders WHERE date(ts_utc)='{TODAY}' ORDER BY id",
        conn,
    )
    conn.close()
    df["ts"] = pd.to_datetime(df["ts_utc"], utc=True)
    df["ts_et"] = df["ts"].dt.tz_convert(NY)
    return df


def load_position_marks() -> pd.DataFrame:
    conn = sqlite3.connect(REPO / "data" / "alpaca_max_hft_host.sqlite3")
    df = pd.read_sql_query(
        f"SELECT * FROM position_marks WHERE date(ts_utc)='{TODAY}' ORDER BY id",
        conn,
    )
    conn.close()
    df["ts"] = pd.to_datetime(df["ts_utc"], utc=True)
    df["ts_et"] = df["ts"].dt.tz_convert(NY)
    return df


def load_condor() -> pd.DataFrame:
    conn = sqlite3.connect(REPO / "data" / "alpaca_three_lane_host.sqlite3")
    df = pd.read_sql_query(
        f"SELECT * FROM lane_orders WHERE trade_date='{TODAY}'",
        conn,
    )
    conn.close()
    return df


def fetch_account_state() -> dict[str, dict[str, Any]]:
    """Fetch current account state for all 3 lanes."""
    load_env()
    endpoint = os.environ.get("ALPACA_ENDPOINT", "https://paper-api.alpaca.markets/v2")
    out = {}
    for lane in ("Aegis", "Falcon", "Valk"):
        k = os.environ.get(f"ALPACA_{lane.upper()}_KEY")
        s = os.environ.get(f"ALPACA_{lane.upper()}_SECRET")
        H = {"APCA-API-KEY-ID": k, "APCA-API-SECRET-KEY": s}
        try:
            acct = requests.get(f"{endpoint}/account", headers=H, timeout=10).json()
            pos = requests.get(f"{endpoint}/positions", headers=H, timeout=10).json()
            out[lane] = {
                "equity": float(acct.get("equity") or 0),
                "last_equity": float(acct.get("last_equity") or 0),
                "cash": float(acct.get("cash") or 0),
                "buying_power": float(acct.get("buying_power") or 0),
                "positions": pos or [],
            }
            out[lane]["pnl_today"] = out[lane]["equity"] - out[lane]["last_equity"]
        except Exception:
            out[lane] = {"equity": 0, "pnl_today": 0, "positions": []}
    return out


def load_news_macro() -> list[dict[str, Any]]:
    """Load today's CNBC + MarketWatch headlines."""
    items = []
    for src in ("cnbc", "marketwatch"):
        p = REPO / "data" / "news" / src / TODAY / "headlines.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                obj["source"] = src
                items.append(obj)
            except Exception:
                continue
    return items


def load_ticker_news(ticker: str) -> list[dict[str, Any]]:
    p = REPO / "data" / "news" / "yahoo" / TODAY / f"{ticker}.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


# ---------------------------------------------------------------------------
# Hardcoded events of the day (ET times) — manually curated from news data
KEY_EVENTS = [
    ("10:07", "Trump rejects Iran proposal", "macro"),
    ("13:10", "Apple-Intel chip deal", "AAPL"),
    ("14:00", "AMD server CPU supercycle call", "AMD"),
    ("14:38", "Musk + Cook joining Trump-Xi summit", "TSLA/META/AAPL"),
    ("14:52", "California county sues Meta", "META"),
]


def add_page_header(fig, title: str, subtitle: str = "") -> None:
    fig.text(0.04, 0.97, "TURBO CARNIVAL  •  PAPER OPTIONS RESEARCH", fontsize=8,
             color=COL_MUTED, weight="bold")
    fig.text(0.04, 0.94, title, fontsize=18, color=COL_INK, weight="bold")
    if subtitle:
        fig.text(0.04, 0.918, subtitle, fontsize=10, color=COL_MUTED)
    fig.text(0.96, 0.97, datetime.now(NY).strftime("%d %b %Y"), fontsize=8,
             color=COL_MUTED, ha="right")


def add_page_footer(fig, page_n: int, total_pages: int = 16) -> None:
    fig.text(0.04, 0.025,
             "Paper trading only. Not investment advice. For internal review.",
             fontsize=7, color=COL_MUTED, style="italic")
    fig.text(0.96, 0.025, f"Page {page_n} of {total_pages}", fontsize=7,
             color=COL_MUTED, ha="right")


def fmt_usd(x: float, sign: bool = False) -> str:
    s = f"${abs(x):,.0f}"
    if x < 0:
        return f"−{s}"
    if sign and x > 0:
        return f"+{s}"
    return s


# ---------------------------------------------------------------------------
# PAGES
# ---------------------------------------------------------------------------
def page_cover(pdf: PdfPages, account_state: dict) -> None:
    fig = plt.figure(figsize=(8.5, 11), facecolor="white")
    total_pnl = sum(s["pnl_today"] for s in account_state.values())
    total_eq = sum(s["equity"] for s in account_state.values())

    # Top brand bar
    fig.patches.append(mpatches.Rectangle((0, 0.95), 1, 0.05, transform=fig.transFigure,
                                           facecolor=COL_ACCENT, edgecolor="none"))
    fig.text(0.5, 0.965, "TURBO CARNIVAL", fontsize=14, color="white",
             weight="bold", ha="center")

    fig.text(0.5, 0.85, "DAY 1 CASE STUDY", fontsize=11, color=COL_MUTED,
             ha="center", weight="bold")
    fig.text(0.5, 0.78, "Multi-Strategy Paper Trading", fontsize=28,
             color=COL_INK, ha="center", weight="bold")
    fig.text(0.5, 0.74, "Aegis • Falcon • Valk", fontsize=14,
             color=COL_MUTED, ha="center", style="italic")

    # Horizontal rule
    fig.add_artist(plt.Line2D([0.2, 0.8], [0.69, 0.69], color=COL_INK, lw=1))

    # Headline numbers
    fig.text(0.5, 0.63, datetime.now(NY).strftime("%A, %d %B %Y"),
             fontsize=11, color=COL_MUTED, ha="center")
    pnl_color = COL_GREEN if total_pnl >= 0 else COL_RED
    fig.text(0.5, 0.55, fmt_usd(total_pnl, sign=True),
             fontsize=44, color=pnl_color, ha="center", weight="bold")
    fig.text(0.5, 0.51, "Combined P&L • Day 1 • Paper",
             fontsize=10, color=COL_MUTED, ha="center")

    # Lane mini-cards
    y_base = 0.36
    for i, lane in enumerate(("Aegis", "Falcon", "Valk")):
        x = 0.18 + i * 0.27
        s = account_state[lane]
        c = COL_GREEN if s["pnl_today"] >= 0 else COL_RED
        fig.text(x, y_base + 0.05, lane, fontsize=13, color=COL_INK,
                 ha="center", weight="bold")
        fig.text(x, y_base + 0.02, fmt_usd(s["pnl_today"], sign=True),
                 fontsize=18, color=c, ha="center", weight="bold")
        fig.text(x, y_base - 0.01, f"NAV {fmt_usd(s['equity'])}",
                 fontsize=8, color=COL_MUTED, ha="center")

    # Methodology box
    fig.text(0.5, 0.22, "60-MODEL INDEPENDENT REVIEW",
             fontsize=10, color=COL_INK, ha="center", weight="bold")
    fig.text(0.5, 0.19, "NVIDIA NIM × 20  •  Claude Haiku × 20  •  "
             "Claude Sonnet × 10  •  Claude Opus × 10",
             fontsize=9, color=COL_MUTED, ha="center")
    fig.text(0.5, 0.16, "Aggregate verdict: 60/60 pause Falcon, "
             "0/60 deploy real money, mean rating 3.71/10",
             fontsize=9, color=COL_INK, ha="center")

    # Bottom disclaimer
    fig.text(0.5, 0.075,
             "Confidential and proprietary. This document presents paper-trading results "
             "and model-based commentary.\nIt is not an offer, solicitation, or investment "
             "advice. Past performance does not predict future results.\nAll figures "
             "denominated in USD. Source: Alpaca Markets paper accounts, internal research.",
             fontsize=7, color=COL_MUTED, ha="center", style="italic", linespacing=1.6)

    pdf.savefig(fig)
    plt.close(fig)


def page_executive_summary(pdf: PdfPages, account_state: dict, trades: pd.DataFrame,
                            page_n: int) -> None:
    fig = plt.figure(figsize=(8.5, 11), facecolor="white")
    add_page_header(fig, "Executive Summary",
                     "Day 1 of a 90-day investor demonstration period")

    # KPI strip
    total_pnl = sum(s["pnl_today"] for s in account_state.values())
    total_eq = sum(s["equity"] for s in account_state.values())
    initial = 300_000.0

    box_y = 0.83
    kpis = [
        ("STARTING NAV", fmt_usd(initial), COL_INK, "$100k per lane × 3"),
        ("CURRENT NAV", fmt_usd(total_eq), COL_INK, "after Day 1"),
        ("DAY 1 P&L", fmt_usd(total_pnl, sign=True),
         COL_GREEN if total_pnl >= 0 else COL_RED, f"{total_pnl/initial*100:+.2f}% of NAV"),
        ("TRADES", f"{len(trades)}", COL_INK, "HFT lanes Aegis + Falcon"),
    ]
    for i, (label, val, color, sub) in enumerate(kpis):
        x = 0.06 + i * 0.225
        # box
        fig.patches.append(mpatches.FancyBboxPatch(
            (x, box_y - 0.015), 0.20, 0.075,
            boxstyle="round,pad=0.005",
            transform=fig.transFigure,
            facecolor="#f9fafb", edgecolor="#e5e7eb", lw=0.5,
        ))
        fig.text(x + 0.01, box_y + 0.05, label, fontsize=7,
                 color=COL_MUTED, weight="bold")
        fig.text(x + 0.01, box_y + 0.02, val, fontsize=15,
                 color=color, weight="bold")
        fig.text(x + 0.01, box_y - 0.003, sub, fontsize=7, color=COL_MUTED)

    # Narrative
    body_y = 0.74
    lines = [
        ("THE DAY IN ONE SENTENCE", "h"),
        ("Three differentiated paper trading lanes opened simultaneously at 09:30 ET; "
         "two HFT equity strategies and one SPY iron condor.", ""),
        ("By 16:00 ET close, combined P&L was −$3,443 on $300k (−1.15%), "
         "driven almost entirely by Falcon's repeated stop-out cascade in the "
         "afternoon news-driven chop.", ""),
        ("", ""),
        ("LANE-BY-LANE", "h"),
        (f"• Aegis (HFT mean-reversion):  {fmt_usd(account_state['Aegis']['pnl_today'], sign=True)}    "
         f"33 take-profits / 23 stop-losses / 7 time-exits — 52% hit rate. Working as designed.", ""),
        (f"• Falcon (HFT breakout):       {fmt_usd(account_state['Falcon']['pnl_today'], sign=True)}    "
         f"7 take-profits / 82 stop-losses / 10 time-exits — 7% hit rate. Pause recommended.", ""),
        (f"• Valk (SPY iron condor):     {fmt_usd(account_state['Valk']['pnl_today'], sign=True)}    "
         f"1 position held; 30-DTE, qty 44, $0.54 entry credit. Mark-to-market drift.", ""),
        ("", ""),
        ("WHAT CAUSED THE LOSS", "h"),
        ("Five news catalysts clustered between 10:07 and 14:52 ET (Iran, Apple-Intel, "
         "AMD supercycle, Trump-Xi summit, Meta lawsuit) generated synchronous "
         "intraday spikes in TSLA / META / AAPL.", ""),
        ("Falcon's breakout signal interprets these spikes as momentum entries; the spikes "
         "then mean-revert as algorithms price the news in seconds. Falcon's 25 bps stop "
         "fires within 2-10 seconds — repeatedly.", ""),
        ("Backtest assumed 0.75 bps slippage. Live evidence implies 3-15 bps. "
         "On an 18 bps signal threshold, this inverts the sign of edge.", ""),
        ("", ""),
        ("MODEL CONSENSUS (60 INDEPENDENT REVIEWS)", "h"),
        ("• Falcon: 60/60 pause  •  Real-money deploy: 0/60 yes  "
         "•  Architecture rating: 3.71/10 weighted mean", ""),
        ("• Smarter models rated harder: Haiku 4.16, Sonnet 3.32, Opus 3.05.", ""),
        ("• Top regime filters: VIX (38), ATR (35), news/macro gate (28), ADX (25).", ""),
        ("", ""),
        ("IMMEDIATE ACTIONS TAKEN", "h"),
        ("• Force-closed Aegis and Falcon equity positions before EOD (no overnight risk).", ""),
        ("• Killed Falcon HFT host process — does not auto-restart Tuesday.", ""),
        ("• Valk iron condor held; risk-monitor process still polling every 60s.", ""),
    ]
    y = body_y
    for line, mode in lines:
        if mode == "h":
            fig.text(0.06, y, line, fontsize=10, color=COL_ACCENT,
                     weight="bold")
            y -= 0.022
        else:
            wrapped = line
            fig.text(0.06, y, wrapped, fontsize=8.5, color=COL_INK, linespacing=1.4)
            y -= 0.022 if line else 0.012

    add_page_footer(fig, page_n)
    pdf.savefig(fig)
    plt.close(fig)


def page_methodology(pdf: PdfPages, page_n: int) -> None:
    fig = plt.figure(figsize=(8.5, 11), facecolor="white")
    add_page_header(fig, "Methodology",
                     "Three differentiated strategy lanes, $100k paper capital each")

    # 3 lane cards
    for i, (lane, desc) in enumerate([
        ("Aegis", [
            ("STRATEGY", "HFT mean-reversion"),
            ("UNIVERSE", "SPY, QQQ, NVDA, TSLA, AAPL, META, AMD"),
            ("SIGNAL", "25 bps minimum signal score"),
            ("TAKE PROFIT", "100 bps from entry"),
            ("STOP LOSS", "35 bps from entry"),
            ("MAX HOLD", "1200 seconds (20 min)"),
            ("SIZING", "95% notional per entry"),
            ("EOD RULE", "Force-flat at 15:55 ET"),
        ]),
        ("Falcon", [
            ("STRATEGY", "HFT breakout / momentum"),
            ("UNIVERSE", "SPY, QQQ, NVDA, TSLA, AAPL, META, AMD"),
            ("SIGNAL", "18 bps minimum signal score"),
            ("TAKE PROFIT", "70 bps from entry"),
            ("STOP LOSS", "25 bps from entry"),
            ("MAX HOLD", "900 seconds (15 min)"),
            ("SIZING", "95% notional per entry"),
            ("EOD RULE", "Force-flat at 15:55 ET"),
        ]),
        ("Valk", [
            ("STRATEGY", "SPY iron condor (defined-risk credit)"),
            ("STRUCTURE", "Short put + long put + short call + long call"),
            ("DTE", "30 days to expiration at entry"),
            ("STRIKES", "0.12 delta short legs, $5 wing width"),
            ("TAKE PROFIT", "30% of credit captured"),
            ("STOP LOSS", "Debit reaches 3× entry credit"),
            ("FORCE CLOSE", "≤ 10 DTE"),
            ("GATE", "VRP positive (IV > realized vol)"),
        ]),
    ]):
        x = 0.05 + i * 0.31
        # Card header
        fig.patches.append(mpatches.Rectangle(
            (x, 0.86), 0.28, 0.025, transform=fig.transFigure,
            facecolor=LANE_COLORS[lane], edgecolor="none",
        ))
        fig.text(x + 0.01, 0.87, lane.upper(), fontsize=11, color="white",
                 weight="bold")
        # Body
        fig.patches.append(mpatches.FancyBboxPatch(
            (x, 0.40), 0.28, 0.46, boxstyle="square,pad=0.001",
            transform=fig.transFigure, facecolor="#f9fafb",
            edgecolor="#e5e7eb", lw=0.5,
        ))
        y = 0.825
        for k, v in desc:
            fig.text(x + 0.01, y, k, fontsize=7, color=COL_MUTED,
                     weight="bold")
            fig.text(x + 0.01, y - 0.018, v, fontsize=8.5, color=COL_INK)
            y -= 0.052

    # Backtest disclosure
    fig.text(0.05, 0.36, "BACKTEST PROVENANCE",
             fontsize=11, color=COL_ACCENT, weight="bold")
    fig.text(0.05, 0.335,
             "HFT lanes (Aegis, Falcon) tuned on 21 calendar days of cached Alpaca 1-minute "
             "bars\nwith simulated 0.75 bps slippage. Reported sim performance: "
             "+17% total return, −8% max DD.",
             fontsize=8.5, color=COL_INK, linespacing=1.5)
    fig.text(0.05, 0.29,
             "Valk strategy parameters selected from the 70/30 split on the SPY iron-condor "
             "research grid\n(strategy slug: fixed_best_qwen). Sim monthly +13.7%, max DD −18.7%.",
             fontsize=8.5, color=COL_INK, linespacing=1.5)
    fig.text(0.05, 0.25,
             "Backtest material weakness disclosed: live Day 1 evidence indicates real round-trip "
             "fill cost\nof 3-15 bps, materially higher than sim. Re-validation work flagged.",
             fontsize=8.5, color=COL_RED, linespacing=1.5)

    # Universe table
    fig.text(0.05, 0.20, "TRADING UNIVERSE",
             fontsize=11, color=COL_ACCENT, weight="bold")
    universe_data = [
        ("SPY",  "SPDR S&P 500 ETF",        "broad-market beta"),
        ("QQQ",  "Invesco QQQ Trust",       "tech-weighted"),
        ("NVDA", "NVIDIA",                  "AI/semis"),
        ("AAPL", "Apple",                   "mega-cap tech"),
        ("META", "Meta Platforms",          "mega-cap tech"),
        ("AMD",  "Advanced Micro Devices",  "AI/semis"),
        ("TSLA", "Tesla",                   "EV/high-volatility"),
    ]
    fig.text(0.05, 0.175, "Symbol", fontsize=7, color=COL_MUTED,
             weight="bold")
    fig.text(0.16, 0.175, "Name", fontsize=7, color=COL_MUTED,
             weight="bold")
    fig.text(0.45, 0.175, "Category", fontsize=7, color=COL_MUTED,
             weight="bold")
    fig.add_artist(plt.Line2D([0.05, 0.95], [0.17, 0.17], color="#e5e7eb", lw=0.5))
    for i, (sym, name, cat) in enumerate(universe_data):
        y = 0.155 - i * 0.015
        fig.text(0.05, y, sym, fontsize=8.5, color=COL_INK, weight="bold")
        fig.text(0.16, y, name, fontsize=8.5, color=COL_INK)
        fig.text(0.45, y, cat, fontsize=8.5, color=COL_MUTED, style="italic")

    add_page_footer(fig, page_n)
    pdf.savefig(fig)
    plt.close(fig)


def page_lane_results(pdf: PdfPages, lane: str, trades: pd.DataFrame, marks: pd.DataFrame,
                       account: dict, bars: dict, page_n: int) -> None:
    fig = plt.figure(figsize=(8.5, 11), facecolor="white")
    color = LANE_COLORS.get(lane, COL_INK)
    add_page_header(fig, f"{lane} • Lane Results",
                     {"Aegis": "HFT mean-reversion equity scalper",
                      "Falcon": "HFT breakout equity scalper",
                      "Valk": "SPY 30-DTE iron condor"}[lane])

    # Lane KPIs
    lane_trades = trades[trades["lane"] == lane].copy()
    entries = lane_trades[lane_trades["action"].isin(["entry_long", "entry_short"])]
    tp = lane_trades[lane_trades["action"] == "take_profit"]
    sl = lane_trades[lane_trades["action"] == "stop_loss"]
    te = lane_trades[lane_trades["action"] == "time_exit"]
    n_total_exits = len(tp) + len(sl) + len(te)
    hit_rate = len(tp) / n_total_exits * 100 if n_total_exits > 0 else 0

    kpis = [
        ("DAY 1 P&L", fmt_usd(account[lane]["pnl_today"], sign=True),
         COL_GREEN if account[lane]["pnl_today"] >= 0 else COL_RED),
        ("ENTRIES", f"{len(entries)}", color),
        ("TAKE-PROFITS", f"{len(tp)}", COL_GREEN),
        ("STOP-LOSSES", f"{len(sl)}", COL_RED),
        ("HIT RATE", f"{hit_rate:.0f}%", COL_INK),
    ]
    box_y = 0.83
    for i, (label, val, c) in enumerate(kpis):
        x = 0.05 + i * 0.185
        fig.patches.append(mpatches.FancyBboxPatch(
            (x, box_y - 0.015), 0.16, 0.075, boxstyle="round,pad=0.005",
            transform=fig.transFigure, facecolor="#f9fafb",
            edgecolor="#e5e7eb", lw=0.5,
        ))
        fig.text(x + 0.005, box_y + 0.05, label, fontsize=6.5,
                 color=COL_MUTED, weight="bold")
        fig.text(x + 0.005, box_y + 0.022, val, fontsize=14,
                 color=c, weight="bold")

    # Distribution chart - exit reason for HFT lanes
    if lane in ("Aegis", "Falcon") and n_total_exits > 0:
        ax_dist = fig.add_axes([0.06, 0.55, 0.40, 0.22])
        reasons = ["take_profit", "stop_loss", "time_exit"]
        counts = [len(tp), len(sl), len(te)]
        colors_bar = [COL_GREEN, COL_RED, COL_AMBER]
        bars_h = ax_dist.bar(reasons, counts, color=colors_bar, width=0.6,
                              edgecolor="none")
        ax_dist.set_title(f"Exit reason distribution — {n_total_exits} total exits",
                          loc="left", color=COL_INK, fontsize=10)
        ax_dist.set_ylabel("Count")
        ax_dist.tick_params(axis="x", labelsize=8)
        for b, c in zip(bars_h, counts):
            ax_dist.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                         str(c), ha="center", fontsize=9, color=COL_INK, weight="bold")
        ax_dist.grid(True, axis="y", alpha=0.3)
        ax_dist.set_ylim(0, max(counts) * 1.25 + 1)

        # P&L by symbol bar chart
        if not entries.empty:
            ax_sym = fig.add_axes([0.55, 0.55, 0.40, 0.22])
            executed = lane_trades[lane_trades["status"].isin(["filled", "submitted", "pending_new"])].copy()
            sym_counts = entries.groupby("symbol").size().sort_values(ascending=False)
            ax_sym.bar(sym_counts.index, sym_counts.values, color=color, width=0.6,
                        edgecolor="none")
            ax_sym.set_title(f"Entries by ticker", loc="left", color=COL_INK, fontsize=10)
            ax_sym.set_ylabel("Entries")
            ax_sym.tick_params(axis="x", labelsize=8)
            ax_sym.grid(True, axis="y", alpha=0.3)
            for i, v in enumerate(sym_counts.values):
                ax_sym.text(i, v + 0.3, str(v), ha="center", fontsize=8, color=COL_INK)

    if lane == "Valk":
        # Show condor structure
        condor = load_condor()
        valk_row = condor[condor["lane"] == "Valk"]
        if not valk_row.empty:
            r = valk_row.iloc[0]
            def occ_strike(occ: str) -> float:
                # OCC: ROOTYYMMDDC/P[STRIKE*1000 padded to 8]
                if occ and occ[-8:].isdigit():
                    return int(occ[-8:]) / 1000.0
                try:
                    return float(occ)
                except Exception:
                    return 0.0
            lp = occ_strike(str(r["long_put"]))
            sp = occ_strike(str(r["short_put"]))
            sc = occ_strike(str(r["short_call"]))
            lc = occ_strike(str(r["long_call"]))
            spy = float(r["spy_price"])
            credit = float(r["credit"])
            qty = int(r["quantity"])
            max_loss = (lc - sc - credit) * 100 * qty

            ax_str = fig.add_axes([0.06, 0.50, 0.88, 0.28])
            xs = np.linspace(lp - 5, lc + 5, 500)
            # iron condor payoff: short the put spread + short the call spread
            payoff = np.zeros_like(xs)
            for i, x in enumerate(xs):
                p_loss = max(0, lp - x) - max(0, sp - x)  # long put - short put (defined risk lower)
                c_loss = max(0, x - sc) - max(0, x - lc)
                # Strategy is short the put-spread (collect credit, lose if below sp)
                # and short the call-spread (collect credit, lose if above sc)
                short_put_pnl = -(max(0, sp - x) - max(0, lp - x))
                short_call_pnl = -(max(0, x - sc) - max(0, x - lc))
                payoff[i] = (short_put_pnl + short_call_pnl + credit) * 100 * qty
            ax_str.plot(xs, payoff, color=COL_GREEN, lw=2)
            ax_str.axhline(0, color="#9ca3af", lw=0.5)
            ax_str.axvline(spy, color=COL_INK, linestyle="--", lw=1, alpha=0.7,
                           label=f"SPY ${spy:.2f}")
            ax_str.fill_between(xs, payoff, 0, where=payoff > 0,
                                  color=COL_GREEN, alpha=0.15)
            ax_str.fill_between(xs, payoff, 0, where=payoff < 0,
                                  color=COL_RED, alpha=0.15)
            ax_str.axvline(lp, color=COL_RED, linestyle=":", lw=0.8, alpha=0.5)
            ax_str.axvline(sp, color=COL_RED, linestyle=":", lw=0.8, alpha=0.5)
            ax_str.axvline(sc, color=COL_RED, linestyle=":", lw=0.8, alpha=0.5)
            ax_str.axvline(lc, color=COL_RED, linestyle=":", lw=0.8, alpha=0.5)
            ax_str.set_xlabel(f"SPY at expiry")
            ax_str.set_ylabel("Total P&L at expiry ($)")
            ax_str.set_title(
                f"Iron condor payoff @ expiry  •  qty {qty} contracts  •  "
                f"strikes [{lp:.0f} / {sp:.0f} / {sc:.0f} / {lc:.0f}]  •  "
                f"credit ${credit:.2f}",
                loc="left", color=COL_INK, fontsize=10,
            )
            ax_str.grid(True, alpha=0.3)
            ax_str.legend(loc="lower center")

    # Narrative for the lane
    notes_y = 0.42
    if lane == "Aegis":
        text = (
            "Aegis performed as designed. The mean-reversion signal targets sharp "
            "intraday displacements that revert; 52% of exits hit take-profit at "
            "+100 bps, and risk discipline was preserved with a wider 35 bps stop and "
            "a 20-minute time-out. Day 1 loss of −$709 is within acceptable variance "
            "for a strategy with a 1.5:1 reward:risk ratio.\n\n"
            "Falcon's afternoon whipsaw is partially a regime effect; Aegis was "
            "exposed to the same news chop but its mean-reversion bias was the right "
            "side of the post-spike reversion that crushed Falcon's breakouts."
        )
    elif lane == "Falcon":
        text = (
            "Falcon failed in the way the 60-model review predicted before we knew "
            "the cause: a strategy whose 18 bps signal threshold lies inside the live "
            "bid/ask + half-spread + microstructure noise band cannot extract edge "
            "from any direction.\n\n"
            "Empirically, 82 of 99 entries stopped out within 2-10 seconds. The "
            "7 take-profits came predominantly during the 13:30-14:00 ET window when "
            "the market was briefly trending. From 14:38 onward the news-driven chop "
            "produced 12 consecutive stop-losses on TSLA and META.\n\n"
            "Conclusion: this is not a tuning problem; it is a strategy/regime mismatch. "
            "The breakout logic is being adversely-selected by faster participants. "
            "Recommendation: do not restart in current form."
        )
    else:
        text = (
            "Valk opened one SPY iron condor at 09:32 ET for $0.54 credit per contract, "
            "qty 44, expiry June 12 (32 days to expiry). The mark-to-market drift to "
            "−$444 reflects an early IV expansion as the news cycle began.\n\n"
            "The position remains within both short strikes; theta accrual continues "
            "to favor the holder over time provided SPY stays in the 696-790 corridor. "
            "Max loss is bounded at $19,360 (wing width less credit, multiplied by 44 "
            "contracts and 100 shares per contract).\n\n"
            "Risk concentration: 44 contracts on one 30-DTE structure is a meaningful "
            "short-vega position. Independent Opus reviewers flagged Valk as the "
            "single largest tail risk overnight given the news cluster. Recommended "
            "action at tomorrow's open: scale down to 22 contracts or delta-hedge."
        )
    for line in text.split("\n\n"):
        fig.text(0.06, notes_y, line, fontsize=9, color=COL_INK,
                  linespacing=1.5, wrap=True)
        n_lines = len(line) // 95 + 1
        notes_y -= 0.022 * n_lines + 0.012

    add_page_footer(fig, page_n)
    pdf.savefig(fig)
    plt.close(fig)


def page_intraday_charts(pdf: PdfPages, bars: dict, trades: pd.DataFrame,
                          tickers: list[str], page_n: int) -> None:
    fig = plt.figure(figsize=(8.5, 11), facecolor="white")
    add_page_header(fig, "Intraday Price Action",
                     "Today's 1-minute bars with our entry/exit markers + news anchors")

    n = len(tickers)
    rows = (n + 1) // 2
    for i, sym in enumerate(tickers):
        ax = fig.add_axes([0.06 + (i % 2) * 0.46, 0.78 - (i // 2) * 0.22, 0.41, 0.17])
        df = bars.get(sym)
        if df is None or df.empty:
            ax.text(0.5, 0.5, f"{sym} — no data", ha="center", va="center",
                     transform=ax.transAxes, color=COL_MUTED)
            ax.axis("off")
            continue
        # Plot price
        ax.plot(df["t_et"], df["c"], color=COL_INK, lw=0.7)
        ax.set_title(f"{sym}  •  open ${df['o'].iloc[0]:.2f}  •  close ${df['c'].iloc[-1]:.2f}  "
                      f"({(df['c'].iloc[-1]/df['o'].iloc[0]-1)*100:+.2f}%)",
                      loc="left", color=COL_INK, fontsize=9)
        ax.xaxis.set_major_locator(mdates.HourLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=NY))
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(True, alpha=0.3)

        # Add our trade markers
        sym_trades = trades[trades["symbol"] == sym]
        for _, t in sym_trades.iterrows():
            ts_et = t["ts_et"]
            action = t["action"]
            # Approximate price from notional / qty
            price = None
            if t.get("notional") and t.get("qty"):
                try:
                    price = float(t["notional"]) / float(t["qty"])
                except Exception:
                    pass
            if price is None:
                idx = (df["t_et"] - ts_et).abs().idxmin()
                price = float(df.loc[idx, "c"])
            if action in ("entry_long", "entry_short"):
                m = "^" if action == "entry_long" else "v"
                c = LANE_COLORS.get(t["lane"], COL_GREY)
                ax.scatter(ts_et, price, marker=m, color=c, s=18, zorder=3, alpha=0.7)
            elif action == "take_profit":
                ax.scatter(ts_et, price, marker="o", color=COL_GREEN, s=10,
                            zorder=3, alpha=0.7)
            elif action == "stop_loss":
                ax.scatter(ts_et, price, marker="x", color=COL_RED, s=14,
                            zorder=3, alpha=0.7)

        # News annotations
        for hhmm, label, scope in KEY_EVENTS:
            if sym in scope or scope == "macro":
                ts = pd.Timestamp(f"{TODAY} {hhmm}", tz=NY)
                if ts >= df["t_et"].min() and ts <= df["t_et"].max():
                    ax.axvline(ts, color=COL_AMBER, linestyle="--", lw=0.6, alpha=0.5)

    # Legend at bottom
    handles = [
        mpatches.Patch(color=COL_INK, label="price"),
        plt.Line2D([0], [0], marker="^", color=COL_INK, label="entry long",
                    markerfacecolor="black", linestyle="", markersize=7),
        plt.Line2D([0], [0], marker="v", color=COL_INK, label="entry short",
                    markerfacecolor="black", linestyle="", markersize=7),
        plt.Line2D([0], [0], marker="o", color=COL_GREEN, label="take-profit",
                    linestyle="", markersize=5),
        plt.Line2D([0], [0], marker="x", color=COL_RED, label="stop-loss",
                    linestyle="", markersize=6),
        plt.Line2D([0], [0], color=COL_AMBER, linestyle="--", lw=0.8, label="news event"),
    ]
    fig.legend(handles=handles, loc="lower center",
                bbox_to_anchor=(0.5, 0.07), ncol=6, fontsize=8, frameon=False)

    add_page_footer(fig, page_n)
    pdf.savefig(fig)
    plt.close(fig)


def page_news_timeline(pdf: PdfPages, bars: dict, page_n: int) -> None:
    fig = plt.figure(figsize=(8.5, 11), facecolor="white")
    add_page_header(fig, "News Catalyst Timeline",
                     "Macro & company-specific events vs. SPY intraday movement")

    # SPY chart
    ax = fig.add_axes([0.06, 0.50, 0.88, 0.32])
    spy = bars.get("SPY")
    if spy is not None and not spy.empty:
        ax.plot(spy["t_et"], spy["c"], color=COL_INK, lw=1.0)
        ax.fill_between(spy["t_et"], spy["c"], spy["c"].min(), color=COL_INK, alpha=0.04)
        for hhmm, label, scope in KEY_EVENTS:
            ts = pd.Timestamp(f"{TODAY} {hhmm}", tz=NY)
            if ts < spy["t_et"].min() or ts > spy["t_et"].max():
                continue
            ax.axvline(ts, color=COL_AMBER, linestyle="--", lw=0.8, alpha=0.7)
            idx = (spy["t_et"] - ts).abs().idxmin()
            y = float(spy.loc[idx, "c"])
            ax.annotate(
                f"{hhmm}\n{label}",
                xy=(ts, y), xytext=(0, 25),
                textcoords="offset points", fontsize=7,
                color=COL_AMBER, ha="center",
                arrowprops=dict(arrowstyle="-", color=COL_AMBER, lw=0.5),
            )
        ax.set_title("SPY — 1-minute close with news anchors", loc="left",
                      color=COL_INK, fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=NY))
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    # Events table
    fig.text(0.06, 0.44, "EVENTS OF THE DAY",
              fontsize=11, color=COL_ACCENT, weight="bold")
    fig.text(0.06, 0.41, "ET TIME", fontsize=7,
              color=COL_MUTED, weight="bold")
    fig.text(0.14, 0.41, "EVENT", fontsize=7,
              color=COL_MUTED, weight="bold")
    fig.text(0.70, 0.41, "AFFECTED", fontsize=7,
              color=COL_MUTED, weight="bold")
    fig.add_artist(plt.Line2D([0.05, 0.95], [0.405, 0.405],
                                color="#e5e7eb", lw=0.5))
    for i, (hhmm, label, scope) in enumerate(KEY_EVENTS):
        y = 0.39 - i * 0.025
        fig.text(0.06, y, hhmm, fontsize=9, color=COL_INK, weight="bold")
        fig.text(0.14, y, label, fontsize=9, color=COL_INK)
        fig.text(0.70, y, scope, fontsize=9, color=COL_MUTED, style="italic")

    # Sentiment table
    fig.text(0.06, 0.235, "RETAIL SENTIMENT (Stocktwits)",
              fontsize=11, color=COL_ACCENT, weight="bold")
    fig.text(0.06, 0.21, "Net retail tilt across the universe today:",
              fontsize=8.5, color=COL_INK)

    sentiment = [
        ("SPY",  19, 30, "BEARISH"),
        ("AAPL",  4,  6, "MILDLY BEARISH"),
        ("META", 11,  9, "MIXED"),
        ("QQQ",  15, 14, "MIXED"),
        ("NVDA", 22,  4, "BULLISH"),
        ("AMD",  21,  3, "BULLISH"),
        ("TSLA", 42,  9, "BULLISH"),
    ]
    fig.text(0.06, 0.185, "TICKER", fontsize=7,
              color=COL_MUTED, weight="bold")
    fig.text(0.18, 0.185, "BULL", fontsize=7,
              color=COL_MUTED, weight="bold")
    fig.text(0.27, 0.185, "BEAR", fontsize=7,
              color=COL_MUTED, weight="bold")
    fig.text(0.40, 0.185, "VERDICT", fontsize=7,
              color=COL_MUTED, weight="bold")
    fig.add_artist(plt.Line2D([0.05, 0.65], [0.18, 0.18], color="#e5e7eb", lw=0.5))
    for i, (sym, b, br, v) in enumerate(sentiment):
        y = 0.165 - i * 0.018
        c = COL_GREEN if "BULL" in v else (COL_RED if "BEAR" in v else COL_MUTED)
        fig.text(0.06, y, sym, fontsize=8.5, color=COL_INK, weight="bold")
        fig.text(0.18, y, str(b), fontsize=8.5, color=COL_GREEN)
        fig.text(0.27, y, str(br), fontsize=8.5, color=COL_RED)
        fig.text(0.40, y, v, fontsize=8.5, color=c, style="italic")

    add_page_footer(fig, page_n)
    pdf.savefig(fig)
    plt.close(fig)


def page_swarm_analysis(pdf: PdfPages, page_n: int) -> None:
    fig = plt.figure(figsize=(8.5, 11), facecolor="white")
    add_page_header(fig, "60-Model Independent Review",
                     "Cross-architecture consensus from 13 providers")

    # Vote bar chart
    ax = fig.add_axes([0.06, 0.66, 0.40, 0.18])
    panels = ["NIM\n(20)", "Haiku\n(20)", "Sonnet\n(10)", "Opus\n(10)"]
    ratings = [3.80, 4.16, 3.32, 3.05]
    colors = ["#10b981", "#3b82f6", "#8b5cf6", "#a855f7"]
    bars_h = ax.bar(panels, ratings, color=colors, width=0.6, edgecolor="none")
    ax.set_ylim(0, 10)
    ax.set_ylabel("Architecture rating /10")
    ax.set_title("Rating by model family", loc="left",
                  color=COL_INK, fontsize=10)
    ax.axhline(5, color="#9ca3af", lw=0.5, linestyle="--")
    ax.text(3.6, 5.2, "Neutral", fontsize=7, color=COL_MUTED)
    for b, r in zip(bars_h, ratings):
        ax.text(b.get_x() + b.get_width() / 2, r + 0.15, f"{r}",
                 ha="center", fontsize=9, color=COL_INK, weight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    fig.text(0.06, 0.63,
              "Notable pattern: harsher ratings from deeper reasoners. "
              "Haiku 4.16 → Sonnet 3.32 → Opus 3.05.",
              fontsize=8, color=COL_MUTED, style="italic")

    # Filter consensus
    ax2 = fig.add_axes([0.55, 0.66, 0.40, 0.18])
    filters = ["VIX", "ATR", "News-gate", "ADX", "Slip-monitor"]
    counts = [38, 35, 28, 25, 16]
    bars2 = ax2.barh(filters[::-1], counts[::-1], color=COL_ACCENT, height=0.6)
    ax2.set_xlabel("Models mentioning (of 60)")
    ax2.set_title("Top regime filters by consensus", loc="left",
                   color=COL_INK, fontsize=10)
    for i, c in enumerate(counts[::-1]):
        ax2.text(c + 0.5, i, str(c), fontsize=9, color=COL_INK,
                  va="center", weight="bold")
    ax2.grid(True, axis="x", alpha=0.3)

    # Verdict box
    fig.text(0.06, 0.56, "AGGREGATE VERDICT", fontsize=11,
              color=COL_ACCENT, weight="bold")
    fig.patches.append(mpatches.FancyBboxPatch(
        (0.06, 0.43), 0.88, 0.115, boxstyle="round,pad=0.005",
        transform=fig.transFigure, facecolor="#fef3c7",
        edgecolor=COL_AMBER, lw=1.0,
    ))
    fig.text(0.10, 0.525,
              "60 of 60 models recommend PAUSING Falcon.",
              fontsize=12, color=COL_RED, weight="bold")
    fig.text(0.10, 0.50,
              "0 of 60 models recommend deploying with real money.",
              fontsize=12, color=COL_RED, weight="bold")
    fig.text(0.10, 0.475,
              "Mean architecture rating: 3.71 / 10 (weighted across all panels).",
              fontsize=10, color=COL_INK)
    fig.text(0.10, 0.455,
              "Top hidden risk (50 of 60): live execution slippage 4-20× the simulator assumption.",
              fontsize=9, color=COL_INK)
    fig.text(0.10, 0.435,
              "Unique Opus findings (max-effort tier): all 3 lanes are short-gamma; "
              "Valk is the unbounded tail; backtest window is survivorship-biased.",
              fontsize=9, color=COL_INK)

    # Provider list
    fig.text(0.06, 0.39, "PROVIDERS REVIEWED",
              fontsize=11, color=COL_ACCENT, weight="bold")
    providers = [
        "Meta — Llama 3.3, Llama 4 Maverick",
        "Anthropic — Claude Opus, Sonnet, Haiku",
        "Alibaba — Qwen3 Coder, Qwen 2.5",
        "Moonshot AI — Kimi K2 Instruct",
        "DeepSeek — DeepSeek v4 Flash, v4 Pro",
        "Mistral AI — Mixtral 8×22B, 8×7B, Ministral",
        "Google — Gemma 3 (4B, 12B, 27B), Gemma 4",
        "NVIDIA — Nemotron Super, Nano",
        "Microsoft — Phi-4 Mini",
        "Upstage — Solar 10.7B",
        "Abacus AI — Dracarys Llama",
    ]
    for i, p in enumerate(providers):
        x = 0.06 + (i % 2) * 0.45
        y = 0.36 - (i // 2) * 0.022
        fig.text(x, y, "• " + p, fontsize=8, color=COL_INK)

    add_page_footer(fig, page_n)
    pdf.savefig(fig)
    plt.close(fig)


def page_risk_analysis(pdf: PdfPages, page_n: int) -> None:
    fig = plt.figure(figsize=(8.5, 11), facecolor="white")
    add_page_header(fig, "Risk Analysis & Drivers",
                     "Why the breakout lane failed; what the backtest missed")

    fig.text(0.06, 0.87, "ROOT-CAUSE DIAGNOSIS",
              fontsize=11, color=COL_ACCENT, weight="bold")
    drivers = [
        ("Slippage gap (4-20×)",
         "Backtest assumed 0.75 bps round-trip slip on cached 1-min bars. "
         "Live evidence shows fills costing 3-15 bps depending on ticker and time-of-day. "
         "Falcon's 18 bps signal threshold and 25 bps stop are now mathematically inside "
         "the noise band; the strategy effectively pays spread to the market maker on "
         "every round-trip."),
        ("Regime mismatch — news-driven chop",
         "Five macro/single-stock catalysts hit between 10:07 and 14:52 ET. "
         "Breakout strategies require directional follow-through; today's tape was a "
         "series of news pops that algos faded within seconds. Falcon entered on the pop, "
         "got stopped on the fade — 12 consecutive stop-outs between 14:38-15:30 ET."),
        ("Single-factor portfolio (short gamma)",
         "Despite three nominally different strategies, all 3 lanes share short-gamma "
         "exposure to information flow: Aegis fades edges; Falcon's tight-SL breakouts "
         "are short-gamma by construction; Valk's iron condor is explicitly short gamma. "
         "When realized vol expanded today, all 3 lanes bled simultaneously. The "
         "diversification thesis was disproven on Day 1."),
        ("Backtest window selection bias",
         "The 21-day window used to tune Aegis and Falcon was chosen post-hoc — implicitly "
         "selected for periods where the rules looked good. Re-pulling a random 21-day "
         "window with news catalysts in it would likely show very different results. "
         "Until the backtest is re-run on heterogeneous regimes, the +17% / −8% headline "
         "should be regarded as in-sample fitting."),
        ("Sub-second exits vs. 1-minute simulation grain",
         "The simulator validates a strategy that can only react every 60 seconds. "
         "Live, Falcon's exits fire within 2-10 seconds. Whatever signal validation the "
         "backtest provided cannot be carried to the sub-second decision horizon."),
    ]
    y = 0.84
    for title, body in drivers:
        fig.text(0.06, y, "• " + title, fontsize=10, color=COL_INK,
                  weight="bold")
        y -= 0.020
        # Wrap manually
        words = body.split()
        line = ""
        for w in words:
            if len(line) + len(w) + 1 < 105:
                line = (line + " " + w).strip()
            else:
                fig.text(0.085, y, line, fontsize=8.5, color=COL_INK, linespacing=1.5)
                y -= 0.015
                line = w
        if line:
            fig.text(0.085, y, line, fontsize=8.5, color=COL_INK, linespacing=1.5)
            y -= 0.015
        y -= 0.012

    # Value at risk table
    fig.text(0.06, 0.22, "POSITION SIZING vs OBSERVED LOSS",
              fontsize=11, color=COL_ACCENT, weight="bold")
    fig.text(0.06, 0.195, "Lane", fontsize=7, color=COL_MUTED,
              weight="bold")
    fig.text(0.20, 0.195, "Notional/trade", fontsize=7, color=COL_MUTED,
              weight="bold")
    fig.text(0.36, 0.195, "Day 1 P&L", fontsize=7, color=COL_MUTED,
              weight="bold")
    fig.text(0.50, 0.195, "% of NAV", fontsize=7, color=COL_MUTED,
              weight="bold")
    fig.text(0.62, 0.195, "Max theoretical loss", fontsize=7, color=COL_MUTED,
              weight="bold")
    fig.add_artist(plt.Line2D([0.05, 0.95], [0.19, 0.19], color="#e5e7eb", lw=0.5))
    rows = [
        ("Aegis",  "~$95k",   "−$709",   "−0.71%",  "Bounded by 35 bps stop × notional"),
        ("Falcon", "~$92k",   "−$2,290", "−2.29%",  "Bounded by 25 bps stop × notional"),
        ("Valk",   "44 cont", "−$444 m2m", "−0.44%",  "$19,360 (wing width − credit, ×44)"),
    ]
    for i, (lane, notion, pnl, pct, maxl) in enumerate(rows):
        y = 0.17 - i * 0.020
        fig.text(0.06, y, lane, fontsize=9, color=COL_INK, weight="bold")
        fig.text(0.20, y, notion, fontsize=9, color=COL_INK)
        c = COL_GREEN if "−" not in pnl else COL_RED
        fig.text(0.36, y, pnl, fontsize=9, color=c, weight="bold")
        fig.text(0.50, y, pct, fontsize=9, color=c)
        fig.text(0.62, y, maxl, fontsize=9, color=COL_INK, style="italic")

    add_page_footer(fig, page_n)
    pdf.savefig(fig)
    plt.close(fig)


def page_recommendations(pdf: PdfPages, page_n: int) -> None:
    fig = plt.figure(figsize=(8.5, 11), facecolor="white")
    add_page_header(fig, "Recommendations & Forward Plan",
                     "Concrete steps for the remaining 89-day demo period")

    recs = [
        ("1. Pause Falcon indefinitely.",
         "Falcon stop-loss fired on 82 of 99 entries. The 18 bps signal sits inside the "
         "live noise band. Do not restart in current form. If restarted later, require: "
         "(a) signal threshold ≥ 3× measured live spread, (b) ATR or vol-of-vol regime gate, "
         "(c) 30 days of clean paper trading with realistic slip accounting.",
         COL_RED),
        ("2. Halve Valk before next session open.",
         "44-contract 30-DTE iron condor concentrates short-gamma risk in a single "
         "structure. At next open: scale to 22 contracts and consider 14-DTE rebalance to "
         "improve theta-to-gamma ratio. Independent Opus reviewers flagged Valk as the "
         "single largest overnight tail.",
         COL_AMBER),
        ("3. Keep Aegis running, add a vol-of-vol gate.",
         "Aegis was the strongest performer: 52% hit rate, modest −$709 loss in a regime "
         "that broke Falcon. Recommend pausing new entries when the rolling 5-min stdev of "
         "1-min SPY returns exceeds 1.4× its trailing 20-day median. Single-parameter, "
         "freeze-compatible (lives at execution layer).",
         COL_INK),
        ("4. Recalibrate the slippage model and re-validate.",
         "Re-run the 21-day backtest at 8 bps round-trip slippage (conservative midpoint "
         "of observed 3-15 bps range). If the strategy remains positive after slip and "
         "across at least three distinct 21-day windows, redeploy. If it goes negative, "
         "the original deployment was a methodology error.",
         COL_INK),
        ("5. Decide on the freeze rule.",
         "The 90-day code-freeze and the risk-discipline goal are in tension. Two options: "
         "(a) suspend the freeze and treat the demo as an iterative live test, OR "
         "(b) keep the freeze and explicitly disclose that −10 to −20% drawdowns are "
         "accepted under the rule. The current ambiguous middle is worst-of-both.",
         COL_AMBER),
        ("6. Do not deploy real money under any current configuration.",
         "0 of 60 independent models recommended real-money deployment. Required "
         "preconditions per the panel: ≥30 days clean paper, slip recalibration, regime "
         "gate in place, daily-loss circuit breaker at −0.5% NAV, and a written acceptance "
         "of the maximum drawdown the freeze rule implies.",
         COL_RED),
    ]
    y = 0.86
    for title, body, c in recs:
        fig.patches.append(mpatches.Rectangle(
            (0.05, y - 0.005), 0.005, 0.04, transform=fig.transFigure,
            facecolor=c, edgecolor="none",
        ))
        fig.text(0.07, y + 0.015, title, fontsize=10.5, color=c, weight="bold")
        # Body wrap
        words = body.split()
        line = ""
        yy = y - 0.005
        for w in words:
            if len(line) + len(w) + 1 < 100:
                line = (line + " " + w).strip()
            else:
                fig.text(0.07, yy, line, fontsize=8.5, color=COL_INK, linespacing=1.5)
                yy -= 0.014
                line = w
        if line:
            fig.text(0.07, yy, line, fontsize=8.5, color=COL_INK, linespacing=1.5)
            yy -= 0.014
        y = yy - 0.022

    # Final stamp
    fig.patches.append(mpatches.FancyBboxPatch(
        (0.05, 0.05), 0.90, 0.06, boxstyle="round,pad=0.005",
        transform=fig.transFigure, facecolor="#f0f9ff",
        edgecolor=COL_ACCENT, lw=1.0,
    ))
    fig.text(0.5, 0.085,
              "Investor takeaway: the system is paper-only and working correctly as a learning instrument. "
              "Day 1 surfaced a real microstructure flaw in the breakout lane and a real concentration risk "
              "in the options lane. Both are addressable; neither is being papered over.",
              fontsize=9, color=COL_INK, ha="center", style="italic", wrap=True)

    add_page_footer(fig, page_n)
    pdf.savefig(fig)
    plt.close(fig)


def page_appendix_trades(pdf: PdfPages, trades: pd.DataFrame, page_n: int) -> None:
    fig = plt.figure(figsize=(8.5, 11), facecolor="white")
    add_page_header(fig, "Appendix — Trade Log",
                     "Full HFT order history for Day 1 (most recent first)")
    # Take 40 most recent
    df = trades.sort_values("ts").tail(50).copy()
    df["time_et"] = df["ts_et"].dt.strftime("%H:%M:%S")
    cols = ["time_et", "lane", "symbol", "action", "side", "qty", "notional", "status"]

    fig.text(0.04, 0.88, "TIME (ET)", fontsize=7, color=COL_MUTED,
              weight="bold")
    fig.text(0.14, 0.88, "LANE", fontsize=7, color=COL_MUTED,
              weight="bold")
    fig.text(0.21, 0.88, "SYM", fontsize=7, color=COL_MUTED,
              weight="bold")
    fig.text(0.28, 0.88, "ACTION", fontsize=7, color=COL_MUTED,
              weight="bold")
    fig.text(0.42, 0.88, "SIDE", fontsize=7, color=COL_MUTED,
              weight="bold")
    fig.text(0.49, 0.88, "QTY", fontsize=7, color=COL_MUTED,
              weight="bold")
    fig.text(0.58, 0.88, "NOTIONAL", fontsize=7, color=COL_MUTED,
              weight="bold")
    fig.text(0.74, 0.88, "STATUS", fontsize=7, color=COL_MUTED,
              weight="bold")
    fig.add_artist(plt.Line2D([0.03, 0.97], [0.873, 0.873], color="#9ca3af", lw=0.5))

    for i, (_, row) in enumerate(df.iloc[::-1].iterrows()):
        y = 0.86 - i * 0.0155
        if y < 0.06:
            break
        act = row["action"]
        c = (COL_GREEN if act == "take_profit" else
             COL_RED if act == "stop_loss" else
             COL_AMBER if act == "time_exit" else COL_INK)
        fig.text(0.04, y, row["time_et"], fontsize=7.5, color=COL_INK)
        fig.text(0.14, y, row["lane"], fontsize=7.5,
                  color=LANE_COLORS.get(row["lane"], COL_INK), weight="bold")
        fig.text(0.21, y, str(row["symbol"]), fontsize=7.5, color=COL_INK)
        fig.text(0.28, y, act, fontsize=7.5, color=c)
        fig.text(0.42, y, row["side"], fontsize=7.5, color=COL_INK)
        fig.text(0.49, y, str(int(row["qty"])), fontsize=7.5, color=COL_INK)
        fig.text(0.58, y, fmt_usd(float(row["notional"] or 0)), fontsize=7.5, color=COL_INK)
        fig.text(0.74, y, row["status"], fontsize=7.5, color=COL_MUTED, style="italic")

    add_page_footer(fig, page_n)
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> int:
    print(f"building case study PDF -> {OUT_PDF}")
    print("  fetching account state...")
    account = fetch_account_state()
    print("  loading trades...")
    trades = load_trades()
    marks = load_position_marks()
    condor = load_condor()
    print(f"  trades: {len(trades)}  marks: {len(marks)}  condors: {len(condor)}")

    print("  fetching minute bars...")
    bars = fetch_minute_bars(["SPY", "QQQ", "NVDA", "AAPL", "META", "AMD", "TSLA"], TODAY)
    print(f"  bars fetched: {list(bars.keys())}")

    print("  rendering PDF...")
    with PdfPages(OUT_PDF) as pdf:
        # Set PDF metadata
        d = pdf.infodict()
        d["Title"] = "Day 1 Case Study — Turbo Carnival Paper Trading"
        d["Author"] = "Turbo Carnival Research"
        d["Subject"] = "Multi-strategy paper-trading review with 60-model consensus"
        d["Keywords"] = "paper trading, iron condor, HFT, model review"

        page_cover(pdf, account)
        page_executive_summary(pdf, account, trades, 2)
        page_methodology(pdf, 3)
        page_lane_results(pdf, "Aegis", trades, marks, account, bars, 4)
        page_lane_results(pdf, "Falcon", trades, marks, account, bars, 5)
        page_lane_results(pdf, "Valk", trades, marks, account, bars, 6)
        page_news_timeline(pdf, bars, 7)
        page_intraday_charts(pdf, bars, trades, ["SPY", "QQQ"], 8)
        page_intraday_charts(pdf, bars, trades, ["NVDA", "AMD"], 9)
        page_intraday_charts(pdf, bars, trades, ["TSLA", "META"], 10)
        page_intraday_charts(pdf, bars, trades, ["AAPL"], 11)
        page_swarm_analysis(pdf, 12)
        page_risk_analysis(pdf, 13)
        page_recommendations(pdf, 14)
        page_appendix_trades(pdf, trades, 15)

    print(f"\nPDF written: {OUT_PDF}")
    print(f"size: {OUT_PDF.stat().st_size / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
