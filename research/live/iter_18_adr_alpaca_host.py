"""iter_18 ADR Dual-Listing Arbitrage — ALPACA-ONLY Paper-Shadow Host (HK simulated).

================================================================================
HONEST DISCLAIMER — READ THIS BEFORE TRUSTING ANY P&L FROM THIS RUN
================================================================================
This host is a SPLIT-LEG HALF-REAL paper shadow:
  - US ADR leg: REAL paper execution through Alpaca (real fills, slippage,
                rejects, reconciliation against broker truth).
  - HK ord  leg: SIMULATED in code. Virtual entries/exits are logged at the
                most recent HK Close from yfinance (end-of-day, free, delayed).

It is NOT a full validation of the strategy. It validates only:
  + Signal logic (parity calc, threshold trigger, ex-div blackout, stop/exit)
  + US execution (Alpaca fills, latency, IOC behaviour, partial fills)
  + Broker-truth reconciliation on the US side (position vs. expected qty)
  + Operational reliability (DB, logging, kill switches, cron scheduling)

It DOES NOT validate:
  - HKEX execution (real broker fills, partial fills, halts, after-auction)
  - HKEX slippage (~10-30 bps round-trip is realistic; sim assumes 0 bps)
  - HKEX trading halts / typhoon closures (sim never halts)
  - Intraday HK price moves between US close and HK open (sim uses HK Close)
  - Cross-listing operational reality (HKD funding, settlement T+2, FX risk)
  - Real USD/HKD FX risk (sim uses the 7.80 peg; in practice peg holds tightly
    but intraday band is +/- ~0.5%, which can flip an 80 bps signal)

WHEN TO USE THIS: As the first 30-day paper shadow BEFORE wiring up a real HKEX
broker (IBKR, Futu, Tiger, or HK-domiciled prime). It pins down the BIGGEST
single risk surface (signal stability + US execution + reconciliation) cheaply
and quickly, so when you later swap in real HK execution you can attribute any
slippage divergence to the HK leg specifically.

DO NOT extrapolate the combined P&L number to expected live performance. The
simulated HK leg overstates edge by an unknown amount (likely 20-50 bps per
round-trip from execution costs, plus episodic halt/slippage tail risk).

See README_ADR_ALPACA.md for the full migration path to real HK execution.

================================================================================
STRATEGY (sourced from iter_18_cleanup_summary.json — PASS verdict):
  Pairs (default --universe core, 5 pairs):
    BABA / 9988.HK 1:8
    JD   / 9618.HK 1:2
    BIDU / 9888.HK 1:8
    NTES / 9999.HK 1:5
    TCOM / 9961.HK 1:1
  Pairs (--universe expanded, 12 pairs that PASSED iter_18_expanded):
    Above 5 + LI/2015, NIO/9866, XPEV/9868, BILI/9626, TME/1698,
    MNSO/9896, ZTO/2057  (YMM/2643 excluded — failed expanded)
  Entry:    |deviation_bps| > 80
  Exit:     |deviation_bps| < 20
  Stop:     |deviation_bps| > 250
  Cost:     30 bps round-trip (for sim accounting only — US side uses real fills)
  Blackout: +/- 3 sessions around ex-dividend on EITHER leg
  Sizing:   20% NAV per pair, max 5 pairs simultaneously = 100% NAV
  FX:       USD/HKD = 7.80 peg (Hong Kong Monetary Authority)

================================================================================
DEPLOYMENT
First run (interactive confirm on first 10 trades, recommended):
    /Users/weeoto/turbo-carnival/.venv/bin/python \\
        /Users/weeoto/turbo-carnival/research/live/iter_18_adr_alpaca_host.py \\
        --account AEGIS --confirm --cycle us_close

One-shot cron-style:
    /Users/weeoto/turbo-carnival/.venv/bin/python \\
        /Users/weeoto/turbo-carnival/research/live/iter_18_adr_alpaca_host.py \\
        --account AEGIS --cycle us_close

Dry run (compute signal & log, NO orders, NO sim updates persisted as fills):
    /Users/weeoto/turbo-carnival/.venv/bin/python \\
        /Users/weeoto/turbo-carnival/research/live/iter_18_adr_alpaca_host.py \\
        --account AEGIS --dry-run

Force flatten everything now (real US + virtual HK):
    /Users/weeoto/turbo-carnival/.venv/bin/python \\
        /Users/weeoto/turbo-carnival/research/live/iter_18_adr_alpaca_host.py \\
        --account AEGIS --flatten-now

================================================================================
KILL SWITCHES
  Automatic (script triggers, flattens, exits non-zero so cron can page):
    1. Per-pair simulated loss > 5% NAV
    2. Total gross exposure > 100% NAV
    3. Kill file present at startup: research/live/STOP_ADR_ALPACA
  Manual:
    A. touch /Users/weeoto/turbo-carnival/research/live/STOP_ADR_ALPACA
       (halts next cycle — does NOT auto-flatten; pair --flatten-now if you
       want positions closed)
    B. SIGINT / SIGTERM the process.
    C. --flatten-now sub-command (above).

Author : Deployment Engineer A36
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
REPO = Path("/Users/weeoto/turbo-carnival")
ENV_FILE = REPO / "council" / ".env.local"
ROOT = REPO / "research"
LIVE_DIR = ROOT / "live"
LOG_PATH = LIVE_DIR / "iter_18_adr_alpaca_host.log"
DECISIONS_DB = LIVE_DIR / "adr_alpaca_decisions.sqlite3"
NAV_STATE_PATH = LIVE_DIR / "adr_alpaca_nav_state.json"
KILL_FLAG_PATH = LIVE_DIR / "STOP_ADR_ALPACA"
CONFIRM_COUNTER_PATH = LIVE_DIR / "adr_alpaca_confirm_counter.json"
VIRTUAL_HK_STATE_PATH = LIVE_DIR / "adr_alpaca_virtual_hk_positions.json"

NY = ZoneInfo("America/New_York")
ALPACA_TRADING = "https://paper-api.alpaca.markets/v2"
ALPACA_DATA = "https://data.alpaca.markets/v2"

# --- Strategy params (frozen, copied from iter_18_cleanup_summary.json) ----
# Core 5 pairs (iter_18_cleanup PASS @ mean_bps_net=65.79)
CORE_PAIRS: list[tuple[str, str, int]] = [
    ("BABA", "9988.HK", 8),
    ("JD",   "9618.HK", 2),
    ("BIDU", "9888.HK", 8),
    ("NTES", "9999.HK", 5),
    ("TCOM", "9961.HK", 1),
]

# Expanded 12 pairs that PASSED iter_18_expanded.py (YMM/2643 excluded — failed)
EXPANDED_PAIRS: list[tuple[str, str, int]] = CORE_PAIRS + [
    ("LI",   "2015.HK", 2),
    ("NIO",  "9866.HK", 1),
    ("XPEV", "9868.HK", 2),
    ("BILI", "9626.HK", 1),
    ("TME",  "1698.HK", 2),
    ("MNSO", "9896.HK", 4),
    ("ZTO",  "2057.HK", 1),
]

ENTRY_BPS = 80.0
EXIT_BPS = 20.0
STOP_BPS = 250.0
ROUND_TRIP_COST_BPS = 30.0
EX_DIV_SKIP_WINDOW_BARS = 3
USD_HKD_PEG = 7.80
USD_PER_HKD = 1.0 / USD_HKD_PEG

# Sizing & risk
PER_PAIR_NAV_FRACTION = 0.20
MAX_OPEN_PAIRS = 5
MAX_TOTAL_GROSS_NAV_FRAC = 1.00
PER_PAIR_LOSS_KILL_FRAC = 0.05
CONFIRM_TRADE_COUNT = 10

# Alpaca order knobs (mirrors svxy_sweet_spot_host.py)
ORDER_PRICE_SLIP_BPS = 25  # 25 bps marketable-limit aggression
ORDER_POLL_SECS = 6  # poll up to N seconds for IOC fill status
MIN_ORDER_QTY = 1  # whole-share US orders

# Backtest verification
BACKTEST_LOOKBACK_DAYS = 60
BACKTEST_DIVERGENCE_FRACTION = 0.30  # exit if expected vs realised diverges > 30%

# Caches written by iter_18_cleanup.py / iter_18_expanded.py
CACHE_CORE = ROOT / "data" / "cache" / "us_hk_pairs_daily_raw.parquet"
CACHE_EXPANDED = ROOT / "data" / "cache" / "us_hk_pairs_expanded_daily_raw.parquet"
DIV_CACHE_CORE = ROOT / "data" / "cache" / "us_hk_pairs_dividends.parquet"
DIV_CACHE_EXPANDED = ROOT / "data" / "cache" / "us_hk_pairs_expanded_dividends.parquet"

LOG_FMT = "%(asctime)s %(levelname)s %(message)s"
logger = logging.getLogger("iter_18_adr_alpaca_host")
STOPPED = False


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("iter_18_adr_alpaca_host")
    log.setLevel(logging.INFO)
    for h in list(log.handlers):
        log.removeHandler(h)
    fh = logging.FileHandler(LOG_PATH, mode="a")
    fh.setFormatter(logging.Formatter(LOG_FMT))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter(LOG_FMT))
    log.addHandler(fh)
    log.addHandler(sh)
    log.propagate = False
    return log


def print_startup_banner(log: logging.Logger, universe_name: str, n_pairs: int) -> None:
    banner = (
        "=" * 78 + "\n"
        "WARNING: HK leg is SIMULATED in this paper-shadow host.\n"
        "  Real combined edge will differ from this log's P&L by:\n"
        "    - HKEX execution costs ~10-30 bps round-trip (sim: 0)\n"
        "    - HKEX timing slippage (sim uses HK Close end-of-day)\n"
        "    - HKEX halts/typhoon closures (sim never halts)\n"
        "    - Intraday FX moves (sim uses 7.80 peg)\n"
        "  USE THIS PAPER SHADOW TO VALIDATE: signal logic + US execution +\n"
        "    broker-truth reconciliation on Alpaca side ONLY.\n"
        "  DO NOT extrapolate combined P&L to expected live performance until\n"
        "    HK leg is wired to a real broker (IBKR / Futu / Tiger).\n"
        "  See research/live/README_ADR_ALPACA.md for the migration path.\n"
        + "=" * 78
    )
    for line in banner.split("\n"):
        log.warning(line)
    log.info(f"Universe: {universe_name} ({n_pairs} pairs)")


# ---------------------------------------------------------------------------
# Env / credentials
# ---------------------------------------------------------------------------
def load_env() -> None:
    """Merge council/.env.local into os.environ (does not override existing)."""
    if not ENV_FILE.exists():
        logger.warning(f"env file not found: {ENV_FILE} — relying on process env")
        return
    for raw in ENV_FILE.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def resolve_credentials(account: str) -> tuple[str, str]:
    """Return (key, secret). Falls back to generic ALPACA_API_KEY/SECRET."""
    account = account.upper()
    key = os.environ.get(f"ALPACA_{account}_KEY") or os.environ.get("ALPACA_API_KEY")
    sec = (
        os.environ.get(f"ALPACA_{account}_SECRET")
        or os.environ.get("ALPACA_API_SECRET")
    )
    if not key or not sec:
        raise SystemExit(
            f"Missing Alpaca paper credentials. Expected ALPACA_{account}_KEY/"
            f"ALPACA_{account}_SECRET (or ALPACA_API_KEY / ALPACA_API_SECRET)."
        )
    return key, sec


# ---------------------------------------------------------------------------
# Alpaca REST helpers
# ---------------------------------------------------------------------------
def _headers(key: str, secret: str) -> dict:
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def alpaca_get_account(key: str, secret: str) -> Optional[dict]:
    try:
        r = requests.get(
            f"{ALPACA_TRADING}/account", headers=_headers(key, secret), timeout=10
        )
        if r.status_code == 200:
            return r.json()
        logger.warning(f"account GET {r.status_code}: {r.text[:200]}")
    except Exception as exc:
        logger.warning(f"account exception: {exc}")
    return None


def alpaca_get_position(key: str, secret: str, symbol: str) -> Optional[dict]:
    try:
        r = requests.get(
            f"{ALPACA_TRADING}/positions/{symbol}",
            headers=_headers(key, secret),
            timeout=10,
        )
        if r.status_code == 200:
            return r.json()
        if r.status_code == 404:
            return None
        logger.warning(f"position GET {symbol} {r.status_code}: {r.text[:200]}")
    except Exception as exc:
        logger.warning(f"position exception: {exc}")
    return None


def alpaca_list_positions(key: str, secret: str) -> list[dict]:
    try:
        r = requests.get(
            f"{ALPACA_TRADING}/positions", headers=_headers(key, secret), timeout=10
        )
        if r.status_code == 200:
            return r.json() or []
    except Exception as exc:
        logger.warning(f"positions list exception: {exc}")
    return []


def alpaca_get_latest_trade(key: str, secret: str, symbol: str) -> Optional[float]:
    try:
        r = requests.get(
            f"{ALPACA_DATA}/stocks/{symbol}/trades/latest",
            headers=_headers(key, secret),
            timeout=10,
        )
        if r.status_code == 200:
            j = r.json()
            if j.get("trade") and j["trade"].get("p"):
                return float(j["trade"]["p"])
    except Exception as exc:
        logger.warning(f"trade GET {symbol} exception: {exc}")
    return None


def alpaca_get_asset_tradable(key: str, secret: str, symbol: str) -> bool:
    try:
        r = requests.get(
            f"{ALPACA_TRADING}/assets/{symbol}",
            headers=_headers(key, secret),
            timeout=8,
        )
        if r.status_code == 200:
            j = r.json()
            return bool(j.get("tradable")) and j.get("status") == "active"
    except Exception:
        pass
    return False


def alpaca_get_order(key: str, secret: str, order_id: str) -> Optional[dict]:
    try:
        r = requests.get(
            f"{ALPACA_TRADING}/orders/{order_id}",
            headers=_headers(key, secret),
            timeout=8,
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def alpaca_submit_marketable_limit_ioc(
    key: str,
    secret: str,
    symbol: str,
    qty: float,
    side: str,
    ref_price: float,
    client_tag: str,
) -> Optional[dict]:
    """Submit a marketable IOC limit order at ref_price +/- ORDER_PRICE_SLIP_BPS.

    For short orders Alpaca requires the asset to be marginable+shortable AND
    the account to be margin-enabled. Paper accounts default to margin so
    shorts are typically OK on US ADRs in this universe.
    """
    slip = ref_price * (ORDER_PRICE_SLIP_BPS / 10_000.0)
    if side == "buy":
        limit_price = round(ref_price + slip, 2)
    else:
        limit_price = round(ref_price - slip, 2)
    payload = {
        "symbol": symbol,
        "qty": str(int(round(qty))),
        "side": side,
        "type": "limit",
        "limit_price": str(limit_price),
        "time_in_force": "ioc",
        "client_order_id": f"adr-alpaca-{client_tag}-{int(time.time() * 1000)}",
    }
    try:
        r = requests.post(
            f"{ALPACA_TRADING}/orders",
            headers=_headers(key, secret),
            json=payload,
            timeout=10,
        )
        if r.status_code in (200, 201):
            return r.json()
        logger.error(f"order failed {r.status_code}: {r.text[:300]}")
    except Exception as exc:
        logger.error(f"order exception: {exc}")
    return None


# ---------------------------------------------------------------------------
# SQLite reconciliation store
# ---------------------------------------------------------------------------
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS decisions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc        TEXT    NOT NULL,
    cycle         TEXT    NOT NULL,
    pair          TEXT    NOT NULL,
    us_symbol     TEXT    NOT NULL,
    hk_symbol     TEXT    NOT NULL,
    ratio         INTEGER NOT NULL,
    us_price_usd  REAL,
    hk_price_hkd  REAL,
    fx_usd_hkd    REAL,
    dev_bps       REAL,
    ex_div_block  INTEGER,
    action        TEXT    NOT NULL,
    reason        TEXT,
    direction     TEXT,
    us_qty        INTEGER,
    hk_qty        INTEGER,
    note          TEXT
);

CREATE TABLE IF NOT EXISTS orders (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc          TEXT    NOT NULL,
    cycle           TEXT    NOT NULL,
    pair            TEXT    NOT NULL,
    leg             TEXT    NOT NULL,      -- 'US' (real) or 'HK_SIM' (virtual)
    symbol          TEXT    NOT NULL,
    side            TEXT    NOT NULL,      -- 'buy' or 'sell'
    qty             INTEGER NOT NULL,
    submitted_ok    INTEGER NOT NULL,
    fill_price      REAL,                  -- avg fill (US: real, HK: sim close)
    fill_qty        INTEGER,
    status          TEXT,
    order_id        TEXT,
    client_order_id TEXT,
    is_simulated    INTEGER NOT NULL,      -- 0 = real US, 1 = virtual HK
    note            TEXT
);

CREATE TABLE IF NOT EXISTS virtual_hk_positions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc       TEXT NOT NULL,
    pair         TEXT NOT NULL,
    hk_symbol    TEXT NOT NULL,
    qty          INTEGER NOT NULL,
    avg_cost_hkd REAL NOT NULL,
    open_close   TEXT NOT NULL,   -- 'open' or 'close'
    note         TEXT
);

CREATE TABLE IF NOT EXISTS positions_snapshot (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc       TEXT NOT NULL,
    cycle        TEXT NOT NULL,
    symbol       TEXT NOT NULL,
    is_simulated INTEGER NOT NULL,
    qty          REAL,
    avg_cost     REAL,
    market_value REAL,
    unreal_pnl   REAL,
    note         TEXT
);

CREATE TABLE IF NOT EXISTS eod_summary (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc           TEXT    NOT NULL,
    cycle            TEXT    NOT NULL,
    nav_usd          REAL,
    open_pair_count  INTEGER,
    gross_exposure   REAL,
    real_us_pnl      REAL,
    sim_hk_pnl       REAL,
    total_combined_pnl REAL,
    payload_json     TEXT
);

CREATE TABLE IF NOT EXISTS kill_events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc    TEXT NOT NULL,
    cycle     TEXT,
    reason    TEXT NOT NULL,
    detail    TEXT
);
"""


def open_db() -> sqlite3.Connection:
    DECISIONS_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DECISIONS_DB))
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Kill-switch primitives
# ---------------------------------------------------------------------------
class KillSwitch(Exception):
    """Raised on hard-kill conditions. Cycle aborts and positions flatten."""


def write_kill_flag(reason: str, detail: str = "") -> None:
    KILL_FLAG_PATH.parent.mkdir(parents=True, exist_ok=True)
    KILL_FLAG_PATH.write_text(
        json.dumps(
            {
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "reason": reason,
                "detail": detail,
            },
            indent=2,
        )
    )


def kill_flag_present() -> Optional[dict]:
    if not KILL_FLAG_PATH.exists():
        return None
    try:
        return json.loads(KILL_FLAG_PATH.read_text())
    except json.JSONDecodeError:
        return {"reason": "unknown", "detail": "corrupt STOP_ADR_ALPACA file"}


def record_kill(
    conn: sqlite3.Connection, cycle: str, reason: str, detail: str
) -> None:
    conn.execute(
        "INSERT INTO kill_events (ts_utc, cycle, reason, detail) VALUES (?, ?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), cycle, reason, detail),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Confirmation counter
# ---------------------------------------------------------------------------
def load_confirm_counter() -> int:
    if not CONFIRM_COUNTER_PATH.exists():
        return 0
    try:
        return int(json.loads(CONFIRM_COUNTER_PATH.read_text()).get("count", 0))
    except (json.JSONDecodeError, ValueError):
        return 0


def bump_confirm_counter() -> int:
    cur = load_confirm_counter() + 1
    CONFIRM_COUNTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIRM_COUNTER_PATH.write_text(json.dumps({"count": cur}, indent=2))
    return cur


def interactive_confirm(prompt: str, allow_skip: bool) -> bool:
    if allow_skip:
        return True
    try:
        sys.stdout.write(f"\n>>> CONFIRM: {prompt}  [y/N]: ")
        sys.stdout.flush()
        ans = sys.stdin.readline().strip().lower()
    except EOFError:
        return False
    return ans in {"y", "yes"}


# ---------------------------------------------------------------------------
# Backtest divergence verification (re-uses iter_18_cleanup summary)
# ---------------------------------------------------------------------------
def verify_backtest_divergence(log: logging.Logger, universe: str) -> None:
    """Check that the realised backtest trade rate is within tolerance.

    Reads research/results/iter_18_cleanup_summary.json (or expanded variant)
    and compares the expected trades-per-year to the actual count in the last
    BACKTEST_LOOKBACK_DAYS of the trade log. Bails out loud if divergence
    exceeds BACKTEST_DIVERGENCE_FRACTION.
    """
    if universe == "expanded":
        summary_path = ROOT / "results" / "iter_18_expanded_summary.json"
        trades_csv = ROOT / "results" / "iter_18_expanded_trades.csv"
    else:
        summary_path = ROOT / "results" / "iter_18_cleanup_summary.json"
        trades_csv = ROOT / "results" / "iter_18_cleanup_trades.csv"

    if not summary_path.exists():
        raise SystemExit(
            f"FATAL: missing backtest summary {summary_path}. "
            "Run scripts/iter_18_cleanup.py (or iter_18_expanded.py) first."
        )

    summary = json.loads(summary_path.read_text())
    agg = summary.get("aggregate", {})
    total_trades = int(agg.get("total_trades", 0))
    elapsed_years = float(agg.get("elapsed_years", 0.0)) or 1.0
    trades_per_year = total_trades / max(elapsed_years, 1e-9)
    expected_60d = trades_per_year * (BACKTEST_LOOKBACK_DAYS / 365.25)

    actual_60d = 0
    if trades_csv.exists():
        try:
            tdf = pd.read_csv(trades_csv, parse_dates=["entry_date", "exit_date"])
            cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(
                days=BACKTEST_LOOKBACK_DAYS
            )
            actual_60d = int((tdf["exit_date"] >= cutoff).sum())
        except (KeyError, ValueError) as exc:
            log.warning(f"Could not parse {trades_csv} for divergence check: {exc}")

    log.info(
        f"Backtest verification ({universe}): expected ~{expected_60d:.1f} "
        f"trades in last {BACKTEST_LOOKBACK_DAYS}d, actual {actual_60d} "
        f"(annual rate {trades_per_year:.1f})"
    )
    if expected_60d > 0:
        divergence = abs(actual_60d - expected_60d) / expected_60d
    else:
        divergence = 0.0
    log.info(f"Backtest divergence vs spec rate: {divergence * 100:.1f}%")
    if divergence > BACKTEST_DIVERGENCE_FRACTION:
        raise SystemExit(
            f"FATAL: backtest divergence {divergence * 100:.1f}% exceeds "
            f"{BACKTEST_DIVERGENCE_FRACTION * 100:.0f}% threshold. "
            "Re-run scripts/iter_18_cleanup.py (or expanded) and investigate."
        )
    log.info("Backtest verification PASS.")


# ---------------------------------------------------------------------------
# Dividend calendar
# ---------------------------------------------------------------------------
def load_dividend_calendar(
    log: logging.Logger, universe: str
) -> dict[str, set]:
    """Return {ticker -> set(ex_date)} for blackout calc."""
    if universe == "expanded":
        div_caches = [DIV_CACHE_EXPANDED, DIV_CACHE_CORE]
    else:
        div_caches = [DIV_CACHE_CORE]
    cal: dict[str, set] = {}
    for div_cache in div_caches:
        if not div_cache.exists():
            log.warning(f"Dividend cache {div_cache} missing.")
            continue
        try:
            df = pd.read_parquet(div_cache)
        except (OSError, ValueError) as exc:
            log.warning(f"Could not read dividend cache {div_cache}: {exc}")
            continue
        for col in df.columns:
            s = df[col].dropna()
            if s.empty:
                continue
            bare = col.replace(".HK", "")
            cal.setdefault(bare, set()).update(
                pd.to_datetime(s.index).normalize().tolist()
            )
            cal.setdefault(col, set()).update(
                pd.to_datetime(s.index).normalize().tolist()
            )
    log.info(f"Dividend calendar loaded: {len(cal)} tickers")
    return cal


def is_ex_div_blocked(
    today: pd.Timestamp,
    us_sym: str,
    hk_sym: str,
    cal: dict[str, set],
) -> bool:
    today = pd.Timestamp(today).normalize()
    bare_hk = hk_sym.replace(".HK", "")
    for sym in (us_sym, hk_sym, bare_hk):
        if sym not in cal:
            continue
        for ex in cal[sym]:
            if abs((today - ex).days) <= EX_DIV_SKIP_WINDOW_BARS:
                return True
    return False


# ---------------------------------------------------------------------------
# Market data: HK closes from yfinance, US prices from Alpaca
# ---------------------------------------------------------------------------
def fetch_hk_close_yf(hk_symbol: str, log: logging.Logger) -> Optional[float]:
    """Pull the latest HK close (HKD) from yfinance. End-of-day only."""
    try:
        import yfinance as yf

        hist = yf.Ticker(hk_symbol).history(period="5d", interval="1d")
        if hist.empty:
            log.warning(f"yfinance returned empty for {hk_symbol}")
            return None
        return float(hist["Close"].iloc[-1])
    except Exception as exc:
        log.warning(f"yfinance HK fetch for {hk_symbol} failed: {exc}")
        return None


def fetch_hk_closes_batch(
    hk_symbols: list[str], log: logging.Logger
) -> dict[str, Optional[float]]:
    """Batch yfinance pull for all HK tickers. Falls back to per-ticker."""
    out: dict[str, Optional[float]] = {sym: None for sym in hk_symbols}
    try:
        import yfinance as yf

        raw = yf.download(
            hk_symbols,
            period="5d",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=True,
            group_by="ticker",
        )
        if isinstance(raw.columns, pd.MultiIndex):
            for sym in hk_symbols:
                if sym in raw.columns.get_level_values(0):
                    sub = raw[sym]
                    if "Close" in sub.columns:
                        s = sub["Close"].dropna()
                        if len(s) > 0:
                            out[sym] = float(s.iloc[-1])
        else:
            # Single-ticker fallback shape
            if "Close" in raw.columns and len(hk_symbols) == 1:
                s = raw["Close"].dropna()
                if len(s) > 0:
                    out[hk_symbols[0]] = float(s.iloc[-1])
    except Exception as exc:
        log.warning(f"yfinance batch fetch failed: {exc} — using per-ticker fallback")
        for sym in hk_symbols:
            if out[sym] is None:
                out[sym] = fetch_hk_close_yf(sym, log)
    # Any remaining Nones get a per-ticker retry
    for sym, px in out.items():
        if px is None:
            out[sym] = fetch_hk_close_yf(sym, log)
    return out


# ---------------------------------------------------------------------------
# Virtual HK position state — persisted to JSON
# ---------------------------------------------------------------------------
@dataclass
class VirtualHKPosition:
    pair: str
    hk_symbol: str
    qty: int = 0
    avg_cost_hkd: float = 0.0
    opened_at_utc: Optional[str] = None
    note: str = ""


def load_virtual_hk_positions() -> dict[str, VirtualHKPosition]:
    """Load virtual HK positions from disk. Returns {pair_key -> VirtualHKPosition}."""
    if not VIRTUAL_HK_STATE_PATH.exists():
        return {}
    try:
        data = json.loads(VIRTUAL_HK_STATE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    out: dict[str, VirtualHKPosition] = {}
    for pair, p in data.items():
        out[pair] = VirtualHKPosition(
            pair=pair,
            hk_symbol=p.get("hk_symbol", ""),
            qty=int(p.get("qty", 0)),
            avg_cost_hkd=float(p.get("avg_cost_hkd", 0.0)),
            opened_at_utc=p.get("opened_at_utc"),
            note=p.get("note", ""),
        )
    return out


def persist_virtual_hk_positions(positions: dict[str, VirtualHKPosition]) -> None:
    VIRTUAL_HK_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        pair: {
            "hk_symbol": p.hk_symbol,
            "qty": p.qty,
            "avg_cost_hkd": p.avg_cost_hkd,
            "opened_at_utc": p.opened_at_utc,
            "note": p.note,
        }
        for pair, p in positions.items()
        if p.qty != 0
    }
    VIRTUAL_HK_STATE_PATH.write_text(json.dumps(serializable, indent=2))


def record_virtual_hk_event(
    conn: sqlite3.Connection,
    pair: str,
    hk_symbol: str,
    qty: int,
    avg_cost_hkd: float,
    open_close: str,
    note: str = "",
) -> None:
    conn.execute(
        """INSERT INTO virtual_hk_positions
           (ts_utc, pair, hk_symbol, qty, avg_cost_hkd, open_close, note)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now(timezone.utc).isoformat(),
            pair,
            hk_symbol,
            qty,
            avg_cost_hkd,
            open_close,
            note,
        ),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Per-pair state (combines real US + virtual HK)
# ---------------------------------------------------------------------------
@dataclass
class PairState:
    us_sym: str
    hk_sym: str
    ratio: int
    us_qty: int = 0           # positive = long US, negative = short US (real)
    hk_qty: int = 0           # positive = long HK ord (VIRTUAL)
    avg_us_cost: float = 0.0  # USD (real, from Alpaca position)
    avg_hk_cost: float = 0.0  # HKD (virtual, from sim entry)
    direction: Optional[str] = None

    @property
    def open(self) -> bool:
        return self.us_qty != 0 or self.hk_qty != 0


def reconstruct_pair_states(
    pairs: list[tuple[str, str, int]],
    alpaca_positions: list[dict],
    virtual_hk: dict[str, VirtualHKPosition],
) -> dict[str, PairState]:
    """Merge real Alpaca US positions with virtual HK sim positions into PairState."""
    alpaca_by_symbol = {p["symbol"]: p for p in alpaca_positions}
    states: dict[str, PairState] = {}
    for us, hk, ratio in pairs:
        pair_key = f"{us}/{hk}"
        st = PairState(us_sym=us, hk_sym=hk, ratio=ratio)
        if us in alpaca_by_symbol:
            p = alpaca_by_symbol[us]
            try:
                st.us_qty = int(round(float(p.get("qty", 0))))
                st.avg_us_cost = float(p.get("avg_entry_price", 0) or 0)
            except (TypeError, ValueError):
                pass
        if pair_key in virtual_hk:
            v = virtual_hk[pair_key]
            st.hk_qty = v.qty
            st.avg_hk_cost = v.avg_cost_hkd
        if st.us_qty < 0 and st.hk_qty > 0:
            st.direction = "short_us_long_hk"
        elif st.us_qty > 0 and st.hk_qty < 0:
            st.direction = "long_us_short_hk"
        states[pair_key] = st
    return states


# ---------------------------------------------------------------------------
# NAV
# ---------------------------------------------------------------------------
def fetch_nav_usd(key: str, secret: str, log: logging.Logger) -> float:
    acct = alpaca_get_account(key, secret)
    if acct and acct.get("equity"):
        try:
            return float(acct["equity"])
        except (TypeError, ValueError):
            pass
    if NAV_STATE_PATH.exists():
        try:
            return float(
                json.loads(NAV_STATE_PATH.read_text()).get("nav_usd", 100_000.0)
            )
        except (json.JSONDecodeError, ValueError, OSError):
            pass
    log.warning("NAV unavailable; defaulting to $100,000 (paper).")
    return 100_000.0


def persist_nav(nav_usd: float) -> None:
    NAV_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    NAV_STATE_PATH.write_text(
        json.dumps(
            {"ts_utc": datetime.now(timezone.utc).isoformat(), "nav_usd": nav_usd},
            indent=2,
        )
    )


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------
def compute_sizing(
    nav_usd: float, us_px: float, hk_px_hkd: float, ratio: int
) -> tuple[int, int]:
    """Return (us_qty_abs, hk_qty_abs) for a dollar-neutral pair entry.

    Per-pair notional: PER_PAIR_NAV_FRACTION * NAV on EACH leg, sized in USD.
    HK leg is in HKD shares — converted via the peg, rounded down to 100-share
    board lot (HKEX standard for these tickers).
    """
    per_leg_usd = PER_PAIR_NAV_FRACTION * nav_usd
    us_qty_abs = max(int(math.floor(per_leg_usd / max(us_px, 1e-6))), 0)
    hk_px_usd = hk_px_hkd * USD_PER_HKD
    hk_qty_abs = max(int(math.floor(per_leg_usd / max(hk_px_usd, 1e-6))), 0)
    hk_qty_abs = (hk_qty_abs // 100) * 100  # HKEX board lot
    _ = ratio
    return us_qty_abs, hk_qty_abs


# ---------------------------------------------------------------------------
# Decision logic — identical to IBKR host
# ---------------------------------------------------------------------------
@dataclass
class PairDecision:
    pair: str
    us_sym: str
    hk_sym: str
    ratio: int
    us_px: Optional[float]
    hk_px: Optional[float]
    fx: float
    dev_bps: Optional[float]
    ex_div_block: bool
    action: str          # 'open' | 'close' | 'hold' | 'stop' | 'skip'
    direction: Optional[str] = None
    reason: str = ""
    us_qty: int = 0
    hk_qty: int = 0
    note: str = ""


def decide_pair(
    st: PairState,
    us_px: Optional[float],
    hk_px: Optional[float],
    fx: float,
    blocked: bool,
    nav_usd: float,
) -> PairDecision:
    pair = f"{st.us_sym}/{st.hk_sym}"
    if us_px is None or hk_px is None:
        return PairDecision(
            pair, st.us_sym, st.hk_sym, st.ratio, us_px, hk_px, fx,
            None, blocked, "skip", reason="missing_price",
        )

    parity_usd = (hk_px / fx) * st.ratio
    dev_bps = (
        10_000.0 * (us_px - parity_usd) / parity_usd if parity_usd > 0 else 0.0
    )

    if not st.open:
        if blocked:
            return PairDecision(
                pair, st.us_sym, st.hk_sym, st.ratio, us_px, hk_px, fx,
                dev_bps, blocked, "skip", reason="ex_div_blackout",
            )
        if abs(dev_bps) > ENTRY_BPS:
            direction = "short_us_long_hk" if dev_bps > 0 else "long_us_short_hk"
            us_qty, hk_qty = compute_sizing(nav_usd, us_px, hk_px, st.ratio)
            return PairDecision(
                pair, st.us_sym, st.hk_sym, st.ratio, us_px, hk_px, fx,
                dev_bps, blocked, "open", direction=direction,
                reason=f"|dev|={abs(dev_bps):.1f}>{ENTRY_BPS}",
                us_qty=us_qty, hk_qty=hk_qty,
            )
        return PairDecision(
            pair, st.us_sym, st.hk_sym, st.ratio, us_px, hk_px, fx,
            dev_bps, blocked, "hold", reason="below_entry_threshold",
        )

    # Already open
    if abs(dev_bps) > STOP_BPS:
        return PairDecision(
            pair, st.us_sym, st.hk_sym, st.ratio, us_px, hk_px, fx,
            dev_bps, blocked, "stop", direction=st.direction,
            reason=f"|dev|={abs(dev_bps):.1f}>{STOP_BPS}",
            us_qty=st.us_qty, hk_qty=st.hk_qty,
        )
    if abs(dev_bps) < EXIT_BPS and not blocked:
        return PairDecision(
            pair, st.us_sym, st.hk_sym, st.ratio, us_px, hk_px, fx,
            dev_bps, blocked, "close", direction=st.direction,
            reason=f"|dev|={abs(dev_bps):.1f}<{EXIT_BPS}",
            us_qty=st.us_qty, hk_qty=st.hk_qty,
        )
    return PairDecision(
        pair, st.us_sym, st.hk_sym, st.ratio, us_px, hk_px, fx,
        dev_bps, blocked, "hold", direction=st.direction,
        reason="within_band", us_qty=st.us_qty, hk_qty=st.hk_qty,
    )


def log_decision(conn: sqlite3.Connection, cycle: str, d: PairDecision) -> None:
    conn.execute(
        """INSERT INTO decisions (
            ts_utc, cycle, pair, us_symbol, hk_symbol, ratio,
            us_price_usd, hk_price_hkd, fx_usd_hkd, dev_bps,
            ex_div_block, action, reason, direction,
            us_qty, hk_qty, note
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now(timezone.utc).isoformat(),
            cycle,
            d.pair,
            d.us_sym,
            d.hk_sym,
            d.ratio,
            d.us_px,
            d.hk_px,
            d.fx,
            d.dev_bps,
            1 if d.ex_div_block else 0,
            d.action,
            d.reason,
            d.direction,
            d.us_qty,
            d.hk_qty,
            d.note,
        ),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# US execution (real Alpaca)
# ---------------------------------------------------------------------------
def execute_us_order(
    key: str,
    secret: str,
    symbol: str,
    side: str,
    qty: int,
    ref_price: float,
    label: str,
    log: logging.Logger,
) -> dict:
    """Submit IOC limit, poll for fill, return result dict."""
    if qty <= 0:
        return {
            "submitted_ok": False,
            "status": "skipped",
            "note": "zero qty",
            "fill_qty": 0,
            "fill_price": None,
            "order_id": "",
        }
    if ref_price is None or ref_price <= 0:
        return {
            "submitted_ok": False,
            "status": "skipped",
            "note": "no ref price",
            "fill_qty": 0,
            "fill_price": None,
            "order_id": "",
        }
    result = alpaca_submit_marketable_limit_ioc(
        key, secret, symbol, qty, side, ref_price, label
    )
    if not result:
        return {
            "submitted_ok": False,
            "status": "submit_failed",
            "note": "alpaca returned None",
            "fill_qty": 0,
            "fill_price": None,
            "order_id": "",
        }
    order_id = result.get("id", "")
    client_order_id = result.get("client_order_id", "")
    status = result.get("status")
    log.info(
        f"  US ORDER {label}: {side.upper()} {qty} {symbol} "
        f"id={order_id} status={status}"
    )

    # Poll for fill (IOC: typically resolves within seconds)
    fill_qty = 0
    fill_price: Optional[float] = None
    if order_id:
        for _ in range(ORDER_POLL_SECS):
            time.sleep(1)
            upd = alpaca_get_order(key, secret, order_id)
            if upd:
                status = upd.get("status", status)
                if upd.get("filled_qty"):
                    try:
                        fill_qty = int(float(upd["filled_qty"]))
                    except (TypeError, ValueError):
                        pass
                if upd.get("filled_avg_price"):
                    try:
                        fill_price = float(upd["filled_avg_price"])
                    except (TypeError, ValueError):
                        pass
                if status in {"filled", "canceled", "expired", "rejected"}:
                    break
    return {
        "submitted_ok": True,
        "status": status,
        "fill_qty": fill_qty,
        "fill_price": fill_price,
        "order_id": order_id,
        "client_order_id": client_order_id,
        "note": "",
    }


def log_order(
    conn: sqlite3.Connection,
    cycle: str,
    pair: str,
    leg: str,
    symbol: str,
    side: str,
    qty: int,
    result: dict,
    is_simulated: bool,
) -> None:
    conn.execute(
        """INSERT INTO orders (
            ts_utc, cycle, pair, leg, symbol, side, qty,
            submitted_ok, fill_price, fill_qty, status,
            order_id, client_order_id, is_simulated, note
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now(timezone.utc).isoformat(),
            cycle,
            pair,
            leg,
            symbol,
            side,
            qty,
            1 if result.get("submitted_ok") else 0,
            result.get("fill_price"),
            result.get("fill_qty"),
            result.get("status"),
            result.get("order_id", ""),
            result.get("client_order_id", ""),
            1 if is_simulated else 0,
            result.get("note", ""),
        ),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Combined execution (real US + virtual HK)
# ---------------------------------------------------------------------------
def execute_decision(
    key: str,
    secret: str,
    conn: sqlite3.Connection,
    cycle: str,
    d: PairDecision,
    virtual_hk: dict[str, VirtualHKPosition],
    log: logging.Logger,
    allow_skip_confirm: bool,
) -> None:
    """Translate one decision into real US order + virtual HK update."""
    if d.action in ("hold", "skip"):
        return

    # Confirmation gate
    cnt = load_confirm_counter()
    if cnt < CONFIRM_TRADE_COUNT:
        log.info(
            f"  Confirmation gate active (trade #{cnt + 1} of first "
            f"{CONFIRM_TRADE_COUNT})"
        )
        prompt = (
            f"{d.pair} action={d.action} dir={d.direction} "
            f"dev_bps={d.dev_bps:.1f} us_qty={d.us_qty} hk_qty={d.hk_qty} "
            f"us_px=${d.us_px:.2f} hk_px=HK${d.hk_px:.2f} — approve?"
        )
        if not interactive_confirm(prompt, allow_skip=allow_skip_confirm):
            log.warning(f"  Operator declined {d.pair} {d.action}")
            d.note = "operator_declined"
            return
        bump_confirm_counter()

    if d.action == "open":
        if d.direction == "short_us_long_hk":
            us_side, hk_side = "sell", "buy"
        else:
            us_side, hk_side = "buy", "sell"
        us_qty, hk_qty = d.us_qty, d.hk_qty
    elif d.action in ("close", "stop"):
        us_side = "buy" if d.us_qty < 0 else "sell"
        hk_side = "buy" if d.hk_qty < 0 else "sell"
        us_qty, hk_qty = abs(d.us_qty), abs(d.hk_qty)
    else:
        return

    # --- US leg: REAL Alpaca order ---
    if us_qty >= MIN_ORDER_QTY:
        us_res = execute_us_order(
            key, secret, d.us_sym, us_side, us_qty, d.us_px or 0.0,
            f"{d.pair}-{d.action}", log,
        )
    else:
        us_res = {
            "submitted_ok": False,
            "status": "skipped",
            "note": "qty<min",
            "fill_qty": 0,
            "fill_price": None,
            "order_id": "",
        }
    log_order(
        conn, cycle, d.pair, "US", d.us_sym, us_side, us_qty, us_res,
        is_simulated=False,
    )

    # --- HK leg: VIRTUAL update ---
    # Simulated fill price = the HK Close we used to compute the deviation.
    # Simulated fill qty = the spec qty (sim never partial-fills, never rejects).
    sim_status = "filled_sim"
    sim_fill_qty = hk_qty
    sim_fill_price = d.hk_px

    if d.action == "open":
        signed_qty = hk_qty if hk_side == "buy" else -hk_qty
        virtual_hk[d.pair] = VirtualHKPosition(
            pair=d.pair,
            hk_symbol=d.hk_sym,
            qty=signed_qty,
            avg_cost_hkd=float(d.hk_px or 0.0),
            opened_at_utc=datetime.now(timezone.utc).isoformat(),
            note=f"sim open dir={d.direction}",
        )
        record_virtual_hk_event(
            conn, d.pair, d.hk_sym, signed_qty, float(d.hk_px or 0.0),
            "open", note=f"dir={d.direction}",
        )
    else:  # close / stop
        # Calculate sim P&L on the way out for the log
        prev = virtual_hk.get(d.pair)
        if prev:
            sim_pnl_hkd = (
                (float(d.hk_px or 0.0) - prev.avg_cost_hkd) * prev.qty
            )
            sim_pnl_usd = sim_pnl_hkd * USD_PER_HKD
            log.info(
                f"  HK SIM CLOSE {d.pair}: qty={prev.qty} avg={prev.avg_cost_hkd:.4f} "
                f"exit={d.hk_px:.4f} sim_pnl_hkd={sim_pnl_hkd:+,.2f} "
                f"sim_pnl_usd={sim_pnl_usd:+,.2f}"
            )
            record_virtual_hk_event(
                conn, d.pair, d.hk_sym, -prev.qty, float(d.hk_px or 0.0),
                "close",
                note=f"sim_pnl_hkd={sim_pnl_hkd:.2f} sim_pnl_usd={sim_pnl_usd:.2f} "
                     f"action={d.action}",
            )
            virtual_hk.pop(d.pair, None)

    sim_res = {
        "submitted_ok": True,
        "status": sim_status,
        "fill_qty": sim_fill_qty,
        "fill_price": sim_fill_price,
        "order_id": "SIM",
        "client_order_id": f"sim-{d.pair}-{d.action}-{int(time.time())}",
        "note": "HK leg simulated — see banner for limitations",
    }
    log_order(
        conn, cycle, d.pair, "HK_SIM", d.hk_sym, hk_side, hk_qty, sim_res,
        is_simulated=True,
    )

    # Persist virtual HK state on every trade
    persist_virtual_hk_positions(virtual_hk)

    # If US leg failed (or partially filled) but HK virtual succeeded as-spec,
    # we now have a degenerate naked-or-oversized virtual HK position. Two cases:
    #   1. us_filled == 0          → revert virtual HK entirely
    #   2. 0 < us_filled < us_qty  → scale virtual HK to the actual US fill ratio
    us_filled = us_res.get("fill_qty", 0) or 0
    if d.action == "open" and us_qty > 0 and us_filled == 0:
        log.warning(
            f"  US leg for {d.pair} did NOT fill (status={us_res.get('status')}); "
            "reverting virtual HK open to keep books honest."
        )
        virtual_hk.pop(d.pair, None)
        record_virtual_hk_event(
            conn, d.pair, d.hk_sym, 0, 0.0, "close",
            note="auto-revert: US leg failed to fill",
        )
        persist_virtual_hk_positions(virtual_hk)
    elif d.action == "open" and us_qty > 0 and 0 < us_filled < us_qty:
        # Partial US fill — proportionally rescale virtual HK so the books
        # reflect actual exposure. Without this the virtual HK pretends a
        # full pair is open while the US leg is undersized → P&L mismark.
        fill_ratio = us_filled / us_qty
        prev = virtual_hk.get(d.pair)
        if prev:
            new_qty = int(round(prev.qty * fill_ratio))
            qty_delta = new_qty - prev.qty
            virtual_hk[d.pair] = VirtualHKPosition(
                pair=prev.pair,
                hk_symbol=prev.hk_symbol,
                qty=new_qty,
                avg_cost_hkd=prev.avg_cost_hkd,
                opened_at_utc=prev.opened_at_utc,
                note=(prev.note or "") + f" [scaled to {fill_ratio:.1%}]",
            )
            log.warning(
                f"  US leg for {d.pair} partially filled ({us_filled}/{us_qty}, "
                f"ratio={fill_ratio:.1%}); scaling virtual HK qty {prev.qty} -> {new_qty}."
            )
            record_virtual_hk_event(
                conn, d.pair, d.hk_sym, qty_delta, prev.avg_cost_hkd,
                "scale",
                note=f"partial US fill ratio={fill_ratio:.4f} prev_qty={prev.qty} new_qty={new_qty}",
            )
            persist_virtual_hk_positions(virtual_hk)


# ---------------------------------------------------------------------------
# Per-pair P&L (real US + sim HK)
# ---------------------------------------------------------------------------
def per_pair_pnl(
    st: PairState,
    us_px: Optional[float],
    hk_px_hkd: Optional[float],
    fx: float,
    alpaca_positions: list[dict],
) -> tuple[float, float]:
    """Return (real_us_pnl_usd, sim_hk_pnl_usd).

    The real US P&L comes from broker truth where available; otherwise computed
    locally. Sim HK P&L is always computed locally — it's a paper number.
    """
    if not st.open or us_px is None or hk_px_hkd is None:
        return 0.0, 0.0

    real_us_pnl = 0.0
    # Prefer broker-reported unrealised P&L (more accurate; includes corp actions)
    for p in alpaca_positions:
        if p.get("symbol") == st.us_sym:
            try:
                real_us_pnl = float(p.get("unrealized_pl", 0) or 0)
            except (TypeError, ValueError):
                real_us_pnl = (us_px - st.avg_us_cost) * st.us_qty
            break
    else:
        real_us_pnl = (us_px - st.avg_us_cost) * st.us_qty

    sim_hk_pnl_hkd = (hk_px_hkd - st.avg_hk_cost) * st.hk_qty
    sim_hk_pnl_usd = sim_hk_pnl_hkd / max(fx, 1e-9)
    return float(real_us_pnl), float(sim_hk_pnl_usd)


# ---------------------------------------------------------------------------
# Flatten (real US + virtual HK)
# ---------------------------------------------------------------------------
def flatten_all(
    key: str,
    secret: str,
    conn: sqlite3.Connection,
    cycle: str,
    pairs: list[tuple[str, str, int]],
    virtual_hk: dict[str, VirtualHKPosition],
    log: logging.Logger,
) -> None:
    log.warning("FLATTEN ALL: closing every iter_18 ADR Alpaca position.")
    alpaca_positions = alpaca_list_positions(key, secret)
    alpaca_by_symbol = {p["symbol"]: p for p in alpaca_positions}

    for us, hk, _ratio in pairs:
        pair_key = f"{us}/{hk}"

        # Real US flatten
        if us in alpaca_by_symbol:
            try:
                qty = int(round(float(alpaca_by_symbol[us].get("qty", 0))))
            except (TypeError, ValueError):
                qty = 0
            if qty != 0:
                side = "buy" if qty < 0 else "sell"
                px = alpaca_get_latest_trade(key, secret, us) or 1.0
                r = execute_us_order(
                    key, secret, us, side, abs(qty), px, f"flatten-{us}", log,
                )
                log_order(
                    conn, cycle, pair_key, "US", us, side, abs(qty), r,
                    is_simulated=False,
                )

        # Virtual HK flatten
        if pair_key in virtual_hk:
            prev = virtual_hk[pair_key]
            if prev.qty != 0:
                exit_px = fetch_hk_close_yf(hk, log) or prev.avg_cost_hkd
                sim_pnl_hkd = (exit_px - prev.avg_cost_hkd) * prev.qty
                sim_pnl_usd = sim_pnl_hkd * USD_PER_HKD
                hk_side = "buy" if prev.qty < 0 else "sell"
                record_virtual_hk_event(
                    conn, pair_key, hk, -prev.qty, exit_px, "close",
                    note=f"FLATTEN sim_pnl_usd={sim_pnl_usd:.2f}",
                )
                log_order(
                    conn, cycle, pair_key, "HK_SIM", hk, hk_side, abs(prev.qty),
                    {
                        "submitted_ok": True,
                        "status": "filled_sim",
                        "fill_qty": abs(prev.qty),
                        "fill_price": exit_px,
                        "order_id": "SIM",
                        "client_order_id": f"sim-flatten-{pair_key}",
                        "note": f"flatten sim_pnl_usd={sim_pnl_usd:.2f}",
                    },
                    is_simulated=True,
                )
                virtual_hk.pop(pair_key, None)

    persist_virtual_hk_positions(virtual_hk)


# ---------------------------------------------------------------------------
# Position snapshot persistence
# ---------------------------------------------------------------------------
def write_positions_snapshot(
    conn: sqlite3.Connection,
    cycle: str,
    alpaca_positions: list[dict],
    virtual_hk: dict[str, VirtualHKPosition],
    hk_closes: dict[str, Optional[float]],
) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    for p in alpaca_positions:
        try:
            qty = float(p.get("qty", 0) or 0)
            avg = float(p.get("avg_entry_price", 0) or 0)
            mv = float(p.get("market_value", 0) or 0)
            upnl = float(p.get("unrealized_pl", 0) or 0)
        except (TypeError, ValueError):
            qty = avg = mv = upnl = 0.0
        conn.execute(
            """INSERT INTO positions_snapshot
               (ts_utc, cycle, symbol, is_simulated, qty, avg_cost,
                market_value, unreal_pnl, note)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (ts, cycle, p.get("symbol", ""), 0, qty, avg, mv, upnl,
             "real alpaca"),
        )
    for pair, v in virtual_hk.items():
        exit_px = hk_closes.get(v.hk_symbol) or v.avg_cost_hkd
        mv_hkd = exit_px * v.qty
        mv_usd = mv_hkd * USD_PER_HKD
        upnl_hkd = (exit_px - v.avg_cost_hkd) * v.qty
        upnl_usd = upnl_hkd * USD_PER_HKD
        conn.execute(
            """INSERT INTO positions_snapshot
               (ts_utc, cycle, symbol, is_simulated, qty, avg_cost,
                market_value, unreal_pnl, note)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (ts, cycle, v.hk_symbol, 1, float(v.qty), v.avg_cost_hkd,
             mv_usd, upnl_usd,
             f"virtual HK sim — DO NOT trust as real broker truth"),
        )
    conn.commit()


# ---------------------------------------------------------------------------
# EOD summary
# ---------------------------------------------------------------------------
def write_eod_summary(
    conn: sqlite3.Connection,
    cycle: str,
    nav_usd: float,
    states: dict[str, PairState],
    decisions: list[PairDecision],
    real_pnls: dict[str, float],
    sim_pnls: dict[str, float],
    log: logging.Logger,
) -> None:
    open_count = sum(1 for s in states.values() if s.open)
    gross = 0.0
    for st in states.values():
        gross += abs(st.us_qty * st.avg_us_cost)
        gross += abs(st.hk_qty * st.avg_hk_cost) * USD_PER_HKD

    payload = {
        "pairs": {
            d.pair: {
                "dev_bps": None if d.dev_bps is None else round(d.dev_bps, 2),
                "action": d.action,
                "direction": d.direction,
                "us_px": d.us_px,
                "hk_px": d.hk_px,
                "ex_div_block": d.ex_div_block,
                "real_us_pnl_usd": round(real_pnls.get(d.pair, 0.0), 2),
                "sim_hk_pnl_usd": round(sim_pnls.get(d.pair, 0.0), 2),
                "combined_pnl_usd": round(
                    real_pnls.get(d.pair, 0.0) + sim_pnls.get(d.pair, 0.0), 2
                ),
                "reason": d.reason,
            }
            for d in decisions
        },
    }
    total_real = sum(real_pnls.values())
    total_sim = sum(sim_pnls.values())
    conn.execute(
        """INSERT INTO eod_summary (
            ts_utc, cycle, nav_usd, open_pair_count, gross_exposure,
            real_us_pnl, sim_hk_pnl, total_combined_pnl, payload_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now(timezone.utc).isoformat(),
            cycle,
            nav_usd,
            open_count,
            gross,
            total_real,
            total_sim,
            total_real + total_sim,
            json.dumps(payload, default=str),
        ),
    )
    conn.commit()
    log.info(
        f"EOD ({cycle}): NAV=${nav_usd:,.2f} open_pairs={open_count} "
        f"gross=${gross:,.2f} real_us_pnl=${total_real:+,.2f} "
        f"sim_hk_pnl=${total_sim:+,.2f} "
        f"combined=${total_real + total_sim:+,.2f} "
        f"[REMINDER: HK side is sim — do NOT extrapolate to live]"
    )


# ---------------------------------------------------------------------------
# One full cycle
# ---------------------------------------------------------------------------
def run_cycle(
    key: str,
    secret: str,
    conn: sqlite3.Connection,
    cycle: str,
    pairs: list[tuple[str, str, int]],
    cal: dict[str, set],
    log: logging.Logger,
    allow_skip_confirm: bool,
    dry_run: bool,
) -> None:
    log.info(f"=== Cycle start: {cycle} dry_run={dry_run} ===")

    # Kill flag
    kf = kill_flag_present()
    if kf is not None:
        raise KillSwitch(
            f"STOP_ADR_ALPACA file present: {kf.get('reason')} / {kf.get('detail')}"
        )

    # NAV + positions (real, broker truth) + virtual HK (sim)
    nav_usd = fetch_nav_usd(key, secret, log)
    persist_nav(nav_usd)
    alpaca_positions = alpaca_list_positions(key, secret)
    virtual_hk = load_virtual_hk_positions()
    states = reconstruct_pair_states(pairs, alpaca_positions, virtual_hk)

    # FX: peg
    fx = USD_HKD_PEG
    log.info(f"FX USD/HKD = {fx:.4f} (peg)")

    # Pull prices
    us_prices: dict[str, Optional[float]] = {}
    for us, _hk, _ratio in pairs:
        us_prices[us] = alpaca_get_latest_trade(key, secret, us)
    hk_symbols = [hk for _us, hk, _ratio in pairs]
    hk_closes = fetch_hk_closes_batch(hk_symbols, log)
    write_positions_snapshot(conn, cycle, alpaca_positions, virtual_hk, hk_closes)

    # Decide per pair
    today = pd.Timestamp.today().normalize()
    decisions: list[PairDecision] = []
    real_pnls: dict[str, float] = {}
    sim_pnls: dict[str, float] = {}
    for us, hk, ratio in pairs:
        st = states[f"{us}/{hk}"]
        blocked = is_ex_div_blocked(today, us, hk, cal)
        d = decide_pair(
            st,
            us_prices.get(us),
            hk_closes.get(hk),
            fx,
            blocked,
            nav_usd,
        )
        decisions.append(d)
        log_decision(conn, cycle, d)
        rp, sp = per_pair_pnl(
            st, us_prices.get(us), hk_closes.get(hk), fx, alpaca_positions,
        )
        real_pnls[d.pair] = rp
        sim_pnls[d.pair] = sp
        log.info(
            f"  {d.pair}: dev_bps="
            + (f"{d.dev_bps:+.1f}" if d.dev_bps is not None else "NA")
            + f" action={d.action} blocked={d.ex_div_block} dir={d.direction} "
            f"real_us_pnl=${rp:+,.2f} sim_hk_pnl=${sp:+,.2f} reason={d.reason}"
        )

    # Risk gates BEFORE we trade
    open_count = sum(1 for s in states.values() if s.open)
    will_open = sum(1 for d in decisions if d.action == "open")
    if open_count + will_open > MAX_OPEN_PAIRS:
        log.warning(
            f"Open+new ({open_count + will_open}) exceeds MAX_OPEN_PAIRS="
            f"{MAX_OPEN_PAIRS}; deferring entries beyond cap."
        )
        slots = max(MAX_OPEN_PAIRS - open_count, 0)
        for d in decisions:
            if d.action == "open" and slots <= 0:
                d.action = "skip"
                d.reason = "cap_reached"
            elif d.action == "open":
                slots -= 1

    # Hard kill: per-pair simulated loss > 5% NAV
    for d in decisions:
        combined = real_pnls.get(d.pair, 0.0) + sim_pnls.get(d.pair, 0.0)
        if combined < -PER_PAIR_LOSS_KILL_FRAC * nav_usd:
            raise KillSwitch(
                f"Per-pair combined loss for {d.pair} = ${combined:,.2f} "
                f"exceeds {PER_PAIR_LOSS_KILL_FRAC * 100:.0f}% of NAV "
                f"${nav_usd:,.2f}"
            )

    # Hard kill: total real-capital gross exposure > 100% NAV.
    # Virtual HK does not consume real margin, so it must NOT be counted toward
    # the kill threshold — otherwise asymmetric US-fail / HK-sim-success cycles
    # double-count notional and trigger false-positive kills (see 2026-05-13).
    gross = 0.0
    for st in states.values():
        gross += abs(st.us_qty * st.avg_us_cost)
    if gross > MAX_TOTAL_GROSS_NAV_FRAC * nav_usd:
        raise KillSwitch(
            f"Real US gross exposure ${gross:,.2f} > {MAX_TOTAL_GROSS_NAV_FRAC * 100:.0f}% "
            f"of NAV ${nav_usd:,.2f}"
        )

    # Execute
    if dry_run:
        log.info("DRY RUN — skipping order placement and virtual HK updates.")
    else:
        for d in decisions:
            if d.action in ("open", "close", "stop"):
                execute_decision(
                    key, secret, conn, cycle, d, virtual_hk, log,
                    allow_skip_confirm,
                )

    # EOD summary
    write_eod_summary(
        conn, cycle, nav_usd, states, decisions, real_pnls, sim_pnls, log,
    )
    log.info(f"=== Cycle done: {cycle} ===")


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------
def _signal_handler(_signum, _frame) -> None:
    global STOPPED
    STOPPED = True
    logger.info("stop signal received")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="iter_18 ADR paper-shadow host (Alpaca US real + HK sim)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--account",
        default=os.environ.get("ALPACA_PAPER_ACCOUNT", "AEGIS"),
        help="Alpaca paper account nickname (default: AEGIS). Reads "
             "ALPACA_<NAME>_KEY/SECRET or falls back to ALPACA_API_KEY/SECRET.",
    )
    p.add_argument(
        "--cycle",
        choices=["hk_close", "us_close", "adhoc"],
        default="adhoc",
        help="Which scheduled cycle this invocation represents.",
    )
    p.add_argument(
        "--universe",
        choices=["core", "expanded"],
        default="core",
        help="core = 5 pairs from iter_18_cleanup; expanded = 12 pairs "
             "from iter_18_expanded (YMM/2643 excluded — failed).",
    )
    p.add_argument(
        "--confirm",
        action="store_true",
        help=f"Require interactive [y/N] approval for the first "
             f"{CONFIRM_TRADE_COUNT} live trades.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute decisions, log to DB, but DO NOT place US orders or "
             "update virtual HK positions.",
    )
    p.add_argument(
        "--skip-backtest-check",
        action="store_true",
        help="Skip the 60-day backtest divergence verification on startup.",
    )
    p.add_argument(
        "--flatten-now",
        action="store_true",
        help="Immediately flatten all iter_18 positions (real US + virtual HK) "
             "and exit.",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    log = setup_logging()
    log.info(f"=== iter_18 ADR Alpaca host start (argv={sys.argv}) ===")

    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    load_env()

    # Pick universe
    pairs = EXPANDED_PAIRS if args.universe == "expanded" else CORE_PAIRS
    print_startup_banner(log, args.universe, len(pairs))

    try:
        key, secret = resolve_credentials(args.account)
    except SystemExit as exc:
        log.error(str(exc))
        return 2

    # Backtest divergence verification
    if not args.skip_backtest_check:
        try:
            verify_backtest_divergence(log, args.universe)
        except SystemExit as exc:
            log.error(str(exc))
            return 2

    # Pre-existing kill flag => refuse to start
    kf = kill_flag_present()
    if kf is not None:
        log.error(
            f"STOP_ADR_ALPACA present at startup: {kf}. "
            f"Remove {KILL_FLAG_PATH} after investigating before re-running."
        )
        return 4

    conn = open_db()
    cal = load_dividend_calendar(log, args.universe)

    # Validate that all US symbols are tradable
    untradable = []
    for us, _hk, _ratio in pairs:
        if not alpaca_get_asset_tradable(key, secret, us):
            untradable.append(us)
    if untradable:
        log.error(
            f"The following US symbols are not tradable on Alpaca paper: "
            f"{untradable}. Investigate before continuing."
        )
        # Don't hard-fail — some pairs may still be tradable. Just warn.

    log.info(f"Account: {args.account} cycle={args.cycle} universe={args.universe}")

    try:
        virtual_hk = load_virtual_hk_positions()

        if args.flatten_now:
            flatten_all(key, secret, conn, args.cycle, pairs, virtual_hk, log)
            log.info("Flatten complete; exiting.")
            return 0

        run_cycle(
            key=key,
            secret=secret,
            conn=conn,
            cycle=args.cycle,
            pairs=pairs,
            cal=cal,
            log=log,
            allow_skip_confirm=not args.confirm,
            dry_run=args.dry_run,
        )
        return 0

    except KillSwitch as kill:
        log.error(f"KILL SWITCH FIRED: {kill}")
        write_kill_flag("hard_kill", str(kill))
        record_kill(conn, args.cycle, "hard_kill", str(kill))
        try:
            virtual_hk = load_virtual_hk_positions()
            flatten_all(key, secret, conn, args.cycle, pairs, virtual_hk, log)
        except Exception as exc:  # noqa: BLE001
            log.error(f"Flatten-on-kill failed: {exc}")
        return 9

    except Exception as exc:  # noqa: BLE001
        log.exception(f"Unhandled error: {exc}")
        record_kill(conn, args.cycle, "unhandled_exception", repr(exc))
        return 5

    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass
        log.info("=== iter_18 ADR Alpaca host exit ===")


if __name__ == "__main__":
    sys.exit(main())
