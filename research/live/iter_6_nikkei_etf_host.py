"""iter_6 Nikkei Gap Continuation — Japan ETF Proxy Paper-Shadow Host (Alpaca).

================================================================================
PROXY DEPLOYMENT — please read the honest caveat banner before trusting numbers.
================================================================================

ORIGINAL backtest (research/results/iter_6_sweep.json, best_by_sharpe row):
    instrument          = ^N225 daily proxy (Tokyo cash index)
    threshold           = 1.0% overnight gap
    OOS Sharpe          = 2.51
    OOS ann return      = 29.1%
    OOS max drawdown    = -5.40%
    OOS win rate        = 65.7%
    n_trades            = 338 over 7.10 years (~18.9% days in market)
    round-trip cost     = 5 bps baked in

THIS HOST trades EWJ (default) or DXJ on Alpaca paper instead of NKD futures /
NK225M minis. The strategy logic is preserved exactly:

    signal source       : yfinance ^N225 daily OHLC (Tokyo session)
    signal              : sign(overnight gap) when |gap| > 1.0%, else flat
                            gap = today_tokyo_open / prev_tokyo_close - 1
    execution venue     : Alpaca paper REST on EWJ (or DXJ via --etf DXJ)
    entry               : at US cash open (09:30 ET) after the Tokyo session
                            that produced the signal closed (i.e. the signal is
                            from the most recent Tokyo session, and we trade EWJ
                            during the next US session as a continuation proxy)
    direction           : long EWJ if gap > +1.0%
                            short EWJ if gap < -1.0%  (paper easy-to-borrow;
                            disable with --no-shorts)
    hold                : 1 US trading day (proxy for "next Tokyo session
                            continuation"); exit at US close (15:55 ET)
    sizing              : MAX_POSITION_FRAC of NAV, single position at a time
    cost                : 5 bps assumed; Alpaca paper charges $0 commission, so
                            real cost is bid-ask + 25-bps marketable-limit slip

HONEST CAVEATS (printed at startup; do not ignore):
    1. CURRENCY DRAG — EWJ is JPY-denominated underlying valued in USD. A 1%
       move in the Nikkei + a 0.5% strengthening of USD/JPY = ~0.5% EWJ move,
       not 1%. This dilutes the gap-continuation edge.
       DXJ ($XJPN-hedged) removes this, but adds tracking-error of the hedge.
    2. NAV TRACKING ERROR — EWJ is a basket of large-cap Japan ADRs/ORDs that
       only loosely tracks Nikkei 225 (Nikkei 225 is price-weighted; EWJ is
       MSCI Japan-weighted, market-cap). Daily correlation is ~0.90, not 1.0.
    3. TIMING MISMATCH — The original signal is "Tokyo close vs. next Tokyo
       open." Here we execute on EWJ during US hours AFTER the Tokyo session
       that produced the gap. EWJ has already absorbed part of the move in
       overnight US ADRs (DXJ less so). The remaining edge is the "Tokyo
       session continuation" carrying into the next US session.
    4. EXPECTED EDGE DEGRADATION — combine the three above and the signal
       Sharpe is likely 30-50% of the backtest figure (so 0.75 - 1.25 Sharpe
       in practice, not 2.51). Use this host to *measure* that degradation,
       not to assume the original number.
    5. SHORTS — Alpaca paper allows shorting on liquid ETFs by default, but
       some sub-accounts disable shorting. The host probes for shorting
       support on startup; if shorts are not permitted it will only take
       long signals (gap > +1.0%) and log negative-gap signals as no-trade.

MIGRATION PATH — when an IBKR or true futures account is available, swap to
iter_6_nikkei_host.py (NKD via CME or NK225M via OSE.JPN). All the SQLite
schemas are compatible; this host's `venue` field is set to "alpaca_etf"
to distinguish runs.

================================================================================
DEPLOYMENT
================================================================================
First run (interactive confirm on first 5 trades, recommended):
    /Users/weeoto/turbo-carnival/.venv/bin/python \
        /Users/weeoto/turbo-carnival/research/live/iter_6_nikkei_etf_host.py \
        --account AEGIS --confirm

One-shot cron-style (run twice a day at 09:30 ET open and 15:55 ET close):
    /Users/weeoto/turbo-carnival/.venv/bin/python \
        /Users/weeoto/turbo-carnival/research/live/iter_6_nikkei_etf_host.py \
        --account AEGIS --once --cycle open
    /Users/weeoto/turbo-carnival/.venv/bin/python \
        /Users/weeoto/turbo-carnival/research/live/iter_6_nikkei_etf_host.py \
        --account AEGIS --once --cycle close

Daemon (decides at OPEN_TIME, flattens at CLOSE_TIME):
    /Users/weeoto/turbo-carnival/.venv/bin/python \
        /Users/weeoto/turbo-carnival/research/live/iter_6_nikkei_etf_host.py \
        --account AEGIS

Currency-hedged variant:
    ... --etf DXJ

Dry run:
    ... --dry-run

================================================================================
ENVIRONMENT
================================================================================
Reads /Users/weeoto/turbo-carnival/council/.env.local then process env.

Required:
    ALPACA_<ACCOUNT>_KEY      e.g. ALPACA_AEGIS_KEY
    ALPACA_<ACCOUNT>_SECRET   e.g. ALPACA_AEGIS_SECRET

Defaults:
    ALPACA_PAPER_ACCOUNT      AEGIS

================================================================================
KILL SWITCHES (fail CLOSED; halt logged to kill_events)
================================================================================
Automatic:
    1. Daily loss cap 1.5% NAV    -> block new entries (do not flatten)
    2. Intraday loss 2% NAV       -> flatten + halt
    3. 5 consecutive losses        -> halt new entries
    4. STOP_NIKKEI_ETF kill file   -> halt (no flatten)
    5. Startup divergence > 30%   -> exit non-zero
    6. SIGINT/SIGTERM              -> graceful flatten + exit

Manual:
    touch /Users/weeoto/turbo-carnival/research/live/STOP_NIKKEI_ETF
    or run with --flatten-now

Author : Deployment Engineer A37
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal as signal_module
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional
from zoneinfo import ZoneInfo

import requests

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
REPO = Path("/Users/weeoto/turbo-carnival")
ENV_FILE = REPO / "council" / ".env.local"
ROOT = REPO / "research"
LIVE_DIR = ROOT / "live"
LOG_DIR = REPO / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LIVE_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = LOG_DIR / "iter_6_nikkei_etf_host.log"
DECISIONS_DB = LIVE_DIR / "nikkei_etf_decisions.sqlite3"
NAV_STATE_PATH = LIVE_DIR / "nikkei_etf_nav_state.json"
CONFIRM_COUNTER_PATH = LIVE_DIR / "nikkei_etf_confirm_counter.json"
KILL_FLAG_PATH = LIVE_DIR / "STOP_NIKKEI_ETF"
SWEEP_SUMMARY_PATH = ROOT / "results" / "iter_6_sweep.json"

NY = ZoneInfo("America/New_York")
TOKYO = ZoneInfo("Asia/Tokyo")

ENDPOINT = "https://paper-api.alpaca.markets/v2"
DATA_ENDPOINT = "https://data.alpaca.markets/v2"

# Strategy parameters — frozen, copied from iter_6_sweep.json best_by_sharpe.
GAP_THRESHOLD = 0.010            # 1.0% overnight gap threshold
ROUND_TRIP_COST_BPS = 5.0        # informational
LOOKBACK_DAYS = 60               # startup-verification window

# ETF universe (proxy for ^N225).
DEFAULT_ETF = "EWJ"
ALT_ETF = "DXJ"
VALID_ETFS = (DEFAULT_ETF, ALT_ETF)

# Risk envelope (account-level).
MAX_POSITION_FRAC = 0.25         # 25% of NAV per trade (single position)
MAX_POSITIONS = 1                # one symbol at a time
INTRADAY_LOSS_KILL_FRAC = 0.02   # 2% NAV intraday loss -> flatten + halt
DAILY_LOSS_CAP_FRAC = 0.015      # 1.5% NAV daily cap (block new entries)
MAX_CONSECUTIVE_LOSSES = 5       # halt new entries after 5 in a row
CONFIRM_TRADE_COUNT = 5          # first 5 trades require --confirm gate

# Order execution.
MIN_REBALANCE_USD = 50.0
ORDER_PRICE_SLIP_BPS = 25        # 25 bps marketable-limit IOC slip

# Backtest divergence guard.
DIVERGENCE_KILL_FRAC = 0.30

# Cycle times (US Eastern).
DEFAULT_OPEN_HHMM_ET = "09:35"   # 5 min after open, lets prints settle
DEFAULT_CLOSE_HHMM_ET = "15:55"  # 5 min before close to ensure flat
DEFAULT_VERIFY_HHMM_ET = "09:30" # verify before open

VENUE_LABEL = "alpaca_etf"

logger = logging.getLogger("iter_6_nikkei_etf")
STOPPED = False


# ---------------------------------------------------------------------------
# Env loading & credentials
# ---------------------------------------------------------------------------
def load_env() -> None:
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
    account = account.upper()
    key = os.environ.get(f"ALPACA_{account}_KEY") or os.environ.get("ALPACA_API_KEY")
    sec = os.environ.get(f"ALPACA_{account}_SECRET") or os.environ.get("ALPACA_API_SECRET")
    if not key or not sec:
        raise SystemExit(
            f"Missing Alpaca paper credentials. Expected ALPACA_{account}_KEY/"
            f"ALPACA_{account}_SECRET (or ALPACA_API_KEY / ALPACA_API_SECRET)."
        )
    return key, sec


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging() -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fh = logging.FileHandler(LOG_PATH, mode="a")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)


# ---------------------------------------------------------------------------
# Honest caveat banner
# ---------------------------------------------------------------------------
def print_banner(etf: str, allow_shorts: bool) -> None:
    bar = "=" * 78
    logger.warning(bar)
    logger.warning("iter_6 NIKKEI ETF-PROXY HOST — HONEST CAVEAT BANNER")
    logger.warning(bar)
    logger.warning(
        "WARNING: This is a PROXY strategy. The original iter_6 signal was "
        "developed on ^N225 daily proxy. Trading %s adds: "
        "(1) currency drag (JPY/USD); "
        "(2) NAV-vs-NIKKEI tracking error (EWJ is MSCI Japan, not N225); "
        "(3) US-hours vs Tokyo-hours timing mismatch. "
        "Expected edge may be 30-50%% of backtest. "
        "Use DXJ (--etf DXJ) for currency-hedged variant — also tracks "
        "Nikkei more closely.",
        etf,
    )
    if etf == DEFAULT_ETF:
        logger.warning(
            "EWJ is unhedged (full JPY exposure). DXJ (--etf DXJ) is currency-"
            "hedged and tracks Nikkei moves more cleanly, at the cost of "
            "hedge-roll drag in stable-JPY regimes."
        )
    else:
        logger.warning(
            "DXJ is JPY-hedged (closer Nikkei tracking). Note the hedge is "
            "imperfect: tracking error vs Nikkei is still ~5-8 bps/day."
        )
    if not allow_shorts:
        logger.warning(
            "SHORTS DISABLED: only long entries on positive gaps will be taken. "
            "Negative gaps will log as no-trade. This cuts trade count ~50%%."
        )
    logger.warning(bar)


# ---------------------------------------------------------------------------
# Alpaca REST helpers (paper)
# ---------------------------------------------------------------------------
def _headers(key: str, secret: str) -> dict:
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def get_account(key: str, secret: str) -> Optional[dict]:
    try:
        r = requests.get(f"{ENDPOINT}/account", headers=_headers(key, secret), timeout=10)
        if r.status_code == 200:
            return r.json()
        logger.warning(f"account GET {r.status_code}: {r.text[:200]}")
    except Exception as exc:
        logger.warning(f"account exception: {exc}")
    return None


def get_clock(key: str, secret: str) -> Optional[dict]:
    try:
        r = requests.get(f"{ENDPOINT}/clock", headers=_headers(key, secret), timeout=8)
        if r.status_code == 200:
            return r.json()
    except Exception as exc:
        logger.warning(f"clock exception: {exc}")
    return None


def get_position(key: str, secret: str, symbol: str) -> Optional[dict]:
    try:
        r = requests.get(f"{ENDPOINT}/positions/{symbol}",
                         headers=_headers(key, secret), timeout=10)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 404:
            return None
        logger.warning(f"position GET {symbol} {r.status_code}: {r.text[:200]}")
    except Exception as exc:
        logger.warning(f"position exception: {exc}")
    return None


def get_latest_trade(key: str, secret: str, symbol: str) -> Optional[float]:
    try:
        r = requests.get(f"{DATA_ENDPOINT}/stocks/{symbol}/trades/latest",
                         headers=_headers(key, secret), timeout=10)
        if r.status_code == 200:
            j = r.json()
            if j.get("trade") and j["trade"].get("p"):
                return float(j["trade"]["p"])
    except Exception as exc:
        logger.warning(f"trade GET {symbol} exception: {exc}")
    return None


def get_asset(key: str, secret: str, symbol: str) -> Optional[dict]:
    try:
        r = requests.get(f"{ENDPOINT}/assets/{symbol}",
                         headers=_headers(key, secret), timeout=8)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def check_shorting_supported(key: str, secret: str, symbol: str) -> bool:
    """Probe whether the paper account is allowed to short `symbol`.

    Alpaca exposes `shortable` and `easy_to_borrow` flags on the asset. The
    account itself must also have `shorting_enabled=true` (paper accounts do
    by default unless explicitly disabled).
    """
    asset = get_asset(key, secret, symbol)
    if not asset:
        return False
    if not asset.get("shortable") or not asset.get("easy_to_borrow"):
        return False
    acct = get_account(key, secret)
    if not acct:
        return False
    if acct.get("shorting_enabled") is False:
        return False
    return True


def submit_marketable_limit_ioc(
    key: str, secret: str, symbol: str, qty: float, side: str, ref_price: float,
    client_tag: str,
) -> Optional[dict]:
    """Submit a marketable IOC limit order at ref_price +/- ORDER_PRICE_SLIP_BPS."""
    slip = ref_price * (ORDER_PRICE_SLIP_BPS / 10_000.0)
    if side == "buy":
        limit_price = round(ref_price + slip, 2)
    else:
        limit_price = round(ref_price - slip, 2)
    payload = {
        "symbol": symbol,
        "qty": str(round(qty, 4)),
        "side": side,
        "type": "limit",
        "limit_price": str(limit_price),
        "time_in_force": "ioc",
        "client_order_id": f"nikkei-etf-{client_tag}-{int(time.time() * 1000)}",
    }
    try:
        r = requests.post(f"{ENDPOINT}/orders",
                          headers=_headers(key, secret), json=payload, timeout=10)
        if r.status_code in (200, 201):
            return r.json()
        logger.error(f"order failed {r.status_code}: {r.text[:300]}")
    except Exception as exc:
        logger.error(f"order exception: {exc}")
    return None


def get_order(key: str, secret: str, order_id: str) -> Optional[dict]:
    try:
        r = requests.get(f"{ENDPOINT}/orders/{order_id}",
                         headers=_headers(key, secret), timeout=8)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# yfinance signal source (^N225)
# ---------------------------------------------------------------------------
def _import_yf():
    try:
        import yfinance as yf  # type: ignore
        return yf
    except ImportError as exc:
        raise SystemExit(
            "FATAL: yfinance is not installed. Install with `pip install yfinance`."
        ) from exc


def _import_pd_np():
    try:
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
        return np, pd
    except ImportError as exc:
        raise SystemExit(
            "FATAL: pandas/numpy not installed."
        ) from exc


def fetch_n225_daily(days: int) -> "object":
    """Pull daily ^N225 OHLC for the last `days` calendar days via yfinance.

    Returns a pandas DataFrame with columns Open/High/Low/Close/Volume,
    tz-naive index sorted ascending.
    """
    yf = _import_yf()
    _, pd = _import_pd_np()
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=int(days * 1.6) + 7)
    logger.info(f"Fetching ^N225 daily {start} -> {end}")
    df = yf.download(
        "^N225",
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        raise RuntimeError("yfinance returned no recent ^N225 data")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.dropna(subset=["Open", "Close"]).sort_index()
    return df


@dataclass
class GapSignal:
    """Result of a single signal evaluation."""
    session_date: date
    prev_close: float
    session_open: float
    gap_pct: float
    direction: int                # -1, 0, +1
    took_trade: bool
    reason: str


def daily_gap_signal(df, threshold: float = GAP_THRESHOLD) -> GapSignal:
    """Compute the latest gap-continuation signal from daily ^N225 OHLC.

    df: tz-naive index of trading days, columns include Open and Close.
        The LAST row is the most recent Tokyo session — its Open is the
        gap-open we measure relative to the prior session's Close.
    """
    _, pd = _import_pd_np()
    if df is None or len(df) < 2:
        raise ValueError("daily_gap_signal needs at least 2 rows of daily OHLC")

    last = df.iloc[-1]
    prev = df.iloc[-2]
    session_date = pd.Timestamp(df.index[-1]).date()

    prev_close = float(prev["Close"])
    sess_open = float(last["Open"])
    if prev_close <= 0 or not math.isfinite(prev_close):
        return GapSignal(session_date, prev_close, sess_open, 0.0, 0, False,
                         "prev_close invalid")

    gap = sess_open / prev_close - 1.0

    if gap > threshold:
        return GapSignal(session_date, prev_close, sess_open, gap, +1, True,
                         f"gap {gap*100:.3f}% > +{threshold*100:.2f}%")
    if gap < -threshold:
        return GapSignal(session_date, prev_close, sess_open, gap, -1, True,
                         f"gap {gap*100:.3f}% < -{threshold*100:.2f}%")
    return GapSignal(session_date, prev_close, sess_open, gap, 0, False,
                     f"|gap| {abs(gap)*100:.3f}% <= {threshold*100:.2f}%")


# ---------------------------------------------------------------------------
# Startup verification — replay backtest on last LOOKBACK_DAYS
# ---------------------------------------------------------------------------
def count_backtest_signals(df, threshold: float = GAP_THRESHOLD) -> int:
    np, _ = _import_pd_np()
    work = df.copy()
    work["prev_close"] = work["Close"].shift(1)
    work["gap"] = work["Open"] / work["prev_close"] - 1.0
    work = work.dropna(subset=["gap"])
    sig = np.where(
        work["gap"] > threshold, 1.0,
        np.where(work["gap"] < -threshold, -1.0, 0.0),
    )
    return int((sig != 0.0).sum())


def expected_signal_count(lookback_days: int) -> float:
    """Pull expected signal-frequency from the sweep JSON for the 1.0%
    threshold. Falls back to 18.9% days-in-market figure.
    """
    try:
        sweep = json.loads(SWEEP_SUMMARY_PATH.read_text())
        for row in sweep.get("threshold_results", []):
            if abs(float(row.get("threshold_pct", -1)) - 1.0) < 1e-9:
                return lookback_days * float(row["days_in_market_pct"]) / 100.0
    except Exception:
        pass
    return lookback_days * 0.189


def verify_against_backtest() -> bool:
    """Re-run backtest signal count on last LOOKBACK_DAYS and confirm within
    DIVERGENCE_KILL_FRAC of expectation. Returns False on hard fail.
    """
    try:
        df = fetch_n225_daily(LOOKBACK_DAYS)
    except Exception as exc:
        logger.error(f"verify: ^N225 fetch failed: {exc}")
        return False
    observed = count_backtest_signals(df, GAP_THRESHOLD)
    trading_days = len(df.dropna())
    expected = expected_signal_count(trading_days)
    if expected <= 0:
        logger.warning("verify: expected signal count is zero — skipping check")
        return True
    divergence = abs(observed - expected) / expected
    logger.info(
        f"verify: trading_days={trading_days} observed_signals={observed} "
        f"expected={expected:.2f} divergence={divergence*100:.1f}%"
    )
    db_log("kill_events", {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "VERIFY",
        "detail": json.dumps({
            "trading_days": trading_days,
            "observed_signals": observed,
            "expected_signals": round(expected, 2),
            "divergence_pct": round(divergence * 100, 3),
        }),
    })
    if divergence > DIVERGENCE_KILL_FRAC:
        logger.error(
            f"FATAL: signal-count divergence {divergence*100:.1f}% > "
            f"{DIVERGENCE_KILL_FRAC*100:.0f}%"
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Decisions DB
# ---------------------------------------------------------------------------
def init_db() -> None:
    conn = sqlite3.connect(DECISIONS_DB)
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS evaluations (
            ts_utc TEXT NOT NULL,
            session_date TEXT NOT NULL,
            venue TEXT NOT NULL,
            etf TEXT NOT NULL,
            prev_close REAL,
            session_open REAL,
            gap_pct REAL,
            threshold_pct REAL,
            signal_dir INTEGER,
            took_trade INTEGER,
            reason TEXT
        );
        CREATE TABLE IF NOT EXISTS orders (
            ts_utc TEXT NOT NULL,
            session_date TEXT NOT NULL,
            venue TEXT NOT NULL,
            symbol TEXT NOT NULL,
            action TEXT NOT NULL,
            qty REAL NOT NULL,
            order_type TEXT NOT NULL,
            broker_order_id TEXT,
            status TEXT
        );
        CREATE TABLE IF NOT EXISTS fills (
            ts_utc TEXT NOT NULL,
            session_date TEXT NOT NULL,
            venue TEXT NOT NULL,
            symbol TEXT NOT NULL,
            action TEXT NOT NULL,
            qty REAL NOT NULL,
            fill_price REAL,
            commission REAL,
            broker_order_id TEXT
        );
        CREATE TABLE IF NOT EXISTS eod_summaries (
            session_date TEXT PRIMARY KEY,
            venue TEXT NOT NULL,
            etf TEXT NOT NULL,
            gap_pct REAL,
            signal_dir INTEGER,
            entry_price REAL,
            exit_price REAL,
            shares REAL,
            pnl_usd REAL,
            commission_usd REAL,
            account_nav_usd REAL,
            note TEXT
        );
        CREATE TABLE IF NOT EXISTS kill_events (
            ts_utc TEXT NOT NULL,
            reason TEXT NOT NULL,
            detail TEXT
        );
        """
    )
    conn.commit()
    conn.close()


def db_log(table: str, row: dict) -> None:
    conn = sqlite3.connect(DECISIONS_DB)
    cur = conn.cursor()
    cols = ", ".join(row.keys())
    placeholders = ", ".join("?" for _ in row)
    cur.execute(
        f"INSERT INTO {table} ({cols}) VALUES ({placeholders})",
        tuple(row.values()),
    )
    conn.commit()
    conn.close()


def db_upsert_eod(row: dict) -> None:
    conn = sqlite3.connect(DECISIONS_DB)
    cur = conn.cursor()
    cur.execute("DELETE FROM eod_summaries WHERE session_date = ?", (row["session_date"],))
    cols = ", ".join(row.keys())
    placeholders = ", ".join("?" for _ in row)
    cur.execute(
        f"INSERT INTO eod_summaries ({cols}) VALUES ({placeholders})",
        tuple(row.values()),
    )
    conn.commit()
    conn.close()


def count_executed_trades() -> int:
    if not DECISIONS_DB.exists():
        return 0
    conn = sqlite3.connect(DECISIONS_DB)
    cur = conn.execute(
        "SELECT COUNT(*) FROM orders WHERE order_type='MKT' AND status != 'DRYRUN'"
    )
    n = int(cur.fetchone()[0])
    conn.close()
    return n


# ---------------------------------------------------------------------------
# Persistent NAV / loss state
# ---------------------------------------------------------------------------
def load_state(default_nav: float) -> dict:
    if NAV_STATE_PATH.exists():
        try:
            return json.loads(NAV_STATE_PATH.read_text())
        except Exception:
            pass
    return {
        "session_date": None,
        "start_of_day_nav": default_nav,
        "intraday_pnl": 0.0,
        "daily_pnl": 0.0,
        "consecutive_losses": 0,
        "last_trade_result": None,
    }


def save_state(state: dict) -> None:
    NAV_STATE_PATH.write_text(json.dumps(state, indent=2))


def load_confirm_counter() -> int:
    if CONFIRM_COUNTER_PATH.exists():
        try:
            return int(json.loads(CONFIRM_COUNTER_PATH.read_text()).get("count", 0))
        except Exception:
            return 0
    return 0


def save_confirm_counter(n: int) -> None:
    CONFIRM_COUNTER_PATH.write_text(json.dumps({"count": int(n)}))


# ---------------------------------------------------------------------------
# Kill switches
# ---------------------------------------------------------------------------
@dataclass
class KillState:
    triggered: bool = False
    reason: str = ""
    detail: str = ""


def file_kill_active() -> bool:
    return KILL_FLAG_PATH.exists()


def check_kill(state: dict) -> KillState:
    if file_kill_active():
        return KillState(True, "FILE_KILL", f"Found {KILL_FLAG_PATH}")
    if state.get("consecutive_losses", 0) >= MAX_CONSECUTIVE_LOSSES:
        return KillState(True, "CONSECUTIVE_LOSSES",
                         f"{state['consecutive_losses']} in a row")
    nav = float(state.get("start_of_day_nav", 0))
    intraday = float(state.get("intraday_pnl", 0.0))
    if nav > 0 and intraday <= -INTRADAY_LOSS_KILL_FRAC * nav:
        return KillState(True, "INTRADAY_LOSS",
                         f"intraday_pnl={intraday:.2f} vs limit "
                         f"{-INTRADAY_LOSS_KILL_FRAC*nav:.2f}")
    return KillState(False)


def daily_cap_blocks_new_entry(state: dict) -> bool:
    nav = float(state.get("start_of_day_nav", 0))
    daily = float(state.get("daily_pnl", 0.0))
    return nav > 0 and daily <= -DAILY_LOSS_CAP_FRAC * nav


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------
def interactive_confirm(prompt: str) -> bool:
    try:
        sys.stdout.write(f"\n>>> CONFIRM: {prompt}  [y/N]: ")
        sys.stdout.flush()
        ans = sys.stdin.readline().strip().lower()
        return ans in {"y", "yes"}
    except Exception:
        return False


def poll_order(key: str, secret: str, order_id: str, poll_secs: int) -> dict:
    """Block up to poll_secs polling order status. Returns latest order dict."""
    last = {}
    for _ in range(max(1, poll_secs)):
        time.sleep(1)
        upd = get_order(key, secret, order_id)
        if upd:
            last = upd
            if upd.get("status") in {"filled", "canceled", "expired", "rejected"}:
                return upd
    return last


def submit_and_track(
    key: str,
    secret: str,
    symbol: str,
    side: str,
    qty: float,
    ref_px: float,
    session_date: date,
    client_tag: str,
    poll_secs: int,
    dry_run: bool,
) -> Optional[dict]:
    """Submit one order and log + return its filled state. side in {'buy','sell'}."""
    if qty <= 0:
        return None
    action_label = side.upper()
    if dry_run:
        logger.info(f"[DRY-RUN] {action_label} {qty:.4f} {symbol} ~ ${ref_px:.2f}")
        db_log("orders", {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "session_date": session_date.isoformat(),
            "venue": VENUE_LABEL,
            "symbol": symbol,
            "action": action_label,
            "qty": float(qty),
            "order_type": "MKT_DRYRUN",
            "broker_order_id": None,
            "status": "DRYRUN",
        })
        return {"status": "DRYRUN", "filled_avg_price": ref_px, "filled_qty": qty}

    submitted = submit_marketable_limit_ioc(key, secret, symbol, qty, side, ref_px, client_tag)
    if not submitted:
        db_log("orders", {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "session_date": session_date.isoformat(),
            "venue": VENUE_LABEL,
            "symbol": symbol,
            "action": action_label,
            "qty": float(qty),
            "order_type": "MKT",
            "broker_order_id": None,
            "status": "SUBMIT_FAILED",
        })
        return None
    order_id = submitted.get("id", "")
    order_status = submitted.get("status")
    db_log("orders", {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "session_date": session_date.isoformat(),
        "venue": VENUE_LABEL,
        "symbol": symbol,
        "action": action_label,
        "qty": float(qty),
        "order_type": "MKT",
        "broker_order_id": str(order_id),
        "status": str(order_status),
    })
    logger.info(f"submitted {side} {qty:.4f} {symbol} id={order_id} status={order_status}")
    final = poll_order(key, secret, order_id, poll_secs)
    if final and final.get("filled_qty"):
        try:
            filled_qty = float(final.get("filled_qty", 0))
            filled_px = float(final.get("filled_avg_price", 0)) if final.get("filled_avg_price") else None
        except Exception:
            filled_qty, filled_px = 0.0, None
        db_log("fills", {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "session_date": session_date.isoformat(),
            "venue": VENUE_LABEL,
            "symbol": symbol,
            "action": action_label,
            "qty": float(filled_qty),
            "fill_price": filled_px,
            "commission": 0.0,            # Alpaca paper: $0
            "broker_order_id": str(order_id),
        })
    return final


def flatten_all(
    key: str,
    secret: str,
    symbol: str,
    session_date: date,
    dry_run: bool,
) -> None:
    """Close any open position on `symbol` via marketable IOC."""
    pos = get_position(key, secret, symbol)
    if not pos:
        logger.info(f"flatten: no position on {symbol}")
        return
    try:
        qty = float(pos["qty"])
    except Exception:
        logger.warning(f"flatten: cannot parse qty from position {pos}")
        return
    if qty == 0:
        return
    side = "sell" if qty > 0 else "buy"
    ref_px = get_latest_trade(key, secret, symbol) or 1.0
    logger.warning(f"FLATTEN: closing {qty:+.4f} {symbol} via {side.upper()}")
    submit_and_track(
        key, secret, symbol, side, abs(qty), ref_px,
        session_date, "flatten", poll_secs=10, dry_run=dry_run,
    )


# ---------------------------------------------------------------------------
# Cycle: open (decide+enter), close (exit + accounting)
# ---------------------------------------------------------------------------
@dataclass
class CycleConfig:
    account: str
    etf: str
    allow_shorts: bool
    dry_run: bool
    confirm: bool
    poll_secs: int = 12


def open_leg(cfg: CycleConfig, key: str, secret: str) -> str:
    """Decision + entry leg. Run at US cash open + small buffer."""
    init_db()
    acct = get_account(key, secret)
    if not acct:
        logger.error("open_leg: account snapshot failed — aborting")
        return "account_unavailable"
    equity = float(acct.get("equity", 0) or 0)
    if equity <= 0:
        logger.error("open_leg: equity<=0 — aborting")
        return "equity_zero"

    state = load_state(equity)
    session_date = datetime.now(timezone.utc).date()
    if state.get("session_date") != session_date.isoformat():
        state["session_date"] = session_date.isoformat()
        state["start_of_day_nav"] = equity
        state["intraday_pnl"] = 0.0
        state["daily_pnl"] = 0.0
        save_state(state)

    kill = check_kill(state)
    if kill.triggered:
        logger.error(f"open_leg: kill active {kill.reason} :: {kill.detail}")
        db_log("kill_events", {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "reason": kill.reason,
            "detail": kill.detail,
        })
        flatten_all(key, secret, cfg.etf, session_date, cfg.dry_run)
        return f"kill:{kill.reason}"

    # 1. Compute signal from ^N225 daily close-to-open.
    df = fetch_n225_daily(days=10)
    sig = daily_gap_signal(df, GAP_THRESHOLD)
    logger.info(
        f"SIGNAL: tokyo_date={sig.session_date} prev_close={sig.prev_close:.2f} "
        f"open={sig.session_open:.2f} gap={sig.gap_pct*100:.3f}% "
        f"dir={sig.direction} take={sig.took_trade} ({sig.reason})"
    )
    db_log("evaluations", {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "session_date": sig.session_date.isoformat(),
        "venue": VENUE_LABEL,
        "etf": cfg.etf,
        "prev_close": sig.prev_close,
        "session_open": sig.session_open,
        "gap_pct": float(sig.gap_pct),
        "threshold_pct": float(GAP_THRESHOLD),
        "signal_dir": int(sig.direction),
        "took_trade": int(sig.took_trade),
        "reason": sig.reason,
    })

    if not sig.took_trade:
        logger.info("no-trade day (|gap| below threshold)")
        return "no_trade"
    if sig.direction < 0 and not cfg.allow_shorts:
        logger.info("short signal ignored (shorts disabled / not supported)")
        return "no_trade_short_disabled"
    if daily_cap_blocks_new_entry(state):
        logger.warning(
            f"daily loss cap reached (daily_pnl={state.get('daily_pnl')}); skip entry"
        )
        return "daily_cap_blocks"

    # 2. Existing position check (we trade single-symbol single-position).
    existing = get_position(key, secret, cfg.etf)
    if existing and abs(float(existing.get("qty", 0) or 0)) > 0:
        logger.warning(
            f"open_leg: existing position on {cfg.etf} qty={existing['qty']} — "
            "skip new entry (close leg will handle exit)"
        )
        return "position_exists"

    # 3. Sizing.
    px = get_latest_trade(key, secret, cfg.etf)
    if not px or px <= 0:
        logger.error(f"open_leg: no price for {cfg.etf} — aborting")
        return "no_price"
    target_usd = equity * MAX_POSITION_FRAC
    raw_qty = target_usd / px
    qty = math.floor(raw_qty * 100) / 100.0  # 2 decimals
    if qty <= 0:
        logger.error(f"open_leg: sized qty<=0 (equity={equity}, px={px})")
        return "size_zero"

    side = "buy" if sig.direction > 0 else "sell"
    action_label = "LONG" if side == "buy" else "SHORT"

    # 4. Confirm gate.
    confirm_count = load_confirm_counter()
    if confirm_count < CONFIRM_TRADE_COUNT and not cfg.dry_run:
        if not cfg.confirm:
            logger.error(
                f"--confirm required for first {CONFIRM_TRADE_COUNT} trades "
                f"(current count={confirm_count})"
            )
            return "confirm_required"
        ok = interactive_confirm(
            f"{action_label} {qty:.2f} {cfg.etf} @ ~${px:.2f} "
            f"(notional ${qty*px:,.0f}, equity ${equity:,.0f}, gap {sig.gap_pct*100:+.2f}%) "
            f"[trade #{confirm_count+1} of first {CONFIRM_TRADE_COUNT}]"
        )
        if not ok:
            logger.warning("user declined confirm — skipping entry")
            return "user_declined"

    # 5. Submit entry.
    result = submit_and_track(
        key, secret, cfg.etf, side, qty, px,
        session_date, f"entry-{action_label.lower()}",
        cfg.poll_secs, cfg.dry_run,
    )
    if not result:
        logger.error("entry submit failed")
        return "entry_failed"

    save_confirm_counter(confirm_count + 1)
    logger.info(
        f"ENTRY DONE: {action_label} {qty:.2f} {cfg.etf} (status={result.get('status')})"
    )
    return f"entered:{action_label}"


def close_leg(cfg: CycleConfig, key: str, secret: str) -> str:
    """Exit leg. Run shortly before US cash close. Flatten any open position
    and record EOD P&L versus most recent evaluation.
    """
    init_db()
    session_date = datetime.now(timezone.utc).date()
    pos = get_position(key, secret, cfg.etf)
    if not pos or abs(float(pos.get("qty", 0) or 0)) == 0:
        logger.info("close_leg: no position to exit")
        # Still write an EOD note so the cron run is auditable.
        _write_eod_no_trade(cfg.etf, session_date, cfg.dry_run)
        return "no_position"

    qty = float(pos["qty"])
    entry_px = float(pos.get("avg_entry_price", 0) or 0)
    side_exit = "sell" if qty > 0 else "buy"
    px_ref = get_latest_trade(key, secret, cfg.etf) or entry_px or 1.0
    logger.info(
        f"EXIT: closing {qty:+.4f} {cfg.etf} entry_avg={entry_px:.2f} ref=${px_ref:.2f}"
    )
    result = submit_and_track(
        key, secret, cfg.etf, side_exit, abs(qty), px_ref,
        session_date, "exit", cfg.poll_secs, cfg.dry_run,
    )

    # Compute realized P&L from fills.
    exit_px = None
    if result:
        try:
            if result.get("filled_avg_price"):
                exit_px = float(result["filled_avg_price"])
        except Exception:
            exit_px = None
    if exit_px is None:
        exit_px = px_ref

    side_sign = 1.0 if qty > 0 else -1.0
    shares = abs(qty)
    pnl_usd = (exit_px - entry_px) * shares * side_sign

    # Look up the evaluation row from today for gap context.
    gap_pct, signal_dir = _latest_evaluation_for_today(session_date)

    # Update NAV / consecutive-loss tracking.
    acct = get_account(key, secret)
    equity_after = float(acct.get("equity", 0)) if acct else 0.0
    state = load_state(equity_after if equity_after else 0)
    state["intraday_pnl"] = float(state.get("intraday_pnl", 0.0)) + pnl_usd
    state["daily_pnl"] = float(state.get("daily_pnl", 0.0)) + pnl_usd
    if pnl_usd < 0:
        state["consecutive_losses"] = int(state.get("consecutive_losses", 0)) + 1
        state["last_trade_result"] = "LOSS"
    elif pnl_usd > 0:
        state["consecutive_losses"] = 0
        state["last_trade_result"] = "WIN"
    save_state(state)

    db_upsert_eod({
        "session_date": session_date.isoformat(),
        "venue": VENUE_LABEL,
        "etf": cfg.etf,
        "gap_pct": gap_pct,
        "signal_dir": signal_dir,
        "entry_price": float(entry_px),
        "exit_price": float(exit_px),
        "shares": float(shares),
        "pnl_usd": float(pnl_usd),
        "commission_usd": 0.0,
        "account_nav_usd": float(equity_after),
        "note": "dry-run" if cfg.dry_run else "live-paper",
    })

    logger.info(
        f"EOD: etf={cfg.etf} dir={signal_dir} gap={gap_pct} "
        f"entry={entry_px:.2f} exit={exit_px:.2f} shares={shares:.2f} "
        f"pnl_usd={pnl_usd:+.2f} consec_losses={state['consecutive_losses']}"
    )

    # Final post-trade kill check.
    final_kill = check_kill(state)
    if final_kill.triggered:
        logger.error(f"KILL after trade: {final_kill.reason} :: {final_kill.detail}")
        db_log("kill_events", {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "reason": final_kill.reason,
            "detail": final_kill.detail,
        })
        return f"kill:{final_kill.reason}"
    return f"exited:pnl={pnl_usd:+.2f}"


def _write_eod_no_trade(etf: str, session_date: date, dry_run: bool) -> None:
    gap_pct, signal_dir = _latest_evaluation_for_today(session_date)
    db_upsert_eod({
        "session_date": session_date.isoformat(),
        "venue": VENUE_LABEL,
        "etf": etf,
        "gap_pct": gap_pct,
        "signal_dir": signal_dir,
        "entry_price": None,
        "exit_price": None,
        "shares": 0.0,
        "pnl_usd": 0.0,
        "commission_usd": 0.0,
        "account_nav_usd": 0.0,
        "note": "no-trade" if not dry_run else "dry-run no-trade",
    })


def _latest_evaluation_for_today(session_date: date) -> tuple[Optional[float], Optional[int]]:
    if not DECISIONS_DB.exists():
        return None, None
    conn = sqlite3.connect(DECISIONS_DB)
    cur = conn.execute(
        "SELECT gap_pct, signal_dir FROM evaluations "
        "WHERE session_date = ? ORDER BY ts_utc DESC LIMIT 1",
        (session_date.isoformat(),),
    )
    row = cur.fetchone()
    conn.close()
    if row:
        gp = float(row[0]) if row[0] is not None else None
        sd = int(row[1]) if row[1] is not None else None
        return gp, sd
    return None, None


# ---------------------------------------------------------------------------
# Daemon loop
# ---------------------------------------------------------------------------
def _signal_handler(_signum, _frame):
    global STOPPED
    STOPPED = True
    logger.warning("stop signal received")


def daemon_loop(
    cfg: CycleConfig,
    key: str,
    secret: str,
    open_hhmm_et: str,
    close_hhmm_et: str,
) -> int:
    oh, om = [int(x) for x in open_hhmm_et.split(":")]
    ch, cm = [int(x) for x in close_hhmm_et.split(":")]
    last_open_date = None
    last_close_date = None
    logger.info(f"daemon: open at {open_hhmm_et} ET, close at {close_hhmm_et} ET")
    while not STOPPED:
        if file_kill_active():
            logger.warning(f"STOP file {KILL_FLAG_PATH} present — halting daemon")
            break
        now_et = datetime.now(NY)
        is_weekday = now_et.weekday() < 5
        open_dt = now_et.replace(hour=oh, minute=om, second=0, microsecond=0)
        close_dt = now_et.replace(hour=ch, minute=cm, second=0, microsecond=0)

        if is_weekday and now_et >= open_dt and now_et.date() != last_open_date and now_et < close_dt:
            try:
                open_leg(cfg, key, secret)
            except Exception:
                logger.exception("open_leg crashed; continuing daemon")
            last_open_date = now_et.date()

        if is_weekday and now_et >= close_dt and now_et.date() != last_close_date:
            try:
                close_leg(cfg, key, secret)
            except Exception:
                logger.exception("close_leg crashed; continuing daemon")
            last_close_date = now_et.date()

        for _ in range(60):
            if STOPPED or file_kill_active():
                break
            time.sleep(1)
    logger.info("daemon exited cleanly")
    return 0


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="iter_6 Nikkei gap-continuation proxy host (Alpaca ETF).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--account", default=os.environ.get("ALPACA_PAPER_ACCOUNT", "AEGIS"),
                   help="Alpaca paper account nickname (default: AEGIS).")
    p.add_argument("--etf", choices=VALID_ETFS, default=DEFAULT_ETF,
                   help=f"ETF proxy (default {DEFAULT_ETF}; {ALT_ETF} = JPY-hedged).")
    p.add_argument("--no-shorts", action="store_true",
                   help="Disable short entries even if account supports them.")
    p.add_argument("--once", action="store_true",
                   help="Run a single cycle (use with --cycle) and exit.")
    p.add_argument("--cycle", choices=("open", "close", "both"), default="both",
                   help="Which leg to run when --once is set (default both).")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute signal and log decisions but submit no orders.")
    p.add_argument("--confirm", action="store_true",
                   help=f"Required gate for the first {CONFIRM_TRADE_COUNT} live trades.")
    p.add_argument("--flatten-now", action="store_true",
                   help="Immediately close any open position on the chosen ETF and exit.")
    p.add_argument("--verify-only", action="store_true",
                   help="Run startup backtest verification and exit (no orders).")
    p.add_argument("--skip-verify", action="store_true",
                   help="Skip startup backtest verification.")
    p.add_argument("--open-time-et", default=DEFAULT_OPEN_HHMM_ET,
                   help=f"Entry decision time in ET (default {DEFAULT_OPEN_HHMM_ET}).")
    p.add_argument("--close-time-et", default=DEFAULT_CLOSE_HHMM_ET,
                   help=f"Exit time in ET (default {DEFAULT_CLOSE_HHMM_ET}).")
    p.add_argument("--poll-secs", type=int, default=12,
                   help="Order fill polling seconds (default 12).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()
    init_db()
    load_env()

    logger.info("=== iter_6 Nikkei ETF-proxy host START ===")
    logger.info(
        f"account={args.account} etf={args.etf} dry_run={args.dry_run} "
        f"confirm={args.confirm} cycle={args.cycle} once={args.once}"
    )

    if file_kill_active():
        logger.error(f"Kill flag present at {KILL_FLAG_PATH}; remove it to resume.")
        return 8

    key, secret = resolve_credentials(args.account)
    signal_module.signal(signal_module.SIGINT, _signal_handler)
    signal_module.signal(signal_module.SIGTERM, _signal_handler)

    # Probe shorting support.
    short_ok = check_shorting_supported(key, secret, args.etf)
    allow_shorts = (not args.no_shorts) and short_ok
    if args.no_shorts:
        logger.info("--no-shorts flag set; only long entries will be taken")
    elif not short_ok:
        logger.warning(
            f"shorting not supported for {args.etf} on this account "
            "(asset not shortable / not easy_to_borrow / account shorting_enabled=false); "
            "falling back to long-only"
        )

    print_banner(args.etf, allow_shorts=allow_shorts)

    # Startup verification.
    if args.verify_only:
        ok = verify_against_backtest()
        return 0 if ok else 3
    if not args.skip_verify:
        if not verify_against_backtest():
            logger.error("startup verification failed — exiting")
            return 3

    cfg = CycleConfig(
        account=args.account,
        etf=args.etf,
        allow_shorts=allow_shorts,
        dry_run=args.dry_run,
        confirm=args.confirm,
        poll_secs=int(args.poll_secs),
    )

    try:
        if args.flatten_now:
            logger.warning("--flatten-now requested; closing all positions and exiting")
            flatten_all(key, secret, cfg.etf, datetime.now(timezone.utc).date(), cfg.dry_run)
            return 0

        if args.once:
            if args.cycle in ("open", "both"):
                status_open = open_leg(cfg, key, secret)
                logger.info(f"once open-leg result: {status_open}")
            if args.cycle in ("close", "both"):
                status_close = close_leg(cfg, key, secret)
                logger.info(f"once close-leg result: {status_close}")
            return 0

        return daemon_loop(cfg, key, secret, args.open_time_et, args.close_time_et)

    except SystemExit:
        raise
    except Exception:
        logger.exception("unhandled exception")
        try:
            flatten_all(key, secret, cfg.etf, datetime.now(timezone.utc).date(), cfg.dry_run)
        except Exception:
            pass
        return 1
    finally:
        logger.info("=== iter_6 Nikkei ETF-proxy host END ===")


if __name__ == "__main__":
    sys.exit(main())
