"""iter_6 Nikkei 225 Gap Continuation — IBKR Paper Shadow Deployment Host.

Production-ready paper-shadow harness for the iter_6 Nikkei gap-continuation
strategy (daily-proxy version), backed by the iter_6 threshold sweep verdict:

    threshold            = 1.0% overnight gap
    OOS Sharpe           = 2.51
    OOS ann return       = 29.1%
    OOS max drawdown     = -5.40%
    OOS win rate         = 65.7%
    n_trades             = 338 over 7.10 years (~18.9% days in market)

Strategy summary (from iter_6_backtest.py and iter_6_sweep.json):
  Instrument : Nikkei 225 (cash index ^N225 for signal; trades NKD CME or
               NK225M OSE for execution)
  Signal     : sign(overnight gap) when |overnight gap| > 1.0%, else flat
  Entry      : at session open (Tokyo open if OSE leg; US-overnight bar open
               if CME NKD leg)
  Exit       : at session close (same session) — flat overnight
  Cost       : 5 bps round-trip already baked into backtest

Deployment notes:
  - This is the DAILY-PROXY host. The spec describes a 5-min intraday opening
    range breakout (09:00-11:30 JST). When paid 5-min Nikkei mini futures bars
    are available, swap `daily_gap_signal` for the 5-min OR-breakout signal —
    the host is structured so only the signal-generation function changes.
  - Default execution venue is CME NKD (Globex), accessible in retail IBKR
    paper accounts and only requires the CME data subscription (~$10/mo).
  - OSE NK225 mini is the alternative — better aligned with the original
    Tokyo-open hypothesis, but requires OSE direct data and a JPY-funded
    sub-account. Use `--venue ose` to select this path.
  - All decisions / orders / fills are mirrored to
    research/live/nikkei_decisions.sqlite3 for reconciliation.
  - Kill switches fail CLOSED: any one flattens and exits non-zero so the
    cron wrapper / supervisor can page.

Author : Deployment Engineer A30
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
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

# ib_insync is required.  Install with: pip install ib_insync
try:
    from ib_insync import (  # type: ignore
        IB,
        Contract,
        Future,
        MarketOrder,
        util,
    )
except ImportError as exc:  # pragma: no cover
    print(
        "FATAL: ib_insync is not installed. Install with `pip install ib_insync`.",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc

import numpy as np
import pandas as pd

# yfinance is used for (a) startup backtest verification and (b) fallback
# daily OHLC if IBKR historical Nikkei cash index data is unavailable.
try:
    import yfinance as yf  # type: ignore
except ImportError as exc:  # pragma: no cover
    print(
        "FATAL: yfinance is not installed. Install with `pip install yfinance`.",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

ROOT = Path("/Users/weeoto/turbo-carnival/research")
LIVE_DIR = ROOT / "live"
LOG_PATH = LIVE_DIR / "iter_6_nikkei_host.log"
DECISIONS_DB = LIVE_DIR / "nikkei_decisions.sqlite3"
NAV_STATE_PATH = LIVE_DIR / "nikkei_nav_state.json"
CONFIRM_COUNTER_PATH = LIVE_DIR / "nikkei_confirm_counter.json"
KILL_FLAG_PATH = LIVE_DIR / "STOP_NIKKEI"
BACKTEST_SUMMARY_PATH = ROOT / "results" / "iter_6_summary.json"
SWEEP_SUMMARY_PATH = ROOT / "results" / "iter_6_sweep.json"

# Strategy parameters — frozen, copied from iter_6_sweep.json best_by_sharpe.
GAP_THRESHOLD = 0.010            # 1.0% overnight gap threshold (winner)
ROUND_TRIP_COST_BPS = 5.0        # informational; broker pays real commission
LOOKBACK_DAYS = 60               # startup-verification window

# Risk envelope (account-level).
ACCOUNT_NAV_USD = 100_000.0
MAX_CONTRACTS = 1                # NKD margin ~$3-4k; 1 contract at $100k NAV
INTRADAY_LOSS_KILL_FRAC = 0.02   # 2% NAV intraday loss -> hard kill
DAILY_LOSS_CAP_FRAC = 0.015      # 1.5% NAV daily cap (stops new entries)
MAX_CONSECUTIVE_LOSSES = 5       # 5 in a row -> hard kill
CONFIRM_TRADE_COUNT = 5          # first 5 trades require --confirm gate

# Backtest divergence guard.
DIVERGENCE_KILL_FRAC = 0.30      # >30% divergence vs backtest signal count -> exit

# Connection.
DEFAULT_IB_HOST = "127.0.0.1"
DEFAULT_IB_PORT = 7497           # IBKR paper TWS
DEFAULT_IB_CLIENT_ID = 26
IB_CONNECT_TIMEOUT_SECS = 15
CONNECTION_LOSS_KILL_SECS = 300  # 5 minutes flat-line on IBKR socket -> kill

# Contract specs (NKD CME mini-Nikkei in USD).
NKD_MULTIPLIER_USD = 5.0         # $5 per index point — CME spec
NKD_EXCHANGE = "CME"
NKD_CURRENCY = "USD"
NKD_SYMBOL = "NKD"

# Contract specs (NK225M OSE mini-Nikkei in JPY).
NK225M_MULTIPLIER_JPY = 100.0    # JPY 100 per index point
NK225M_EXCHANGE = "OSE.JPN"
NK225M_CURRENCY = "JPY"
NK225M_SYMBOL = "N225M"

VENUES = ("cme", "ose")

# Tokyo session timing (informational; used for OSE leg scheduling).
TOKYO_OPEN_HHMM = "09:00"
TOKYO_LUNCH_START = "11:30"
TOKYO_LUNCH_END = "12:30"
TOKYO_CLOSE_HHMM = "15:00"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("iter_6_nikkei")
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(LOG_PATH, mode="a")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# Decisions DB
# ---------------------------------------------------------------------------

def init_db() -> None:
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DECISIONS_DB)
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS evaluations (
            ts_utc TEXT NOT NULL,
            session_date TEXT NOT NULL,
            venue TEXT NOT NULL,
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
            qty INTEGER NOT NULL,
            order_type TEXT NOT NULL,
            broker_order_id INTEGER,
            status TEXT
        );
        CREATE TABLE IF NOT EXISTS fills (
            ts_utc TEXT NOT NULL,
            session_date TEXT NOT NULL,
            venue TEXT NOT NULL,
            symbol TEXT NOT NULL,
            action TEXT NOT NULL,
            qty INTEGER NOT NULL,
            fill_price REAL,
            commission REAL,
            broker_order_id INTEGER
        );
        CREATE TABLE IF NOT EXISTS eod_summaries (
            session_date TEXT PRIMARY KEY,
            venue TEXT NOT NULL,
            gap_pct REAL,
            signal_dir INTEGER,
            entry_price REAL,
            exit_price REAL,
            contracts INTEGER,
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


# ---------------------------------------------------------------------------
# State (NAV, consecutive losses, confirm counter)
# ---------------------------------------------------------------------------

def load_state() -> dict:
    if NAV_STATE_PATH.exists():
        try:
            return json.loads(NAV_STATE_PATH.read_text())
        except Exception:
            pass
    return {
        "session_date": None,
        "start_of_day_nav": ACCOUNT_NAV_USD,
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
# Signal generation — pluggable
# ---------------------------------------------------------------------------

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


SignalFn = Callable[[pd.DataFrame, float], GapSignal]


def daily_gap_signal(df: pd.DataFrame, threshold: float = GAP_THRESHOLD) -> GapSignal:
    """Compute today's gap-continuation signal from daily OHLC.

    df: tz-naive index of trading days, columns include Open and Close.
        The LAST row is the current session — i.e. its Open is "today's open"
        and df.Close.shift(1).iloc[-1] is the prior close.
    """
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


# ---- 5-min OR-breakout placeholder (UPGRADE PATH) -------------------------
def opening_range_breakout_signal(
    bars_5min: pd.DataFrame, threshold: float = GAP_THRESHOLD
) -> GapSignal:
    """Placeholder for the true spec'd strategy: 09:00-11:30 JST 5-min OR
    breakout. Activate this when paid 5-min Nikkei mini futures bars are
    available — pass the resulting function into `run_once` via SIGNAL_FN.

    Suggested logic (NOT yet active):
      - Define OR = (max High, min Low) over the first N 5-min bars after
        the JST open (e.g. first 6 bars = first 30 minutes).
      - Long if price > OR.high after the OR window closes; short if price
        < OR.low.
      - Time-stop at 11:25 JST; flat by 11:30 JST (Tokyo lunch).
      - Optional: ATR regime gate, min OR width filter.
    """
    raise NotImplementedError(
        "5-min OR-breakout signal not implemented — paid intraday data required"
    )


# ---------------------------------------------------------------------------
# Startup verification — replay backtest on last LOOKBACK_DAYS
# ---------------------------------------------------------------------------

def fetch_recent_daily(logger: logging.Logger, days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """Pull daily ^N225 OHLC for the last `days` calendar days via yfinance."""
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=int(days * 1.6) + 7)  # buffer for weekends/holidays
    logger.info(f"Startup verification: fetching ^N225 daily {start} -> {end}")
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


def count_backtest_signals(df: pd.DataFrame, threshold: float = GAP_THRESHOLD) -> int:
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
    threshold. days_in_market_pct = 18.8932 in the OOS window.
    """
    try:
        sweep = json.loads(SWEEP_SUMMARY_PATH.read_text())
        for row in sweep.get("threshold_results", []):
            if abs(float(row.get("threshold_pct", -1)) - 1.0) < 1e-9:
                return lookback_days * float(row["days_in_market_pct"]) / 100.0
    except Exception:
        pass
    return lookback_days * 0.189  # fallback to sweep figure


def verify_against_backtest(logger: logging.Logger) -> None:
    """Re-run backtest logic on last LOOKBACK_DAYS and confirm signal count
    is within DIVERGENCE_KILL_FRAC of backtest expectation. Hard exit on fail.
    """
    df = fetch_recent_daily(logger, LOOKBACK_DAYS)
    observed = count_backtest_signals(df, GAP_THRESHOLD)
    trading_days = len(df.dropna())
    expected = expected_signal_count(trading_days)

    if expected <= 0:
        logger.warning("Expected signal count is zero — skipping divergence check")
        return

    divergence = abs(observed - expected) / expected
    logger.info(
        f"Startup verify: trading_days={trading_days} observed_signals={observed} "
        f"expected={expected:.2f} divergence={divergence*100:.1f}%"
    )

    if divergence > DIVERGENCE_KILL_FRAC:
        db_log("kill_events", {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "reason": "STARTUP_DIVERGENCE",
            "detail": json.dumps({
                "observed": observed,
                "expected": expected,
                "divergence_pct": round(divergence * 100, 3),
                "limit_pct": DIVERGENCE_KILL_FRAC * 100,
            }),
        })
        logger.error(
            f"FATAL: signal-count divergence {divergence*100:.1f}% > "
            f"{DIVERGENCE_KILL_FRAC*100:.0f}%. Exiting."
        )
        raise SystemExit(3)


# ---------------------------------------------------------------------------
# IBKR connection
# ---------------------------------------------------------------------------

@dataclass
class IBConnection:
    host: str
    port: int
    client_id: int
    ib: IB = field(default_factory=IB)
    last_heartbeat_ts: float = 0.0

    def connect(self, logger: logging.Logger) -> None:
        logger.info(
            f"Connecting to IBKR at {self.host}:{self.port} client_id={self.client_id} ..."
        )
        self.ib.connect(
            self.host,
            self.port,
            clientId=self.client_id,
            timeout=IB_CONNECT_TIMEOUT_SECS,
            readonly=False,
        )
        self.last_heartbeat_ts = time.time()
        logger.info(
            f"IBKR connected: serverVersion={self.ib.client.serverVersion()} "
            f"accounts={self.ib.managedAccounts()}"
        )

    def disconnect(self) -> None:
        if self.ib.isConnected():
            self.ib.disconnect()

    def alive(self) -> bool:
        return self.ib.isConnected()

    def heartbeat(self, logger: logging.Logger) -> None:
        if self.alive():
            try:
                self.ib.reqCurrentTime()
                self.last_heartbeat_ts = time.time()
            except Exception as exc:
                logger.warning(f"Heartbeat failed: {exc}")


def build_nkd_future(ib: IB, logger: logging.Logger) -> Contract:
    """Resolve the front-month CME NKD contract."""
    base = Future(symbol=NKD_SYMBOL, exchange=NKD_EXCHANGE, currency=NKD_CURRENCY)
    details = ib.reqContractDetails(base)
    if not details:
        raise RuntimeError("No NKD contract details returned by IBKR")
    # Sort by expiry, pick the soonest non-expired.
    today = datetime.now().strftime("%Y%m%d")
    candidates = sorted(
        [d.contract for d in details if d.contract.lastTradeDateOrContractMonth >= today],
        key=lambda c: c.lastTradeDateOrContractMonth,
    )
    if not candidates:
        raise RuntimeError("No non-expired NKD contract found")
    chosen = candidates[0]
    logger.info(
        f"NKD front-month: {chosen.localSymbol} expiry={chosen.lastTradeDateOrContractMonth}"
    )
    return chosen


def build_nk225m_future(ib: IB, logger: logging.Logger) -> Contract:
    """Resolve the front-month OSE NK225 mini contract."""
    base = Future(symbol=NK225M_SYMBOL, exchange=NK225M_EXCHANGE, currency=NK225M_CURRENCY)
    details = ib.reqContractDetails(base)
    if not details:
        raise RuntimeError("No NK225M contract details returned by IBKR (OSE data subscription required)")
    today = datetime.now().strftime("%Y%m%d")
    candidates = sorted(
        [d.contract for d in details if d.contract.lastTradeDateOrContractMonth >= today],
        key=lambda c: c.lastTradeDateOrContractMonth,
    )
    if not candidates:
        raise RuntimeError("No non-expired NK225M contract found")
    chosen = candidates[0]
    logger.info(
        f"NK225M front-month: {chosen.localSymbol} expiry={chosen.lastTradeDateOrContractMonth}"
    )
    return chosen


def resolve_contract(venue: str, ib: IB, logger: logging.Logger) -> tuple[Contract, float, str]:
    """Returns (contract, multiplier_in_account_ccy, currency_label)."""
    if venue == "cme":
        return build_nkd_future(ib, logger), NKD_MULTIPLIER_USD, NKD_CURRENCY
    if venue == "ose":
        return build_nk225m_future(ib, logger), NK225M_MULTIPLIER_JPY, NK225M_CURRENCY
    raise ValueError(f"Unknown venue: {venue!r}; must be one of {VENUES}")


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


def check_kill(state: dict, logger: logging.Logger) -> KillState:
    if file_kill_active():
        return KillState(True, "FILE_KILL", f"Found {KILL_FLAG_PATH}")
    if state.get("consecutive_losses", 0) >= MAX_CONSECUTIVE_LOSSES:
        return KillState(True, "CONSECUTIVE_LOSSES",
                         f"{state['consecutive_losses']} in a row")
    nav = float(state.get("start_of_day_nav", ACCOUNT_NAV_USD))
    intraday = float(state.get("intraday_pnl", 0.0))
    if nav > 0 and intraday <= -INTRADAY_LOSS_KILL_FRAC * nav:
        return KillState(True, "INTRADAY_LOSS",
                         f"intraday_pnl={intraday:.2f} vs limit "
                         f"{-INTRADAY_LOSS_KILL_FRAC*nav:.2f}")
    return KillState(False)


def daily_cap_blocks_new_entry(state: dict) -> bool:
    nav = float(state.get("start_of_day_nav", ACCOUNT_NAV_USD))
    daily = float(state.get("daily_pnl", 0.0))
    return nav > 0 and daily <= -DAILY_LOSS_CAP_FRAC * nav


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------

def submit_market(
    ib: IB,
    contract: Contract,
    action: str,
    qty: int,
    venue: str,
    session_date: date,
    logger: logging.Logger,
    dry_run: bool,
) -> Optional[int]:
    if qty <= 0:
        return None
    action = action.upper()
    assert action in ("BUY", "SELL")
    if dry_run:
        logger.info(f"[DRY-RUN] {action} {qty} {contract.localSymbol} MKT")
        db_log("orders", {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "session_date": session_date.isoformat(),
            "venue": venue,
            "symbol": contract.localSymbol,
            "action": action,
            "qty": int(qty),
            "order_type": "MKT_DRYRUN",
            "broker_order_id": None,
            "status": "DRYRUN",
        })
        return None

    order = MarketOrder(action, qty)
    trade = ib.placeOrder(contract, order)
    # Wait briefly for ack.
    ib.sleep(1.0)
    broker_id = trade.order.orderId if trade.order else None
    status = trade.orderStatus.status if trade.orderStatus else "Submitted"
    logger.info(
        f"Order: {action} {qty} {contract.localSymbol} broker_id={broker_id} status={status}"
    )
    db_log("orders", {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "session_date": session_date.isoformat(),
        "venue": venue,
        "symbol": contract.localSymbol,
        "action": action,
        "qty": int(qty),
        "order_type": "MKT",
        "broker_order_id": broker_id,
        "status": status,
    })
    return broker_id


def wait_for_fill(
    ib: IB, broker_id: int, timeout_secs: int, logger: logging.Logger
) -> Optional[dict]:
    if broker_id is None:
        return None
    t0 = time.time()
    while time.time() - t0 < timeout_secs:
        ib.sleep(0.5)
        for trade in ib.trades():
            if trade.order.orderId == broker_id and trade.orderStatus.status == "Filled":
                fills = trade.fills
                if fills:
                    f = fills[-1]
                    return {
                        "price": float(f.execution.price),
                        "qty": int(f.execution.shares),
                        "commission": float(getattr(f.commissionReport, "commission", 0.0) or 0.0),
                    }
    logger.warning(f"Order {broker_id} did not fill within {timeout_secs}s")
    return None


def flatten_all(
    ib: IB,
    contract: Contract,
    venue: str,
    session_date: date,
    logger: logging.Logger,
    dry_run: bool,
) -> None:
    """Close any open position on this contract via market order."""
    positions = [p for p in ib.positions() if p.contract.conId == contract.conId]
    for pos in positions:
        qty = int(pos.position)
        if qty == 0:
            continue
        action = "SELL" if qty > 0 else "BUY"
        logger.warning(f"FLATTEN: closing {qty} of {contract.localSymbol} via {action}")
        submit_market(ib, contract, action, abs(qty), venue, session_date, logger, dry_run)


# ---------------------------------------------------------------------------
# One cycle: evaluate, enter, exit
# ---------------------------------------------------------------------------

@dataclass
class CycleConfig:
    venue: str
    dry_run: bool
    confirm: bool
    fill_wait_secs: int = 60
    eod_hold_secs: int = 0          # if >0, sleep before closing; for end-of-cycle
    signal_fn: SignalFn = daily_gap_signal


def run_once(
    cfg: CycleConfig,
    conn: IBConnection,
    logger: logging.Logger,
) -> None:
    """One evaluation+entry+exit cycle. The 'session' is one trading day.

    Intended cron use:
      - CME path: invoke at ~US-cash-open (09:30 ET); exit at ~US-cash-close
        (16:00 ET).
      - OSE path: invoke at ~09:00 JST; exit at ~14:55 JST (just before close).
    """
    init_db()
    state = load_state()
    session_date = datetime.now(timezone.utc).date()

    # Reset daily counters at the start of a new session.
    if state.get("session_date") != session_date.isoformat():
        state["session_date"] = session_date.isoformat()
        state["start_of_day_nav"] = float(state.get("start_of_day_nav", ACCOUNT_NAV_USD))
        state["intraday_pnl"] = 0.0
        state["daily_pnl"] = 0.0
        save_state(state)

    kill = check_kill(state, logger)
    if kill.triggered:
        logger.error(f"KILL active before entry: {kill.reason} :: {kill.detail}")
        db_log("kill_events", {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "reason": kill.reason,
            "detail": kill.detail,
        })
        flatten_all_or_warn(cfg, conn, session_date, logger)
        raise SystemExit(5)

    # 1. Compute signal from daily OHLC ending today.
    df = fetch_recent_daily(logger, days=10)
    sig = cfg.signal_fn(df, GAP_THRESHOLD)
    logger.info(
        f"SIGNAL: date={sig.session_date} prev_close={sig.prev_close:.2f} "
        f"open={sig.session_open:.2f} gap={sig.gap_pct*100:.3f}% "
        f"dir={sig.direction} take={sig.took_trade} ({sig.reason})"
    )
    db_log("evaluations", {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "session_date": sig.session_date.isoformat(),
        "venue": cfg.venue,
        "prev_close": sig.prev_close,
        "session_open": sig.session_open,
        "gap_pct": float(sig.gap_pct),
        "threshold_pct": float(GAP_THRESHOLD),
        "signal_dir": int(sig.direction),
        "took_trade": int(sig.took_trade),
        "reason": sig.reason,
    })

    if not sig.took_trade:
        logger.info("No-trade day. Exiting cycle cleanly.")
        return

    if daily_cap_blocks_new_entry(state):
        logger.warning(
            f"Daily loss cap reached (daily_pnl={state.get('daily_pnl')}); "
            "skipping new entry."
        )
        return

    # 2. Resolve contract and confirm gate.
    contract, multiplier, ccy = resolve_contract(cfg.venue, conn.ib, logger)
    qty = MAX_CONTRACTS
    action_entry = "BUY" if sig.direction > 0 else "SELL"
    action_exit = "SELL" if sig.direction > 0 else "BUY"

    confirm_count = load_confirm_counter()
    if confirm_count < CONFIRM_TRADE_COUNT:
        if not cfg.confirm and not cfg.dry_run:
            logger.error(
                f"--confirm required for first {CONFIRM_TRADE_COUNT} trades "
                f"(current count={confirm_count}). Re-run with --confirm to proceed."
            )
            raise SystemExit(6)

    # 3. Entry order.
    entry_id = submit_market(
        conn.ib, contract, action_entry, qty,
        cfg.venue, session_date, logger, cfg.dry_run,
    )
    entry_fill = wait_for_fill(conn.ib, entry_id, cfg.fill_wait_secs, logger) if entry_id else None
    if entry_fill:
        db_log("fills", {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "session_date": session_date.isoformat(),
            "venue": cfg.venue,
            "symbol": contract.localSymbol,
            "action": action_entry,
            "qty": entry_fill["qty"],
            "fill_price": entry_fill["price"],
            "commission": entry_fill["commission"],
            "broker_order_id": entry_id,
        })

    # 4. Hold until session-close exit trigger.
    if cfg.eod_hold_secs > 0:
        logger.info(f"Holding for {cfg.eod_hold_secs}s until EOD exit...")
        t_end = time.time() + cfg.eod_hold_secs
        while time.time() < t_end:
            # Periodic kill checks; bail early if anything trips.
            time.sleep(min(30.0, t_end - time.time()))
            cur_state = load_state()
            cur_kill = check_kill(cur_state, logger)
            if cur_kill.triggered:
                logger.error(f"KILL during hold: {cur_kill.reason} :: {cur_kill.detail}")
                break

    # 5. Exit order.
    exit_id = submit_market(
        conn.ib, contract, action_exit, qty,
        cfg.venue, session_date, logger, cfg.dry_run,
    )
    exit_fill = wait_for_fill(conn.ib, exit_id, cfg.fill_wait_secs, logger) if exit_id else None
    if exit_fill:
        db_log("fills", {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "session_date": session_date.isoformat(),
            "venue": cfg.venue,
            "symbol": contract.localSymbol,
            "action": action_exit,
            "qty": exit_fill["qty"],
            "fill_price": exit_fill["price"],
            "commission": exit_fill["commission"],
            "broker_order_id": exit_id,
        })

    # 6. P&L bookkeeping (USD-equivalent — best effort).
    entry_px = entry_fill["price"] if entry_fill else float(sig.session_open)
    exit_px = exit_fill["price"] if exit_fill else float("nan")
    commission_total = (
        (entry_fill["commission"] if entry_fill else 0.0)
        + (exit_fill["commission"] if exit_fill else 0.0)
    )
    if math.isfinite(entry_px) and math.isfinite(exit_px):
        side = 1.0 if sig.direction > 0 else -1.0
        gross_native = (exit_px - entry_px) * multiplier * qty * side
        # NKD pays in USD directly; NK225M pays in JPY — convert at a flat rate
        # for bookkeeping. The cron wrapper can refine this from IBKR account
        # values overnight.
        pnl_usd = gross_native if ccy == "USD" else gross_native / 150.0
        pnl_usd -= commission_total
    else:
        pnl_usd = 0.0

    state["intraday_pnl"] = float(state.get("intraday_pnl", 0.0)) + pnl_usd
    state["daily_pnl"] = float(state.get("daily_pnl", 0.0)) + pnl_usd
    if pnl_usd < 0:
        state["consecutive_losses"] = int(state.get("consecutive_losses", 0)) + 1
        state["last_trade_result"] = "LOSS"
    elif pnl_usd > 0:
        state["consecutive_losses"] = 0
        state["last_trade_result"] = "WIN"
    save_state(state)
    save_confirm_counter(load_confirm_counter() + 1)

    db_upsert_eod({
        "session_date": session_date.isoformat(),
        "venue": cfg.venue,
        "gap_pct": float(sig.gap_pct),
        "signal_dir": int(sig.direction),
        "entry_price": float(entry_px) if math.isfinite(entry_px) else None,
        "exit_price": float(exit_px) if math.isfinite(exit_px) else None,
        "contracts": int(qty),
        "pnl_usd": float(pnl_usd),
        "commission_usd": float(commission_total),
        "account_nav_usd": float(state["start_of_day_nav"]) + float(state["daily_pnl"]),
        "note": "dry-run" if cfg.dry_run else "live-paper",
    })

    logger.info(
        f"EOD: gap={sig.gap_pct*100:.3f}% dir={sig.direction} "
        f"entry={entry_px:.2f} exit={exit_px:.2f} qty={qty} "
        f"pnl_usd={pnl_usd:.2f} (commission={commission_total:.2f}) "
        f"daily_pnl={state['daily_pnl']:.2f} "
        f"consec_losses={state['consecutive_losses']}"
    )

    # Final post-trade kill check (consecutive losses might have triggered).
    final_kill = check_kill(state, logger)
    if final_kill.triggered:
        logger.error(f"KILL after trade: {final_kill.reason} :: {final_kill.detail}")
        db_log("kill_events", {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "reason": final_kill.reason,
            "detail": final_kill.detail,
        })
        raise SystemExit(7)


def flatten_all_or_warn(
    cfg: CycleConfig,
    conn: IBConnection,
    session_date: date,
    logger: logging.Logger,
) -> None:
    try:
        contract, _, _ = resolve_contract(cfg.venue, conn.ib, logger)
        flatten_all(conn.ib, contract, cfg.venue, session_date, logger, cfg.dry_run)
    except Exception as exc:
        logger.error(f"flatten_all_or_warn failed: {exc}")


# ---------------------------------------------------------------------------
# CLI / supervisor loop
# ---------------------------------------------------------------------------

class GracefulExit(Exception):
    pass


def install_signal_handlers(logger: logging.Logger) -> None:
    def _handler(signum, _frame):
        logger.warning(f"Received signal {signum}; requesting graceful shutdown.")
        raise GracefulExit()
    signal_module.signal(signal_module.SIGINT, _handler)
    signal_module.signal(signal_module.SIGTERM, _handler)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="iter_6 Nikkei gap-continuation paper-shadow host (daily proxy)."
    )
    p.add_argument(
        "--venue",
        choices=VENUES,
        default="cme",
        help="Execution venue: 'cme' for NKD (default, retail-friendly) or 'ose' for NK225M.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute signals and log decisions but do not place orders.",
    )
    p.add_argument(
        "--confirm",
        action="store_true",
        help=f"Required gate for the first {CONFIRM_TRADE_COUNT} live trades.",
    )
    p.add_argument(
        "--flatten-now",
        action="store_true",
        help="Immediately close any open position on the chosen venue and exit.",
    )
    p.add_argument(
        "--verify-only",
        action="store_true",
        help="Run startup backtest verification and exit (no IBKR connection).",
    )
    p.add_argument(
        "--hold-secs",
        type=int,
        default=0,
        help="Seconds to hold between entry and exit. 0 = exit immediately after entry "
             "(use cron to schedule the open/close separately, or set ~21000 for "
             "intraday hold from US open to close).",
    )
    p.add_argument(
        "--fill-wait-secs",
        type=int,
        default=60,
        help="How long to wait for a market order to fill before warning.",
    )
    p.add_argument(
        "--host",
        default=os.environ.get("IB_HOST", DEFAULT_IB_HOST),
        help="IBKR gateway host (env IB_HOST overrides default).",
    )
    p.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("IB_PORT", DEFAULT_IB_PORT)),
        help="IBKR gateway port (env IB_PORT overrides default; 7497=paper TWS).",
    )
    p.add_argument(
        "--client-id",
        type=int,
        default=int(os.environ.get("IB_CLIENT_ID", DEFAULT_IB_CLIENT_ID)),
        help="IBKR API client_id (env IB_CLIENT_ID overrides default).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logging()
    install_signal_handlers(logger)
    init_db()

    logger.info("=== iter_6 Nikkei paper-shadow host START ===")
    logger.info(
        f"venue={args.venue} dry_run={args.dry_run} confirm={args.confirm} "
        f"hold_secs={args.hold_secs} host={args.host} port={args.port} "
        f"client_id={args.client_id}"
    )

    if file_kill_active():
        logger.error(
            f"Kill flag present at {KILL_FLAG_PATH}; remove it to resume."
        )
        return 8

    # Startup verification: replay backtest on last LOOKBACK_DAYS.
    if args.verify_only:
        verify_against_backtest(logger)
        logger.info("Startup verification complete (verify-only). Exiting.")
        return 0

    try:
        verify_against_backtest(logger)
    except SystemExit:
        raise
    except Exception as exc:
        logger.error(f"Startup verification raised: {exc}")
        return 4

    conn = IBConnection(host=args.host, port=args.port, client_id=args.client_id)
    try:
        conn.connect(logger)
    except Exception as exc:
        logger.error(f"IBKR connection failed: {exc}")
        return 9

    cfg = CycleConfig(
        venue=args.venue,
        dry_run=args.dry_run,
        confirm=args.confirm,
        fill_wait_secs=args.fill_wait_secs,
        eod_hold_secs=args.hold_secs,
        signal_fn=daily_gap_signal,    # upgrade: swap to opening_range_breakout_signal
    )

    try:
        if args.flatten_now:
            logger.warning("--flatten-now requested. Closing all positions and exiting.")
            flatten_all_or_warn(cfg, conn, datetime.now(timezone.utc).date(), logger)
            return 0

        run_once(cfg, conn, logger)
        return 0

    except GracefulExit:
        logger.warning("Graceful shutdown requested; attempting to flatten if needed.")
        try:
            flatten_all_or_warn(cfg, conn, datetime.now(timezone.utc).date(), logger)
        finally:
            return 130

    except SystemExit:
        raise

    except Exception as exc:
        logger.exception(f"Unhandled exception: {exc}")
        try:
            flatten_all_or_warn(cfg, conn, datetime.now(timezone.utc).date(), logger)
        except Exception:
            pass
        return 1

    finally:
        try:
            conn.disconnect()
        except Exception:
            pass
        logger.info("=== iter_6 Nikkei paper-shadow host END ===")


if __name__ == "__main__":
    sys.exit(main())
