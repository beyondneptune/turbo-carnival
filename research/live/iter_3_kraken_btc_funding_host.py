"""iter_3 BTC Perpetual Funding Cash-and-Carry — KRAKEN Paper Shadow Host.

Production-ready paper-shadow harness for the iter_3 strategy, ported to use
KRAKEN end-to-end (the user already has a verified Kraken spot account):

  - Perp leg:  Kraken Futures (`krakenfutures` via ccxt), symbol `PF_XBTUSD`
               (Flexible / LINEAR USD-margined perpetual; ccxt unified
               symbol `BTC/USD:USD`), routed via the demo-futures sandbox
               (`set_sandbox_mode(True)` -> demo-futures.kraken.com).
               NB: Kraken also offers `PI_XBTUSD` (Inverse, settled in BTC,
               ccxt symbol `BTC/USD:USD:BTC`). We intentionally use the
               LINEAR PF_ product so spot-leg BTC quantity = perp-leg BTC
               quantity 1:1, keeping the cash-and-carry hedge exact.
  - Spot leg:  Kraken Spot (`kraken` via ccxt), symbol `BTC/USD`, real account
               but HARD-CAPPED at $100 USD notional per trade so it functions
               as a "tiny size for testing" paper proxy.

Why Kraken-only:
  - One brokerage relationship the operator has already KYC'd, instead of
    bringing Binance + OKX online from cold.
  - Eliminates inter-exchange counterparty risk fragmentation; both legs settle
    under the same legal entity (Payward Inc. / Payward Ventures).
  - Demo-futures sandbox is a faithful clone of production, no KYC required to
    obtain its keys.

Strategy parameters (identical to iter_3_summary.json — frozen by 90-day cycle):
  - Entry:    rolling-30 annualized funding APR > 10%
  - Exit:     rolling-30 annualized funding APR < 4%
  - Cost:     30 bps round-trip total budget (15 bps per side)
  - Sizing:   20% NAV per entry, max 5 simultaneous = 100% NAV
              (subject to the $100 spot hard cap below for the spot leg)

Kraken funding cadence:
  - Kraken Futures pays funding every HOUR (24x/day), unlike Binance/Bybit
    which pay every 8 hours (3x/day). The rolling-30 window therefore covers
    30 hours of funding history, not 240 hours; this is faster-reacting but
    consistent with the iter_3 mathematical definition (rolling mean of N
    *funding cycles* in annualized terms).

Safety envelope (same as Binance version):
  - basis blowout > 2%               -> flatten + kill
  - either exchange unreachable >10m -> flatten + kill
  - NAV drawdown > 8%                -> flatten + kill
  - BTC 1h move > 10%                -> pause new entries
  - kill file STOP_BTC_FUNDING       -> flatten + kill
  - first 5 trades require --confirm
  - spot leg hard-capped at $100 notional (real-money budget cap)

All decisions / orders / fills / reconciliations are mirrored to
research/live/kraken_btc_funding_decisions.sqlite3.

Author : Deployment Engineer A35
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
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

try:
    import ccxt  # type: ignore
except ImportError as exc:  # pragma: no cover
    print(
        "FATAL: ccxt is not installed. Install with `pip install ccxt`.",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

ROOT = Path("/Users/weeoto/turbo-carnival/research")
LIVE_DIR = ROOT / "live"
ENV_PATH = Path("/Users/weeoto/turbo-carnival/council/.env.local")
LOG_PATH = LIVE_DIR / "iter_3_kraken_btc_funding_host.log"


def _load_env_local() -> None:
    """Read council/.env.local into os.environ (setdefault — does not clobber)."""
    if not ENV_PATH.exists():
        return
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env_local()
DECISIONS_DB = LIVE_DIR / "kraken_btc_funding_decisions.sqlite3"
NAV_STATE_PATH = LIVE_DIR / "kraken_btc_funding_nav_state.json"
KILL_FLAG_PATH = LIVE_DIR / "STOP_BTC_FUNDING"
CONFIRM_COUNTER_PATH = LIVE_DIR / "kraken_btc_funding_confirm_counter.json"
SUMMARY_JSON = ROOT / "results" / "iter_3_summary.json"

# Strategy parameters — frozen, copied from iter_3_backtest.py / summary.
# Kraken Futures: we trade the LINEAR perpetual (USD-margined).
#   - ccxt unified symbol:  BTC/USD:USD   (settle=USD, linear=True)
#   - Kraken native code:   PF_XBTUSD     (Flexible / linear)
# The INVERSE alternative (PI_XBTUSD, ccxt symbol BTC/USD:BTC) is NOT used:
# its quantity is denominated in USD-contracts (1 contract = $1), which would
# break the BTC-quantity sizing we use to keep the spot/perp hedge balanced.
SYMBOL_PERP = "BTC/USD:USD"           # ccxt unified for krakenfutures linear perp
SYMBOL_PERP_NATIVE = "PF_XBTUSD"      # native Kraken Futures code (linear)
SYMBOL_PERP_INVERSE_NATIVE = "PI_XBTUSD"   # NOT used — kept for reference
SYMBOL_SPOT = "BTC/USD"               # ccxt unified spot symbol on kraken

ENTRY_APR = 0.10                # 10% rolling APR
EXIT_APR = 0.04                 # 4% rolling APR
ROLL_WINDOW = 30                # 30 funding cycles
FUND_PER_DAY = 24               # Kraken Futures funds hourly (24 windows/day)
FUND_PER_YEAR = 365 * FUND_PER_DAY
ROUND_TRIP_COST_BUDGET = 0.0030 # 30 bps total
PER_SIDE_COST_BUDGET = 0.0015   # 15 bps per side

# Sizing / risk
PER_ENTRY_NAV_FRACTION = 0.20   # 20% NAV per entry
MAX_OPEN_POSITIONS = 5          # 5 * 20% = 100% NAV
DEFAULT_NAV_USD = 22_000.0      # Kraken Futures Demo auto-replenish ceiling (mixed currencies)

# Demo-only mode flag — set by --demo-only argument in main().
# When True: skip real Kraken Spot leg; treat existing BTC inventory in the
# Kraken Futures Demo account as the implicit long leg of cash-and-carry.
DEMO_ONLY_MODE: bool = False

# Spot leg HARD CAP (real money, intentionally tiny for "testing in production")
SPOT_LEG_MAX_USD = 100.0        # max $100 notional per spot trade

# Risk envelope (hard kills)
MAX_BASIS_DIVERGENCE = 0.02     # 2% perp/spot basis -> kill
EXCHANGE_OUTAGE_KILL_SECS = 600 # 10 minutes
MAX_NAV_DRAWDOWN = 0.08         # 8% drawdown -> kill
VOL_EVENT_MOVE = 0.10           # 10% BTC move in 1h -> auto-pause
CONFIRM_TRADE_COUNT = 5         # interactive --confirm for first 5 trades

# Two-exchange coordination
LEG_FILL_TIMEOUT_SECS = 30      # both legs must fill within 30s
RECONCILE_SECS = 3600           # reconcile / decision cycle = 1 hour
KILL_FILE_POLL_SECS = 60        # kill file checked at least every 60s

# Funding window timing — Kraken Futures pays each top-of-hour UTC.
FUNDING_WINDOWS_UTC = tuple(range(24))  # every hour 00..23 UTC
ENTRY_LOOKAHEAD_MIN_MINUTES = 5         # skip "borderline" entries inside 5 min

# Startup verification
STARTUP_HISTORY_DAYS = 30
# 30d * 24/day = 720 cycles on Kraken. Use a similar above-threshold tolerance
# as the Binance version, but compute the base fraction from in-regime data
# generically — keep the same ~47% point estimate (the same global BTC funding
# distribution gates both); tolerance +/- 30%.
STARTUP_TOLERANCE_FRAC = 0.30
EXPECTED_ABOVE_10PCT_BASE_FRAC = 2768.0 / 5874.0


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("iter_3_kraken_host")
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fh = logging.FileHandler(LOG_PATH, mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# SQLite store
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc       TEXT    NOT NULL,
    cycle_type   TEXT    NOT NULL,
    funding_rate REAL,
    roll_apr     REAL,
    signal       TEXT,
    n_open       INTEGER,
    nav_usd      REAL,
    note         TEXT
);

CREATE TABLE IF NOT EXISTS orders (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc         TEXT    NOT NULL,
    position_id    INTEGER,
    leg            TEXT,
    exchange       TEXT,
    symbol         TEXT,
    side           TEXT,
    qty            REAL,
    price_hint     REAL,
    status         TEXT,
    exchange_oid   TEXT,
    fill_price     REAL,
    note           TEXT
);

CREATE TABLE IF NOT EXISTS positions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    opened_ts_utc  TEXT    NOT NULL,
    closed_ts_utc  TEXT,
    qty_btc        REAL    NOT NULL,
    perp_entry_px  REAL,
    spot_entry_px  REAL,
    perp_exit_px   REAL,
    spot_exit_px   REAL,
    funding_collected_usd REAL DEFAULT 0,
    status         TEXT    NOT NULL,
    exit_reason    TEXT
);

CREATE TABLE IF NOT EXISTS reconciliations (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc         TEXT    NOT NULL,
    perp_position_btc REAL,
    spot_position_btc REAL,
    basis_pct      REAL,
    futures_ok     INTEGER,
    spot_ok        INTEGER,
    note           TEXT
);

CREATE TABLE IF NOT EXISTS daily_summary (
    day_utc         TEXT PRIMARY KEY,
    funding_collected_usd REAL,
    basis_drift_max_pct   REAL,
    perp_pnl_usd    REAL,
    spot_pnl_usd    REAL,
    net_pnl_usd     REAL,
    n_open_eod      INTEGER,
    note            TEXT
);
"""


def open_db() -> sqlite3.Connection:
    DECISIONS_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DECISIONS_DB)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def db_log_decision(
    conn: sqlite3.Connection,
    cycle_type: str,
    funding_rate: Optional[float],
    roll_apr: Optional[float],
    signal_str: str,
    n_open: int,
    nav_usd: float,
    note: str,
) -> None:
    conn.execute(
        "INSERT INTO decisions"
        " (ts_utc, cycle_type, funding_rate, roll_apr, signal, n_open, nav_usd, note)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            datetime.now(timezone.utc).isoformat(),
            cycle_type,
            funding_rate,
            roll_apr,
            signal_str,
            n_open,
            nav_usd,
            note,
        ),
    )
    conn.commit()


def db_log_order(
    conn: sqlite3.Connection,
    position_id: Optional[int],
    leg: str,
    exchange: str,
    symbol: str,
    side: str,
    qty: float,
    price_hint: Optional[float],
    status: str,
    exchange_oid: Optional[str],
    fill_price: Optional[float],
    note: str,
) -> int:
    cur = conn.execute(
        "INSERT INTO orders"
        " (ts_utc, position_id, leg, exchange, symbol, side, qty, price_hint,"
        "  status, exchange_oid, fill_price, note)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            datetime.now(timezone.utc).isoformat(),
            position_id,
            leg,
            exchange,
            symbol,
            side,
            qty,
            price_hint,
            status,
            exchange_oid,
            fill_price,
            note,
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def db_open_position(conn: sqlite3.Connection, qty_btc: float,
                     perp_px: float, spot_px: float) -> int:
    cur = conn.execute(
        "INSERT INTO positions"
        " (opened_ts_utc, qty_btc, perp_entry_px, spot_entry_px, status)"
        " VALUES (?, ?, ?, ?, 'opening')",
        (datetime.now(timezone.utc).isoformat(), qty_btc, perp_px, spot_px),
    )
    conn.commit()
    return int(cur.lastrowid)


def db_mark_position(conn: sqlite3.Connection, pid: int, status: str,
                     exit_reason: Optional[str] = None,
                     perp_exit_px: Optional[float] = None,
                     spot_exit_px: Optional[float] = None,
                     funding_collected_usd: Optional[float] = None) -> None:
    fields = ["status = ?"]
    args: list[Any] = [status]
    if status in ("closed", "aborted"):
        fields.append("closed_ts_utc = ?")
        args.append(datetime.now(timezone.utc).isoformat())
    if exit_reason is not None:
        fields.append("exit_reason = ?")
        args.append(exit_reason)
    if perp_exit_px is not None:
        fields.append("perp_exit_px = ?")
        args.append(perp_exit_px)
    if spot_exit_px is not None:
        fields.append("spot_exit_px = ?")
        args.append(spot_exit_px)
    if funding_collected_usd is not None:
        fields.append("funding_collected_usd = COALESCE(funding_collected_usd,0) + ?")
        args.append(funding_collected_usd)
    args.append(pid)
    conn.execute(f"UPDATE positions SET {', '.join(fields)} WHERE id = ?", args)
    conn.commit()


def db_log_reconcile(conn: sqlite3.Connection, perp_btc: float, spot_btc: float,
                     basis_pct: float, futures_ok: bool, spot_ok: bool,
                     note: str) -> None:
    conn.execute(
        "INSERT INTO reconciliations"
        " (ts_utc, perp_position_btc, spot_position_btc, basis_pct,"
        "  futures_ok, spot_ok, note)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            datetime.now(timezone.utc).isoformat(),
            perp_btc,
            spot_btc,
            basis_pct,
            int(futures_ok),
            int(spot_ok),
            note,
        ),
    )
    conn.commit()


def db_load_open_positions(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    cur = conn.execute(
        "SELECT id, opened_ts_utc, qty_btc, perp_entry_px, spot_entry_px,"
        " funding_collected_usd, status"
        " FROM positions WHERE status IN ('open','opening','closing')"
    )
    rows = []
    for r in cur.fetchall():
        rows.append(
            dict(
                id=r[0], opened_ts_utc=r[1], qty_btc=r[2],
                perp_entry_px=r[3], spot_entry_px=r[4],
                funding_collected_usd=r[5] or 0.0, status=r[6],
            )
        )
    return rows


# ---------------------------------------------------------------------------
# NAV state
# ---------------------------------------------------------------------------

@dataclass
class NavState:
    nav_usd: float = DEFAULT_NAV_USD
    peak_nav_usd: float = DEFAULT_NAV_USD
    last_update_ts: str = ""
    cumulative_funding_usd: float = 0.0
    cumulative_fees_usd: float = 0.0

    @classmethod
    def load(cls) -> "NavState":
        if NAV_STATE_PATH.exists():
            try:
                with open(NAV_STATE_PATH) as f:
                    data = json.load(f)
                return cls(**data)
            except Exception:  # noqa: BLE001
                pass
        return cls()

    def save(self) -> None:
        self.last_update_ts = datetime.now(timezone.utc).isoformat()
        NAV_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(NAV_STATE_PATH, "w") as f:
            json.dump(asdict(self), f, indent=2)

    def update_peak(self) -> None:
        self.peak_nav_usd = max(self.peak_nav_usd, self.nav_usd)

    def drawdown(self) -> float:
        if self.peak_nav_usd <= 0:
            return 0.0
        return max(0.0, (self.peak_nav_usd - self.nav_usd) / self.peak_nav_usd)


# ---------------------------------------------------------------------------
# Confirm counter (first N trades require --confirm)
# ---------------------------------------------------------------------------

def load_confirm_counter() -> int:
    if not CONFIRM_COUNTER_PATH.exists():
        return 0
    try:
        with open(CONFIRM_COUNTER_PATH) as f:
            return int(json.load(f).get("trades_done", 0))
    except Exception:  # noqa: BLE001
        return 0


def save_confirm_counter(n: int) -> None:
    CONFIRM_COUNTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIRM_COUNTER_PATH, "w") as f:
        json.dump({"trades_done": int(n)}, f)


def require_confirm_if_needed(logger: logging.Logger, confirm_flag: bool,
                              trades_done: int, trade_label: str) -> bool:
    if trades_done >= CONFIRM_TRADE_COUNT:
        return True
    if not confirm_flag:
        logger.warning(
            f"Trade #{trades_done + 1} ({trade_label}) requires --confirm "
            f"(first {CONFIRM_TRADE_COUNT} trades gated). Skipping."
        )
        return False
    try:
        ans = input(
            f"\n[CONFIRM] Place trade #{trades_done + 1}: {trade_label}? [y/N] "
        ).strip().lower()
    except EOFError:
        ans = "n"
    if ans != "y":
        logger.warning(f"User declined confirm for {trade_label}")
        return False
    return True


# ---------------------------------------------------------------------------
# Kill switches
# ---------------------------------------------------------------------------

class KillSwitchTriggered(Exception):
    """Raised when any kill condition fires."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


SHUTDOWN_REQUESTED = False


def install_signal_handlers(logger: logging.Logger) -> None:
    def _handler(signum: int, _frame: Any) -> None:
        global SHUTDOWN_REQUESTED
        SHUTDOWN_REQUESTED = True
        logger.warning(f"Received signal {signum}; requesting clean shutdown.")

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def kill_file_present() -> bool:
    return KILL_FLAG_PATH.exists()


# ---------------------------------------------------------------------------
# Exchange clients
# ---------------------------------------------------------------------------

def build_krakenfutures_sandbox(logger: logging.Logger) -> ccxt.Exchange:
    """Build a Kraken Futures client pointed at the demo-futures sandbox.

    Demo sandbox URL: https://demo-futures.kraken.com
    Keys are obtained by registering separately at demo-futures.kraken.com
    (NO KYC, isolated from the live account).
    """
    api = os.environ.get("KRAKEN_FUTURES_API_KEY", "")
    sec = os.environ.get("KRAKEN_FUTURES_API_SECRET", "")
    if not api or not sec:
        logger.warning(
            "KRAKEN_FUTURES_API_KEY/SECRET not set. Will run in"
            " read-only mode (no orders)."
        )
    ex = ccxt.krakenfutures({
        "apiKey": api,
        "secret": sec,
        "enableRateLimit": True,
    })
    # Demo / sandbox endpoint
    ex.set_sandbox_mode(True)
    try:
        ex.load_markets()
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Kraken Futures sandbox load_markets failed: {exc}")
    return ex


def build_kraken_spot(logger: logging.Logger) -> ccxt.Exchange:
    """Build a Kraken Spot client. This is the REAL (verified) account; the
    $100/trade hard cap (SPOT_LEG_MAX_USD) is enforced upstream of order
    placement to keep real-money exposure bounded to "testing-tiny"."""
    api = os.environ.get("KRAKEN_SPOT_API_KEY", "")
    sec = os.environ.get("KRAKEN_SPOT_API_SECRET", "")
    if not api or not sec:
        logger.warning(
            "KRAKEN_SPOT_API_KEY/SECRET not set. Will run in"
            " read-only mode (no orders)."
        )
    ex = ccxt.kraken({
        "apiKey": api,
        "secret": sec,
        "enableRateLimit": True,
    })
    try:
        ex.load_markets()
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Kraken Spot load_markets failed: {exc}")
    return ex


def resolve_perp_symbol(ex: ccxt.Exchange, logger: logging.Logger) -> str:
    """Return the unified symbol for the LINEAR BTC perpetual on Kraken
    Futures (PF_XBTUSD, ccxt symbol BTC/USD:USD).

    We deliberately refuse to fall back to the inverse PI_XBTUSD product:
    its quantity is denominated in USD contracts (1 contract = $1), not BTC,
    so silently substituting it would break the spot/perp hedge sizing."""
    try:
        markets = ex.markets or {}
    except Exception:  # noqa: BLE001
        markets = {}
    # Preferred: ccxt unified symbol for the linear perp
    if SYMBOL_PERP in markets:
        m = markets[SYMBOL_PERP]
        if m.get("linear") is True and (m.get("id") or "").startswith("PF_"):
            return SYMBOL_PERP
        logger.warning(
            f"{SYMBOL_PERP!r} resolved but is not the linear PF_ perp "
            f"(id={m.get('id')!r}). Falling through to native lookup."
        )
    # Native-id lookup (still must be LINEAR)
    for m_id, m in markets.items():
        if m.get("id") == SYMBOL_PERP_NATIVE and m.get("linear") is True:
            sym = m.get("symbol") or m_id
            logger.info(f"Resolved perp symbol via native id -> {sym}")
            return sym
    logger.error(
        f"Could not resolve LINEAR BTC perpetual ({SYMBOL_PERP_NATIVE}) on "
        f"{ex.id}. Refusing to fall back to inverse {SYMBOL_PERP_INVERSE_NATIVE} "
        f"because the contract semantics differ. Defaulting to {SYMBOL_PERP!r}; "
        f"order placement will likely fail until markets are loaded."
    )
    return SYMBOL_PERP


# ---------------------------------------------------------------------------
# Funding signal
# ---------------------------------------------------------------------------

def fetch_recent_funding(ex: ccxt.Exchange, perp_symbol: str,
                         logger: logging.Logger,
                         lookback_cycles: int = 720) -> pd.DataFrame:
    """Pull the last `lookback_cycles` funding rates from Kraken Futures.

    ccxt's krakenfutures provides fetch_funding_rate_history; lookback default
    is sized for ~30 days at hourly cadence (30*24=720)."""
    try:
        rows = ex.fetch_funding_rate_history(perp_symbol, limit=lookback_cycles)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            f"fetch_funding_rate_history failed on {ex.id} {perp_symbol}: {exc}"
        )
        return pd.DataFrame()
    out = []
    for r in rows:
        fr = r.get("fundingRate")
        if fr is None:
            continue
        out.append(
            {
                "timestamp": int(r.get("timestamp") or 0),
                "datetime": r.get("datetime"),
                "funding_rate": float(fr),
            }
        )
    df = pd.DataFrame(out).sort_values("timestamp").reset_index(drop=True)
    if not df.empty:
        df["annualized"] = df["funding_rate"] * FUND_PER_YEAR
        df["roll_mean_apr"] = (
            df["annualized"].rolling(ROLL_WINDOW, min_periods=ROLL_WINDOW).mean()
        )
    return df


def fetch_current_funding_rate(ex: ccxt.Exchange, perp_symbol: str,
                               logger: logging.Logger) -> Optional[float]:
    """Single-point fetch (used as belt + suspenders alongside history)."""
    try:
        fr = ex.fetch_funding_rate(perp_symbol)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"fetch_funding_rate failed on {ex.id} {perp_symbol}: {exc}"
        )
        return None
    rate = fr.get("fundingRate")
    if rate is None:
        return None
    return float(rate)


def compute_signal(df: pd.DataFrame) -> tuple[Optional[float], Optional[float], str]:
    """Return (current_funding, roll_apr, signal_label)."""
    if df.empty or df["roll_mean_apr"].isna().all():
        return None, None, "INSUFFICIENT_DATA"
    last = df.iloc[-1]
    roll = float(last["roll_mean_apr"]) if pd.notna(last["roll_mean_apr"]) else None
    cur = float(last["funding_rate"])
    if roll is None:
        return cur, None, "WARMUP"
    if roll > ENTRY_APR:
        return cur, roll, "ENTRY"
    if roll < EXIT_APR:
        return cur, roll, "EXIT"
    return cur, roll, "HOLD"


def next_funding_window_ts(now: Optional[datetime] = None) -> datetime:
    """Next top-of-hour UTC (Kraken Futures funds every hour)."""
    if now is None:
        now = datetime.now(timezone.utc)
    base = now.replace(minute=0, second=0, microsecond=0)
    return base + timedelta(hours=1)


# ---------------------------------------------------------------------------
# Price helpers
# ---------------------------------------------------------------------------

def safe_ticker_last(ex: ccxt.Exchange, symbol: str,
                     logger: logging.Logger) -> Optional[float]:
    try:
        t = ex.fetch_ticker(symbol)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"fetch_ticker {ex.id} {symbol} failed: {exc}")
        return None
    last = t.get("last") or t.get("close")
    if last is None:
        return None
    return float(last)


def compute_basis_pct(perp_px: float, spot_px: float) -> float:
    if spot_px <= 0:
        return 0.0
    return (perp_px - spot_px) / spot_px


def fetch_recent_btc_move(ex: ccxt.Exchange, symbol: str,
                          logger: logging.Logger,
                          minutes: int = 60) -> Optional[float]:
    """Return |return| over the last `minutes` minutes using 1m OHLCV."""
    try:
        ohlcv = ex.fetch_ohlcv(symbol, timeframe="1m", limit=minutes + 5)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"fetch_ohlcv for vol check failed: {exc}")
        return None
    if not ohlcv or len(ohlcv) < 2:
        return None
    closes = [row[4] for row in ohlcv[-(minutes + 1):]]
    if len(closes) < 2 or closes[0] <= 0:
        return None
    return abs(closes[-1] - closes[0]) / closes[0]


# ---------------------------------------------------------------------------
# Order placement / atomic two-leg coordination
# ---------------------------------------------------------------------------

@dataclass
class LegResult:
    leg: str
    exchange_id: str
    symbol: str
    side: str
    qty: float
    status: str
    exchange_oid: Optional[str] = None
    fill_price: Optional[float] = None
    error: Optional[str] = None


def place_market_order(ex: ccxt.Exchange, symbol: str, side: str, qty: float,
                       logger: logging.Logger,
                       dry_run: bool, reduce_only: bool = False) -> LegResult:
    """Place a single market order. On dry_run we just synthesize a fill."""
    leg_id = f"{ex.id}:{symbol}:{side}"
    if dry_run:
        px = safe_ticker_last(ex, symbol, logger) or 0.0
        logger.info(f"[DRY] {leg_id} qty={qty} px~{px}")
        return LegResult(
            leg=leg_id, exchange_id=ex.id, symbol=symbol, side=side,
            qty=qty, status="filled", exchange_oid="DRY",
            fill_price=px, error=None,
        )
    params: dict[str, Any] = {}
    if reduce_only:
        # Kraken Futures /sendorder accepts `reduceOnly` (camelCase, boolean).
        # Kraken Spot has no native reduce-only flag (spot is non-margin in
        # our setup), so this param is a no-op on the kraken (spot) client;
        # we only set reduce_only=True on perp leg paths.
        params["reduceOnly"] = True
    try:
        order = ex.create_order(symbol, "market", side, qty, None, params)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"create_order failed on {leg_id}: {exc}")
        return LegResult(
            leg=leg_id, exchange_id=ex.id, symbol=symbol, side=side,
            qty=qty, status="rejected", error=str(exc),
        )
    return LegResult(
        leg=leg_id, exchange_id=ex.id, symbol=symbol, side=side,
        qty=qty, status=str(order.get("status") or "submitted"),
        exchange_oid=str(order.get("id") or ""),
        fill_price=(float(order.get("average") or 0.0) or
                    float(order.get("price") or 0.0) or None),
    )


def _signed_position_size(ex: ccxt.Exchange, symbol: str) -> float:
    """Best-effort current net position size on a symbol. Returns 0.0 on failure.

    Used to snapshot the position BEFORE submitting an order so the
    fetch_positions fallback in poll_fill can verify the *delta* matches the
    expected order quantity instead of conflating cumulative size with a fill.
    """
    try:
        positions = ex.fetch_positions([symbol])
    except Exception:  # noqa: BLE001
        return 0.0
    total = 0.0
    for p in positions:
        if p.get("symbol") != symbol:
            continue
        contracts = float(p.get("contracts") or 0)
        side = (p.get("side") or "").lower()
        if side == "short":
            total -= contracts
        else:
            total += contracts
    return total


def poll_fill(
    ex: ccxt.Exchange,
    order_id: str,
    symbol: str,
    logger: logging.Logger,
    timeout_secs: int,
    *,
    expected_qty: Optional[float] = None,
    expected_side: Optional[str] = None,
    pre_position_size: Optional[float] = None,
    price_hint: Optional[float] = None,
) -> tuple[bool, Optional[float]]:
    """Poll fetch_order until filled or timeout. Returns (filled, fill_price).

    Primary path: fetch_order returns "closed" → use the order's average price.

    Fallback (Kraken Futures demo): when fetch_order keeps failing, verify the
    fill by comparing the *current* position size against the *pre-order*
    snapshot. Accept only if the size delta matches the expected qty (with
    direction matching expected_side). Use `price_hint` as the fill price
    because fetch_positions().entryPrice is the BLENDED average across all
    open positions on the symbol, not this individual order's fill.
    """
    deadline = time.time() + timeout_secs
    fetch_order_failures = 0
    while time.time() < deadline:
        try:
            o = ex.fetch_order(order_id, symbol)
        except Exception as exc:  # noqa: BLE001
            fetch_order_failures += 1
            logger.warning(f"fetch_order {ex.id} {order_id} failed: {exc}")
            if fetch_order_failures >= 5:
                ok, px = _delta_verified_fill(
                    ex, symbol, logger,
                    expected_qty=expected_qty,
                    expected_side=expected_side,
                    pre_position_size=pre_position_size,
                    price_hint=price_hint,
                    tag="poll_fill fallback (mid-poll)",
                )
                if ok:
                    return True, px
            time.sleep(1.0)
            continue
        status = o.get("status")
        if status == "closed":
            px = float(o.get("average") or 0.0) or float(o.get("price") or 0.0) or None
            return True, px
        if status in ("canceled", "rejected", "expired"):
            return False, None
        time.sleep(1.0)

    # Final fallback at timeout
    ok, px = _delta_verified_fill(
        ex, symbol, logger,
        expected_qty=expected_qty,
        expected_side=expected_side,
        pre_position_size=pre_position_size,
        price_hint=price_hint,
        tag="poll_fill final fallback",
    )
    return ok, px


def _delta_verified_fill(
    ex: ccxt.Exchange,
    symbol: str,
    logger: logging.Logger,
    *,
    expected_qty: Optional[float],
    expected_side: Optional[str],
    pre_position_size: Optional[float],
    price_hint: Optional[float],
    tag: str,
) -> tuple[bool, Optional[float]]:
    """Verify a fill by comparing post-order position size delta to expected qty.

    If we have no pre-snapshot or expected qty we cannot disambiguate this fill
    from prior cumulative positions, so we refuse to claim a fill — that's the
    safe behavior. Returns (filled, fill_price_to_record).
    """
    if expected_qty is None or pre_position_size is None:
        # Best-effort legacy behavior: accept any non-zero position, but log
        # a clear warning that the fill price may be the blended entry.
        try:
            positions = ex.fetch_positions([symbol])
            for p in positions:
                if p.get("symbol") == symbol and (p.get("contracts") or 0) != 0:
                    entry = p.get("entryPrice")
                    if entry:
                        logger.warning(
                            f"{tag}: no delta context — using blended entryPrice "
                            f"{entry} as fill (UNVERIFIED). Pass expected_qty + "
                            f"pre_position_size to get a verified fill."
                        )
                        return True, float(entry)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"{tag} failed: {exc}")
        return False, None

    try:
        post_size = _signed_position_size(ex, symbol)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"{tag} fetch_positions failed: {exc}")
        return False, None

    delta = post_size - pre_position_size
    expected_signed = expected_qty if (expected_side or "").lower() == "buy" else -expected_qty
    # Allow 5% tolerance on the delta (rounding, fees, separate fills).
    tol = max(abs(expected_signed) * 0.05, 1e-6)
    if abs(delta - expected_signed) <= tol:
        # Delta matches expected quantity → genuine fill detected.
        # Use price_hint as the fill price (blended entryPrice is unreliable
        # when multiple positions are open).
        if price_hint and price_hint > 0:
            logger.info(
                f"{tag}: position delta {pre_position_size:+.6f} -> "
                f"{post_size:+.6f} = {delta:+.6f} matches expected "
                f"{expected_signed:+.6f}. Using price_hint={price_hint} as fill."
            )
            return True, float(price_hint)
        # No hint — fall back to blended entry with warning.
        try:
            positions = ex.fetch_positions([symbol])
            for p in positions:
                if p.get("symbol") == symbol and (p.get("contracts") or 0) != 0:
                    entry = p.get("entryPrice")
                    if entry:
                        logger.warning(
                            f"{tag}: delta matched but no price_hint — using "
                            f"blended entryPrice {entry} (may differ from real fill)."
                        )
                        return True, float(entry)
        except Exception:  # noqa: BLE001
            pass
        return False, None

    logger.warning(
        f"{tag}: position delta {pre_position_size:+.6f} -> {post_size:+.6f} "
        f"= {delta:+.6f} does NOT match expected {expected_signed:+.6f} "
        f"(tol={tol}). Refusing to claim fill."
    )
    return False, None


def cap_spot_qty(qty_btc: float, spot_px: float,
                 logger: logging.Logger) -> float:
    """Enforce SPOT_LEG_MAX_USD cap on the real-money spot leg."""
    if spot_px <= 0:
        return 0.0
    capped_qty = min(qty_btc, SPOT_LEG_MAX_USD / spot_px)
    if capped_qty < qty_btc:
        logger.warning(
            f"Spot leg HARD CAP: requested qty={qty_btc:.6f} BTC "
            f"-> capped to {capped_qty:.6f} BTC "
            f"(spot_px=${spot_px:.2f}, max_usd=${SPOT_LEG_MAX_USD:.2f})"
        )
    # round to typical Kraken spot precision (5dp)
    return round(capped_qty, 5)


def open_position_atomic(
    kf: ccxt.Exchange,
    ks: ccxt.Exchange,
    perp_symbol: str,
    qty_btc: float,
    logger: logging.Logger,
    conn: sqlite3.Connection,
    dry_run: bool,
) -> Optional[int]:
    """Open BOTH legs atomically. If one leg fails / times out, reverse the
    filled leg. Returns position_id on success, None on abort.

    `kf` = krakenfutures sandbox (perp short).
    `ks` = kraken spot (real, $100-capped long)."""
    perp_px_hint = safe_ticker_last(kf, perp_symbol, logger)
    if perp_px_hint is None:
        logger.error("Cannot price perp leg; aborting open.")
        return None
    if DEMO_ONLY_MODE or ks is None:
        # Demo-only: existing BTC inventory in the demo account IS the long leg.
        # Use perp price as spot proxy; basis check trivially passes.
        spot_px_hint = perp_px_hint
        basis = 0.0
    else:
        spot_px_hint = safe_ticker_last(ks, SYMBOL_SPOT, logger)
        if spot_px_hint is None:
            logger.error("Cannot price spot leg; aborting open.")
            return None
        basis = compute_basis_pct(perp_px_hint, spot_px_hint)
        if abs(basis) > MAX_BASIS_DIVERGENCE:
            logger.error(
                f"Pre-trade basis {basis*100:.2f}% exceeds "
                f"{MAX_BASIS_DIVERGENCE*100:.2f}%; aborting open."
            )
            return None

    if DEMO_ONLY_MODE or ks is None:
        # Demo-only: no $100 spot cap; the "spot" is just held BTC inventory.
        # Position size is bounded by NAV % already.
        final_qty = qty_btc
    else:
        # Spot leg cap applied here. Both legs use the same final qty so the
        # hedge remains exact (no naked perp exposure). If the cap reduces the
        # qty, we shrink the perp leg to match.
        spot_qty = cap_spot_qty(qty_btc, spot_px_hint, logger)
        if spot_qty <= 0:
            logger.error("Spot qty after cap is zero; aborting open.")
            return None
        final_qty = spot_qty
        if final_qty < qty_btc:
            logger.info(
                f"Resizing both legs to spot-capped qty={final_qty:.6f} BTC "
                f"(notional ~${final_qty * spot_px_hint:.2f})"
            )

    pid = db_open_position(conn, final_qty, perp_px_hint, spot_px_hint)
    logger.info(
        f"Opening position id={pid} qty={final_qty} BTC "
        f"perp_hint={perp_px_hint} spot_hint={spot_px_hint} "
        f"basis={basis*100:.3f}%"
    )

    # 1) perp short on Kraken Futures sandbox.
    # Snapshot pre-order net position so poll_fill can verify the size DELTA
    # matches expected qty (Kraken demo's fetchOrder 404 fallback otherwise
    # conflates cumulative position with this individual fill).
    pre_perp_size = _signed_position_size(kf, perp_symbol)
    perp_result = place_market_order(
        kf, perp_symbol, "sell", final_qty, logger, dry_run=dry_run,
    )
    db_log_order(
        conn, pid, "perp_short", kf.id, perp_symbol, "sell", final_qty,
        perp_px_hint, perp_result.status, perp_result.exchange_oid,
        perp_result.fill_price, perp_result.error or "",
    )
    if perp_result.status in ("rejected",):
        logger.error("Perp leg rejected; aborting open. No reverse needed.")
        db_mark_position(conn, pid, "aborted", exit_reason="perp_rejected")
        return None

    if perp_result.status != "filled" and not dry_run:
        filled, fill_px = poll_fill(
            kf, perp_result.exchange_oid or "", perp_symbol,
            logger, LEG_FILL_TIMEOUT_SECS,
            expected_qty=final_qty, expected_side="sell",
            pre_position_size=pre_perp_size, price_hint=perp_px_hint,
        )
        if not filled:
            logger.error("Perp leg failed to fill within timeout; aborting.")
            try:
                kf.cancel_order(perp_result.exchange_oid or "", perp_symbol)
            except Exception:  # noqa: BLE001
                pass
            db_mark_position(conn, pid, "aborted",
                             exit_reason="perp_fill_timeout")
            return None
        perp_result.fill_price = fill_px
        perp_result.status = "filled"

    if DEMO_ONLY_MODE or ks is None:
        # Demo-only: skip the spot leg entirely. The BTC inventory in the demo
        # account serves as the implicit long leg. Log a synthetic spot fill
        # for auditability.
        db_log_order(
            conn, pid, "spot_long_virtual", "demo_inventory", SYMBOL_SPOT,
            "buy", final_qty, spot_px_hint, "virtual_filled", "",
            spot_px_hint, "demo-only: using existing demo BTC inventory",
        )
        db_mark_position(conn, pid, "open")
        logger.info(
            f"Position id={pid} OPEN. perp_fill={perp_result.fill_price} "
            f"spot=VIRTUAL (demo inventory) @ {spot_px_hint}"
        )
        return pid

    # 2) spot long on Kraken Spot (real, but capped)
    pre_spot_size = _signed_position_size(ks, SYMBOL_SPOT)
    spot_result = place_market_order(
        ks, SYMBOL_SPOT, "buy", final_qty, logger, dry_run=dry_run,
    )
    db_log_order(
        conn, pid, "spot_long", ks.id, SYMBOL_SPOT, "buy", final_qty,
        spot_px_hint, spot_result.status, spot_result.exchange_oid,
        spot_result.fill_price, spot_result.error or "",
    )

    spot_ok = spot_result.status == "filled"
    if not spot_ok and spot_result.status != "rejected" and not dry_run:
        filled, fill_px = poll_fill(
            ks, spot_result.exchange_oid or "", SYMBOL_SPOT,
            logger, LEG_FILL_TIMEOUT_SECS,
            expected_qty=final_qty, expected_side="buy",
            pre_position_size=pre_spot_size, price_hint=spot_px_hint,
        )
        if filled:
            spot_result.fill_price = fill_px
            spot_result.status = "filled"
            spot_ok = True

    if not spot_ok:
        logger.error(
            "Spot leg failed; reversing perp short with a BUY to flatten."
        )
        reverse_result = place_market_order(
            kf, perp_symbol, "buy", final_qty, logger,
            dry_run=dry_run, reduce_only=True,
        )
        db_log_order(
            conn, pid, "perp_short_reverse", kf.id, perp_symbol,
            "buy", final_qty, perp_px_hint, reverse_result.status,
            reverse_result.exchange_oid, reverse_result.fill_price,
            "reverse-after-spot-fail",
        )
        db_mark_position(conn, pid, "aborted", exit_reason="spot_fail_reversed")
        return None

    db_mark_position(conn, pid, "open")
    logger.info(
        f"Position id={pid} OPEN. perp_fill={perp_result.fill_price} "
        f"spot_fill={spot_result.fill_price}"
    )
    return pid


def close_position_atomic(
    kf: ccxt.Exchange,
    ks: ccxt.Exchange,
    perp_symbol: str,
    pos: dict[str, Any],
    logger: logging.Logger,
    conn: sqlite3.Connection,
    dry_run: bool,
    reason: str,
) -> bool:
    pid = int(pos["id"])
    qty = float(pos["qty_btc"])
    perp_px_hint = safe_ticker_last(kf, perp_symbol, logger)
    if DEMO_ONLY_MODE or ks is None:
        spot_px_hint = perp_px_hint  # virtual spot in demo-only
    else:
        spot_px_hint = safe_ticker_last(ks, SYMBOL_SPOT, logger)
    db_mark_position(conn, pid, "closing", exit_reason=reason)
    logger.info(f"Closing position id={pid} qty={qty} reason={reason}")

    # 1) perp BUY to cover
    pre_perp_size = _signed_position_size(kf, perp_symbol)
    perp_result = place_market_order(
        kf, perp_symbol, "buy", qty, logger,
        dry_run=dry_run, reduce_only=True,
    )
    db_log_order(
        conn, pid, "perp_close", kf.id, perp_symbol, "buy", qty,
        perp_px_hint, perp_result.status, perp_result.exchange_oid,
        perp_result.fill_price, perp_result.error or "",
    )
    if perp_result.status != "filled" and not dry_run:
        filled, fill_px = poll_fill(
            kf, perp_result.exchange_oid or "", perp_symbol,
            logger, LEG_FILL_TIMEOUT_SECS,
            expected_qty=qty, expected_side="buy",
            pre_position_size=pre_perp_size, price_hint=perp_px_hint,
        )
        if filled:
            perp_result.fill_price = fill_px
            perp_result.status = "filled"

    if DEMO_ONLY_MODE or ks is None:
        # Demo-only: skip real spot SELL. The BTC inventory stays in the
        # demo account (it was never traded, just held).
        db_log_order(
            conn, pid, "spot_close_virtual", "demo_inventory", SYMBOL_SPOT,
            "sell", qty, spot_px_hint, "virtual_filled", "",
            spot_px_hint, "demo-only: BTC inventory remains in demo account",
        )
        both_ok = (perp_result.status == "filled")
        db_mark_position(
            conn, pid, "closed" if both_ok else "aborted",
            exit_reason=reason if both_ok else f"{reason}_partial",
            perp_exit_px=perp_result.fill_price,
            spot_exit_px=spot_px_hint,
        )
        return both_ok

    # 2) spot SELL
    pre_spot_size = _signed_position_size(ks, SYMBOL_SPOT)
    spot_result = place_market_order(
        ks, SYMBOL_SPOT, "sell", qty, logger, dry_run=dry_run,
    )
    db_log_order(
        conn, pid, "spot_close", ks.id, SYMBOL_SPOT, "sell", qty,
        spot_px_hint, spot_result.status, spot_result.exchange_oid,
        spot_result.fill_price, spot_result.error or "",
    )
    if spot_result.status != "filled" and not dry_run:
        filled, fill_px = poll_fill(
            ks, spot_result.exchange_oid or "", SYMBOL_SPOT,
            logger, LEG_FILL_TIMEOUT_SECS,
            expected_qty=qty, expected_side="sell",
            pre_position_size=pre_spot_size, price_hint=spot_px_hint,
        )
        if filled:
            spot_result.fill_price = fill_px
            spot_result.status = "filled"

    both_ok = (perp_result.status == "filled" and spot_result.status == "filled")
    db_mark_position(
        conn, pid, "closed" if both_ok else "aborted",
        exit_reason=reason if both_ok else f"{reason}_partial",
        perp_exit_px=perp_result.fill_price, spot_exit_px=spot_result.fill_price,
    )
    return both_ok


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------

def fetch_perp_net_position(kf: ccxt.Exchange, perp_symbol: str,
                            logger: logging.Logger) -> Optional[float]:
    try:
        positions = kf.fetch_positions([perp_symbol])
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"fetch_positions krakenfutures failed: {exc}")
        return None
    total = 0.0
    for p in positions:
        contracts = float(p.get("contracts") or 0.0)
        side = (p.get("side") or "").lower()
        if side == "short":
            total -= contracts
        else:
            total += contracts
    return total


def fetch_spot_btc_balance(ks: ccxt.Exchange,
                           logger: logging.Logger) -> Optional[float]:
    try:
        bal = ks.fetch_balance()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"fetch_balance kraken spot failed: {exc}")
        return None
    # Kraken's raw /0/private/Balance reports Bitcoin as "XXBT" (legacy)
    # or "XBT". ccxt normalizes this to "BTC" in the unified `total`/`free`
    # dicts, but we probe all three keys to survive any edge cases.
    total_map = bal.get("total", {}) if isinstance(bal.get("total"), dict) else {}
    free_map = bal.get("free", {}) if isinstance(bal.get("free"), dict) else {}
    for code in ("BTC", "XBT", "XXBT"):
        btc = bal.get(code) or total_map.get(code) or free_map.get(code)
        if btc is None:
            continue
        if isinstance(btc, dict):
            v = btc.get("total", 0.0) or 0.0
            return float(v)
        return float(btc)
    return None


def reconcile(
    kf: ccxt.Exchange,
    ks: ccxt.Exchange,
    perp_symbol: str,
    open_positions: list[dict[str, Any]],
    logger: logging.Logger,
    conn: sqlite3.Connection,
) -> tuple[bool, bool, Optional[float], Optional[float], Optional[float]]:
    """Returns (futures_ok, spot_ok, perp_btc, spot_btc, basis_pct)."""
    perp_btc = fetch_perp_net_position(kf, perp_symbol, logger)
    perp_px = safe_ticker_last(kf, perp_symbol, logger)
    futures_ok = perp_px is not None
    if DEMO_ONLY_MODE or ks is None:
        # Demo-only: skip Kraken Spot probes (no real spot connection).
        # The "spot leg" is BTC inventory held in the Kraken Futures Demo
        # account, which is already tracked via fetch_balance on kf if needed.
        spot_btc = None
        spot_px = perp_px
        spot_ok = futures_ok  # treat as healthy if futures is healthy
        basis = 0.0 if futures_ok else None
    else:
        spot_btc = fetch_spot_btc_balance(ks, logger)
        spot_px = safe_ticker_last(ks, SYMBOL_SPOT, logger)
        spot_ok = spot_px is not None
        basis = None
        if futures_ok and spot_ok:
            basis = compute_basis_pct(perp_px or 0.0, spot_px or 0.0)

    expected_qty = sum(p["qty_btc"] for p in open_positions
                       if p["status"] in ("open", "closing"))
    note_parts = [f"expected_qty={expected_qty:.6f}"]
    if perp_btc is not None:
        note_parts.append(f"perp_actual={perp_btc:.6f}")
        if abs(abs(perp_btc) - expected_qty) > 0.001:
            note_parts.append("PERP_DRIFT")
    if spot_btc is not None:
        note_parts.append(f"spot_actual={spot_btc:.6f}")
    if basis is not None:
        note_parts.append(f"basis_pct={basis*100:.3f}")

    db_log_reconcile(
        conn,
        perp_btc=perp_btc or 0.0,
        spot_btc=spot_btc or 0.0,
        basis_pct=basis or 0.0,
        futures_ok=futures_ok, spot_ok=spot_ok,
        note=" | ".join(note_parts),
    )
    return futures_ok, spot_ok, perp_btc, spot_btc, basis


# ---------------------------------------------------------------------------
# Startup verification
# ---------------------------------------------------------------------------

def startup_verification(kf: ccxt.Exchange, perp_symbol: str,
                         logger: logging.Logger) -> bool:
    """Pull last 30d funding, confirm signal would have fired within tolerance."""
    cycles_30d = STARTUP_HISTORY_DAYS * FUND_PER_DAY  # 720
    df = fetch_recent_funding(kf, perp_symbol, logger,
                              lookback_cycles=cycles_30d + ROLL_WINDOW)
    if df.empty or len(df) < ROLL_WINDOW + 1:
        logger.error(
            f"Startup verification FAIL: only {len(df)} funding rows "
            f"(need {ROLL_WINDOW + 1}+)."
        )
        return False
    slice_df = df.dropna(subset=["roll_mean_apr"]).tail(cycles_30d)
    n_total = len(slice_df)
    n_above = int((slice_df["annualized"] > ENTRY_APR).sum())
    expected = EXPECTED_ABOVE_10PCT_BASE_FRAC * n_total
    lo = expected * (1.0 - STARTUP_TOLERANCE_FRAC)
    hi = expected * (1.0 + STARTUP_TOLERANCE_FRAC)
    logger.info(
        f"Startup verification: last {n_total} cycles, "
        f"{n_above} above {ENTRY_APR*100:.0f}% (expected ~{expected:.1f} "
        f"+/- {STARTUP_TOLERANCE_FRAC*100:.0f}% -> [{lo:.1f}, {hi:.1f}])."
    )
    if n_total == 0:
        logger.error("Startup verification FAIL: 0 cycles in slice.")
        return False
    if n_above < lo - 1 and lo > 1:
        logger.warning(
            f"Startup verification SOFT-FAIL: {n_above} < lower bound "
            f"{lo:.1f}. Regime may be quiet; continuing in WATCH mode."
        )
        return True
    if n_above > hi + 1:
        logger.warning(
            f"Startup verification SOFT-WARN: {n_above} > upper bound "
            f"{hi:.1f}. Regime may be hot; continuing."
        )
        return True
    return True


# ---------------------------------------------------------------------------
# Daily summary
# ---------------------------------------------------------------------------

def write_daily_summary_if_due(conn: sqlite3.Connection, nav: NavState,
                               logger: logging.Logger) -> None:
    now = datetime.now(timezone.utc)
    if not (now.hour == 0 and now.minute < 5):
        return
    day = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    cur = conn.execute(
        "SELECT COALESCE(SUM(funding_collected_usd),0), COUNT(*) "
        "FROM positions WHERE status IN ('open','closing')"
    )
    funding_sum, n_open = cur.fetchone()
    cur = conn.execute(
        "SELECT MAX(ABS(basis_pct)) FROM reconciliations WHERE ts_utc >= ?",
        ((now - timedelta(days=1)).isoformat(),),
    )
    basis_max = cur.fetchone()[0] or 0.0
    try:
        conn.execute(
            "INSERT OR REPLACE INTO daily_summary"
            " (day_utc, funding_collected_usd, basis_drift_max_pct,"
            "  perp_pnl_usd, spot_pnl_usd, net_pnl_usd, n_open_eod, note)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                day, float(funding_sum or 0.0), float(basis_max * 100.0),
                None, None, None, int(n_open),
                f"nav_usd={nav.nav_usd:.2f} cum_funding={nav.cumulative_funding_usd:.2f}",
            ),
        )
        conn.commit()
        logger.info(
            f"Daily summary stored for {day}: funding={funding_sum:.2f} "
            f"basis_max={basis_max*100:.3f}% n_open={n_open}"
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"daily_summary write failed: {exc}")


# ---------------------------------------------------------------------------
# Flatten-all
# ---------------------------------------------------------------------------

def flatten_all(
    kf: ccxt.Exchange, ks: ccxt.Exchange, perp_symbol: str,
    logger: logging.Logger, conn: sqlite3.Connection, dry_run: bool,
    reason: str,
) -> None:
    open_positions = db_load_open_positions(conn)
    logger.warning(f"FLATTEN: {len(open_positions)} positions reason={reason}")
    for p in open_positions:
        try:
            close_position_atomic(kf, ks, perp_symbol, p, logger, conn,
                                  dry_run=dry_run, reason=reason)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Flatten close failed for pos {p['id']}: {exc}")


# ---------------------------------------------------------------------------
# Decision cycle
# ---------------------------------------------------------------------------

@dataclass
class CycleContext:
    kf: ccxt.Exchange
    ks: ccxt.Exchange
    perp_symbol: str
    conn: sqlite3.Connection
    nav: NavState
    logger: logging.Logger
    dry_run: bool
    confirm_flag: bool
    exchange_last_seen: dict[str, float] = field(default_factory=dict)


def run_decision_cycle(ctx: CycleContext) -> None:
    now = datetime.now(timezone.utc)
    ctx.logger.info(f"=== decision cycle @ {now.isoformat()} ===")

    open_positions = db_load_open_positions(ctx.conn)
    n_open = len(open_positions)
    ctx.logger.info(f"open positions: {n_open}")

    futures_ok, spot_ok, perp_btc, spot_btc, basis = reconcile(
        ctx.kf, ctx.ks, ctx.perp_symbol, open_positions, ctx.logger, ctx.conn,
    )
    if futures_ok:
        ctx.exchange_last_seen["krakenfutures"] = time.time()
    if spot_ok:
        ctx.exchange_last_seen["kraken"] = time.time()

    if n_open > 0:
        for ex_name in ("krakenfutures", "kraken"):
            last = ctx.exchange_last_seen.get(ex_name, 0.0)
            silent = time.time() - last
            if silent > EXCHANGE_OUTAGE_KILL_SECS:
                raise KillSwitchTriggered(
                    f"{ex_name} unreachable for {silent:.0f}s with open positions"
                )

    if basis is not None and abs(basis) > MAX_BASIS_DIVERGENCE and n_open > 0:
        raise KillSwitchTriggered(
            f"basis {basis*100:.2f}% exceeds {MAX_BASIS_DIVERGENCE*100:.2f}%"
        )

    move = fetch_recent_btc_move(ctx.kf, ctx.perp_symbol, ctx.logger, minutes=60)
    if move is not None and move > VOL_EVENT_MOVE:
        ctx.logger.warning(
            f"VOL EVENT: |1h move|={move*100:.2f}% > {VOL_EVENT_MOVE*100:.0f}% "
            "-> pause new entries this cycle."
        )
        vol_pause = True
    else:
        vol_pause = False

    ctx.nav.update_peak()
    dd = ctx.nav.drawdown()
    if dd > MAX_NAV_DRAWDOWN:
        raise KillSwitchTriggered(
            f"NAV drawdown {dd*100:.2f}% > {MAX_NAV_DRAWDOWN*100:.0f}%"
        )

    # Signal — pull enough history to fill the rolling window comfortably.
    df = fetch_recent_funding(ctx.kf, ctx.perp_symbol, ctx.logger,
                              lookback_cycles=ROLL_WINDOW * 4)
    cur_fund, roll_apr, sig = compute_signal(df)
    if roll_apr is not None:
        ctx.logger.info(
            f"signal: cur_funding={cur_fund} "
            f"roll_apr={roll_apr*100:.2f}% -> {sig}"
        )
    else:
        ctx.logger.info(
            f"signal: cur_funding={cur_fund} roll_apr=NA -> {sig}"
        )
    db_log_decision(
        ctx.conn, "hourly", cur_fund, roll_apr, sig, n_open, ctx.nav.nav_usd,
        note=f"basis_pct={(basis or 0)*100:.3f} vol_pause={vol_pause}",
    )

    if sig == "EXIT":
        for p in open_positions:
            close_position_atomic(
                ctx.kf, ctx.ks, ctx.perp_symbol, p, ctx.logger, ctx.conn,
                dry_run=ctx.dry_run, reason="signal_exit",
            )
        return

    if sig == "ENTRY" and not vol_pause and n_open < MAX_OPEN_POSITIONS:
        nxt = next_funding_window_ts(now)
        mins = (nxt - now).total_seconds() / 60.0
        ctx.logger.info(f"next funding window in {mins:.1f} min @ {nxt.isoformat()}")
        if (roll_apr is not None and roll_apr < ENTRY_APR + 0.01
                and mins < ENTRY_LOOKAHEAD_MIN_MINUTES):
            ctx.logger.info(
                "Borderline roll_apr and < lookahead window -> skip entry."
            )
            return

        perp_px = safe_ticker_last(ctx.kf, ctx.perp_symbol, ctx.logger)
        if perp_px is None or perp_px <= 0:
            ctx.logger.error("No perp price; cannot size entry.")
            return
        notional_usd = ctx.nav.nav_usd * PER_ENTRY_NAV_FRACTION
        qty_btc = round(notional_usd / perp_px, 6)
        if qty_btc <= 0:
            ctx.logger.error(f"Computed qty {qty_btc} <= 0; skip.")
            return

        trades_done = load_confirm_counter()
        if DEMO_ONLY_MODE or ctx.ks is None:
            spot_note = "spot leg=VIRTUAL (demo BTC inventory)"
        else:
            spot_note = f"spot leg will be capped at ${SPOT_LEG_MAX_USD:.0f}"
        label = (
            f"OPEN qty~{qty_btc} BTC notional~${notional_usd:.0f} "
            f"perp_px~{perp_px} ({spot_note})"
        )
        if not require_confirm_if_needed(ctx.logger, ctx.confirm_flag,
                                         trades_done, label):
            return

        pid = open_position_atomic(
            ctx.kf, ctx.ks, ctx.perp_symbol, qty_btc, ctx.logger, ctx.conn,
            dry_run=ctx.dry_run,
        )
        if pid is not None:
            save_confirm_counter(trades_done + 1)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main_loop(args: argparse.Namespace) -> int:
    global DEMO_ONLY_MODE
    DEMO_ONLY_MODE = bool(getattr(args, "demo_only", False))

    logger = setup_logging()
    install_signal_handlers(logger)
    logger.info("=== iter_3 KRAKEN BTC perp funding host starting ===")
    logger.info(f"dry_run={args.dry_run} flatten_now={args.flatten_now} "
                f"confirm={args.confirm} skip_startup_check={args.skip_startup_check} "
                f"demo_only={DEMO_ONLY_MODE}")

    kf = build_krakenfutures_sandbox(logger)
    if DEMO_ONLY_MODE:
        ks = None  # type: ignore[assignment]
        logger.info(
            "DEMO-ONLY MODE: Kraken Spot leg DISABLED. Using existing BTC "
            "inventory in Kraken Futures Demo account as the implicit long "
            "leg. Only the perp short will be traded."
        )
    else:
        ks = build_kraken_spot(logger)
    perp_symbol = resolve_perp_symbol(kf, logger)
    logger.info(f"Resolved perp symbol on krakenfutures = {perp_symbol}")
    if ks is not None:
        logger.info(f"Spot symbol on kraken = {SYMBOL_SPOT}")
        logger.info(f"Spot leg HARD CAP = ${SPOT_LEG_MAX_USD:.2f} per trade")

    conn = open_db()
    nav = NavState.load()
    logger.info(f"NAV state loaded: nav={nav.nav_usd:.2f} "
                f"peak={nav.peak_nav_usd:.2f}")

    if args.flatten_now:
        flatten_all(kf, ks, perp_symbol, logger, conn,
                    dry_run=args.dry_run, reason="manual_flatten")
        return 0

    if not args.skip_startup_check:
        ok = startup_verification(kf, perp_symbol, logger)
        if not ok:
            logger.error("Startup verification failed; exiting.")
            return 3
    else:
        logger.warning("Skipping startup verification (--skip-startup-check).")

    ctx = CycleContext(
        kf=kf,
        ks=ks,
        perp_symbol=perp_symbol,
        conn=conn,
        nav=nav,
        logger=logger,
        dry_run=args.dry_run,
        confirm_flag=args.confirm,
    )
    ctx.exchange_last_seen["krakenfutures"] = time.time()
    ctx.exchange_last_seen["kraken"] = time.time()

    last_cycle_ts = 0.0
    rc = 0
    while True:
        try:
            if SHUTDOWN_REQUESTED:
                logger.warning("Shutdown signal honored. Flushing state.")
                break
            if kill_file_present():
                logger.warning(
                    f"Kill file {KILL_FLAG_PATH} present. Flattening + exiting."
                )
                flatten_all(kf, ks, perp_symbol, logger, conn,
                            dry_run=args.dry_run, reason="kill_file")
                rc = 4
                break

            now_ts = time.time()
            if now_ts - last_cycle_ts >= RECONCILE_SECS:
                try:
                    run_decision_cycle(ctx)
                    last_cycle_ts = now_ts
                except KillSwitchTriggered as ks_err:
                    logger.error(f"KILL SWITCH: {ks_err.reason}")
                    flatten_all(kf, ks, perp_symbol, logger, conn,
                                dry_run=args.dry_run,
                                reason=f"kill:{ks_err.reason}")
                    rc = 5
                    break
                except Exception as exc:  # noqa: BLE001
                    logger.error(f"Decision cycle error: {exc}")
                    logger.error(traceback.format_exc())
                    time.sleep(30)

            write_daily_summary_if_due(conn, nav, logger)
            nav.save()

            time.sleep(KILL_FILE_POLL_SECS)
        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt; clean shutdown.")
            break

    nav.save()
    conn.close()
    logger.info(f"Exiting with code {rc}.")
    return rc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true",
                   help="Compute signals + log decisions, but never place "
                        "real orders. Synthesizes fills at last ticker.")
    p.add_argument("--confirm", action="store_true",
                   help="Interactively confirm each of the first "
                        f"{CONFIRM_TRADE_COUNT} trades.")
    p.add_argument("--flatten-now", action="store_true",
                   help="Close ALL open positions immediately, then exit.")
    p.add_argument("--skip-startup-check", action="store_true",
                   help="Skip the 30-day funding history sanity gate. "
                        "Operator override; logs a warning.")
    p.add_argument("--demo-only", action="store_true",
                   help="Single-account demo mode: use Kraken Futures Demo "
                        "BTC inventory as the implicit spot leg. Skip the "
                        "real Kraken Spot leg entirely. No real money at "
                        "risk. Recommended for week-1 paper shadow.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return main_loop(args)
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("iter_3_kraken_host").error(f"Top-level crash: {exc}")
        logging.getLogger("iter_3_kraken_host").error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
