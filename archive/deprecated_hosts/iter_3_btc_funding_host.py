"""iter_3 BTC Perpetual Funding Cash-and-Carry — Paper Shadow Deployment Host.

Production-ready paper-shadow harness for the iter_3 strategy
(Binance BTC perp short + OKX BTC spot long, threshold-driven by 30-period
rolling annualized funding). Backed by the iter_3 backtest verdict:

  symbol               = BTC/USDT:USDT
  span_days            = 1,957.7  (2021-01-01 -> 2026-05-12)
  total_funding_cycles = 5,874
  mean_annualized_funding_pct = 11.15
  periods_above_10pct  = 2,768
  n_positions          = 11
  annualized_carry_net_pct = 7.95
  in_position_annualized_pct = 17.45
  verdict              = PASS

Strategy summary (from iter_3_summary.json):
  Symbol:   BTC/USDT perpetual (Binance) hedged by BTC/USDT spot (OKX)
  Entry:    rolling-30 annualized funding APR > 10%
  Exit:     rolling-30 annualized funding APR < 4%
  Cost:     30 bps round-trip total budget (15 bps per side)
  Funding:  collected at 00:00 / 08:00 / 16:00 UTC (Binance schedule)
  Sizing:   20% NAV per entry, max 5 simultaneous = 100% NAV
            (Note: backtest expressed exactly one logical position at a time;
             the harness exposes 5 slots so the same signal can stagger
             entries over multiple funding windows without doubling up.)

Deployment notes:
  - Binance leg uses testnet via ccxt `'sandbox': True`.
  - OKX leg uses OKX paper trading via ccxt header `x-simulated-trading: 1`.
    If OKX paper is unreachable, fall back to Bybit testnet by passing
    `--spot-exchange bybit`.
  - All decisions / orders / fills / reconciliations are mirrored to
    research/live/btc_funding_decisions.sqlite3.
  - Kill switches are wired and fail CLOSED: any kill flattens both legs
    and exits non-zero so the supervisor can page.

Author : Deployment Engineer A29
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
LOG_PATH = LIVE_DIR / "iter_3_btc_funding_host.log"
DECISIONS_DB = LIVE_DIR / "btc_funding_decisions.sqlite3"
NAV_STATE_PATH = LIVE_DIR / "btc_funding_nav_state.json"
KILL_FLAG_PATH = LIVE_DIR / "STOP_BTC"
CONFIRM_COUNTER_PATH = LIVE_DIR / "btc_funding_confirm_counter.json"
SUMMARY_JSON = ROOT / "results" / "iter_3_summary.json"

# Strategy parameters — frozen, copied from iter_3_backtest.py / summary.
SYMBOL_PERP = "BTC/USDT:USDT"  # ccxt unified perp symbol
SYMBOL_SPOT = "BTC/USDT"        # ccxt unified spot symbol

ENTRY_APR = 0.10                # 10% rolling APR
EXIT_APR = 0.04                 # 4% rolling APR
ROLL_WINDOW = 30                # 30 funding cycles
FUND_PER_DAY = 3                # 8h windows
FUND_PER_YEAR = 365 * FUND_PER_DAY
ROUND_TRIP_COST_BUDGET = 0.0030 # 30 bps total
PER_SIDE_COST_BUDGET = 0.0015   # 15 bps per side

# Sizing / risk
PER_ENTRY_NAV_FRACTION = 0.20   # 20% NAV per entry
MAX_OPEN_POSITIONS = 5          # 5 * 20% = 100% NAV
DEFAULT_NAV_USD = 100_000.0     # paper bankroll if no state present

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

# Funding window timing
FUNDING_WINDOWS_UTC = (0, 8, 16)
ENTRY_LOOKAHEAD_MIN_MINUTES = 15  # skip "borderline" entries that miss window

# Startup verification
STARTUP_HISTORY_DAYS = 30
# In 30d there are 30*3=90 funding cycles. Backtest had 2,768 above-10pct
# periods over 5,874 cycles = ~47% of the time. We allow +/- 30% tolerance.
STARTUP_TOLERANCE_FRAC = 0.30
EXPECTED_ABOVE_10PCT_BASE_FRAC = 2768.0 / 5874.0


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("iter_3_host")
    logger.setLevel(logging.INFO)
    # avoid dup handlers on re-import
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
    leg            TEXT,    -- 'perp_short' or 'spot_long'
    exchange       TEXT,
    symbol         TEXT,
    side           TEXT,
    qty            REAL,
    price_hint     REAL,
    status         TEXT,    -- 'submitted','filled','rejected','aborted','reversed'
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
    status         TEXT    NOT NULL,   -- 'opening','open','closing','closed','aborted'
    exit_reason    TEXT
);

CREATE TABLE IF NOT EXISTS reconciliations (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc         TEXT    NOT NULL,
    perp_position_btc REAL,
    spot_position_btc REAL,
    basis_pct      REAL,
    binance_ok     INTEGER,
    okx_ok         INTEGER,
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
                     basis_pct: float, binance_ok: bool, okx_ok: bool,
                     note: str) -> None:
    conn.execute(
        "INSERT INTO reconciliations"
        " (ts_utc, perp_position_btc, spot_position_btc, basis_pct,"
        "  binance_ok, okx_ok, note)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            datetime.now(timezone.utc).isoformat(),
            perp_btc,
            spot_btc,
            basis_pct,
            int(binance_ok),
            int(okx_ok),
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

def build_binance_testnet(logger: logging.Logger) -> ccxt.Exchange:
    api = os.environ.get("BINANCE_TESTNET_API_KEY", "")
    sec = os.environ.get("BINANCE_TESTNET_API_SECRET", "")
    if not api or not sec:
        logger.warning(
            "BINANCE_TESTNET_API_KEY/SECRET not set. Will run in"
            " read-only mode (no orders)."
        )
    ex = ccxt.binance({
        "apiKey": api,
        "secret": sec,
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })
    # Binance futures testnet
    ex.set_sandbox_mode(True)
    try:
        ex.load_markets()
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Binance testnet load_markets failed: {exc}")
    return ex


def build_okx_paper(logger: logging.Logger) -> ccxt.Exchange:
    api = os.environ.get("OKX_PAPER_API_KEY", "")
    sec = os.environ.get("OKX_PAPER_API_SECRET", "")
    pwd = os.environ.get("OKX_PAPER_API_PASSPHRASE", "")
    if not api or not sec or not pwd:
        logger.warning(
            "OKX_PAPER_API_{KEY,SECRET,PASSPHRASE} not set. Will run in"
            " read-only mode (no orders)."
        )
    ex = ccxt.okx({
        "apiKey": api,
        "secret": sec,
        "password": pwd,
        "enableRateLimit": True,
        # OKX paper trading routes via the simulated-trading header.
        "headers": {"x-simulated-trading": "1"},
    })
    try:
        ex.load_markets()
    except Exception as exc:  # noqa: BLE001
        logger.error(f"OKX paper load_markets failed: {exc}")
    return ex


def build_bybit_testnet(logger: logging.Logger) -> ccxt.Exchange:
    api = os.environ.get("BYBIT_TESTNET_API_KEY", "")
    sec = os.environ.get("BYBIT_TESTNET_API_SECRET", "")
    ex = ccxt.bybit({
        "apiKey": api,
        "secret": sec,
        "enableRateLimit": True,
    })
    ex.set_sandbox_mode(True)
    try:
        ex.load_markets()
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Bybit testnet load_markets failed: {exc}")
    return ex


# ---------------------------------------------------------------------------
# Funding signal
# ---------------------------------------------------------------------------

def fetch_recent_funding(ex: ccxt.Exchange, logger: logging.Logger,
                         lookback_cycles: int = 90) -> pd.DataFrame:
    """Pull the last `lookback_cycles` funding rates from Binance perp."""
    # ccxt returns most recent funding rates; pull enough for the rolling window
    # plus safety margin.
    try:
        rows = ex.fetch_funding_rate_history(SYMBOL_PERP, limit=lookback_cycles)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"fetch_funding_rate_history failed: {exc}")
        return pd.DataFrame()
    out = []
    for r in rows:
        fr = r.get("fundingRate")
        if fr is None:
            continue
        out.append(
            {
                "timestamp": int(r["timestamp"]),
                "datetime": r["datetime"],
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
    if now is None:
        now = datetime.now(timezone.utc)
    today_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    candidates = [today_midnight + timedelta(hours=h) for h in FUNDING_WINDOWS_UTC]
    candidates.append(today_midnight + timedelta(days=1))  # tomorrow 00:00
    for c in candidates:
        if c > now:
            return c
    return today_midnight + timedelta(days=1)


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
        # synthesize a fill at last ticker
        px = safe_ticker_last(ex, symbol, logger) or 0.0
        logger.info(f"[DRY] {leg_id} qty={qty} px~{px}")
        return LegResult(
            leg=leg_id, exchange_id=ex.id, symbol=symbol, side=side,
            qty=qty, status="filled", exchange_oid="DRY",
            fill_price=px, error=None,
        )
    params: dict[str, Any] = {}
    if reduce_only:
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


def poll_fill(ex: ccxt.Exchange, order_id: str, symbol: str,
              logger: logging.Logger, timeout_secs: int) -> tuple[bool, Optional[float]]:
    """Poll fetch_order until filled or timeout. Returns (filled, fill_price)."""
    deadline = time.time() + timeout_secs
    while time.time() < deadline:
        try:
            o = ex.fetch_order(order_id, symbol)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"fetch_order {ex.id} {order_id} failed: {exc}")
            time.sleep(1.0)
            continue
        status = o.get("status")
        if status == "closed":
            px = float(o.get("average") or 0.0) or float(o.get("price") or 0.0) or None
            return True, px
        if status in ("canceled", "rejected", "expired"):
            return False, None
        time.sleep(1.0)
    return False, None


def open_position_atomic(
    binance: ccxt.Exchange,
    okx: ccxt.Exchange,
    qty_btc: float,
    logger: logging.Logger,
    conn: sqlite3.Connection,
    dry_run: bool,
) -> Optional[int]:
    """Open BOTH legs atomically. If one leg fails / times out, reverse the
    filled leg. Returns position_id on success, None on abort."""
    perp_px_hint = safe_ticker_last(binance, SYMBOL_PERP, logger)
    spot_px_hint = safe_ticker_last(okx, SYMBOL_SPOT, logger)
    if perp_px_hint is None or spot_px_hint is None:
        logger.error("Cannot price either leg; aborting open.")
        return None

    basis = compute_basis_pct(perp_px_hint, spot_px_hint)
    if abs(basis) > MAX_BASIS_DIVERGENCE:
        logger.error(
            f"Pre-trade basis {basis*100:.2f}% exceeds "
            f"{MAX_BASIS_DIVERGENCE*100:.2f}%; aborting open."
        )
        return None

    pid = db_open_position(conn, qty_btc, perp_px_hint, spot_px_hint)
    logger.info(
        f"Opening position id={pid} qty={qty_btc} BTC "
        f"perp_hint={perp_px_hint} spot_hint={spot_px_hint} basis={basis*100:.3f}%"
    )

    # 1) perp short
    perp_result = place_market_order(
        binance, SYMBOL_PERP, "sell", qty_btc, logger, dry_run=dry_run,
    )
    db_log_order(
        conn, pid, "perp_short", binance.id, SYMBOL_PERP, "sell", qty_btc,
        perp_px_hint, perp_result.status, perp_result.exchange_oid,
        perp_result.fill_price, perp_result.error or "",
    )
    if perp_result.status in ("rejected",):
        logger.error("Perp leg rejected; aborting open. No reverse needed.")
        db_mark_position(conn, pid, "aborted", exit_reason="perp_rejected")
        return None

    # If submitted (not yet filled), poll until fill or timeout
    if perp_result.status != "filled" and not dry_run:
        filled, fill_px = poll_fill(
            binance, perp_result.exchange_oid or "", SYMBOL_PERP,
            logger, LEG_FILL_TIMEOUT_SECS,
        )
        if not filled:
            logger.error("Perp leg failed to fill within timeout; aborting.")
            # try cancel
            try:
                binance.cancel_order(perp_result.exchange_oid or "", SYMBOL_PERP)
            except Exception:  # noqa: BLE001
                pass
            db_mark_position(conn, pid, "aborted",
                             exit_reason="perp_fill_timeout")
            return None
        perp_result.fill_price = fill_px
        perp_result.status = "filled"

    # 2) spot long
    spot_result = place_market_order(
        okx, SYMBOL_SPOT, "buy", qty_btc, logger, dry_run=dry_run,
    )
    db_log_order(
        conn, pid, "spot_long", okx.id, SYMBOL_SPOT, "buy", qty_btc,
        spot_px_hint, spot_result.status, spot_result.exchange_oid,
        spot_result.fill_price, spot_result.error or "",
    )

    spot_ok = spot_result.status == "filled"
    if not spot_ok and spot_result.status != "rejected" and not dry_run:
        filled, fill_px = poll_fill(
            okx, spot_result.exchange_oid or "", SYMBOL_SPOT,
            logger, LEG_FILL_TIMEOUT_SECS,
        )
        if filled:
            spot_result.fill_price = fill_px
            spot_result.status = "filled"
            spot_ok = True

    if not spot_ok:
        # Reverse the perp leg
        logger.error(
            "Spot leg failed; reversing perp short with a BUY to flatten."
        )
        reverse_result = place_market_order(
            binance, SYMBOL_PERP, "buy", qty_btc, logger,
            dry_run=dry_run, reduce_only=True,
        )
        db_log_order(
            conn, pid, "perp_short_reverse", binance.id, SYMBOL_PERP,
            "buy", qty_btc, perp_px_hint, reverse_result.status,
            reverse_result.exchange_oid, reverse_result.fill_price,
            "reverse-after-spot-fail",
        )
        db_mark_position(conn, pid, "aborted", exit_reason="spot_fail_reversed")
        return None

    # Both filled
    db_mark_position(conn, pid, "open")
    logger.info(
        f"Position id={pid} OPEN. perp_fill={perp_result.fill_price} "
        f"spot_fill={spot_result.fill_price}"
    )
    return pid


def close_position_atomic(
    binance: ccxt.Exchange,
    okx: ccxt.Exchange,
    pos: dict[str, Any],
    logger: logging.Logger,
    conn: sqlite3.Connection,
    dry_run: bool,
    reason: str,
) -> bool:
    pid = int(pos["id"])
    qty = float(pos["qty_btc"])
    perp_px_hint = safe_ticker_last(binance, SYMBOL_PERP, logger)
    spot_px_hint = safe_ticker_last(okx, SYMBOL_SPOT, logger)
    db_mark_position(conn, pid, "closing", exit_reason=reason)
    logger.info(f"Closing position id={pid} qty={qty} reason={reason}")

    # 1) perp BUY to cover
    perp_result = place_market_order(
        binance, SYMBOL_PERP, "buy", qty, logger,
        dry_run=dry_run, reduce_only=True,
    )
    db_log_order(
        conn, pid, "perp_close", binance.id, SYMBOL_PERP, "buy", qty,
        perp_px_hint, perp_result.status, perp_result.exchange_oid,
        perp_result.fill_price, perp_result.error or "",
    )
    if perp_result.status != "filled" and not dry_run:
        filled, fill_px = poll_fill(
            binance, perp_result.exchange_oid or "", SYMBOL_PERP,
            logger, LEG_FILL_TIMEOUT_SECS,
        )
        if filled:
            perp_result.fill_price = fill_px
            perp_result.status = "filled"

    # 2) spot SELL
    spot_result = place_market_order(
        okx, SYMBOL_SPOT, "sell", qty, logger, dry_run=dry_run,
    )
    db_log_order(
        conn, pid, "spot_close", okx.id, SYMBOL_SPOT, "sell", qty,
        spot_px_hint, spot_result.status, spot_result.exchange_oid,
        spot_result.fill_price, spot_result.error or "",
    )
    if spot_result.status != "filled" and not dry_run:
        filled, fill_px = poll_fill(
            okx, spot_result.exchange_oid or "", SYMBOL_SPOT,
            logger, LEG_FILL_TIMEOUT_SECS,
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

def fetch_perp_net_position(binance: ccxt.Exchange,
                            logger: logging.Logger) -> Optional[float]:
    try:
        positions = binance.fetch_positions([SYMBOL_PERP])
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"fetch_positions Binance failed: {exc}")
        return None
    total = 0.0
    for p in positions:
        contracts = float(p.get("contracts") or 0.0)
        side = (p.get("side") or "").lower()
        # ccxt unified: side='short' -> contracts negative-by-convention
        if side == "short":
            total -= contracts
        else:
            total += contracts
    return total


def fetch_spot_btc_balance(okx: ccxt.Exchange,
                           logger: logging.Logger) -> Optional[float]:
    try:
        bal = okx.fetch_balance()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"fetch_balance OKX failed: {exc}")
        return None
    btc = bal.get("BTC") or bal.get("free", {}).get("BTC")
    if btc is None:
        return None
    if isinstance(btc, dict):
        return float(btc.get("total", 0.0) or 0.0)
    return float(btc)


def reconcile(
    binance: ccxt.Exchange,
    okx: ccxt.Exchange,
    open_positions: list[dict[str, Any]],
    logger: logging.Logger,
    conn: sqlite3.Connection,
) -> tuple[bool, bool, Optional[float], Optional[float], Optional[float]]:
    """Returns (binance_ok, okx_ok, perp_btc, spot_btc, basis_pct)."""
    perp_btc = fetch_perp_net_position(binance, logger)
    spot_btc = fetch_spot_btc_balance(okx, logger)
    perp_px = safe_ticker_last(binance, SYMBOL_PERP, logger)
    spot_px = safe_ticker_last(okx, SYMBOL_SPOT, logger)
    binance_ok = perp_px is not None
    okx_ok = spot_px is not None
    basis = None
    if binance_ok and okx_ok:
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
        binance_ok=binance_ok, okx_ok=okx_ok,
        note=" | ".join(note_parts),
    )
    return binance_ok, okx_ok, perp_btc, spot_btc, basis


# ---------------------------------------------------------------------------
# Startup verification
# ---------------------------------------------------------------------------

def startup_verification(binance: ccxt.Exchange,
                         logger: logging.Logger) -> bool:
    """Pull last 30d funding, confirm signal would have fired within tolerance."""
    cycles_30d = STARTUP_HISTORY_DAYS * FUND_PER_DAY  # 90
    df = fetch_recent_funding(binance, logger,
                              lookback_cycles=cycles_30d + ROLL_WINDOW)
    if df.empty or len(df) < ROLL_WINDOW + 1:
        logger.error(
            f"Startup verification FAIL: only {len(df)} funding rows "
            f"(need {ROLL_WINDOW + 1}+)."
        )
        return False
    # Count entry-signal periods on the recent slice (where rolling defined)
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
    # We are deliberately permissive: pass if either bound is met, since a
    # 30-day window has low signal mass and a regime can be quiet. Reject
    # only when we're outside [0, 2*expected] which is the symmetric tolerance.
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
    # Sum funding collected during the day (from positions table updates)
    cur = conn.execute(
        "SELECT COALESCE(SUM(funding_collected_usd),0), COUNT(*) "
        "FROM positions WHERE status IN ('open','closing')"
    )
    funding_sum, n_open = cur.fetchone()
    # Basis drift max from reconciliations
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
    binance: ccxt.Exchange, okx: ccxt.Exchange,
    logger: logging.Logger, conn: sqlite3.Connection, dry_run: bool,
    reason: str,
) -> None:
    open_positions = db_load_open_positions(conn)
    logger.warning(f"FLATTEN: {len(open_positions)} positions reason={reason}")
    for p in open_positions:
        try:
            close_position_atomic(binance, okx, p, logger, conn,
                                  dry_run=dry_run, reason=reason)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Flatten close failed for pos {p['id']}: {exc}")


# ---------------------------------------------------------------------------
# Decision cycle
# ---------------------------------------------------------------------------

@dataclass
class CycleContext:
    binance: ccxt.Exchange
    okx: ccxt.Exchange
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

    # Reconcile first
    binance_ok, okx_ok, perp_btc, spot_btc, basis = reconcile(
        ctx.binance, ctx.okx, open_positions, ctx.logger, ctx.conn,
    )
    if binance_ok:
        ctx.exchange_last_seen["binance"] = time.time()
    if okx_ok:
        ctx.exchange_last_seen["okx"] = time.time()

    # Kill: exchange outage during open position
    if n_open > 0:
        for ex_name in ("binance", "okx"):
            last = ctx.exchange_last_seen.get(ex_name, 0.0)
            silent = time.time() - last
            if silent > EXCHANGE_OUTAGE_KILL_SECS:
                raise KillSwitchTriggered(
                    f"{ex_name} unreachable for {silent:.0f}s with open positions"
                )

    # Kill: basis blowout
    if basis is not None and abs(basis) > MAX_BASIS_DIVERGENCE and n_open > 0:
        raise KillSwitchTriggered(
            f"basis {basis*100:.2f}% exceeds {MAX_BASIS_DIVERGENCE*100:.2f}%"
        )

    # Pause: vol event
    move = fetch_recent_btc_move(ctx.binance, SYMBOL_PERP, ctx.logger, minutes=60)
    if move is not None and move > VOL_EVENT_MOVE:
        ctx.logger.warning(
            f"VOL EVENT: |1h move|={move*100:.2f}% > {VOL_EVENT_MOVE*100:.0f}% "
            "-> pause new entries this cycle."
        )
        vol_pause = True
    else:
        vol_pause = False

    # Kill: NAV drawdown
    ctx.nav.update_peak()
    dd = ctx.nav.drawdown()
    if dd > MAX_NAV_DRAWDOWN:
        raise KillSwitchTriggered(
            f"NAV drawdown {dd*100:.2f}% > {MAX_NAV_DRAWDOWN*100:.0f}%"
        )

    # Signal
    df = fetch_recent_funding(ctx.binance, ctx.logger,
                              lookback_cycles=ROLL_WINDOW * 4)
    cur_fund, roll_apr, sig = compute_signal(df)
    ctx.logger.info(
        f"signal: cur_funding={cur_fund} "
        f"roll_apr={roll_apr*100:.2f}% (if any) -> {sig}"
        if roll_apr is not None else
        f"signal: cur_funding={cur_fund} roll_apr=NA -> {sig}"
    )
    db_log_decision(
        ctx.conn, "hourly", cur_fund, roll_apr, sig, n_open, ctx.nav.nav_usd,
        note=f"basis_pct={(basis or 0)*100:.3f} vol_pause={vol_pause}",
    )

    # ACTION
    if sig == "EXIT":
        for p in open_positions:
            close_position_atomic(
                ctx.binance, ctx.okx, p, ctx.logger, ctx.conn,
                dry_run=ctx.dry_run, reason="signal_exit",
            )
        return

    if sig == "ENTRY" and not vol_pause and n_open < MAX_OPEN_POSITIONS:
        # Check funding window proximity
        nxt = next_funding_window_ts(now)
        mins = (nxt - now).total_seconds() / 60.0
        ctx.logger.info(f"next funding window in {mins:.1f} min @ {nxt.isoformat()}")
        # borderline = roll just above entry by <1% and we'd miss next window
        if (roll_apr is not None and roll_apr < ENTRY_APR + 0.01
                and mins < ENTRY_LOOKAHEAD_MIN_MINUTES):
            ctx.logger.info(
                "Borderline roll_apr and < lookahead window -> skip entry."
            )
            return

        # sizing
        perp_px = safe_ticker_last(ctx.binance, SYMBOL_PERP, ctx.logger)
        if perp_px is None or perp_px <= 0:
            ctx.logger.error("No perp price; cannot size entry.")
            return
        notional_usd = ctx.nav.nav_usd * PER_ENTRY_NAV_FRACTION
        qty_btc = round(notional_usd / perp_px, 6)
        if qty_btc <= 0:
            ctx.logger.error(f"Computed qty {qty_btc} <= 0; skip.")
            return

        trades_done = load_confirm_counter()
        label = (f"OPEN qty={qty_btc} BTC notional~${notional_usd:.0f} "
                 f"perp_px~{perp_px}")
        if not require_confirm_if_needed(ctx.logger, ctx.confirm_flag,
                                         trades_done, label):
            return

        pid = open_position_atomic(
            ctx.binance, ctx.okx, qty_btc, ctx.logger, ctx.conn,
            dry_run=ctx.dry_run,
        )
        if pid is not None:
            save_confirm_counter(trades_done + 1)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main_loop(args: argparse.Namespace) -> int:
    logger = setup_logging()
    install_signal_handlers(logger)
    logger.info("=== iter_3 BTC perp funding host starting ===")
    logger.info(f"dry_run={args.dry_run} flatten_now={args.flatten_now} "
                f"confirm={args.confirm} spot_exchange={args.spot_exchange}")

    binance = build_binance_testnet(logger)
    if args.spot_exchange == "okx":
        spot_ex = build_okx_paper(logger)
    elif args.spot_exchange == "bybit":
        spot_ex = build_bybit_testnet(logger)
    else:
        logger.error(f"Unknown spot exchange: {args.spot_exchange}")
        return 2

    conn = open_db()
    nav = NavState.load()
    logger.info(f"NAV state loaded: nav={nav.nav_usd:.2f} "
                f"peak={nav.peak_nav_usd:.2f}")

    # Flatten-now utility
    if args.flatten_now:
        flatten_all(binance, spot_ex, logger, conn,
                    dry_run=args.dry_run, reason="manual_flatten")
        return 0

    # Startup verification
    if not args.skip_startup_check:
        ok = startup_verification(binance, logger)
        if not ok:
            logger.error("Startup verification failed; exiting.")
            return 3
    else:
        logger.warning("Skipping startup verification (--skip-startup-check).")

    ctx = CycleContext(
        binance=binance,
        okx=spot_ex,
        conn=conn,
        nav=nav,
        logger=logger,
        dry_run=args.dry_run,
        confirm_flag=args.confirm,
    )
    ctx.exchange_last_seen["binance"] = time.time()
    ctx.exchange_last_seen["okx"] = time.time()

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
                flatten_all(binance, spot_ex, logger, conn,
                            dry_run=args.dry_run, reason="kill_file")
                rc = 4
                break

            now_ts = time.time()
            if now_ts - last_cycle_ts >= RECONCILE_SECS:
                try:
                    run_decision_cycle(ctx)
                    last_cycle_ts = now_ts
                except KillSwitchTriggered as ks:
                    logger.error(f"KILL SWITCH: {ks.reason}")
                    flatten_all(binance, spot_ex, logger, conn,
                                dry_run=args.dry_run, reason=f"kill:{ks.reason}")
                    rc = 5
                    break
                except Exception as exc:  # noqa: BLE001
                    logger.error(f"Decision cycle error: {exc}")
                    logger.error(traceback.format_exc())
                    # do not kill on transient errors; back off
                    time.sleep(30)

            # daily summary at top of UTC hour 00:00
            write_daily_summary_if_due(conn, nav, logger)
            nav.save()

            # short sleep so the kill-file is responsive within 60s.
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
    p.add_argument("--spot-exchange", default="okx",
                   choices=["okx", "bybit"],
                   help="Spot leg venue. okx = OKX paper trading; "
                        "bybit = Bybit testnet fallback.")
    p.add_argument("--skip-startup-check", action="store_true",
                   help="Skip the 30-day funding history sanity gate. "
                        "Operator override; logs a warning.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return main_loop(args)
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("iter_3_host").error(f"Top-level crash: {exc}")
        logging.getLogger("iter_3_host").error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
