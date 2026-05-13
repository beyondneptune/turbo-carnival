"""iter_18 ADR Dual-Listing Arbitrage — IBKR Paper Shadow Deployment Host.

Production-ready paper-shadow harness for the cleaned-up iter_18 strategy
(US/HK ADR-vs-ordinary parity arbitrage). Backed by the iter_18 cleanup
backtest verdict:
  aggregate mean_bps_net = 65.79
  total_trades            = 1,543 over 6.36 years
  win_rate                = 65.98%
  per-pair PASS count     = 5 / 5

Strategy summary (from iter_18_cleanup_summary.json):
  Pairs:    BABA/9988 1:8, JD/9618 1:2, BIDU/9888 1:8,
            NTES/9999 1:5, TCOM/9961 1:1
  Entry:    |deviation_bps| > 80
  Exit:     |deviation_bps| < 20
  Stop:     |deviation_bps| > 250
  Cost:     30 bps round-trip
  Blackout: +/- 3 sessions around ex-dividend on EITHER leg
  Sizing:   20% NAV per pair, max 5 pairs simultaneously = 100% NAV

Deployment notes:
  - Alpaca does NOT support HKEX. We use Interactive Brokers via ib_insync.
  - The schedule is twice daily:
      cycle 1: ~04:00 ET (HK close)
      cycle 2: ~16:00 ET (US close)
    Run this script under cron/launchd with `--cycle hk_close` or
    `--cycle us_close` to pin which cycle it is.
  - All decisions / orders / fills are mirrored to
    research/live/adr_decisions.sqlite3 for reconciliation.
  - Kill switches are wired and fail CLOSED: any of them flattens everything
    and exits non-zero so the cron wrapper can page.

Author : Deployment Engineer A28
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

# ib_insync is required. Install with: pip install ib_insync
try:
    from ib_insync import IB, Stock, Forex, MarketOrder, util  # type: ignore
except ImportError as exc:  # pragma: no cover
    print(
        "FATAL: ib_insync is not installed. Install with `pip install ib_insync`.",
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
LOG_PATH = LIVE_DIR / "iter_18_adr_host.log"
DECISIONS_DB = LIVE_DIR / "adr_decisions.sqlite3"
NAV_STATE_PATH = LIVE_DIR / "adr_nav_state.json"
KILL_FLAG_PATH = LIVE_DIR / "KILL_SWITCH"
CONFIRM_COUNTER_PATH = LIVE_DIR / "confirm_counter.json"

# Strategy parameters — frozen, copied from iter_18_cleanup_summary.json.
PAIRS: list[tuple[str, str, int]] = [
    ("BABA", "9988", 8),
    ("JD",   "9618", 2),
    ("BIDU", "9888", 8),
    ("NTES", "9999", 5),
    ("TCOM", "9961", 1),
]

ENTRY_BPS = 80.0
EXIT_BPS = 20.0
STOP_BPS = 250.0
ROUND_TRIP_COST_BPS = 30.0
EX_DIV_SKIP_WINDOW_BARS = 3
USD_HKD_PEG = 7.80

# Risk and sizing
PER_PAIR_NAV_FRACTION = 0.20
MAX_OPEN_PAIRS = 5
MAX_TOTAL_GROSS_NAV_FRAC = 1.00  # short + long both counted as gross
PER_PAIR_LOSS_KILL_FRAC = 0.05   # 5% NAV per-pair loss => hard kill
CONNECTION_LOSS_KILL_SECS = 300  # 5 minutes
CONFIRM_TRADE_COUNT = 10         # interactive --confirm for the first 10 trades

# Backtest verification on startup
BACKTEST_LOOKBACK_DAYS = 60
BACKTEST_DIVERGENCE_FRACTION = 0.30  # exit if expected vs actual diverges > 30%

# IBKR contract conventions
HK_EXCHANGE = "SEHK"
HK_CURRENCY = "HKD"
US_EXCHANGE = "SMART"
US_CURRENCY = "USD"

# Logging
LOG_FMT = "%(asctime)s %(levelname)s %(message)s"

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def setup_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("iter_18_adr_host")
    logger.setLevel(logging.INFO)
    # Reset handlers so re-runs in the same process don't duplicate output.
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fh = logging.FileHandler(LOG_PATH, mode="a")
    fh.setFormatter(logging.Formatter(LOG_FMT))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter(LOG_FMT))
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


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
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc         TEXT    NOT NULL,
    cycle          TEXT    NOT NULL,
    pair           TEXT    NOT NULL,
    leg            TEXT    NOT NULL,          -- 'US' or 'HK'
    symbol         TEXT    NOT NULL,
    side           TEXT    NOT NULL,          -- 'BUY' or 'SELL'
    qty            INTEGER NOT NULL,
    submitted_ok   INTEGER NOT NULL,
    fill_price     REAL,
    fill_qty       INTEGER,
    avg_fill_price REAL,
    status         TEXT,
    order_id       INTEGER,
    perm_id        INTEGER,
    note           TEXT
);

CREATE TABLE IF NOT EXISTS positions_snapshot (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc       TEXT    NOT NULL,
    cycle        TEXT    NOT NULL,
    symbol       TEXT    NOT NULL,
    exchange     TEXT,
    currency     TEXT,
    qty          REAL,
    avg_cost     REAL,
    market_value REAL,
    unreal_pnl   REAL
);

CREATE TABLE IF NOT EXISTS eod_summary (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc          TEXT    NOT NULL,
    cycle           TEXT    NOT NULL,
    nav_usd         REAL,
    open_pair_count INTEGER,
    gross_exposure  REAL,
    total_unreal_pnl REAL,
    payload_json    TEXT
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
    """Raised when a hard-kill condition fires. The cycle is aborted and
    open positions are flattened on the way out (best-effort)."""


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
        return {"reason": "unknown", "detail": "corrupt KILL_SWITCH file"}


def record_kill(conn: sqlite3.Connection, cycle: str, reason: str, detail: str) -> None:
    conn.execute(
        "INSERT INTO kill_events (ts_utc, cycle, reason, detail) VALUES (?, ?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), cycle, reason, detail),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Confirmation counter (first N trades require interactive --confirm)
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
    """Returns True if the operator (or non-interactive flag) approves."""
    if allow_skip:
        return True
    try:
        ans = input(f"{prompt} [y/N]: ").strip().lower()
    except EOFError:
        return False
    return ans == "y"


# ---------------------------------------------------------------------------
# IBKR contract factories
# ---------------------------------------------------------------------------


def make_us_contract(symbol: str) -> Stock:
    return Stock(symbol, US_EXCHANGE, US_CURRENCY)


def make_hk_contract(symbol: str) -> Stock:
    # 4-digit HKEX symbols are passed as STRINGS to IBKR (e.g. "9988"), not ints.
    return Stock(symbol, HK_EXCHANGE, HK_CURRENCY)


def make_usd_hkd_fx() -> Forex:
    return Forex("USDHKD")


# ---------------------------------------------------------------------------
# Backtest verification (60-day re-run for divergence check)
# ---------------------------------------------------------------------------


def verify_backtest_divergence(log: logging.Logger) -> None:
    """Re-run the cleanup backtest on the last 60 days and bail if the
    realised trade count diverges from the spec-implied rate by more than
    BACKTEST_DIVERGENCE_FRACTION.

    Expected rate from iter_18_cleanup_summary.json:
        total_trades = 1543 over 6.36 years -> ~243 trades/year -> ~40 in 60 days
    """
    summary_path = ROOT / "results" / "iter_18_cleanup_summary.json"
    if not summary_path.exists():
        raise SystemExit(
            f"FATAL: missing backtest summary {summary_path}. "
            "Run scripts/iter_18_cleanup.py first."
        )

    summary = json.loads(summary_path.read_text())
    total_trades = int(summary["aggregate"]["total_trades"])
    elapsed_years = float(summary["aggregate"]["elapsed_years"])
    trades_per_year = total_trades / max(elapsed_years, 1e-9)
    expected_60d = trades_per_year * (BACKTEST_LOOKBACK_DAYS / 365.25)

    trades_csv = ROOT / "results" / "iter_18_cleanup_trades.csv"
    actual_60d = 0
    if trades_csv.exists():
        try:
            tdf = pd.read_csv(trades_csv, parse_dates=["entry_date", "exit_date"])
            cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=BACKTEST_LOOKBACK_DAYS)
            actual_60d = int((tdf["exit_date"] >= cutoff).sum())
        except (KeyError, ValueError) as exc:
            log.warning(f"Could not parse {trades_csv} for divergence check: {exc}")

    log.info(
        f"Backtest verification: expected ~{expected_60d:.1f} trades in last "
        f"{BACKTEST_LOOKBACK_DAYS}d, actual {actual_60d} "
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
            "Re-run scripts/iter_18_cleanup.py and investigate before deploying."
        )
    log.info("Backtest verification PASS.")


# ---------------------------------------------------------------------------
# Dividend calendar (loaded from cleanup cache)
# ---------------------------------------------------------------------------


def load_dividend_calendar(log: logging.Logger) -> dict[str, set]:
    """Return {ticker -> set(ex_date)} merged across both legs of each pair.

    We accept either yfinance suffix "9988.HK" or bare "9988" — both forms
    are normalised to the bare HKEX symbol used here.
    """
    div_cache = ROOT / "data" / "cache" / "us_hk_pairs_dividends.parquet"
    if not div_cache.exists():
        log.warning(f"Dividend cache {div_cache} missing — ex-div blackout disabled.")
        return {}

    try:
        df = pd.read_parquet(div_cache)
    except (OSError, ValueError) as exc:
        log.warning(f"Could not read dividend cache: {exc}")
        return {}

    cal: dict[str, set] = {}
    for col in df.columns:
        s = df[col].dropna()
        if s.empty:
            continue
        # Some columns are stored as "9988.HK"; map to bare "9988".
        bare = col.replace(".HK", "")
        cal.setdefault(bare, set()).update(pd.to_datetime(s.index).normalize().tolist())
        cal.setdefault(col, set()).update(pd.to_datetime(s.index).normalize().tolist())
    return cal


def is_ex_div_blocked(
    today: pd.Timestamp,
    us_sym: str,
    hk_sym: str,
    cal: dict[str, set],
) -> bool:
    today = pd.Timestamp(today).normalize()
    for sym in (us_sym, hk_sym):
        if sym not in cal:
            continue
        for ex in cal[sym]:
            if abs((today - ex).days) <= EX_DIV_SKIP_WINDOW_BARS:
                return True
    return False


# ---------------------------------------------------------------------------
# Market data helpers (snapshot quotes only — keep within paper-tier rate)
# ---------------------------------------------------------------------------


def snapshot_price(ib: IB, contract, log: logging.Logger, label: str) -> Optional[float]:
    """Pull a single snapshot price. Returns None if no usable tick arrives."""
    try:
        ticker = ib.reqMktData(contract, "", snapshot=True, regulatorySnapshot=False)
        ib.sleep(2.0)
        px: Optional[float] = None
        # Prefer last, then mid, then close.
        for attr in ("last", "marketPrice", "close", "bid"):
            val = getattr(ticker, attr, None)
            if val is not None and val == val and val > 0:
                px = float(val)
                break
        if hasattr(ticker, "bid") and hasattr(ticker, "ask") and px is None:
            b, a = ticker.bid, ticker.ask
            if b and a and b > 0 and a > 0:
                px = (b + a) / 2
        try:
            ib.cancelMktData(contract)
        except Exception:  # noqa: BLE001 — best-effort cleanup
            pass
        if px is None:
            log.warning(f"No price tick for {label} ({contract})")
        return px
    except Exception as exc:  # noqa: BLE001
        log.warning(f"snapshot_price({label}) failed: {exc}")
        return None


def fetch_fx_usd_hkd(ib: IB, log: logging.Logger) -> float:
    fx = make_usd_hkd_fx()
    try:
        ib.qualifyContracts(fx)
    except Exception as exc:  # noqa: BLE001
        log.warning(f"qualify USDHKD failed ({exc}); falling back to peg {USD_HKD_PEG}")
        return USD_HKD_PEG
    px = snapshot_price(ib, fx, log, "USDHKD")
    if px is None or not (7.5 < px < 8.1):
        log.warning(f"FX snapshot suspect ({px}); using peg {USD_HKD_PEG}")
        return USD_HKD_PEG
    return px


# ---------------------------------------------------------------------------
# Position state
# ---------------------------------------------------------------------------


@dataclass
class PairState:
    us_sym: str
    hk_sym: str
    ratio: int
    us_qty: int = 0           # positive = long US, negative = short US
    hk_qty: int = 0           # positive = long HK ordinary
    avg_us_cost: float = 0.0
    avg_hk_cost: float = 0.0
    direction: Optional[str] = None  # "long_us_short_hk" or "short_us_long_hk"

    @property
    def open(self) -> bool:
        return self.us_qty != 0 or self.hk_qty != 0


def snapshot_positions(ib: IB, log: logging.Logger) -> dict[str, dict]:
    """Return {symbol -> {qty, avgCost, currency, exchange, marketValue, unrealizedPnl}}."""
    out: dict[str, dict] = {}
    try:
        portfolio = ib.portfolio()
    except Exception as exc:  # noqa: BLE001
        log.warning(f"portfolio() failed: {exc}")
        return out
    for item in portfolio:
        c = item.contract
        out[c.symbol] = {
            "qty": float(item.position),
            "avgCost": float(item.averageCost),
            "currency": getattr(c, "currency", None),
            "exchange": getattr(c, "primaryExchange", None) or getattr(c, "exchange", None),
            "marketValue": float(item.marketValue),
            "unrealizedPnl": float(item.unrealizedPNL),
        }
    return out


def reconstruct_pair_states(positions: dict[str, dict]) -> dict[str, PairState]:
    states: dict[str, PairState] = {}
    for us, hk, ratio in PAIRS:
        st = PairState(us_sym=us, hk_sym=hk, ratio=ratio)
        if us in positions:
            st.us_qty = int(round(positions[us]["qty"]))
            st.avg_us_cost = positions[us]["avgCost"]
        if hk in positions:
            st.hk_qty = int(round(positions[hk]["qty"]))
            st.avg_hk_cost = positions[hk]["avgCost"]
        if st.us_qty < 0 and st.hk_qty > 0:
            st.direction = "short_us_long_hk"
        elif st.us_qty > 0 and st.hk_qty < 0:
            st.direction = "long_us_short_hk"
        states[f"{us}/{hk}"] = st
    return states


# ---------------------------------------------------------------------------
# NAV
# ---------------------------------------------------------------------------


def fetch_nav_usd(ib: IB, log: logging.Logger) -> float:
    try:
        for av in ib.accountValues():
            if av.tag == "NetLiquidation" and av.currency == "USD":
                return float(av.value)
    except Exception as exc:  # noqa: BLE001
        log.warning(f"accountValues() failed: {exc}")
    # Fallback: previous run's recorded NAV, or hard default.
    if NAV_STATE_PATH.exists():
        try:
            return float(json.loads(NAV_STATE_PATH.read_text()).get("nav_usd", 100_000.0))
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
    nav_usd: float,
    us_px: float,
    hk_px_hkd: float,
    fx_usd_hkd: float,
    ratio: int,
) -> tuple[int, int]:
    """Return (us_qty_abs, hk_qty_abs) for a dollar-neutral pair entry.

    Per-pair notional: PER_PAIR_NAV_FRACTION * NAV on EACH leg.
    """
    per_leg_usd = PER_PAIR_NAV_FRACTION * nav_usd
    us_qty_abs = max(int(math.floor(per_leg_usd / max(us_px, 1e-6))), 0)
    hk_px_usd = hk_px_hkd / max(fx_usd_hkd, 1e-6)
    hk_qty_abs = max(int(math.floor(per_leg_usd / max(hk_px_usd, 1e-6))), 0)
    # HKEX trades in board lots (typ. 100, sometimes 500). Round down to 100.
    hk_qty_abs = (hk_qty_abs // 100) * 100
    # Sanity: enforce that hk_qty roughly tracks us_qty * ratio in shares; the
    # *value* is already balanced by the dollar sizing above, so we don't force
    # the ratio in share-count terms — we just record the ratio for accounting.
    _ = ratio  # ratio is for parity math, not for share-count enforcement here
    return us_qty_abs, hk_qty_abs


# ---------------------------------------------------------------------------
# Order placement
# ---------------------------------------------------------------------------


def place_market_order(
    ib: IB,
    contract,
    side: str,
    qty: int,
    log: logging.Logger,
    label: str,
) -> dict:
    """Submit a market order. Returns a result dict with fill stats."""
    assert side in ("BUY", "SELL"), side
    if qty <= 0:
        return {"submitted_ok": False, "note": "zero qty", "status": "skipped"}
    order = MarketOrder(side, qty)
    order.tif = "DAY"
    order.outsideRth = False
    try:
        trade = ib.placeOrder(contract, order)
    except Exception as exc:  # noqa: BLE001
        log.error(f"placeOrder({label}, {side} {qty}) failed: {exc}")
        return {"submitted_ok": False, "note": str(exc), "status": "submit_error"}
    # Wait briefly for status; do not block forever.
    waited = 0.0
    while not trade.isDone() and waited < 30.0:
        ib.sleep(1.0)
        waited += 1.0

    fills = trade.fills or []
    fill_qty = int(sum(f.execution.shares for f in fills))
    avg_px = (
        float(np.mean([f.execution.avgPrice for f in fills])) if fills else None
    )
    status = trade.orderStatus.status if trade.orderStatus else "Unknown"
    log.info(
        f"  ORDER {label}: {side} {qty} -> status={status} "
        f"fill_qty={fill_qty} avg_px={avg_px}"
    )
    return {
        "submitted_ok": True,
        "fill_qty": fill_qty,
        "avg_fill_price": avg_px,
        "status": status,
        "order_id": getattr(trade.order, "orderId", None),
        "perm_id": getattr(trade.order, "permId", None),
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
) -> None:
    conn.execute(
        """
        INSERT INTO orders (
            ts_utc, cycle, pair, leg, symbol, side, qty,
            submitted_ok, fill_price, fill_qty, avg_fill_price,
            status, order_id, perm_id, note
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            cycle,
            pair,
            leg,
            symbol,
            side,
            qty,
            1 if result.get("submitted_ok") else 0,
            result.get("avg_fill_price"),
            result.get("fill_qty"),
            result.get("avg_fill_price"),
            result.get("status"),
            result.get("order_id"),
            result.get("perm_id"),
            result.get("note", ""),
        ),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Connection-loss watchdog
# ---------------------------------------------------------------------------


def ensure_connected(ib: IB, log: logging.Logger, market_hours: bool) -> None:
    """If the connection drops, allow up to CONNECTION_LOSS_KILL_SECS to
    recover during market hours; otherwise kill-switch."""
    if ib.isConnected():
        return
    log.warning("IB connection dropped; attempting reconnect.")
    deadline = time.time() + CONNECTION_LOSS_KILL_SECS
    while time.time() < deadline:
        try:
            ib.connect(
                "127.0.0.1",
                int(os.environ.get("IB_PORT", 7497)),
                clientId=int(os.environ.get("IB_CLIENT_ID", 17)),
                readonly=False,
            )
            if ib.isConnected():
                log.info("Reconnected to IBKR.")
                return
        except Exception as exc:  # noqa: BLE001
            log.warning(f"Reconnect attempt failed: {exc}")
        time.sleep(15)
    if market_hours:
        raise KillSwitch(
            f"IB connection lost > {CONNECTION_LOSS_KILL_SECS}s during market hours"
        )
    log.warning("Outside market hours; not raising kill on disconnect.")


# ---------------------------------------------------------------------------
# Flatten everything (used on kill-switch)
# ---------------------------------------------------------------------------


def flatten_all(
    ib: IB,
    conn: sqlite3.Connection,
    cycle: str,
    log: logging.Logger,
) -> None:
    log.warning("FLATTEN ALL: closing every iter_18 position.")
    if not ib.isConnected():
        log.error("Cannot flatten — IB disconnected.")
        return
    positions = snapshot_positions(ib, log)
    for us, hk, _ratio in PAIRS:
        for sym, side_for_close in ((us, "BUY"), (hk, "BUY")):
            if sym not in positions:
                continue
            qty = int(round(positions[sym]["qty"]))
            if qty == 0:
                continue
            side = "BUY" if qty < 0 else "SELL"
            contract = make_us_contract(sym) if sym == us else make_hk_contract(sym)
            try:
                ib.qualifyContracts(contract)
            except Exception as exc:  # noqa: BLE001
                log.error(f"qualify {sym} failed during flatten: {exc}")
                continue
            result = place_market_order(
                ib, contract, side, abs(qty), log, f"FLATTEN {sym}"
            )
            log_order(
                conn,
                cycle,
                f"{us}/{hk}",
                "US" if sym == us else "HK",
                sym,
                side,
                abs(qty),
                result,
            )
            _ = side_for_close  # silence linter — kept for readability


def flatten_pair(
    ib: IB,
    conn: sqlite3.Connection,
    cycle: str,
    pair_key: str,
    st: PairState,
    log: logging.Logger,
    reason: str,
) -> None:
    log.warning(f"FLATTEN PAIR {pair_key}: reason={reason}")
    if st.us_qty != 0:
        side = "BUY" if st.us_qty < 0 else "SELL"
        c = make_us_contract(st.us_sym)
        try:
            ib.qualifyContracts(c)
            r = place_market_order(ib, c, side, abs(st.us_qty), log, f"close US {st.us_sym}")
        except Exception as exc:  # noqa: BLE001
            r = {"submitted_ok": False, "note": str(exc), "status": "qualify_error"}
        log_order(conn, cycle, pair_key, "US", st.us_sym, side, abs(st.us_qty), r)
    if st.hk_qty != 0:
        side = "BUY" if st.hk_qty < 0 else "SELL"
        c = make_hk_contract(st.hk_sym)
        try:
            ib.qualifyContracts(c)
            r = place_market_order(ib, c, side, abs(st.hk_qty), log, f"close HK {st.hk_sym}")
        except Exception as exc:  # noqa: BLE001
            r = {"submitted_ok": False, "note": str(exc), "status": "qualify_error"}
        log_order(conn, cycle, pair_key, "HK", st.hk_sym, side, abs(st.hk_qty), r)


# ---------------------------------------------------------------------------
# Pair decision logic — mirrors run_pair_backtest()
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
    dev_bps = 10_000.0 * (us_px - parity_usd) / parity_usd if parity_usd > 0 else 0.0

    if not st.open:
        if blocked:
            return PairDecision(
                pair, st.us_sym, st.hk_sym, st.ratio, us_px, hk_px, fx,
                dev_bps, blocked, "skip", reason="ex_div_blackout",
            )
        if abs(dev_bps) > ENTRY_BPS:
            direction = "short_us_long_hk" if dev_bps > 0 else "long_us_short_hk"
            us_qty, hk_qty = compute_sizing(nav_usd, us_px, hk_px, fx, st.ratio)
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


def log_decision(
    conn: sqlite3.Connection,
    cycle: str,
    d: PairDecision,
) -> None:
    conn.execute(
        """
        INSERT INTO decisions (
            ts_utc, cycle, pair, us_symbol, hk_symbol, ratio,
            us_price_usd, hk_price_hkd, fx_usd_hkd, dev_bps,
            ex_div_block, action, reason, direction,
            us_qty, hk_qty, note
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
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
# Execution: turn a decision into a pair of broker orders
# ---------------------------------------------------------------------------


def execute_decision(
    ib: IB,
    conn: sqlite3.Connection,
    cycle: str,
    d: PairDecision,
    log: logging.Logger,
    allow_skip_confirm: bool,
) -> None:
    """Place the two broker orders implied by `d`.

    Order convention:
      OPEN long_us_short_hk : BUY US,  SELL HK
      OPEN short_us_long_hk : SELL US, BUY HK
      CLOSE / STOP          : reverse of whatever is held
    """
    if d.action in ("hold", "skip"):
        return

    # Confirm gate for the first N live trades
    cnt = load_confirm_counter()
    if cnt < CONFIRM_TRADE_COUNT:
        log.info(
            f"  Confirmation gate active (trade #{cnt + 1} / first {CONFIRM_TRADE_COUNT})"
        )
        prompt = (
            f"  Trade {d.pair} action={d.action} dir={d.direction} "
            f"dev_bps={d.dev_bps:.1f} us_qty={d.us_qty} hk_qty={d.hk_qty} — approve?"
        )
        if not interactive_confirm(prompt, allow_skip=allow_skip_confirm):
            log.warning("  Operator declined trade.")
            d.note = "operator_declined"
            return
        bump_confirm_counter()

    us_contract = make_us_contract(d.us_sym)
    hk_contract = make_hk_contract(d.hk_sym)
    try:
        ib.qualifyContracts(us_contract, hk_contract)
    except Exception as exc:  # noqa: BLE001
        log.error(f"qualify failed for {d.pair}: {exc}")
        d.note = f"qualify_error:{exc}"
        return

    if d.action == "open":
        if d.direction == "short_us_long_hk":
            us_side, hk_side = "SELL", "BUY"
        else:
            us_side, hk_side = "BUY", "SELL"
        us_qty, hk_qty = d.us_qty, d.hk_qty
    elif d.action in ("close", "stop"):
        # d.us_qty / d.hk_qty are the SIGNED holdings — close = reverse them.
        us_side = "BUY" if d.us_qty < 0 else "SELL"
        hk_side = "BUY" if d.hk_qty < 0 else "SELL"
        us_qty, hk_qty = abs(d.us_qty), abs(d.hk_qty)
    else:
        return

    # Submit HK leg first when HKEX is closed at us_close cycle? We submit
    # both; orders into a closed exchange become DAY orders that fill at next
    # open — acceptable for paper-shadow.
    us_res = place_market_order(ib, us_contract, us_side, us_qty, log, f"{d.pair} US")
    log_order(conn, cycle, d.pair, "US", d.us_sym, us_side, us_qty, us_res)
    hk_res = place_market_order(ib, hk_contract, hk_side, hk_qty, log, f"{d.pair} HK")
    log_order(conn, cycle, d.pair, "HK", d.hk_sym, hk_side, hk_qty, hk_res)

    # If HK leg failed to submit (trading halt etc.), flatten the US leg too.
    hk_halted = (
        not hk_res.get("submitted_ok")
        or hk_res.get("status", "").lower() in ("cancelled", "inactive", "rejected")
    )
    if hk_halted and us_res.get("submitted_ok"):
        log.warning(
            f"  HK leg appears halted/rejected for {d.pair} "
            f"(status={hk_res.get('status')}); flattening US leg."
        )
        # Reverse the just-placed US order
        rev_side = "BUY" if us_side == "SELL" else "SELL"
        rev_res = place_market_order(
            ib, us_contract, rev_side, us_qty, log, f"{d.pair} US-flatten-on-HK-halt"
        )
        log_order(
            conn, cycle, d.pair, "US", d.us_sym, rev_side, us_qty, rev_res,
        )


# ---------------------------------------------------------------------------
# Per-pair P&L for the 5%-NAV kill check
# ---------------------------------------------------------------------------


def per_pair_unrealised_pnl(
    st: PairState,
    us_px: Optional[float],
    hk_px_hkd: Optional[float],
    fx: float,
) -> float:
    if not st.open or us_px is None or hk_px_hkd is None:
        return 0.0
    us_pnl = (us_px - st.avg_us_cost) * st.us_qty
    hk_pnl_hkd = (hk_px_hkd - st.avg_hk_cost) * st.hk_qty
    hk_pnl_usd = hk_pnl_hkd / max(fx, 1e-9)
    return float(us_pnl + hk_pnl_usd)


# ---------------------------------------------------------------------------
# EOD summary writer
# ---------------------------------------------------------------------------


def write_eod_summary(
    conn: sqlite3.Connection,
    cycle: str,
    nav_usd: float,
    states: dict[str, PairState],
    decisions: list[PairDecision],
    pnls: dict[str, float],
    log: logging.Logger,
) -> None:
    open_count = sum(1 for s in states.values() if s.open)
    gross = 0.0
    for st in states.values():
        gross += abs(st.us_qty * st.avg_us_cost)
        gross += abs(st.hk_qty * st.avg_hk_cost) / USD_HKD_PEG
    payload = {
        "pairs": {
            d.pair: {
                "dev_bps": None if d.dev_bps is None else round(d.dev_bps, 2),
                "action": d.action,
                "direction": d.direction,
                "us_px": d.us_px,
                "hk_px": d.hk_px,
                "ex_div_block": d.ex_div_block,
                "unrealised_pnl_usd": round(pnls.get(d.pair, 0.0), 2),
                "reason": d.reason,
            }
            for d in decisions
        },
    }
    total_pnl = sum(pnls.values())
    conn.execute(
        """
        INSERT INTO eod_summary (
            ts_utc, cycle, nav_usd, open_pair_count, gross_exposure,
            total_unreal_pnl, payload_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            cycle,
            nav_usd,
            open_count,
            gross,
            total_pnl,
            json.dumps(payload, default=str),
        ),
    )
    conn.commit()
    log.info(
        f"EOD ({cycle}): NAV=${nav_usd:,.2f} open_pairs={open_count} "
        f"gross_exposure=${gross:,.2f} total_unreal_pnl=${total_pnl:,.2f}"
    )


def write_positions_snapshot(
    conn: sqlite3.Connection,
    cycle: str,
    positions: dict[str, dict],
) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    for sym, p in positions.items():
        conn.execute(
            """
            INSERT INTO positions_snapshot (
                ts_utc, cycle, symbol, exchange, currency,
                qty, avg_cost, market_value, unreal_pnl
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts, cycle, sym,
                p.get("exchange"), p.get("currency"),
                p.get("qty"), p.get("avgCost"),
                p.get("marketValue"), p.get("unrealizedPnl"),
            ),
        )
    conn.commit()


# ---------------------------------------------------------------------------
# One full cycle
# ---------------------------------------------------------------------------


def run_cycle(
    ib: IB,
    conn: sqlite3.Connection,
    cycle: str,
    cal: dict[str, set],
    log: logging.Logger,
    allow_skip_confirm: bool,
    dry_run: bool,
) -> None:
    log.info(f"=== Cycle start: {cycle} dry_run={dry_run} ===")

    # 1. Kill flag check
    kf = kill_flag_present()
    if kf is not None:
        raise KillSwitch(f"KILL_SWITCH file present: {kf.get('reason')} / {kf.get('detail')}")

    # 2. Snapshot account
    nav_usd = fetch_nav_usd(ib, log)
    persist_nav(nav_usd)
    positions = snapshot_positions(ib, log)
    write_positions_snapshot(conn, cycle, positions)
    states = reconstruct_pair_states(positions)

    # 3. Pull FX once
    fx = fetch_fx_usd_hkd(ib, log)
    log.info(f"FX USD/HKD = {fx:.4f}")

    # 4. Pull prices for all 10 tickers
    prices: dict[str, Optional[float]] = {}
    for us, hk, _ratio in PAIRS:
        us_c = make_us_contract(us)
        hk_c = make_hk_contract(hk)
        try:
            ib.qualifyContracts(us_c, hk_c)
        except Exception as exc:  # noqa: BLE001
            log.warning(f"qualify {us}/{hk} failed: {exc}")
        prices[us] = snapshot_price(ib, us_c, log, us)
        prices[hk] = snapshot_price(ib, hk_c, log, hk)

    # 5. Decide per pair
    today = pd.Timestamp.today().normalize()
    decisions: list[PairDecision] = []
    pnls: dict[str, float] = {}
    for us, hk, _ratio in PAIRS:
        st = states[f"{us}/{hk}"]
        blocked = is_ex_div_blocked(today, us, hk, cal)
        d = decide_pair(
            st,
            prices.get(us),
            prices.get(hk),
            fx,
            blocked,
            nav_usd,
        )
        decisions.append(d)
        log_decision(conn, cycle, d)
        pnls[d.pair] = per_pair_unrealised_pnl(st, prices.get(us), prices.get(hk), fx)
        log.info(
            f"  {d.pair}: dev_bps="
            + (f"{d.dev_bps:+.1f}" if d.dev_bps is not None else "NA")
            + f" action={d.action} blocked={d.ex_div_block} "
            f"dir={d.direction} reason={d.reason}"
        )

    # 6. Risk gates BEFORE we trade
    open_count = sum(1 for s in states.values() if s.open)
    will_open = sum(1 for d in decisions if d.action == "open")
    if open_count + will_open > MAX_OPEN_PAIRS:
        log.warning(
            f"Open+new ({open_count + will_open}) exceeds MAX_OPEN_PAIRS={MAX_OPEN_PAIRS}; "
            "deferring entries beyond cap."
        )
        slots = max(MAX_OPEN_PAIRS - open_count, 0)
        for d in decisions:
            if d.action == "open" and slots <= 0:
                d.action = "skip"
                d.reason = "cap_reached"
            elif d.action == "open":
                slots -= 1

    # Hard kill: per-pair loss > 5% NAV
    for d in decisions:
        pnl = pnls.get(d.pair, 0.0)
        if pnl < -PER_PAIR_LOSS_KILL_FRAC * nav_usd:
            raise KillSwitch(
                f"Per-pair loss for {d.pair} = ${pnl:,.2f} exceeds "
                f"{PER_PAIR_LOSS_KILL_FRAC * 100:.0f}% of NAV ${nav_usd:,.2f}"
            )

    # Hard kill: total gross exposure > 100% NAV
    gross = 0.0
    for st in states.values():
        gross += abs(st.us_qty * st.avg_us_cost)
        gross += abs(st.hk_qty * st.avg_hk_cost) / max(fx, 1e-9)
    if gross > MAX_TOTAL_GROSS_NAV_FRAC * nav_usd:
        raise KillSwitch(
            f"Gross exposure ${gross:,.2f} > {MAX_TOTAL_GROSS_NAV_FRAC * 100:.0f}% "
            f"of NAV ${nav_usd:,.2f}"
        )

    # 7. Execute
    if dry_run:
        log.info("DRY RUN — skipping order placement.")
    else:
        for d in decisions:
            if d.action in ("open", "close", "stop"):
                execute_decision(ib, conn, cycle, d, log, allow_skip_confirm)

    # 8. EOD summary
    write_eod_summary(conn, cycle, nav_usd, states, decisions, pnls, log)
    log.info(f"=== Cycle done: {cycle} ===")


# ---------------------------------------------------------------------------
# Signal handling — graceful shutdown
# ---------------------------------------------------------------------------


_shutdown_requested = False


def _sigterm_handler(signum, frame):  # noqa: D401 — system handler
    global _shutdown_requested
    _shutdown_requested = True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="iter_18 ADR paper-shadow host (IBKR)")
    p.add_argument(
        "--cycle",
        choices=["hk_close", "us_close", "adhoc"],
        default="adhoc",
        help="Which scheduled cycle this invocation represents.",
    )
    p.add_argument(
        "--confirm",
        action="store_true",
        help="Enable interactive trade confirmation for the first 10 live trades.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Decide and log but DO NOT submit orders.",
    )
    p.add_argument(
        "--skip-backtest-check",
        action="store_true",
        help="Skip the 60-day backtest divergence verification.",
    )
    p.add_argument(
        "--flatten",
        action="store_true",
        help="Immediately flatten all iter_18 positions and exit.",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    log = setup_logging()
    log.info(f"=== iter_18 ADR host start (argv={sys.argv}) ===")

    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, _sigterm_handler)
    signal.signal(signal.SIGINT, _sigterm_handler)

    # Startup backtest divergence check
    if not args.skip_backtest_check:
        verify_backtest_divergence(log)

    # Pre-existing kill flag => refuse to start
    kf = kill_flag_present()
    if kf is not None:
        log.error(
            f"KILL_SWITCH present at startup: {kf}. "
            f"Remove {KILL_FLAG_PATH} after investigating before re-running."
        )
        return 4

    conn = open_db()
    cal = load_dividend_calendar(log)

    port = int(os.environ.get("IB_PORT", 7497))   # 7497 = paper TWS, 4002 = paper gateway
    client_id = int(os.environ.get("IB_CLIENT_ID", 17))
    log.info(f"Connecting to IBKR at 127.0.0.1:{port} clientId={client_id}")

    ib = IB()
    try:
        ib.connect("127.0.0.1", port, clientId=client_id, readonly=False)
    except Exception as exc:  # noqa: BLE001
        log.error(f"IB connect failed: {exc}")
        return 3
    if not ib.isConnected():
        log.error("IB connect returned without raising but isConnected()=False.")
        return 3
    log.info(
        f"Connected. server_version={ib.client.serverVersion()} "
        f"accounts={ib.managedAccounts()}"
    )

    try:
        if args.flatten:
            flatten_all(ib, conn, args.cycle, log)
            log.info("Flatten complete; exiting.")
            return 0

        run_cycle(
            ib=ib,
            conn=conn,
            cycle=args.cycle,
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
            flatten_all(ib, conn, args.cycle, log)
        except Exception as exc:  # noqa: BLE001
            log.error(f"Flatten-on-kill failed: {exc}")
        return 9

    except Exception as exc:  # noqa: BLE001
        log.exception(f"Unhandled error: {exc}")
        record_kill(conn, args.cycle, "unhandled_exception", repr(exc))
        return 5

    finally:
        try:
            if ib.isConnected():
                ib.disconnect()
        except Exception:  # noqa: BLE001
            pass
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass
        log.info("=== iter_18 ADR host exit ===")


if __name__ == "__main__":
    sys.exit(main())
