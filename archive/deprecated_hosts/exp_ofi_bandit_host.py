"""Order Flow Imbalance (OFI) with Thompson Sampling — SPY Paper Host.

Experimental paper-shadow host for the OFI signal on SPY, using a 3-arm
Thompson-sampling bandit to learn the |OFI| threshold online.

Strategy mechanics
------------------
Signal
  - Poll Alpaca's `/v2/stocks/SPY/trades/latest` every 5 seconds and append the
    latest print to a 60-second rolling window of trades.
  - Each trade is classified buy/sell via Lee-Ready: trade price vs. the mid
    of the latest NBBO quote. Trades above mid count as buys, below mid as
    sells; ties (==mid) fall back to a tick-test (vs. last classified price).
  - OFI = (sum_buy_volume - sum_sell_volume) / total_volume in window
  - OFI is in [-1, +1].

Bandit
  - Three threshold arms: 0.20, 0.40, 0.60.
  - Each arm tracks a Beta(alpha=wins+1, beta=losses+1) posterior.
  - Every 5s we sample one value from each arm's Beta and pick argmax as the
    "current_threshold" used for that tick's go/no-go decision.

Trade rule
  - If |OFI| > current_threshold AND no open position AND risk-clear:
      direction = "buy"  if OFI > 0 else "sell"
      submit marketable-limit IOC for 10 SPY shares
      hold 30 seconds
      flatten with opposite-side marketable-limit IOC
      outcome = win if net P&L (after a flat round-trip cost budget) > 1 bp
                of entry notional, else loss
      update Beta posterior of the firing arm

Risk envelope (hard caps)
  - Max 1 concurrent position
  - Max 50 trades/day
  - Fixed size 10 shares (~$5k notional at SPY ~ $500)
  - Daily loss cap: -$200 -> halt for day
  - 5 consecutive losses -> halt for day
  - Kill file: research/live/STOP_OFI_BANDIT
  - --dry-run: compute and log signal/decisions, NEVER submit orders

All decisions, ticks, orders, fills, and bandit-state snapshots are mirrored
to research/live/exp_ofi_bandit_decisions.sqlite3.

Author: Deployment Engineer (turbo-carnival research/live)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal as signal_mod
import sqlite3
import sys
import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Optional

import numpy as np
import requests


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

ROOT = Path("/Users/weeoto/turbo-carnival")
LIVE_DIR = ROOT / "research" / "live"
ENV_PATH = ROOT / "council" / ".env.local"

DECISIONS_DB = LIVE_DIR / "exp_ofi_bandit_decisions.sqlite3"
KILL_FLAG_PATH = LIVE_DIR / "STOP_OFI_BANDIT"
LOG_PATH = LIVE_DIR / "exp_ofi_bandit_host.log"
STATE_PATH = LIVE_DIR / "exp_ofi_bandit_state.json"

ALPACA_TRADING = "https://paper-api.alpaca.markets/v2"
ALPACA_DATA = "https://data.alpaca.markets/v2"

SYMBOL = "SPY"

# Strategy / bandit knobs (frozen for this experiment)
POLL_SECS = 5
WINDOW_SECS = 60
HOLD_SECS = 30
ARMS = (0.20, 0.40, 0.60)
ORDER_SIZE = 10  # shares
ORDER_PRICE_SLIP_BPS = 5.0  # marketable-limit IOC slippage budget
WIN_THRESHOLD_BPS = 1.0     # net > 1 bp of entry notional => win

# Risk envelope
MAX_CONCURRENT = 1
MAX_TRADES_PER_DAY = 50
DAILY_LOSS_CAP_USD = 200.0
MAX_CONSEC_LOSSES = 5

# Cost assumption (flat round-trip budget used purely for win/loss labelling).
# We assume the marketable-IOC pair eats at most 2 * SLIP bps in friction;
# anything earned above that is a win.
ROUND_TRIP_COST_BPS = 2.0 * ORDER_PRICE_SLIP_BPS  # = 10 bps

# Tolerance for the loop scheduler (don't burn CPU)
SLEEP_FLOOR = 0.05


# ---------------------------------------------------------------------------
# Env loader & credentials
# ---------------------------------------------------------------------------

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


def resolve_credentials() -> tuple[str, str]:
    """Return (key, secret) for the VALK account (~$99,615)."""
    key = os.environ.get("ALPACA_VALK_KEY")
    secret = os.environ.get("ALPACA_VALK_SECRET")
    if not key or not secret:
        raise SystemExit(
            "FATAL: Missing ALPACA_VALK_KEY / ALPACA_VALK_SECRET in "
            f"{ENV_PATH}"
        )
    return key, secret


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("exp_ofi_bandit_host")
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

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ticks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc        TEXT    NOT NULL,
    last_price    REAL,
    bid           REAL,
    ask           REAL,
    mid           REAL,
    window_n      INTEGER,
    buy_vol       REAL,
    sell_vol      REAL,
    total_vol     REAL,
    ofi           REAL,
    abs_ofi       REAL,
    sample_arm0   REAL,
    sample_arm1   REAL,
    sample_arm2   REAL,
    chosen_arm    INTEGER,
    threshold     REAL,
    decision      TEXT,
    note          TEXT
);

CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_open_utc     TEXT    NOT NULL,
    ts_close_utc    TEXT,
    arm_idx         INTEGER NOT NULL,
    threshold       REAL    NOT NULL,
    ofi_at_entry    REAL    NOT NULL,
    direction       TEXT    NOT NULL,        -- 'long' | 'short'
    qty             INTEGER NOT NULL,
    entry_price     REAL,
    exit_price      REAL,
    pnl_usd         REAL,
    pnl_bps         REAL,
    outcome         TEXT,                    -- 'win' | 'loss' | 'flat'
    entry_order_id  TEXT,
    exit_order_id   TEXT,
    is_dry_run      INTEGER NOT NULL,
    note            TEXT
);

CREATE TABLE IF NOT EXISTS bandit_state (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc       TEXT NOT NULL,
    arm_idx      INTEGER NOT NULL,
    threshold    REAL NOT NULL,
    wins         INTEGER NOT NULL,
    losses       INTEGER NOT NULL,
    trades       INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS kills (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc      TEXT NOT NULL,
    reason      TEXT NOT NULL,
    detail      TEXT
);
"""


def open_db() -> sqlite3.Connection:
    DECISIONS_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DECISIONS_DB))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# ---------------------------------------------------------------------------
# Alpaca REST helpers
# ---------------------------------------------------------------------------

def _headers(key: str, secret: str) -> dict:
    return {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
        "Content-Type": "application/json",
    }


def alpaca_get_account(key: str, secret: str) -> Optional[dict]:
    try:
        r = requests.get(
            f"{ALPACA_TRADING}/account", headers=_headers(key, secret), timeout=10
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None


def alpaca_get_latest_trade(key: str, secret: str, symbol: str) -> Optional[dict]:
    """Return the latest trade dict {p, s, t, ...} or None."""
    try:
        r = requests.get(
            f"{ALPACA_DATA}/stocks/{symbol}/trades/latest",
            headers=_headers(key, secret),
            timeout=10,
        )
        if r.status_code == 200:
            j = r.json()
            return j.get("trade")
    except Exception:
        return None
    return None


def alpaca_get_latest_quote(key: str, secret: str, symbol: str) -> Optional[dict]:
    """Return latest NBBO quote {bp, ap, bs, as, t} or None."""
    try:
        r = requests.get(
            f"{ALPACA_DATA}/stocks/{symbol}/quotes/latest",
            headers=_headers(key, secret),
            timeout=10,
        )
        if r.status_code == 200:
            j = r.json()
            return j.get("quote")
    except Exception:
        return None
    return None


def alpaca_get_position(
    key: str, secret: str, symbol: str
) -> Optional[dict]:
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
    except Exception:
        return None
    return None


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
        return None
    return None


def alpaca_submit_marketable_limit_ioc(
    key: str,
    secret: str,
    symbol: str,
    qty: int,
    side: str,
    ref_price: float,
    client_tag: str,
    logger: logging.Logger,
) -> Optional[dict]:
    """Submit a marketable-limit IOC at ref_price +/- ORDER_PRICE_SLIP_BPS."""
    slip = ref_price * (ORDER_PRICE_SLIP_BPS / 10_000.0)
    if side == "buy":
        limit_price = round(ref_price + slip, 2)
    else:
        limit_price = round(ref_price - slip, 2)
    payload = {
        "symbol": symbol,
        "qty": str(int(qty)),
        "side": side,
        "type": "limit",
        "limit_price": str(limit_price),
        "time_in_force": "ioc",
        "client_order_id": f"ofi-bandit-{client_tag}-{int(time.time() * 1000)}",
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


def wait_for_order_terminal(
    key: str,
    secret: str,
    order_id: str,
    timeout_secs: float = 5.0,
) -> Optional[dict]:
    """Poll order until status is terminal or timeout."""
    deadline = time.time() + timeout_secs
    terminal = {"filled", "canceled", "expired", "rejected", "done_for_day"}
    last = None
    while time.time() < deadline:
        last = alpaca_get_order(key, secret, order_id)
        if last is None:
            time.sleep(0.2)
            continue
        if last.get("status") in terminal:
            return last
        time.sleep(0.25)
    return last


# ---------------------------------------------------------------------------
# Lee-Ready trade classifier + rolling window + OFI
# ---------------------------------------------------------------------------

@dataclass
class TradeEvt:
    ts_unix: float
    price: float
    size: float
    side: str  # 'buy' | 'sell' | 'unk'


class RollingTrades:
    """60-second rolling window of trades with Lee-Ready classification.

    Lee-Ready (simplified for last-trade polling, not full tape):
      - quote rule:  price > mid  -> buy
                     price < mid  -> sell
                     price == mid -> tick test (vs. last classified price)
      - tick test:   price > last_diff_price -> buy
                     price < last_diff_price -> sell
                     else: unknown (do not include in OFI)
    """

    def __init__(self, window_secs: int = WINDOW_SECS) -> None:
        self.window_secs = window_secs
        self.buf: Deque[TradeEvt] = deque()
        self._last_diff_price: Optional[float] = None
        self._last_seen_trade_ts: Optional[float] = None

    def _evict(self, now_unix: float) -> None:
        cutoff = now_unix - self.window_secs
        while self.buf and self.buf[0].ts_unix < cutoff:
            self.buf.popleft()

    def add(
        self,
        ts_unix: float,
        price: float,
        size: float,
        mid: Optional[float],
    ) -> TradeEvt:
        """Classify trade and append. De-dupes by ts_unix (latest-trade
        endpoint returns the same print across polls until a new one occurs).
        """
        if self._last_seen_trade_ts is not None and ts_unix <= self._last_seen_trade_ts:
            # Same print as before; do not re-add.
            evt = TradeEvt(ts_unix=ts_unix, price=price, size=size, side="dup")
            return evt
        self._last_seen_trade_ts = ts_unix

        side = "unk"
        if mid is not None and mid > 0:
            if price > mid:
                side = "buy"
            elif price < mid:
                side = "sell"
            else:
                # tick test
                if self._last_diff_price is not None:
                    if price > self._last_diff_price:
                        side = "buy"
                    elif price < self._last_diff_price:
                        side = "sell"
        else:
            # no quote -> pure tick test
            if self._last_diff_price is not None:
                if price > self._last_diff_price:
                    side = "buy"
                elif price < self._last_diff_price:
                    side = "sell"

        if (
            self._last_diff_price is None
            or not math.isclose(price, self._last_diff_price)
        ):
            self._last_diff_price = price

        evt = TradeEvt(ts_unix=ts_unix, price=price, size=size, side=side)
        self.buf.append(evt)
        self._evict(ts_unix)
        return evt

    def compute_ofi(self, now_unix: float) -> tuple[float, float, float, int]:
        """Return (buy_vol, sell_vol, total_vol, n_in_window)."""
        self._evict(now_unix)
        buy_v = 0.0
        sell_v = 0.0
        total = 0.0
        for t in self.buf:
            if t.side == "buy":
                buy_v += t.size
                total += t.size
            elif t.side == "sell":
                sell_v += t.size
                total += t.size
        return buy_v, sell_v, total, len(self.buf)


def parse_trade_ts(t: dict) -> Optional[float]:
    """Return unix-seconds from Alpaca trade dict's 't' (RFC3339)."""
    s = t.get("t")
    if not s:
        return None
    try:
        # Alpaca returns RFC3339 with nanosecond precision sometimes.
        # Trim to microseconds.
        if "." in s:
            head, dot, tail = s.partition(".")
            # tail like "123456789Z"
            digits = ""
            for c in tail:
                if c.isdigit():
                    digits += c
                else:
                    break
            digits = digits[:6].ljust(6, "0")
            s = f"{head}.{digits}+00:00"
        else:
            s = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        return dt.timestamp()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Thompson sampling bandit
# ---------------------------------------------------------------------------

@dataclass
class Arm:
    idx: int
    threshold: float
    wins: int = 0
    losses: int = 0

    @property
    def trades(self) -> int:
        return self.wins + self.losses

    def beta_params(self) -> tuple[float, float]:
        # +1 priors to keep Beta defined and behave as flat prior.
        return float(self.wins) + 1.0, float(self.losses) + 1.0


class ThompsonBandit:
    def __init__(self, thresholds: tuple[float, ...] = ARMS) -> None:
        self.arms = [Arm(idx=i, threshold=t) for i, t in enumerate(thresholds)]
        self.rng = np.random.default_rng()

    def sample(self) -> tuple[int, list[float]]:
        samples = [
            float(self.rng.beta(a.beta_params()[0], a.beta_params()[1]))
            for a in self.arms
        ]
        chosen = int(np.argmax(samples))
        return chosen, samples

    def update(self, arm_idx: int, win: bool) -> None:
        a = self.arms[arm_idx]
        if win:
            a.wins += 1
        else:
            a.losses += 1

    def serialize(self) -> list[dict]:
        return [
            {
                "idx": a.idx,
                "threshold": a.threshold,
                "wins": a.wins,
                "losses": a.losses,
                "trades": a.trades,
            }
            for a in self.arms
        ]

    def load(self, payload: list[dict]) -> None:
        by_idx = {int(d["idx"]): d for d in payload}
        for a in self.arms:
            d = by_idx.get(a.idx)
            if not d:
                continue
            a.wins = int(d.get("wins", 0))
            a.losses = int(d.get("losses", 0))


# ---------------------------------------------------------------------------
# Persistent risk state
# ---------------------------------------------------------------------------

@dataclass
class DailyState:
    date_utc: str
    trades_today: int = 0
    pnl_usd: float = 0.0
    consec_losses: int = 0
    halted: bool = False
    halt_reason: str = ""

    @classmethod
    def fresh(cls) -> "DailyState":
        return cls(date_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d"))

    def maybe_rollover(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self.date_utc:
            self.date_utc = today
            self.trades_today = 0
            self.pnl_usd = 0.0
            self.consec_losses = 0
            self.halted = False
            self.halt_reason = ""


def load_state(bandit: ThompsonBandit) -> DailyState:
    if not STATE_PATH.exists():
        return DailyState.fresh()
    try:
        payload = json.loads(STATE_PATH.read_text())
        if "bandit" in payload:
            bandit.load(payload["bandit"])
        ds = payload.get("daily") or {}
        st = DailyState(
            date_utc=ds.get(
                "date_utc", datetime.now(timezone.utc).strftime("%Y-%m-%d")
            ),
            trades_today=int(ds.get("trades_today", 0)),
            pnl_usd=float(ds.get("pnl_usd", 0.0)),
            consec_losses=int(ds.get("consec_losses", 0)),
            halted=bool(ds.get("halted", False)),
            halt_reason=str(ds.get("halt_reason", "")),
        )
        st.maybe_rollover()
        return st
    except Exception:
        return DailyState.fresh()


def save_state(bandit: ThompsonBandit, state: DailyState) -> None:
    try:
        payload = {
            "bandit": bandit.serialize(),
            "daily": asdict(state),
            "saved_at_utc": utc_now_iso(),
        }
        STATE_PATH.write_text(json.dumps(payload, indent=2))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Kill flag
# ---------------------------------------------------------------------------

def kill_flag_present() -> Optional[str]:
    if not KILL_FLAG_PATH.exists():
        return None
    try:
        body = KILL_FLAG_PATH.read_text().strip()
    except Exception:
        body = ""
    return body or "kill file present"


def write_kill_flag(reason: str) -> None:
    try:
        KILL_FLAG_PATH.write_text(reason)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Trade execution + outcome scoring
# ---------------------------------------------------------------------------

@dataclass
class OpenPosition:
    arm_idx: int
    threshold: float
    ofi_at_entry: float
    direction: str          # 'long' | 'short'
    qty: int
    entry_price: float
    entry_ts_unix: float
    entry_order_id: str
    ts_open_utc: str


def submit_entry(
    key: str,
    secret: str,
    arm_idx: int,
    threshold: float,
    ofi_at_entry: float,
    ref_price: float,
    logger: logging.Logger,
    dry_run: bool,
) -> Optional[OpenPosition]:
    direction = "long" if ofi_at_entry > 0 else "short"
    side = "buy" if direction == "long" else "sell"
    ts_open_utc = utc_now_iso()
    if dry_run:
        logger.info(
            f"[DRY-RUN] would SUBMIT entry {side} {ORDER_SIZE} {SYMBOL} @ ~{ref_price:.2f} "
            f"(arm={arm_idx} thr={threshold:.2f} OFI={ofi_at_entry:+.3f})"
        )
        return OpenPosition(
            arm_idx=arm_idx,
            threshold=threshold,
            ofi_at_entry=ofi_at_entry,
            direction=direction,
            qty=ORDER_SIZE,
            entry_price=ref_price,
            entry_ts_unix=time.time(),
            entry_order_id="DRY",
            ts_open_utc=ts_open_utc,
        )

    res = alpaca_submit_marketable_limit_ioc(
        key=key,
        secret=secret,
        symbol=SYMBOL,
        qty=ORDER_SIZE,
        side=side,
        ref_price=ref_price,
        client_tag=f"entry-{arm_idx}",
        logger=logger,
    )
    if not res:
        return None
    order_id = res.get("id", "")
    final = wait_for_order_terminal(key, secret, order_id) or res
    status = final.get("status", "")
    filled_qty = float(final.get("filled_qty") or 0)
    avg_price = final.get("filled_avg_price")
    if status != "filled" or filled_qty <= 0 or not avg_price:
        logger.warning(
            f"entry {order_id} status={status} filled_qty={filled_qty} — skipping"
        )
        return None
    return OpenPosition(
        arm_idx=arm_idx,
        threshold=threshold,
        ofi_at_entry=ofi_at_entry,
        direction=direction,
        qty=int(filled_qty),
        entry_price=float(avg_price),
        entry_ts_unix=time.time(),
        entry_order_id=order_id,
        ts_open_utc=ts_open_utc,
    )


def submit_exit(
    key: str,
    secret: str,
    pos: OpenPosition,
    ref_price: float,
    logger: logging.Logger,
    dry_run: bool,
) -> tuple[Optional[str], Optional[float]]:
    """Return (exit_order_id, exit_avg_price)."""
    side = "sell" if pos.direction == "long" else "buy"
    if dry_run:
        logger.info(
            f"[DRY-RUN] would SUBMIT exit {side} {pos.qty} {SYMBOL} @ ~{ref_price:.2f}"
        )
        return "DRY", ref_price

    res = alpaca_submit_marketable_limit_ioc(
        key=key,
        secret=secret,
        symbol=SYMBOL,
        qty=pos.qty,
        side=side,
        ref_price=ref_price,
        client_tag=f"exit-{pos.arm_idx}",
        logger=logger,
    )
    if not res:
        return None, None
    order_id = res.get("id", "")
    final = wait_for_order_terminal(key, secret, order_id) or res
    status = final.get("status", "")
    filled_qty = float(final.get("filled_qty") or 0)
    avg_price = final.get("filled_avg_price")
    if status != "filled" or filled_qty <= 0 or not avg_price:
        # Best-effort: try a second exit at a wider slip (assume worst-case fill at ref).
        logger.warning(
            f"exit {order_id} status={status} filled_qty={filled_qty} — using ref_price fallback"
        )
        return order_id, ref_price
    return order_id, float(avg_price)


def score_outcome(
    pos: OpenPosition,
    exit_price: float,
) -> tuple[float, float, str]:
    """Return (pnl_usd, pnl_bps, outcome)."""
    if pos.direction == "long":
        pnl_per_share = exit_price - pos.entry_price
    else:
        pnl_per_share = pos.entry_price - exit_price
    pnl_usd = pnl_per_share * pos.qty
    entry_notional = pos.entry_price * pos.qty
    if entry_notional <= 0:
        return 0.0, 0.0, "flat"
    pnl_bps = (pnl_per_share / pos.entry_price) * 10_000.0
    # Net of round-trip cost budget for label
    net_bps = pnl_bps - ROUND_TRIP_COST_BPS
    if net_bps > WIN_THRESHOLD_BPS:
        outcome = "win"
    else:
        outcome = "loss"
    return pnl_usd, pnl_bps, outcome


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def log_tick(
    conn: sqlite3.Connection,
    *,
    last_price: Optional[float],
    bid: Optional[float],
    ask: Optional[float],
    mid: Optional[float],
    window_n: int,
    buy_vol: float,
    sell_vol: float,
    total_vol: float,
    ofi: float,
    samples: list[float],
    chosen_arm: int,
    threshold: float,
    decision: str,
    note: str,
) -> None:
    conn.execute(
        """INSERT INTO ticks
            (ts_utc, last_price, bid, ask, mid, window_n,
             buy_vol, sell_vol, total_vol, ofi, abs_ofi,
             sample_arm0, sample_arm1, sample_arm2,
             chosen_arm, threshold, decision, note)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            utc_now_iso(),
            last_price,
            bid,
            ask,
            mid,
            window_n,
            buy_vol,
            sell_vol,
            total_vol,
            ofi,
            abs(ofi),
            samples[0] if len(samples) > 0 else None,
            samples[1] if len(samples) > 1 else None,
            samples[2] if len(samples) > 2 else None,
            chosen_arm,
            threshold,
            decision,
            note,
        ),
    )
    conn.commit()


def log_trade(
    conn: sqlite3.Connection,
    pos: OpenPosition,
    exit_order_id: Optional[str],
    exit_price: Optional[float],
    pnl_usd: float,
    pnl_bps: float,
    outcome: str,
    ts_close_utc: str,
    is_dry_run: bool,
    note: str,
) -> None:
    conn.execute(
        """INSERT INTO trades
            (ts_open_utc, ts_close_utc, arm_idx, threshold, ofi_at_entry,
             direction, qty, entry_price, exit_price, pnl_usd, pnl_bps,
             outcome, entry_order_id, exit_order_id, is_dry_run, note)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            pos.ts_open_utc,
            ts_close_utc,
            pos.arm_idx,
            pos.threshold,
            pos.ofi_at_entry,
            pos.direction,
            pos.qty,
            pos.entry_price,
            exit_price,
            pnl_usd,
            pnl_bps,
            outcome,
            pos.entry_order_id,
            exit_order_id or "",
            1 if is_dry_run else 0,
            note,
        ),
    )
    conn.commit()


def log_bandit_snapshot(
    conn: sqlite3.Connection, bandit: ThompsonBandit
) -> None:
    now = utc_now_iso()
    rows = [
        (now, a.idx, a.threshold, a.wins, a.losses, a.trades) for a in bandit.arms
    ]
    conn.executemany(
        """INSERT INTO bandit_state
            (ts_utc, arm_idx, threshold, wins, losses, trades)
           VALUES (?,?,?,?,?,?)""",
        rows,
    )
    conn.commit()


def log_kill(conn: sqlite3.Connection, reason: str, detail: str) -> None:
    conn.execute(
        "INSERT INTO kills (ts_utc, reason, detail) VALUES (?,?,?)",
        (utc_now_iso(), reason, detail),
    )
    conn.commit()


def hold_and_exit(
    *,
    key: str,
    secret: str,
    pos: OpenPosition,
    logger: logging.Logger,
    dry_run: bool,
) -> tuple[Optional[str], float]:
    """Hold for HOLD_SECS, then exit at marketable-IOC. Returns
    (exit_order_id, exit_price)."""
    target_close = pos.entry_ts_unix + HOLD_SECS
    while True:
        now = time.time()
        if now >= target_close:
            break
        # Allow kill-file fast exit during the hold.
        if kill_flag_present() is not None:
            logger.warning("kill flag detected during hold — exiting early")
            break
        time.sleep(min(0.5, max(SLEEP_FLOOR, target_close - now)))
    # Fetch a fresh reference price for the exit.
    ref_price = pos.entry_price
    last_trade = alpaca_get_latest_trade(key, secret, SYMBOL)
    if last_trade and last_trade.get("p"):
        try:
            ref_price = float(last_trade["p"])
        except Exception:
            pass
    eid, eprice = submit_exit(
        key=key,
        secret=secret,
        pos=pos,
        ref_price=ref_price,
        logger=logger,
        dry_run=dry_run,
    )
    return eid, eprice if eprice is not None else ref_price


def run_host(args: argparse.Namespace) -> int:
    log = setup_logging()
    key, secret = resolve_credentials()
    conn = open_db()
    bandit = ThompsonBandit()
    state = load_state(bandit)

    log.info("=" * 78)
    log.info("OFI Thompson-Sampling Bandit Host — SPY")
    log.info(
        f"dry_run={args.dry_run}  duration_secs={args.duration or 'oo'}  "
        f"arms={ARMS}  size={ORDER_SIZE}sh  hold={HOLD_SECS}s  poll={POLL_SECS}s  "
        f"window={WINDOW_SECS}s"
    )
    log.info(
        f"max_trades/day={MAX_TRADES_PER_DAY}  daily_loss_cap=${DAILY_LOSS_CAP_USD:.0f}  "
        f"max_consec_losses={MAX_CONSEC_LOSSES}"
    )
    log.info(f"db={DECISIONS_DB}")
    log.info(f"kill_file={KILL_FLAG_PATH}")

    # Account sanity check (skip if endpoint unreachable in dry-run).
    acct = alpaca_get_account(key, secret)
    if acct:
        log.info(
            f"account_id={acct.get('id')} equity=${float(acct.get('equity', 0)):.2f} "
            f"status={acct.get('status')}"
        )
    else:
        log.warning("could not fetch /account (continuing; orders may still work)")

    # Kill-flag at startup -> refuse to run.
    if kill_flag_present() is not None:
        log.error(f"refusing to start: kill flag present at {KILL_FLAG_PATH}")
        return 3

    win = RollingTrades(window_secs=WINDOW_SECS)
    open_pos: Optional[OpenPosition] = None
    start_unix = time.time()
    last_snapshot_unix = 0.0
    SNAPSHOT_EVERY_SECS = 30.0

    def _shutdown(*_a: Any) -> None:
        log.info("received signal; persisting state and exiting")
        save_state(bandit, state)
        raise SystemExit(0)

    try:
        signal_mod.signal(signal_mod.SIGINT, _shutdown)
        signal_mod.signal(signal_mod.SIGTERM, _shutdown)
    except Exception:
        pass

    try:
        while True:
            tick_start = time.time()
            state.maybe_rollover()

            # Termination checks.
            if args.duration and (tick_start - start_unix) >= args.duration:
                log.info(f"reached --duration={args.duration}s, exiting")
                break
            kf = kill_flag_present()
            if kf is not None:
                log.error(f"kill flag present: {kf}")
                log_kill(conn, "kill_file", kf)
                break

            # ---- Data ingest ----
            trade = alpaca_get_latest_trade(key, secret, SYMBOL)
            quote = alpaca_get_latest_quote(key, secret, SYMBOL)
            bid = ask = mid = None
            if quote:
                try:
                    bp = float(quote.get("bp") or 0)
                    ap = float(quote.get("ap") or 0)
                    if bp > 0 and ap > 0 and ap >= bp:
                        bid, ask = bp, ap
                        mid = (bp + ap) / 2.0
                except Exception:
                    pass

            last_price = None
            if trade and trade.get("p"):
                try:
                    p = float(trade["p"])
                    s = float(trade.get("s") or 0)
                    ts_unix = parse_trade_ts(trade) or time.time()
                    win.add(ts_unix, p, s, mid)
                    last_price = p
                except Exception as exc:
                    log.warning(f"could not parse trade: {exc}")

            buy_v, sell_v, total_v, n = win.compute_ofi(time.time())
            ofi = 0.0 if total_v <= 0 else (buy_v - sell_v) / total_v

            # ---- Bandit sample ----
            chosen, samples = bandit.sample()
            threshold = bandit.arms[chosen].threshold

            decision = "no_signal"
            note = ""

            # ---- Risk gates ----
            if state.halted:
                decision = "halted"
                note = state.halt_reason
            elif state.trades_today >= MAX_TRADES_PER_DAY:
                decision = "halted"
                note = "max_trades_today"
                state.halted = True
                state.halt_reason = note
            elif state.pnl_usd <= -DAILY_LOSS_CAP_USD:
                decision = "halted"
                note = f"daily_loss_cap pnl={state.pnl_usd:.2f}"
                state.halted = True
                state.halt_reason = note
            elif state.consec_losses >= MAX_CONSEC_LOSSES:
                decision = "halted"
                note = f"consec_losses={state.consec_losses}"
                state.halted = True
                state.halt_reason = note
            elif open_pos is not None:
                decision = "in_position"
            elif n < 2 or total_v <= 0:
                decision = "warmup"
                note = f"n={n} total_vol={total_v}"
            elif last_price is None:
                decision = "no_data"
            elif abs(ofi) <= threshold:
                decision = "below_threshold"
                note = f"|OFI|={abs(ofi):.3f} <= thr={threshold:.2f}"
            else:
                decision = "fire"

            log_tick(
                conn,
                last_price=last_price,
                bid=bid,
                ask=ask,
                mid=mid,
                window_n=n,
                buy_vol=buy_v,
                sell_vol=sell_v,
                total_vol=total_v,
                ofi=ofi,
                samples=samples,
                chosen_arm=chosen,
                threshold=threshold,
                decision=decision,
                note=note,
            )

            log.info(
                f"tick ofi={ofi:+.3f} n={n} vol={total_v:.0f} "
                f"thr={threshold:.2f}(arm={chosen}) "
                f"samples=[{samples[0]:.2f},{samples[1]:.2f},{samples[2]:.2f}] "
                f"decision={decision}"
            )

            # ---- Trade execution ----
            if decision == "fire" and last_price is not None:
                pos = submit_entry(
                    key=key,
                    secret=secret,
                    arm_idx=chosen,
                    threshold=threshold,
                    ofi_at_entry=ofi,
                    ref_price=last_price,
                    logger=log,
                    dry_run=args.dry_run,
                )
                if pos is None:
                    log.warning("entry submission failed/rejected")
                else:
                    open_pos = pos
                    eid, eprice = hold_and_exit(
                        key=key,
                        secret=secret,
                        pos=pos,
                        logger=log,
                        dry_run=args.dry_run,
                    )
                    pnl_usd, pnl_bps, outcome = score_outcome(pos, eprice)
                    bandit.update(pos.arm_idx, win=(outcome == "win"))
                    state.trades_today += 1
                    state.pnl_usd += pnl_usd
                    if outcome == "win":
                        state.consec_losses = 0
                    else:
                        state.consec_losses += 1
                    ts_close = utc_now_iso()
                    log_trade(
                        conn,
                        pos=pos,
                        exit_order_id=eid,
                        exit_price=eprice,
                        pnl_usd=pnl_usd,
                        pnl_bps=pnl_bps,
                        outcome=outcome,
                        ts_close_utc=ts_close,
                        is_dry_run=args.dry_run,
                        note=f"thr={pos.threshold:.2f} ofi_entry={pos.ofi_at_entry:+.3f}",
                    )
                    log.info(
                        f"TRADE {pos.direction} arm={pos.arm_idx} "
                        f"entry={pos.entry_price:.2f} exit={eprice:.2f} "
                        f"pnl=${pnl_usd:+.2f} ({pnl_bps:+.1f}bps) -> {outcome.upper()}  "
                        f"day_pnl=${state.pnl_usd:+.2f} consec_L={state.consec_losses}"
                    )
                    open_pos = None
                    save_state(bandit, state)
                    log_bandit_snapshot(conn, bandit)

            # ---- Periodic snapshots ----
            if (time.time() - last_snapshot_unix) >= SNAPSHOT_EVERY_SECS:
                log_bandit_snapshot(conn, bandit)
                save_state(bandit, state)
                last_snapshot_unix = time.time()

            # ---- Sleep until next poll ----
            elapsed = time.time() - tick_start
            sleep_for = max(SLEEP_FLOOR, POLL_SECS - elapsed)
            time.sleep(sleep_for)
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt, exiting cleanly")
    except SystemExit:
        raise
    except Exception as exc:
        log.error(f"unhandled error: {exc}\n{traceback.format_exc()}")
        log_kill(conn, "exception", str(exc))
        save_state(bandit, state)
        return 2

    save_state(bandit, state)
    log_bandit_snapshot(conn, bandit)
    log.info("clean exit")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "OFI Thompson-Sampling Bandit paper host for SPY (Alpaca VALK)."
        )
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute signals & log decisions; do NOT submit any Alpaca orders.",
    )
    p.add_argument(
        "--duration",
        type=int,
        default=0,
        help="If > 0, exit cleanly after this many wall-clock seconds (test helper).",
    )
    return p.parse_args()


def main() -> int:
    _load_env_local()
    args = parse_args()
    return run_host(args)


if __name__ == "__main__":
    sys.exit(main())
