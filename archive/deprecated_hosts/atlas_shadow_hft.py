"""ATLAS Shadow HFT Lab — non-trading prediction logger.

Mirrors the HERMES Rust HFT engine's data + predictor logic but:
  - NEVER places orders
  - Logs every prediction to SQLite (no selectivity gate)
  - Multi-horizon: predicts SPY move at 30s, 60s, 90s lookaheads
  - Auto-evaluates predictions against actual SPY moves
  - Maintains an online RLS predictor on ES log-returns -> SPY log-returns

After N days we'll have:
  - calibration curve (predicted bps bucket -> actual mean)
  - directional hit rate per horizon
  - MAE per horizon
  - "would HERMES's 4-gate have helped or hurt?" comparison

Run:
    .venv/bin/python research/live/atlas_shadow_hft.py
Stop:
    echo stop > research/live/STOP_ATLAS_SHADOW
"""
from __future__ import annotations

import json
import math
import os
import signal
import sqlite3
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path("/Users/weeoto/turbo-carnival")
ENV_FILE = ROOT / "council" / ".env.local"
DB_PATH  = ROOT / "research" / "live" / "atlas_shadow_predictions.sqlite3"
KILL_FILE = ROOT / "research" / "live" / "STOP_ATLAS_SHADOW"


# ---- env loading -------------------------------------------------------

def load_env() -> dict:
    env = {}
    if not ENV_FILE.exists():
        return env
    for ln in ENV_FILE.read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#") or "=" not in ln:
            continue
        k, v = ln.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


ENV = load_env()
ALPACA_KEY    = ENV.get("ALPACA_ATLAS_KEY", "")
ALPACA_SECRET = ENV.get("ALPACA_ATLAS_SECRET", "")
ALPACA_BASE   = "https://paper-api.alpaca.markets"
ALPACA_DATA   = "https://data.alpaca.markets"


# ---- SQLite schema -----------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS shadow_predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts_utc TEXT NOT NULL,
  spy_bid REAL,
  spy_ask REAL,
  spy_mid REAL,
  spread_bps REAL,
  es_samples INTEGER,
  pred_bps_30s REAL,
  pred_bps_60s REAL,
  pred_bps_90s REAL,
  confidence REAL,
  hermes_gate_would_fire INTEGER,
  -- actual outcomes (filled later by evaluator)
  actual_bps_30s REAL,
  actual_bps_60s REAL,
  actual_bps_90s REAL,
  actual_spy_at_30s REAL,
  actual_spy_at_60s REAL,
  actual_spy_at_90s REAL,
  evaluated INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_shadow_ts ON shadow_predictions(ts_utc);
CREATE INDEX IF NOT EXISTS idx_shadow_evaluated ON shadow_predictions(evaluated);

CREATE TABLE IF NOT EXISTS rls_state (
  id INTEGER PRIMARY KEY,
  ts_utc TEXT,
  beta REAL,
  cov REAL,
  samples INTEGER
);
"""


def db_init() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.executescript(SCHEMA)
    return conn


# ---- data fetchers (mirroring Rust engine) -----------------------------

def fetch_spy_quote() -> dict | None:
    try:
        r = requests.get(
            f"{ALPACA_DATA}/v2/stocks/SPY/quotes/latest",
            headers={"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET},
            timeout=5,
        )
        if r.status_code != 200:
            return None
        q = r.json().get("quote") or {}
        bid = float(q.get("bp", 0) or 0)
        ask = float(q.get("ap", 0) or 0)
        if bid <= 0 or ask <= 0:
            return None
        return {
            "bid":  bid,
            "ask":  ask,
            "mid":  (bid + ask) / 2,
            "spread_bps": ((ask - bid) / ((ask + bid) / 2)) * 1e4,
            "ts":   q.get("t"),
        }
    except Exception:
        return None


def fetch_spy_trade() -> float | None:
    """Get the most recent SPY trade price (for actual-outcome evaluation)."""
    try:
        r = requests.get(
            f"{ALPACA_DATA}/v2/stocks/SPY/trades/latest",
            headers={"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET},
            timeout=5,
        )
        if r.status_code != 200:
            return None
        t = r.json().get("trade") or {}
        return float(t.get("p", 0) or 0) or None
    except Exception:
        return None


def fetch_es_bars() -> list[dict]:
    """Pull ES=F 1-min bars from Yahoo (same source as Rust engine after fix)."""
    try:
        r = requests.get(
            "https://query1.finance.yahoo.com/v8/finance/chart/ES%3DF",
            params={"interval": "1m", "range": "1d"},
            headers={"User-Agent": "atlas-shadow/1.0"},
            timeout=6,
        )
        if r.status_code != 200:
            return []
        d = r.json()
        result = (d.get("chart") or {}).get("result") or []
        if not result:
            return []
        meta = result[0].get("meta", {})
        ts   = result[0].get("timestamp") or []
        close = (result[0].get("indicators", {}).get("quote") or [{}])[0].get("close") or []
        bars = []
        for t_i, c in zip(ts, close):
            if c is None:
                continue
            bars.append({"ts": int(t_i), "close": float(c)})
        return bars[-30:]  # last 30 minutes
    except Exception:
        return []


# ---- online RLS predictor (mirrors Rust signal.rs) ---------------------

class OnlineRLS:
    """Single-feature recursive least squares: y_t = beta * x_t + noise.

    x_t = log-return of ES over last bar
    y_t = log-return of SPY over equivalent window

    Forgetting factor lambda < 1 lets old samples decay.
    """
    def __init__(self, lam: float = 0.99, p0: float = 100.0):
        self.beta = 0.0
        self.cov  = p0      # 1x1 covariance, scalar
        self.lam  = lam
        self.samples = 0
        self.recent_errors = deque(maxlen=50)  # for confidence estimate

    def update(self, x: float, y: float) -> None:
        # RLS recursion in 1-D:
        # gain = cov * x / (lambda + cov * x^2)
        # error = y - beta * x
        # beta  += gain * error
        # cov   = (1 - gain * x) * cov / lambda
        denom = self.lam + self.cov * x * x
        if denom <= 0:
            return
        gain = self.cov * x / denom
        err  = y - self.beta * x
        self.beta += gain * err
        self.cov   = (1.0 - gain * x) * self.cov / self.lam
        self.samples += 1
        self.recent_errors.append(abs(err))

    def predict(self, x: float) -> float:
        return self.beta * x

    def confidence(self) -> float:
        """Heuristic 0-1 confidence: low when cov is high or recent errors are high."""
        if self.samples < 5:
            return 0.0
        # Map cov to a soft score; smaller cov = tighter estimate = higher confidence
        cov_score = math.exp(-self.cov * 1e4)  # cov is small (variance of beta)
        # Recent error score: low MAE = high confidence
        if not self.recent_errors:
            err_score = 0.0
        else:
            mean_err = sum(self.recent_errors) / len(self.recent_errors)
            # Anything under 0.5 bps mean abs error -> 1.0, scaling down
            err_score = max(0.0, 1.0 - mean_err * 1e3)
        return min(1.0, 0.5 * cov_score + 0.5 * err_score)


# ---- Hermes 4-gate predicate (for comparison logging) ------------------

def hermes_would_fire(pred_bps: float, conf: float, spread_bps: float,
                      latency_p99_ms: float) -> bool:
    """Replicates HERMES selective gate (without firing)."""
    return (
        latency_p99_ms < 250.0
        and conf > 0.55
        and abs(pred_bps) > 1.5
        and abs(pred_bps) > 3.0 * spread_bps
    )


# ---- main loop ---------------------------------------------------------

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_predict(conn: sqlite3.Connection, row: dict) -> int:
    cur = conn.execute("""
        INSERT INTO shadow_predictions
          (ts_utc, spy_bid, spy_ask, spy_mid, spread_bps, es_samples,
           pred_bps_30s, pred_bps_60s, pred_bps_90s, confidence,
           hermes_gate_would_fire)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (
        row["ts_utc"], row["spy_bid"], row["spy_ask"], row["spy_mid"],
        row["spread_bps"], row["es_samples"],
        row["pred_bps_30s"], row["pred_bps_60s"], row["pred_bps_90s"],
        row["confidence"], int(row["hermes_gate_would_fire"]),
    ))
    conn.commit()
    return cur.lastrowid


def evaluate_pending(conn: sqlite3.Connection) -> int:
    """Look at unevaluated predictions and update if enough time has passed."""
    now = datetime.now(timezone.utc)
    cur = conn.execute("""
        SELECT id, ts_utc, spy_mid FROM shadow_predictions
        WHERE evaluated=0 ORDER BY id ASC LIMIT 200
    """)
    rows = cur.fetchall()
    if not rows:
        return 0

    spy_now = fetch_spy_trade()
    if spy_now is None:
        return 0

    n_updated = 0
    for rid, ts_utc, mid in rows:
        try:
            t0 = datetime.fromisoformat(ts_utc)
        except Exception:
            continue
        age = (now - t0).total_seconds()
        if age < 30:
            continue  # too early for 30s eval
        update = {}
        if age >= 30 and age < 60:
            update["actual_spy_at_30s"] = spy_now
            update["actual_bps_30s"] = ((spy_now - mid) / mid) * 1e4
        elif age >= 60 and age < 90:
            update["actual_spy_at_60s"] = spy_now
            update["actual_bps_60s"] = ((spy_now - mid) / mid) * 1e4
            if "actual_bps_30s" not in update:
                # backfill 30s using current price as proxy
                update["actual_spy_at_30s"] = spy_now
                update["actual_bps_30s"] = ((spy_now - mid) / mid) * 1e4
        elif age >= 90:
            update["actual_spy_at_90s"] = spy_now
            update["actual_bps_90s"] = ((spy_now - mid) / mid) * 1e4
            if "actual_bps_30s" not in update:
                update["actual_spy_at_30s"] = spy_now
                update["actual_bps_30s"] = ((spy_now - mid) / mid) * 1e4
            if "actual_bps_60s" not in update:
                update["actual_spy_at_60s"] = spy_now
                update["actual_bps_60s"] = ((spy_now - mid) / mid) * 1e4
            update["evaluated"] = 1
        if not update:
            continue
        cols = ", ".join(f"{k}=?" for k in update.keys())
        conn.execute(f"UPDATE shadow_predictions SET {cols} WHERE id=?",
                     (*update.values(), rid))
        n_updated += 1
    conn.commit()
    return n_updated


def kill_requested() -> bool:
    return KILL_FILE.exists()


def main() -> int:
    if not ALPACA_KEY or not ALPACA_SECRET:
        print("ALPACA_ATLAS_KEY/SECRET not set — abort.", file=sys.stderr)
        return 1
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if KILL_FILE.exists():
        KILL_FILE.unlink()

    conn = db_init()
    rls = OnlineRLS(lam=0.99)
    prev_es_close: float | None = None
    prev_spy_mid: float | None = None

    stop = False
    def _sig(_signum, _frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    print(f"[ATLAS shadow] starting at {now_utc_iso()}")
    print(f"  db:        {DB_PATH}")
    print(f"  kill_file: {KILL_FILE}")
    print(f"  hermes_gate replica: latency<250 AND conf>0.55 AND |pred|>1.5 AND |pred|>3*spread")
    print()

    cycle = 0
    last_eval = time.time()
    while not stop and not kill_requested():
        cycle += 1
        t_cycle = datetime.now(timezone.utc)
        spy = fetch_spy_quote()
        es_bars = fetch_es_bars()

        if not spy or len(es_bars) < 2:
            time.sleep(15)
            continue

        # Update RLS with the latest bar pair (if we have a previous baseline)
        latest_es_close = es_bars[-1]["close"]
        if prev_es_close is not None and prev_spy_mid is not None:
            es_ret  = math.log(latest_es_close / prev_es_close)
            spy_ret = math.log(spy["mid"] / prev_spy_mid)
            # x_t = ES log-return at previous interval, y_t = SPY log-return
            rls.update(es_ret, spy_ret)

        # Predict the next-interval SPY move based on the most-recent ES
        # log-return. Multi-horizon = same prediction scaled by horizon ratio.
        if prev_es_close is not None:
            latest_es_ret = math.log(latest_es_close / prev_es_close)
            base_pred_bps = rls.predict(latest_es_ret) * 1e4
        else:
            base_pred_bps = 0.0

        # Naive multi-horizon scaling — for now they're the same;
        # later we can plug in horizon-specific betas.
        pred_30s = base_pred_bps
        pred_60s = base_pred_bps * 1.0
        pred_90s = base_pred_bps * 1.0

        conf = rls.confidence()
        # No latency probe in shadow lab — assume nominal 100ms for gate replica
        hermes_fire = hermes_would_fire(base_pred_bps, conf, spy["spread_bps"], 100.0)

        row = {
            "ts_utc":         t_cycle.isoformat(),
            "spy_bid":        spy["bid"],
            "spy_ask":        spy["ask"],
            "spy_mid":        spy["mid"],
            "spread_bps":     round(spy["spread_bps"], 3),
            "es_samples":     rls.samples,
            "pred_bps_30s":   round(pred_30s, 4),
            "pred_bps_60s":   round(pred_60s, 4),
            "pred_bps_90s":   round(pred_90s, 4),
            "confidence":     round(conf, 4),
            "hermes_gate_would_fire": hermes_fire,
        }
        rid = log_predict(conn, row)

        if cycle % 10 == 1 or hermes_fire:
            print(f"[{t_cycle.strftime('%H:%M:%S')}] cycle {cycle:>4} "
                  f"spy=${spy['mid']:.2f} spread={spy['spread_bps']:.2f}bps "
                  f"pred={base_pred_bps:+.3f}bps conf={conf:.2f} "
                  f"samples={rls.samples} hermes_fire={hermes_fire} (id={rid})")

        # Evaluator pass every ~30s
        if time.time() - last_eval > 30:
            n = evaluate_pending(conn)
            if n:
                print(f"  [evaluator] updated {n} rows")
            last_eval = time.time()

        # Persist RLS state occasionally for diagnostics
        if cycle % 20 == 0:
            conn.execute("DELETE FROM rls_state")
            conn.execute(
                "INSERT INTO rls_state(ts_utc, beta, cov, samples) VALUES (?,?,?,?)",
                (now_utc_iso(), rls.beta, rls.cov, rls.samples),
            )
            conn.commit()

        prev_es_close = latest_es_close
        prev_spy_mid  = spy["mid"]
        time.sleep(15)  # 15s cycle (twice the rate of HERMES)

    print(f"[ATLAS shadow] exiting at {now_utc_iso()} after {cycle} cycles")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
