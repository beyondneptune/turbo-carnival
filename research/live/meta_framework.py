#!/usr/bin/env python3
"""
META Validation Framework — Strategy Registry + Auto-Promote/Kill Engine.

Implements the operator-level platform described in
/Users/weeoto/turbo-carnival/research/candidates/iter_25_meta_validation_framework.md

This is NOT a trading strategy. It ingests the other strategies and emits state
transitions: research -> shadow -> tiny_live -> live -> killed.

Design philosophy (per iter_25 spec):
- Git is the source of truth; the sqlite mirror exists for fast queries.
- Promotion to `live` requires every gate to pass (broker truth, kill criteria
  declared, kill criteria never breached, etc.). No code path can bypass.
- Auto-kill is autonomous (defensive action cannot wait for human). Auto-promote
  to `live` is the only gate that benefits from a human-in-the-loop, but for
  the tiny_live tier (the iter_25 spec's intermediate) we trust the gate so the
  registry stays moving.
- Each strategy host owns its in-process kill check; this framework is the
  redundant outer layer that touches STOP files to halt hosts.

Run modes (CLI):
    python meta_framework.py init                    # create db + pre-register
    python meta_framework.py register --id X ...      # add a strategy
    python meta_framework.py list                     # show registered
    python meta_framework.py metrics --strategy X --days N
    python meta_framework.py promote-check            # auto-promote
    python meta_framework.py kill-check               # auto-kill
    python meta_framework.py force-kill --strategy X --reason "..."
    python meta_framework.py global-kill              # panic button
    python meta_framework.py dashboard-snapshot       # JSON for dashboard
    python meta_framework.py record-daily --strategy X --date ... \
        --ann-return-pct N --sharpe N --max-dd-pct N --trades N \
        --win-rate N --broker-truth-match true --drawdown-pct-today N
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Constants and paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path("/Users/weeoto/turbo-carnival")
LIVE_DIR = REPO_ROOT / "research" / "live"
DB_PATH = LIVE_DIR / "meta_registry.sqlite3"
LOG_PATH = LIVE_DIR / "meta_framework.log"

VALID_STATUSES = {"research", "shadow", "tiny_live", "live", "killed", "paused", "quarantine"}
VALID_BROKERS = {"alpaca", "ibkr", "binance", "okx", "coinbase", "ccxt"}
PROMOTABLE_FROM_SHADOW = "shadow"
PROMOTABLE_FROM_TINY = "tiny_live"
MIN_SHADOW_DAYS_FOR_PROMOTE = 30
MIN_TINY_LIVE_DAYS_FOR_LIVE = 14
GLOBAL_KILL_LOG = LIVE_DIR / "META_GLOBAL_KILL.log"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LIVE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("meta_framework")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def open_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def verify_git_sha(sha: str) -> bool:
    """Return True iff `sha` is a real commit reachable from the turbo-carnival repo."""
    if not sha or len(sha) < 7:
        return False
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "cat-file", "-t", sha],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and result.stdout.strip() == "commit"


def current_repo_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return ""


# ---------------------------------------------------------------------------
# Schema bootstrap
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS strategies (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version_sha TEXT NOT NULL,
    status TEXT NOT NULL,
    deployed_at TEXT NOT NULL,
    host_path TEXT NOT NULL,
    broker TEXT NOT NULL,
    account_id TEXT,
    verdict_criteria_json TEXT NOT NULL,
    capacity_ceiling_usd REAL NOT NULL,
    kill_criteria_json TEXT NOT NULL,
    stop_file_path TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS daily_metrics (
    strategy_id TEXT NOT NULL,
    date TEXT NOT NULL,
    ann_return_pct REAL,
    sharpe REAL,
    max_dd_pct REAL,
    trades INTEGER,
    win_rate REAL,
    broker_truth_match INTEGER,   -- sqlite has no BOOLEAN; store 0/1
    drawdown_pct_today REAL,
    PRIMARY KEY (strategy_id, date),
    FOREIGN KEY (strategy_id) REFERENCES strategies(id)
);

CREATE TABLE IF NOT EXISTS state_transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id TEXT NOT NULL,
    from_status TEXT,
    to_status TEXT NOT NULL,
    reason TEXT,
    at TEXT NOT NULL,
    FOREIGN KEY (strategy_id) REFERENCES strategies(id)
);

CREATE INDEX IF NOT EXISTS idx_metrics_strategy_date
    ON daily_metrics(strategy_id, date DESC);
CREATE INDEX IF NOT EXISTS idx_transitions_strategy_at
    ON state_transitions(strategy_id, at DESC);
CREATE INDEX IF NOT EXISTS idx_strategies_status
    ON strategies(status);
"""


def init_db() -> None:
    with closing(open_db()) as conn:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    logger.info("Initialized db at %s", DB_PATH)


# ---------------------------------------------------------------------------
# Registration (with lesson enforcement)
# ---------------------------------------------------------------------------

REQUIRED_KILL_KEYS = {"kill_3d_dd_pct", "kill_daily_loss_pct", "kill_rolling_sharpe"}


def _validate_kill_criteria(kc: dict[str, Any]) -> Optional[str]:
    """Return None if kc is valid, otherwise an error string."""
    if not isinstance(kc, dict):
        return "kill_criteria_json must be a JSON object"
    missing = REQUIRED_KILL_KEYS - set(kc.keys())
    if missing:
        return f"kill_criteria_json missing required keys: {sorted(missing)}"
    for k in REQUIRED_KILL_KEYS:
        if not isinstance(kc[k], (int, float)):
            return f"kill_criteria_json key {k} must be numeric"
    return None


def _validate_verdict_criteria(vc: dict[str, Any]) -> Optional[str]:
    if not isinstance(vc, dict):
        return "verdict_criteria_json must be a JSON object"
    # promote thresholds; soft requirement — any of these may be unset, but at
    # least one of (promote_min_sharpe, promote_max_dd_pct, promote_min_ann_return_pct)
    # should exist or there is no promotion path.
    keys = {"promote_min_sharpe", "promote_max_dd_pct", "promote_min_ann_return_pct"}
    if not keys & set(vc.keys()):
        return f"verdict_criteria_json must include at least one of {sorted(keys)}"
    return None


def register_strategy(
    *,
    strategy_id: str,
    name: str,
    version_sha: str,
    status: str,
    host_path: str,
    broker: str,
    account_id: Optional[str],
    verdict_criteria: dict[str, Any],
    capacity_ceiling_usd: float,
    kill_criteria: dict[str, Any],
    stop_file_path: Optional[str] = None,
    notes: Optional[str] = None,
    allow_existing: bool = False,
) -> None:
    """
    Insert (or upsert if allow_existing) a strategy row.

    Hard rules (iter_25 §2 / §11 lesson 10):
    - kill_criteria_json must be present with required keys
    - version_sha must resolve to a real git commit in the turbo-carnival repo
    - status must be one of VALID_STATUSES
    - broker must be in VALID_BROKERS
    - capacity_ceiling_usd must be > 0
    """
    if status not in VALID_STATUSES:
        raise ValueError(f"status must be one of {sorted(VALID_STATUSES)}; got {status!r}")
    if broker not in VALID_BROKERS:
        raise ValueError(f"broker must be one of {sorted(VALID_BROKERS)}; got {broker!r}")
    if capacity_ceiling_usd <= 0:
        raise ValueError("capacity_ceiling_usd must be > 0 (Lesson 9: capacity bound named upfront)")

    err = _validate_kill_criteria(kill_criteria)
    if err:
        raise ValueError(
            "register rejected — Lesson 10 (kill criteria written before deploy): " + err
        )

    err = _validate_verdict_criteria(verdict_criteria)
    if err:
        raise ValueError("register rejected — " + err)

    if not verify_git_sha(version_sha):
        raise ValueError(
            f"register rejected — version_sha {version_sha!r} does not resolve to a real "
            f"git commit in {REPO_ROOT}. The registry pins each strategy to a commit."
        )

    if not Path(host_path).is_file():
        # Soft-warn only: host may be in flight. But the path itself must look real.
        logger.warning("host_path %s does not exist on disk yet", host_path)

    with closing(open_db()) as conn:
        existing = conn.execute(
            "SELECT id, status FROM strategies WHERE id = ?", (strategy_id,)
        ).fetchone()
        if existing and not allow_existing:
            raise ValueError(
                f"strategy {strategy_id!r} already registered (status={existing['status']}). "
                "Use --allow-existing to upsert."
            )

        from_status = existing["status"] if existing else None
        now_iso = utc_now_iso()

        conn.execute(
            """
            INSERT INTO strategies (
                id, name, version_sha, status, deployed_at, host_path, broker,
                account_id, verdict_criteria_json, capacity_ceiling_usd,
                kill_criteria_json, stop_file_path, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                version_sha = excluded.version_sha,
                status = excluded.status,
                host_path = excluded.host_path,
                broker = excluded.broker,
                account_id = excluded.account_id,
                verdict_criteria_json = excluded.verdict_criteria_json,
                capacity_ceiling_usd = excluded.capacity_ceiling_usd,
                kill_criteria_json = excluded.kill_criteria_json,
                stop_file_path = excluded.stop_file_path,
                notes = excluded.notes
            """,
            (
                strategy_id,
                name,
                version_sha,
                status,
                now_iso,
                host_path,
                broker,
                account_id,
                json.dumps(verdict_criteria, sort_keys=True),
                float(capacity_ceiling_usd),
                json.dumps(kill_criteria, sort_keys=True),
                stop_file_path,
                notes,
            ),
        )
        conn.execute(
            "INSERT INTO state_transitions (strategy_id, from_status, to_status, reason, at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                strategy_id,
                from_status,
                status,
                "register" if not existing else "re-register (upsert)",
                now_iso,
            ),
        )
        conn.commit()

    logger.info("registered strategy id=%s status=%s sha=%s", strategy_id, status, version_sha[:8])


# ---------------------------------------------------------------------------
# Metric ingest
# ---------------------------------------------------------------------------

def record_daily_metric(
    *,
    strategy_id: str,
    date: str,
    ann_return_pct: Optional[float],
    sharpe: Optional[float],
    max_dd_pct: Optional[float],
    trades: Optional[int],
    win_rate: Optional[float],
    broker_truth_match: Optional[bool],
    drawdown_pct_today: Optional[float],
) -> None:
    with closing(open_db()) as conn:
        existing = conn.execute(
            "SELECT id FROM strategies WHERE id = ?", (strategy_id,)
        ).fetchone()
        if not existing:
            raise ValueError(f"unknown strategy {strategy_id!r} — register first")
        conn.execute(
            """
            INSERT INTO daily_metrics (
                strategy_id, date, ann_return_pct, sharpe, max_dd_pct, trades,
                win_rate, broker_truth_match, drawdown_pct_today
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(strategy_id, date) DO UPDATE SET
                ann_return_pct = excluded.ann_return_pct,
                sharpe = excluded.sharpe,
                max_dd_pct = excluded.max_dd_pct,
                trades = excluded.trades,
                win_rate = excluded.win_rate,
                broker_truth_match = excluded.broker_truth_match,
                drawdown_pct_today = excluded.drawdown_pct_today
            """,
            (
                strategy_id,
                date,
                ann_return_pct,
                sharpe,
                max_dd_pct,
                trades,
                win_rate,
                None if broker_truth_match is None else int(bool(broker_truth_match)),
                drawdown_pct_today,
            ),
        )
        conn.commit()
    logger.info("recorded daily metric strategy=%s date=%s", strategy_id, date)


# ---------------------------------------------------------------------------
# Promotion logic
# ---------------------------------------------------------------------------

def _recent_metrics(conn: sqlite3.Connection, strategy_id: str, n_days: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM daily_metrics WHERE strategy_id = ? "
        "ORDER BY date DESC LIMIT ?",
        (strategy_id, n_days),
    ).fetchall()


def _has_ever_broker_truth_failure(conn: sqlite3.Connection, strategy_id: str) -> bool:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM daily_metrics "
        "WHERE strategy_id = ? AND broker_truth_match = 0",
        (strategy_id,),
    ).fetchone()
    return bool(row and row["n"] > 0)


def _verdict_pass(metrics: list[sqlite3.Row], criteria: dict[str, Any]) -> tuple[bool, list[str]]:
    """Return (passes, reasons-for-pass-or-fail)."""
    if not metrics:
        return False, ["no metrics on file"]

    # Compute averages and worst-case
    sharpes = [m["sharpe"] for m in metrics if m["sharpe"] is not None]
    anns = [m["ann_return_pct"] for m in metrics if m["ann_return_pct"] is not None]
    dds = [m["max_dd_pct"] for m in metrics if m["max_dd_pct"] is not None]

    reasons: list[str] = []
    ok = True

    if "promote_min_sharpe" in criteria:
        if not sharpes:
            ok = False
            reasons.append("no sharpe samples")
        else:
            avg_sharpe = sum(sharpes) / len(sharpes)
            threshold = float(criteria["promote_min_sharpe"])
            if avg_sharpe >= threshold:
                reasons.append(f"sharpe avg={avg_sharpe:.2f} >= {threshold}")
            else:
                ok = False
                reasons.append(f"sharpe avg={avg_sharpe:.2f} < {threshold}")

    if "promote_max_dd_pct" in criteria:
        if not dds:
            ok = False
            reasons.append("no max_dd samples")
        else:
            worst_dd = max(dds)  # max_dd_pct is positive
            threshold = float(criteria["promote_max_dd_pct"])
            if worst_dd <= threshold:
                reasons.append(f"max_dd worst={worst_dd:.2f} <= {threshold}")
            else:
                ok = False
                reasons.append(f"max_dd worst={worst_dd:.2f} > {threshold}")

    if "promote_min_ann_return_pct" in criteria:
        if not anns:
            ok = False
            reasons.append("no ann_return samples")
        else:
            avg_ann = sum(anns) / len(anns)
            threshold = float(criteria["promote_min_ann_return_pct"])
            if avg_ann >= threshold:
                reasons.append(f"ann_return avg={avg_ann:.2f} >= {threshold}")
            else:
                ok = False
                reasons.append(f"ann_return avg={avg_ann:.2f} < {threshold}")

    # Broker-truth match must hold for the window
    truth_fails = [m for m in metrics if m["broker_truth_match"] == 0]
    if truth_fails:
        ok = False
        reasons.append(f"broker_truth_match=False on {len(truth_fails)} day(s) in window")

    return ok, reasons


def _kill_check(metrics: list[sqlite3.Row], criteria: dict[str, Any]) -> Optional[str]:
    """Return None if no kill triggered, else the reason."""
    if not metrics:
        return None

    # 3-day rolling drawdown breach
    last3 = metrics[:3]
    dd3 = [m["drawdown_pct_today"] for m in last3 if m["drawdown_pct_today"] is not None]
    if dd3 and "kill_3d_dd_pct" in criteria:
        worst3 = max(dd3)
        if worst3 > float(criteria["kill_3d_dd_pct"]):
            return f"3-day DD breach: worst3={worst3:.2f}% > {criteria['kill_3d_dd_pct']}"

    # Daily loss cap
    if "kill_daily_loss_pct" in criteria:
        worst_day = None
        for m in metrics:
            if m["drawdown_pct_today"] is not None:
                if worst_day is None or m["drawdown_pct_today"] > worst_day:
                    worst_day = m["drawdown_pct_today"]
        if worst_day is not None and worst_day > float(criteria["kill_daily_loss_pct"]):
            return f"daily loss cap breach: {worst_day:.2f}% > {criteria['kill_daily_loss_pct']}"

    # Rolling Sharpe collapse
    if "kill_rolling_sharpe" in criteria:
        sharpes = [m["sharpe"] for m in metrics if m["sharpe"] is not None]
        if sharpes:
            avg_sharpe = sum(sharpes) / len(sharpes)
            if avg_sharpe < float(criteria["kill_rolling_sharpe"]):
                return f"rolling sharpe collapse: {avg_sharpe:.2f} < {criteria['kill_rolling_sharpe']}"

    # Broker-truth divergence (any single day) — for live tiers this is a hard kill
    truth_fail = [m for m in metrics if m["broker_truth_match"] == 0]
    if truth_fail:
        return f"broker-truth divergence on {len(truth_fail)} day(s) in window"

    # 60-day DD if declared
    if "kill_60d_dd_pct" in criteria:
        dd60 = [m["max_dd_pct"] for m in metrics if m["max_dd_pct"] is not None]
        if dd60:
            worst60 = max(dd60)
            if worst60 > float(criteria["kill_60d_dd_pct"]):
                return f"60-day DD breach: worst={worst60:.2f}% > {criteria['kill_60d_dd_pct']}"

    return None


def _transition(
    conn: sqlite3.Connection,
    strategy_id: str,
    new_status: str,
    reason: str,
) -> None:
    row = conn.execute(
        "SELECT status FROM strategies WHERE id = ?", (strategy_id,)
    ).fetchone()
    from_status = row["status"] if row else None
    now_iso = utc_now_iso()
    conn.execute(
        "UPDATE strategies SET status = ? WHERE id = ?", (new_status, strategy_id)
    )
    conn.execute(
        "INSERT INTO state_transitions (strategy_id, from_status, to_status, reason, at) "
        "VALUES (?, ?, ?, ?, ?)",
        (strategy_id, from_status, new_status, reason, now_iso),
    )
    logger.info(
        "TRANSITION %s: %s -> %s (%s)", strategy_id, from_status, new_status, reason
    )


def promote_check() -> dict[str, Any]:
    """
    Run promotion logic for all strategies in shadow / tiny_live.

    shadow -> tiny_live: requires >= MIN_SHADOW_DAYS_FOR_PROMOTE daily metrics
        and all promote_* criteria pass
    tiny_live -> live: requires >= MIN_TINY_LIVE_DAYS_FOR_LIVE daily metrics,
        criteria still pass, and broker_truth_match has NEVER been False
    """
    result: dict[str, Any] = {
        "checked_at": utc_now_iso(),
        "promotions": [],
        "no_action": [],
        "kills": [],
    }
    with closing(open_db()) as conn:
        # SHADOW -> TINY_LIVE
        shadow_rows = conn.execute(
            "SELECT * FROM strategies WHERE status = ?", (PROMOTABLE_FROM_SHADOW,)
        ).fetchall()
        for row in shadow_rows:
            strategy_id = row["id"]
            metrics = _recent_metrics(conn, strategy_id, MIN_SHADOW_DAYS_FOR_PROMOTE)
            if len(metrics) < MIN_SHADOW_DAYS_FOR_PROMOTE:
                result["no_action"].append({
                    "id": strategy_id,
                    "status": "shadow",
                    "reason": f"only {len(metrics)} daily metrics (need {MIN_SHADOW_DAYS_FOR_PROMOTE})",
                })
                continue
            criteria = json.loads(row["verdict_criteria_json"])
            kill_criteria = json.loads(row["kill_criteria_json"])
            kill_reason = _kill_check(metrics, kill_criteria)
            if kill_reason:
                _transition(conn, strategy_id, "killed", f"shadow kill: {kill_reason}")
                _touch_stop_file(row)
                result["kills"].append({"id": strategy_id, "reason": kill_reason})
                continue
            passes, reasons = _verdict_pass(metrics, criteria)
            if passes:
                _transition(
                    conn,
                    strategy_id,
                    "tiny_live",
                    "shadow->tiny_live: " + "; ".join(reasons),
                )
                result["promotions"].append({
                    "id": strategy_id,
                    "from": "shadow",
                    "to": "tiny_live",
                    "reasons": reasons,
                })
            else:
                result["no_action"].append({
                    "id": strategy_id,
                    "status": "shadow",
                    "reason": "verdict fail: " + "; ".join(reasons),
                })

        # TINY_LIVE -> LIVE
        tiny_rows = conn.execute(
            "SELECT * FROM strategies WHERE status = ?", (PROMOTABLE_FROM_TINY,)
        ).fetchall()
        for row in tiny_rows:
            strategy_id = row["id"]
            metrics = _recent_metrics(conn, strategy_id, MIN_TINY_LIVE_DAYS_FOR_LIVE)
            if len(metrics) < MIN_TINY_LIVE_DAYS_FOR_LIVE:
                result["no_action"].append({
                    "id": strategy_id,
                    "status": "tiny_live",
                    "reason": f"only {len(metrics)} daily metrics (need {MIN_TINY_LIVE_DAYS_FOR_LIVE})",
                })
                continue
            criteria = json.loads(row["verdict_criteria_json"])
            kill_criteria = json.loads(row["kill_criteria_json"])
            kill_reason = _kill_check(metrics, kill_criteria)
            if kill_reason:
                _transition(conn, strategy_id, "killed", f"tiny_live kill: {kill_reason}")
                _touch_stop_file(row)
                result["kills"].append({"id": strategy_id, "reason": kill_reason})
                continue
            passes, reasons = _verdict_pass(metrics, criteria)
            # HARD GATE: broker_truth_match must have NEVER been False for live
            if passes and _has_ever_broker_truth_failure(conn, strategy_id):
                result["no_action"].append({
                    "id": strategy_id,
                    "status": "tiny_live",
                    "reason": "blocked from live: broker_truth_match was False at least once historically (Lesson 4)",
                })
                continue
            if passes:
                _transition(
                    conn,
                    strategy_id,
                    "live",
                    "tiny_live->live: " + "; ".join(reasons),
                )
                result["promotions"].append({
                    "id": strategy_id,
                    "from": "tiny_live",
                    "to": "live",
                    "reasons": reasons,
                })
            else:
                result["no_action"].append({
                    "id": strategy_id,
                    "status": "tiny_live",
                    "reason": "verdict fail: " + "; ".join(reasons),
                })

        conn.commit()
    return result


def kill_check() -> dict[str, Any]:
    """Auto-kill any tiny_live or live strategy whose kill criteria triggered."""
    result: dict[str, Any] = {
        "checked_at": utc_now_iso(),
        "kills": [],
        "ok": [],
    }
    with closing(open_db()) as conn:
        rows = conn.execute(
            "SELECT * FROM strategies WHERE status IN ('tiny_live', 'live')"
        ).fetchall()
        for row in rows:
            strategy_id = row["id"]
            kill_criteria = json.loads(row["kill_criteria_json"])
            # Use a 60-day window so we catch rolling drawdown criteria
            metrics = _recent_metrics(conn, strategy_id, 60)
            reason = _kill_check(metrics, kill_criteria)
            if reason:
                _transition(conn, strategy_id, "killed", f"auto-kill: {reason}")
                _touch_stop_file(row)
                result["kills"].append({"id": strategy_id, "reason": reason})
            else:
                result["ok"].append({"id": strategy_id, "status": row["status"]})
        conn.commit()
    return result


def force_kill(strategy_id: str, reason: str) -> None:
    """Operator-issued emergency kill."""
    with closing(open_db()) as conn:
        row = conn.execute(
            "SELECT * FROM strategies WHERE id = ?", (strategy_id,)
        ).fetchone()
        if not row:
            raise ValueError(f"unknown strategy {strategy_id!r}")
        _transition(conn, strategy_id, "killed", f"force-kill: {reason}")
        _touch_stop_file(row)
        conn.commit()
    logger.warning("FORCE-KILLED %s: %s", strategy_id, reason)


def _touch_stop_file(row: sqlite3.Row) -> None:
    """If the strategy declared a stop_file_path, touch it to halt the host."""
    stop_path = row["stop_file_path"]
    if not stop_path:
        return
    try:
        Path(stop_path).touch()
        logger.info("touched STOP file %s for %s", stop_path, row["id"])
    except OSError as exc:
        logger.error("failed to touch STOP file %s: %s", stop_path, exc)


# ---------------------------------------------------------------------------
# Global kill
# ---------------------------------------------------------------------------

def global_kill(reason: str = "operator global kill") -> dict[str, Any]:
    """Touch STOP files for ALL registered strategies and mark them paused.

    Note: this does NOT set status=killed (operator may want to recover); it
    sets status=paused so a follow-up review can reverse or finalize.
    """
    result: dict[str, Any] = {
        "fired_at": utc_now_iso(),
        "reason": reason,
        "touched": [],
        "missing_stop_file": [],
    }
    with closing(open_db()) as conn:
        rows = conn.execute(
            "SELECT * FROM strategies WHERE status NOT IN ('killed', 'research')"
        ).fetchall()
        for row in rows:
            stop_path = row["stop_file_path"]
            if not stop_path:
                result["missing_stop_file"].append(row["id"])
                continue
            try:
                Path(stop_path).touch()
                result["touched"].append({"id": row["id"], "stop_file": stop_path})
            except OSError as exc:
                logger.error("global-kill: failed to touch %s: %s", stop_path, exc)
            _transition(conn, row["id"], "paused", f"global-kill: {reason}")
        conn.commit()

    # Append to an append-only log so the panic record cannot be lost
    with open(GLOBAL_KILL_LOG, "a") as f:
        f.write(json.dumps(result) + "\n")

    logger.critical("GLOBAL KILL fired: %s — touched %d STOP files",
                    reason, len(result["touched"]))
    return result


# ---------------------------------------------------------------------------
# Operator views
# ---------------------------------------------------------------------------

def list_strategies() -> list[dict[str, Any]]:
    with closing(open_db()) as conn:
        rows = conn.execute(
            "SELECT id, name, status, broker, version_sha, deployed_at, "
            "capacity_ceiling_usd FROM strategies ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]


def show_metrics(strategy_id: str, days: int) -> list[dict[str, Any]]:
    with closing(open_db()) as conn:
        rows = conn.execute(
            "SELECT * FROM daily_metrics WHERE strategy_id = ? "
            "ORDER BY date DESC LIMIT ?",
            (strategy_id, days),
        ).fetchall()
        return [dict(r) for r in rows]


def dashboard_snapshot() -> dict[str, Any]:
    """JSON shape suited for the monitoring dashboard."""
    snap: dict[str, Any] = {
        "generated_at": utc_now_iso(),
        "framework_sha": current_repo_sha(),
        "strategies": [],
        "totals_by_status": {},
    }
    with closing(open_db()) as conn:
        rows = conn.execute("SELECT * FROM strategies").fetchall()
        for row in rows:
            latest = conn.execute(
                "SELECT * FROM daily_metrics WHERE strategy_id = ? "
                "ORDER BY date DESC LIMIT 1",
                (row["id"],),
            ).fetchone()
            last_transition = conn.execute(
                "SELECT * FROM state_transitions WHERE strategy_id = ? "
                "ORDER BY at DESC LIMIT 1",
                (row["id"],),
            ).fetchone()
            snap["strategies"].append({
                "id": row["id"],
                "name": row["name"],
                "status": row["status"],
                "broker": row["broker"],
                "account_id": row["account_id"],
                "version_sha": row["version_sha"],
                "host_path": row["host_path"],
                "capacity_ceiling_usd": row["capacity_ceiling_usd"],
                "latest_metric": dict(latest) if latest else None,
                "last_transition": dict(last_transition) if last_transition else None,
            })
        for r in conn.execute(
            "SELECT status, COUNT(*) AS n FROM strategies GROUP BY status"
        ).fetchall():
            snap["totals_by_status"][r["status"]] = r["n"]
    return snap


# ---------------------------------------------------------------------------
# Pre-registration of the 4 surviving strategies
# ---------------------------------------------------------------------------

def preregister_known_strategies(version_sha: Optional[str] = None) -> list[str]:
    """
    Register the 4 surviving strategies. Criteria are taken from each iter_N
    spec file. version_sha defaults to current HEAD of the turbo-carnival repo.
    """
    sha = version_sha or current_repo_sha()
    if not sha:
        raise RuntimeError("could not resolve a git SHA to pin strategies to")

    registered: list[str] = []

    # 1. SVXY sweet spot  (host: research/live/svxy_sweet_spot_host.py)
    register_strategy(
        strategy_id="svxy_sweet_spot",
        name="SVXY contango sweet spot (VIX/VIX3M < 1.00)",
        version_sha=sha,
        status="shadow",
        host_path=str(LIVE_DIR / "svxy_sweet_spot_host.py"),
        broker="alpaca",
        account_id="AEGIS",
        verdict_criteria={
            # Backtest expectation: Sharpe 0.96, max DD -8.36%
            "promote_min_sharpe": 0.80,
            "promote_max_dd_pct": 10.0,
        },
        capacity_ceiling_usd=50_000.0,
        kill_criteria={
            # README_SVXY.md hard kills: SVXY -15% intraday, account DD -12%
            "kill_3d_dd_pct": 8.0,
            "kill_daily_loss_pct": 6.0,
            "kill_rolling_sharpe": -0.5,
            "kill_60d_dd_pct": 12.0,
        },
        stop_file_path=str(LIVE_DIR / "STOP_SVXY"),
        notes="Pre-registered iter_25 step 7. STOP_SVXY halts host without flatten; "
              "--flatten-now flag flattens before halt.",
        allow_existing=True,
    )
    registered.append("svxy_sweet_spot")

    # 2. iter_18 ADR cleanup
    register_strategy(
        strategy_id="iter_18_adr",
        name="iter_18 US-HK ADR dual-listing arbitrage (cleaned)",
        version_sha=sha,
        status="shadow",
        host_path=str(LIVE_DIR / "iter_18_adr_host.py"),
        broker="ibkr",
        account_id="DU_PAPER",
        verdict_criteria={
            # iter_18 §11 promote: Sharpe >= 0.9 with bootstrap CI excluding 0.3,
            # avg net P&L >= 25 bps/trade, max DD <= 8% NAV
            "promote_min_sharpe": 0.9,
            "promote_max_dd_pct": 8.0,
            "promote_min_ann_return_pct": 6.0,
        },
        capacity_ceiling_usd=100_000.0,
        kill_criteria={
            # iter_18 §11 kill: 3-day NAV DD > 3.5%, daily loss cap -2.5% NAV
            "kill_3d_dd_pct": 3.5,
            "kill_daily_loss_pct": 2.5,
            "kill_rolling_sharpe": 0.0,
            "kill_60d_dd_pct": 8.0,
        },
        stop_file_path=str(LIVE_DIR / "KILL_SWITCH"),
        notes="iter_18 cleanup PASS: mean_bps_net 65.79 over 1543 trades, 6.36yr. "
              "Touches KILL_SWITCH (existing host convention) to halt.",
        allow_existing=True,
    )
    registered.append("iter_18_adr")

    # 3. iter_3 BTC funding (parallel build)
    register_strategy(
        strategy_id="iter_3_btc_funding",
        name="iter_3 BTC perpetual funding arbitrage",
        version_sha=sha,
        status="shadow",
        host_path=str(LIVE_DIR / "iter_3_btc_funding_host.py"),
        broker="ccxt",
        account_id="BINANCE_OKX",
        verdict_criteria={
            # iter_3 §11 promote: net carry > 6%/yr, Sharpe > 1.5, max DD < 4%
            "promote_min_sharpe": 1.5,
            "promote_max_dd_pct": 4.0,
            "promote_min_ann_return_pct": 6.0,
        },
        capacity_ceiling_usd=50_000.0,
        kill_criteria={
            # iter_3 §11 kill: 7-day DD > 3%, daily loss -1.5%
            "kill_3d_dd_pct": 2.0,
            "kill_daily_loss_pct": 1.5,
            "kill_rolling_sharpe": 0.0,
            "kill_60d_dd_pct": 6.0,
        },
        stop_file_path=str(LIVE_DIR / "STOP_BTC_FUNDING"),
        notes="iter_3 host being built in parallel. Touches STOP_BTC_FUNDING to halt.",
        allow_existing=True,
    )
    registered.append("iter_3_btc_funding")

    # 4. iter_6 Nikkei gap (parallel build)
    register_strategy(
        strategy_id="iter_6_nikkei_gap",
        name="iter_6 Nikkei futures open-range breakout",
        version_sha=sha,
        status="shadow",
        host_path=str(LIVE_DIR / "iter_6_nikkei_host.py"),
        broker="ibkr",
        account_id="DU_PAPER",
        verdict_criteria={
            # iter_6 §11 promote: Sharpe > 1.2, max DD < 8%, profit factor > 1.25
            "promote_min_sharpe": 1.2,
            "promote_max_dd_pct": 8.0,
        },
        capacity_ceiling_usd=30_000.0,
        kill_criteria={
            # iter_6 §11 kill: 3-day rolling DD > 5%, single-day loss > $1500
            "kill_3d_dd_pct": 5.0,
            "kill_daily_loss_pct": 3.0,
            "kill_rolling_sharpe": 0.0,
            "kill_60d_dd_pct": 10.0,
        },
        stop_file_path=str(LIVE_DIR / "STOP_NIKKEI"),
        notes="iter_6 host being built in parallel. Touches STOP_NIKKEI to halt.",
        allow_existing=True,
    )
    registered.append("iter_6_nikkei_gap")

    return registered


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_json_arg(arg: Optional[str]) -> Optional[dict[str, Any]]:
    if arg is None:
        return None
    if Path(arg).is_file():
        with open(arg) as f:
            return json.load(f)
    return json.loads(arg)


def _print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str, sort_keys=True))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="meta_framework",
        description="iter_25 META validation framework — strategy registry "
                    "and auto-promote/auto-kill engine.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # init
    p_init = sub.add_parser(
        "init", help="initialize the sqlite db (and optionally pre-register the 4 strategies)"
    )
    p_init.add_argument(
        "--with-defaults", action="store_true",
        help="also pre-register the 4 surviving strategies (svxy, iter_18, iter_3, iter_6)",
    )
    p_init.add_argument(
        "--version-sha", help="git SHA to pin pre-registered strategies to (default: HEAD)"
    )

    # register
    p_reg = sub.add_parser("register", help="register a new strategy")
    p_reg.add_argument("--id", required=True)
    p_reg.add_argument("--name", required=True)
    p_reg.add_argument("--version-sha", required=True)
    p_reg.add_argument("--status", required=True, choices=sorted(VALID_STATUSES))
    p_reg.add_argument("--host-path", required=True)
    p_reg.add_argument("--broker", required=True, choices=sorted(VALID_BROKERS))
    p_reg.add_argument("--account-id")
    p_reg.add_argument("--criteria-json", required=True,
                       help="JSON literal or path to file with verdict_criteria")
    p_reg.add_argument("--capacity-ceiling-usd", type=float, required=True)
    p_reg.add_argument("--kill-criteria-json", required=True,
                       help="JSON literal or path to file with kill_criteria")
    p_reg.add_argument("--stop-file-path")
    p_reg.add_argument("--notes")
    p_reg.add_argument("--allow-existing", action="store_true")

    # list
    sub.add_parser("list", help="list all registered strategies")

    # metrics
    p_metrics = sub.add_parser("metrics", help="show recent metrics for one strategy")
    p_metrics.add_argument("--strategy", required=True)
    p_metrics.add_argument("--days", type=int, default=30)

    # record-daily
    p_rec = sub.add_parser("record-daily", help="ingest one daily_metrics row")
    p_rec.add_argument("--strategy", required=True)
    p_rec.add_argument("--date", required=True)
    p_rec.add_argument("--ann-return-pct", type=float)
    p_rec.add_argument("--sharpe", type=float)
    p_rec.add_argument("--max-dd-pct", type=float)
    p_rec.add_argument("--trades", type=int)
    p_rec.add_argument("--win-rate", type=float)
    p_rec.add_argument("--broker-truth-match", choices=["true", "false"])
    p_rec.add_argument("--drawdown-pct-today", type=float)

    # promote-check
    sub.add_parser("promote-check", help="run auto-promote logic")

    # kill-check
    sub.add_parser("kill-check", help="run auto-kill logic")

    # force-kill
    p_fk = sub.add_parser("force-kill", help="operator emergency kill of one strategy")
    p_fk.add_argument("--strategy", required=True)
    p_fk.add_argument("--reason", required=True)

    # global-kill
    p_gk = sub.add_parser("global-kill", help="panic button — touch STOP files for all strategies")
    p_gk.add_argument("--reason", default="operator global kill")
    p_gk.add_argument("--confirm", action="store_true",
                      help="required confirmation flag (prevents accidents)")

    # dashboard-snapshot
    sub.add_parser("dashboard-snapshot", help="JSON snapshot of current portfolio state")

    args = parser.parse_args(argv)

    if args.command == "init":
        init_db()
        if args.with_defaults:
            ids = preregister_known_strategies(version_sha=args.version_sha)
            _print_json({"initialized": True, "pre_registered": ids})
        else:
            _print_json({"initialized": True})
        return 0

    # All other commands need the db to exist
    if not DB_PATH.exists():
        logger.error("db does not exist at %s — run `init` first", DB_PATH)
        return 2

    if args.command == "register":
        verdict_criteria = _parse_json_arg(args.criteria_json) or {}
        kill_criteria = _parse_json_arg(args.kill_criteria_json) or {}
        register_strategy(
            strategy_id=args.id,
            name=args.name,
            version_sha=args.version_sha,
            status=args.status,
            host_path=args.host_path,
            broker=args.broker,
            account_id=args.account_id,
            verdict_criteria=verdict_criteria,
            capacity_ceiling_usd=args.capacity_ceiling_usd,
            kill_criteria=kill_criteria,
            stop_file_path=args.stop_file_path,
            notes=args.notes,
            allow_existing=args.allow_existing,
        )
        _print_json({"registered": args.id})
        return 0

    if args.command == "list":
        _print_json(list_strategies())
        return 0

    if args.command == "metrics":
        _print_json(show_metrics(args.strategy, args.days))
        return 0

    if args.command == "record-daily":
        btm: Optional[bool]
        if args.broker_truth_match is None:
            btm = None
        else:
            btm = args.broker_truth_match == "true"
        record_daily_metric(
            strategy_id=args.strategy,
            date=args.date,
            ann_return_pct=args.ann_return_pct,
            sharpe=args.sharpe,
            max_dd_pct=args.max_dd_pct,
            trades=args.trades,
            win_rate=args.win_rate,
            broker_truth_match=btm,
            drawdown_pct_today=args.drawdown_pct_today,
        )
        _print_json({"recorded": True})
        return 0

    if args.command == "promote-check":
        _print_json(promote_check())
        return 0

    if args.command == "kill-check":
        _print_json(kill_check())
        return 0

    if args.command == "force-kill":
        force_kill(args.strategy, args.reason)
        _print_json({"force_killed": args.strategy, "reason": args.reason})
        return 0

    if args.command == "global-kill":
        if not args.confirm:
            logger.error("global-kill requires --confirm (this halts every host)")
            return 2
        _print_json(global_kill(reason=args.reason))
        return 0

    if args.command == "dashboard-snapshot":
        _print_json(dashboard_snapshot())
        return 0

    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
