"""24/7 paper_v1 watchdog for the Mac mini.

This process is intentionally conservative:

- It never uses ``--allow-synthetic``.
- It never mutates strategy logic, model features, or thresholds.
- It waits while augment/evidence processes are already running.
- It runs the official paper_v1 evidence orchestrator at most once, only after
  the strict priced-cache readiness check passes with ``require_spy_core=True``.

The point is operational continuity, not autonomous research.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import signal
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "cache"
DEFAULT_DB_PATH = REPO_ROOT / "data" / "trading_system.db"
DEFAULT_STATUS_PATH = REPO_ROOT / "data" / "paper_v1_status.json"
DEFAULT_WATCHDOG_STATUS_PATH = REPO_ROOT / "data" / "paper_v1_watchdog_status.json"
DEFAULT_LOG_PATH = REPO_ROOT / "data" / "logs" / "paper_v1_watchdog.log"
DEFAULT_LOCK_PATH = REPO_ROOT / "data" / "paper_v1_watchdog.lock"

PROCESS_NEEDLES = (
    "augment_chain_pricing.py",
    "run_paper_v1_evidence.py",
)

logger = logging.getLogger("paper_v1_watchdog")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def find_active_processes() -> list[dict[str, Any]]:
    """Return paper_v1 augment/evidence processes, excluding this watchdog."""
    current_pid = os.getpid()
    out: list[dict[str, Any]] = []
    for proc in psutil.process_iter(attrs=["pid", "cmdline", "create_time"]):
        try:
            info = proc.info
            pid = int(info.get("pid") or 0)
            if pid == current_pid:
                continue
            cmdline_parts = info.get("cmdline") or []
            cmdline = " ".join(str(part) for part in cmdline_parts)
        except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
            continue
        if not cmdline:
            continue
        if any(needle in cmdline for needle in PROCESS_NEEDLES):
            out.append(
                {
                    "pid": pid,
                    "cmdline": cmdline,
                    "create_time": info.get("create_time"),
                }
            )
    return out


def official_run_exists(db_path: Path) -> bool:
    if not db_path.exists():
        return False
    try:
        conn = sqlite3.connect(f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        return False
    try:
        row = conn.execute(
            "SELECT 1 FROM runs "
            "WHERE run_id LIKE 'paper_v1_evidence_%' "
            "AND run_id NOT LIKE 'paper_v1_unofficial_%' "
            "LIMIT 1"
        ).fetchone()
        return row is not None
    except sqlite3.OperationalError:
        return False
    finally:
        conn.close()


def run_readiness(cache_dir: Path) -> dict[str, Any]:
    scripts_dir = str(REPO_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from check_paper_v1_readiness import check_readiness

    return check_readiness(cache_dir, require_spy_core=True)


def run_official_evidence(
    *,
    cache_dir: Path,
    db_path: Path,
    status_path: Path,
) -> int:
    run_id = "paper_v1_evidence_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_paper_v1_evidence.py"),
        "--run-id",
        run_id,
        "--cache-dir",
        str(cache_dir),
        "--db-path",
        str(db_path),
        "--status-path",
        str(status_path),
    ]
    logger.info("starting official evidence run_id=%s", run_id)
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    for line in proc.stdout.splitlines()[-80:]:
        logger.info("evidence: %s", line)
    logger.info("official evidence exited returncode=%s run_id=%s", proc.returncode, run_id)
    return int(proc.returncode)


def acquire_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fh = lock_path.open("w")
    try:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"another watchdog already holds {lock_path}", file=sys.stderr)
        raise SystemExit(1)
    lock_fh.write(f"{os.getpid()}\n")
    lock_fh.flush()
    return lock_fh


def watchdog_loop(args: argparse.Namespace) -> int:
    log_path = Path(args.log_path).expanduser().resolve()
    status_path = Path(args.status_path).expanduser().resolve()
    watchdog_status_path = Path(args.watchdog_status_path).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    db_path = Path(args.db_path).expanduser().resolve()
    lock_path = Path(args.lock_path).expanduser().resolve()
    configure_logging(log_path)
    lock_fh = acquire_lock(lock_path)

    stop_requested = False

    def _handle_stop(signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True
        logger.info("received signal=%s; stopping after current tick", signum)

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    logger.info(
        "watchdog starting interval=%ss cache_dir=%s db_path=%s",
        args.interval_seconds,
        cache_dir,
        db_path,
    )
    last_ready: dict[str, Any] | None = None
    official_started = False

    while not stop_requested:
        tick = {
            "updated_at_utc": utc_now_iso(),
            "phase": "watching",
            "interval_seconds": int(args.interval_seconds),
            "pid": os.getpid(),
            "log_path": str(log_path),
            "cache_dir": str(cache_dir),
            "db_path": str(db_path),
        }
        try:
            active = find_active_processes()
            tick["active_processes"] = active
            if active:
                tick["action"] = "waiting_for_active_process"
                logger.info(
                    "active paper_v1 process(es): %s",
                    ", ".join(f"{p['pid']}" for p in active),
                )
                write_json_atomic(watchdog_status_path, tick)
                time.sleep(args.interval_seconds)
                continue

            if official_started or official_run_exists(db_path):
                tick["phase"] = "idle"
                tick["action"] = "official_run_already_present"
                write_json_atomic(watchdog_status_path, tick)
                logger.info("official paper_v1 evidence run already present; idle")
                time.sleep(args.interval_seconds)
                continue

            readiness = run_readiness(cache_dir)
            last_ready = readiness
            tick["readiness"] = {
                "ready": readiness.get("ready"),
                "blocker_count": readiness.get("blocker_count"),
                "warning_count": readiness.get("warning_count"),
                "years": {
                    year: {
                        "satellite_priced_file_count": payload.get(
                            "satellite_priced_file_count"
                        ),
                        "satellite_event_count": payload.get("satellite_event_count"),
                        "spy_core": payload.get("spy_core"),
                    }
                    for year, payload in (readiness.get("years") or {}).items()
                    if isinstance(payload, dict)
                },
            }
            if not readiness.get("ready"):
                tick["action"] = "readiness_blocked"
                logger.info(
                    "readiness blocked blockers=%s warnings=%s",
                    readiness.get("blocker_count"),
                    readiness.get("warning_count"),
                )
                write_json_atomic(watchdog_status_path, tick)
                time.sleep(args.interval_seconds)
                continue

            tick["action"] = "starting_official_evidence"
            write_json_atomic(watchdog_status_path, tick)
            rc = run_official_evidence(
                cache_dir=cache_dir,
                db_path=db_path,
                status_path=status_path,
            )
            official_started = True
            tick["updated_at_utc"] = utc_now_iso()
            tick["phase"] = "idle"
            tick["action"] = "official_evidence_finished"
            tick["official_evidence_returncode"] = rc
            write_json_atomic(watchdog_status_path, tick)
        except Exception as exc:  # pragma: no cover - operational guardrail
            logger.exception("watchdog tick failed")
            tick["phase"] = "error"
            tick["action"] = "tick_failed"
            tick["error"] = repr(exc)
            if last_ready is not None:
                tick["last_readiness_ready"] = last_ready.get("ready")
            write_json_atomic(watchdog_status_path, tick)
        time.sleep(args.interval_seconds)

    logger.info("watchdog stopped")
    lock_fh.close()
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Conservative 24/7 watcher for paper_v1 readiness and official run."
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--status-path", type=Path, default=DEFAULT_STATUS_PATH)
    parser.add_argument(
        "--watchdog-status-path",
        type=Path,
        default=DEFAULT_WATCHDOG_STATUS_PATH,
    )
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--lock-path", type=Path, default=DEFAULT_LOCK_PATH)
    parser.add_argument("--interval-seconds", type=int, default=300)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.interval_seconds < 30:
        print("interval_seconds must be >= 30", file=sys.stderr)
        return 2
    return watchdog_loop(args)


if __name__ == "__main__":
    raise SystemExit(main())
