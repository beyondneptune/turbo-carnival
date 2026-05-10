"""Tests for the conservative paper_v1 watchdog daemon."""
from __future__ import annotations

import importlib
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
SCHEMA_PATH = REPO_ROOT / "sql" / "schema.sql"


def _load_mod():
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    if "paper_v1_watchdog" in sys.modules:
        return importlib.reload(sys.modules["paper_v1_watchdog"])
    return importlib.import_module("paper_v1_watchdog")


def test_official_run_exists_ignores_unofficial_runs(tmp_path):
    mod = _load_mod()
    db = tmp_path / "trading_system.db"
    conn = sqlite3.connect(db)
    conn.executescript(SCHEMA_PATH.read_text())
    conn.execute(
        "INSERT INTO runs (run_id, run_type, status, initial_nav_usd, started_at_utc) "
        "VALUES (?, 'paper_frozen', 'completed', 10000, '2026-05-10T00:00:00Z')",
        ("paper_v1_unofficial_smoke_20260510T000000Z",),
    )
    conn.commit()
    conn.close()

    assert mod.official_run_exists(db) is False


def test_official_run_exists_detects_real_evidence_run(tmp_path):
    mod = _load_mod()
    db = tmp_path / "trading_system.db"
    conn = sqlite3.connect(db)
    conn.executescript(SCHEMA_PATH.read_text())
    conn.execute(
        "INSERT INTO runs (run_id, run_type, status, initial_nav_usd, started_at_utc) "
        "VALUES (?, 'paper_frozen', 'completed', 10000, '2026-05-10T00:00:00Z')",
        ("paper_v1_evidence_20260510T000000Z",),
    )
    conn.commit()
    conn.close()

    assert mod.official_run_exists(db) is True


class _FakeProc:
    def __init__(self, pid: int, cmdline: list[str]):
        self.info = {"pid": pid, "cmdline": cmdline, "create_time": 123.0}


def test_find_active_processes_keeps_only_paper_v1_processes(monkeypatch):
    mod = _load_mod()
    fake = [
        _FakeProc(10, ["python", "scripts/augment_chain_pricing.py"]),
        _FakeProc(11, ["python", "scripts/run_paper_v1_evidence.py"]),
        _FakeProc(12, ["python", "other.py"]),
        _FakeProc(13, []),
    ]
    monkeypatch.setattr(mod.psutil, "process_iter", lambda attrs=None: iter(fake))
    monkeypatch.setattr(mod.os, "getpid", lambda: 99)

    out = mod.find_active_processes()

    assert [p["pid"] for p in out] == [10, 11]


def test_main_rejects_too_fast_interval():
    mod = _load_mod()
    assert mod.main(["--interval-seconds", "1"]) == 2
