"""Tests for `scripts/run_phase0_demo.py` CLI behavior.

Covers:
- auto-generated run_id pattern
- explicit --run-id
- append-only refusal on duplicate run_id
- --reset-demo override
- --help and invalid-arg handling

Approach: import the module and monkeypatch the module-level `DB_PATH` constant
to a tmp_path SQLite file, then call `main([...])` directly. This is hermetic
(each test gets its own DB file), avoids subprocess overhead, and gives direct
access to argparse's SystemExit for --help / invalid-arg paths.

Subprocess + cwd was rejected because the script's DB_PATH is computed from
`Path(__file__).resolve().parent.parent`, not the cwd, so cwd manipulation has
no effect; only patching the module constant cleanly isolates the DB.
"""

from __future__ import annotations

import importlib
import re
import sqlite3
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _import_demo_module():
    """Import scripts/run_phase0_demo.py as a module, fresh each time."""
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    if "run_phase0_demo" in sys.modules:
        return importlib.reload(sys.modules["run_phase0_demo"])
    return importlib.import_module("run_phase0_demo")


@pytest.fixture
def demo_module(tmp_path, monkeypatch):
    """Import the demo script module with DB_PATH redirected to a tmp file."""
    mod = _import_demo_module()
    test_db = tmp_path / "test.db"
    monkeypatch.setattr(mod, "DB_PATH", test_db)
    return mod


def _count_runs(db_path: Path, run_id: str | None = None) -> int:
    conn = sqlite3.connect(db_path)
    try:
        if run_id is None:
            cur = conn.execute("SELECT COUNT(*) FROM runs")
        else:
            cur = conn.execute("SELECT COUNT(*) FROM runs WHERE run_id = ?", (run_id,))
        return int(cur.fetchone()[0])
    finally:
        conn.close()


def test_auto_generated_run_id_fresh_db(demo_module, capsys):
    """No --run-id arg: script generates a run_id matching the timestamp pattern."""
    rc = demo_module.main([])
    assert rc == 0

    out = capsys.readouterr().out
    # First line: "Phase 0 demo seed complete (run_id=...)"
    m = re.search(r"run_id=(\S+)\)", out)
    assert m is not None, f"could not find run_id in stdout: {out!r}"
    run_id = m.group(1)
    assert re.fullmatch(r"paper_v0_demo_\d{8}T\d{6}Z", run_id), (
        f"auto-generated run_id {run_id!r} does not match expected pattern"
    )

    db_path = demo_module.DB_PATH
    assert _count_runs(db_path) == 1
    assert _count_runs(db_path, run_id) == 1


def test_explicit_run_id_fresh_db(demo_module):
    """--run-id test_xyz: that exact run_id ends up in the runs table."""
    rc = demo_module.main(["--run-id", "test_xyz"])
    assert rc == 0

    db_path = demo_module.DB_PATH
    assert _count_runs(db_path, "test_xyz") == 1
    assert _count_runs(db_path) == 1


def test_duplicate_run_id_without_reset_refused(demo_module, capsys):
    """Second invocation with the same --run-id and no --reset-demo must fail."""
    rc1 = demo_module.main(["--run-id", "test_xyz"])
    assert rc1 == 0
    capsys.readouterr()  # drain first run's stdout

    rc2 = demo_module.main(["--run-id", "test_xyz"])
    assert rc2 == 1, "duplicate run without --reset-demo should exit non-zero"

    captured = capsys.readouterr()
    assert "already exists" in captured.err, (
        f"expected 'already exists' in stderr, got: {captured.err!r}"
    )

    # runs table still has exactly one row for that run_id (append-only safeguard)
    db_path = demo_module.DB_PATH
    assert _count_runs(db_path, "test_xyz") == 1
    assert _count_runs(db_path) == 1


def test_reset_demo_overwrites_existing(demo_module, capsys):
    """Same setup as duplicate test, but second invocation passes --reset-demo."""
    rc1 = demo_module.main(["--run-id", "test_xyz"])
    assert rc1 == 0
    capsys.readouterr()

    rc2 = demo_module.main(["--run-id", "test_xyz", "--reset-demo"])
    assert rc2 == 0, "with --reset-demo the second run should succeed"

    db_path = demo_module.DB_PATH
    # Cleaned + re-inserted -> still exactly one row
    assert _count_runs(db_path, "test_xyz") == 1
    assert _count_runs(db_path) == 1


def test_help_exits_zero_and_lists_flags(demo_module, capsys):
    """--help: argparse calls sys.exit(0); stdout must mention both flags."""
    with pytest.raises(SystemExit) as exc_info:
        demo_module.main(["--help"])
    # argparse's --help exits 0
    assert exc_info.value.code == 0

    out = capsys.readouterr().out
    assert "--run-id" in out, f"--run-id missing from --help output: {out!r}"
    assert "--reset-demo" in out, f"--reset-demo missing from --help output: {out!r}"


def test_invalid_arg_fails(demo_module, capsys):
    """Unknown flag: argparse exits non-zero (typically 2)."""
    with pytest.raises(SystemExit) as exc_info:
        demo_module.main(["--unknown-flag"])
    code = exc_info.value.code
    # argparse uses 2 for usage errors; just assert non-zero
    assert code is not None and code != 0, (
        f"unknown flag should produce non-zero exit, got {code!r}"
    )
