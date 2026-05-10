from __future__ import annotations

from pathlib import Path

from experiments import run_window_experiments as mod


def test_reset_window_scratch_removes_rerun_artifacts(tmp_path: Path) -> None:
    win_dir = tmp_path / "exp001"
    spy_dir = win_dir / "spy_priced"
    spy_dir.mkdir(parents=True)
    for name in mod.WINDOW_SCRATCH_FILES:
        (win_dir / name).write_text("old")
    (spy_dir / "2026-01-05.parquet").write_text("old")
    keep = win_dir / "notes.txt"
    keep.write_text("preserve")

    mod.reset_window_scratch(win_dir)

    assert win_dir.exists()
    assert keep.read_text() == "preserve"
    for name in mod.WINDOW_SCRATCH_FILES:
        assert not (win_dir / name).exists()
    assert not spy_dir.exists()
