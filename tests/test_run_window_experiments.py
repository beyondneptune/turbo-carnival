from __future__ import annotations

from pathlib import Path

from experiments import run_window_experiments as mod
from scripts import run_paper_v1_evidence as evidence


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


def test_trade_stats_aggregates_core_and_satellite(tmp_path: Path) -> None:
    conn = evidence.open_db(tmp_path / "experiment.db")
    try:
        evidence.insert_run_row(conn, "r")
        for sleeve, pnl in (("core", 150.0), ("core", -25.0), ("satellite", 75.0)):
            strategy_type = "iron_condor" if sleeve == "core" else "long_straddle"
            decision_id = evidence.insert_gate_decision(
                conn,
                run_id="r",
                decision_at_utc="2026-01-01T14:30:00+00:00",
                underlying="SPY",
                sleeve=sleeve,
                candidate_summary="ENTRY:test",
                risk_pass=True,
                block_reason=None,
                final_decision="approved",
                model_version="test",
            )
            conn.execute(
                """
                INSERT INTO trades (
                    run_id, decision_id, sleeve, strategy_type, underlying_ticker,
                    status, entry_at_utc, realized_pnl_usd
                ) VALUES ('r', ?, ?, ?, 'SPY', 'closed', ?, ?)
                """,
                (
                    decision_id,
                    sleeve,
                    strategy_type,
                    "2026-01-01T14:30:00+00:00",
                    pnl,
                ),
            )
        stats = mod.trade_stats(conn, "r")
    finally:
        conn.close()

    assert stats["trade_count"] == 3
    assert stats["core_trade_count"] == 2
    assert stats["satellite_trade_count"] == 1
    assert stats["realized_pnl_usd"] == 200.0
    assert stats["core_realized_pnl_usd"] == 125.0
    assert stats["satellite_realized_pnl_usd"] == 75.0
    assert stats["return_on_10k"] == 0.02
    assert stats["win_rate"] == 2 / 3
