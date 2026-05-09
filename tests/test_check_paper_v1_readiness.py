"""Hermetic tests for ``scripts/check_paper_v1_readiness.py``.

The checker is intentionally read-only: tests build tmp_path parquet fixtures
and never touch the real cache or network.
"""
from __future__ import annotations

import importlib
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _import_module():
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    if "check_paper_v1_readiness" in sys.modules:
        return importlib.reload(sys.modules["check_paper_v1_readiness"])
    return importlib.import_module("check_paper_v1_readiness")


@pytest.fixture
def mod():
    return _import_module()


def _write_earnings_dates(
    cache_dir: Path,
    year: str,
    ticker: str,
    dates: list[date],
) -> None:
    target_dir = cache_dir / year / "earnings_dates"
    target_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"earnings_date": dates}).to_parquet(
        target_dir / f"{ticker}.parquet",
        index=False,
    )


def _priced_df(
    ticker: str,
    event_date: date,
    *,
    mid: float = 1.25,
    underlying_close: float = 100.0,
    atm_iv: float = 0.30,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": [
                f"O:{ticker}{event_date.strftime('%y%m%d')}C00100000",
                f"O:{ticker}{event_date.strftime('%y%m%d')}P00100000",
            ],
            "underlying_ticker": [ticker, ticker],
            "contract_type": ["call", "put"],
            "strike_price": [100.0, 100.0],
            "expiration_date": [
                (pd.Timestamp(event_date) + pd.Timedelta(days=9)).date().isoformat(),
                (pd.Timestamp(event_date) + pd.Timedelta(days=9)).date().isoformat(),
            ],
            "as_of_date": [event_date.isoformat(), event_date.isoformat()],
            "timing": ["entry", "entry"],
            "event_date": [event_date.isoformat(), event_date.isoformat()],
            "bid": [mid - 0.05, mid - 0.05],
            "ask": [mid + 0.05, mid + 0.05],
            "mid": [mid, mid],
            "underlying_close": [underlying_close, underlying_close],
            "is_atm": [True, True],
            "atm_iv": [atm_iv, atm_iv],
        }
    )


def _write_priced(
    cache_dir: Path,
    year: str,
    ticker: str,
    event_date: date,
    *,
    df: pd.DataFrame | None = None,
    mid: float = 1.25,
    underlying_close: float = 100.0,
    atm_iv: float = 0.30,
) -> Path:
    target_dir = cache_dir / year / "chains_priced" / ticker
    target_dir.mkdir(parents=True, exist_ok=True)
    out = target_dir / f"{event_date.isoformat()}.parquet"
    if df is None:
        df = _priced_df(
            ticker,
            event_date,
            mid=mid,
            underlying_close=underlying_close,
            atm_iv=atm_iv,
        )
    df.to_parquet(out, index=False)
    return out


def _write_ready_satellite_cache(cache_dir: Path) -> None:
    events = {
        "year1": date(2024, 8, 1),
        "year2": date(2025, 8, 1),
    }
    for year, event_date in events.items():
        _write_earnings_dates(cache_dir, year, "AAPL", [event_date])
        _write_priced(cache_dir, year, "AAPL", event_date)


def test_check_readiness_ready_with_satellite_priced_and_spy_warning(mod, tmp_path):
    _write_ready_satellite_cache(tmp_path)

    summary = mod.check_readiness(tmp_path, universe=("AAPL",))

    assert summary["ready"] is True
    assert summary["blockers"] == []
    assert {w["code"] for w in summary["warnings"]} == {"no_spy_priced_core"}
    assert summary["years"]["year1"]["satellite_priced_file_count"] == 1
    assert summary["years"]["year2"]["satellite_priced_file_count"] == 1

    text = mod.format_text_summary(summary)
    assert "paper_v1 readiness: READY with warnings" in text
    assert "year1: satellite priced 1/1" in text


def test_check_readiness_reports_missing_priced_file(mod, tmp_path):
    _write_ready_satellite_cache(tmp_path)
    missing_event = date(2025, 11, 3)
    _write_earnings_dates(
        tmp_path,
        "year2",
        "AAPL",
        [date(2025, 8, 1), missing_event],
    )

    summary = mod.check_readiness(tmp_path, universe=("AAPL",))

    assert summary["ready"] is False
    blockers = summary["blockers"]
    assert any(
        b["code"] == "missing_priced_file"
        and b["ticker"] == "AAPL"
        and b["event_date"] == missing_event.isoformat()
        for b in blockers
    )
    assert summary["years"]["year2"]["satellite_priced_file_count"] == 1
    assert summary["years"]["year2"]["satellite_missing_priced_count"] == 1


def test_check_readiness_validates_required_columns(mod, tmp_path):
    _write_ready_satellite_cache(tmp_path)
    event = date(2025, 8, 1)
    bad = _priced_df("AAPL", event).drop(columns=["ask"])
    _write_priced(tmp_path, "year2", "AAPL", event, df=bad)

    summary = mod.check_readiness(tmp_path, universe=("AAPL",))

    assert summary["ready"] is False
    assert any(b["code"] == "missing_priced_columns" for b in summary["blockers"])


def test_check_readiness_requires_finite_mid_and_underlying_close(mod, tmp_path):
    _write_ready_satellite_cache(tmp_path)
    event = date(2024, 8, 1)
    _write_priced(
        tmp_path,
        "year1",
        "AAPL",
        event,
        mid=float("nan"),
        underlying_close=float("nan"),
    )

    summary = mod.check_readiness(tmp_path, universe=("AAPL",))
    codes = {b["code"] for b in summary["blockers"]}

    assert "no_finite_mid" in codes
    assert "no_finite_underlying_close" in codes


def test_check_readiness_requires_finite_atm_iv_for_runner_compat(mod, tmp_path):
    _write_ready_satellite_cache(tmp_path)
    event = date(2024, 8, 1)
    _write_priced(tmp_path, "year1", "AAPL", event, atm_iv=float("nan"))

    summary = mod.check_readiness(tmp_path, universe=("AAPL",))

    assert summary["ready"] is False
    assert any(b["code"] == "no_finite_atm_iv" for b in summary["blockers"])


def test_spy_core_detection_warns_when_partial(mod, tmp_path):
    _write_ready_satellite_cache(tmp_path)
    spy_entry = date(2025, 5, 12)
    _write_priced(tmp_path, "year2", "SPY", spy_entry)

    summary = mod.check_readiness(tmp_path, universe=("AAPL",))
    y2_spy = summary["years"]["year2"]["spy_core"]

    assert y2_spy["present"] is True
    assert y2_spy["priced_files"] == 1
    assert y2_spy["expected_monday_files"] > 1
    assert any(w["code"] == "partial_spy_priced_core" for w in summary["warnings"])
    assert summary["ready"] is True


def test_require_spy_core_promotes_missing_spy_to_blocker(mod, tmp_path):
    _write_ready_satellite_cache(tmp_path)

    summary = mod.check_readiness(
        tmp_path,
        universe=("AAPL",),
        require_spy_core=True,
    )

    assert summary["ready"] is False
    assert any(b["code"] == "no_spy_priced_core" for b in summary["blockers"])


def test_main_json_output_and_exit_code(mod, tmp_path, capsys):
    _write_ready_satellite_cache(tmp_path)

    rc = mod.main(["--cache-dir", str(tmp_path), "--universe", "AAPL", "--json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 0
    assert payload["ready"] is True
    assert payload["years"]["year1"]["satellite_event_count"] == 1
    assert captured.err == ""


def test_main_returns_nonzero_for_blockers(mod, tmp_path, capsys):
    _write_ready_satellite_cache(tmp_path)
    event = date(2024, 8, 1)
    _write_priced(tmp_path, "year1", "AAPL", event, mid=np.nan)

    rc = mod.main(["--cache-dir", str(tmp_path), "--universe", "AAPL"])
    captured = capsys.readouterr()

    assert rc == 1
    assert "paper_v1 readiness: BLOCKED" in captured.out
