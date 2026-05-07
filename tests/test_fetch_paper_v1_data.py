"""Hermetic tests for ``scripts/fetch_paper_v1_data.py``.

All Polygon and yfinance calls are mocked; no test exercises the real network.
The script is imported as a module via the same scripts-on-sys.path trick used
by ``tests/test_run_daily_paper.py`` so the `from ingest...` imports resolve.
"""
from __future__ import annotations

import importlib
import json
import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _import_module():
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    if "fetch_paper_v1_data" in sys.modules:
        return importlib.reload(sys.modules["fetch_paper_v1_data"])
    return importlib.import_module("fetch_paper_v1_data")


@pytest.fixture
def mod(monkeypatch, tmp_path):
    """Module under test, with the default status file redirected to tmp_path."""
    m = _import_module()
    monkeypatch.setattr(m, "STATUS_FILE", tmp_path / "status.json")
    monkeypatch.setattr(m, "DEFAULT_CACHE_DIR", tmp_path / "cache")
    return m


# ---------------------------------------------------------------------------
# cache_path helpers
# ---------------------------------------------------------------------------


def test_cache_path_year1_layout(tmp_path, mod):
    p = mod.cache_path_year1("spy_daily.parquet", base_dir=tmp_path)
    assert p == tmp_path / "year1" / "spy_daily.parquet"


def test_cache_path_year2_layout(tmp_path, mod):
    p = mod.cache_path_year2("chains/AAPL/2024-08-01.parquet", base_dir=tmp_path)
    assert p == tmp_path / "year2" / "chains" / "AAPL" / "2024-08-01.parquet"


def test_cache_paths_do_not_create_dirs(tmp_path, mod):
    """Path helpers are pure: they must not mkdir as a side effect."""
    mod.cache_path_year1("foo.parquet", base_dir=tmp_path)
    mod.cache_path_year2("bar.parquet", base_dir=tmp_path)
    assert not (tmp_path / "year1").exists()
    assert not (tmp_path / "year2").exists()


# ---------------------------------------------------------------------------
# write_status
# ---------------------------------------------------------------------------


def test_write_status_creates_json(tmp_path, mod):
    sf = tmp_path / "status.json"
    doc = mod.write_status("stage_a", "running", status_file=sf, foo=1)
    assert sf.exists()
    on_disk = json.loads(sf.read_text())
    assert on_disk == doc
    assert on_disk["stages"]["stage_a"]["status"] == "running"
    assert on_disk["stages"]["stage_a"]["foo"] == 1
    assert on_disk["phase"] == "running"
    assert "started_at_utc" in on_disk
    assert "last_update_utc" in on_disk


def test_write_status_atomic_no_tmp_left_behind(tmp_path, mod):
    sf = tmp_path / "status.json"
    mod.write_status("stage_a", "running", status_file=sf)
    # The atomic-write helper renames the tmp file; nothing should remain.
    leftover = list(tmp_path.glob("status.json.tmp"))
    assert leftover == []
    # And the final file is valid JSON.
    json.loads(sf.read_text())


def test_write_status_merges_existing_stages(tmp_path, mod):
    sf = tmp_path / "status.json"
    mod.write_status("stage_a", "running", status_file=sf)
    mod.write_status("stage_b", "complete", status_file=sf, rows=42)
    on_disk = json.loads(sf.read_text())
    assert set(on_disk["stages"].keys()) == {"stage_a", "stage_b"}
    assert on_disk["stages"]["stage_a"]["status"] == "running"
    assert on_disk["stages"]["stage_b"]["status"] == "complete"
    assert on_disk["stages"]["stage_b"]["rows"] == 42


def test_write_status_overwrites_same_stage(tmp_path, mod):
    sf = tmp_path / "status.json"
    mod.write_status("stage_a", "running", status_file=sf, done=0)
    mod.write_status("stage_a", "complete", status_file=sf, done=10, total=10)
    on_disk = json.loads(sf.read_text())
    assert on_disk["stages"]["stage_a"]["status"] == "complete"
    assert on_disk["stages"]["stage_a"]["done"] == 10
    assert on_disk["stages"]["stage_a"]["total"] == 10


def test_write_status_phase_override(tmp_path, mod):
    sf = tmp_path / "status.json"
    mod.write_status("init", "running", status_file=sf)
    assert json.loads(sf.read_text())["phase"] == "running"
    mod.write_status("complete", "complete", status_file=sf, phase="cache_complete")
    assert json.loads(sf.read_text())["phase"] == "cache_complete"


def test_write_status_recovers_from_corrupt_file(tmp_path, mod):
    sf = tmp_path / "status.json"
    sf.write_text("{not json")
    # Should not raise — start fresh.
    doc = mod.write_status("stage_a", "running", status_file=sf)
    assert doc["stages"]["stage_a"]["status"] == "running"


# ---------------------------------------------------------------------------
# fetch_spy_daily
# ---------------------------------------------------------------------------


def _fake_aggregates_df() -> pd.DataFrame:
    idx = pd.DatetimeIndex(
        ["2024-05-07", "2024-05-08", "2024-05-09"], name="date"
    )
    df = pd.DataFrame(
        {
            "open": [500.0, 501.0, 502.0],
            "high": [502.0, 503.0, 504.0],
            "low": [499.0, 500.0, 501.0],
            "close": [501.0, 502.0, 503.0],
            "volume": [1000, 1100, 1200],
            "vwap": [500.5, 501.5, 502.5],
            "transactions": [10, 11, 12],
        },
        index=idx,
    )
    return df


def test_fetch_spy_daily_calls_api_when_no_cache(tmp_path, mod):
    target = tmp_path / "spy_daily.parquet"
    client = MagicMock()
    client.fetch_aggregates.return_value = _fake_aggregates_df()
    df = mod.fetch_spy_daily(client, date(2024, 5, 7), date(2025, 5, 6), target)
    assert client.fetch_aggregates.called
    assert target.exists()
    assert len(df) == 3


def test_fetch_spy_daily_skips_api_when_cache_exists(tmp_path, mod):
    target = tmp_path / "spy_daily.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    _fake_aggregates_df().to_parquet(target)
    client = MagicMock()
    client.fetch_aggregates.side_effect = AssertionError("must not call API")
    df = mod.fetch_spy_daily(client, date(2024, 5, 7), date(2025, 5, 6), target)
    assert not client.fetch_aggregates.called
    assert len(df) == 3


def test_fetch_spy_daily_writes_atomically_no_tmp_left(tmp_path, mod):
    target = tmp_path / "spy_daily.parquet"
    client = MagicMock()
    client.fetch_aggregates.return_value = _fake_aggregates_df()
    mod.fetch_spy_daily(client, date(2024, 5, 7), date(2025, 5, 6), target)
    assert not list(tmp_path.glob("*.parquet.tmp"))


# ---------------------------------------------------------------------------
# fetch_chain_at_event
# ---------------------------------------------------------------------------


def _fake_contracts_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "O:AAPL240801C00200000",
                "underlying_ticker": "AAPL",
                "contract_type": "call",
                "strike_price": 200.0,
                "expiration_date": "2024-08-09",
                "exercise_style": "american",
                "shares_per_contract": 100,
            },
            {
                "ticker": "O:AAPL240801P00200000",
                "underlying_ticker": "AAPL",
                "contract_type": "put",
                "strike_price": 200.0,
                "expiration_date": "2024-08-09",
                "exercise_style": "american",
                "shares_per_contract": 100,
            },
        ]
    )


def test_fetch_chain_at_event_writes_file(tmp_path, mod):
    client = MagicMock()
    client.fetch_options_contracts.return_value = _fake_contracts_df()
    chains_dir = tmp_path / "chains"
    df = mod.fetch_chain_at_event(client, "AAPL", date(2024, 8, 1), chains_dir)
    assert df is not None and not df.empty
    out = chains_dir / "AAPL" / "2024-08-01.parquet"
    assert out.exists()
    on_disk = pd.read_parquet(out)
    # Stamped metadata columns are present.
    assert "as_of_date" in on_disk.columns
    assert "event_date" in on_disk.columns
    assert on_disk["event_date"].iloc[0] == "2024-08-01"


def test_fetch_chain_at_event_skips_when_cache_exists(tmp_path, mod):
    chains_dir = tmp_path / "chains"
    out_dir = chains_dir / "AAPL"
    out_dir.mkdir(parents=True, exist_ok=True)
    cached = _fake_contracts_df()
    cached["as_of_date"] = "2024-07-31"
    cached["event_date"] = "2024-08-01"
    cached.to_parquet(out_dir / "2024-08-01.parquet")
    client = MagicMock()
    client.fetch_options_contracts.side_effect = AssertionError("must not call API")
    df = mod.fetch_chain_at_event(client, "AAPL", date(2024, 8, 1), chains_dir)
    assert df is not None
    assert not client.fetch_options_contracts.called


def test_fetch_chain_at_event_returns_none_on_empty_chain(tmp_path, mod):
    """Polygon may return no contracts for an illiquid name on a particular date."""
    client = MagicMock()
    client.fetch_options_contracts.return_value = pd.DataFrame()
    chains_dir = tmp_path / "chains"
    df = mod.fetch_chain_at_event(client, "ZZZZ", date(2024, 8, 1), chains_dir)
    assert df is None
    # Importantly, no file is written for an empty result.
    assert not (chains_dir / "ZZZZ" / "2024-08-01.parquet").exists()


def test_fetch_chain_at_event_returns_none_on_polygon_error(tmp_path, mod):
    from ingest.polygon_free import PolygonError

    client = MagicMock()
    client.fetch_options_contracts.side_effect = PolygonError("HTTP 500")
    chains_dir = tmp_path / "chains"
    df = mod.fetch_chain_at_event(client, "AAPL", date(2024, 8, 1), chains_dir)
    assert df is None
    assert not (chains_dir / "AAPL" / "2024-08-01.parquet").exists()


def test_fetch_chain_at_event_uses_t_minus_1_as_of(tmp_path, mod):
    """T-1 should land on the prior weekday (or earlier if it was a weekend)."""
    client = MagicMock()
    client.fetch_options_contracts.return_value = _fake_contracts_df()
    chains_dir = tmp_path / "chains"
    # Aug 5 2024 is a Monday → T-1 = Sunday 8/4 → trading-day-on-or-before = Friday 8/2.
    mod.fetch_chain_at_event(client, "AAPL", date(2024, 8, 5), chains_dir)
    args, kwargs = client.fetch_options_contracts.call_args
    assert kwargs["as_of"] == "2024-08-02"


# ---------------------------------------------------------------------------
# fetch_spy_iv_daily
# ---------------------------------------------------------------------------


def test_fetch_spy_iv_daily_full_range_no_cache(tmp_path, mod):
    target = tmp_path / "iv.parquet"
    client = MagicMock()
    client.fetch_options_contracts.return_value = _fake_contracts_df()
    df = mod.fetch_spy_iv_daily(client, date(2024, 5, 7), date(2024, 5, 13), target)
    # 5/7 (Tue) -> 5/13 (Mon) inclusive = Tue, Wed, Thu, Fri, Mon = 5 weekdays.
    assert len(df) == 5
    assert target.exists()


def test_fetch_spy_iv_daily_resumes_from_max_existing(tmp_path, mod):
    target = tmp_path / "iv.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    # Pretend we already have rows up through 2024-05-09.
    existing = pd.DataFrame(
        [
            {"date": "2024-05-07", "atm_iv": 0.12, "available": True},
            {"date": "2024-05-08", "atm_iv": 0.13, "available": True},
            {"date": "2024-05-09", "atm_iv": 0.14, "available": True},
        ]
    )
    existing.to_parquet(target, index=False)
    client = MagicMock()
    client.fetch_options_contracts.return_value = _fake_contracts_df()

    df = mod.fetch_spy_iv_daily(client, date(2024, 5, 7), date(2024, 5, 13), target)
    # Existing 3 rows + Friday 5/10 + Monday 5/13 = 5 rows total.
    assert len(df) == 5
    # Only the two new days should have triggered API calls.
    assert client.fetch_options_contracts.call_count == 2


def test_fetch_spy_iv_daily_no_op_when_cache_already_complete(tmp_path, mod):
    target = tmp_path / "iv.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.DataFrame(
        [
            {"date": "2024-05-07", "atm_iv": 0.12, "available": True},
            {"date": "2024-05-08", "atm_iv": 0.13, "available": True},
        ]
    )
    existing.to_parquet(target, index=False)
    client = MagicMock()
    client.fetch_options_contracts.side_effect = AssertionError("must not call API")
    df = mod.fetch_spy_iv_daily(client, date(2024, 5, 7), date(2024, 5, 8), target)
    assert len(df) == 2
    assert client.fetch_options_contracts.call_count == 0


def test_fetch_spy_iv_daily_records_unavailable_on_polygon_error(tmp_path, mod):
    from ingest.polygon_free import PolygonError

    target = tmp_path / "iv.parquet"
    client = MagicMock()
    client.fetch_options_contracts.side_effect = PolygonError("HTTP 500")
    df = mod.fetch_spy_iv_daily(client, date(2024, 5, 7), date(2024, 5, 7), target)
    assert len(df) == 1
    assert bool(df.iloc[0]["available"]) is False


# ---------------------------------------------------------------------------
# fetch_earnings_dates_for_universe
# ---------------------------------------------------------------------------


def test_fetch_earnings_dates_for_universe_uses_cache(tmp_path, mod, monkeypatch):
    """fetch_calendar's per-ticker cache means re-running is offline."""
    from ingest import earnings_calendar as ec

    monkeypatch.setattr(
        ec,
        "fetch_ticker_earnings",
        lambda ticker, start=None, end=None: [date(2024, 8, 1), date(2024, 11, 1)],
    )
    out = mod.fetch_earnings_dates_for_universe(
        ["AAPL"], date(2024, 5, 7), date(2025, 5, 6), tmp_path / "ed"
    )
    assert out["AAPL"] == [date(2024, 8, 1), date(2024, 11, 1)]
    cached_path = tmp_path / "ed" / "AAPL.parquet"
    assert cached_path.exists()
    # Second call must not invoke the network.
    monkeypatch.setattr(
        ec, "fetch_ticker_earnings", lambda *a, **k: pytest.fail("should be cached")
    )
    out2 = mod.fetch_earnings_dates_for_universe(
        ["AAPL"], date(2024, 5, 7), date(2025, 5, 6), tmp_path / "ed"
    )
    assert out2 == out


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


def test_parse_args_defaults(mod):
    ns = mod.parse_args([])
    assert ns.year == "both"
    assert ns.skip_iv is False
    assert ns.skip_chains is False
    assert ns.max_events_per_ticker is None
    assert isinstance(ns.cache_dir, Path)


def test_parse_args_year_choice(mod):
    ns = mod.parse_args(["--year", "year2"])
    assert ns.year == "year2"


def test_parse_args_skip_flags(mod):
    ns = mod.parse_args(["--skip-iv", "--skip-chains"])
    assert ns.skip_iv is True
    assert ns.skip_chains is True


def test_parse_args_cap(mod):
    ns = mod.parse_args(["--max-events-per-ticker", "2"])
    assert ns.max_events_per_ticker == 2


def test_parse_args_invalid_year(mod):
    with pytest.raises(SystemExit):
        mod.parse_args(["--year", "year3"])


def test_parse_args_cache_dir_override(tmp_path, mod):
    ns = mod.parse_args(["--cache-dir", str(tmp_path)])
    assert ns.cache_dir == tmp_path


# ---------------------------------------------------------------------------
# main integration smoke
# ---------------------------------------------------------------------------


def test_main_smoke_runs_year1_and_year2(tmp_path, mod, monkeypatch):
    """End-to-end: with all clients mocked, year1 + year2 finishes in <1s."""
    # Tighten the windows to a handful of weekdays so the IV loop is short.
    monkeypatch.setattr(mod, "YEAR1_START", date(2024, 5, 7))
    monkeypatch.setattr(mod, "YEAR1_END", date(2024, 5, 9))
    monkeypatch.setattr(mod, "YEAR2_START", date(2025, 5, 7))
    monkeypatch.setattr(mod, "YEAR2_END", date(2025, 5, 9))
    # Tiny universe so chain calls are bounded.
    monkeypatch.setattr(mod, "PAPER_V1_SATELLITE", ("AAPL",))

    # Stub PolygonFreeClient.
    fake_client = MagicMock()
    fake_client.fetch_aggregates.return_value = _fake_aggregates_df()
    fake_client.fetch_options_contracts.return_value = _fake_contracts_df()
    monkeypatch.setattr(mod, "PolygonFreeClient", lambda *a, **k: fake_client)

    # Stub yfinance via fetch_ticker_earnings (used internally by fetch_calendar).
    from ingest import earnings_calendar as ec
    monkeypatch.setattr(
        ec, "fetch_ticker_earnings",
        lambda ticker, start=None, end=None: [date(2024, 5, 8)] if start and start.year == 2024 else [date(2025, 5, 8)],
    )

    cache_dir = tmp_path / "cache"
    status = tmp_path / "status.json"
    rc = mod.main([
        "--cache-dir", str(cache_dir),
        "--status-file", str(status),
        "--max-events-per-ticker", "1",
    ])
    assert rc == 0

    # Both year directories populated.
    assert (cache_dir / "year1" / "spy_daily.parquet").exists()
    assert (cache_dir / "year2" / "spy_daily.parquet").exists()
    assert (cache_dir / "year1" / "earnings_dates" / "AAPL.parquet").exists()
    assert (cache_dir / "year2" / "earnings_dates" / "AAPL.parquet").exists()
    # Chains dir created and contains at least one event file.
    chains1 = list((cache_dir / "year1" / "chains" / "AAPL").glob("*.parquet"))
    assert len(chains1) >= 1
    # IV daily exists for both years.
    assert (cache_dir / "year1" / "spy_iv_daily.parquet").exists()
    assert (cache_dir / "year2" / "spy_iv_daily.parquet").exists()
    # Manifest enumerates every parquet.
    assert (cache_dir / "manifest.json").exists()
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    assert manifest["count"] >= 4
    # Final status is cache_complete with year1+year2 stages all "complete".
    doc = json.loads(status.read_text())
    assert doc["phase"] == "cache_complete"
    assert doc["stages"]["fetch_year1_spy_daily"]["status"] == "complete"
    assert doc["stages"]["fetch_year2_spy_daily"]["status"] == "complete"
    assert doc["stages"]["fetch_year1_earnings"]["status"] == "complete"
    assert doc["stages"]["fetch_year2_earnings"]["status"] == "complete"
    assert doc["stages"]["fetch_year1_chains"]["status"] == "complete"
    assert doc["stages"]["fetch_year2_chains"]["status"] == "complete"
    assert doc["stages"]["fetch_year1_iv_daily"]["status"] == "complete"
    assert doc["stages"]["fetch_year2_iv_daily"]["status"] == "complete"


def test_main_year_filter_single(tmp_path, mod, monkeypatch):
    """``--year year1`` fetches only year1; year2 stays empty."""
    monkeypatch.setattr(mod, "YEAR1_START", date(2024, 5, 7))
    monkeypatch.setattr(mod, "YEAR1_END", date(2024, 5, 8))
    monkeypatch.setattr(mod, "YEAR2_START", date(2025, 5, 7))
    monkeypatch.setattr(mod, "YEAR2_END", date(2025, 5, 8))
    monkeypatch.setattr(mod, "PAPER_V1_SATELLITE", ("AAPL",))

    fake_client = MagicMock()
    fake_client.fetch_aggregates.return_value = _fake_aggregates_df()
    fake_client.fetch_options_contracts.return_value = _fake_contracts_df()
    monkeypatch.setattr(mod, "PolygonFreeClient", lambda *a, **k: fake_client)

    from ingest import earnings_calendar as ec
    monkeypatch.setattr(ec, "fetch_ticker_earnings", lambda *a, **k: [date(2024, 5, 8)])

    cache_dir = tmp_path / "cache"
    status = tmp_path / "status.json"
    rc = mod.main([
        "--year", "year1",
        "--cache-dir", str(cache_dir),
        "--status-file", str(status),
        "--max-events-per-ticker", "1",
    ])
    assert rc == 0
    assert (cache_dir / "year1" / "spy_daily.parquet").exists()
    assert not (cache_dir / "year2").exists()


def test_main_skip_iv_and_chains(tmp_path, mod, monkeypatch):
    monkeypatch.setattr(mod, "YEAR1_START", date(2024, 5, 7))
    monkeypatch.setattr(mod, "YEAR1_END", date(2024, 5, 8))
    monkeypatch.setattr(mod, "YEAR2_START", date(2025, 5, 7))
    monkeypatch.setattr(mod, "YEAR2_END", date(2025, 5, 8))
    monkeypatch.setattr(mod, "PAPER_V1_SATELLITE", ("AAPL",))

    fake_client = MagicMock()
    fake_client.fetch_aggregates.return_value = _fake_aggregates_df()
    fake_client.fetch_options_contracts.side_effect = AssertionError(
        "should not be called: --skip-iv and --skip-chains in effect"
    )
    monkeypatch.setattr(mod, "PolygonFreeClient", lambda *a, **k: fake_client)

    from ingest import earnings_calendar as ec
    monkeypatch.setattr(ec, "fetch_ticker_earnings", lambda *a, **k: [date(2024, 5, 8)])

    cache_dir = tmp_path / "cache"
    status = tmp_path / "status.json"
    rc = mod.main([
        "--year", "year1",
        "--cache-dir", str(cache_dir),
        "--status-file", str(status),
        "--skip-iv",
        "--skip-chains",
    ])
    assert rc == 0
    doc = json.loads(status.read_text())
    assert doc["stages"]["fetch_year1_chains"]["status"] == "skipped"
    assert doc["stages"]["fetch_year1_iv_daily"]["status"] == "skipped"
    assert not (cache_dir / "year1" / "chains").exists()
    assert not (cache_dir / "year1" / "spy_iv_daily.parquet").exists()


def test_main_propagates_polygon_auth_error(tmp_path, mod, monkeypatch):
    from ingest.polygon_free import PolygonAuthError

    def boom(*_a, **_k):
        raise PolygonAuthError("no key")

    monkeypatch.setattr(mod, "PolygonFreeClient", boom)
    rc = mod.main([
        "--year", "year1",
        "--cache-dir", str(tmp_path / "cache"),
        "--status-file", str(tmp_path / "status.json"),
    ])
    assert rc == 2
    doc = json.loads((tmp_path / "status.json").read_text())
    assert doc["phase"] == "error"
