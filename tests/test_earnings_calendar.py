from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from ingest import earnings_calendar as ec
from ingest.earnings_calendar import (
    cache_path,
    fetch_calendar,
    fetch_ticker_earnings,
    load_cached,
    save_to_cache,
)


# --- cache layout ----------------------------------------------------------


def test_cache_path_layout(tmp_path: Path):
    assert cache_path("AAPL", tmp_path) == tmp_path / "AAPL.parquet"
    assert cache_path("MSFT", tmp_path) == tmp_path / "MSFT.parquet"


def test_save_then_load_round_trip_preserves_dates(tmp_path: Path):
    dates = [date(2024, 2, 1), date(2024, 5, 2), date(2024, 8, 1), date(2024, 11, 1)]
    save_to_cache("AAPL", dates, tmp_path)
    loaded = load_cached("AAPL", tmp_path)
    assert loaded is not None
    assert loaded == dates  # sorted order, exact equality


def test_save_to_cache_creates_directory(tmp_path: Path):
    nested = tmp_path / "subdir" / "earnings"
    assert not nested.exists()
    save_to_cache("AAPL", [date(2024, 5, 2)], nested)
    assert (nested / "AAPL.parquet").exists()


def test_save_to_cache_dedupes_and_sorts(tmp_path: Path):
    raw = [date(2024, 5, 2), date(2024, 2, 1), date(2024, 5, 2), date(2024, 11, 1)]
    save_to_cache("AAPL", raw, tmp_path)
    loaded = load_cached("AAPL", tmp_path)
    assert loaded == [date(2024, 2, 1), date(2024, 5, 2), date(2024, 11, 1)]


def test_load_cached_missing_returns_none(tmp_path: Path):
    assert load_cached("NOPE", tmp_path) is None


def test_load_cached_empty_list_round_trip(tmp_path: Path):
    save_to_cache("EMPTY", [], tmp_path)
    loaded = load_cached("EMPTY", tmp_path)
    assert loaded == []


# --- fetch_calendar dispatch ----------------------------------------------


def test_fetch_calendar_empty_ticker_list(tmp_path: Path):
    assert fetch_calendar([], start="2024-01-01", end="2025-12-31", base_dir=tmp_path) == {}


def test_fetch_calendar_cache_hit_skips_yfinance(tmp_path: Path):
    """If a cache file exists and use_cache=True, yfinance must not be called."""
    cached = [date(2024, 5, 2), date(2024, 8, 1), date(2024, 11, 1)]
    save_to_cache("AAPL", cached, tmp_path)

    with mock.patch.object(ec, "fetch_ticker_earnings") as mocked:
        result = fetch_calendar(
            ["AAPL"],
            start="2024-01-01",
            end="2025-12-31",
            base_dir=tmp_path,
        )
    mocked.assert_not_called()
    assert result == {"AAPL": cached}


def test_fetch_calendar_cache_filters_by_window(tmp_path: Path):
    cached = [date(2023, 11, 1), date(2024, 5, 2), date(2024, 8, 1), date(2026, 2, 1)]
    save_to_cache("AAPL", cached, tmp_path)

    result = fetch_calendar(
        ["AAPL"],
        start=date(2024, 1, 1),
        end=date(2025, 12, 31),
        base_dir=tmp_path,
    )
    assert result["AAPL"] == [date(2024, 5, 2), date(2024, 8, 1)]


def test_fetch_calendar_no_cache_calls_fetch_and_persists(tmp_path: Path):
    """use_cache=False or no file present -> fetch, then write cache."""
    fake_dates = [date(2024, 5, 2), date(2024, 8, 1)]
    with mock.patch.object(ec, "fetch_ticker_earnings", return_value=fake_dates) as mocked:
        result = fetch_calendar(
            ["AAPL"],
            start="2024-01-01",
            end="2025-12-31",
            base_dir=tmp_path,
            use_cache=False,
        )
    mocked.assert_called_once()
    assert result == {"AAPL": fake_dates}
    # cache should now exist
    assert cache_path("AAPL", tmp_path).exists()
    assert load_cached("AAPL", tmp_path) == fake_dates


def test_fetch_calendar_per_ticker_failure_omitted(tmp_path: Path):
    """If one ticker raises during fetch, it is omitted; others succeed."""

    def flaky(ticker, *, start, end):
        if ticker == "BAD":
            raise RuntimeError("simulated network failure")
        return [date(2024, 5, 2)]

    with mock.patch.object(ec, "fetch_ticker_earnings", side_effect=flaky):
        result = fetch_calendar(
            ["AAPL", "BAD", "MSFT"],
            start="2024-01-01",
            end="2025-12-31",
            base_dir=tmp_path,
            use_cache=False,
        )
    assert "BAD" not in result
    assert result["AAPL"] == [date(2024, 5, 2)]
    assert result["MSFT"] == [date(2024, 5, 2)]


# --- fetch_ticker_earnings semantics ---------------------------------------


def test_fetch_ticker_earnings_invalid_arg_raises():
    with pytest.raises(ValueError):
        fetch_ticker_earnings("")
    with pytest.raises(ValueError):
        fetch_ticker_earnings("   ")


def test_fetch_ticker_earnings_yfinance_returns_none(monkeypatch):
    """Delisted-symbol path: yfinance returns None -> empty list, no crash."""

    class _FakeTicker:
        def __init__(self, sym):
            pass

        def get_earnings_dates(self, limit=100):
            return None

    monkeypatch.setattr(ec.yf, "Ticker", _FakeTicker)
    assert fetch_ticker_earnings("ZZZZZZZ_FAKE") == []


def test_fetch_ticker_earnings_yfinance_raises_returns_empty(monkeypatch):
    """Network/parse failure path: yfinance raises -> empty list, no crash."""

    class _FakeTicker:
        def __init__(self, sym):
            pass

        def get_earnings_dates(self, limit=100):
            raise RuntimeError("boom")

    monkeypatch.setattr(ec.yf, "Ticker", _FakeTicker)
    assert fetch_ticker_earnings("AAPL") == []


def test_fetch_ticker_earnings_filters_window(monkeypatch):
    """yfinance returns a tz-aware DatetimeIndex; we filter to [start, end]."""
    idx = pd.DatetimeIndex(
        [
            "2023-11-01 21:00:00",  # before start
            "2024-05-02 21:00:00",
            "2024-08-01 21:00:00",
            "2024-11-01 21:00:00",
            "2026-02-01 21:00:00",  # after end
        ],
        tz="America/New_York",
    )
    df = pd.DataFrame({"EPS Estimate": [1.0] * len(idx)}, index=idx)

    class _FakeTicker:
        def __init__(self, sym):
            pass

        def get_earnings_dates(self, limit=100):
            return df

    monkeypatch.setattr(ec.yf, "Ticker", _FakeTicker)
    out = fetch_ticker_earnings("AAPL", start="2024-01-01", end="2025-12-31")
    assert out == [date(2024, 5, 2), date(2024, 8, 1), date(2024, 11, 1)]


# --- live network smoke test ----------------------------------------------


@pytest.mark.network
def test_fetch_ticker_earnings_live_aapl():
    try:
        dates = fetch_ticker_earnings("AAPL", start="2024-01-01", end="2025-12-31")
    except Exception as exc:
        pytest.skip(f"network/yfinance unavailable: {exc!r}")
    if not dates:
        pytest.skip("yfinance returned no earnings dates for AAPL in window")
    assert len(dates) >= 3
    assert all(date(2024, 1, 1) <= d <= date(2025, 12, 31) for d in dates)


@pytest.mark.network
def test_fetch_ticker_earnings_live_invalid_returns_empty():
    """A bogus symbol must not crash; returns []."""
    try:
        out = fetch_ticker_earnings("ZZZZZZZ_FAKE")
    except ValueError:
        raise  # programmer error path -> let pytest fail
    except Exception as exc:
        pytest.skip(f"network/yfinance unavailable: {exc!r}")
    assert out == []
