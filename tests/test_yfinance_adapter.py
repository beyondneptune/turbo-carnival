from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ingest.yfinance_adapter import (
    cache_path,
    fetch_daily_prices,
    load_from_parquet,
    save_to_parquet,
)


def pytest_configure(config):
    config.addinivalue_line("markers", "network: tests that require internet access")


def _make_synthetic_df(n: int = 5) -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.date_range("2026-01-02", periods=n, freq="B"), name="date")
    df = pd.DataFrame(
        {
            "open": [100.0 + i for i in range(n)],
            "high": [101.0 + i for i in range(n)],
            "low": [99.0 + i for i in range(n)],
            "close": [100.5 + i for i in range(n)],
            "volume": [1_000_000 + i * 1000 for i in range(n)],
        },
        index=idx,
    )
    return df


def test_cache_path_layout(tmp_path: Path):
    assert cache_path("SPY", tmp_path) == tmp_path / "prices" / "SPY.parquet"
    assert cache_path("AAPL", tmp_path) == tmp_path / "prices" / "AAPL.parquet"


def test_save_to_parquet_creates_directory_and_files(tmp_path: Path):
    prices = {"SPY": _make_synthetic_df(5)}
    assert not (tmp_path / "prices").exists()
    save_to_parquet(prices, tmp_path)
    assert (tmp_path / "prices").is_dir()
    assert cache_path("SPY", tmp_path).exists()


def test_save_to_parquet_preserves_schema_on_round_trip(tmp_path: Path):
    original = _make_synthetic_df(5)
    save_to_parquet({"SPY": original}, tmp_path)
    loaded = load_from_parquet("SPY", tmp_path)
    assert loaded is not None
    assert list(loaded.columns) == list(original.columns)
    assert loaded.index.name == "date"
    assert isinstance(loaded.index, pd.DatetimeIndex)


def test_load_from_parquet_returns_none_for_missing_file(tmp_path: Path):
    assert load_from_parquet("NOPE", tmp_path) is None


def test_load_from_parquet_round_trip(tmp_path: Path):
    original = _make_synthetic_df(5)
    save_to_parquet({"SPY": original}, tmp_path)
    loaded = load_from_parquet("SPY", tmp_path)
    assert loaded is not None
    pd.testing.assert_frame_equal(
        loaded.sort_index(),
        original.sort_index(),
        check_freq=False,
    )


def test_save_to_parquet_handles_multiple_tickers(tmp_path: Path):
    prices = {"SPY": _make_synthetic_df(3), "AAPL": _make_synthetic_df(4)}
    save_to_parquet(prices, tmp_path)
    assert cache_path("SPY", tmp_path).exists()
    assert cache_path("AAPL", tmp_path).exists()
    aapl = load_from_parquet("AAPL", tmp_path)
    assert aapl is not None
    assert len(aapl) == 4


def test_fetch_daily_prices_empty_ticker_list_returns_empty_dict():
    result = fetch_daily_prices([], "2026-01-01", "2026-01-10")
    assert result == {}


@pytest.mark.network
def test_fetch_daily_prices_live_spy():
    try:
        result = fetch_daily_prices(["SPY"], "2026-01-02", "2026-01-10")
    except Exception as exc:  # network-dependent: skip on any failure
        pytest.skip(f"network/yfinance unavailable: {exc!r}")
    if not result or "SPY" not in result:
        pytest.skip("yfinance returned no data for SPY in test window")
    df = result["SPY"]
    assert len(df) > 0
    assert "close" in df.columns
    assert df.index.name == "date"
