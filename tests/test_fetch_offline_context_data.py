"""Hermetic tests for scripts/fetch_offline_context_data.py."""
from __future__ import annotations

import importlib
import json
import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _import_module():
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    if "fetch_offline_context_data" in sys.modules:
        return importlib.reload(sys.modules["fetch_offline_context_data"])
    return importlib.import_module("fetch_offline_context_data")


def test_safe_symbol_name_handles_indexes():
    mod = _import_module()
    assert mod.safe_symbol_name("^VIX") == "IDX_VIX"
    assert mod.safe_symbol_name("BRK.B") == "BRK.B"
    assert mod.safe_symbol_name("ABC/DEF") == "ABC_DEF"


def test_default_tickers_are_unique_and_include_context_sets():
    mod = _import_module()
    tickers = mod.default_tickers()
    assert len(tickers) == len(set(t.upper() for t in tickers))
    assert "AAPL" in tickers
    assert "SPY" in tickers
    assert "XLK" in tickers
    assert "^VIX" in tickers


def test_fetch_yfinance_daily_writes_per_ticker_parquet(tmp_path, monkeypatch):
    mod = _import_module()
    idx = pd.DatetimeIndex(["2024-05-07", "2024-05-08"], name="date")
    prices = {"AAPL": pd.DataFrame({"close": [100.0, 101.0]}, index=idx)}
    monkeypatch.setattr(mod, "fetch_daily_prices", lambda *a, **k: prices)

    summary = mod.fetch_yfinance_daily(
        ["AAPL"],
        start=date(2024, 5, 7),
        end=date(2024, 5, 8),
        output_dir=tmp_path / "offline",
        status_file=tmp_path / "status.json",
        chunk_size=10,
        refresh=False,
    )

    assert summary["written"] == 1
    out = tmp_path / "offline" / "yfinance" / "daily" / "AAPL.parquet"
    assert out.exists()
    df = pd.read_parquet(out)
    assert list(df.columns) == ["date", "close"]


def test_fetch_yfinance_daily_uses_alias_but_writes_original_symbol(tmp_path, monkeypatch):
    mod = _import_module()
    idx = pd.DatetimeIndex(["2024-05-07"], name="date")
    prices = {"XYZ": pd.DataFrame({"close": [70.0]}, index=idx)}

    seen = {}

    def fake_fetch(tickers, *_args, **_kwargs):
        seen["tickers"] = tickers
        return prices

    monkeypatch.setattr(mod, "fetch_daily_prices", fake_fetch)

    summary = mod.fetch_yfinance_daily(
        ["SQ"],
        start=date(2024, 5, 7),
        end=date(2024, 5, 8),
        output_dir=tmp_path / "offline",
        status_file=tmp_path / "status.json",
        chunk_size=10,
        refresh=False,
    )

    assert summary["written"] == 1
    assert seen["tickers"] == ["XYZ"]
    out = tmp_path / "offline" / "yfinance" / "daily" / "SQ.parquet"
    df = pd.read_parquet(out)
    assert df["source_ticker"].unique().tolist() == ["XYZ"]


def test_fetch_fred_series_writes_normalized_parquet(tmp_path, monkeypatch):
    mod = _import_module()

    class FakeResponse:
        text = "observation_date,DGS10\n2024-05-07,4.5\n2024-05-08,.\n"

        def raise_for_status(self):
            return None

    monkeypatch.setattr(mod.requests, "get", lambda *a, **k: FakeResponse())

    summary = mod.fetch_fred_series(
        {"DGS10": "10-year Treasury yield"},
        start=date(2024, 5, 7),
        end=date(2024, 5, 8),
        output_dir=tmp_path / "offline",
        status_file=tmp_path / "status.json",
        refresh=False,
    )

    assert summary["written"] == 1
    df = pd.read_parquet(tmp_path / "offline" / "fred" / "DGS10.parquet")
    assert list(df.columns) == ["date", "value", "series_id", "description"]
    assert df["series_id"].unique().tolist() == ["DGS10"]


def test_fetch_yfinance_actions_writes_empty_cache(tmp_path, monkeypatch):
    mod = _import_module()
    fake_ticker = MagicMock()
    fake_ticker.actions = pd.DataFrame()
    monkeypatch.setattr(mod.yf, "Ticker", lambda ticker: fake_ticker)

    summary = mod.fetch_yfinance_actions(
        ["AAPL"],
        start=date(2024, 5, 7),
        end=date(2024, 5, 8),
        output_dir=tmp_path / "offline",
        status_file=tmp_path / "status.json",
        refresh=False,
    )

    assert summary["written"] == 1
    assert (tmp_path / "offline" / "yfinance" / "actions" / "AAPL.parquet").exists()


def test_main_skip_modes_writes_manifest(tmp_path):
    mod = _import_module()
    rc = mod.main([
        "--output-dir", str(tmp_path / "offline"),
        "--status-file", str(tmp_path / "status.json"),
        "--skip-yfinance",
        "--skip-actions",
        "--skip-fred",
    ])

    assert rc == 0
    manifest = json.loads((tmp_path / "offline" / "manifest.json").read_text())
    assert manifest["summaries"] == {}
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["stages"]["complete"]["status"] == "complete"
