"""Hermetic tests for ``scripts/augment_chain_pricing.py``.

All Polygon calls are mocked via a fake client; no test exercises the real
network. The script is imported as a module via the same scripts-on-sys.path
trick used by ``tests/test_fetch_paper_v1_data.py``.
"""
from __future__ import annotations

import importlib
import json
import math
import sys
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _import_module():
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    if "augment_chain_pricing" in sys.modules:
        return importlib.reload(sys.modules["augment_chain_pricing"])
    return importlib.import_module("augment_chain_pricing")


@pytest.fixture
def mod(monkeypatch, tmp_path):
    """Module under test, with default paths redirected to tmp_path."""
    m = _import_module()
    monkeypatch.setattr(m, "STATUS_FILE", tmp_path / "status.json")
    monkeypatch.setattr(m, "DEFAULT_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(m, "DEFAULT_LOG_FILE", tmp_path / "paper_v1_augment.log")
    return m


# ---------------------------------------------------------------------------
# Cache path helper
# ---------------------------------------------------------------------------


def test_cache_path_priced_layout(tmp_path, mod):
    p = mod.cache_path_priced("AAPL", date(2024, 8, 1), year="year1", base_dir=tmp_path)
    assert p == tmp_path / "year1" / "chains_priced" / "AAPL" / "2024-08-01.parquet"


def test_cache_path_priced_no_side_effects(tmp_path, mod):
    """Path helper must not create directories."""
    mod.cache_path_priced("AAPL", date(2024, 8, 1), year="year1", base_dir=tmp_path)
    assert not (tmp_path / "year1").exists()


def test_cache_path_chain_layout(tmp_path, mod):
    p = mod.cache_path_chain("MSFT", date(2025, 1, 30), year="year2", base_dir=tmp_path)
    assert p == tmp_path / "year2" / "chains" / "MSFT" / "2025-01-30.parquet"


def test_cache_path_underlying_daily_layout(tmp_path, mod):
    p = mod.cache_path_underlying_daily("AAPL", year="year1", base_dir=tmp_path)
    assert p == tmp_path / "year1" / "AAPL_daily.parquet"


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------


def test_trading_day_on_or_before_weekend(mod):
    # Sunday 2024-08-04 -> Friday 2024-08-02
    assert mod.trading_day_on_or_before(date(2024, 8, 4)) == date(2024, 8, 2)


def test_trading_day_on_or_before_weekday(mod):
    assert mod.trading_day_on_or_before(date(2024, 8, 1)) == date(2024, 8, 1)


def test_add_trading_days_skips_weekend(mod):
    # 2024-08-01 is Thursday; +5 trading days = next Thursday 2024-08-08
    assert mod.add_trading_days(date(2024, 8, 1), 5) == date(2024, 8, 8)


def test_next_friday_on_or_after(mod):
    # Wed 2024-08-07 -> Fri 2024-08-09
    assert mod.next_friday_on_or_after(date(2024, 8, 7)) == date(2024, 8, 9)
    # Friday 2024-08-09 -> same day
    assert mod.next_friday_on_or_after(date(2024, 8, 9)) == date(2024, 8, 9)


# ---------------------------------------------------------------------------
# ATM strike picker
# ---------------------------------------------------------------------------


def test_pick_atm_strike_round_to_nearest(mod):
    # 5 strikes around 100, spot=99.5 -> 100 (closer than 99 by 0.5 vs 0.5; tie-break to lower)
    # Tie at 99.5: |100-99.5|==|99-99.5|; we tie-break by smaller strike per stable sort.
    assert mod.pick_atm_strike([95.0, 99.0, 100.0, 105.0, 110.0], 99.5) == 99.0


def test_pick_atm_strike_far_strikes(mod):
    # spot=99.5, no exact ATM -> closest is 100.
    assert mod.pick_atm_strike([90.0, 100.0, 110.0], 99.5) == 100.0


def test_pick_atm_strike_handles_nan_spot(mod):
    # NaN underlying close → fall back to median.
    out = mod.pick_atm_strike([90.0, 95.0, 100.0, 105.0, 110.0], float("nan"))
    assert out == 100.0


def test_pick_atm_strike_empty_raises(mod):
    with pytest.raises(ValueError):
        mod.pick_atm_strike([], 100.0)


# ---------------------------------------------------------------------------
# IV inversion smoke
# ---------------------------------------------------------------------------


def test_compute_atm_iv_round_trip(mod):
    """BS price → recovered IV should be close to input."""
    from features.greeks import bs_price

    spot = 100.0
    strike = 100.0
    sigma_in = 0.25
    r = 0.04
    entry = date(2024, 6, 3)
    expiry = entry + timedelta(days=30)
    T = 30 / 365.0
    call_mid = bs_price("call", spot, strike, T, r, sigma_in)
    put_mid = bs_price("put", spot, strike, T, r, sigma_in)
    iv = mod.compute_atm_iv_from_straddle(
        call_mid=call_mid,
        put_mid=put_mid,
        spot=spot,
        strike=strike,
        entry_date=entry,
        expiration_date=expiry,
        risk_free_rate=r,
    )
    assert abs(iv - sigma_in) < 5e-3


def test_compute_atm_iv_nan_inputs(mod):
    """NaN mid prices -> NaN atm_iv (no crash)."""
    iv = mod.compute_atm_iv_from_straddle(
        call_mid=float("nan"),
        put_mid=float("nan"),
        spot=100.0,
        strike=100.0,
        entry_date=date(2024, 6, 3),
        expiration_date=date(2024, 7, 3),
    )
    assert math.isnan(iv)


def test_compute_atm_iv_sub_intrinsic_returns_nan(mod):
    """Sub-intrinsic price → solver fails → atm_iv=NaN, no exception."""
    iv = mod.compute_atm_iv_from_straddle(
        call_mid=-1.0,  # garbage
        put_mid=-1.0,
        spot=100.0,
        strike=100.0,
        entry_date=date(2024, 6, 3),
        expiration_date=date(2024, 7, 3),
    )
    assert math.isnan(iv)


# ---------------------------------------------------------------------------
# Underlying close lookup
# ---------------------------------------------------------------------------


def test_lookup_close_on_exact_match(mod):
    idx = pd.DatetimeIndex(["2024-06-03", "2024-06-04", "2024-06-05"], name="date")
    df = pd.DataFrame({"close": [100.0, 101.0, 102.0]}, index=idx)
    assert mod.lookup_close_on(df, date(2024, 6, 4)) == 101.0


def test_lookup_close_on_falls_back_to_prior(mod):
    idx = pd.DatetimeIndex(["2024-06-03", "2024-06-05"], name="date")
    df = pd.DataFrame({"close": [100.0, 102.0]}, index=idx)
    # 6/4 missing -> use 6/3
    assert mod.lookup_close_on(df, date(2024, 6, 4)) == 100.0


def test_lookup_close_on_empty_returns_nan(mod):
    assert math.isnan(mod.lookup_close_on(pd.DataFrame(), date(2024, 6, 4)))


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


REQUIRED_COLUMNS = {
    "ticker", "underlying_ticker", "contract_type", "strike_price",
    "expiration_date", "as_of_date", "timing", "event_date",
    "bid", "ask", "mid", "underlying_close", "is_atm", "atm_iv",
}


def _make_chain_metadata(ticker: str, event_date: date, expiry: date, strikes: list[float]) -> pd.DataFrame:
    """Build a minimal metadata-only chain df like the fetcher writes."""
    rows = []
    for k in strikes:
        for ct in ("call", "put"):
            tag = "C" if ct == "call" else "P"
            rows.append({
                "ticker": f"O:{ticker}{expiry.strftime('%y%m%d')}{tag}{int(k * 1000):08d}",
                "underlying_ticker": ticker,
                "contract_type": ct,
                "strike_price": float(k),
                "expiration_date": expiry.isoformat(),
                "exercise_style": "american",
                "shares_per_contract": 100,
                "as_of_date": (event_date - timedelta(days=1)).isoformat(),
                "event_date": event_date.isoformat(),
            })
    return pd.DataFrame(rows)


def _underlying_daily(start: date, end: date, base_close: float = 100.0) -> pd.DataFrame:
    days = pd.bdate_range(start, end)
    return pd.DataFrame({"close": [base_close + i * 0.1 for i in range(len(days))]},
                        index=pd.DatetimeIndex(days, name="date"))


def test_output_schema_columns_match_spec(tmp_path, mod):
    ticker = "AAPL"
    event = date(2024, 8, 1)
    expiry = date(2024, 8, 9)
    chain = _make_chain_metadata(ticker, event, expiry, [95.0, 100.0, 105.0])
    daily = _underlying_daily(date(2024, 7, 25), date(2024, 8, 12), base_close=99.5)

    fake_client = MagicMock()
    # Per-fetch close: arbitrary positive
    fake_client.fetch_options_aggregates.return_value = pd.DataFrame(
        {"close": [2.5]},
        index=pd.DatetimeIndex(["2024-07-31"], name="date"),
    )

    target = tmp_path / "out.parquet"
    wrote = mod.augment_satellite_event(
        client=fake_client,
        ticker=ticker,
        event_date=event,
        chain_metadata=chain,
        underlying_daily=daily,
        target=target,
    )
    assert wrote is True
    assert target.exists()
    df = pd.read_parquet(target)
    assert set(df.columns) == REQUIRED_COLUMNS
    # 4 rows: (call, put) x (entry, exit)
    assert len(df) == 4
    assert set(df["timing"].unique()) == {"entry", "exit"}
    assert set(df["contract_type"].unique()) == {"call", "put"}
    # is_atm True for all rows (only ATM strike priced)
    assert df["is_atm"].all()


def test_output_atm_strike_is_picked_correctly(tmp_path, mod):
    """Synthetic chain with strikes around 100; underlying_close=99.5 → picks 100 (or 99.5-tie)."""
    ticker = "AAPL"
    event = date(2024, 8, 1)
    expiry = date(2024, 8, 9)
    # Strikes 95, 99, 100, 105, 110 — closest to 99.5 is 99 (tie with 100; tie-break smaller)
    chain = _make_chain_metadata(ticker, event, expiry, [95.0, 99.0, 100.0, 105.0, 110.0])
    daily = _underlying_daily(date(2024, 7, 25), date(2024, 8, 12), base_close=99.0)

    # Force entry-date close = 99.5
    daily.loc[pd.Timestamp("2024-07-31")] = [99.5]

    fake_client = MagicMock()
    fake_client.fetch_options_aggregates.return_value = pd.DataFrame(
        {"close": [3.0]},
        index=pd.DatetimeIndex(["2024-07-31"], name="date"),
    )

    target = tmp_path / "out.parquet"
    mod.augment_satellite_event(
        client=fake_client,
        ticker=ticker,
        event_date=event,
        chain_metadata=chain,
        underlying_daily=daily,
        target=target,
    )
    df = pd.read_parquet(target)
    # ATM strike picked (tie-break smaller per pick_atm_strike contract).
    assert df["strike_price"].nunique() == 1
    picked = float(df["strike_price"].iloc[0])
    assert picked == 99.0  # tie-broken to smaller


def test_output_includes_atm_iv_broadcast(tmp_path, mod):
    """atm_iv computed once and broadcast to all rows."""
    from features.greeks import bs_price

    ticker = "AAPL"
    event = date(2024, 8, 1)
    expiry = date(2024, 9, 6)  # ~36d for clean IV
    chain = _make_chain_metadata(ticker, event, expiry, [95.0, 100.0, 105.0])
    daily = _underlying_daily(date(2024, 7, 25), date(2024, 8, 12), base_close=100.0)

    # Build BS-consistent prices for sigma=0.30
    sigma = 0.30
    T = (expiry - date(2024, 7, 31)).days / 365.0
    call_mid = bs_price("call", 100.0, 100.0, T, 0.04, sigma)
    put_mid = bs_price("put", 100.0, 100.0, T, 0.04, sigma)

    fake_client = MagicMock()
    counter = {"n": 0}

    def _agg(option_ticker, from_date, to_date):
        # The contract ticker has format O:AAPL240906C00100000 — the C or P
        # right before the 8-digit strike disambiguates call vs put.
        counter["n"] += 1
        close = call_mid if "C0" in option_ticker else put_mid
        return pd.DataFrame({"close": [close]},
                            index=pd.DatetimeIndex([from_date], name="date"))

    fake_client.fetch_options_aggregates.side_effect = _agg

    target = tmp_path / "out.parquet"
    mod.augment_satellite_event(
        client=fake_client,
        ticker=ticker,
        event_date=event,
        chain_metadata=chain,
        underlying_daily=daily,
        target=target,
    )
    df = pd.read_parquet(target)
    # IV recovered close to input.
    iv_vals = df["atm_iv"].dropna().unique()
    assert len(iv_vals) == 1
    assert abs(float(iv_vals[0]) - sigma) < 0.02


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_idempotency_skips_existing_priced_parquet(tmp_path, mod):
    ticker = "AAPL"
    event = date(2024, 8, 1)
    expiry = date(2024, 8, 9)
    chain = _make_chain_metadata(ticker, event, expiry, [95.0, 100.0, 105.0])
    daily = _underlying_daily(date(2024, 7, 25), date(2024, 8, 12))

    target = tmp_path / "out.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    # Pre-write a marker priced parquet
    pd.DataFrame([{
        "ticker": "pre-existing", "underlying_ticker": ticker, "contract_type": "call",
        "strike_price": 100.0, "expiration_date": expiry.isoformat(),
        "as_of_date": event.isoformat(), "timing": "entry",
        "event_date": event.isoformat(),
        "bid": float("nan"), "ask": float("nan"), "mid": 1.0,
        "underlying_close": 100.0, "is_atm": True, "atm_iv": 0.3,
    }]).to_parquet(target)

    fake_client = MagicMock()
    fake_client.fetch_options_aggregates.side_effect = AssertionError("must not call API")

    wrote = mod.augment_satellite_event(
        client=fake_client,
        ticker=ticker,
        event_date=event,
        chain_metadata=chain,
        underlying_daily=daily,
        target=target,
    )
    assert wrote is False
    # The pre-existing file is untouched.
    df = pd.read_parquet(target)
    assert df.iloc[0]["ticker"] == "pre-existing"


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------


def test_atomic_write_no_tmp_left_on_success(tmp_path, mod):
    df = pd.DataFrame({"a": [1, 2, 3]})
    target = tmp_path / "x.parquet"
    mod._atomic_write_parquet(df, target)
    assert target.exists()
    assert not list(tmp_path.glob("*.parquet.tmp"))


def test_atomic_write_crash_leaves_old_or_new(tmp_path, mod, monkeypatch):
    """If rename fails mid-way, old file (or no file) remains — never half-written.

    We simulate a crash AFTER the .tmp is written but before os.replace runs.
    The original target — if absent — stays absent; the .tmp cleanup is the
    caller's responsibility, but the contract is "old or new, never half."
    """
    target = tmp_path / "y.parquet"
    df = pd.DataFrame({"a": [1, 2, 3]})

    # Pre-existing file = "old" state.
    df.to_parquet(target)
    old_size = target.stat().st_size

    # Patch os.replace to raise after the .tmp is created.
    real_replace = mod.os.replace
    def _boom(src, dst):
        raise RuntimeError("simulated kill -9 mid-rename")
    monkeypatch.setattr(mod.os, "replace", _boom)

    new_df = pd.DataFrame({"a": [10, 20, 30, 40]})
    with pytest.raises(RuntimeError, match="simulated"):
        mod._atomic_write_parquet(new_df, target)

    # Old file untouched.
    assert target.exists()
    assert target.stat().st_size == old_size
    # Restore for the rest of the suite.
    monkeypatch.setattr(mod.os, "replace", real_replace)


# ---------------------------------------------------------------------------
# Status writer
# ---------------------------------------------------------------------------


def test_status_writer_creates_stages_dict(tmp_path, mod):
    sf = tmp_path / "status.json"
    doc = mod.write_status(
        "augment_year1_pricing",
        "running",
        status_file=sf,
        done=0,
        total=10,
    )
    assert sf.exists()
    on_disk = json.loads(sf.read_text())
    assert on_disk == doc
    assert "augment_year1_pricing" in on_disk["stages"]
    stage = on_disk["stages"]["augment_year1_pricing"]
    assert stage["status"] == "running"
    assert stage["done"] == 0
    assert stage["total"] == 10


def test_status_writer_files_written_field(tmp_path, mod):
    sf = tmp_path / "status.json"
    mod.write_status(
        "augment_year1_pricing",
        "running",
        status_file=sf,
        done=5,
        total=10,
        files_written=3,
    )
    on_disk = json.loads(sf.read_text())
    stage = on_disk["stages"]["augment_year1_pricing"]
    assert stage["files_written"] == 3


def test_status_writer_phase_override(tmp_path, mod):
    sf = tmp_path / "status.json"
    mod.write_status("init", "running", status_file=sf, phase="augmenting")
    assert json.loads(sf.read_text())["phase"] == "augmenting"
    mod.write_status("complete", "complete", status_file=sf, phase="augment_complete")
    assert json.loads(sf.read_text())["phase"] == "augment_complete"


def test_status_writer_atomic_no_tmp_left(tmp_path, mod):
    sf = tmp_path / "status.json"
    mod.write_status("stage_a", "running", status_file=sf)
    assert not list(tmp_path.glob("status.json.tmp"))


# ---------------------------------------------------------------------------
# NaN tolerance
# ---------------------------------------------------------------------------


def test_nan_tolerance_when_polygon_returns_empty(tmp_path, mod):
    """Empty Polygon results -> rows written with NaN mid; no crash."""
    ticker = "AAPL"
    event = date(2024, 8, 1)
    expiry = date(2024, 8, 9)
    chain = _make_chain_metadata(ticker, event, expiry, [95.0, 100.0, 105.0])
    daily = _underlying_daily(date(2024, 7, 25), date(2024, 8, 12))

    fake_client = MagicMock()
    # Always empty -> mid should be NaN.
    fake_client.fetch_options_aggregates.return_value = pd.DataFrame()

    target = tmp_path / "out.parquet"
    wrote = mod.augment_satellite_event(
        client=fake_client,
        ticker=ticker,
        event_date=event,
        chain_metadata=chain,
        underlying_daily=daily,
        target=target,
    )
    assert wrote is True
    df = pd.read_parquet(target)
    assert df["mid"].isna().all()
    # bid/ask always NaN per Polygon Free aggregates contract.
    assert df["bid"].isna().all()
    assert df["ask"].isna().all()
    # atm_iv NaN (can't invert NaN mids).
    assert df["atm_iv"].isna().all()


def test_nan_tolerance_polygon_error(tmp_path, mod):
    """Polygon raises -> mid NaN, no crash, file written."""
    from ingest.polygon_free import PolygonError

    ticker = "AAPL"
    event = date(2024, 8, 1)
    expiry = date(2024, 8, 9)
    chain = _make_chain_metadata(ticker, event, expiry, [95.0, 100.0, 105.0])
    daily = _underlying_daily(date(2024, 7, 25), date(2024, 8, 12))

    fake_client = MagicMock()
    fake_client.fetch_options_aggregates.side_effect = PolygonError("transient")

    target = tmp_path / "out.parquet"
    wrote = mod.augment_satellite_event(
        client=fake_client,
        ticker=ticker,
        event_date=event,
        chain_metadata=chain,
        underlying_daily=daily,
        target=target,
    )
    assert wrote is True
    df = pd.read_parquet(target)
    assert df["mid"].isna().all()


# ---------------------------------------------------------------------------
# SPY core metadata fallback
# ---------------------------------------------------------------------------


def test_spy_core_fetches_contract_metadata_when_no_cached_spy_chain(tmp_path, mod):
    """SPY core should not require pre-existing chains/SPY metadata files."""
    cache_dir = tmp_path / "cache"
    status_file = tmp_path / "status.json"
    yd = cache_dir / "year1"
    yd.mkdir(parents=True)

    entry = date(2024, 6, 3)  # Monday
    expiry = mod.next_friday_on_or_after(entry + timedelta(days=mod.CORE_DTE_TARGET_DAYS))
    days = pd.bdate_range(entry, expiry)
    spy_daily = pd.DataFrame(
        {"close": [500.0] * len(days)},
        index=pd.DatetimeIndex(days, name="date"),
    )
    spy_daily.to_parquet(yd / "spy_daily.parquet")

    chain = _make_chain_metadata("SPY", entry, expiry, [470.0, 475.0, 500.0, 525.0, 530.0])
    fake_client = MagicMock()
    fake_client.fetch_options_contracts.return_value = chain
    fake_client.fetch_options_aggregates.return_value = pd.DataFrame(
        {"close": [2.0]},
        index=pd.DatetimeIndex([entry], name="date"),
    )

    summary = mod._augment_spy_core_for_year(
        year="year1",
        client=fake_client,
        base_dir=cache_dir,
        status_file=status_file,
        year_start=entry,
        year_end=entry,
    )

    assert summary == {"skipped": False, "files_written": 1, "total": 1}
    fake_client.fetch_options_contracts.assert_called_once_with(
        "SPY",
        expiration_date=expiry.isoformat(),
        as_of=entry.isoformat(),
        limit=1000,
    )
    assert (cache_dir / "year1" / "chains" / "SPY" / f"{entry.isoformat()}.parquet").exists()
    priced = cache_dir / "year1" / "chains_priced" / "SPY" / f"{entry.isoformat()}.parquet"
    assert priced.exists()
    out = pd.read_parquet(priced)
    assert set(out.columns) == REQUIRED_COLUMNS
    assert len(out) == 8
    assert set(out["timing"].unique()) == {"entry", "exit"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_parse_args_defaults(mod):
    ns = mod.parse_args([])
    assert ns.year == "both"
    assert ns.max_events is None
    assert ns.skip_spy_core is False
    assert ns.log_file == mod.DEFAULT_LOG_FILE


def test_parse_args_year(mod):
    ns = mod.parse_args(["--year", "year1"])
    assert ns.year == "year1"


def test_parse_args_max_events(mod):
    ns = mod.parse_args(["--max-events", "5"])
    assert ns.max_events == 5


def test_parse_args_skip_spy_core(mod):
    ns = mod.parse_args(["--skip-spy-core"])
    assert ns.skip_spy_core is True


def test_parse_args_log_file(mod, tmp_path):
    log_file = tmp_path / "custom.log"
    ns = mod.parse_args(["--log-file", str(log_file)])
    assert ns.log_file == log_file


def test_parse_args_invalid_year(mod):
    with pytest.raises(SystemExit):
        mod.parse_args(["--year", "year3"])


def test_configure_logging_writes_log_file(mod, tmp_path):
    log_file = tmp_path / "logs" / "augment.log"
    mod.configure_logging(log_file)
    mod.logger.info("hello augment log")

    text = log_file.read_text()
    assert "hello augment log" in text


# ---------------------------------------------------------------------------
# End-to-end synthetic main run
# ---------------------------------------------------------------------------


def _stage_year_cache(
    *,
    cache_dir: Path,
    year: str,
    universe: list[str],
    event_dates: dict[str, list[date]],
):
    """Lay down a minimal year cache: spy_daily, earnings_dates/, chains/."""
    yd = cache_dir / year
    (yd / "earnings_dates").mkdir(parents=True, exist_ok=True)
    (yd / "chains").mkdir(parents=True, exist_ok=True)

    # SPY daily (so SPY core has spot lookups; also covered by -daily fallback).
    days = pd.bdate_range(date(2024, 5, 7), date(2024, 6, 30))
    spy_daily = pd.DataFrame(
        {"open": [500.0] * len(days), "high": [501.0] * len(days),
         "low": [499.0] * len(days), "close": [500.0] * len(days),
         "volume": [1000] * len(days), "vwap": [500.0] * len(days),
         "transactions": [10] * len(days)},
        index=pd.DatetimeIndex(days, name="date"),
    )
    spy_daily.to_parquet(yd / "spy_daily.parquet")

    for ticker in universe:
        eds = event_dates.get(ticker, [])
        ed_df = pd.DataFrame({"earnings_date": [d.isoformat() for d in eds]})
        ed_df.to_parquet(yd / "earnings_dates" / f"{ticker}.parquet", index=False)
        ticker_chain_dir = yd / "chains" / ticker
        ticker_chain_dir.mkdir(parents=True, exist_ok=True)
        for ev in eds:
            expiry = ev + timedelta(days=8)
            while expiry.weekday() != 4:
                expiry += timedelta(days=1)
            chain = _make_chain_metadata(ticker, ev, expiry, [95.0, 100.0, 105.0])
            chain.to_parquet(ticker_chain_dir / f"{ev.isoformat()}.parquet", index=False)


def test_main_end_to_end_synthetic(tmp_path, mod, monkeypatch):
    """End-to-end with a 3-event scope: writes 3 priced parquets + status augment_complete."""
    cache_dir = tmp_path / "cache"
    status = tmp_path / "status.json"

    # Tighten universe to one ticker with 3 events; year2 not run.
    monkeypatch.setattr(mod, "PAPER_V1_SATELLITE", ("AAPL",))
    monkeypatch.setattr(mod, "YEAR1_START", date(2024, 5, 7))
    monkeypatch.setattr(mod, "YEAR1_END", date(2024, 6, 30))

    _stage_year_cache(
        cache_dir=cache_dir,
        year="year1",
        universe=["AAPL"],
        event_dates={"AAPL": [date(2024, 5, 20), date(2024, 6, 3), date(2024, 6, 17)]},
    )

    fake_client = MagicMock()
    # Daily aggregates -> small frame for AAPL.
    days = pd.bdate_range(date(2024, 5, 7), date(2024, 6, 30))
    daily_df = pd.DataFrame(
        {"open": [99.5] * len(days), "high": [100.5] * len(days),
         "low": [98.5] * len(days), "close": [100.0] * len(days),
         "volume": [1000] * len(days), "vwap": [100.0] * len(days),
         "transactions": [10] * len(days)},
        index=pd.DatetimeIndex(days, name="date"),
    )
    fake_client.fetch_aggregates.return_value = daily_df

    # Option aggregates -> $2 close.
    fake_client.fetch_options_aggregates.return_value = pd.DataFrame(
        {"close": [2.0]},
        index=pd.DatetimeIndex(["2024-05-20"], name="date"),
    )
    monkeypatch.setattr(mod, "PolygonFreeClient", lambda *a, **k: fake_client)

    rc = mod.main([
        "--year", "year1",
        "--cache-dir", str(cache_dir),
        "--status-file", str(status),
        "--max-events", "3",
        "--skip-spy-core",
    ])
    assert rc == 0
    assert mod.DEFAULT_LOG_FILE.exists()
    assert "starting chain pricing augment" in mod.DEFAULT_LOG_FILE.read_text()

    # 3 priced parquets.
    out_dir = cache_dir / "year1" / "chains_priced" / "AAPL"
    files = sorted(out_dir.glob("*.parquet"))
    assert len(files) == 3

    # Status JSON shape.
    doc = json.loads(status.read_text())
    assert doc["phase"] == "augment_complete"
    assert "augment_year1_pricing" in doc["stages"]
    stage = doc["stages"]["augment_year1_pricing"]
    assert stage["status"] == "complete"
    assert stage["done"] == 3
    assert stage["total"] == 3
    assert stage["files_written"] == 3
    assert stage["current_substage"] == "year_complete"


def test_main_resume_skips_already_priced(tmp_path, mod, monkeypatch):
    """A pre-existing priced parquet must not trigger any Polygon calls for that event."""
    cache_dir = tmp_path / "cache"
    status = tmp_path / "status.json"

    monkeypatch.setattr(mod, "PAPER_V1_SATELLITE", ("AAPL",))
    monkeypatch.setattr(mod, "YEAR1_START", date(2024, 5, 7))
    monkeypatch.setattr(mod, "YEAR1_END", date(2024, 6, 30))

    _stage_year_cache(
        cache_dir=cache_dir,
        year="year1",
        universe=["AAPL"],
        event_dates={"AAPL": [date(2024, 5, 20)]},
    )
    # Pre-write the priced parquet.
    priced = mod.cache_path_priced("AAPL", date(2024, 5, 20),
                                   year="year1", base_dir=cache_dir)
    priced.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "ticker": "pre", "underlying_ticker": "AAPL", "contract_type": "call",
        "strike_price": 100.0, "expiration_date": "2024-05-31",
        "as_of_date": "2024-05-17", "timing": "entry", "event_date": "2024-05-20",
        "bid": float("nan"), "ask": float("nan"), "mid": 1.0,
        "underlying_close": 100.0, "is_atm": True, "atm_iv": 0.3,
    }]).to_parquet(priced)

    # Pre-write the underlying daily so no fetch happens for it either.
    daily_target = mod.cache_path_underlying_daily("AAPL", year="year1", base_dir=cache_dir)
    daily_target.parent.mkdir(parents=True, exist_ok=True)
    days = pd.bdate_range(date(2024, 5, 7), date(2024, 5, 31))
    daily_df = pd.DataFrame(
        {"close": [100.0] * len(days)},
        index=pd.DatetimeIndex(days, name="date"),
    )
    daily_df.to_parquet(daily_target)

    fake_client = MagicMock()
    # Any Polygon call would be a bug.
    fake_client.fetch_options_aggregates.side_effect = AssertionError("must not call")
    fake_client.fetch_aggregates.side_effect = AssertionError("must not call")
    monkeypatch.setattr(mod, "PolygonFreeClient", lambda *a, **k: fake_client)

    rc = mod.main([
        "--year", "year1",
        "--cache-dir", str(cache_dir),
        "--status-file", str(status),
        "--skip-spy-core",
    ])
    assert rc == 0
    # Pre-existing file untouched.
    df = pd.read_parquet(priced)
    assert df.iloc[0]["ticker"] == "pre"


def test_main_polygon_auth_error_exits_2(tmp_path, mod, monkeypatch):
    """Bad credentials -> exit 2, phase=error."""
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
