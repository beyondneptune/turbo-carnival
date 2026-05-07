"""Tests for the locked paper_v1 Top-50 satellite universe.

These tests pin the snapshot: 50 unique uppercase tickers, no overlap with
the core sleeve, and synthetic candidates that all pass the liquidity gates
defined in `signals.earnings_universe`."""
from __future__ import annotations

import pandas as pd

from signals.earnings_universe import (
    REQUIRED_COLUMNS,
    UniverseFilters,
    apply_liquidity_filters,
)
from signals.paper_v1_universe import (
    PAPER_V1_CORE,
    PAPER_V1_SATELLITE,
    get_universe_candidates,
)


# ---------------------------------------------------------------------------
# Ticker list shape
# ---------------------------------------------------------------------------


def test_satellite_has_exactly_50_tickers():
    assert len(PAPER_V1_SATELLITE) == 50


def test_satellite_tickers_are_unique():
    assert len(set(PAPER_V1_SATELLITE)) == 50


def test_satellite_tickers_are_all_uppercase():
    for t in PAPER_V1_SATELLITE:
        assert isinstance(t, str)
        assert t == t.upper(), f"ticker not uppercase: {t!r}"
        assert t.strip() == t, f"ticker has whitespace: {t!r}"
        assert len(t) > 0, "empty ticker"


def test_core_and_satellite_do_not_overlap():
    core = set(PAPER_V1_CORE)
    sat = set(PAPER_V1_SATELLITE)
    assert core.isdisjoint(sat), f"overlap: {core & sat}"


# ---------------------------------------------------------------------------
# Candidate frame contract
# ---------------------------------------------------------------------------


def test_candidates_returns_dataframe():
    df = get_universe_candidates()
    assert isinstance(df, pd.DataFrame)


def test_candidates_has_exactly_50_rows():
    df = get_universe_candidates()
    assert len(df) == 50


def test_candidates_has_all_required_columns():
    df = get_universe_candidates()
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"missing required column: {col}"


def test_candidates_tickers_match_satellite_list():
    df = get_universe_candidates()
    assert list(df["ticker"]) == list(PAPER_V1_SATELLITE)


# ---------------------------------------------------------------------------
# End-to-end liquidity-gate pass
# ---------------------------------------------------------------------------


def test_all_50_pass_default_liquidity_filters():
    df = get_universe_candidates()
    eligible = apply_liquidity_filters(df)
    assert len(eligible) == 50, (
        f"expected all 50 to pass, got {len(eligible)}: "
        f"missing={set(df['ticker']) - set(eligible['ticker'])}"
    )


def test_all_50_pass_explicit_default_filters():
    df = get_universe_candidates()
    eligible = apply_liquidity_filters(df, filters=UniverseFilters())
    assert set(eligible["ticker"]) == set(PAPER_V1_SATELLITE)
