from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from signals.earnings_universe import (
    REQUIRED_COLUMNS,
    UniverseFilters,
    apply_liquidity_filters,
    build_universe,
    explain_exclusion,
    select_top_n,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candidates(n: int = 100, seed: int = 0) -> pd.DataFrame:
    """Synthetic candidate frame; mixed pass/fail by construction (random thresholds)."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "ticker": [f"T{i:03d}" for i in range(n)],
            "options_adv_30d": rng.integers(1_000, 100_000, n),
            "options_oi_30d": rng.integers(2_000, 200_000, n),
            "atm_spread_pct_30d": rng.uniform(0.01, 0.15, n),
            "earnings_events_count": rng.integers(0, 20, n),
        }
    )


def _passing_row(ticker: str, adv: int = 50_000) -> dict:
    """A row that clears every gate by a comfortable margin. Override `adv` to control rank."""
    return {
        "ticker": ticker,
        "options_adv_30d": adv,
        "options_oi_30d": 100_000,
        "atm_spread_pct_30d": 0.04,
        "earnings_events_count": 12,
    }


# ---------------------------------------------------------------------------
# Required-column validation
# ---------------------------------------------------------------------------


def test_apply_liquidity_filters_raises_on_missing_column():
    df = _make_candidates(5).drop(columns=["options_adv_30d"])
    with pytest.raises(ValueError, match="options_adv_30d"):
        apply_liquidity_filters(df)


def test_apply_liquidity_filters_raises_lists_all_missing_columns():
    df = _make_candidates(5).drop(columns=["options_oi_30d", "atm_spread_pct_30d"])
    with pytest.raises(ValueError) as excinfo:
        apply_liquidity_filters(df)
    msg = str(excinfo.value)
    assert "options_oi_30d" in msg
    assert "atm_spread_pct_30d" in msg


# ---------------------------------------------------------------------------
# Single-gate behaviour
# ---------------------------------------------------------------------------


def test_three_passing_candidates_all_returned_and_ranked():
    df = pd.DataFrame(
        [
            _passing_row("AAA", adv=70_000),
            _passing_row("BBB", adv=50_000),
            _passing_row("CCC", adv=30_000),
        ]
    )
    out = build_universe(df)
    assert list(out["ticker"]) == ["AAA", "BBB", "CCC"]
    assert list(out["rank"]) == [1, 2, 3]


def test_one_candidate_fails_adv():
    rows = [
        _passing_row("AAA"),
        _passing_row("BBB"),
        _passing_row("FAIL"),
    ]
    rows[2]["options_adv_30d"] = 4_000  # below 5_000 floor
    out = apply_liquidity_filters(pd.DataFrame(rows))
    assert "FAIL" not in set(out["ticker"])
    assert set(out["ticker"]) == {"AAA", "BBB"}


def test_one_candidate_fails_oi():
    rows = [_passing_row("AAA"), _passing_row("FAIL")]
    rows[1]["options_oi_30d"] = 8_000  # below 10_000 floor
    out = apply_liquidity_filters(pd.DataFrame(rows))
    assert list(out["ticker"]) == ["AAA"]


def test_one_candidate_fails_spread():
    rows = [_passing_row("AAA"), _passing_row("FAIL")]
    rows[1]["atm_spread_pct_30d"] = 0.10  # above 0.08 ceiling
    out = apply_liquidity_filters(pd.DataFrame(rows))
    assert list(out["ticker"]) == ["AAA"]


def test_one_candidate_fails_event_count():
    rows = [_passing_row("AAA"), _passing_row("FAIL")]
    rows[1]["earnings_events_count"] = 5  # below 8 floor
    out = apply_liquidity_filters(pd.DataFrame(rows))
    assert list(out["ticker"]) == ["AAA"]


def test_boundary_values_pass_inclusive():
    """Exactly-at-threshold rows should pass (>= / <= inclusive)."""
    rows = [
        {
            "ticker": "EDGE",
            "options_adv_30d": 5_000,
            "options_oi_30d": 10_000,
            "atm_spread_pct_30d": 0.08,
            "earnings_events_count": 8,
        }
    ]
    out = apply_liquidity_filters(pd.DataFrame(rows))
    assert list(out["ticker"]) == ["EDGE"]


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------


def test_nan_values_fail_without_exception():
    rows = [
        _passing_row("AAA"),
        _passing_row("NAN_ADV"),
        _passing_row("NAN_OI"),
        _passing_row("NAN_SPREAD"),
        _passing_row("NAN_EVENTS"),
    ]
    df = pd.DataFrame(rows)
    df.loc[df["ticker"] == "NAN_ADV", "options_adv_30d"] = np.nan
    df.loc[df["ticker"] == "NAN_OI", "options_oi_30d"] = np.nan
    df.loc[df["ticker"] == "NAN_SPREAD", "atm_spread_pct_30d"] = np.nan
    df.loc[df["ticker"] == "NAN_EVENTS", "earnings_events_count"] = np.nan
    out = apply_liquidity_filters(df)
    assert list(out["ticker"]) == ["AAA"]


# ---------------------------------------------------------------------------
# Top-N truncation
# ---------------------------------------------------------------------------


def test_top_n_truncates_to_50_when_60_pass():
    rows = [_passing_row(f"P{i:03d}", adv=10_000 + i * 100) for i in range(60)]
    out = build_universe(pd.DataFrame(rows))
    assert len(out) == 50
    advs = out["options_adv_30d"].tolist()
    assert advs == sorted(advs, reverse=True)
    assert list(out["rank"]) == list(range(1, 51))
    # The 50 returned should be the top 50 ADV values from the 60 inputs.
    expected_top_advs = sorted([10_000 + i * 100 for i in range(60)], reverse=True)[:50]
    assert advs == expected_top_advs


def test_top_n_returns_all_when_fewer_than_50_pass():
    rows = [_passing_row(f"P{i:03d}", adv=10_000 + i * 100) for i in range(30)]
    out = build_universe(pd.DataFrame(rows))
    assert len(out) == 30
    assert list(out["rank"]) == list(range(1, 31))


def test_select_top_n_with_custom_filter_n():
    rows = [_passing_row(f"P{i:03d}", adv=10_000 + i) for i in range(20)]
    out = select_top_n(pd.DataFrame(rows), UniverseFilters(top_n_by_adv=5))
    assert len(out) == 5
    assert list(out["rank"]) == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# build_universe end-to-end
# ---------------------------------------------------------------------------


def test_build_universe_end_to_end_mixed_pass_fail():
    """100 rows: half engineered to pass, half to fail; expect top-50 of the passing half."""
    passing = [_passing_row(f"P{i:02d}", adv=20_000 + i * 100) for i in range(50)]
    failing = []
    for i in range(50):
        row = _passing_row(f"F{i:02d}", adv=80_000)  # high ADV so they would dominate if not filtered
        # Vary which gate fails so the fail-bucket is heterogeneous.
        if i % 4 == 0:
            row["options_adv_30d"] = 1_000
        elif i % 4 == 1:
            row["options_oi_30d"] = 5_000
        elif i % 4 == 2:
            row["atm_spread_pct_30d"] = 0.20
        else:
            row["earnings_events_count"] = 2
        failing.append(row)

    df = pd.DataFrame(passing + failing)
    out = build_universe(df)
    assert len(out) == 50
    # Every returned ticker must come from the passing set.
    assert set(out["ticker"]).issubset({r["ticker"] for r in passing})
    # Sorted descending by ADV.
    advs = out["options_adv_30d"].tolist()
    assert advs == sorted(advs, reverse=True)


def test_build_universe_includes_as_of_when_provided():
    df = pd.DataFrame([_passing_row("AAA"), _passing_row("BBB", adv=40_000)])
    snapshot = date(2026, 5, 6)
    out = build_universe(df, as_of=snapshot)
    assert "as_of" in out.columns
    assert (out["as_of"] == snapshot).all()


def test_build_universe_omits_as_of_when_not_provided():
    df = pd.DataFrame([_passing_row("AAA")])
    out = build_universe(df)
    assert "as_of" not in out.columns


def test_build_universe_output_columns_in_documented_order():
    df = pd.DataFrame([_passing_row("AAA")])
    out = build_universe(df, as_of=date(2026, 5, 6))
    assert list(out.columns) == [
        "ticker",
        "options_adv_30d",
        "options_oi_30d",
        "atm_spread_pct_30d",
        "earnings_events_count",
        "rank",
        "as_of",
    ]


def test_build_universe_on_synthetic_universe_does_not_explode():
    """Smoke: random 100-row synthetic frame should run cleanly and respect cap."""
    df = _make_candidates(100)
    out = build_universe(df)
    assert len(out) <= 50
    if len(out) > 0:
        advs = out["options_adv_30d"].tolist()
        assert advs == sorted(advs, reverse=True)


# ---------------------------------------------------------------------------
# explain_exclusion
# ---------------------------------------------------------------------------


def test_explain_exclusion_eligible_row_returns_empty():
    row = pd.Series(_passing_row("AAA"))
    assert explain_exclusion(row) == []


def test_explain_exclusion_lists_all_failing_gates():
    row = pd.Series(
        {
            "ticker": "BAD",
            "options_adv_30d": 1_000,        # fails min_options_adv
            "options_oi_30d": 5_000,         # fails min_options_oi
            "atm_spread_pct_30d": 0.04,      # passes
            "earnings_events_count": 12,     # passes
        }
    )
    failed = explain_exclusion(row)
    assert failed == ["min_options_adv", "min_options_oi"]


def test_explain_exclusion_all_gates_fail():
    row = pd.Series(
        {
            "ticker": "ALLBAD",
            "options_adv_30d": 100,
            "options_oi_30d": 100,
            "atm_spread_pct_30d": 0.50,
            "earnings_events_count": 0,
        }
    )
    failed = explain_exclusion(row)
    assert failed == [
        "min_options_adv",
        "min_options_oi",
        "max_atm_spread_pct",
        "min_historical_events",
    ]


def test_explain_exclusion_handles_nan():
    row = pd.Series(
        {
            "ticker": "NANROW",
            "options_adv_30d": np.nan,
            "options_oi_30d": 100_000,
            "atm_spread_pct_30d": 0.04,
            "earnings_events_count": 12,
        }
    )
    failed = explain_exclusion(row)
    assert "min_options_adv" in failed


def test_explain_exclusion_raises_on_missing_field():
    row = pd.Series({"ticker": "X", "options_adv_30d": 10_000})
    with pytest.raises(ValueError):
        explain_exclusion(row)


# ---------------------------------------------------------------------------
# Configuration sanity
# ---------------------------------------------------------------------------


def test_required_columns_constant_matches_dataclass_assumptions():
    """Defensive: REQUIRED_COLUMNS should never silently drift from the gate columns."""
    assert "ticker" in REQUIRED_COLUMNS
    for col in (
        "options_adv_30d",
        "options_oi_30d",
        "atm_spread_pct_30d",
        "earnings_events_count",
    ):
        assert col in REQUIRED_COLUMNS


def test_universe_filters_defaults_match_strategic_lock():
    f = UniverseFilters()
    assert f.min_options_adv == 5_000
    assert f.min_options_oi == 10_000
    assert f.max_atm_spread_pct == 0.08
    assert f.min_historical_events == 8
    assert f.top_n_by_adv == 50


def test_select_top_n_rejects_non_positive_n():
    df = pd.DataFrame([_passing_row("A")])
    with pytest.raises(ValueError):
        select_top_n(df, UniverseFilters(top_n_by_adv=0))
