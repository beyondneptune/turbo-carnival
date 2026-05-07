"""Tests for signals/earnings_satellite.py — twin-model targets + position structuring."""
from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from signals.earnings_satellite import (
    EarningsEvent,
    StraddlePosition,
    build_targets,
    compute_expected_event_risk,
    compute_risk_floor,
    model_a_target,
    model_b_target,
    select_atm_strike,
    select_post_event_expiry,
    structure_straddle,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SQRT_5_OVER_252 = math.sqrt(5 / 252)


def _make_chain(strikes: list[float], expiry: date) -> pd.DataFrame:
    """Build a synthetic chain with deterministic mids: call_mid = max(spot - K, 0.5)
    + 1.0, put_mid = max(K - spot, 0.5) + 1.0. spot is implicit — for tests we just
    use simple linearly-increasing prices so we can verify ATM selection rather than
    pricing accuracy."""
    rows = []
    for i, K in enumerate(strikes):
        # Simple deterministic mids; bid=mid-0.05, ask=mid+0.05.
        call_mid = 2.0 + 0.10 * i
        put_mid = 2.5 - 0.05 * i
        rows.append(
            {
                "strike": K,
                "expiry": expiry,
                "option_type": "call",
                "bid": call_mid - 0.05,
                "ask": call_mid + 0.05,
            }
        )
        rows.append(
            {
                "strike": K,
                "expiry": expiry,
                "option_type": "put",
                "bid": put_mid - 0.05,
                "ask": put_mid + 0.05,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# select_atm_strike
# ---------------------------------------------------------------------------


def test_select_atm_strike_exact_match():
    assert select_atm_strike(100.0, [95.0, 100.0, 105.0]) == 100.0


def test_select_atm_strike_between_strikes_picks_closer():
    # 102 is closer to 100 than to 105 (diff 2 vs 3) → 100.
    assert select_atm_strike(102.0, [100.0, 105.0]) == 100.0


def test_select_atm_strike_ties_break_low():
    # 102.5 is equidistant from 100 and 105 → tie broken to lower.
    assert select_atm_strike(102.5, [100.0, 105.0]) == 100.0


def test_select_atm_strike_unsorted_input():
    # Non-monotone input must still produce the same answer.
    assert select_atm_strike(102.5, [105.0, 100.0]) == 100.0


def test_select_atm_strike_empty_raises():
    with pytest.raises(ValueError):
        select_atm_strike(100.0, [])


def test_select_atm_strike_non_positive_spot_raises():
    with pytest.raises(ValueError):
        select_atm_strike(0.0, [95.0, 100.0])


# ---------------------------------------------------------------------------
# select_post_event_expiry
# ---------------------------------------------------------------------------


def test_select_post_event_expiry_event_tuesday_picks_following_friday():
    # Event Tue 2026-05-05; weekly expiries are Fridays.
    # Earliest acceptable = Tue + 3 cal days = Fri 2026-05-08. That Friday DOES qualify.
    event = date(2026, 5, 5)  # Tue
    expiries = [
        date(2026, 5, 8),   # Fri  (3 days after Tue → eligible boundary)
        date(2026, 5, 15),
        date(2026, 5, 22),
    ]
    picked = select_post_event_expiry(event, expiries)
    assert picked == date(2026, 5, 8)


def test_select_post_event_expiry_event_friday_skips_to_next_friday():
    # Event Fri 2026-05-08 → earliest acceptable = Mon 2026-05-11.
    # The same-day Friday expiry is rejected (only 0 cal days). Next Friday wins.
    event = date(2026, 5, 8)
    expiries = [
        date(2026, 5, 8),   # same day — must be skipped
        date(2026, 5, 15),  # 7 cal days out — eligible
        date(2026, 5, 22),
    ]
    picked = select_post_event_expiry(event, expiries)
    assert picked == date(2026, 5, 15)


def test_select_post_event_expiry_event_thursday_picks_immediate_next_friday():
    # Event Thu 2026-05-07 → earliest acceptable = Sun 2026-05-10.
    # Friday 2026-05-08 is only 1 day out → rejected. Friday 2026-05-15 wins.
    event = date(2026, 5, 7)
    expiries = [
        date(2026, 5, 8),   # 1 day after — rejected
        date(2026, 5, 15),  # 8 days after — eligible
    ]
    picked = select_post_event_expiry(event, expiries)
    assert picked == date(2026, 5, 15)


def test_select_post_event_expiry_unsorted_input():
    event = date(2026, 5, 5)
    # Same as the Tuesday test but with the expiry list shuffled.
    expiries = [date(2026, 5, 22), date(2026, 5, 8), date(2026, 5, 15)]
    picked = select_post_event_expiry(event, expiries)
    assert picked == date(2026, 5, 8)


def test_select_post_event_expiry_no_eligible_raises():
    event = date(2026, 5, 5)
    expiries = [date(2026, 5, 5), date(2026, 5, 6)]  # both within 3 days
    with pytest.raises(ValueError):
        select_post_event_expiry(event, expiries)


def test_select_post_event_expiry_empty_list_raises():
    with pytest.raises(ValueError):
        select_post_event_expiry(date(2026, 5, 5), [])


def test_select_post_event_expiry_negative_min_days_raises():
    with pytest.raises(ValueError):
        select_post_event_expiry(
            date(2026, 5, 5), [date(2026, 5, 8)], min_calendar_days_after=-1
        )


# ---------------------------------------------------------------------------
# structure_straddle
# ---------------------------------------------------------------------------


def test_structure_straddle_picks_atm_and_sums_premium():
    expiry = date(2026, 5, 15)
    strikes = [95.0, 100.0, 105.0, 110.0]
    chain = _make_chain(strikes, expiry)

    event = EarningsEvent(
        ticker="ABC",
        event_date=date(2026, 5, 11),
        decision_date=date(2026, 5, 8),
        spot_at_decision=101.0,         # closest to 100
        nearest_expiry=expiry,
        pre_event_atm_iv_5d=0.30,
    )
    pos = structure_straddle(event, chain)

    assert isinstance(pos, StraddlePosition)
    assert pos.ticker == "ABC"
    assert pos.expiry == expiry
    assert pos.strike == 100.0
    # mids constructed deterministically: index 1 → call_mid 2.10, put_mid 2.45.
    assert pos.entry_call_price == pytest.approx(2.10, abs=1e-9)
    assert pos.entry_put_price == pytest.approx(2.45, abs=1e-9)
    assert pos.entry_total_premium == pytest.approx(4.55, abs=1e-9)
    assert pos.contracts == 1


def test_structure_straddle_filters_to_event_expiry():
    """A chain spanning two expiries must be filtered to event.nearest_expiry."""
    target_expiry = date(2026, 5, 15)
    other_expiry = date(2026, 5, 22)
    strikes = [95.0, 100.0, 105.0]
    chain_a = _make_chain(strikes, target_expiry)
    chain_b = _make_chain(strikes, other_expiry)
    chain = pd.concat([chain_a, chain_b], ignore_index=True)

    event = EarningsEvent(
        ticker="XYZ",
        event_date=date(2026, 5, 11),
        decision_date=date(2026, 5, 8),
        spot_at_decision=100.0,
        nearest_expiry=target_expiry,
        pre_event_atm_iv_5d=0.40,
    )
    pos = structure_straddle(event, chain)
    assert pos.expiry == target_expiry
    assert pos.strike == 100.0


def test_structure_straddle_accepts_precomputed_mid_column():
    """If the chain already has a 'mid' column, it should be used directly."""
    expiry = date(2026, 5, 15)
    chain = pd.DataFrame(
        [
            {"strike": 100.0, "expiry": expiry, "option_type": "call", "mid": 3.00},
            {"strike": 100.0, "expiry": expiry, "option_type": "put", "mid": 2.50},
        ]
    )
    event = EarningsEvent(
        ticker="MID",
        event_date=date(2026, 5, 11),
        decision_date=date(2026, 5, 8),
        spot_at_decision=100.0,
        nearest_expiry=expiry,
        pre_event_atm_iv_5d=0.25,
    )
    pos = structure_straddle(event, chain)
    assert pos.entry_call_price == 3.00
    assert pos.entry_put_price == 2.50
    assert pos.entry_total_premium == 5.50


def test_structure_straddle_missing_columns_raises():
    chain = pd.DataFrame([{"strike": 100.0, "option_type": "call", "bid": 1.0, "ask": 1.1}])
    event = EarningsEvent(
        ticker="X",
        event_date=date(2026, 5, 11),
        decision_date=date(2026, 5, 8),
        spot_at_decision=100.0,
        nearest_expiry=date(2026, 5, 15),
        pre_event_atm_iv_5d=0.30,
    )
    with pytest.raises(ValueError):
        structure_straddle(event, chain)


def test_structure_straddle_empty_filtered_chain_raises():
    expiry = date(2026, 5, 15)
    chain = _make_chain([100.0], date(2026, 5, 22))  # different expiry
    event = EarningsEvent(
        ticker="X",
        event_date=date(2026, 5, 11),
        decision_date=date(2026, 5, 8),
        spot_at_decision=100.0,
        nearest_expiry=expiry,
        pre_event_atm_iv_5d=0.30,
    )
    with pytest.raises(ValueError):
        structure_straddle(event, chain)


# ---------------------------------------------------------------------------
# compute_expected_event_risk
# ---------------------------------------------------------------------------


def test_compute_expected_event_risk_known_value():
    # iv=0.30 → 0.30 × √(5/252).
    expected = 0.30 * SQRT_5_OVER_252
    got = compute_expected_event_risk(0.30)
    assert got == pytest.approx(expected, abs=1e-12)
    # And approximately 0.0422 within 1e-4 (per spec).
    assert got == pytest.approx(0.0422, abs=1e-4)


def test_compute_expected_event_risk_zero_iv_returns_zero():
    assert compute_expected_event_risk(0.0) == 0.0


def test_compute_expected_event_risk_rejects_negative():
    with pytest.raises(ValueError):
        compute_expected_event_risk(-0.10)


def test_compute_expected_event_risk_rejects_nan():
    with pytest.raises(ValueError):
        compute_expected_event_risk(float("nan"))


# ---------------------------------------------------------------------------
# compute_risk_floor
# ---------------------------------------------------------------------------


def test_compute_risk_floor_known_series():
    # Series of 21 IVs from 0.10 to 0.50 in even 0.02 steps. 20th percentile of the
    # SCALED series is iv_p20 × √(5/252); pandas linear interp at q=0.20 over 21 points
    # picks the 4th point (index 4 = 0.18).
    ivs = pd.Series([round(0.10 + 0.02 * i, 4) for i in range(21)])
    floor = compute_risk_floor(ivs, quantile=0.20)
    expected = 0.18 * SQRT_5_OVER_252
    assert floor == pytest.approx(expected, abs=1e-9)


def test_compute_risk_floor_drops_nans():
    ivs = pd.Series([0.10, np.nan, 0.20, 0.30, 0.40])
    floor = compute_risk_floor(ivs, quantile=0.25)
    # After NaN drop: [0.10, 0.20, 0.30, 0.40]; 25th percentile → 0.175 (linear interp).
    expected = 0.175 * SQRT_5_OVER_252
    assert floor == pytest.approx(expected, abs=1e-9)


def test_compute_risk_floor_empty_series_raises():
    with pytest.raises(ValueError):
        compute_risk_floor(pd.Series([], dtype=float))


def test_compute_risk_floor_negative_iv_raises():
    with pytest.raises(ValueError):
        compute_risk_floor(pd.Series([0.10, -0.05, 0.30]))


def test_compute_risk_floor_quantile_out_of_range_raises():
    with pytest.raises(ValueError):
        compute_risk_floor(pd.Series([0.10, 0.20]), quantile=0.0)
    with pytest.raises(ValueError):
        compute_risk_floor(pd.Series([0.10, 0.20]), quantile=1.0)


# ---------------------------------------------------------------------------
# model_a_target / model_b_target
# ---------------------------------------------------------------------------


def test_model_a_target_is_identity():
    assert model_a_target(0.0) == 0.0
    assert model_a_target(0.1234) == 0.1234
    assert model_a_target(-0.05) == -0.05


def test_model_a_target_rejects_nan():
    with pytest.raises(ValueError):
        model_a_target(float("nan"))


def test_model_b_target_floor_does_not_bind():
    # r=0.10, iv=0.30, risk_floor=0.02. expected_event_risk ≈ 0.0422 > 0.02.
    # denominator = 0.0422; target ≈ 0.10/0.0422 ≈ 2.37.
    target = model_b_target(0.10, 0.30, 0.02)
    expected = 0.10 / (0.30 * SQRT_5_OVER_252)
    assert target == pytest.approx(expected, abs=1e-9)
    assert target == pytest.approx(2.37, abs=1e-2)


def test_model_b_target_floor_binds():
    # r=0.10, iv=0.05 → expected_event_risk ≈ 0.0070 < 0.02; denominator = 0.02; target = 5.0.
    target = model_b_target(0.10, 0.05, 0.02)
    assert target == pytest.approx(5.0, abs=1e-9)


def test_model_b_target_zero_return():
    assert model_b_target(0.0, 0.30, 0.02) == 0.0


def test_model_b_target_rejects_zero_floor():
    with pytest.raises(ValueError):
        model_b_target(0.10, 0.30, 0.0)


def test_model_b_target_rejects_nan_return():
    with pytest.raises(ValueError):
        model_b_target(float("nan"), 0.30, 0.02)


# ---------------------------------------------------------------------------
# build_targets
# ---------------------------------------------------------------------------


def _make_events(n: int) -> pd.DataFrame:
    """20 synthetic events with deterministic IVs and returns."""
    rng = np.random.default_rng(seed=42)
    ivs = np.linspace(0.10, 0.60, n)             # spans the IV space
    rets = rng.normal(loc=0.02, scale=0.10, size=n)  # synthetic returns
    return pd.DataFrame(
        {
            "ticker": [f"T{i}" for i in range(n)],
            "event_date": [date(2026, 5, 1) + timedelta(days=i) for i in range(n)],
            "pre_event_atm_iv_5d": ivs,
            "realized_5d_straddle_return": rets,
        }
    )


def test_build_targets_end_to_end():
    events = _make_events(20)
    out = build_targets(events)

    # All expected columns present.
    for col in ("target_a", "target_b", "expected_event_risk", "risk_floor_used"):
        assert col in out.columns
    assert len(out) == 20

    # target_a is identity.
    np.testing.assert_allclose(
        out["target_a"].to_numpy(),
        events["realized_5d_straddle_return"].to_numpy(),
    )

    # expected_event_risk = iv × √(5/252).
    expected_risk = events["pre_event_atm_iv_5d"].to_numpy() * SQRT_5_OVER_252
    np.testing.assert_allclose(out["expected_event_risk"].to_numpy(), expected_risk)

    # risk_floor_used is broadcast (same value across every row).
    floor_vals = out["risk_floor_used"].unique()
    assert len(floor_vals) == 1
    floor = float(floor_vals[0])
    assert floor > 0

    # And it equals the 20th percentile of expected_risk.
    expected_floor = float(pd.Series(expected_risk).quantile(0.20))
    assert floor == pytest.approx(expected_floor, abs=1e-12)

    # target_b matches return / max(expected_risk, floor) row-by-row.
    denom = np.maximum(expected_risk, floor)
    expected_b = events["realized_5d_straddle_return"].to_numpy() / denom
    np.testing.assert_allclose(out["target_b"].to_numpy(), expected_b)


def test_build_targets_honours_provided_risk_floor():
    events = _make_events(20)
    out = build_targets(events, risk_floor=0.05)

    # The provided floor must be used verbatim — no recomputation.
    floor_vals = out["risk_floor_used"].unique()
    assert len(floor_vals) == 1
    assert float(floor_vals[0]) == pytest.approx(0.05, abs=1e-12)

    # And target_b reflects clipping at 0.05 wherever expected_event_risk < 0.05.
    expected_risk = events["pre_event_atm_iv_5d"].to_numpy() * SQRT_5_OVER_252
    denom = np.maximum(expected_risk, 0.05)
    expected_b = events["realized_5d_straddle_return"].to_numpy() / denom
    np.testing.assert_allclose(out["target_b"].to_numpy(), expected_b)


def test_build_targets_does_not_mutate_input():
    events = _make_events(5)
    snapshot = events.copy()
    build_targets(events)
    pd.testing.assert_frame_equal(events, snapshot)


def test_build_targets_missing_column_raises():
    events = _make_events(5).drop(columns=["pre_event_atm_iv_5d"])
    with pytest.raises(ValueError):
        build_targets(events)


def test_build_targets_empty_frame_raises():
    empty = pd.DataFrame(
        columns=[
            "ticker",
            "event_date",
            "pre_event_atm_iv_5d",
            "realized_5d_straddle_return",
        ]
    )
    with pytest.raises(ValueError):
        build_targets(empty)


def test_build_targets_rejects_negative_iv():
    events = _make_events(5)
    events.loc[2, "pre_event_atm_iv_5d"] = -0.10
    with pytest.raises(ValueError):
        build_targets(events)


def test_build_targets_rejects_provided_zero_floor():
    events = _make_events(5)
    with pytest.raises(ValueError):
        build_targets(events, risk_floor=0.0)
