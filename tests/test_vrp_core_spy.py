from __future__ import annotations

import math
from datetime import date, timedelta

import pandas as pd
import pytest

from features.greeks import bs_delta, bs_price
from signals.vrp_core_spy import (
    IronCondorEntry,
    IronCondorSpec,
    compute_entry_credit,
    construct_iron_condor,
    evaluate_exit,
    find_short_strikes_by_delta,
    position_count_check,
    select_expiry,
)


# ---------------------------------------------------------------------------
# Synthetic chain helpers
# ---------------------------------------------------------------------------


def _bs_chain(
    spot: float,
    strikes: list[float],
    T: float,
    r: float = 0.04,
    sigma: float = 0.20,
    q: float = 0.0,
    spread_bps_of_mid: float = 5.0,
) -> pd.DataFrame:
    """Build a (call, put) BS-priced chain with deltas. Bid/ask are mid +/- (spread_bps_of_mid bps)."""
    rows: list[dict] = []
    for K in strikes:
        for ot in ("call", "put"):
            mid = bs_price(ot, spot, K, T, r, sigma, q)
            d = bs_delta(ot, spot, K, T, r, sigma, q)
            half_spread = max(mid * spread_bps_of_mid * 1e-4, 0.005)
            bid = max(mid - half_spread, 0.0)
            ask = mid + half_spread
            rows.append({"strike": K, "option_type": ot, "delta": d, "bid": bid, "ask": ask})
    return pd.DataFrame(rows)


def _hardcoded_chain() -> pd.DataFrame:
    """Tiny hand-built chain with known deltas for direct strike-selection tests.
    Calls: deltas decrease with rising strike. Puts: |delta| decreases with falling strike.
    """
    rows = [
        # calls
        {"strike": 400.0, "option_type": "call", "delta": 0.50, "bid": 5.00, "ask": 5.10},
        {"strike": 410.0, "option_type": "call", "delta": 0.30, "bid": 2.50, "ask": 2.60},
        {"strike": 420.0, "option_type": "call", "delta": 0.12, "bid": 1.00, "ask": 1.10},
        {"strike": 425.0, "option_type": "call", "delta": 0.07, "bid": 0.50, "ask": 0.55},
        {"strike": 430.0, "option_type": "call", "delta": 0.04, "bid": 0.20, "ask": 0.25},
        # puts
        {"strike": 400.0, "option_type": "put", "delta": -0.50, "bid": 5.00, "ask": 5.10},
        {"strike": 390.0, "option_type": "put", "delta": -0.30, "bid": 2.50, "ask": 2.60},
        {"strike": 380.0, "option_type": "put", "delta": -0.12, "bid": 1.00, "ask": 1.10},
        {"strike": 375.0, "option_type": "put", "delta": -0.07, "bid": 0.50, "ask": 0.55},
        {"strike": 370.0, "option_type": "put", "delta": -0.04, "bid": 0.20, "ask": 0.25},
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# select_expiry
# ---------------------------------------------------------------------------


def test_select_expiry_picks_midpoint():
    today = date(2026, 5, 6)
    expiries = [today + timedelta(days=d) for d in (15, 25, 35, 50, 90)]
    picked = select_expiry(expiries, today)
    # 35 DTE is the in-window candidate; midpoint of 30..45 is 37.5, so 35 (|diff|=2.5) wins.
    assert (picked - today).days == 35


def test_select_expiry_ties_prefer_longer():
    today = date(2026, 5, 6)
    # 30 and 45 are both equidistant from midpoint 37.5; we expect longer (45) to win on tie.
    expiries = [today + timedelta(days=30), today + timedelta(days=45)]
    picked = select_expiry(expiries, today)
    assert (picked - today).days == 45


def test_select_expiry_picks_closest_to_midpoint():
    today = date(2026, 5, 6)
    # 32 (|diff|=5.5) vs 41 (|diff|=3.5): 41 wins.
    expiries = [today + timedelta(days=32), today + timedelta(days=41)]
    picked = select_expiry(expiries, today)
    assert (picked - today).days == 41


def test_select_expiry_raises_when_none_in_range():
    today = date(2026, 5, 6)
    expiries = [today + timedelta(days=10), today + timedelta(days=20), today + timedelta(days=60)]
    with pytest.raises(ValueError, match="no expiry"):
        select_expiry(expiries, today)


def test_select_expiry_invalid_window_raises():
    today = date(2026, 5, 6)
    with pytest.raises(ValueError, match="invalid DTE window"):
        select_expiry([today + timedelta(days=30)], today, target_dte_min=45, target_dte_max=30)


# ---------------------------------------------------------------------------
# find_short_strikes_by_delta
# ---------------------------------------------------------------------------


def test_find_short_strikes_hardcoded():
    chain = _hardcoded_chain()
    sc, sp = find_short_strikes_by_delta(chain, target_short_delta=0.12, delta_tolerance=0.03)
    # 12-delta call sits at 420; 12-delta put sits at 380.
    assert sc == 420.0
    assert sp == 380.0


def test_find_short_strikes_picks_closest_within_tolerance():
    chain = _hardcoded_chain()
    # Target 0.10: 0.12-delta is closest (|diff|=0.02), within tolerance 0.03.
    sc, sp = find_short_strikes_by_delta(chain, target_short_delta=0.10, delta_tolerance=0.03)
    assert sc == 420.0
    assert sp == 380.0


def test_find_short_strikes_raises_when_no_call_in_tolerance():
    chain = _hardcoded_chain()
    # Target 0.20: closest call is 0.12 (|diff|=0.08), outside tolerance 0.02.
    with pytest.raises(ValueError, match="no call within delta_tolerance"):
        find_short_strikes_by_delta(chain, target_short_delta=0.20, delta_tolerance=0.02)


def test_find_short_strikes_raises_when_no_put_in_tolerance():
    # Build a chain where calls cover delta 0.20 well but puts don't.
    rows = [
        {"strike": 100.0, "option_type": "call", "delta": 0.20, "bid": 1.0, "ask": 1.1},
        {"strike": 95.0, "option_type": "put", "delta": -0.50, "bid": 1.0, "ask": 1.1},
    ]
    chain = pd.DataFrame(rows)
    with pytest.raises(ValueError, match="no put within delta_tolerance"):
        find_short_strikes_by_delta(chain, target_short_delta=0.20, delta_tolerance=0.05)


def test_find_short_strikes_missing_columns_raises():
    bad = pd.DataFrame({"strike": [100.0], "option_type": ["call"]})  # no 'delta'
    with pytest.raises(ValueError, match="missing required columns"):
        find_short_strikes_by_delta(bad)


def test_find_short_strikes_invalid_target_raises():
    chain = _hardcoded_chain()
    with pytest.raises(ValueError, match="target_short_delta must be positive"):
        find_short_strikes_by_delta(chain, target_short_delta=0.0)
    with pytest.raises(ValueError, match="delta_tolerance must be non-negative"):
        find_short_strikes_by_delta(chain, delta_tolerance=-0.01)


# ---------------------------------------------------------------------------
# IronCondorSpec properties + construct_iron_condor
# ---------------------------------------------------------------------------


def test_iron_condor_spec_widths_and_max_loss():
    spec = IronCondorSpec(
        underlying="SPY",
        expiry=date(2026, 6, 19),
        short_call_strike=420.0,
        long_call_strike=425.0,
        short_put_strike=380.0,
        long_put_strike=375.0,
    )
    assert spec.call_wing_width == 5.0
    assert spec.put_wing_width == 5.0
    # max_loss = max(5, 5) * 1 * 100 = 500
    assert spec.max_loss == 500.0
    assert spec.is_valid()


def test_iron_condor_spec_asymmetric_widths():
    spec = IronCondorSpec(
        underlying="SPY",
        expiry=date(2026, 6, 19),
        short_call_strike=420.0,
        long_call_strike=427.0,  # 7-wide call wing
        short_put_strike=380.0,
        long_put_strike=375.0,   # 5-wide put wing
        quantity=2,
    )
    assert spec.call_wing_width == 7.0
    assert spec.put_wing_width == 5.0
    # max_loss takes the wider wing: max(7, 5) * 2 * 100 = 1400
    assert spec.max_loss == 1400.0
    assert spec.is_valid()


def test_iron_condor_spec_invalid_when_call_wing_inverted():
    spec = IronCondorSpec(
        underlying="SPY",
        expiry=date(2026, 6, 19),
        short_call_strike=425.0,   # short above long => inverted call wing
        long_call_strike=420.0,
        short_put_strike=380.0,
        long_put_strike=375.0,
    )
    assert not spec.is_valid()


def test_iron_condor_spec_invalid_when_put_wing_inverted():
    spec = IronCondorSpec(
        underlying="SPY",
        expiry=date(2026, 6, 19),
        short_call_strike=420.0,
        long_call_strike=425.0,
        short_put_strike=375.0,    # short below long => inverted put wing
        long_put_strike=380.0,
    )
    assert not spec.is_valid()


def test_iron_condor_spec_invalid_when_body_crossed():
    # short_put >= short_call: condor is crossed.
    spec = IronCondorSpec(
        underlying="SPY",
        expiry=date(2026, 6, 19),
        short_call_strike=380.0,
        long_call_strike=385.0,
        short_put_strike=420.0,
        long_put_strike=415.0,
    )
    assert not spec.is_valid()


def test_iron_condor_spec_invalid_when_zero_quantity():
    spec = IronCondorSpec(
        underlying="SPY",
        expiry=date(2026, 6, 19),
        short_call_strike=420.0,
        long_call_strike=425.0,
        short_put_strike=380.0,
        long_put_strike=375.0,
        quantity=0,
    )
    assert not spec.is_valid()


def test_construct_iron_condor_basic():
    chain = _hardcoded_chain()
    spec = construct_iron_condor(
        chain,
        underlying="SPY",
        expiry=date(2026, 6, 19),
        target_short_delta=0.12,
        wing_width=5.0,
    )
    assert spec.underlying == "SPY"
    assert spec.short_call_strike == 420.0
    assert spec.long_call_strike == 425.0
    assert spec.short_put_strike == 380.0
    assert spec.long_put_strike == 375.0
    assert spec.short_call_strike < spec.long_call_strike
    assert spec.short_put_strike > spec.long_put_strike
    assert spec.is_valid()


def test_construct_iron_condor_uses_bs_chain():
    """End-to-end on a BS-derived chain — sanity-check that a 30-delta-ish target picks reasonable strikes."""
    spot = 420.0
    strikes = [float(k) for k in range(380, 461, 5)]
    T = 35.0 / 365.0
    chain = _bs_chain(spot=spot, strikes=strikes, T=T)
    spec = construct_iron_condor(
        chain,
        underlying="SPY",
        expiry=date(2026, 5, 6) + timedelta(days=35),
        target_short_delta=0.15,
        wing_width=5.0,
    )
    # Short strikes should be OTM relative to spot.
    assert spec.short_call_strike > spot
    assert spec.short_put_strike < spot
    # Wings should sit further OTM than shorts.
    assert spec.long_call_strike == spec.short_call_strike + 5.0
    assert spec.long_put_strike == spec.short_put_strike - 5.0
    assert spec.is_valid()


def test_construct_iron_condor_zero_wing_width_raises():
    chain = _hardcoded_chain()
    with pytest.raises(ValueError, match="wing_width must be positive"):
        construct_iron_condor(
            chain, underlying="SPY", expiry=date(2026, 6, 19), wing_width=0.0
        )


# ---------------------------------------------------------------------------
# compute_entry_credit
# ---------------------------------------------------------------------------


def test_compute_entry_credit_hardcoded():
    chain = _hardcoded_chain()
    spec = IronCondorSpec(
        underlying="SPY",
        expiry=date(2026, 6, 19),
        short_call_strike=420.0,
        long_call_strike=425.0,
        short_put_strike=380.0,
        long_put_strike=375.0,
    )
    credit = compute_entry_credit(chain, spec)
    # mids: short_call=1.05, long_call=0.525, short_put=1.05, long_put=0.525
    # credit = 1.05 + 1.05 - 0.525 - 0.525 = 1.05
    assert math.isclose(credit, 1.05, abs_tol=1e-9)


def test_compute_entry_credit_positive_on_normal_chain():
    """On a standard BS chain, selling closer-to-money and buying further-OTM yields positive credit."""
    spot = 420.0
    strikes = [float(k) for k in range(380, 461, 5)]
    T = 35.0 / 365.0
    chain = _bs_chain(spot=spot, strikes=strikes, T=T)
    spec = construct_iron_condor(
        chain,
        underlying="SPY",
        expiry=date(2026, 5, 6) + timedelta(days=35),
        target_short_delta=0.15,
        wing_width=5.0,
    )
    credit = compute_entry_credit(chain, spec)
    assert credit > 0


def test_compute_entry_credit_missing_strike_raises():
    chain = _hardcoded_chain()
    spec = IronCondorSpec(
        underlying="SPY",
        expiry=date(2026, 6, 19),
        short_call_strike=420.0,
        long_call_strike=999.0,    # not in chain
        short_put_strike=380.0,
        long_put_strike=375.0,
    )
    with pytest.raises(ValueError, match="no call at strike"):
        compute_entry_credit(chain, spec)


def test_compute_entry_credit_missing_columns_raises():
    bad = pd.DataFrame({"strike": [400.0], "option_type": ["call"]})  # no bid/ask
    spec = IronCondorSpec(
        underlying="SPY",
        expiry=date(2026, 6, 19),
        short_call_strike=420.0,
        long_call_strike=425.0,
        short_put_strike=380.0,
        long_put_strike=375.0,
    )
    with pytest.raises(ValueError, match="missing required columns"):
        compute_entry_credit(bad, spec)


# ---------------------------------------------------------------------------
# evaluate_exit
# ---------------------------------------------------------------------------


def _entry_at_credit(credit_per_contract: float, quantity: int = 1) -> IronCondorEntry:
    spec = IronCondorSpec(
        underlying="SPY",
        expiry=date(2026, 6, 19),
        short_call_strike=420.0,
        long_call_strike=425.0,
        short_put_strike=380.0,
        long_put_strike=375.0,
        quantity=quantity,
    )
    total_credit = credit_per_contract * quantity * spec.multiplier
    return IronCondorEntry(
        spec=spec,
        entry_credit=credit_per_contract,
        entry_at=date(2026, 5, 6),
        short_call_delta=0.12,
        short_put_delta=-0.12,
        expected_pnl_50pct_target=0.50 * total_credit,
    )


def test_evaluate_exit_profit_target():
    entry = _entry_at_credit(credit_per_contract=1.05)  # total = $105
    # Need PnL >= 0.50 * 105 = 52.50
    decision = evaluate_exit(entry, today=date(2026, 5, 20), current_pnl=53.0, days_to_expiry=30)
    assert decision.should_exit is True
    assert decision.reason == "profit_target"
    assert decision.target_pnl == pytest.approx(52.50)


def test_evaluate_exit_profit_target_at_exact_threshold():
    entry = _entry_at_credit(credit_per_contract=1.00)  # total = $100; target = $50
    decision = evaluate_exit(entry, today=date(2026, 5, 20), current_pnl=50.0, days_to_expiry=30)
    assert decision.should_exit is True
    assert decision.reason == "profit_target"


def test_evaluate_exit_stop_loss():
    entry = _entry_at_credit(credit_per_contract=1.05)  # total = $105; stop = -$210
    decision = evaluate_exit(entry, today=date(2026, 5, 20), current_pnl=-220.0, days_to_expiry=30)
    assert decision.should_exit is True
    assert decision.reason == "stop_loss"
    assert decision.target_pnl == pytest.approx(-210.0)


def test_evaluate_exit_stop_loss_at_exact_threshold():
    entry = _entry_at_credit(credit_per_contract=1.00)  # total = $100; stop = -$200
    decision = evaluate_exit(entry, today=date(2026, 5, 20), current_pnl=-200.0, days_to_expiry=30)
    assert decision.should_exit is True
    assert decision.reason == "stop_loss"


def test_evaluate_exit_21dte_close():
    entry = _entry_at_credit(credit_per_contract=1.05)
    # PnL between stop (-210) and target (52.5), DTE <= 21 => 21dte_close.
    decision = evaluate_exit(entry, today=date(2026, 5, 20), current_pnl=10.0, days_to_expiry=21)
    assert decision.should_exit is True
    assert decision.reason == "21dte_close"


def test_evaluate_exit_21dte_does_not_override_profit_target():
    entry = _entry_at_credit(credit_per_contract=1.05)
    # PnL above target AND DTE <=21: profit_target wins.
    decision = evaluate_exit(entry, today=date(2026, 5, 20), current_pnl=80.0, days_to_expiry=15)
    assert decision.reason == "profit_target"


def test_evaluate_exit_21dte_does_not_override_stop_loss():
    entry = _entry_at_credit(credit_per_contract=1.05)
    # PnL below stop AND DTE <=21: stop_loss wins.
    decision = evaluate_exit(entry, today=date(2026, 5, 20), current_pnl=-300.0, days_to_expiry=15)
    assert decision.reason == "stop_loss"


def test_evaluate_exit_hold():
    entry = _entry_at_credit(credit_per_contract=1.05)
    # PnL in [stop, target), DTE > 21 => hold.
    decision = evaluate_exit(entry, today=date(2026, 5, 20), current_pnl=10.0, days_to_expiry=30)
    assert decision.should_exit is False
    assert decision.reason == "hold"
    assert decision.target_pnl == pytest.approx(52.50)


def test_evaluate_exit_scales_with_quantity():
    # quantity=2 doubles total credit, hence doubles thresholds.
    entry = _entry_at_credit(credit_per_contract=1.00, quantity=2)  # total = $200; target = $100
    # PnL = $80 with quantity=1 would trigger target ($50); with quantity=2 it should NOT.
    decision = evaluate_exit(entry, today=date(2026, 5, 20), current_pnl=80.0, days_to_expiry=30)
    assert decision.reason == "hold"
    decision2 = evaluate_exit(entry, today=date(2026, 5, 20), current_pnl=120.0, days_to_expiry=30)
    assert decision2.reason == "profit_target"


# ---------------------------------------------------------------------------
# position_count_check
# ---------------------------------------------------------------------------


def test_position_count_check_under_limit():
    assert position_count_check(open_positions=3, max_simultaneous=6) is True


def test_position_count_check_at_limit():
    assert position_count_check(open_positions=6, max_simultaneous=6) is False


def test_position_count_check_over_limit():
    assert position_count_check(open_positions=7, max_simultaneous=6) is False


def test_position_count_check_zero_positions():
    assert position_count_check(open_positions=0, max_simultaneous=6) is True


def test_position_count_check_invalid_inputs():
    with pytest.raises(ValueError, match="open_positions must be non-negative"):
        position_count_check(open_positions=-1)
    with pytest.raises(ValueError, match="max_simultaneous must be positive"):
        position_count_check(open_positions=0, max_simultaneous=0)
