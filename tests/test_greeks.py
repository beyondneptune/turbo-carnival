from __future__ import annotations

import math

import pytest
import QuantLib as ql

from features.greeks import (
    Greeks,
    Leg,
    aggregate_position_greeks,
    bs_delta,
    bs_gamma,
    bs_price,
    bs_rho,
    bs_theta,
    bs_vega,
    compute_greeks,
    position_dollar_gamma,
    vega_to_vol_point,
)


# ---------------------------------------------------------------------------
# Sanity checks: directionality and bounds
# ---------------------------------------------------------------------------


def test_atm_call_delta_in_range():
    d = bs_delta("call", 100.0, 100.0, 0.25, 0.05, 0.2)
    assert 0.5 < d < 0.6


def test_atm_put_delta_in_range():
    d = bs_delta("put", 100.0, 100.0, 0.25, 0.05, 0.2)
    assert -0.6 < d < -0.4


def test_deep_itm_call_delta_near_one():
    d = bs_delta("call", 200.0, 100.0, 0.25, 0.05, 0.2)
    assert abs(d - 1.0) < 0.01


def test_deep_otm_call_delta_near_zero():
    d = bs_delta("call", 50.0, 100.0, 0.25, 0.05, 0.2)
    assert d < 0.01
    assert d >= 0.0


def test_deep_itm_put_delta_near_minus_one():
    d = bs_delta("put", 50.0, 100.0, 0.25, 0.05, 0.2)
    assert abs(d - (-1.0)) < 0.01


def test_deep_otm_put_delta_near_zero():
    d = bs_delta("put", 200.0, 100.0, 0.25, 0.05, 0.2)
    assert -0.01 < d <= 0.0


def test_gamma_positive_for_call_and_put():
    g_call = bs_gamma(100.0, 100.0, 0.25, 0.05, 0.2)
    assert g_call > 0
    # gamma is identical for call and put under BS — verify scalar agreement
    g_via_call_compute = compute_greeks("call", 100.0, 100.0, 0.25, 0.05, 0.2).gamma
    g_via_put_compute = compute_greeks("put", 100.0, 100.0, 0.25, 0.05, 0.2).gamma
    assert g_via_call_compute > 0
    assert g_via_put_compute > 0
    assert math.isclose(g_via_call_compute, g_via_put_compute, rel_tol=1e-12)


def test_vega_positive_for_call_and_put():
    v = bs_vega(100.0, 100.0, 0.25, 0.05, 0.2)
    assert v > 0
    # vega is identical for call and put — sanity-check via compute_greeks
    v_call = compute_greeks("call", 100.0, 100.0, 0.25, 0.05, 0.2).vega
    v_put = compute_greeks("put", 100.0, 100.0, 0.25, 0.05, 0.2).vega
    assert math.isclose(v_call, v_put, rel_tol=1e-12)


def test_theta_negative_for_long_call_and_put():
    t_call = bs_theta("call", 100.0, 100.0, 0.25, 0.05, 0.2)
    t_put = bs_theta("put", 100.0, 100.0, 0.25, 0.05, 0.2)
    assert t_call < 0
    assert t_put < 0


# ---------------------------------------------------------------------------
# Put-call parity for delta
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q", [0.0, 0.02])
def test_put_call_parity_delta(q: float):
    S, K, T, r, sigma = 100.0, 105.0, 0.5, 0.04, 0.25
    d_c = bs_delta("call", S, K, T, r, sigma, q)
    d_p = bs_delta("put", S, K, T, r, sigma, q)
    expected = math.exp(-q * T)
    assert math.isclose(d_c - d_p, expected, abs_tol=1e-12)


@pytest.mark.parametrize("q", [0.0, 0.02])
def test_put_call_parity_price(q: float):
    """C - P = S e^(-qT) - K e^(-rT)"""
    S, K, T, r, sigma = 100.0, 105.0, 0.5, 0.04, 0.25
    c = bs_price("call", S, K, T, r, sigma, q)
    p = bs_price("put", S, K, T, r, sigma, q)
    expected = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert math.isclose(c - p, expected, abs_tol=1e-10)


# ---------------------------------------------------------------------------
# QuantLib cross-validation
# ---------------------------------------------------------------------------


def _ql_european_price(option_type: str, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today
    # Use ceil(T*365) days so floor() rounding doesn't shorten T relative to ours.
    # We deliberately use T = days/365 in test cases to get exact alignment with Actual365Fixed.
    expiry = today + int(round(T * 365))
    payoff = ql.PlainVanillaPayoff(ql.Option.Call if option_type == "call" else ql.Option.Put, K)
    exercise = ql.EuropeanExercise(expiry)
    option = ql.EuropeanOption(payoff, exercise)
    spot = ql.QuoteHandle(ql.SimpleQuote(S))
    rate = ql.YieldTermStructureHandle(ql.FlatForward(today, r, ql.Actual365Fixed()))
    div = ql.YieldTermStructureHandle(ql.FlatForward(today, q, ql.Actual365Fixed()))
    vol = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, ql.UnitedStates(ql.UnitedStates.NYSE), sigma, ql.Actual365Fixed())
    )
    process = ql.BlackScholesMertonProcess(spot, div, rate, vol)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    return option.NPV()


# Each case: (option_type, S, K, days_to_expiry, r, sigma, q)
# We use days/365 for T so QuantLib's Actual365Fixed day count aligns exactly with ours.
_QL_CASES = [
    ("call", 100.0, 100.0, 30, 0.05, 0.20, 0.0),    # ATM short-dated
    ("call", 100.0, 110.0, 90, 0.04, 0.25, 0.0),    # OTM
    ("put", 100.0, 90.0, 60, 0.03, 0.30, 0.0),      # OTM put
    ("call", 100.0, 100.0, 180, 0.05, 0.20, 0.02),  # with dividend yield
    ("put", 100.0, 100.0, 270, 0.05, 0.25, 0.02),   # long-dated put with div
]


@pytest.mark.parametrize("option_type,S,K,days,r,sigma,q", _QL_CASES)
def test_quantlib_price_cross_validation(option_type, S, K, days, r, sigma, q):
    T = days / 365.0
    mine = bs_price(option_type, S, K, T, r, sigma, q)
    theirs = _ql_european_price(option_type, S, K, T, r, sigma, q)
    # With T = days/365 and Actual365Fixed, both implementations agree to floating-point precision.
    assert abs(mine - theirs) < 1e-6, f"diff={abs(mine - theirs):.3e} mine={mine} ql={theirs}"


# ---------------------------------------------------------------------------
# compute_greeks bundles all five
# ---------------------------------------------------------------------------


def test_compute_greeks_returns_consistent_bundle():
    g = compute_greeks("call", 100.0, 100.0, 0.25, 0.05, 0.2)
    assert math.isclose(g.delta, bs_delta("call", 100.0, 100.0, 0.25, 0.05, 0.2), rel_tol=1e-15)
    assert math.isclose(g.gamma, bs_gamma(100.0, 100.0, 0.25, 0.05, 0.2), rel_tol=1e-15)
    assert math.isclose(g.theta, bs_theta("call", 100.0, 100.0, 0.25, 0.05, 0.2), rel_tol=1e-15)
    assert math.isclose(g.vega, bs_vega(100.0, 100.0, 0.25, 0.05, 0.2), rel_tol=1e-15)
    assert math.isclose(g.rho, bs_rho("call", 100.0, 100.0, 0.25, 0.05, 0.2), rel_tol=1e-15)


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        bs_delta("straddle", 100.0, 100.0, 0.25, 0.05, 0.2)  # type: ignore[arg-type]


def test_zero_sigma_raises():
    with pytest.raises(ValueError):
        bs_price("call", 100.0, 100.0, 0.25, 0.05, 0.0)


def test_zero_T_raises():
    with pytest.raises(ValueError):
        bs_price("call", 100.0, 100.0, 0.0, 0.05, 0.2)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_empty_legs_returns_all_zero():
    g = aggregate_position_greeks([])
    assert g == Greeks(0.0, 0.0, 0.0, 0.0, 0.0)


def test_long_stock_aggregate_delta():
    leg = Leg(instrument_type="stock", direction="long", quantity=10, multiplier=1, delta=1.0)
    g = aggregate_position_greeks([leg])
    assert g.delta == 10.0
    assert g.gamma == 0.0
    assert g.theta == 0.0
    assert g.vega == 0.0
    assert g.rho == 0.0


def test_short_call_aggregate_delta():
    leg = Leg(
        instrument_type="call",
        direction="short",
        quantity=1,
        multiplier=100,
        delta=0.5,
        gamma=0.04,
        theta=-10.0,
        vega=20.0,
        rho=13.0,
    )
    g = aggregate_position_greeks([leg])
    assert g.delta == -50.0
    assert g.gamma == -4.0
    assert g.theta == 1000.0
    assert g.vega == -2000.0
    assert g.rho == -1300.0


def test_iron_condor_delta_near_zero():
    # Symmetric body: long 95 put, short 100 put, short 100 call, long 105 call.
    # (Iron condor wings widen this slightly; here we use a balanced butterfly-like
    # construction to confirm the sign mechanics — body deltas cancel cleanly.)
    S, T, r, sigma = 100.0, 0.25, 0.05, 0.2
    short_put = bs_delta("put", S, 100.0, T, r, sigma)
    long_put = bs_delta("put", S, 95.0, T, r, sigma)
    short_call = bs_delta("call", S, 100.0, T, r, sigma)
    long_call = bs_delta("call", S, 105.0, T, r, sigma)
    legs = [
        Leg("put", "long", 1, 100, delta=long_put),
        Leg("put", "short", 1, 100, delta=short_put),
        Leg("call", "short", 1, 100, delta=short_call),
        Leg("call", "long", 1, 100, delta=long_call),
    ]
    g = aggregate_position_greeks(legs)
    # Symmetric construction around ATM: aggregate |delta| should be small relative
    # to a single naked option leg (~50 delta-shares).
    assert abs(g.delta) < 25.0


def test_long_atm_straddle_delta_near_zero_vega_doubled():
    S, K, T, r, sigma = 100.0, 100.0, 0.25, 0.05, 0.2
    g_call = compute_greeks("call", S, K, T, r, sigma)
    g_put = compute_greeks("put", S, K, T, r, sigma)
    legs = [
        Leg("call", "long", 1, 100, delta=g_call.delta, gamma=g_call.gamma, theta=g_call.theta, vega=g_call.vega, rho=g_call.rho),
        Leg("put", "long", 1, 100, delta=g_put.delta, gamma=g_put.gamma, theta=g_put.theta, vega=g_put.vega, rho=g_put.rho),
    ]
    g = aggregate_position_greeks(legs)
    # Per-share parity: delta_call - delta_put = 1; delta_call + delta_put = 2 delta_call - 1.
    # At ATM short-dated this is small (~0.14 per-share -> 14 per 1 contract), but should be << one naked leg.
    assert abs(g.delta) < 25.0
    expected_vega = 2 * 100 * g_call.vega
    assert math.isclose(g.vega, expected_vega, rel_tol=1e-12)
    # Long straddle: gamma > 0, theta < 0
    assert g.gamma > 0
    assert g.theta < 0


def test_aggregation_signs_via_long_short_offset():
    """Long 1 + short 1 of identical contracts => zero aggregate Greeks."""
    g_call = compute_greeks("call", 100.0, 100.0, 0.25, 0.05, 0.2)
    legs = [
        Leg("call", "long", 1, 100, delta=g_call.delta, gamma=g_call.gamma, theta=g_call.theta, vega=g_call.vega, rho=g_call.rho),
        Leg("call", "short", 1, 100, delta=g_call.delta, gamma=g_call.gamma, theta=g_call.theta, vega=g_call.vega, rho=g_call.rho),
    ]
    g = aggregate_position_greeks(legs)
    assert g.delta == 0.0
    assert g.gamma == 0.0
    assert g.theta == 0.0
    assert g.vega == 0.0
    assert g.rho == 0.0


def test_negative_quantity_rejected():
    leg = Leg(instrument_type="stock", direction="long", quantity=-1, multiplier=1, delta=1.0)
    with pytest.raises(ValueError):
        aggregate_position_greeks([leg])


def test_zero_multiplier_rejected():
    leg = Leg(instrument_type="call", direction="long", quantity=1, multiplier=0, delta=0.5)
    with pytest.raises(ValueError):
        aggregate_position_greeks([leg])


# ---------------------------------------------------------------------------
# Unit-conversion helpers for the risk/envelope.py boundary
# ---------------------------------------------------------------------------
# Consumers (risk.envelope.CandidatePosition) MUST populate `.vega` and
# `.gamma` via `vega_to_vol_point` / `position_dollar_gamma`, NOT raw
# `bs_vega` / `bs_gamma`. Wiring raw BS values would be 100x too large for
# vega and dimensionally wrong for gamma.


def test_vega_to_vol_point_basic_divide_by_100():
    assert vega_to_vol_point(19.5) == pytest.approx(0.195, abs=1e-12)


def test_vega_to_vol_point_zero():
    assert vega_to_vol_point(0.0) == 0.0


def test_vega_to_vol_point_negative():
    assert vega_to_vol_point(-50.0) == pytest.approx(-0.5, abs=1e-12)


def test_vega_to_vol_point_envelope_boundary_atm_call():
    """ATM call: bs_vega ~$19.5/sigma -> vega_to_vol_point ~$0.195/vol-pt.
    Multiplied by a 30-vol-point shock should give ~-$5.85 P&L when short.
    Mirrors the convention enforced by tests/test_envelope.py.
    """
    v_per_sigma = bs_vega(100.0, 100.0, 0.25, 0.05, 0.20)
    assert 19.0 < v_per_sigma < 20.5  # ~19.6 at these params
    v_per_vol_pt = vega_to_vol_point(v_per_sigma)
    assert 0.190 < v_per_vol_pt < 0.205
    # Short position: -1 * v_per_vol_pt * 30 vol points
    pnl = -v_per_vol_pt * 30.0
    assert -6.20 < pnl < -5.65  # ~-$5.85
    assert math.isclose(pnl, -v_per_sigma * 30.0 / 100.0, rel_tol=1e-12)


def test_position_dollar_gamma_short_call_hand_computation():
    """gamma=0.001, spot=100, qty=10, mult=100, sign=-1.
    Expected: -1 * 10 * 100 * 0.001 * 100 * 100 * 1e-4 = -1.0.
    """
    g = position_dollar_gamma(
        leg_gamma_per_dollar=0.001,
        spot=100.0,
        quantity=10,
        multiplier=100,
        direction_sign=-1,
    )
    assert g == pytest.approx(-1.0, abs=1e-12)


def test_position_dollar_gamma_long_sign_flips():
    """Sign convention: +1 long, -1 short -- result flips sign only."""
    long_g = position_dollar_gamma(0.001, 100.0, 10, 100, +1)
    short_g = position_dollar_gamma(0.001, 100.0, 10, 100, -1)
    assert long_g == pytest.approx(+1.0, abs=1e-12)
    assert short_g == pytest.approx(-1.0, abs=1e-12)
    assert long_g == -short_g


def test_position_dollar_gamma_zero_quantity():
    assert position_dollar_gamma(0.001, 100.0, 0, 100, +1) == 0.0


def test_position_dollar_gamma_invalid_sign_raises():
    with pytest.raises(ValueError):
        position_dollar_gamma(0.001, 100.0, 1, 100, 0)
    with pytest.raises(ValueError):
        position_dollar_gamma(0.001, 100.0, 1, 100, 2)


def test_position_dollar_gamma_negative_quantity_raises():
    with pytest.raises(ValueError):
        position_dollar_gamma(0.001, 100.0, -1, 100, +1)


def test_position_dollar_gamma_zero_multiplier_raises():
    with pytest.raises(ValueError):
        position_dollar_gamma(0.001, 100.0, 1, 0, +1)


def test_position_dollar_gamma_zero_spot_raises():
    with pytest.raises(ValueError):
        position_dollar_gamma(0.001, 0.0, 1, 100, +1)


def test_position_dollar_gamma_scales_quadratically_with_spot():
    """gamma * spot^2 scaling: doubling spot quadruples result."""
    g_at_100 = position_dollar_gamma(0.001, 100.0, 1, 100, +1)
    g_at_200 = position_dollar_gamma(0.001, 200.0, 1, 100, +1)
    assert math.isclose(g_at_200, 4.0 * g_at_100, rel_tol=1e-12)
