from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from features.greeks import bs_price as _bs_price
from features.vol_surface import (
    SurfaceFit,
    fit_surface_from_chain,
    get_atm_term_structure,
    implied_vol,
    query_iv_by_delta,
    query_iv_by_strike,
)


# ---------------------------------------------------------------------------
# Synthetic chain helpers
# ---------------------------------------------------------------------------


def _synthetic_flat_chain(
    spot: float,
    vol: float,
    expiries_days: list[int],
    strikes: list[float],
    r: float = 0.05,
    q: float = 0.0,
) -> pd.DataFrame:
    """Build a chain where every option is priced at exactly the given vol via BS."""
    rows = []
    for exp_d in expiries_days:
        T = exp_d / 365.0
        for K in strikes:
            for ot in ("call", "put"):
                price = _bs_price(ot, spot, K, T, r, vol, q)
                rows.append(
                    {
                        "strike": float(K),
                        "expiry_days": int(exp_d),
                        "market_price": float(price),
                        "option_type": ot,
                    }
                )
    return pd.DataFrame(rows)


def _synthetic_skew_chain(
    spot: float,
    expiries_days: list[int],
    strikes: list[float],
    r: float = 0.05,
    q: float = 0.0,
    base_vol: float = 0.20,
    skew_slope: float = -0.5,
) -> pd.DataFrame:
    """Linear-in-log-moneyness skew: iv(K) = base_vol + skew_slope * ln(K/S).
    Negative slope => puts (low strikes) have higher IV (typical equity skew)."""
    rows = []
    for exp_d in expiries_days:
        T = exp_d / 365.0
        for K in strikes:
            iv = base_vol + skew_slope * math.log(K / spot)
            for ot in ("call", "put"):
                price = _bs_price(ot, spot, K, T, r, iv, q)
                rows.append(
                    {
                        "strike": float(K),
                        "expiry_days": int(exp_d),
                        "market_price": float(price),
                        "option_type": ot,
                    }
                )
    return pd.DataFrame(rows)


def _synthetic_term_structure_chain(
    spot: float,
    expiry_to_vol: dict[int, float],
    strikes: list[float],
    r: float = 0.05,
    q: float = 0.0,
) -> pd.DataFrame:
    """Each expiry has a flat (in strike) IV taken from expiry_to_vol."""
    rows = []
    for exp_d, vol in expiry_to_vol.items():
        T = exp_d / 365.0
        for K in strikes:
            for ot in ("call", "put"):
                price = _bs_price(ot, spot, K, T, r, vol, q)
                rows.append(
                    {
                        "strike": float(K),
                        "expiry_days": int(exp_d),
                        "market_price": float(price),
                        "option_type": ot,
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# IV solver round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "option_type,S,K,T,r,sigma",
    [
        ("call", 100.0, 100.0, 0.25, 0.05, 0.20),
        ("put", 100.0, 100.0, 0.25, 0.05, 0.20),
        ("call", 100.0, 110.0, 0.5, 0.04, 0.30),
        ("put", 100.0, 90.0, 0.5, 0.04, 0.30),
        ("call", 100.0, 100.0, 1.0, 0.03, 0.50),
        ("put", 100.0, 100.0, 0.1, 0.05, 0.15),
    ],
)
def test_iv_round_trip(option_type, S, K, T, r, sigma):
    price = _bs_price(option_type, S, K, T, r, sigma)
    iv = implied_vol(price, option_type, S, K, T, r)
    assert abs(iv - sigma) < 1e-5, f"recovered {iv} vs true {sigma}"


def test_iv_round_trip_with_dividend_yield():
    S, K, T, r, q, sigma = 100.0, 100.0, 0.5, 0.04, 0.02, 0.25
    price = _bs_price("call", S, K, T, r, sigma, q)
    iv = implied_vol(price, "call", S, K, T, r, q)
    assert abs(iv - sigma) < 1e-5


def test_iv_deep_itm_call():
    """Deep ITM call: vega is small but brentq should still recover vol."""
    S, K, T, r, sigma = 200.0, 100.0, 0.25, 0.05, 0.20
    price = _bs_price("call", S, K, T, r, sigma)
    iv = implied_vol(price, "call", S, K, T, r)
    assert abs(iv - sigma) < 1e-4


def test_iv_deep_otm_call():
    S, K, T, r, sigma = 50.0, 100.0, 0.25, 0.05, 0.40
    price = _bs_price("call", S, K, T, r, sigma)
    iv = implied_vol(price, "call", S, K, T, r)
    assert abs(iv - sigma) < 1e-4


def test_iv_deep_itm_put():
    S, K, T, r, sigma = 50.0, 100.0, 0.25, 0.05, 0.30
    price = _bs_price("put", S, K, T, r, sigma)
    iv = implied_vol(price, "put", S, K, T, r)
    assert abs(iv - sigma) < 1e-4


def test_iv_deep_otm_put():
    S, K, T, r, sigma = 200.0, 100.0, 0.25, 0.05, 0.40
    price = _bs_price("put", S, K, T, r, sigma)
    iv = implied_vol(price, "put", S, K, T, r)
    assert abs(iv - sigma) < 1e-4


def test_iv_negative_price_raises():
    with pytest.raises(ValueError):
        implied_vol(-1.0, "call", 100.0, 100.0, 0.25, 0.05)


def test_iv_invalid_option_type_raises():
    with pytest.raises(ValueError):
        implied_vol(5.0, "straddle", 100.0, 100.0, 0.25, 0.05)  # type: ignore[arg-type]


def test_iv_zero_T_raises():
    with pytest.raises(ValueError):
        implied_vol(5.0, "call", 100.0, 100.0, 0.0, 0.05)


def test_iv_below_intrinsic_raises():
    """Call price below S - K e^{-rT} violates no-arb; no positive IV solves it."""
    S, K, T, r = 100.0, 50.0, 0.25, 0.05
    intrinsic = S - K * math.exp(-r * T)
    with pytest.raises(ValueError):
        implied_vol(intrinsic - 1.0, "call", S, K, T, r)


# ---------------------------------------------------------------------------
# Surface fitting: flat surface
# ---------------------------------------------------------------------------


def test_flat_surface_recovers_vol():
    spot = 100.0
    flat_vol = 0.20
    expiries = [30, 60, 90]
    strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
    chain = _synthetic_flat_chain(spot, flat_vol, expiries, strikes)
    surface = fit_surface_from_chain(chain, spot=spot, risk_free_rate=0.05)

    # Every fitted IV in the surface points should be ~flat_vol.
    assert (surface.points["iv"].sub(flat_vol).abs() < 1e-4).all()

    # Querying inside the fitted range returns ~flat_vol.
    for K in [85.0, 100.0, 115.0]:
        for d in [40, 60, 80]:
            iv = query_iv_by_strike(surface, K, d)
            assert abs(iv - flat_vol) < 1e-3, f"K={K} d={d}: got {iv}"


def test_flat_surface_returns_surfacefit_with_expected_columns():
    chain = _synthetic_flat_chain(100.0, 0.2, [30, 60], [90.0, 100.0, 110.0])
    surface = fit_surface_from_chain(chain, spot=100.0, risk_free_rate=0.05)
    assert isinstance(surface, SurfaceFit)
    assert {"strike", "expiry_days", "iv", "log_moneyness", "option_type"} <= set(
        surface.points.columns
    )
    assert surface.spot == 100.0
    assert surface.risk_free_rate == 0.05
    assert surface.div_yield == 0.0


def test_surfacefit_is_frozen():
    chain = _synthetic_flat_chain(100.0, 0.2, [30], [100.0])
    surface = fit_surface_from_chain(chain, spot=100.0, risk_free_rate=0.05)
    with pytest.raises(Exception):
        surface.spot = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Surface fitting: skew (smile)
# ---------------------------------------------------------------------------


def test_skew_surface_puts_higher_iv_than_calls():
    spot = 100.0
    expiries = [30, 60, 90]
    strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
    chain = _synthetic_skew_chain(spot, expiries, strikes, base_vol=0.25, skew_slope=-0.5)
    surface = fit_surface_from_chain(chain, spot=spot, risk_free_rate=0.05)

    iv_low = query_iv_by_strike(surface, 0.9 * spot, 60.0)
    iv_atm = query_iv_by_strike(surface, spot, 60.0)
    iv_high = query_iv_by_strike(surface, 1.1 * spot, 60.0)

    assert iv_low > iv_atm > iv_high, f"expected monotone skew; got {iv_low}, {iv_atm}, {iv_high}"


def test_skew_surface_recovers_linear_skew():
    spot = 100.0
    base_vol = 0.25
    slope = -0.4
    expiries = [30, 60, 90]
    strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
    chain = _synthetic_skew_chain(spot, expiries, strikes, base_vol=base_vol, skew_slope=slope)
    surface = fit_surface_from_chain(chain, spot=spot, risk_free_rate=0.05)

    # The fitted IV at strike=spot, expiry=60 should be ~base_vol (since ln(S/S)=0).
    iv_atm = query_iv_by_strike(surface, spot, 60.0)
    assert abs(iv_atm - base_vol) < 1e-3


# ---------------------------------------------------------------------------
# Surface fitting: term structure
# ---------------------------------------------------------------------------


def test_term_structure_monotone_contango():
    spot = 100.0
    expiry_to_vol = {30: 0.18, 60: 0.20, 90: 0.22, 180: 0.25}
    strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
    chain = _synthetic_term_structure_chain(spot, expiry_to_vol, strikes)
    surface = fit_surface_from_chain(chain, spot=spot, risk_free_rate=0.05)

    ts = get_atm_term_structure(surface)
    # Sorted by expiry_days, monotone increasing.
    assert list(ts.index) == sorted(expiry_to_vol.keys())
    diffs = np.diff(ts.values)
    assert (diffs > 0).all(), f"expected monotone increasing term structure, got {ts.values}"


def test_term_structure_recovers_known_atm_iv():
    spot = 100.0
    expiry_to_vol = {30: 0.18, 60: 0.22, 120: 0.28}
    strikes = [90.0, 100.0, 110.0]
    chain = _synthetic_term_structure_chain(spot, expiry_to_vol, strikes)
    surface = fit_surface_from_chain(chain, spot=spot, risk_free_rate=0.05)

    ts = get_atm_term_structure(surface)
    for d, expected in expiry_to_vol.items():
        assert abs(ts.loc[d] - expected) < 1e-3, f"d={d}: got {ts.loc[d]}, expected {expected}"


# ---------------------------------------------------------------------------
# Delta queries
# ---------------------------------------------------------------------------


def test_delta_query_flat_surface_returns_flat_vol():
    spot = 100.0
    flat_vol = 0.20
    expiries = [30, 60, 90]
    strikes = [80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0]
    chain = _synthetic_flat_chain(spot, flat_vol, expiries, strikes)
    surface = fit_surface_from_chain(chain, spot=spot, risk_free_rate=0.05)

    # ATM-ish delta (~0.5 call) on a flat surface returns flat_vol.
    iv = query_iv_by_delta(surface, 0.5, 60.0, "call")
    assert abs(iv - flat_vol) < 1e-3

    iv_put = query_iv_by_delta(surface, -0.5, 60.0, "put")
    assert abs(iv_put - flat_vol) < 1e-3


def test_delta_query_skewed_surface_otm_vs_itm_call():
    """Skewed surface with negative slope: OTM call (low delta, high strike) has lower IV
    than ITM call (high delta, low strike)."""
    spot = 100.0
    expiries = [30, 60, 90]
    strikes = [70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0]
    chain = _synthetic_skew_chain(spot, expiries, strikes, base_vol=0.25, skew_slope=-0.5)
    surface = fit_surface_from_chain(chain, spot=spot, risk_free_rate=0.05)

    iv_otm = query_iv_by_delta(surface, 0.25, 60.0, "call")  # OTM call (high strike)
    iv_itm = query_iv_by_delta(surface, 0.75, 60.0, "call")  # ITM call (low strike)
    assert iv_itm > iv_otm, f"ITM IV {iv_itm} should exceed OTM IV {iv_otm} under negative skew"


def test_delta_query_invalid_call_delta_raises():
    chain = _synthetic_flat_chain(100.0, 0.2, [30, 60], [90.0, 100.0, 110.0])
    surface = fit_surface_from_chain(chain, spot=100.0, risk_free_rate=0.05)
    with pytest.raises(ValueError):
        query_iv_by_delta(surface, 1.5, 30.0, "call")
    with pytest.raises(ValueError):
        query_iv_by_delta(surface, -0.1, 30.0, "call")


def test_delta_query_invalid_put_delta_raises():
    chain = _synthetic_flat_chain(100.0, 0.2, [30, 60], [90.0, 100.0, 110.0])
    surface = fit_surface_from_chain(chain, spot=100.0, risk_free_rate=0.05)
    with pytest.raises(ValueError):
        query_iv_by_delta(surface, 0.5, 30.0, "put")
    with pytest.raises(ValueError):
        query_iv_by_delta(surface, -1.5, 30.0, "put")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_chain_raises():
    empty = pd.DataFrame(columns=["strike", "expiry_days", "market_price", "option_type"])
    with pytest.raises(ValueError):
        fit_surface_from_chain(empty, spot=100.0, risk_free_rate=0.05)


def test_chain_missing_columns_raises():
    chain = pd.DataFrame({"strike": [100.0], "expiry_days": [30]})
    with pytest.raises(ValueError):
        fit_surface_from_chain(chain, spot=100.0, risk_free_rate=0.05)


def test_invalid_spot_raises():
    chain = _synthetic_flat_chain(100.0, 0.2, [30], [100.0])
    with pytest.raises(ValueError):
        fit_surface_from_chain(chain, spot=0.0, risk_free_rate=0.05)


def test_single_point_chain_returns_constant_iv():
    """Single point (one strike, one expiry, one option type) => any nearby query should
    return that single IV (degenerate case: nearest-fallback)."""
    spot = 100.0
    flat_vol = 0.30
    T = 30 / 365.0
    price = _bs_price("call", spot, 100.0, T, 0.05, flat_vol)
    chain = pd.DataFrame(
        [{"strike": 100.0, "expiry_days": 30, "market_price": price, "option_type": "call"}]
    )
    surface = fit_surface_from_chain(chain, spot=spot, risk_free_rate=0.05)
    # Single-point degenerate case: any query returns the lone fitted IV.
    iv = query_iv_by_strike(surface, 100.0, 30.0)
    assert abs(iv - flat_vol) < 1e-4


def test_chain_with_unsolvable_rows_drops_them_with_warning():
    """A chain that includes rows priced below intrinsic: those rows should be dropped
    (with a RuntimeWarning) and the remaining good rows should still produce a valid fit."""
    spot = 100.0
    chain = _synthetic_flat_chain(spot, 0.2, [30, 60], [90.0, 100.0, 110.0])
    # Inject one bad row: a call priced below intrinsic (impossible).
    bad_row = pd.DataFrame(
        [{"strike": 50.0, "expiry_days": 30, "market_price": 0.01, "option_type": "call"}]
    )
    chain = pd.concat([chain, bad_row], ignore_index=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        surface = fit_surface_from_chain(chain, spot=spot, risk_free_rate=0.05)
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)
    # All remaining points should have IV ~0.2.
    assert (surface.points["iv"].sub(0.2).abs() < 1e-4).all()


def test_invalid_strike_raises_on_query():
    chain = _synthetic_flat_chain(100.0, 0.2, [30, 60], [90.0, 100.0, 110.0])
    surface = fit_surface_from_chain(chain, spot=100.0, risk_free_rate=0.05)
    with pytest.raises(ValueError):
        query_iv_by_strike(surface, 0.0, 30.0)
    with pytest.raises(ValueError):
        query_iv_by_strike(surface, 100.0, 0.0)
