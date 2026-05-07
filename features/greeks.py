"""Black-Scholes Greeks for European options + per-position aggregation. Pure math, no I/O."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from scipy.stats import norm

OptionType = Literal["call", "put"]
Direction = Literal["long", "short"]


@dataclass(frozen=True)
class Greeks:
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


@dataclass(frozen=True)
class Leg:
    instrument_type: Literal["call", "put", "stock"]
    direction: Direction
    quantity: int
    multiplier: int
    delta: float
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0


def _validate(S: float, K: float, T: float, sigma: float) -> None:
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        raise ValueError(f"S, K, T, sigma must all be positive (got S={S}, K={K}, T={T}, sigma={sigma})")


def _d1(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """d1 = (ln(S/K) + (r - q + sigma^2/2) T) / (sigma sqrt(T))"""
    return (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))


def _d2(d1: float, T: float, sigma: float) -> float:
    """d2 = d1 - sigma sqrt(T)"""
    return d1 - sigma * math.sqrt(T)


def bs_price(option_type: OptionType, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Black-Scholes-Merton price for European call or put with continuous dividend yield q."""
    _validate(S, K, T, sigma)
    d1 = _d1(S, K, T, r, sigma, q)
    d2 = _d2(d1, T, sigma)
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    if option_type == "call":
        return S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    if option_type == "put":
        return K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def bs_delta(option_type: OptionType, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Call: e^(-qT) N(d1). Put: -e^(-qT) N(-d1)."""
    _validate(S, K, T, sigma)
    d1 = _d1(S, K, T, r, sigma, q)
    disc_q = math.exp(-q * T)
    if option_type == "call":
        return disc_q * norm.cdf(d1)
    if option_type == "put":
        return -disc_q * norm.cdf(-d1)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def bs_gamma(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """e^(-qT) phi(d1) / (S sigma sqrt(T)). Same for call and put."""
    _validate(S, K, T, sigma)
    d1 = _d1(S, K, T, r, sigma, q)
    return math.exp(-q * T) * norm.pdf(d1) / (S * sigma * math.sqrt(T))


def bs_theta(option_type: OptionType, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Per-year theta (annualised, in price-units-per-year). Divide by 365 for per-day."""
    _validate(S, K, T, sigma)
    d1 = _d1(S, K, T, r, sigma, q)
    d2 = _d2(d1, T, sigma)
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    common = -S * disc_q * norm.pdf(d1) * sigma / (2.0 * math.sqrt(T))
    if option_type == "call":
        return common - r * K * disc_r * norm.cdf(d2) + q * S * disc_q * norm.cdf(d1)
    if option_type == "put":
        return common + r * K * disc_r * norm.cdf(-d2) - q * S * disc_q * norm.cdf(-d1)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def bs_vega(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """S e^(-qT) phi(d1) sqrt(T). Same for call and put. Per 1.0 sigma (i.e. 100 vol points)."""
    _validate(S, K, T, sigma)
    d1 = _d1(S, K, T, r, sigma, q)
    return S * math.exp(-q * T) * norm.pdf(d1) * math.sqrt(T)


def bs_rho(option_type: OptionType, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Per 1.0 r (i.e. 100 rate bps)."""
    _validate(S, K, T, sigma)
    d1 = _d1(S, K, T, r, sigma, q)
    d2 = _d2(d1, T, sigma)
    disc_r = math.exp(-r * T)
    if option_type == "call":
        return K * T * disc_r * norm.cdf(d2)
    if option_type == "put":
        return -K * T * disc_r * norm.cdf(-d2)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def compute_greeks(option_type: OptionType, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> Greeks:
    """Returns all 5 Greeks in one call."""
    return Greeks(
        delta=bs_delta(option_type, S, K, T, r, sigma, q),
        gamma=bs_gamma(S, K, T, r, sigma, q),
        theta=bs_theta(option_type, S, K, T, r, sigma, q),
        vega=bs_vega(S, K, T, r, sigma, q),
        rho=bs_rho(option_type, S, K, T, r, sigma, q),
    )


# ----------------------------------------------------------------------------
# Unit conversions for the risk/envelope.py boundary
# ----------------------------------------------------------------------------
# bs_vega(...) returns vega per 1.0 sigma (= per 100 vol points; standard
# Black-Scholes textbook convention). risk/envelope.py expects vega per 1 vol
# point (see test_envelope.py: vega=-100, vol_change=30 -> -$3000 P&L).
# bs_gamma(...) returns per-share gamma in 1/$ units; risk/envelope.py expects
# a position-level dollar gamma already pre-scaled for the 1% spot move
# convention used in `0.5 * gamma * (spot_change_pct * notional)^2`.
#
# ALWAYS convert at the source-of-input boundary. Consumers populating
# `risk.envelope.CandidatePosition.vega` and `.gamma` MUST route through
# `vega_to_vol_point(...)` and `position_dollar_gamma(...)` respectively --
# NOT raw `bs_vega` / `bs_gamma`. Wiring raw BS Greeks directly would be 100x
# too large for vega and dimensionally wrong for gamma.


def vega_to_vol_point(vega_per_sigma: float) -> float:
    """Convert BS vega (per 1.0 sigma) to per-1-vol-point. Divide by 100.

    Use to populate `risk.envelope.CandidatePosition.vega` from `bs_vega(...)`.
    """
    return float(vega_per_sigma) / 100.0


def position_dollar_gamma(
    leg_gamma_per_dollar: float,
    spot: float,
    quantity: int,
    multiplier: int,
    direction_sign: int,  # +1 long, -1 short
) -> float:
    """Convert per-share BS gamma (1/$) to dollar gamma at the position level.

    Dollar gamma definition: 0.5 * gamma * (1% spot move)^2 in dollars.
    Returns: direction_sign * quantity * multiplier * gamma * spot * spot * 1e-4.

    The 1e-4 = (0.01)^2 = 1%-squared scaling so the result is dollars
    of P&L per 1% spot move squared, which matches risk/envelope.py's
    `0.5 * gamma * (spot_change_pct * notional)^2` convention only when
    `notional == spot * quantity * multiplier`. Use this helper to
    pre-compute the position gamma input rather than doing the math inline.

    Use to populate `risk.envelope.CandidatePosition.gamma` from `bs_gamma(...)`.
    """
    if direction_sign not in (-1, 1):
        raise ValueError(f"direction_sign must be +1 or -1, got {direction_sign}")
    if quantity < 0:
        raise ValueError(f"quantity must be non-negative, got {quantity}")
    if multiplier <= 0:
        raise ValueError(f"multiplier must be positive, got {multiplier}")
    if spot <= 0:
        raise ValueError(f"spot must be positive, got {spot}")
    return float(direction_sign) * float(quantity) * float(multiplier) * float(leg_gamma_per_dollar) * float(spot) * float(spot) * 1e-4


def aggregate_position_greeks(legs: list[Leg]) -> Greeks:
    """Sum signed contributions across legs (sign * quantity * multiplier * leg.greek)."""
    if not legs:
        return Greeks(0.0, 0.0, 0.0, 0.0, 0.0)
    delta = gamma = theta = vega = rho = 0.0
    for leg in legs:
        if leg.quantity < 0:
            raise ValueError(f"leg.quantity must be non-negative, got {leg.quantity}")
        if leg.multiplier <= 0:
            raise ValueError(f"leg.multiplier must be positive, got {leg.multiplier}")
        sign = 1 if leg.direction == "long" else -1
        weight = sign * leg.quantity * leg.multiplier
        delta += weight * leg.delta
        gamma += weight * leg.gamma
        theta += weight * leg.theta
        vega += weight * leg.vega
        rho += weight * leg.rho
    return Greeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)
