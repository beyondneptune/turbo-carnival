"""Implied volatility surface fitting from option chains.
Non-parametric (scipy interpolation only) per Cycle 1 scope."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.optimize import brentq

from features.greeks import bs_price as _bs_price, bs_delta as _bs_delta


OptionType = Literal["call", "put"]

_IV_LO: float = 1e-6
_IV_HI: float = 5.0


@dataclass(frozen=True)
class SurfaceFit:
    """A fitted IV surface. Immutable.

    Internally: stores the input chain points and an interpolator over (log_moneyness, T).
    Queryable by either (strike, expiry_days) or (delta, expiry_days)."""

    points: pd.DataFrame  # columns: strike, expiry_days, iv, log_moneyness, option_type
    spot: float
    risk_free_rate: float
    div_yield: float = 0.0
    method: Literal["linear", "cubic", "nearest"] = "linear"


def implied_vol(
    market_price: float,
    option_type: OptionType,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    *,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """Returns IV that prices the option to market_price under Black-Scholes.
    Raises ValueError if no solution in [1e-6, 5.0] (i.e. >500% vol).
    """
    if market_price < 0:
        raise ValueError(f"market_price must be non-negative, got {market_price}")
    if S <= 0 or K <= 0 or T <= 0:
        raise ValueError(f"S, K, T must be positive (got S={S}, K={K}, T={T})")
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    # Arbitrage bounds. Below intrinsic => no positive IV solves it.
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    if option_type == "call":
        intrinsic = max(S * disc_q - K * disc_r, 0.0)
        upper_bound = S * disc_q  # call price <= S e^{-qT}
    else:
        intrinsic = max(K * disc_r - S * disc_q, 0.0)
        upper_bound = K * disc_r  # put price <= K e^{-rT}

    if market_price < intrinsic - tol:
        raise ValueError(
            f"market_price {market_price} is below intrinsic value {intrinsic} "
            f"for {option_type} S={S} K={K} T={T} r={r} q={q}"
        )
    if market_price > upper_bound + tol:
        raise ValueError(
            f"market_price {market_price} exceeds no-arb upper bound {upper_bound} "
            f"for {option_type} S={S} K={K} T={T} r={r} q={q}"
        )

    def f(sigma: float) -> float:
        return _bs_price(option_type, S, K, T, r, sigma, q) - market_price

    f_lo = f(_IV_LO)
    f_hi = f(_IV_HI)
    if f_lo * f_hi > 0:
        # No sign change in [_IV_LO, _IV_HI]; nudge endpoints to handle numerical edge cases.
        if abs(f_lo) < tol:
            return _IV_LO
        if abs(f_hi) < tol:
            return _IV_HI
        raise ValueError(
            f"No IV solution in [{_IV_LO}, {_IV_HI}] for {option_type} "
            f"price={market_price} S={S} K={K} T={T}"
        )
    return float(brentq(f, _IV_LO, _IV_HI, xtol=tol, maxiter=max_iter))


def fit_surface_from_chain(
    chain: pd.DataFrame,
    spot: float,
    risk_free_rate: float,
    div_yield: float = 0.0,
    method: Literal["linear", "cubic", "nearest"] = "linear",
) -> SurfaceFit:
    """Input chain DataFrame must have columns:
        strike: float
        expiry_days: int (calendar days to expiry)
        market_price: float (mid)
        option_type: 'call' or 'put'
    Computes IV per row, returns SurfaceFit.
    Drops rows where IV solving fails (with a warning).
    """
    if chain is None or len(chain) == 0:
        raise ValueError("chain is empty")
    required = {"strike", "expiry_days", "market_price", "option_type"}
    missing = required - set(chain.columns)
    if missing:
        raise ValueError(f"chain missing required columns: {sorted(missing)}")
    if spot <= 0:
        raise ValueError(f"spot must be positive, got {spot}")

    rows: list[dict] = []
    failures = 0
    for _, row in chain.iterrows():
        K = float(row["strike"])
        exp_d = int(row["expiry_days"])
        mp = float(row["market_price"])
        ot: OptionType = row["option_type"]
        if exp_d <= 0 or K <= 0:
            failures += 1
            continue
        T = exp_d / 365.0
        try:
            iv = implied_vol(mp, ot, spot, K, T, risk_free_rate, div_yield)
        except ValueError:
            failures += 1
            continue
        rows.append(
            {
                "strike": K,
                "expiry_days": exp_d,
                "iv": iv,
                "log_moneyness": math.log(K / spot),
                "option_type": ot,
            }
        )

    if failures:
        warnings.warn(
            f"fit_surface_from_chain: dropped {failures} rows where IV solve failed",
            RuntimeWarning,
        )
    if not rows:
        raise ValueError("fit_surface_from_chain: no rows produced a valid IV")

    points = pd.DataFrame(rows)
    return SurfaceFit(
        points=points,
        spot=float(spot),
        risk_free_rate=float(risk_free_rate),
        div_yield=float(div_yield),
        method=method,
    )


def _interp_iv(surface: SurfaceFit, log_moneyness: float, expiry_days: float) -> float:
    """Interpolate IV at (log_moneyness, expiry_days) using griddata.
    Falls back to nearest if linear/cubic returns NaN (out of convex hull or degenerate)."""
    pts = surface.points
    xy = np.column_stack([pts["log_moneyness"].to_numpy(), pts["expiry_days"].to_numpy()])
    z = pts["iv"].to_numpy()
    target = np.array([[log_moneyness, float(expiry_days)]])

    # Degenerate case: a single unique grid point => nearest only.
    n_unique = len({(round(x, 12), round(y, 12)) for x, y in xy})
    if n_unique < 2:
        return float(z[0])
    # cubic requires >=4 non-collinear points; griddata handles collinearity by returning NaN.
    method = surface.method
    try:
        val = griddata(xy, z, target, method=method)[0]
    except Exception:
        val = np.nan
    if np.isnan(val):
        # Fall back to nearest (always defined).
        val = griddata(xy, z, target, method="nearest")[0]
    return float(val)


def query_iv_by_strike(surface: SurfaceFit, strike: float, expiry_days: float) -> float:
    """Interpolate IV at (log_moneyness=ln(strike/spot), expiry_days)."""
    if strike <= 0:
        raise ValueError(f"strike must be positive, got {strike}")
    if expiry_days <= 0:
        raise ValueError(f"expiry_days must be positive, got {expiry_days}")
    lm = math.log(strike / surface.spot)
    return _interp_iv(surface, lm, float(expiry_days))


def query_iv_by_delta(
    surface: SurfaceFit,
    delta: float,
    expiry_days: float,
    option_type: OptionType = "call",
) -> float:
    """Find the strike that gives the requested delta at the requested expiry, then return its IV.
    For calls: 0.05 (deep OTM) to 0.95 (deep ITM). For puts: -0.95 to -0.05.
    Implementation: solve for strike via root-finding using BS delta + interpolated IV; iterate to fixed-point.
    """
    if expiry_days <= 0:
        raise ValueError(f"expiry_days must be positive, got {expiry_days}")
    if option_type == "call":
        if not (0.0 < delta < 1.0):
            raise ValueError(f"call delta must be in (0, 1), got {delta}")
    elif option_type == "put":
        if not (-1.0 < delta < 0.0):
            raise ValueError(f"put delta must be in (-1, 0), got {delta}")
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    S = surface.spot
    r = surface.risk_free_rate
    q = surface.div_yield
    T = float(expiry_days) / 365.0

    # Strike search bounds: span an order of magnitude around spot.
    K_lo = max(S * 1e-3, 1e-6)
    K_hi = S * 1e3

    def delta_at_strike(K: float) -> float:
        iv = _interp_iv(surface, math.log(K / S), float(expiry_days))
        # Guard against degenerate IV.
        iv = float(np.clip(iv, _IV_LO, _IV_HI))
        return _bs_delta(option_type, S, K, T, r, iv, q)

    def f(K: float) -> float:
        return delta_at_strike(K) - delta

    # BS delta is monotonic in K (call: decreasing, put: increasing) for fixed sigma.
    # Even with IV varying via interpolation, in practice on smooth surfaces this remains
    # well-behaved enough for brentq once we confirm a sign change.
    f_lo = f(K_lo)
    f_hi = f(K_hi)
    if f_lo * f_hi > 0:
        # Try a denser scan to find a bracket.
        ks = np.geomspace(K_lo, K_hi, 64)
        vals = np.array([f(k) for k in ks])
        sign_changes = np.where(np.diff(np.sign(vals)) != 0)[0]
        if len(sign_changes) == 0:
            raise ValueError(
                f"Could not bracket delta={delta} for option_type={option_type} "
                f"expiry_days={expiry_days}"
            )
        i = sign_changes[0]
        K_lo, K_hi = float(ks[i]), float(ks[i + 1])

    K_star = float(brentq(f, K_lo, K_hi, xtol=1e-6, maxiter=100))
    return _interp_iv(surface, math.log(K_star / S), float(expiry_days))


def get_atm_term_structure(surface: SurfaceFit) -> pd.Series:
    """Returns IV at strike=spot for each unique expiry_days in the surface (sorted)."""
    expiries = sorted(surface.points["expiry_days"].unique().tolist())
    values = [query_iv_by_strike(surface, surface.spot, float(d)) for d in expiries]
    return pd.Series(data=values, index=pd.Index(expiries, name="expiry_days"), name="atm_iv")
