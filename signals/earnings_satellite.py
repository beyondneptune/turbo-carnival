"""Phase 3 earnings satellite — twin-model target computation + position structuring.

Pure functions and dataclasses. Does NOT train models (that's twin_model_trainer.py)
and does NOT execute trades. Provides the contract for what each model targets and
how a candidate position is structured at T-1 close.

Spec source:
    metadata/strategic_lock.md  § Twin-Model Design
    metadata/capital_policy.md  § Satellite

Twin-model targets:
    Model A (baseline)        : Target = net_5day_straddle_return_after_costs
    Model B (cond-aligned)    : Target = return / max(expected_event_risk, risk_floor)

    expected_event_risk = pre_event_ATM_IV_5d × √(5/252)
    risk_floor          = 20th percentile of expected_event_risk in training data

Position structuring (T-1 close):
    - ATM strike at decision time (round to nearest available, ties broken low)
    - Nearest weekly expiry ≥ 3 calendar days post-earnings
    - Hold T-1 close → T+5 close, no partial exits
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRADING_DAYS_PER_YEAR: int = 252
HOLD_PERIOD_DAYS: int = 5  # T+5 evaluation horizon
DEFAULT_RISK_FLOOR_QUANTILE: float = 0.20
DEFAULT_MIN_CAL_DAYS_AFTER_EVENT: int = 3


# Square root of (5 / 252) used to scale 5-day expected move under Brownian motion.
# Exposed as a module constant so callers can verify the magic number.
_SQRT_HOLD_OVER_YEAR: float = math.sqrt(HOLD_PERIOD_DAYS / TRADING_DAYS_PER_YEAR)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EarningsEvent:
    """Single earnings event observation, from announcement metadata + chain snapshot.

    The realized fields are filled in at the T+5 evaluation step (post hoc); they are
    None for not-yet-evaluated events. They do NOT participate in feature engineering
    — they are the targets/outcomes the twin-model framework predicts.
    """

    ticker: str
    event_date: date                # the day earnings is announced
    decision_date: date             # T-1 close (the day BEFORE event)
    spot_at_decision: float         # ATM ref price
    nearest_expiry: date            # weekly expiration ≥ 3 cal days post-event
    pre_event_atm_iv_5d: float      # 5-day ATM IV ahead of event
    realized_5d_return: float | None = None             # underlying 5-day return T-1 → T+5
    realized_5d_straddle_return: float | None = None    # straddle PnL after costs (target A)


@dataclass(frozen=True)
class StraddlePosition:
    """Long ATM straddle structured at T-1 close.

    All prices are per-contract dollar quotes (NOT cents). entry_total_premium is the
    debit paid (call_mid + put_mid). Gross P&L at T+5 is computed elsewhere via
    paper_exec; this class only carries the entry contract.
    """

    ticker: str
    decision_date: date
    expiry: date
    strike: float                   # ATM = closest available strike to spot, ties broken low
    entry_call_price: float         # mid
    entry_put_price: float
    entry_total_premium: float      # call + put
    contracts: int = 1


# ---------------------------------------------------------------------------
# Strike + expiry selection
# ---------------------------------------------------------------------------


def select_atm_strike(spot: float, available_strikes: list[float]) -> float:
    """Returns the strike closest to spot (ties broken to the lower strike).

    Args:
        spot: Reference price at decision time (T-1 close).
        available_strikes: Strikes listed in the chain. Need not be sorted.

    Returns:
        The strike with minimum |strike - spot|. If two strikes tie, the lower one wins
        — this matches the convention for delta-neutral structures: a slightly low strike
        gives a marginally positive call delta and marginally negative put delta whose
        magnitudes are closer to symmetric than the high-strike alternative.

    Raises:
        ValueError: if available_strikes is empty or spot is non-positive.
    """
    if spot <= 0:
        raise ValueError(f"spot must be positive, got {spot}")
    if not available_strikes:
        raise ValueError("available_strikes is empty")

    # Sort ascending so equal-distance ties resolve to the lower strike.
    sorted_strikes = sorted(float(k) for k in available_strikes)
    best_strike = sorted_strikes[0]
    best_diff = abs(best_strike - spot)
    for k in sorted_strikes[1:]:
        diff = abs(k - spot)
        # Strict less-than preserves the lower-strike-on-tie behaviour.
        if diff < best_diff:
            best_strike = k
            best_diff = diff
    return best_strike


def select_post_event_expiry(
    event_date: date,
    available_expiries: list[date],
    *,
    min_calendar_days_after: int = DEFAULT_MIN_CAL_DAYS_AFTER_EVENT,
) -> date:
    """Returns the nearest weekly expiry that is ≥ event_date + min_calendar_days_after.

    The earliest acceptable expiry equals event_date + min_calendar_days_after; any expiry
    on or after that boundary qualifies. Among qualifying expiries we pick the closest one
    (smallest delta).

    Args:
        event_date: The day earnings is announced (T+0).
        available_expiries: Listed weekly expirations. Need not be sorted.
        min_calendar_days_after: Minimum gap in calendar days. 3 is the project default and
            ensures the straddle can survive at least one full post-event close-to-close move
            before assignment risk dominates.

    Raises:
        ValueError: if available_expiries is empty, min_calendar_days_after is negative,
            or no expiry meets the gap requirement.
    """
    if min_calendar_days_after < 0:
        raise ValueError(
            f"min_calendar_days_after must be non-negative, got {min_calendar_days_after}"
        )
    if not available_expiries:
        raise ValueError("available_expiries is empty")

    earliest_acceptable = event_date + timedelta(days=min_calendar_days_after)
    eligible = [e for e in available_expiries if e >= earliest_acceptable]
    if not eligible:
        raise ValueError(
            f"no weekly expiry on or after {earliest_acceptable.isoformat()} "
            f"(event={event_date.isoformat()}, min_days={min_calendar_days_after}, "
            f"available={[e.isoformat() for e in sorted(available_expiries)]})"
        )
    return min(eligible)


# ---------------------------------------------------------------------------
# Straddle structuring
# ---------------------------------------------------------------------------


def _ensure_mid(chain: pd.DataFrame) -> pd.DataFrame:
    """Return a chain with a 'mid' column. If absent, compute (bid+ask)/2."""
    if "mid" in chain.columns:
        return chain
    if not {"bid", "ask"}.issubset(chain.columns):
        raise ValueError(
            "chain must have either a 'mid' column or both 'bid' and 'ask' columns"
        )
    out = chain.copy()
    out["mid"] = 0.5 * (out["bid"].astype(float) + out["ask"].astype(float))
    return out


def structure_straddle(
    event: EarningsEvent,
    chain: pd.DataFrame,
) -> StraddlePosition:
    """Build a long ATM straddle from a chain snapshot at T-1 close.

    Args:
        event: EarningsEvent carrying the decision context (spot, expiry, etc.).
        chain: DataFrame with columns 'strike', 'expiry', 'option_type'
            ('call' | 'put'), and either 'mid' or both 'bid'/'ask'. The chain may
            include rows for non-target expiries; this function filters down to
            event.nearest_expiry before picking the ATM strike.

    Returns:
        StraddlePosition with entry_total_premium = call_mid + put_mid.

    Raises:
        ValueError: if required columns are missing, the filtered chain is empty,
            or the ATM strike lacks a matching call AND put row.
    """
    required = {"strike", "expiry", "option_type"}
    missing = required - set(chain.columns)
    if missing:
        raise ValueError(f"chain missing required columns: {sorted(missing)}")

    chain = _ensure_mid(chain)

    expiry_chain = chain[chain["expiry"] == event.nearest_expiry]
    if expiry_chain.empty:
        raise ValueError(
            f"chain has no rows for expiry={event.nearest_expiry.isoformat()} "
            f"(ticker={event.ticker})"
        )

    available_strikes = sorted({float(k) for k in expiry_chain["strike"].tolist()})
    atm_strike = select_atm_strike(event.spot_at_decision, available_strikes)

    call_rows = expiry_chain[
        (expiry_chain["strike"] == atm_strike) & (expiry_chain["option_type"] == "call")
    ]
    put_rows = expiry_chain[
        (expiry_chain["strike"] == atm_strike) & (expiry_chain["option_type"] == "put")
    ]
    if call_rows.empty:
        raise ValueError(f"no call at ATM strike {atm_strike} for {event.ticker}")
    if put_rows.empty:
        raise ValueError(f"no put at ATM strike {atm_strike} for {event.ticker}")

    call_mid = float(call_rows.iloc[0]["mid"])
    put_mid = float(put_rows.iloc[0]["mid"])
    if math.isnan(call_mid) or math.isnan(put_mid):
        raise ValueError(
            f"NaN mid at strike {atm_strike} for {event.ticker} "
            f"(call_mid={call_mid}, put_mid={put_mid})"
        )
    if call_mid < 0 or put_mid < 0:
        raise ValueError(
            f"negative mid at strike {atm_strike} for {event.ticker} "
            f"(call_mid={call_mid}, put_mid={put_mid})"
        )

    return StraddlePosition(
        ticker=event.ticker,
        decision_date=event.decision_date,
        expiry=event.nearest_expiry,
        strike=atm_strike,
        entry_call_price=call_mid,
        entry_put_price=put_mid,
        entry_total_premium=call_mid + put_mid,
        contracts=1,
    )


# ---------------------------------------------------------------------------
# Twin-model targets
# ---------------------------------------------------------------------------


def compute_expected_event_risk(pre_event_atm_iv_5d: float) -> float:
    """expected_event_risk = pre_event_atm_iv_5d × √(5/252).

    Reasoning: under geometric-Brownian motion with annualised vol σ, the standard
    deviation of an h-day log return is σ × √(h/252). Here h = 5 trading days, so this
    is the model-implied 5-day move size that the satellite straddle is buying or
    selling against.

    Raises ValueError if pre_event_atm_iv_5d is negative or non-finite.
    """
    if not math.isfinite(pre_event_atm_iv_5d):
        raise ValueError(f"pre_event_atm_iv_5d must be finite, got {pre_event_atm_iv_5d}")
    if pre_event_atm_iv_5d < 0:
        raise ValueError(f"pre_event_atm_iv_5d must be non-negative, got {pre_event_atm_iv_5d}")
    return float(pre_event_atm_iv_5d) * _SQRT_HOLD_OVER_YEAR


def compute_risk_floor(
    pre_event_atm_iv_5d_series: pd.Series,
    *,
    quantile: float = DEFAULT_RISK_FLOOR_QUANTILE,
) -> float:
    """20th percentile of (iv × √(5/252)) across the training series.

    Used to clip Model B's denominator. When the pre-event ATM IV is anomalously low,
    the raw expected_event_risk produces an exploding target; the floor caps that.

    Args:
        pre_event_atm_iv_5d_series: ATM IV values (NOT yet scaled). NaNs are dropped.
        quantile: percentile in (0, 1). Default 0.20 per spec.

    Raises:
        ValueError: if the series is empty/all-NaN, contains negatives, or quantile
            is outside (0, 1).
    """
    if not 0 < quantile < 1:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    iv = pd.to_numeric(pre_event_atm_iv_5d_series, errors="coerce").dropna()
    if iv.empty:
        raise ValueError("pre_event_atm_iv_5d_series is empty after dropping NaNs")
    if (iv < 0).any():
        raise ValueError("pre_event_atm_iv_5d_series contains negative values")
    expected_risk = iv * _SQRT_HOLD_OVER_YEAR
    return float(expected_risk.quantile(quantile))


def model_a_target(realized_5d_straddle_return: float) -> float:
    """Identity. Defined explicitly so it's clear what Model A targets.

    Model A predicts the raw 5-day straddle return after costs — the "naive" baseline
    against which Model B's condition-aligned target is paired-t-tested.
    """
    if not math.isfinite(realized_5d_straddle_return):
        raise ValueError(
            f"realized_5d_straddle_return must be finite, got {realized_5d_straddle_return}"
        )
    return float(realized_5d_straddle_return)


def model_b_target(
    realized_5d_straddle_return: float,
    pre_event_atm_iv_5d: float,
    risk_floor: float,
) -> float:
    """Model B target = return / max(expected_event_risk, risk_floor).

    Args:
        realized_5d_straddle_return: net 5-day straddle return after costs (Model A's target).
        pre_event_atm_iv_5d: ATM 5-day IV at decision time. Used to compute
            expected_event_risk = iv × √(5/252).
        risk_floor: lower bound on the denominator (typically the 20th percentile of
            expected_event_risk in training data — see compute_risk_floor).

    Raises:
        ValueError: if any input is non-finite or risk_floor ≤ 0.
    """
    if not math.isfinite(realized_5d_straddle_return):
        raise ValueError(
            f"realized_5d_straddle_return must be finite, got {realized_5d_straddle_return}"
        )
    if risk_floor <= 0:
        raise ValueError(f"risk_floor must be positive, got {risk_floor}")
    expected_event_risk = compute_expected_event_risk(pre_event_atm_iv_5d)
    denominator = max(expected_event_risk, risk_floor)
    return float(realized_5d_straddle_return) / denominator


# ---------------------------------------------------------------------------
# Batch target builder
# ---------------------------------------------------------------------------


_REQUIRED_EVENT_COLUMNS: tuple[str, ...] = (
    "ticker",
    "event_date",
    "pre_event_atm_iv_5d",
    "realized_5d_straddle_return",
)


def build_targets(
    events: pd.DataFrame,
    *,
    risk_floor: float | None = None,
) -> pd.DataFrame:
    """Compute Model A and Model B targets for a DataFrame of events.

    Args:
        events: DataFrame with columns:
            ticker, event_date, pre_event_atm_iv_5d, realized_5d_straddle_return.
        risk_floor: If None, computed once from events at quantile=0.20 (the default
            per strategic_lock.md). Otherwise used as-is. The same risk_floor applies
            to every row in the returned frame; this is intentional — it prevents
            within-set leakage and matches how the floor is applied at training time.

    Returns:
        Copy of `events` with these added columns:
            target_a              : realized_5d_straddle_return (identity)
            target_b              : return / max(expected_event_risk, risk_floor_used)
            expected_event_risk   : pre_event_atm_iv_5d × √(5/252)
            risk_floor_used       : the floor value applied to every row

    Raises:
        ValueError: missing required columns, empty frame, or computed risk_floor ≤ 0.
    """
    missing = [c for c in _REQUIRED_EVENT_COLUMNS if c not in events.columns]
    if missing:
        raise ValueError(f"events DataFrame missing required columns: {missing}")
    if events.empty:
        raise ValueError("events DataFrame is empty")

    out = events.copy()
    iv = pd.to_numeric(out["pre_event_atm_iv_5d"], errors="coerce")
    ret = pd.to_numeric(out["realized_5d_straddle_return"], errors="coerce")

    if iv.isna().any():
        raise ValueError("events.pre_event_atm_iv_5d has non-numeric/NaN entries")
    if ret.isna().any():
        raise ValueError("events.realized_5d_straddle_return has non-numeric/NaN entries")
    if (iv < 0).any():
        raise ValueError("events.pre_event_atm_iv_5d contains negative values")

    expected_risk = iv * _SQRT_HOLD_OVER_YEAR

    if risk_floor is None:
        risk_floor_used = float(expected_risk.quantile(DEFAULT_RISK_FLOOR_QUANTILE))
    else:
        risk_floor_used = float(risk_floor)
    if risk_floor_used <= 0:
        raise ValueError(f"risk_floor_used must be positive, got {risk_floor_used}")

    denom = expected_risk.clip(lower=risk_floor_used)

    out["target_a"] = ret.astype(float)
    out["target_b"] = (ret / denom).astype(float)
    out["expected_event_risk"] = expected_risk.astype(float)
    out["risk_floor_used"] = float(risk_floor_used)
    return out
