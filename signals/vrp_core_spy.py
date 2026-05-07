"""Phase 2 VRP core — SPY iron condor structuring and entry/exit decisions.
Pure functions; no I/O."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Literal

import pandas as pd


ExitReason = Literal[
    "profit_target",
    "stop_loss",
    "21dte_close",
    "expiry_pin",
    "hold",
    "force_close",
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IronCondorSpec:
    """An iron condor structure: short call + long call (further OTM) + short put + long put (further OTM).

    Convention: short_call_strike < long_call_strike (call wing extends upward),
    short_put_strike > long_put_strike (put wing extends downward).
    """

    underlying: str
    expiry: date
    short_call_strike: float
    long_call_strike: float
    short_put_strike: float
    long_put_strike: float
    quantity: int = 1
    multiplier: int = 100

    @property
    def call_wing_width(self) -> float:
        """Distance between long and short call strikes (always non-negative for a valid spec)."""
        return self.long_call_strike - self.short_call_strike

    @property
    def put_wing_width(self) -> float:
        """Distance between short and long put strikes (always non-negative for a valid spec)."""
        return self.short_put_strike - self.long_put_strike

    @property
    def max_loss(self) -> float:
        """Worst-case loss before credit is netted, in dollars.

        For an iron condor, max loss occurs in only one wing at expiry (the underlying
        cannot be both above the call wing and below the put wing simultaneously). So
        max_loss = max(call_width, put_width) * quantity * multiplier — the credit
        received reduces the realised loss but is not subtracted here; callers pair
        this with entry_credit to compute net max loss.
        """
        widest = max(self.call_wing_width, self.put_wing_width)
        return widest * self.quantity * self.multiplier

    def is_valid(self) -> bool:
        """Structural sanity: short strikes inside, long strikes further OTM, positive quantity."""
        if self.quantity <= 0 or self.multiplier <= 0:
            return False
        if self.short_call_strike <= 0 or self.long_call_strike <= 0:
            return False
        if self.short_put_strike <= 0 or self.long_put_strike <= 0:
            return False
        # Call wing: short strike below long strike (sell closer to spot, buy further OTM up).
        if self.short_call_strike >= self.long_call_strike:
            return False
        # Put wing: short strike above long strike (sell closer to spot, buy further OTM down).
        if self.short_put_strike <= self.long_put_strike:
            return False
        # Body: short put below short call (otherwise the condor is inverted/crossed).
        if self.short_put_strike >= self.short_call_strike:
            return False
        return True


@dataclass(frozen=True)
class IronCondorEntry:
    """A live iron condor position with entry-time book-keeping."""

    spec: IronCondorSpec
    entry_credit: float        # net credit received per spread, in dollars (NOT cents)
    entry_at: date
    short_call_delta: float
    short_put_delta: float
    expected_pnl_50pct_target: float


@dataclass(frozen=True)
class ExitDecision:
    should_exit: bool
    reason: ExitReason
    target_pnl: float | None = None


# ---------------------------------------------------------------------------
# Strike selection
# ---------------------------------------------------------------------------


def find_short_strikes_by_delta(
    chain: pd.DataFrame,
    target_short_delta: float = 0.12,
    delta_tolerance: float = 0.03,
) -> tuple[float, float]:
    """Pick short call (delta closest to +target) and short put (delta closest to -target) from chain.

    Args:
        chain: DataFrame with columns 'strike', 'option_type' ('call'|'put'), 'delta'.
            Call deltas should be in (0, 1); put deltas in (-1, 0).
        target_short_delta: Magnitude of the desired short-leg delta (e.g. 0.12 = 12 delta).
            Must be positive. The function searches for calls near +target and puts near -target.
        delta_tolerance: Maximum allowed |chain_delta - target| for a candidate to qualify.

    Returns:
        (short_call_strike, short_put_strike).

    Raises:
        ValueError: if no call within +/- delta_tolerance of +target_short_delta exists,
            or no put within +/- delta_tolerance of -target_short_delta exists.
    """
    if target_short_delta <= 0:
        raise ValueError(f"target_short_delta must be positive, got {target_short_delta}")
    if delta_tolerance < 0:
        raise ValueError(f"delta_tolerance must be non-negative, got {delta_tolerance}")
    required_cols = {"strike", "option_type", "delta"}
    missing = required_cols - set(chain.columns)
    if missing:
        raise ValueError(f"chain missing required columns: {sorted(missing)}")

    calls = chain[chain["option_type"] == "call"]
    puts = chain[chain["option_type"] == "put"]
    if calls.empty:
        raise ValueError("chain has no call options")
    if puts.empty:
        raise ValueError("chain has no put options")

    call_diffs = (calls["delta"] - target_short_delta).abs()
    best_call_idx = call_diffs.idxmin()
    if call_diffs.loc[best_call_idx] > delta_tolerance:
        raise ValueError(
            f"no call within delta_tolerance={delta_tolerance} of target +{target_short_delta} "
            f"(closest |diff|={call_diffs.loc[best_call_idx]:.4f})"
        )

    put_diffs = (puts["delta"] - (-target_short_delta)).abs()
    best_put_idx = put_diffs.idxmin()
    if put_diffs.loc[best_put_idx] > delta_tolerance:
        raise ValueError(
            f"no put within delta_tolerance={delta_tolerance} of target -{target_short_delta} "
            f"(closest |diff|={put_diffs.loc[best_put_idx]:.4f})"
        )

    short_call_strike = float(calls.loc[best_call_idx, "strike"])
    short_put_strike = float(puts.loc[best_put_idx, "strike"])
    return short_call_strike, short_put_strike


def construct_iron_condor(
    chain: pd.DataFrame,
    underlying: str,
    expiry: date,
    *,
    target_short_delta: float = 0.12,
    wing_width: float = 5.0,
) -> IronCondorSpec:
    """Pick short strikes by delta, set long strikes wing_width away in OTM direction.

    The long call strike sits wing_width above the short call strike; the long put
    strike sits wing_width below the short put strike. The function does NOT verify
    that those long strikes exist in the chain — that's the caller's responsibility.

    Raises ValueError if no short strikes can be picked or wing_width is non-positive.
    """
    if wing_width <= 0:
        raise ValueError(f"wing_width must be positive, got {wing_width}")

    short_call_strike, short_put_strike = find_short_strikes_by_delta(
        chain, target_short_delta=target_short_delta
    )
    long_call_strike = short_call_strike + wing_width
    long_put_strike = short_put_strike - wing_width

    spec = IronCondorSpec(
        underlying=underlying,
        expiry=expiry,
        short_call_strike=short_call_strike,
        long_call_strike=long_call_strike,
        short_put_strike=short_put_strike,
        long_put_strike=long_put_strike,
    )
    if not spec.is_valid():
        raise ValueError(
            f"constructed iron condor failed structural validation: "
            f"short_put={short_put_strike} long_put={long_put_strike} "
            f"short_call={short_call_strike} long_call={long_call_strike}"
        )
    return spec


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


def _lookup_mid(chain: pd.DataFrame, strike: float, option_type: str) -> float:
    """Return mid price of the (strike, option_type) row. Raises if not found or NaN."""
    matches = chain[(chain["strike"] == strike) & (chain["option_type"] == option_type)]
    if matches.empty:
        raise ValueError(f"chain has no {option_type} at strike {strike}")
    if len(matches) > 1:
        raise ValueError(
            f"chain has {len(matches)} rows for {option_type} at strike {strike}; expected exactly 1"
        )
    row = matches.iloc[0]
    bid = float(row["bid"])
    ask = float(row["ask"])
    if math.isnan(bid) or math.isnan(ask):
        raise ValueError(f"NaN bid/ask for {option_type} at strike {strike}")
    if bid < 0 or ask < 0:
        raise ValueError(f"negative bid/ask for {option_type} at strike {strike}: bid={bid} ask={ask}")
    if ask < bid:
        raise ValueError(f"crossed market for {option_type} at strike {strike}: bid={bid} ask={ask}")
    return 0.5 * (bid + ask)


def compute_entry_credit(
    chain: pd.DataFrame,
    spec: IronCondorSpec,
) -> float:
    """Net credit received for opening one iron condor spread, in dollars (NOT cents).

    credit = (short_call_mid + short_put_mid) - (long_call_mid + long_put_mid)

    chain must contain rows at every leg strike with columns 'strike', 'option_type',
    'bid', 'ask'. A negative or zero result indicates an unfavourable/inverted
    market — caller should treat that as a no-go.
    """
    required_cols = {"strike", "option_type", "bid", "ask"}
    missing = required_cols - set(chain.columns)
    if missing:
        raise ValueError(f"chain missing required columns: {sorted(missing)}")

    short_call_mid = _lookup_mid(chain, spec.short_call_strike, "call")
    long_call_mid = _lookup_mid(chain, spec.long_call_strike, "call")
    short_put_mid = _lookup_mid(chain, spec.short_put_strike, "put")
    long_put_mid = _lookup_mid(chain, spec.long_put_strike, "put")
    return short_call_mid + short_put_mid - long_call_mid - long_put_mid


# ---------------------------------------------------------------------------
# Expiry selection
# ---------------------------------------------------------------------------


def select_expiry(
    available_expiries: list[date],
    today: date,
    target_dte_min: int = 30,
    target_dte_max: int = 45,
) -> date:
    """Return the expiry with DTE in [target_dte_min, target_dte_max] closest to the midpoint.

    Ties (equal distance to midpoint) are broken by picking the longer-dated expiry —
    longer DTE => more time premium, slower theta decay surprises.

    Raises ValueError if no expiry in the window or if the window is malformed.
    """
    if target_dte_min < 0 or target_dte_max < target_dte_min:
        raise ValueError(
            f"invalid DTE window: min={target_dte_min} max={target_dte_max}"
        )
    midpoint = (target_dte_min + target_dte_max) / 2.0

    candidates: list[tuple[float, int, date]] = []
    for exp in available_expiries:
        dte = (exp - today).days
        if target_dte_min <= dte <= target_dte_max:
            candidates.append((abs(dte - midpoint), dte, exp))

    if not candidates:
        raise ValueError(
            f"no expiry in [{target_dte_min}, {target_dte_max}] DTE relative to {today.isoformat()} "
            f"(available DTEs: {[(e - today).days for e in available_expiries]})"
        )

    # Sort by distance-to-midpoint ascending, then DTE descending (prefer longer on tie).
    candidates.sort(key=lambda t: (t[0], -t[1]))
    return candidates[0][2]


# ---------------------------------------------------------------------------
# Exit logic
# ---------------------------------------------------------------------------


def evaluate_exit(
    entry: IronCondorEntry,
    today: date,
    current_pnl: float,
    days_to_expiry: int,
) -> ExitDecision:
    """Apply Phase 2 VRP exit rules.

    Total credit received = entry_credit * quantity * multiplier. (entry_credit is per-contract.)

    Rules in priority order:
        1. profit_target : current_pnl >= +0.50 * total_credit
        2. stop_loss     : current_pnl <= -2.00 * total_credit
        3. 21dte_close   : days_to_expiry <= 21 (and not already triggered above)
        4. hold          : otherwise
    """
    spec = entry.spec
    total_credit = entry.entry_credit * spec.quantity * spec.multiplier
    target_pnl = 0.50 * total_credit
    stop_pnl = -2.00 * total_credit

    if current_pnl >= target_pnl:
        return ExitDecision(should_exit=True, reason="profit_target", target_pnl=target_pnl)
    if current_pnl <= stop_pnl:
        return ExitDecision(should_exit=True, reason="stop_loss", target_pnl=stop_pnl)
    if days_to_expiry <= 21:
        return ExitDecision(should_exit=True, reason="21dte_close", target_pnl=target_pnl)
    return ExitDecision(should_exit=False, reason="hold", target_pnl=target_pnl)


# ---------------------------------------------------------------------------
# Position-count gate
# ---------------------------------------------------------------------------


def position_count_check(open_positions: int, max_simultaneous: int = 6) -> bool:
    """True if we can open another position (i.e. open_positions < max_simultaneous)."""
    if open_positions < 0:
        raise ValueError(f"open_positions must be non-negative, got {open_positions}")
    if max_simultaneous <= 0:
        raise ValueError(f"max_simultaneous must be positive, got {max_simultaneous}")
    return open_positions < max_simultaneous
