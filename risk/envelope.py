"""Phase 2 pre-trade risk envelope.

Deterministic Python; no LLM, never can be overridden mid-cycle.

Implements the five hard gates from `metadata/capital_policy.md`:

  1. Single-position vega        > 5% NAV  -> reject
  2. Total portfolio vega        > 15% NAV -> reject (new positions)
  3. Single-name concentration   > 20% NAV -> reject
  4. Cash buffer floor           < 20% NAV -> stop opening
  5. Scenario stress drawdown    > 25% NAV (worst of 4) -> reject

The risk envelope runs *before* any trade is opened. Failure on any gate is a
hard rejection; gates are evaluated in parallel (all five always evaluated)
so the caller can see *every* failed gate, not just the first.

Units & conventions
-------------------
  - All dollar quantities in USD.
  - `delta`, `gamma`, `theta`, `vega`, `rho`: aggregate Greeks of the *position*,
    in conventional dollar units (i.e. delta is dollars-per-1%-spot-move where
    appropriate, vega is dollars-per-1-vol-point).
  - `notional_usd`: gross notional dollar exposure of the underlying leg
    (e.g. shares * price * multiplier).
  - Scenario stress shifts:
      spot_change is fractional (e.g. -0.05 = -5%).
      vol_change is in vol points (e.g. +10 = +10 IV points, +30 = adversarial).
  - First-order P&L approximation only:
        P&L = delta * spot_change * notional
            + 0.5 * gamma * (spot_change * notional) ** 2
            + vega * vol_change
    This is intentionally crude. It is a *gate*, not a pricing engine.

No external state: every check is a pure function of `(candidate, portfolio)`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np  # noqa: F401  (kept for parity with rest of risk stack)
import pandas as pd  # noqa: F401  (kept for parity with rest of risk stack)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidatePosition:
    """A candidate trade being evaluated by the risk envelope.

    A candidate is a *single proposed position* for one underlying. Multi-leg
    structures (iron condors, straddles) are represented as a single
    CandidatePosition with aggregate Greeks across all legs.
    """

    underlying: str
    sleeve: Literal["core", "satellite", "demo"]
    notional_usd: float            # gross notional (e.g. price * qty * mult)
    delta: float                   # aggregate delta of the position
    gamma: float
    theta: float
    vega: float
    rho: float
    max_loss_usd: float            # max possible loss (e.g. condor wing width)


@dataclass(frozen=True)
class PortfolioState:
    """Snapshot of the portfolio at the moment a candidate is being evaluated."""

    nav_usd: float
    cash_usd: float
    open_positions: list[CandidatePosition] = field(default_factory=list)


@dataclass(frozen=True)
class RiskCheckResult:
    """Aggregate result from `evaluate_envelope`.

    `passed` is True iff every gate passed. `failed_gates` lists the names
    of the gates that failed (empty when `passed`). `details` carries the
    numeric value reported by each gate keyed by gate name -- always populated
    for *every* gate, pass or fail, so downstream telemetry can record the
    full picture.
    """

    passed: bool
    failed_gates: list[str] = field(default_factory=list)
    details: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Individual gates
# ---------------------------------------------------------------------------


def check_single_position_vega(
    candidate: CandidatePosition,
    nav_usd: float,
    max_pct: float = 0.05,
) -> tuple[bool, float]:
    """Single-position vega cap (default 5% NAV).

    Returns (passed, candidate_vega_pct_of_nav).
    Magnitude is taken because a +vega and -vega exposure are both risky
    relative to a vol shock.
    """
    if nav_usd <= 0:
        return False, float("inf")
    candidate_vega_pct = abs(candidate.vega) / nav_usd
    return candidate_vega_pct <= max_pct, candidate_vega_pct


def check_total_vega(
    candidate: CandidatePosition,
    portfolio: PortfolioState,
    max_pct: float = 0.15,
) -> tuple[bool, float]:
    """Total portfolio vega cap (default 15% NAV).

    Sum of |vega| across all open positions plus the candidate.
    Returns (passed, total_vega_pct_of_nav).
    """
    if portfolio.nav_usd <= 0:
        return False, float("inf")
    open_vega = sum(abs(p.vega) for p in portfolio.open_positions)
    total_vega = open_vega + abs(candidate.vega)
    total_vega_pct = total_vega / portfolio.nav_usd
    return total_vega_pct <= max_pct, total_vega_pct


def check_concentration(
    candidate: CandidatePosition,
    portfolio: PortfolioState,
    max_pct: float = 0.20,
) -> tuple[bool, float]:
    """Single-name concentration cap (default 20% NAV).

    Concentration = sum(|notional_usd|) for *the candidate's underlying* across
    open positions plus the candidate, divided by NAV. Other underlyings do
    not contribute. Returns (passed, concentration_pct_of_nav).
    """
    if portfolio.nav_usd <= 0:
        return False, float("inf")
    same_name_open = sum(
        abs(p.notional_usd) for p in portfolio.open_positions
        if p.underlying == candidate.underlying
    )
    total_notional = same_name_open + abs(candidate.notional_usd)
    concentration_pct = total_notional / portfolio.nav_usd
    return concentration_pct <= max_pct, concentration_pct


def check_cash_buffer(
    candidate: CandidatePosition,
    portfolio: PortfolioState,
    min_pct: float = 0.20,
) -> tuple[bool, float]:
    """Cash buffer floor (default 20% NAV).

    Available cash *after* opening the trade must remain >= min_pct * nav.
    The candidate's `max_loss_usd` is the conservative cash reservation
    requirement (defined-risk strategies must escrow worst-case loss).
    Returns (passed, cash_after_pct_of_nav).
    """
    if portfolio.nav_usd <= 0:
        return False, 0.0
    cash_after = portfolio.cash_usd - abs(candidate.max_loss_usd)
    cash_after_pct = cash_after / portfolio.nav_usd
    return cash_after_pct >= min_pct, cash_after_pct


# ---------------------------------------------------------------------------
# Scenario stress
# ---------------------------------------------------------------------------


# Pre-registered stress scenarios. spot_change is fractional, vol_change is in
# vol points. These are the same scenarios the postmortem and Phase 2 risk
# review reference; do not edit without resetting the cycle.
SCENARIOS: tuple[tuple[str, float, float], ...] = (
    ("scen_a", -0.05, 10.0),     # spot -5%, vol +10
    ("scen_b", -0.10, 30.0),     # spot -10%, vol +30 (adversarial)
    ("scen_c", +0.05, 5.0),      # spot +5%, vol +5
    ("scen_d", 0.0, -5.0),       # spot flat, vol -5
)


def _position_pnl(
    pos: CandidatePosition,
    spot_change: float,
    vol_change: float,
) -> float:
    """First-order P&L approximation for a single position under a shock.

    P&L = delta * dS + 0.5 * gamma * dS^2 + vega * d_vol

    where dS = spot_change * notional. This is the formula specified by the
    risk envelope contract; it is a hard-gate approximation, not a pricing
    model. Theta / rho are deliberately excluded -- the gate evaluates an
    instantaneous shock, not a time evolution.
    """
    dS = spot_change * pos.notional_usd
    return pos.delta * dS + 0.5 * pos.gamma * (dS ** 2) + pos.vega * vol_change


def check_scenario_stress(
    candidate: CandidatePosition,
    portfolio: PortfolioState,
    *,
    max_dd_pct: float = 0.25,
) -> tuple[bool, float]:
    """Scenario stress gate (default worst-case drawdown < 25% NAV).

    Evaluates four pre-registered scenarios over the entire portfolio
    (existing open positions + candidate), takes the worst (most negative)
    aggregated P&L, expresses it as a fraction of NAV, and rejects if it
    exceeds `max_dd_pct`.

    Returns (passed, worst_scenario_pct_of_nav).
    `worst_scenario_pct_of_nav` is reported as a *positive* fraction
    representing the magnitude of drawdown (e.g. 0.30 = 30% NAV drop).
    For scenarios where every position gains, the value is 0.0.
    """
    if portfolio.nav_usd <= 0:
        return False, float("inf")

    all_positions = list(portfolio.open_positions) + [candidate]
    worst_pnl = 0.0  # most negative P&L found
    for _name, spot_change, vol_change in SCENARIOS:
        scenario_pnl = sum(
            _position_pnl(p, spot_change, vol_change) for p in all_positions
        )
        if scenario_pnl < worst_pnl:
            worst_pnl = scenario_pnl

    worst_dd_pct = -worst_pnl / portfolio.nav_usd  # positive number = drop
    return worst_dd_pct <= max_dd_pct, worst_dd_pct


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


_GATE_NAMES = (
    "single_position_vega",
    "total_vega",
    "concentration",
    "cash_buffer",
    "scenario_stress",
)


def evaluate_envelope(
    candidate: CandidatePosition,
    portfolio: PortfolioState,
) -> RiskCheckResult:
    """Run all five hard gates and return an aggregate `RiskCheckResult`.

    Every gate is always evaluated; gates do *not* short-circuit. This means
    the caller (and downstream telemetry) sees the full set of failures, not
    only the first. `passed` is True iff every gate passed; `failed_gates`
    lists the names of the failed gates; `details` records the numeric value
    each gate computed (vega %, concentration %, worst drawdown %, etc.).
    """
    pass_spv, val_spv = check_single_position_vega(candidate, portfolio.nav_usd)
    pass_tv, val_tv = check_total_vega(candidate, portfolio)
    pass_conc, val_conc = check_concentration(candidate, portfolio)
    pass_cash, val_cash = check_cash_buffer(candidate, portfolio)
    pass_stress, val_stress = check_scenario_stress(candidate, portfolio)

    results = (
        ("single_position_vega", pass_spv, val_spv),
        ("total_vega", pass_tv, val_tv),
        ("concentration", pass_conc, val_conc),
        ("cash_buffer", pass_cash, val_cash),
        ("scenario_stress", pass_stress, val_stress),
    )

    failed_gates = [name for name, passed, _ in results if not passed]
    details = {name: float(value) for name, _passed, value in results}

    return RiskCheckResult(
        passed=len(failed_gates) == 0,
        failed_gates=failed_gates,
        details=details,
    )
