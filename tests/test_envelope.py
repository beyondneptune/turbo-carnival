"""Tests for `risk.envelope` -- the Phase 2 pre-trade risk gate envelope.

Each gate is exercised in isolation (positive and negative cases), and the
aggregator `evaluate_envelope` is exercised against multi-gate scenarios.
Test scenarios use a $10,000 paper NAV per `metadata/capital_policy.md`.
"""
from __future__ import annotations

import pytest

from risk.envelope import (
    CandidatePosition,
    PortfolioState,
    RiskCheckResult,
    check_cash_buffer,
    check_concentration,
    check_scenario_stress,
    check_single_position_vega,
    check_total_vega,
    evaluate_envelope,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


NAV = 10_000.0


def _flat(
    underlying: str = "SPY",
    sleeve: str = "core",
    notional_usd: float = 1_000.0,
    delta: float = 0.0,
    gamma: float = 0.0,
    theta: float = 0.0,
    vega: float = 0.0,
    rho: float = 0.0,
    max_loss_usd: float = 100.0,
) -> CandidatePosition:
    """Convenience builder -- a flat (zero-greek) candidate by default."""
    return CandidatePosition(
        underlying=underlying,
        sleeve=sleeve,  # type: ignore[arg-type]
        notional_usd=notional_usd,
        delta=delta,
        gamma=gamma,
        theta=theta,
        vega=vega,
        rho=rho,
        max_loss_usd=max_loss_usd,
    )


def _portfolio(
    nav_usd: float = NAV,
    cash_usd: float = 5_000.0,
    open_positions: list[CandidatePosition] | None = None,
) -> PortfolioState:
    return PortfolioState(
        nav_usd=nav_usd,
        cash_usd=cash_usd,
        open_positions=open_positions or [],
    )


# ---------------------------------------------------------------------------
# Single-position vega gate
# ---------------------------------------------------------------------------


def test_single_position_vega_passes_at_4pct():
    cand = _flat(vega=400.0)  # 400 / 10_000 = 4% < 5% cap
    passed, val = check_single_position_vega(cand, NAV)
    assert passed
    assert val == pytest.approx(0.04)


def test_single_position_vega_fails_at_6pct():
    cand = _flat(vega=600.0)  # 600 / 10_000 = 6% > 5% cap
    passed, val = check_single_position_vega(cand, NAV)
    assert not passed
    assert val == pytest.approx(0.06)


def test_single_position_vega_uses_magnitude_of_vega():
    """A short-vol candidate (-vega) is just as risky as a long-vol candidate."""
    cand = _flat(vega=-600.0)
    passed, val = check_single_position_vega(cand, NAV)
    assert not passed
    assert val == pytest.approx(0.06)


def test_single_position_vega_at_exact_5pct_passes():
    """5% is the inclusive boundary."""
    cand = _flat(vega=500.0)
    passed, val = check_single_position_vega(cand, NAV)
    assert passed
    assert val == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Total portfolio vega gate
# ---------------------------------------------------------------------------


def test_total_vega_fails_when_open_plus_candidate_exceed_15pct():
    """Open positions sum |vega| = 12% NAV, candidate adds 4% -> 16% > 15%."""
    open_positions = [
        _flat(vega=600.0),                       # 6%
        _flat(underlying="QQQ", vega=-600.0),    # 6% (magnitude)
    ]
    cand = _flat(underlying="IWM", vega=400.0)   # +4%
    portfolio = _portfolio(open_positions=open_positions)
    passed, val = check_total_vega(cand, portfolio)
    assert not passed
    assert val == pytest.approx(0.16)


def test_total_vega_passes_when_below_15pct():
    open_positions = [_flat(vega=500.0)]  # 5%
    cand = _flat(vega=400.0)              # 4%
    portfolio = _portfolio(open_positions=open_positions)
    passed, val = check_total_vega(cand, portfolio)
    assert passed
    assert val == pytest.approx(0.09)


def test_total_vega_no_open_positions():
    cand = _flat(vega=300.0)  # 3%
    portfolio = _portfolio()
    passed, val = check_total_vega(cand, portfolio)
    assert passed
    assert val == pytest.approx(0.03)


# ---------------------------------------------------------------------------
# Concentration gate
# ---------------------------------------------------------------------------


def test_concentration_fails_when_same_name_exceeds_20pct():
    """Existing SPY 18% notional, candidate adds SPY 5% -> 23% > 20%."""
    open_positions = [_flat(underlying="SPY", notional_usd=1_800.0)]
    cand = _flat(underlying="SPY", notional_usd=500.0)
    portfolio = _portfolio(open_positions=open_positions)
    passed, val = check_concentration(cand, portfolio)
    assert not passed
    assert val == pytest.approx(0.23)


def test_concentration_passes_when_different_underlying():
    """Existing AAPL 18%, candidate SPY 10% -> SPY-only concentration is 10%."""
    open_positions = [_flat(underlying="AAPL", notional_usd=1_800.0)]
    cand = _flat(underlying="SPY", notional_usd=1_000.0)
    portfolio = _portfolio(open_positions=open_positions)
    passed, val = check_concentration(cand, portfolio)
    assert passed
    assert val == pytest.approx(0.10)


def test_concentration_at_exact_20pct_passes():
    open_positions = [_flat(underlying="SPY", notional_usd=1_500.0)]
    cand = _flat(underlying="SPY", notional_usd=500.0)
    portfolio = _portfolio(open_positions=open_positions)
    passed, val = check_concentration(cand, portfolio)
    assert passed
    assert val == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# Cash buffer gate
# ---------------------------------------------------------------------------


def test_cash_buffer_fails_when_post_trade_cash_below_20pct():
    """nav=10_000, cash=1_500, candidate uses 100 -> cash after = 1_400 (14%)."""
    cand = _flat(max_loss_usd=100.0)
    portfolio = _portfolio(cash_usd=1_500.0)
    passed, val = check_cash_buffer(cand, portfolio)
    assert not passed
    assert val == pytest.approx(0.14)


def test_cash_buffer_passes_with_ample_cash():
    cand = _flat(max_loss_usd=400.0)
    portfolio = _portfolio(cash_usd=5_000.0)
    passed, val = check_cash_buffer(cand, portfolio)
    assert passed
    assert val == pytest.approx(0.46)


def test_cash_buffer_at_exact_20pct_passes():
    cand = _flat(max_loss_usd=500.0)
    portfolio = _portfolio(cash_usd=2_500.0)
    passed, val = check_cash_buffer(cand, portfolio)
    assert passed
    assert val == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# Scenario stress gate
# ---------------------------------------------------------------------------


def test_scenario_stress_passes_for_iron_condor():
    """Defined-risk iron condor: small vega, small gamma, low max loss.

    All four scenarios should produce drawdown < 25% NAV.
    """
    cand = _flat(
        underlying="SPY",
        notional_usd=2_000.0,
        delta=0.0,
        gamma=-1e-4,         # short gamma, but tiny
        vega=-5.0,           # short vol, but tiny
        max_loss_usd=400.0,
    )
    portfolio = _portfolio()
    passed, val = check_scenario_stress(cand, portfolio)
    assert passed
    assert val < 0.25


def test_scenario_stress_fails_for_naked_short_vega():
    """A position short $100 of vega gets crushed by scen_b: -100 * 30 = -$3000.

    On a $10_000 NAV that is a 30% drop, breaching the 25% adversarial floor.
    """
    cand = _flat(
        underlying="SPY",
        notional_usd=1_000.0,
        delta=0.0,
        gamma=0.0,
        vega=-100.0,         # short vol
        max_loss_usd=200.0,
    )
    portfolio = _portfolio()
    passed, val = check_scenario_stress(cand, portfolio)
    assert not passed
    assert val == pytest.approx(0.30)


def test_scenario_stress_aggregates_open_positions_with_candidate():
    """Open positions with offsetting vega should cancel out in the stress."""
    open_positions = [
        _flat(underlying="QQQ", vega=+100.0),  # long vol
    ]
    cand = _flat(
        underlying="SPY",
        notional_usd=1_000.0,
        vega=-100.0,                            # short vol cancels
    )
    portfolio = _portfolio(open_positions=open_positions)
    passed, val = check_scenario_stress(cand, portfolio)
    assert passed
    assert val == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# evaluate_envelope aggregator
# ---------------------------------------------------------------------------


def test_evaluate_envelope_passes_when_all_gates_pass():
    cand = _flat(
        underlying="SPY",
        notional_usd=1_500.0,
        delta=0.0,
        gamma=-1e-4,
        vega=-5.0,
        max_loss_usd=400.0,
    )
    portfolio = _portfolio(cash_usd=5_000.0)
    result = evaluate_envelope(cand, portfolio)
    assert isinstance(result, RiskCheckResult)
    assert result.passed
    assert result.failed_gates == []


def test_evaluate_envelope_fails_with_correct_failed_gates_list():
    """Trip the concentration gate in isolation; all other gates should pass."""
    open_positions = [
        _flat(underlying="SPY", notional_usd=1_900.0,
              vega=0.0, gamma=0.0, delta=0.0, max_loss_usd=0.0),
    ]
    cand = _flat(
        underlying="SPY",
        notional_usd=500.0,         # SPY total = 2_400 / 10_000 = 24% -> FAIL
        delta=0.0,
        gamma=0.0,
        vega=0.0,
        max_loss_usd=100.0,
    )
    portfolio = _portfolio(cash_usd=5_000.0, open_positions=open_positions)
    result = evaluate_envelope(cand, portfolio)
    assert not result.passed
    assert result.failed_gates == ["concentration"]


def test_evaluate_envelope_fails_lists_multiple_gates_when_multiple_fail():
    """A vega-heavy candidate should trip both single-position-vega and
    scenario-stress simultaneously, demonstrating gates do not short-circuit.
    """
    cand = _flat(
        underlying="SPY",
        notional_usd=1_000.0,
        vega=-800.0,                # 8% NAV vega -> single-vega FAIL
        max_loss_usd=200.0,         # but cash buffer fine
    )
    portfolio = _portfolio(cash_usd=5_000.0)
    result = evaluate_envelope(cand, portfolio)
    assert not result.passed
    assert "single_position_vega" in result.failed_gates
    assert "scenario_stress" in result.failed_gates


def test_evaluate_envelope_details_dict_has_all_numeric_values():
    cand = _flat(
        underlying="SPY",
        notional_usd=1_500.0,
        delta=0.0,
        gamma=-1e-4,
        vega=-5.0,
        max_loss_usd=400.0,
    )
    portfolio = _portfolio(cash_usd=5_000.0)
    result = evaluate_envelope(cand, portfolio)
    expected_keys = {
        "single_position_vega",
        "total_vega",
        "concentration",
        "cash_buffer",
        "scenario_stress",
    }
    assert set(result.details.keys()) == expected_keys
    for key, val in result.details.items():
        assert isinstance(val, float), f"{key} should be float, got {type(val)}"


def test_evaluate_envelope_details_dict_populated_on_failure_too():
    """Even on failure, details should hold every gate's numeric value."""
    cand = _flat(vega=900.0, max_loss_usd=100.0)  # vega 9% -> fail
    portfolio = _portfolio(cash_usd=5_000.0)
    result = evaluate_envelope(cand, portfolio)
    assert not result.passed
    assert result.details["single_position_vega"] == pytest.approx(0.09)
    # Other gates still recorded
    assert "concentration" in result.details
    assert "cash_buffer" in result.details
    assert "total_vega" in result.details
    assert "scenario_stress" in result.details
