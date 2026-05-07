"""Tests for `eval.gates` -- the Phase 3 pre-registered 7-gate evaluator.

Each gate is exercised in isolation (positive, negative, and boundary cases),
and the aggregators `evaluate_one_regime` / `evaluate_full` are exercised
against multi-gate scenarios at base and +50% cost regimes per
`metadata/strategic_lock.md`.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from eval.gates import (
    ADVERSARIAL_STRESS_MAX,
    DSR_MIN,
    DSR_T_MIN,
    GATE_NAMES,
    MAX_DRAWDOWN_MAX,
    NET_EXCESS_MIN_BPS,
    PBO_MAX,
    REGIME_LABEL_BASE,
    REGIME_LABEL_PLUS50,
    REGIME_UNDERPERFORMANCE_MAX,
    SAMPLE_SIZE_MIN,
    CostRegimeInputs,
    GateEvaluationReport,
    GateResult,
    evaluate_adversarial_stress,
    evaluate_dsr,
    evaluate_full,
    evaluate_max_drawdown,
    evaluate_net_excess,
    evaluate_one_regime,
    evaluate_pbo,
    evaluate_regime_stability,
    evaluate_sample_size,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _passing_inputs(**overrides) -> CostRegimeInputs:
    """Inputs that pass all 7 gates by a comfortable margin."""
    base = dict(
        dsr=0.6,
        dsr_t_stat=4.5,
        pbo=0.30,
        strategy_annualised_return=0.080,   # 800 bps strategy
        best_baseline_annualised_return=0.040,  # 400 bps baseline; 400 bps gap
        max_drawdown=0.10,
        adversarial_stress_drop=0.15,
        worst_regime_underperformance=-0.02,
        n_events=120,
    )
    base.update(overrides)
    return CostRegimeInputs(**base)


# ---------------------------------------------------------------------------
# Threshold sanity (sanity-check the contract values, not the tests)
# ---------------------------------------------------------------------------


def test_threshold_constants_match_strategic_lock() -> None:
    """Pinned threshold values from `metadata/strategic_lock.md`."""
    assert DSR_MIN == 0.0
    assert DSR_T_MIN == 3.84
    assert PBO_MAX == 0.5
    assert NET_EXCESS_MIN_BPS == 200
    assert MAX_DRAWDOWN_MAX == 0.15
    assert ADVERSARIAL_STRESS_MAX == 0.25
    assert REGIME_UNDERPERFORMANCE_MAX == 0.05
    assert SAMPLE_SIZE_MIN == 50
    assert GATE_NAMES == [
        "dsr",
        "pbo",
        "net_excess_bps",
        "max_drawdown",
        "adversarial_stress",
        "regime_stability",
        "sample_size",
    ]


# ---------------------------------------------------------------------------
# Gate 1: DSR
# ---------------------------------------------------------------------------


def test_evaluate_dsr_happy() -> None:
    result = evaluate_dsr(dsr=0.5, t_stat=4.0)
    assert isinstance(result, GateResult)
    assert result.name == "dsr"
    assert result.passed is True
    assert result.observed == {"dsr": 0.5, "t_stat": 4.0}
    assert result.threshold == {"dsr_min": DSR_MIN, "t_min": DSR_T_MIN}


def test_evaluate_dsr_t_too_low() -> None:
    result = evaluate_dsr(dsr=0.5, t_stat=3.0)
    assert result.passed is False


def test_evaluate_dsr_dsr_negative() -> None:
    result = evaluate_dsr(dsr=-0.1, t_stat=4.0)
    assert result.passed is False


def test_evaluate_dsr_boundary_t_at_threshold_passes() -> None:
    # t exactly at 3.84 should pass (>=).
    result = evaluate_dsr(dsr=0.0, t_stat=3.84)
    assert result.passed is True


# ---------------------------------------------------------------------------
# Gate 2: PBO
# ---------------------------------------------------------------------------


def test_evaluate_pbo_passes() -> None:
    assert evaluate_pbo(0.3).passed is True


def test_evaluate_pbo_fails() -> None:
    assert evaluate_pbo(0.6).passed is False


def test_evaluate_pbo_boundary_strict() -> None:
    # 0.5 must fail under the strict-less-than rule.
    assert evaluate_pbo(0.5).passed is False


# ---------------------------------------------------------------------------
# Gate 3: net excess (annualised, in bps)
# ---------------------------------------------------------------------------


def test_evaluate_net_excess_boundary_at_200bps_passes() -> None:
    # 5% strategy - 3% baseline = 200 bps gap exactly -> passes (>=).
    result = evaluate_net_excess(strategy_ann_ret=0.05, best_baseline_ann_ret=0.03)
    assert result.passed is True
    assert result.observed == pytest.approx(200.0)


def test_evaluate_net_excess_fails() -> None:
    # 4% - 3% = 100 bps gap -> fails.
    result = evaluate_net_excess(strategy_ann_ret=0.04, best_baseline_ann_ret=0.03)
    assert result.passed is False
    assert result.observed == pytest.approx(100.0)


def test_evaluate_net_excess_negative_excess_fails() -> None:
    result = evaluate_net_excess(strategy_ann_ret=0.02, best_baseline_ann_ret=0.05)
    assert result.passed is False
    assert result.observed == pytest.approx(-300.0)


# ---------------------------------------------------------------------------
# Gate 4: max drawdown
# ---------------------------------------------------------------------------


def test_evaluate_max_drawdown_passes() -> None:
    assert evaluate_max_drawdown(0.10).passed is True


def test_evaluate_max_drawdown_boundary_strict() -> None:
    # 15% must fail under strict-less-than.
    assert evaluate_max_drawdown(0.15).passed is False


def test_evaluate_max_drawdown_fails() -> None:
    assert evaluate_max_drawdown(0.20).passed is False


# ---------------------------------------------------------------------------
# Gate 5: adversarial stress
# ---------------------------------------------------------------------------


def test_evaluate_adversarial_stress_passes() -> None:
    assert evaluate_adversarial_stress(0.10).passed is True


def test_evaluate_adversarial_stress_boundary_strict() -> None:
    # Boundary 25% fails (strict <).
    assert evaluate_adversarial_stress(0.25).passed is False


def test_evaluate_adversarial_stress_fails() -> None:
    assert evaluate_adversarial_stress(0.30).passed is False


# ---------------------------------------------------------------------------
# Gate 6: regime stability
# ---------------------------------------------------------------------------


def test_evaluate_regime_stability_passes() -> None:
    # 3% behind in worst regime -> within 5% tolerance.
    assert evaluate_regime_stability(-0.03).passed is True


def test_evaluate_regime_stability_fails() -> None:
    # 7% behind in worst regime -> exceeds 5% tolerance.
    assert evaluate_regime_stability(-0.07).passed is False


def test_evaluate_regime_stability_boundary_at_minus_five_pct_passes() -> None:
    # -5% gap exactly is the boundary. Spec wording: 'no partition
    # underperforming best benchmark by > 5%'. Strict-greater-than -> the
    # boundary itself satisfies the gate. Test pins this choice.
    assert evaluate_regime_stability(-0.05).passed is True


def test_evaluate_regime_stability_positive_outperformance_passes() -> None:
    # Strategy outperformed even in worst regime: trivially passes.
    assert evaluate_regime_stability(0.02).passed is True


# ---------------------------------------------------------------------------
# Gate 7: sample size
# ---------------------------------------------------------------------------


def test_evaluate_sample_size_passes_at_floor() -> None:
    assert evaluate_sample_size(50).passed is True


def test_evaluate_sample_size_fails_below_floor() -> None:
    assert evaluate_sample_size(49).passed is False


def test_evaluate_sample_size_well_above_floor_passes() -> None:
    assert evaluate_sample_size(200).passed is True


# ---------------------------------------------------------------------------
# evaluate_one_regime
# ---------------------------------------------------------------------------


def test_evaluate_one_regime_all_pass() -> None:
    result = evaluate_one_regime(_passing_inputs(), regime_label="base")
    assert result.regime_label == "base"
    assert result.all_pass is True
    assert set(result.gates.keys()) == set(GATE_NAMES)
    for gate in result.gates.values():
        assert gate.passed is True


def test_evaluate_one_regime_partial_fail_only_dsr() -> None:
    # Only DSR fails -- everything else still recorded as passing.
    inputs = _passing_inputs(dsr=-0.1)  # negative DSR -> Gate 1 fails
    result = evaluate_one_regime(inputs, regime_label="base")
    assert result.all_pass is False
    assert result.gates["dsr"].passed is False
    other_gates = [n for n in GATE_NAMES if n != "dsr"]
    for name in other_gates:
        assert result.gates[name].passed is True, f"{name} should still pass"


def test_evaluate_one_regime_records_inputs_and_label() -> None:
    inputs = _passing_inputs()
    result = evaluate_one_regime(inputs, regime_label=REGIME_LABEL_PLUS50)
    assert result.inputs is inputs
    assert result.regime_label == "+50% costs"


# ---------------------------------------------------------------------------
# evaluate_full
# ---------------------------------------------------------------------------


def test_evaluate_full_happy_path() -> None:
    base_inputs = _passing_inputs()
    plus50_inputs = _passing_inputs()
    report = evaluate_full(base_inputs, plus50_inputs)
    assert isinstance(report, GateEvaluationReport)
    assert report.overall_pass is True
    assert report.failed_gates_summary == []
    assert report.base.regime_label == REGIME_LABEL_BASE
    assert report.plus50.regime_label == REGIME_LABEL_PLUS50
    assert report.base.all_pass is True
    assert report.plus50.all_pass is True


def test_evaluate_full_plus50_fails_net_excess_only() -> None:
    """Strategy meets 200 bps at base but only 150 bps at +50% costs.

    Net excess gate fails ONLY in the plus50 regime; everything else passes.
    Overall must therefore fail and `failed_gates_summary` must contain
    'plus50.net_excess_bps' -- and only that.
    """
    base_inputs = _passing_inputs(
        strategy_annualised_return=0.05,
        best_baseline_annualised_return=0.03,
    )
    # +50% costs eats 50 bps of the gap, so 5% - 3.5% = 150 bps -> fails.
    plus50_inputs = _passing_inputs(
        strategy_annualised_return=0.05,
        best_baseline_annualised_return=0.035,
    )

    report = evaluate_full(base_inputs, plus50_inputs)
    assert report.overall_pass is False
    assert report.base.all_pass is True
    assert report.plus50.all_pass is False
    assert report.plus50.gates["net_excess_bps"].passed is False
    assert report.failed_gates_summary == ["plus50.net_excess_bps"]


def test_evaluate_full_failed_gates_summary_lists_all_failures() -> None:
    """Both regimes can have multiple failures; summary preserves order."""
    base_inputs = _passing_inputs(dsr=-0.1)  # base fails Gate 1
    plus50_inputs = _passing_inputs(
        max_drawdown=0.20,           # plus50 fails Gate 4
        n_events=10,                 # plus50 fails Gate 7
    )
    report = evaluate_full(base_inputs, plus50_inputs)
    assert report.overall_pass is False
    # Order: all base failures first (in GATE_NAMES order), then plus50
    # failures (also in GATE_NAMES order).
    assert report.failed_gates_summary == [
        "base.dsr",
        "plus50.max_drawdown",
        "plus50.sample_size",
    ]


def test_evaluate_full_base_fails_overall_fails_even_if_plus50_passes() -> None:
    base_inputs = _passing_inputs(pbo=0.6)  # base fails PBO
    plus50_inputs = _passing_inputs()
    report = evaluate_full(base_inputs, plus50_inputs)
    assert report.overall_pass is False
    assert report.base.all_pass is False
    assert report.plus50.all_pass is True
    assert report.failed_gates_summary == ["base.pbo"]


def test_evaluate_full_independent_inputs_per_regime() -> None:
    """Inputs to base and plus50 are decoupled: changing one does not affect
    the other's gate evaluations."""
    base_inputs = _passing_inputs()
    plus50_inputs = replace(base_inputs, n_events=49)  # plus50 fails Gate 7
    report = evaluate_full(base_inputs, plus50_inputs)
    assert report.base.gates["sample_size"].passed is True
    assert report.plus50.gates["sample_size"].passed is False
    assert report.failed_gates_summary == ["plus50.sample_size"]
