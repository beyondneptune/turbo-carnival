"""Phase 3 pre-registered gate evaluator.

Takes pre-computed metric values (from eval/metrics.py + eval/baselines.py) and
applies the 7-gate check defined in `metadata/strategic_lock.md`. Pure data-in,
data-out -- no module imports beyond stdlib + numpy/pandas. Each gate is checked
at Base AND +50% costs; the overall pass requires all 7 gates to hold in both
cost regimes simultaneously.

This module deliberately accepts pre-computed metric VALUES as inputs (rather
than module references) so the evaluator is self-contained and can run from any
source of metrics: live `eval/metrics.py`, manual entry, or replayed history.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Gate names + thresholds
# ---------------------------------------------------------------------------

GateName = Literal[
    "dsr",                    # Gate 1: DSR >= 0, t >= 3.84
    "pbo",                    # Gate 2: PBO < 0.5
    "net_excess_bps",         # Gate 3: net excess vs best baseline >= 200 bps
    "max_drawdown",           # Gate 4: max DD < 15%
    "adversarial_stress",     # Gate 5: adversarial drop < 25% relative
    "regime_stability",       # Gate 6: no HMM regime drop > 5%
    "sample_size",            # Gate 7: n_events >= 50
]

GATE_NAMES: list[GateName] = [
    "dsr",
    "pbo",
    "net_excess_bps",
    "max_drawdown",
    "adversarial_stress",
    "regime_stability",
    "sample_size",
]

# Pre-registered numeric thresholds. These are the contract with
# `metadata/strategic_lock.md` -- changes here require CEO override and
# reset any in-progress cycle.
DSR_MIN = 0.0
DSR_T_MIN = 3.84
PBO_MAX = 0.5
NET_EXCESS_MIN_BPS = 200          # annualised basis points
MAX_DRAWDOWN_MAX = 0.15           # fraction of NAV
ADVERSARIAL_STRESS_MAX = 0.25     # fraction relative loss
REGIME_UNDERPERFORMANCE_MAX = 0.05  # worst gap >= -0.05 (i.e. <= 5% under)
SAMPLE_SIZE_MIN = 50

REGIME_LABEL_BASE = "base"
REGIME_LABEL_PLUS50 = "+50% costs"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateResult:
    """Outcome of a single gate check."""

    name: GateName
    passed: bool
    observed: float | dict       # the measured value(s)
    threshold: float | dict      # the required value(s)
    detail: str = ""             # human-readable note


@dataclass(frozen=True)
class CostRegimeInputs:
    """Inputs for one cost regime (base or +50% costs).

    All return-style fields are fractions, not percentages. For example,
    `strategy_annualised_return = 0.12` means +12% annualised. Drawdown and
    stress drop are expressed as positive fractions of loss.

    `worst_regime_underperformance` is the most-negative gap to the best
    benchmark across HMM regime partitions. -0.07 means the strategy lost
    7 percentage points to the benchmark in the worst regime.
    """

    dsr: float
    dsr_t_stat: float
    pbo: float
    strategy_annualised_return: float
    best_baseline_annualised_return: float
    max_drawdown: float
    adversarial_stress_drop: float
    worst_regime_underperformance: float
    n_events: int


@dataclass(frozen=True)
class CostRegimeResult:
    """Result of running all 7 gate checks for a single cost regime."""

    regime_label: str
    inputs: CostRegimeInputs
    gates: dict[GateName, GateResult]
    all_pass: bool


@dataclass(frozen=True)
class GateEvaluationReport:
    """Final report aggregating both cost regimes.

    `overall_pass` is True iff every gate passes in both regimes.
    `failed_gates_summary` lists `<regime>.<gate_name>` strings -- e.g.
    `plus50.net_excess_bps` -- for any gate that failed in either regime.
    """

    base: CostRegimeResult
    plus50: CostRegimeResult
    overall_pass: bool
    failed_gates_summary: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Individual gate evaluators
# ---------------------------------------------------------------------------


def evaluate_dsr(dsr: float, t_stat: float) -> GateResult:
    """Gate 1: DSR >= 0 AND t-statistic >= 3.84 (Bailey & Lopez de Prado).

    Both conditions must hold; either failing fails the gate.
    """
    passed = dsr >= DSR_MIN and t_stat >= DSR_T_MIN
    return GateResult(
        name="dsr",
        passed=passed,
        observed={"dsr": float(dsr), "t_stat": float(t_stat)},
        threshold={"dsr_min": DSR_MIN, "t_min": DSR_T_MIN},
        detail=(
            f"dsr={dsr:.4f} (>= {DSR_MIN}); "
            f"t={t_stat:.4f} (>= {DSR_T_MIN})"
        ),
    )


def evaluate_pbo(pbo: float) -> GateResult:
    """Gate 2: PBO < 0.5 (strict). Boundary 0.5 fails."""
    passed = pbo < PBO_MAX
    return GateResult(
        name="pbo",
        passed=passed,
        observed=float(pbo),
        threshold=PBO_MAX,
        detail=f"pbo={pbo:.4f} (< {PBO_MAX} required, strict)",
    )


def evaluate_net_excess(
    strategy_ann_ret: float,
    best_baseline_ann_ret: float,
) -> GateResult:
    """Gate 3: annualised excess vs `max(B0..B3)` >= 200 bps.

    The 200 bps threshold is inclusive (strategy beats baseline by >=200 bps).
    """
    excess_fraction = strategy_ann_ret - best_baseline_ann_ret
    excess_bps = excess_fraction * 10_000.0
    passed = excess_bps >= NET_EXCESS_MIN_BPS
    return GateResult(
        name="net_excess_bps",
        passed=passed,
        observed=float(excess_bps),
        threshold=float(NET_EXCESS_MIN_BPS),
        detail=(
            f"strategy={strategy_ann_ret:.4f}, "
            f"baseline={best_baseline_ann_ret:.4f}, "
            f"excess={excess_bps:.2f} bps (>= {NET_EXCESS_MIN_BPS} required)"
        ),
    )


def evaluate_max_drawdown(max_dd: float) -> GateResult:
    """Gate 4: max drawdown < 15% NAV (strict). Boundary 15% fails.

    `max_dd` is a positive fraction (0.10 means a 10% drawdown).
    """
    passed = max_dd < MAX_DRAWDOWN_MAX
    return GateResult(
        name="max_drawdown",
        passed=passed,
        observed=float(max_dd),
        threshold=MAX_DRAWDOWN_MAX,
        detail=f"max_dd={max_dd:.4f} (< {MAX_DRAWDOWN_MAX} required, strict)",
    )


def evaluate_adversarial_stress(stress_drop: float) -> GateResult:
    """Gate 5: adversarial stress drop < 25% (strict). Boundary 25% fails.

    `stress_drop` is a positive fraction representing relative loss in the
    worst-modelled adversarial scenario.
    """
    passed = stress_drop < ADVERSARIAL_STRESS_MAX
    return GateResult(
        name="adversarial_stress",
        passed=passed,
        observed=float(stress_drop),
        threshold=ADVERSARIAL_STRESS_MAX,
        detail=(
            f"stress_drop={stress_drop:.4f} "
            f"(< {ADVERSARIAL_STRESS_MAX} required, strict)"
        ),
    )


def evaluate_regime_stability(worst_underperformance: float) -> GateResult:
    """Gate 6: no HMM partition underperforming best benchmark by > 5%.

    `worst_underperformance` is the most-negative gap across regime partitions.
    A value of -0.03 means the strategy was 3 percentage points behind in the
    worst regime, which is acceptable. -0.07 means 7 pp behind, which fails.
    The gate passes iff `worst_underperformance >= -REGIME_UNDERPERFORMANCE_MAX`.
    """
    passed = worst_underperformance >= -REGIME_UNDERPERFORMANCE_MAX
    return GateResult(
        name="regime_stability",
        passed=passed,
        observed=float(worst_underperformance),
        threshold=-REGIME_UNDERPERFORMANCE_MAX,
        detail=(
            f"worst_underperformance={worst_underperformance:.4f} "
            f"(>= {-REGIME_UNDERPERFORMANCE_MAX} required)"
        ),
    )


def evaluate_sample_size(n_events: int) -> GateResult:
    """Gate 7: sample size floor of 50 events (inclusive).

    The strategic lock notes 'aim for >=150 if feasibility allows', but the
    hard pre-registered floor is 50.
    """
    passed = n_events >= SAMPLE_SIZE_MIN
    return GateResult(
        name="sample_size",
        passed=passed,
        observed=float(n_events),
        threshold=float(SAMPLE_SIZE_MIN),
        detail=f"n_events={n_events} (>= {SAMPLE_SIZE_MIN} required)",
    )


# ---------------------------------------------------------------------------
# Per-regime + full-report aggregators
# ---------------------------------------------------------------------------


def evaluate_one_regime(
    inputs: CostRegimeInputs,
    regime_label: str,
) -> CostRegimeResult:
    """Run all 7 gate checks for a single cost regime.

    `all_pass` is True iff all 7 individual gates pass. Every gate is recorded
    in `gates`, regardless of outcome, so failures can be inspected without
    re-running.
    """
    gates: dict[GateName, GateResult] = {
        "dsr": evaluate_dsr(inputs.dsr, inputs.dsr_t_stat),
        "pbo": evaluate_pbo(inputs.pbo),
        "net_excess_bps": evaluate_net_excess(
            inputs.strategy_annualised_return,
            inputs.best_baseline_annualised_return,
        ),
        "max_drawdown": evaluate_max_drawdown(inputs.max_drawdown),
        "adversarial_stress": evaluate_adversarial_stress(
            inputs.adversarial_stress_drop
        ),
        "regime_stability": evaluate_regime_stability(
            inputs.worst_regime_underperformance
        ),
        "sample_size": evaluate_sample_size(inputs.n_events),
    }
    all_pass = all(gate.passed for gate in gates.values())
    return CostRegimeResult(
        regime_label=regime_label,
        inputs=inputs,
        gates=gates,
        all_pass=all_pass,
    )


def evaluate_full(
    base_inputs: CostRegimeInputs,
    plus50_inputs: CostRegimeInputs,
) -> GateEvaluationReport:
    """Run gate checks at base and +50% cost regimes; aggregate.

    `overall_pass` is True iff every gate passes in both regimes.
    `failed_gates_summary` lists `<regime>.<gate_name>` strings for every
    gate that failed in either regime, in canonical (regime, GATE_NAMES) order.
    """
    base = evaluate_one_regime(base_inputs, REGIME_LABEL_BASE)
    plus50 = evaluate_one_regime(plus50_inputs, REGIME_LABEL_PLUS50)

    failed: list[str] = []
    for prefix, regime_result in (("base", base), ("plus50", plus50)):
        for gate_name in GATE_NAMES:
            gate = regime_result.gates[gate_name]
            if not gate.passed:
                failed.append(f"{prefix}.{gate_name}")

    overall_pass = base.all_pass and plus50.all_pass
    return GateEvaluationReport(
        base=base,
        plus50=plus50,
        overall_pass=overall_pass,
        failed_gates_summary=failed,
    )
