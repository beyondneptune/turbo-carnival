"""Tests for eval.metrics — DSR, PBO, Sharpe, drawdown, regime stability."""
from __future__ import annotations

import numpy as np
import pytest

from eval.metrics import (
    RegimeStabilityResult,
    annualized_return,
    annualized_sharpe,
    deflated_sharpe_ratio,
    evaluate_regime_stability,
    max_drawdown,
    probability_backtest_overfitting,
)


# ---------------------------------------------------------------------------
# annualized_sharpe
# ---------------------------------------------------------------------------


def test_annualized_sharpe_known_value():
    rng = np.random.default_rng(0)
    # Constant per-period mean and std → SR = mean / std * sqrt(252).
    arr = rng.normal(loc=0.001, scale=0.01, size=2000)
    sr = annualized_sharpe(arr)
    expected = float(np.mean(arr) / np.std(arr, ddof=1) * np.sqrt(252))
    assert sr == pytest.approx(expected, rel=1e-12)


def test_annualized_sharpe_zero_std():
    arr = [0.01] * 100
    assert annualized_sharpe(arr) == 0.0


def test_annualized_sharpe_empty():
    assert annualized_sharpe([]) == 0.0


def test_annualized_sharpe_single_obs():
    assert annualized_sharpe([0.05]) == 0.0


def test_annualized_sharpe_periods_per_year_param():
    rng = np.random.default_rng(1)
    arr = rng.normal(0.0005, 0.02, 1000)
    sr_252 = annualized_sharpe(arr, periods_per_year=252)
    sr_12 = annualized_sharpe(arr, periods_per_year=12)
    # Same underlying ratio, just different annualization factor.
    assert sr_252 == pytest.approx(sr_12 * np.sqrt(252.0 / 12.0), rel=1e-12)


# ---------------------------------------------------------------------------
# max_drawdown
# ---------------------------------------------------------------------------


def test_max_drawdown_monotone_up():
    arr = [0.01] * 50
    assert max_drawdown(arr) == pytest.approx(0.0, abs=1e-15)


def test_max_drawdown_empty():
    assert max_drawdown([]) == 0.0


def test_max_drawdown_known_case():
    # Equity goes 1.0 -> 1.10 -> 0.99 → drawdown = 1 - 0.99/1.10 = 0.10.
    # Use returns: +10%, then a return that takes equity from 1.10 to 0.99
    # (i.e. r2 = 0.99/1.10 - 1 ≈ -0.10).
    r1 = 0.10
    r2 = 0.99 / 1.10 - 1.0
    arr = [r1, r2]
    assert max_drawdown(arr) == pytest.approx(0.10, abs=1e-12)


def test_max_drawdown_severe_loss():
    # 50% loss in one period.
    assert max_drawdown([-0.5]) == pytest.approx(0.5, abs=1e-12)


def test_max_drawdown_recovery():
    # Up then down then up — the max trough is between peak and recovery.
    arr = [0.20, -0.20, 0.05]
    # Equity: 1.20, 0.96, 1.008. Peak = 1.20. Worst ratio = 0.96/1.20 = 0.80.
    assert max_drawdown(arr) == pytest.approx(0.20, abs=1e-12)


# ---------------------------------------------------------------------------
# annualized_return
# ---------------------------------------------------------------------------


def test_annualized_return_known():
    arr = [0.001] * 252
    # mean = 0.001; arithmetic annualized = 0.001 * 252 = 0.252.
    assert annualized_return(arr) == pytest.approx(0.252, rel=1e-12)


def test_annualized_return_empty():
    assert annualized_return([]) == 0.0


def test_annualized_return_negative():
    arr = [-0.002] * 252
    assert annualized_return(arr) == pytest.approx(-0.504, rel=1e-12)


# ---------------------------------------------------------------------------
# deflated_sharpe_ratio
# ---------------------------------------------------------------------------


def test_dsr_n_trials_one_positive():
    rng = np.random.default_rng(42)
    arr = rng.normal(0.001, 0.01, 1000)  # positive Sharpe
    dsr, t_stat = deflated_sharpe_ratio(arr, n_trials=1)
    assert dsr > 0.0
    assert t_stat > 0.0
    # With n_trials=1 and target_sr=0, the baseline SR_0 = 0, so DSR is just
    # the variance-corrected t-stat. It should be in the same ballpark.
    assert dsr == pytest.approx(t_stat, rel=0.5)


def test_dsr_deflation_with_many_trials():
    # When we declare many trials, the noise floor SR_0 grows and DSR shrinks.
    rng = np.random.default_rng(7)
    arr = rng.normal(0.0008, 0.012, 1000)
    dsr_1, t_stat = deflated_sharpe_ratio(arr, n_trials=1)
    dsr_100, _ = deflated_sharpe_ratio(arr, n_trials=100)
    # Both should be measured against the same t_stat baseline; 100-trial
    # case should give a smaller DSR than the 1-trial case.
    assert dsr_100 < dsr_1
    # Deflation should bite — DSR with 100 trials < classical t_stat.
    assert dsr_100 < t_stat


def test_dsr_zero_std_returns_zero():
    arr = [0.01] * 200
    dsr, t_stat = deflated_sharpe_ratio(arr, n_trials=10)
    assert dsr == 0.0
    assert t_stat == 0.0


def test_dsr_empty_returns_zero():
    dsr, t_stat = deflated_sharpe_ratio([], n_trials=5)
    assert dsr == 0.0
    assert t_stat == 0.0


def test_dsr_single_obs_returns_zero():
    dsr, t_stat = deflated_sharpe_ratio([0.01], n_trials=5)
    assert dsr == 0.0
    assert t_stat == 0.0


def test_dsr_target_sr_subtracts_hurdle():
    rng = np.random.default_rng(11)
    arr = rng.normal(0.0005, 0.01, 1000)
    dsr_no_hurdle, t0 = deflated_sharpe_ratio(arr, n_trials=1, target_sr=0.0)
    dsr_hurdle, t1 = deflated_sharpe_ratio(arr, n_trials=1, target_sr=1.0)
    # Hurdle should reduce both DSR and t-stat (testing against a stricter null).
    assert dsr_hurdle < dsr_no_hurdle
    assert t1 < t0


# ---------------------------------------------------------------------------
# probability_backtest_overfitting
# ---------------------------------------------------------------------------


def test_pbo_dominant_strategy_zero():
    # Strategy 0 always beats strategy 1: deterministic spread per period.
    rng = np.random.default_rng(0)
    T = 320
    base = rng.normal(0.0, 0.01, T)
    M = np.column_stack([base + 0.005, base - 0.005])
    pbo = probability_backtest_overfitting(M, n_splits=8)
    # Strategy 0 dominates IS and OOS in every split → never below median.
    assert pbo == 0.0


def test_pbo_dominated_strategy_one():
    # Inverse: strategy 1 always wins. The IS-best in every split is
    # strategy 1, so it should also win OOS → still PBO = 0.
    rng = np.random.default_rng(1)
    T = 320
    base = rng.normal(0.0, 0.01, T)
    M = np.column_stack([base - 0.005, base + 0.005])
    pbo = probability_backtest_overfitting(M, n_splits=8)
    assert pbo == 0.0


def test_pbo_random_strategies_around_one_half():
    # Average PBO across many seeds for iid 5-column data; with truly random
    # strategies the IS-best vs OOS-best correlation is ~0, so the long-run
    # mean PBO should sit near 0.5.
    pbos = []
    for seed in range(40):
        rng = np.random.default_rng(seed)
        T = 192
        M = rng.normal(0.0, 0.01, size=(T, 5))
        pbos.append(probability_backtest_overfitting(M, n_splits=8))
    mean_pbo = float(np.mean(pbos))
    assert 0.3 < mean_pbo < 0.7


def test_pbo_multi_strategy_random():
    rng = np.random.default_rng(987)
    T = 384
    # 5 iid strategies — selection is pure noise, PBO should be near 0.5.
    M = rng.normal(0.0, 0.01, size=(T, 5))
    pbo = probability_backtest_overfitting(M, n_splits=8)
    assert 0.2 < pbo < 0.8


def test_pbo_shape_mismatch_raises():
    with pytest.raises(ValueError):
        probability_backtest_overfitting(np.zeros(100), n_splits=8)


def test_pbo_too_few_strategies_raises():
    with pytest.raises(ValueError):
        probability_backtest_overfitting(np.zeros((100, 1)), n_splits=8)


def test_pbo_n_splits_must_be_even():
    M = np.zeros((100, 2))
    with pytest.raises(ValueError):
        probability_backtest_overfitting(M, n_splits=7)


def test_pbo_n_splits_too_small():
    M = np.zeros((100, 2))
    with pytest.raises(ValueError):
        probability_backtest_overfitting(M, n_splits=0)


def test_pbo_too_few_rows_raises():
    M = np.zeros((4, 2))
    with pytest.raises(ValueError):
        probability_backtest_overfitting(M, n_splits=8)


def test_pbo_in_unit_interval():
    rng = np.random.default_rng(2024)
    M = rng.normal(0.0, 0.01, size=(256, 4))
    pbo = probability_backtest_overfitting(M, n_splits=8)
    assert 0.0 <= pbo <= 1.0


# ---------------------------------------------------------------------------
# evaluate_regime_stability
# ---------------------------------------------------------------------------


def test_regime_stability_aligned_passes():
    # Strategy mirrors benchmark exactly → underperformance ≡ 0 in every state.
    rng = np.random.default_rng(0)
    T = 500
    bench = np.concatenate(
        [
            rng.normal(0.001, 0.005, T // 2),
            rng.normal(-0.001, 0.02, T - T // 2),
        ]
    )
    strat = bench.copy()
    res = evaluate_regime_stability(
        strat, bench, n_states=2, threshold=0.05, random_state=0
    )
    assert isinstance(res, RegimeStabilityResult)
    assert res.passed is True
    for k, gap in res.underperformance_per_state.items():
        assert gap == pytest.approx(0.0, abs=1e-12)
    assert res.worst_underperformance == pytest.approx(0.0, abs=1e-12)


def test_regime_stability_divergent_fails():
    rng = np.random.default_rng(7)
    T = 400
    bench = np.concatenate(
        [
            rng.normal(0.05, 0.005, T // 2),
            rng.normal(0.05, 0.02, T - T // 2),
        ]
    )
    strat = np.full_like(bench, -0.10)  # strategy is consistently bad
    res = evaluate_regime_stability(
        strat, bench, n_states=2, threshold=0.05, random_state=0
    )
    assert res.passed is False
    assert res.worst_underperformance < -0.05
    # All states should show massive underperformance (well below -threshold).
    for gap in res.underperformance_per_state.values():
        assert gap < -0.05


def test_regime_stability_one_state():
    # n_states=1 short-circuits the HMM and just compares unconditional means.
    rng = np.random.default_rng(3)
    T = 200
    bench = rng.normal(0.001, 0.01, T)
    strat = bench + 0.0001  # tiny outperformance
    res = evaluate_regime_stability(
        strat, bench, n_states=1, threshold=0.05, random_state=0
    )
    assert res.n_states == 1
    assert res.passed is True
    assert res.underperformance_per_state[0] == pytest.approx(0.0001, abs=1e-12)


def test_regime_stability_too_short_raises():
    with pytest.raises(ValueError):
        evaluate_regime_stability(
            np.zeros(5), np.zeros(5), n_states=2, threshold=0.05
        )


def test_regime_stability_shape_mismatch_raises():
    with pytest.raises(ValueError):
        evaluate_regime_stability(
            np.zeros(100), np.zeros(99), n_states=2, threshold=0.05
        )


def test_regime_stability_2d_input_raises():
    with pytest.raises(ValueError):
        evaluate_regime_stability(
            np.zeros((100, 2)), np.zeros(100), n_states=2, threshold=0.05
        )


def test_regime_stability_non_finite_raises():
    s = np.array([0.01, np.nan, 0.02] * 40)
    b = np.array([0.01, 0.01, 0.01] * 40)
    with pytest.raises(ValueError):
        evaluate_regime_stability(s, b, n_states=2, threshold=0.05)


def test_regime_stability_state_assignments_length():
    rng = np.random.default_rng(11)
    T = 250
    bench = np.concatenate(
        [
            rng.normal(0.0005, 0.004, T // 2),
            rng.normal(-0.0005, 0.02, T - T // 2),
        ]
    )
    strat = bench + rng.normal(0.0, 0.001, T)
    res = evaluate_regime_stability(
        strat, bench, n_states=2, threshold=0.05, random_state=42
    )
    assert res.state_assignments.shape == (T,)
    assert set(np.unique(res.state_assignments)).issubset({0, 1})
    # Both states should be exercised on this synthetic data.
    assert len(set(res.state_assignments)) >= 1
    assert sorted(res.state_returns.keys()) == list(range(2))
    assert sorted(res.benchmark_state_returns.keys()) == list(range(2))
    assert sorted(res.underperformance_per_state.keys()) == list(range(2))


def test_regime_stability_threshold_boundary():
    # Construct a case where worst underperformance is exactly -0.05.
    # strategy = benchmark - 0.05 in every period → gap = -0.05 in every state.
    # passed requires worst > -threshold (strict), so should fail at threshold=0.05.
    rng = np.random.default_rng(5)
    T = 200
    bench = np.concatenate(
        [
            rng.normal(0.001, 0.005, T // 2),
            rng.normal(-0.002, 0.02, T - T // 2),
        ]
    )
    strat = bench - 0.05
    res = evaluate_regime_stability(
        strat, bench, n_states=2, threshold=0.05, random_state=0
    )
    assert res.passed is False
    for gap in res.underperformance_per_state.values():
        assert gap == pytest.approx(-0.05, abs=1e-12)
