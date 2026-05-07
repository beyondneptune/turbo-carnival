"""Tests for `eval.baselines` -- Phase 3 evaluation baselines B0/B1/B2/B3.

Each baseline is exercised in isolation. The aggregator helpers
(`compute_all_baselines`, `best_baseline_per_event`, `best_baseline_summary`)
are exercised against small synthetic inputs. All synthetic data uses
reproducible numpy seeds so test outcomes are deterministic.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from eval.baselines import (
    BaselineSet,
    baseline_b0_no_trade,
    baseline_b1_long_straddle,
    baseline_b2_iron_butterfly,
    baseline_b3_logistic,
    best_baseline_per_event,
    best_baseline_summary,
    compute_all_baselines,
)


# ---------------------------------------------------------------------------
# B0 -- no-trade (cash at risk-free rate)
# ---------------------------------------------------------------------------


def test_b0_risk_free_default():
    """100 events, rf=0.04, days=5 -> all entries equal (1.04)^(5/365) - 1."""
    out = baseline_b0_no_trade(100, risk_free_rate_annual=0.04, holding_days=5)
    expected = (1.04) ** (5.0 / 365.0) - 1.0
    assert out.shape == (100,)
    assert np.allclose(out, expected)
    assert expected == pytest.approx(0.000538, abs=1e-5)


def test_b0_zero_rate():
    """rf = 0 -> every entry exactly 0.0."""
    out = baseline_b0_no_trade(50, risk_free_rate_annual=0.0, holding_days=5)
    assert out.shape == (50,)
    assert np.array_equal(out, np.zeros(50))


def test_b0_zero_events():
    """Empty baseline still produces a (0,) array (no crash)."""
    out = baseline_b0_no_trade(0)
    assert out.shape == (0,)


def test_b0_holding_days_scales_return():
    """Doubling holding period roughly doubles the (small) cash return."""
    short = baseline_b0_no_trade(1, risk_free_rate_annual=0.04, holding_days=5)[0]
    long = baseline_b0_no_trade(1, risk_free_rate_annual=0.04, holding_days=10)[0]
    assert long > short
    assert long == pytest.approx(2 * short, rel=1e-2)


def test_b0_negative_n_events_raises():
    with pytest.raises(ValueError):
        baseline_b0_no_trade(-1)


# ---------------------------------------------------------------------------
# B1 -- equal-weight long straddle (every event)
# ---------------------------------------------------------------------------


def test_b1_pass_through():
    """Input straddle returns are passed through unchanged."""
    inp = np.array([0.05, -0.02, 0.10])
    out = baseline_b1_long_straddle(inp)
    assert np.array_equal(out, inp)


def test_b1_returns_a_copy():
    """Mutating output does not mutate the input array."""
    inp = np.array([0.05, -0.02, 0.10])
    out = baseline_b1_long_straddle(inp)
    out[0] = 999.0
    assert inp[0] == pytest.approx(0.05)


def test_b1_accepts_list_input():
    """List input is converted to ndarray."""
    out = baseline_b1_long_straddle([0.01, 0.02])
    assert isinstance(out, np.ndarray)
    assert np.allclose(out, [0.01, 0.02])


# ---------------------------------------------------------------------------
# B2 -- iron butterfly (closed-form approximation)
# ---------------------------------------------------------------------------


def test_b2_small_move_positive_pnl():
    """|move| << expected_move -> short straddle keeps most of the credit.

    iv=0.30, T=5/252 -> expected_move ~ 0.0422. With |move|=0.01, PnL ~
    +credit - 0.01. The credit at iv=0.30 is approx 0.0267, so PnL > 0.
    """
    moves = np.array([0.01])
    iv = np.array([0.30])
    out = baseline_b2_iron_butterfly(moves, iv)
    assert out.shape == (1,)
    assert out[0] > 0.0
    # Sanity: closed-form pnl ~ 0.0167 (credit 0.0267 - loss 0.01).
    assert out[0] == pytest.approx(0.01667, abs=5e-4)


def test_b2_large_move_negative_pnl_bounded():
    """|move| > wing_width -> loss capped at wing_width.

    iv=0.30, |move|=0.10 (well past the 0.0422 wing). PnL = credit -
    wing_width ~ 0.0267 - 0.0422 ~ -0.0156. The bound is tight: even
    |move| = 1.0 should not breach it (capped at wing distance).
    """
    moves = np.array([0.10])
    iv = np.array([0.30])
    out = baseline_b2_iron_butterfly(moves, iv)
    assert out[0] < 0.0
    assert out[0] == pytest.approx(-0.01558, abs=5e-4)
    # Verify capping: an even larger move should not push PnL further down.
    out_huge = baseline_b2_iron_butterfly(np.array([1.0]), iv)
    assert out_huge[0] == pytest.approx(out[0], abs=1e-9)


def test_b2_zero_move_returns_full_credit():
    """No move -> capture entire credit (loss = 0)."""
    moves = np.array([0.0])
    iv = np.array([0.30])
    out = baseline_b2_iron_butterfly(moves, iv)
    expected_credit = (
        math.sqrt(2.0 / math.pi)
        - 2.0 * (
            1.0 / math.sqrt(2.0 * math.pi) * math.exp(-0.5)
            - (1.0 - 0.8413447460685429)
        )
    ) * 0.30 * math.sqrt(5.0 / 252.0)
    assert out[0] == pytest.approx(expected_credit, abs=1e-6)


def test_b2_higher_iv_yields_higher_credit():
    """Doubling IV roughly doubles both the wing width and the credit."""
    moves = np.zeros(1)
    out_low = baseline_b2_iron_butterfly(moves, np.array([0.20]))
    out_high = baseline_b2_iron_butterfly(moves, np.array([0.40]))
    assert out_high[0] == pytest.approx(2 * out_low[0], rel=1e-6)


def test_b2_sign_symmetric_in_move():
    """A +5% move and a -5% move yield the same PnL (|move| only)."""
    moves_pos = np.array([0.05])
    moves_neg = np.array([-0.05])
    iv = np.array([0.30])
    out_pos = baseline_b2_iron_butterfly(moves_pos, iv)
    out_neg = baseline_b2_iron_butterfly(moves_neg, iv)
    assert out_pos[0] == pytest.approx(out_neg[0], abs=1e-12)


def test_b2_shape_mismatch_raises():
    with pytest.raises(ValueError):
        baseline_b2_iron_butterfly(np.array([0.01, 0.02]), np.array([0.30]))


def test_b2_array_length_matches_input():
    """Output length matches input length for a vector input."""
    moves = np.linspace(-0.05, 0.05, 20)
    iv = np.full(20, 0.25)
    out = baseline_b2_iron_butterfly(moves, iv)
    assert out.shape == (20,)


# ---------------------------------------------------------------------------
# B3 -- dumb logistic (3 features only)
# ---------------------------------------------------------------------------


def _make_b3_synthetic(n: int = 100, seed: int = 0):
    """Build a synthetic dataset where x0's sign perfectly predicts return sign.

    x0 ~ U[-1, 1]; returns = 0.05 * sign(x0) + small noise; x1, x2 are pure
    noise. The logistic regressor should achieve perfect or near-perfect
    test accuracy.
    """
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-1.0, 1.0, size=n)
    x1 = rng.normal(0.0, 1.0, size=n)
    x2 = rng.normal(0.0, 1.0, size=n)
    # Strong signal: sign(x0) -> sign(return). Use 0.05 magnitude with very
    # small jitter so noise can never flip the sign.
    returns = 0.05 * np.sign(x0) + rng.normal(0.0, 1e-4, size=n)
    features = pd.DataFrame({"x0": x0, "x1": x1, "x2": x2})
    return features, returns


def test_b3_strong_signal_yields_high_accuracy_selection():
    """Synthetic where x0 strongly predicts return sign -> model picks
    almost every winner on the test half. We require high accuracy rather
    than 'every', because sklearn's default L2 penalty leaves a small
    intercept that can flip a single near-boundary row.
    """
    features, returns = _make_b3_synthetic(n=100, seed=7)
    out = baseline_b3_logistic(features, returns)
    assert out.shape == (100,)
    split = 100 // 2
    # Train rows (first half) are NaN.
    assert np.all(np.isnan(out[:split]))
    test_out = out[split:]
    assert np.all(~np.isnan(test_out))

    # 'Selected' rows are those where the output is non-zero (the model
    # picked the trade). They are equal to the realized return; the rest
    # are zero (not NaN).
    test_returns = returns[split:]
    selected_mask = test_out != 0.0
    np.testing.assert_allclose(test_out[selected_mask], test_returns[selected_mask])
    np.testing.assert_array_equal(test_out[~selected_mask], np.zeros(np.sum(~selected_mask)))

    # Accuracy: of the selected, at least 90% should have positive returns.
    n_selected = int(np.sum(selected_mask))
    assert n_selected > 0
    pos_among_selected = int(np.sum(test_returns[selected_mask] > 0))
    assert pos_among_selected / n_selected >= 0.90
    # Mean test output is meaningfully positive (the model captures the edge).
    assert float(np.nanmean(test_out)) > 0.015


def test_b3_returns_array_length_matches_input():
    """Output length matches input length."""
    features, returns = _make_b3_synthetic(n=80, seed=1)
    out = baseline_b3_logistic(features, returns)
    assert out.shape == (80,)


def test_b3_train_rows_are_nan_by_design():
    """The doc-string contracts that train rows are NaN; lock that here."""
    features, returns = _make_b3_synthetic(n=40, seed=2)
    out = baseline_b3_logistic(features, returns, train_test_split_idx=10)
    assert np.all(np.isnan(out[:10]))
    assert np.all(~np.isnan(out[10:]))


def test_b3_explicit_split_idx_honored():
    """train_test_split_idx is honored when provided."""
    features, returns = _make_b3_synthetic(n=60, seed=3)
    out = baseline_b3_logistic(features, returns, train_test_split_idx=20)
    assert np.all(np.isnan(out[:20]))
    assert np.all(~np.isnan(out[20:]))


def test_b3_threshold_above_one_selects_nothing():
    """threshold=1.0 + epsilon -> every test-row prediction filtered, all 0s."""
    features, returns = _make_b3_synthetic(n=50, seed=4)
    out = baseline_b3_logistic(features, returns, threshold=1.0 + 1e-9)
    split = 50 // 2
    assert np.all(out[split:] == 0.0)


def test_b3_threshold_below_zero_takes_every_test_event():
    """threshold=-epsilon -> every test prediction selected; output mirrors
    test-half realized straddle returns exactly.
    """
    features, returns = _make_b3_synthetic(n=50, seed=5)
    out = baseline_b3_logistic(features, returns, threshold=-1e-9)
    split = 50 // 2
    np.testing.assert_allclose(out[split:], returns[split:])


def test_b3_wrong_feature_count_raises():
    """B3 contract: exactly 3 features."""
    features = pd.DataFrame({"x0": [0.0, 1.0, 2.0, 3.0]})
    returns = np.array([0.01, -0.01, 0.02, -0.02])
    with pytest.raises(ValueError):
        baseline_b3_logistic(features, returns)


def test_b3_length_mismatch_raises():
    features = pd.DataFrame({"x0": [0.0], "x1": [0.0], "x2": [0.0]})
    returns = np.array([0.0, 0.1])
    with pytest.raises(ValueError):
        baseline_b3_logistic(features, returns)


def test_b3_single_class_train_falls_back_to_zero():
    """If the train set is single-class (e.g. all positive), B3 cannot
    learn -- it falls back to 'trade nothing' on the test set so we don't
    spuriously inherit a baseline edge from the data.
    """
    rng = np.random.default_rng(11)
    n = 40
    features = pd.DataFrame({
        "x0": rng.normal(size=n),
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
    })
    # First 20 rows all positive -> single training class.
    returns = np.concatenate([np.full(20, 0.01), rng.normal(0, 0.05, 20)])
    out = baseline_b3_logistic(features, returns, train_test_split_idx=20)
    assert np.all(np.isnan(out[:20]))
    assert np.all(out[20:] == 0.0)


def test_b3_deterministic_under_fixed_seed():
    """Repeated calls with the same random_state produce identical output."""
    features, returns = _make_b3_synthetic(n=70, seed=6)
    a = baseline_b3_logistic(features, returns, random_state=42)
    b = baseline_b3_logistic(features, returns, random_state=42)
    # NaN-aware equality on train rows; element-equality on test rows.
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
    mask = ~np.isnan(a)
    np.testing.assert_allclose(a[mask], b[mask])


# ---------------------------------------------------------------------------
# Aggregators
# ---------------------------------------------------------------------------


def _make_compute_all_inputs(n: int = 60, seed: int = 0):
    """Build coordinated synthetic inputs for the four baselines."""
    rng = np.random.default_rng(seed)
    straddle_returns = rng.normal(0.0, 0.05, size=n)
    underlying_moves = rng.normal(0.0, 0.03, size=n)
    iv = rng.uniform(0.20, 0.40, size=n)
    features = pd.DataFrame({
        "x0": rng.normal(size=n),
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
    })
    return straddle_returns, underlying_moves, iv, features


def test_compute_all_baselines_returns_baseline_set_with_equal_lengths():
    s, m, iv, f = _make_compute_all_inputs(n=80, seed=10)
    bset = compute_all_baselines(s, m, iv, f)
    assert isinstance(bset, BaselineSet)
    assert bset.b0.shape == (80,)
    assert bset.b1.shape == (80,)
    assert bset.b2.shape == (80,)
    assert bset.b3.shape == (80,)


def test_compute_all_baselines_b1_is_pass_through():
    """Sanity: B1 in the BaselineSet equals the input straddle returns."""
    s, m, iv, f = _make_compute_all_inputs(n=40, seed=11)
    bset = compute_all_baselines(s, m, iv, f)
    np.testing.assert_allclose(bset.b1, s)


def test_compute_all_baselines_length_mismatch_raises():
    s = np.zeros(10)
    m = np.zeros(11)
    iv = np.full(10, 0.30)
    f = pd.DataFrame({"x0": [0.0] * 10, "x1": [0.0] * 10, "x2": [0.0] * 10})
    with pytest.raises(ValueError):
        compute_all_baselines(s, m, iv, f)


def test_best_baseline_per_event_pointwise_max():
    """Pointwise max across the four baselines on a small constructed set."""
    bset = BaselineSet(
        b0=np.array([0.001, 0.001, 0.001]),
        b1=np.array([0.05, -0.10, 0.0]),
        b2=np.array([0.02, 0.03, -0.02]),
        b3=np.array([np.nan, 0.04, 0.10]),
    )
    out = best_baseline_per_event(bset)
    expected = np.array([0.05, 0.04, 0.10])
    np.testing.assert_allclose(out, expected)


def test_best_baseline_per_event_handles_nan_train_rows():
    """B3 NaNs in the train portion should not poison the per-event max."""
    bset = BaselineSet(
        b0=np.array([0.001, 0.001]),
        b1=np.array([0.05, -0.05]),
        b2=np.array([0.02, 0.02]),
        b3=np.array([np.nan, np.nan]),
    )
    out = best_baseline_per_event(bset)
    np.testing.assert_allclose(out, [0.05, 0.02])


def test_best_baseline_summary_keys_and_finiteness():
    """Returns a dict with five keys, all finite floats."""
    s, m, iv, f = _make_compute_all_inputs(n=50, seed=13)
    bset = compute_all_baselines(s, m, iv, f)
    summary = best_baseline_summary(bset)
    expected_keys = {"b0_mean", "b1_mean", "b2_mean", "b3_mean", "best_per_event_mean"}
    assert set(summary.keys()) == expected_keys
    for k, v in summary.items():
        assert isinstance(v, float), f"{k} should be float, got {type(v)}"
        assert math.isfinite(v), f"{k} should be finite, got {v}"


def test_best_baseline_summary_b1_mean_matches_nanmean_of_input_returns():
    """B1 is pass-through; its mean should match nanmean of input returns."""
    s, m, iv, f = _make_compute_all_inputs(n=50, seed=14)
    bset = compute_all_baselines(s, m, iv, f)
    summary = best_baseline_summary(bset)
    assert summary["b1_mean"] == pytest.approx(float(np.nanmean(s)))


def test_best_baseline_summary_best_dominates_each_baseline():
    """best_per_event_mean >= b0_mean / b1_mean / b2_mean / b3_mean.

    Pointwise max in expectation dominates each component's mean.
    """
    s, m, iv, f = _make_compute_all_inputs(n=80, seed=15)
    bset = compute_all_baselines(s, m, iv, f)
    summary = best_baseline_summary(bset)
    best = summary["best_per_event_mean"]
    for k in ("b0_mean", "b1_mean", "b2_mean", "b3_mean"):
        assert best >= summary[k] - 1e-12, f"{k}={summary[k]} > best={best}"
