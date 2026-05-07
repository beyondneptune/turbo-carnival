"""Tests for the Phase 3 twin-model trainer."""
from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from signals.twin_model_trainer import (
    LGBM_PARAMS,
    EARLY_STOPPING_ROUNDS,
    TrainedModel,
    TwinModelComparison,
    compare_twin_models,
    compute_sharpe,
    predict_returns,
    train_and_compare,
    train_one_model,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset(n: int = 200, n_features: int = 5, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(n, n_features)),
        columns=[f"x{i}" for i in range(n_features)],
    )
    return X, rng.normal(size=n)


@dataclass
class _FakeBooster:
    """Minimal stand-in for an lgb.Booster — just predict() driven by a callable."""

    fn: object

    def predict(self, X, num_iteration=None):  # noqa: ARG002 - mimic lgb signature
        return np.asarray(self.fn(X), dtype=float)


def _fake_model(predict_fn) -> TrainedModel:
    """Build a TrainedModel-like object whose booster.predict returns a fixed pattern.

    Note: we can't actually instantiate the dataclass with a fake booster because
    of the type annotation, but Python's dataclass doesn't enforce types at
    runtime. We bypass via object.__setattr__ on a frozen dataclass.
    """
    # Use a real (tiny) booster instance to satisfy the type, then override predict.
    tm = TrainedModel.__new__(TrainedModel)
    object.__setattr__(tm, "booster", _FakeBooster(predict_fn))
    object.__setattr__(tm, "target_name", "FAKE")
    object.__setattr__(tm, "feature_names", [])
    object.__setattr__(tm, "train_score", 0.0)
    object.__setattr__(tm, "val_score", 0.0)
    object.__setattr__(tm, "best_iteration", 1)
    return tm


# ---------------------------------------------------------------------------
# compute_sharpe
# ---------------------------------------------------------------------------


def test_compute_sharpe_constant_returns_zero():
    """Constant series has std=0, must return 0 (no division by zero)."""
    out = compute_sharpe(np.array([0.01, 0.01, 0.01]))
    assert out == 0.0


def test_compute_sharpe_known_series_matches_manual():
    """Sharpe on a known series matches manual computation."""
    rng = np.random.default_rng(123)
    returns = rng.normal(loc=0.001, scale=0.01, size=500)
    expected = float(np.mean(returns)) / float(np.std(returns, ddof=1)) * np.sqrt(252)
    actual = compute_sharpe(returns)
    assert actual == pytest.approx(expected, rel=1e-9)


def test_compute_sharpe_empty_returns_zero():
    """Empty array does not raise and returns 0."""
    out = compute_sharpe(np.array([]))
    assert out == 0.0


def test_compute_sharpe_single_value_returns_zero():
    """Single observation: std undefined, must return 0 cleanly."""
    out = compute_sharpe(np.array([0.05]))
    assert out == 0.0


# ---------------------------------------------------------------------------
# train_one_model
# ---------------------------------------------------------------------------


def test_train_one_model_smoke():
    """Train on synthetic y = 2*x_0 + noise; basic sanity checks."""
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(100, 5)), columns=[f"x{i}" for i in range(5)])
    y = 2.0 * X["x0"].to_numpy() + 0.1 * rng.normal(size=100)

    model = train_one_model(X, y, target_name="A", random_state=7)

    assert isinstance(model, TrainedModel)
    assert isinstance(model.booster, lgb.Booster)
    assert model.target_name == "A"
    assert model.feature_names == [f"x{i}" for i in range(5)]
    assert np.isfinite(model.train_score)
    assert np.isfinite(model.val_score)
    assert model.best_iteration >= 1


def test_train_one_model_validates_inputs():
    """Mismatched X/y length should raise."""
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        train_one_model(X, np.array([1.0, 2.0]), target_name="A")


def test_train_one_model_rejects_nondataframe():
    """Non-DataFrame X should raise TypeError."""
    with pytest.raises(TypeError):
        train_one_model(np.zeros((5, 3)), np.zeros(5), target_name="A")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# predict_returns
# ---------------------------------------------------------------------------


def test_predict_returns_all_selected():
    """Model that always predicts 1.0 above threshold => all events captured."""
    realized = np.array([0.05, -0.03, 0.10])
    X = pd.DataFrame({"a": [1, 2, 3]})
    model = _fake_model(lambda Z: np.ones(len(Z)))

    out = predict_returns(model, X, realized, selection_threshold=0.0)
    np.testing.assert_array_equal(out, realized)


def test_predict_returns_skip_logic():
    """Mixed predictions => only above-threshold trades are kept."""
    preds = np.array([-1.0, 1.0, 1.0, -1.0])
    realized = np.array([0.01, 0.02, 0.03, 0.04])
    X = pd.DataFrame({"a": [0, 0, 0, 0]})
    model = _fake_model(lambda Z, _p=preds: _p)

    out = predict_returns(model, X, realized, selection_threshold=0.0)
    np.testing.assert_array_equal(out, np.array([0.0, 0.02, 0.03, 0.0]))


def test_predict_returns_threshold_strict():
    """Equal-to-threshold predictions are NOT selected (strict > comparison)."""
    preds = np.array([0.0, 0.0, 0.5])
    realized = np.array([0.10, 0.20, 0.30])
    X = pd.DataFrame({"a": [0, 0, 0]})
    model = _fake_model(lambda Z, _p=preds: _p)

    out = predict_returns(model, X, realized, selection_threshold=0.0)
    np.testing.assert_array_equal(out, np.array([0.0, 0.0, 0.30]))


def test_predict_returns_length_mismatch():
    """Mismatched X / realized lengths should raise."""
    model = _fake_model(lambda Z: np.zeros(len(Z)))
    with pytest.raises(ValueError):
        predict_returns(
            model, pd.DataFrame({"a": [1, 2]}), np.array([0.1, 0.2, 0.3])
        )


# ---------------------------------------------------------------------------
# compare_twin_models
# ---------------------------------------------------------------------------


def _build_b_wins_scenario(n: int = 250, seed: int = 11):
    """Construct a test set where B's predictions perfectly select winners
    while A picks indiscriminately. Returns (model_a, model_b, X, realized).
    """
    rng = np.random.default_rng(seed)
    # Per-event realized returns — mix of strong winners and losers.
    realized = rng.normal(loc=0.0, scale=0.02, size=n)
    realized[: n // 2] += 0.05  # First half winners.
    realized[n // 2 :] -= 0.05  # Second half losers.

    X = pd.DataFrame({"x0": np.arange(n, dtype=float)})

    # Model A: predicts +1 for everything (takes all trades — average ~0).
    model_a = _fake_model(lambda Z: np.ones(len(Z)))
    # Model B: predicts +1 only for winners (correctly times the trade).
    b_preds = np.where(np.arange(n) < n // 2, 1.0, -1.0)
    model_b = _fake_model(lambda Z, _p=b_preds: _p)
    return model_a, model_b, X, realized


def test_compare_twin_models_b_clearly_wins():
    """B selects only winners — must beat A on all 3 thresholds."""
    a, b, X, r = _build_b_wins_scenario()
    res = compare_twin_models(a, b, X, r)

    assert isinstance(res, TwinModelComparison)
    assert res.n_events == len(r)
    assert res.mean_return_b > res.mean_return_a
    assert res.sharpe_delta_b_minus_a >= 0.2
    assert res.net_excess_b_minus_a_bps_ann >= 50.0
    assert res.paired_t_stat >= 2.0
    assert res.b_beats_a is True


def test_compare_twin_models_b_clearly_loses():
    """B selects randomly — should not beat A on all 3 thresholds."""
    rng = np.random.default_rng(42)
    n = 200
    realized = rng.normal(loc=0.001, scale=0.02, size=n)
    X = pd.DataFrame({"x0": np.arange(n, dtype=float)})

    # Model A: takes every trade (mean = mean(realized)).
    model_a = _fake_model(lambda Z: np.ones(len(Z)))
    # Model B: random selections.
    b_preds = rng.normal(size=n)
    model_b = _fake_model(lambda Z, _p=b_preds: _p)

    res = compare_twin_models(model_a, model_b, X, realized)
    assert res.b_beats_a is False


def test_compare_twin_models_mixed_excess_only():
    """B beats A on excess return but not Sharpe — must still fail (need ALL 3).

    Construction: B selects a single huge winner, all other trades skipped =>
    high mean but tiny Sharpe (almost all zeros). A takes all trades, smoother.
    """
    n = 200
    realized = np.full(n, 0.001)  # A takes everything, low-vol positive carry.

    # Build B's preds: one giant pick.
    b_preds = -np.ones(n)
    b_preds[0] = 1.0
    realized[0] = 1.0  # one massive winner

    X = pd.DataFrame({"x0": np.arange(n, dtype=float)})
    model_a = _fake_model(lambda Z: np.ones(len(Z)))
    model_b = _fake_model(lambda Z, _p=b_preds: _p)

    res = compare_twin_models(model_a, model_b, X, realized)
    # Force the situation where at least one of the three thresholds fails:
    assert res.b_beats_a is False
    # And confirm at least one threshold did fail:
    assert any(not d["passed"] for d in res.threshold_details.values())


def test_compare_twin_models_threshold_details_shape():
    """threshold_details must have 3 keys, each with value/threshold/passed."""
    a, b, X, r = _build_b_wins_scenario()
    res = compare_twin_models(a, b, X, r)
    assert set(res.threshold_details.keys()) == {"net_excess", "sharpe_delta", "t_stat"}
    for key, d in res.threshold_details.items():
        assert set(d.keys()) == {"value", "threshold", "passed"}
        assert isinstance(d["value"], float)
        assert isinstance(d["threshold"], float)
        assert isinstance(d["passed"], bool)


def test_compare_twin_models_custom_thresholds():
    """Tighter thresholds can flip a passing verdict to failing."""
    a, b, X, r = _build_b_wins_scenario()
    base = compare_twin_models(a, b, X, r)
    assert base.b_beats_a is True

    # Crank the t-stat threshold absurdly high.
    tightened = compare_twin_models(a, b, X, r, t_stat_threshold=1e6)
    assert tightened.b_beats_a is False
    assert tightened.threshold_details["t_stat"]["passed"] is False


# ---------------------------------------------------------------------------
# train_and_compare
# ---------------------------------------------------------------------------


def test_train_and_compare_end_to_end():
    """End-to-end smoke: trains both models and returns a comparison."""
    rng = np.random.default_rng(2024)
    n, n_features = 200, 8
    X = pd.DataFrame(
        rng.normal(size=(n, n_features)),
        columns=[f"x{i}" for i in range(n_features)],
    )
    # Targets and realized returns: reasonable signal for both, noisy.
    y_a = 2.0 * X["x0"].to_numpy() + 0.5 * rng.normal(size=n)
    y_b = X["x0"].to_numpy() + X["x1"].to_numpy() + 0.5 * rng.normal(size=n)
    realized = 0.02 * X["x0"].to_numpy() + 0.01 * rng.normal(size=n)

    model_a, model_b, comparison = train_and_compare(
        X, y_a, y_b, realized, random_state=2024
    )

    assert isinstance(model_a, TrainedModel)
    assert isinstance(model_b, TrainedModel)
    assert model_a.target_name == "A"
    assert model_b.target_name == "B"
    assert isinstance(comparison, TwinModelComparison)
    assert comparison.n_events == int(n * 0.2)
    # Comparisons should produce finite numbers.
    assert np.isfinite(comparison.mean_return_a)
    assert np.isfinite(comparison.mean_return_b)
    assert np.isfinite(comparison.sharpe_a)
    assert np.isfinite(comparison.sharpe_b)
    assert np.isfinite(comparison.paired_t_stat)
    assert isinstance(comparison.b_beats_a, bool)


def test_train_and_compare_validates_lengths():
    """Mismatched array lengths should raise."""
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        train_and_compare(
            X,
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0]),  # wrong length
            np.array([0.01, 0.02, 0.03]),
        )


# ---------------------------------------------------------------------------
# Locked hyperparameters sanity check
# ---------------------------------------------------------------------------


def test_locked_hyperparameters():
    """Per strategic_lock.md: verify the locked params are present."""
    assert LGBM_PARAMS["max_depth"] == 4
    assert LGBM_PARAMS["n_estimators"] == 500
    assert LGBM_PARAMS["learning_rate"] == 0.05
    assert LGBM_PARAMS["objective"] == "regression"
    assert LGBM_PARAMS["metric"] == "mse"
    assert EARLY_STOPPING_ROUNDS == 50
