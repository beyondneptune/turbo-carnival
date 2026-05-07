"""Phase 3 twin-model trainer — train Model A (return target) and Model B
(condition-aligned target), then run head-to-head A/B evaluation. Self-contained:
takes target arrays as inputs, doesn't import the satellite framework.

Per metadata/strategic_lock.md § Twin-Model Design:
  Both models: LightGBM, max_depth=4, n_estimators=500, learning_rate=0.05,
  early_stopping_rounds=50, MSE loss.

  Verdict thresholds (all must pass):
    1. Net excess return: B - A >= +50 bps annualised
    2. Sharpe delta: Sharpe(B) - Sharpe(A) >= +0.2
    3. Paired t-statistic: t >= 2.0
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split


# Locked hyperparameters from strategic_lock.md.
LGBM_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "metric": "mse",
    "max_depth": 4,
    "n_estimators": 500,
    "learning_rate": 0.05,
    "verbose": -1,
}
EARLY_STOPPING_ROUNDS = 50

# Standard trading days per year for annualisation.
TRADING_DAYS_PER_YEAR = 252
HOLD_PERIOD_DAYS = 5  # earnings straddle hold period (T-1 to T+5)
EVENTS_PER_YEAR = TRADING_DAYS_PER_YEAR / HOLD_PERIOD_DAYS  # ~50.4


@dataclass(frozen=True)
class TrainedModel:
    """A trained LightGBM model for either target A or target B."""

    booster: lgb.Booster
    target_name: str  # 'A' or 'B'
    feature_names: list[str]
    train_score: float  # MSE on train
    val_score: float  # MSE on val
    best_iteration: int


@dataclass(frozen=True)
class TwinModelComparison:
    """Output of head-to-head A vs B evaluation on a held-out test set."""

    n_events: int
    mean_return_a: float
    mean_return_b: float
    sharpe_a: float
    sharpe_b: float
    net_excess_b_minus_a_bps_ann: float
    sharpe_delta_b_minus_a: float
    paired_t_stat: float
    paired_p_value: float
    b_beats_a: bool  # True iff all 3 thresholds met
    threshold_details: dict[str, dict[str, Any]] = field(default_factory=dict)


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error helper."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def train_one_model(
    X: pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    target_name: str,
    val_size: float = 0.2,
    random_state: int = 42,
) -> TrainedModel:
    """Fit a single LightGBM model with early stopping.

    Splits (X, y) into train/val; fits using the locked LGBM_PARAMS with
    EARLY_STOPPING_ROUNDS evaluated against val. Returns a TrainedModel
    capturing the booster, in/out-of-sample MSE, and best iteration.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"X must be a DataFrame, got {type(X).__name__}")
    y_arr = np.asarray(y, dtype=float).ravel()
    if len(X) != len(y_arr):
        raise ValueError(f"X has {len(X)} rows but y has {len(y_arr)}")

    feature_names = list(X.columns)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_arr, test_size=val_size, random_state=random_state
    )

    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )

    booster = model.booster_
    best_iter = int(model.best_iteration_) if model.best_iteration_ else booster.current_iteration()

    train_pred = model.predict(X_train, num_iteration=best_iter)
    val_pred = model.predict(X_val, num_iteration=best_iter)

    return TrainedModel(
        booster=booster,
        target_name=target_name,
        feature_names=feature_names,
        train_score=_mse(y_train, train_pred),
        val_score=_mse(y_val, val_pred),
        best_iteration=best_iter,
    )


def predict_returns(
    model: TrainedModel,
    X: pd.DataFrame,
    realized_returns: np.ndarray,
    *,
    selection_threshold: float = 0.0,
) -> np.ndarray:
    """Per-event realized return achieved by the selection strategy.

    For each event, the model predicts the target. If prediction >
    selection_threshold the trade is taken (capture realized return), else
    skip (return 0).
    """
    realized = np.asarray(realized_returns, dtype=float).ravel()
    if len(X) != len(realized):
        raise ValueError(
            f"X has {len(X)} rows but realized_returns has {len(realized)}"
        )

    preds = np.asarray(
        model.booster.predict(X, num_iteration=model.best_iteration), dtype=float
    ).ravel()
    selected = preds > selection_threshold
    out = np.where(selected, realized, 0.0)
    return out


def compute_sharpe(returns: np.ndarray, *, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualised Sharpe ratio. Returns 0 if std is 0 or returns is empty."""
    arr = np.asarray(returns, dtype=float).ravel()
    if arr.size == 0:
        return 0.0
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    if std <= 0.0 or not np.isfinite(std):
        return 0.0
    mean = float(np.mean(arr))
    return mean / std * float(np.sqrt(periods_per_year))


def compare_twin_models(
    model_a: TrainedModel,
    model_b: TrainedModel,
    X_test: pd.DataFrame,
    realized_returns_test: np.ndarray,
    *,
    bps_threshold: float = 50.0,
    sharpe_delta_threshold: float = 0.2,
    t_stat_threshold: float = 2.0,
) -> TwinModelComparison:
    """Run A and B over X_test. Compute mean return, Sharpe, paired t-test on
    per-event differences (B - A). Verdict: b_beats_a iff all 3 thresholds met.
    """
    realized = np.asarray(realized_returns_test, dtype=float).ravel()
    n = len(X_test)
    if n != len(realized):
        raise ValueError(
            f"X_test has {n} rows but realized_returns_test has {len(realized)}"
        )

    ret_a = predict_returns(model_a, X_test, realized)
    ret_b = predict_returns(model_b, X_test, realized)

    mean_a = float(np.mean(ret_a)) if ret_a.size else 0.0
    mean_b = float(np.mean(ret_b)) if ret_b.size else 0.0

    sharpe_a = compute_sharpe(ret_a)
    sharpe_b = compute_sharpe(ret_b)

    # Net excess annualised. Each event spans HOLD_PERIOD_DAYS=5 trading days,
    # so events-per-year = TRADING_DAYS_PER_YEAR / HOLD_PERIOD_DAYS ≈ 50.4.
    # Multiplying mean per-event diff by 252 (as if events were daily) over-
    # annualises by 5×.
    net_excess_ann = (mean_b - mean_a) * EVENTS_PER_YEAR
    net_excess_bps_ann = net_excess_ann * 1e4

    sharpe_delta = sharpe_b - sharpe_a

    diffs = ret_b - ret_a
    # Paired t-test: t = mean(diffs) / (std(diffs)/sqrt(n))
    if n < 2 or float(np.std(diffs, ddof=1)) == 0.0:
        t_stat = 0.0
        p_value = 1.0
    else:
        t_stat_raw, p_val_raw = stats.ttest_rel(ret_b, ret_a)
        t_stat = float(t_stat_raw)
        p_value = float(p_val_raw)
        if not np.isfinite(t_stat):
            t_stat = 0.0
            p_value = 1.0

    threshold_details: dict[str, dict[str, Any]] = {
        "net_excess": {
            "value": net_excess_bps_ann,
            "threshold": bps_threshold,
            "passed": bool(net_excess_bps_ann >= bps_threshold),
        },
        "sharpe_delta": {
            "value": sharpe_delta,
            "threshold": sharpe_delta_threshold,
            "passed": bool(sharpe_delta >= sharpe_delta_threshold),
        },
        "t_stat": {
            "value": t_stat,
            "threshold": t_stat_threshold,
            "passed": bool(t_stat >= t_stat_threshold),
        },
    }
    b_beats_a = all(d["passed"] for d in threshold_details.values())

    return TwinModelComparison(
        n_events=n,
        mean_return_a=mean_a,
        mean_return_b=mean_b,
        sharpe_a=sharpe_a,
        sharpe_b=sharpe_b,
        net_excess_b_minus_a_bps_ann=net_excess_bps_ann,
        sharpe_delta_b_minus_a=sharpe_delta,
        paired_t_stat=t_stat,
        paired_p_value=p_value,
        b_beats_a=b_beats_a,
        threshold_details=threshold_details,
    )


def train_and_compare(
    X: pd.DataFrame,
    y_a: np.ndarray | pd.Series,
    y_b: np.ndarray | pd.Series,
    realized_returns: np.ndarray,
    *,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> tuple[TrainedModel, TrainedModel, TwinModelComparison]:
    """Top-level convenience: split into train/test, train A and B on train,
    compare on test. Returns (model_a, model_b, comparison).
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"X must be a DataFrame, got {type(X).__name__}")
    y_a_arr = np.asarray(y_a, dtype=float).ravel()
    y_b_arr = np.asarray(y_b, dtype=float).ravel()
    realized = np.asarray(realized_returns, dtype=float).ravel()
    n = len(X)
    if not (len(y_a_arr) == n == len(y_b_arr) == len(realized)):
        raise ValueError(
            "X, y_a, y_b, realized_returns must all have the same length; "
            f"got {n}, {len(y_a_arr)}, {len(y_b_arr)}, {len(realized)}"
        )

    indices = np.arange(n)
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state
    )

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_a_train = y_a_arr[train_idx]
    y_b_train = y_b_arr[train_idx]
    realized_test = realized[test_idx]

    model_a = train_one_model(
        X_train,
        y_a_train,
        target_name="A",
        val_size=val_size,
        random_state=random_state,
    )
    model_b = train_one_model(
        X_train,
        y_b_train,
        target_name="B",
        val_size=val_size,
        random_state=random_state,
    )

    comparison = compare_twin_models(model_a, model_b, X_test, realized_test)
    return model_a, model_b, comparison
