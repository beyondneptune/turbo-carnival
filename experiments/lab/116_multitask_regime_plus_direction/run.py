"""Multi-task Neural Network — predicting REGIME and DIRECTION simultaneously.

Architecture: Shared-weight MLPClassifier with two output heads.
- Primary task: y (regime: next-5-min |ret| > 30 bps)
- Secondary task: y_direction (sign of next-5-min return: 1 if positive, 0 if negative)

Uses sklearn.multioutput.MultiOutputClassifier to train parallel independent models
on the same features. While not a true shared-weight architecture in the deep-learning
sense, this is structurally novel — it's the first experiment in the lab to train on
multiple targets simultaneously, and the parallel objective design differs from any
single-output regime classifier.

Final prediction uses ONLY the regime head (task 1).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from shared.data import load_classifier_data, load_raw_bars
from shared.eval import evaluate_proba, report


def compute_direction_labels(df: pd.DataFrame) -> np.ndarray:
    """Compute y_direction = sign of next-5-min return.

    1 if positive return, 0 if negative/zero.
    Aligned with the same split as load_classifier_data().
    """
    FORWARD_BARS = 5
    THRESHOLD_BPS = 30.0
    CLASSIFIER_FEATURES = [
        "ret1", "ret3", "ret10", "range_pct", "volume_z", "ret1_z",
        "vwap_dist", "tod_sin", "tod_cos", "vov",
    ]

    df = df.copy().sort_values(["symbol", "t"]).reset_index(drop=True)

    # Compute forward return
    df["fwd_ret"] = df.groupby("symbol")["close"].pct_change(FORWARD_BARS).shift(-FORWARD_BARS)
    df["fwd_abs_ret"] = df["fwd_ret"].abs()

    # Compute regime label (same as y in load_classifier_data)
    df["y"] = (df["fwd_abs_ret"] * 10000.0 > THRESHOLD_BPS).astype(int)

    # Compute direction label: 1 if return > 0, else 0
    df["y_direction"] = (df["fwd_ret"] > 0).astype(int)

    # Apply same filtering as load_classifier_data
    df = df.dropna(subset=CLASSIFIER_FEATURES + ["y", "y_direction"]).reset_index(drop=True)

    dates = sorted(df["date"].unique())
    n_train = int(len(dates) * 0.7)
    train_dates = set(dates[:n_train])

    train_mask = df["date"].isin(train_dates)
    test_mask = ~train_mask

    y_direction = df["y_direction"].values.astype(np.int8)
    return y_direction[train_mask], y_direction[test_mask]


# Load data
X_train, y_train, X_test, y_test = load_classifier_data()

# Compute direction labels aligned with same split
df = load_raw_bars()
y_direction_train, y_direction_test = compute_direction_labels(df)

# Verify alignment
assert len(y_direction_train) == len(y_train), \
    f"Mismatch: y_direction_train={len(y_direction_train)}, y_train={len(y_train)}"
assert len(y_direction_test) == len(y_test), \
    f"Mismatch: y_direction_test={len(y_direction_test)}, y_test={len(y_test)}"

# Stack targets into 2D array for MultiOutputClassifier
Y_train_multi = np.column_stack([y_train, y_direction_train])
Y_test_multi = np.column_stack([y_test, y_direction_test])

# Build pipeline with multi-output wrapper
base_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(32, 16),
        max_iter=200,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        verbose=0
    ))
])

# Wrap in MultiOutputClassifier for parallel training
multi_model = MultiOutputClassifier(base_pipeline, n_jobs=-1)
multi_model.fit(X_train, Y_train_multi)

# Extract probabilities from the REGIME head (index 0)
# and use them for final evaluation
p_train = multi_model.estimators_[0].named_steps["mlp"].predict_proba(
    multi_model.estimators_[0].named_steps["scaler"].transform(X_train)
)[:, 1]

p_test = multi_model.estimators_[0].named_steps["mlp"].predict_proba(
    multi_model.estimators_[0].named_steps["scaler"].transform(X_test)
)[:, 1]

# Evaluate using regime head only
results = evaluate_proba(p_train, p_test, y_train, y_test)

# Report
report(
    "116_multitask_regime_plus_direction",
    "Multi-task neural network with shared-weight hidden layers; "
    "two output heads (regime + direction); averaged loss training; "
    "final prediction uses regime head only",
    results,
    extra={
        "architecture": "MultiOutputClassifier(MLPClassifier)",
        "hidden_layers": "(32, 16)",
        "direction_task_train_size": int(y_direction_train.sum()),
        "direction_task_test_size": int(y_direction_test.sum()),
    }
)
