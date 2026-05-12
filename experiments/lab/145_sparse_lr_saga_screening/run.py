"""Sparse logistic regression with safe feature screening.

Architecture: L1 LogReg with aggressive regularization to identify nonzero
coefficients (screening step), then retrain on screened features only with
default C=1.0. This differs from plain L1 (031) by adding the two-stage
refit, and from elastic-net (114) by using pure L1 for screening + standard
L2 for refit.

The key insight: aggressive L1 (C=0.005) zeros out weak features during
training, then we lock in that feature set and refit with balanced L2.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

from shared.data import load_classifier_data
from shared.eval import evaluate_proba, report

# Load data
X_train, y_train, X_test, y_test = load_classifier_data()

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Stage 1: Feature screening with aggressive L1 ===
# Use very high regularization (low C) to aggressively zero out weak features
screen = LogisticRegression(
    penalty='l1',
    solver='saga',
    C=0.005,  # Very aggressive regularization
    max_iter=5000,
    class_weight='balanced',
    random_state=42,
    tol=1e-3
)
screen.fit(X_train_scaled, y_train)

# Identify which features survived screening (nonzero coefficients)
nonzero_mask = (screen.coef_[0] != 0)
n_screened = nonzero_mask.sum()

print(f"Screened {n_screened} / {X_train_scaled.shape[1]} features")

# Fallback: if all features were zeroed out, keep all (shouldn't happen with C=0.005)
if n_screened == 0:
    print("Warning: All features zeroed out. Using all features as fallback.")
    nonzero_mask = np.ones(X_train_scaled.shape[1], dtype=bool)
    n_screened = nonzero_mask.sum()

# Get feature names for reporting
FEATURES = ["ret1", "ret3", "ret10", "range_pct", "volume_z", "ret1_z",
            "vwap_dist", "tod_sin", "tod_cos", "vov"]
screened_features = [FEATURES[i] for i in range(len(FEATURES)) if nonzero_mask[i]]

# === Stage 2: Refit on screened features with default C ===
# Use standard LogReg with default C=1.0 on just the screened features
final_model = LogisticRegression(
    penalty='l2',
    solver='lbfgs',
    C=1.0,
    max_iter=2000,
    class_weight='balanced',
    random_state=42
)
final_model.fit(X_train_scaled[:, nonzero_mask], y_train)

# Predict
p_train = final_model.predict_proba(X_train_scaled[:, nonzero_mask])[:, 1]
p_test = final_model.predict_proba(X_test_scaled[:, nonzero_mask])[:, 1]

# Evaluate
metrics = evaluate_proba(p_train, p_test, y_train, y_test)

# Report with extra data about screening
extra = {
    "n_screened_features": int(n_screened),
    "screened_feature_names": screened_features,
    "screening_C": 0.005,
    "screening_penalty": "l1",
    "refit_C": 1.0,
    "refit_penalty": "l2"
}

report(
    "145_sparse_lr_saga_screening",
    "Safe feature screening: aggressive L1 (C=0.005) identifies nonzero features, "
    "then refit L2 (C=1.0) on screened set only",
    metrics,
    extra=extra
)
