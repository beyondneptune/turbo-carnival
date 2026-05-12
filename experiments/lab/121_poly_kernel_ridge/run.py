"""Kernel Ridge Regression with polynomial kernel, used for classification via sigmoid mapping.

Architecture: ridge regression in polynomial kernel feature space, treated as binary
classification via sigmoid link function. KernelRidge is O(n^2) — subsampled for speed.

Novel: Uses KernelRidge (kernel-implicit polynomial expansion) with sigmoid transformation
for classification. No existing experiment uses KernelRidge with this approach.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.special import expit

from shared.data import load_classifier_data
from shared.eval import evaluate_proba, report

# Load data
X_train, y_train, X_test, y_test = load_classifier_data()

# KernelRidge is O(n^2) — subsample for speed
rng = np.random.default_rng(42)
sub = rng.choice(len(X_train), min(5000, len(X_train)), replace=False)

# Build pipeline with KernelRidge
m = Pipeline([
    ('scaler', StandardScaler()),
    ('kr', KernelRidge(kernel='poly', degree=3, alpha=1.0))
])

# Fit on subsampled training data
m.fit(X_train[sub], y_train[sub].astype(float))

# Predict on full train and test sets
y_train_pred = m.predict(X_train)
y_test_pred = m.predict(X_test)

# Apply sigmoid transformation to convert ridge outputs to probabilities
# Scale by 5 to increase steepness of the sigmoid transition
p_train = expit(5 * (y_train_pred - 0.5))
p_test = expit(5 * (y_test_pred - 0.5))

# Evaluate and report
report(
    "121_poly_kernel_ridge",
    "kernel ridge regression with degree-3 polynomial kernel + sigmoid link, subsampled (5k) for efficiency",
    evaluate_proba(p_train, p_test, y_train, y_test)
)
