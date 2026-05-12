"""Mahalanobis-distance-weighted KNN — distance metric accounts for feature covariance.

Architecture: distance-based classifier using Mahalanobis metric instead of Euclidean.
Mahalanobis distance is measured in units of standard deviation along principal axes,
accounting for the inverse covariance structure of the feature space.
This differs fundamentally from 010_knn (Euclidean) and 042 (other sklearn KNN variants).
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from shared.data import load_classifier_data
from shared.eval import evaluate_proba, report

# Load data
Xt, yt, Xv, yv = load_classifier_data()

# Timing for potential subsample reporting
t0 = time.time()

# Standardize features for covariance computation
sc = StandardScaler().fit(Xt)
Xt_s = sc.transform(Xt)
Xv_s = sc.transform(Xv)

print(f"Train shape: {Xt_s.shape}, Test shape: {Xv_s.shape}")

# Compute inverse covariance matrix (Mahalanobis metric requirement)
cov = np.cov(Xt_s, rowvar=False)
inv_cov = np.linalg.pinv(cov)

print(f"Inverse covariance computed, condition number: {np.linalg.cond(cov):.2e}")

# Check if test set is too large; if so, subsample to 5000 for speed
test_size = Xv_s.shape[0]
if test_size > 5000:
    print(f"Test set size {test_size} exceeds threshold. Subsampling to 5000 for speed.")
    subsample_idx = np.random.RandomState(42).choice(test_size, 5000, replace=False)
    Xv_s_use = Xv_s[subsample_idx]
    yv_use = yv[subsample_idx]
    timing_note = "partial (5000 test subsample)"
else:
    Xv_s_use = Xv_s
    yv_use = yv
    timing_note = "full test"

# Fit KNeighborsClassifier with Mahalanobis metric
# n_neighbors=30 (standard choice for this problem scale)
# metric='mahalanobis' with VI (inverse covariance) parameter
# algorithm='brute' for Mahalanobis (tree-based methods don't support it)
# n_jobs=-1 for parallelization
m = KNeighborsClassifier(
    n_neighbors=30,
    metric='mahalanobis',
    metric_params={'VI': inv_cov},
    algorithm='brute',
    n_jobs=-1
)

print("Training Mahalanobis KNN...")
t_fit = time.time()
m.fit(Xt_s, yt)
t_fit_done = time.time()
print(f"Fit time: {t_fit_done - t_fit:.2f}s")

# Predict probabilities on train and test
print("Predicting on train set...")
t_pred = time.time()
p_train = m.predict_proba(Xt_s)[:, 1]
t_pred_train = time.time()
print(f"Train prediction time: {t_pred_train - t_pred:.2f}s")

print(f"Predicting on test set ({timing_note})...")
p_test = m.predict_proba(Xv_s_use)[:, 1]
t_pred_done = time.time()
print(f"Test prediction time: {t_pred_done - t_pred_train:.2f}s")

# Evaluate
metrics = evaluate_proba(p_train, p_test, yt, yv_use)

# Add timing and subsample info
extra = {
    "total_time_seconds": t_pred_done - t0,
    "test_evaluation": timing_note,
    "covariance_condition": float(np.linalg.cond(cov)),
}

report(
    "130_mahalanobis_knn",
    "KNN with Mahalanobis distance metric (accounts for covariance structure), n_neighbors=30",
    metrics,
    extra=extra
)

print(f"Total runtime: {t_pred_done - t0:.2f}s")
