"""Asymmetric FP penalty via per-sample weights in LGBM.

Architecture: LightGBM with class-imbalanced sample weights targeting false-positive cost.
In trading regime gates, blocking a winning trade (false positive) is more expensive
than letting a marginal trade through (false negative). Implement via custom sample
weights: weight=3 for negative samples (class 0), weight=1 for positive samples (class 1).

This differs from balanced class weights by using a fixed 3:1 ratio targeting FP avoidance
specifically, rather than inversely weighting by class frequency.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import lightgbm as lgb

from shared.data import load_classifier_data
from shared.eval import evaluate_proba, report

# Load data
Xt, yt, Xv, yv = load_classifier_data()

# Compute sample weights: 3x for negative samples (FP avoidance)
# Class 0 (negative): weight=3.0 → penalize missing a negative case (FP)
# Class 1 (positive): weight=1.0 → normal cost for missing a positive case (FN)
sample_weights = np.where(yt == 0, 3.0, 1.0)

# Train LightGBM with asymmetric sample weights
m = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    random_state=42,
    verbose=-1
)
m.fit(Xt, yt, sample_weight=sample_weights)

# Generate predictions
p_train = m.predict_proba(Xt)[:, 1]
p_test = m.predict_proba(Xv)[:, 1]

# Evaluate and report
report(
    "129_asymmetric_fp_penalty_lgbm",
    "LGBM with asymmetric sample weights (3x for negatives) targeting false-positive cost in regime gates",
    evaluate_proba(p_train, p_test, yt, yv)
)
