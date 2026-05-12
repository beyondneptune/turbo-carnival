"""Calibrated CV LGBM with 5-fold isotonic calibration — novel architecture.

Architecture: LightGBM classifier with CalibratedClassifierCV wrapper.
- Base estimator: LGBMClassifier (n_estimators=200, max_depth=5, learning_rate=0.05)
- Calibration method: isotonic regression (non-parametric, monotonic)
- CV strategy: 5-fold stratified cross-validation (not 3-fold)
- Class weight: balanced to handle class imbalance

This differs from existing experiments:
- 055_calibrated_isotonic: uses LinearSVC base, not LGBM
- 056_calibrated_sigmoid: uses Sigmoid (Platt) calibration, not isotonic
- 127_beta_calibrated_lgbm: uses manual Beta calibration, not sklearn CalibratedClassifierCV

Averaged isotonic calibrators across 5 folds reduce overfitting compared to
single-split calibration. Strong calibration option for tree-based models.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from shared.data import load_classifier_data
from shared.eval import evaluate_proba, report

# Load data
Xt, yt, Xv, yv = load_classifier_data()

# Create pipeline with scaling + calibrated LGBM
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("lgbm_cal", CalibratedClassifierCV(
        estimator=lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            class_weight='balanced',
            random_state=42,
            verbose=-1
        ),
        method='isotonic',
        cv=5,
        n_jobs=-1
    ))
])

# Fit on training data
pipeline.fit(Xt, yt)

# Get calibrated probabilities on train and test
p_train = pipeline.predict_proba(Xt)[:, 1]
p_test = pipeline.predict_proba(Xv)[:, 1]

# Evaluate
metrics = evaluate_proba(p_train, p_test, yt, yv)

report(
    "135_calibrated_cv_lgbm",
    "LightGBM with CalibratedClassifierCV (5-fold, isotonic). Base: LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, balanced). Averaged isotonic calibrators across 5 folds reduce calibration overfitting.",
    metrics
)
