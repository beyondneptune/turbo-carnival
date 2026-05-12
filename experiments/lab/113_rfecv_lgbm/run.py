"""Recursive Feature Elimination with Cross-Validation (RFECV) + LightGBM.

Architecture: RFECV iteratively removes least-important features using LogisticRegression
as the base estimator, finds the optimal feature subset via cross-validation, then trains
LightGBM on selected features only.

This is a novel **feature selection strategy** combining RFECV (algorithm family: RFE)
with downstream LightGBM (tree-based decision rule). RFECV is not used in existing
experiments 001-112.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

from shared.data import load_classifier_data
from shared.eval import evaluate_proba, report

# Load data
X_train, y_train, X_test, y_test = load_classifier_data()

# Feature names for tracking
FEATURE_NAMES = [
    "ret1", "ret3", "ret10", "range_pct", "volume_z", "ret1_z",
    "vwap_dist", "tod_sin", "tod_cos", "vov",
]

# RFECV with LogisticRegression as the base estimator (fast)
estimator = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
rfecv = RFECV(estimator=estimator, step=1, cv=3, scoring='roc_auc', n_jobs=-1)
rfecv.fit(X_train, y_train)

# Extract selection results
selected_mask = rfecv.support_
selected_features = [FEATURE_NAMES[i] for i in range(len(FEATURE_NAMES)) if selected_mask[i]]
rejected_features = [FEATURE_NAMES[i] for i in range(len(FEATURE_NAMES)) if not selected_mask[i]]
n_selected = rfecv.n_features_

# Filter to selected features
X_train_sel = X_train[:, selected_mask]
X_test_sel = X_test[:, selected_mask]

# Train LightGBM on selected features only
lgbm = lgb.LGBMClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42,
    verbose=-1
)
lgbm.fit(X_train_sel, y_train)

# Evaluate
p_train = lgbm.predict_proba(X_train_sel)[:, 1]
p_test = lgbm.predict_proba(X_test_sel)[:, 1]
metrics = evaluate_proba(p_train, p_test, y_train, y_test)

# Report with feature selection details
extra = {
    "selected_features": selected_features,
    "rejected_features": rejected_features,
    "n_features_selected": int(n_selected),
    "n_features_total": len(FEATURE_NAMES),
}

report(
    "113_rfecv_lgbm",
    "RFECV (LogReg base, roc_auc scoring) finds optimal feature subset; LightGBM trains on selected features only",
    metrics,
    extra=extra
)
