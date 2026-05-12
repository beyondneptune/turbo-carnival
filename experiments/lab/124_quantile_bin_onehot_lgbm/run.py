"""Quantile-binning + one-hot encoding + LightGBM.

Novel architecture: treats continuous features as categorical via decile binning.
LightGBM natively handles categorical features, but we one-hot them explicitly to force
a different decision path than tree splits on continuous features. This is genuinely
novel — no existing experiment quantile-bins then one-hots before LightGBM.

Feature treatment: Raw features → KBinsDiscretizer (10 quantile bins) → one-hot → LightGBM
Decision rule: tree-based (GBDT) on binary one-hot features
Algorithm family: gradient boosting
Composition: single model
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import lightgbm as lgb

from shared.data import load_classifier_data
from shared.eval import evaluate_proba, report

# Load data
Xt, yt, Xv, yv = load_classifier_data()

# Quantile-bin each feature into 10 decile bins, then one-hot encode
disc = KBinsDiscretizer(n_bins=10, encode='onehot-dense', strategy='quantile')
Xt_b = disc.fit_transform(Xt)  # Shape: (n_samples, n_features * 10)
Xv_b = disc.transform(Xv)

# Store feature info for reporting
n_original_features = Xt.shape[1]
n_binary_features = Xt_b.shape[1]

# Train LightGBM classifier on one-hot features
m = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=5,
    class_weight='balanced',
    random_state=42,
    verbose=-1
)
m.fit(Xt_b, yt)

# Predict probabilities
p_train = m.predict_proba(Xt_b)[:, 1]
p_test = m.predict_proba(Xv_b)[:, 1]

# Evaluate
metrics = evaluate_proba(p_train, p_test, yt, yv)

# Add feature engineering details
metrics['n_original_features'] = int(n_original_features)
metrics['n_binary_features_onehot'] = int(n_binary_features)
metrics['n_bins'] = 10

# Report
report(
    "124_quantile_bin_onehot_lgbm",
    "Quantile-binning (10 deciles) + one-hot + LightGBM. Treats continuous features as categorical.",
    metrics,
    extra={
        "architecture": "quantile binning → one-hot encoding → gradient boosting",
        "n_original_features": n_original_features,
        "n_binary_onehot_features": n_binary_features,
        "lgbm_n_estimators": 200,
        "lgbm_max_depth": 5,
        "strategy": "quantile"
    }
)
