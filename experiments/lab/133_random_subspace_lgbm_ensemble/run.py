"""Random Subspace LightGBM Ensemble — column subsampling at ensemble level.

Architecture: Train 10 separate LightGBM models, each on a RANDOM SUBSET of 5 features
(out of 10 total). Final prediction = mean of all 10 model probabilities.

Different from bagging (044 lgbm_rf, 049 bagging_decision_tree, 050 bagging_logreg):
- Each base learner sees a DIFFERENT FEATURE SUBSET (column subsampling at ensemble level)
- Genuinely simulates random subspace method from Ho 1998
- Underlies original Random Forest's feature randomness
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

# Random subspace ensemble configuration
rng = np.random.default_rng(42)
N_MODELS = 10
K_FEATS = 5

models = []
feature_masks = []

# Train each model on a random subset of features
for i in range(N_MODELS):
    # Random feature subset selection
    feat_idx = rng.choice(Xt.shape[1], size=K_FEATS, replace=False)
    feat_idx = np.sort(feat_idx)  # Sort for consistency

    # Train LightGBM on this feature subset
    m = lgb.LGBMClassifier(
        n_estimators=150,
        max_depth=5,
        class_weight='balanced',
        random_state=42 + i,
        verbose=-1
    )
    m.fit(Xt[:, feat_idx], yt)
    models.append(m)
    feature_masks.append(feat_idx.tolist())

# Average predictions across all 10 models
p_train = np.mean(
    [m.predict_proba(Xt[:, mk])[:, 1] for m, mk in zip(models, feature_masks)],
    axis=0
)
p_test = np.mean(
    [m.predict_proba(Xv[:, mk])[:, 1] for m, mk in zip(models, feature_masks)],
    axis=0
)

# Evaluate and report
metrics = evaluate_proba(p_train, p_test, yt, yv)

# Include feature masks in extra field
extra = {"feature_masks": feature_masks}

report(
    "133_random_subspace_lgbm_ensemble",
    "10 LightGBM models on random 5-feature subsets; ensemble average. "
    "Implements Ho 1998 random subspace method.",
    metrics,
    extra=extra
)
