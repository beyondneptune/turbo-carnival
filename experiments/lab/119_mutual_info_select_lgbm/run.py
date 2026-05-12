"""Mutual Information feature selection + LightGBM.

Architecture: SelectKBest (mutual_info_classif) → LightGBM.

MI is a non-linear dependency measure between each feature and the target.
Differs from permutation importance (uses model predictions) and L1 (linear
coefficient zeroing). MI captures any monotone or non-monotone relationship.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.feature_selection import mutual_info_classif, SelectKBest
import lightgbm as lgb

from shared.data import load_classifier_data, CLASSIFIER_FEATURES
from shared.eval import evaluate_proba, report

Xt, yt, Xv, yv = load_classifier_data()

# Compute mutual information scores for all features
mi_scores = mutual_info_classif(Xt, yt, random_state=42)

# Select top 7 features by MI score
selector = SelectKBest(score_func=mutual_info_classif, k=7)
selector.fit(Xt, yt)
selected_idx = selector.get_support(indices=True)

# Transform data to selected features
Xt_sel = selector.transform(Xt)
Xv_sel = selector.transform(Xv)

# Get feature names
selected_features = [CLASSIFIER_FEATURES[i] for i in selected_idx]

# Build MI score dict with all features
mi_dict = {CLASSIFIER_FEATURES[i]: float(mi_scores[i]) for i in range(len(CLASSIFIER_FEATURES))}

# Train LightGBM on selected features
m = lgb.LGBMClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42,
    verbose=-1
)
m.fit(Xt_sel, yt)

# Predict and evaluate
report(
    "119_mutual_info_select_lgbm",
    "Mutual information feature selection (k=7) → LightGBM; non-linear dependency measure",
    evaluate_proba(m.predict_proba(Xt_sel)[:, 1], m.predict_proba(Xv_sel)[:, 1], yt, yv),
    extra={
        "mi_scores": mi_dict,
        "selected_features": selected_features,
    }
)
