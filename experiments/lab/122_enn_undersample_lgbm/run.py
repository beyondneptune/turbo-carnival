"""Edited Nearest Neighbors (ENN) undersampling + LightGBM.

Architecture: Class imbalance handler (remove majority samples whose class
disagrees with k-NN majority) + tree-based classifier. ENN sharpens decision
boundaries more aggressively than Tomek-links by considering full k-neighborhood,
not just nearest pair.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from imblearn.under_sampling import EditedNearestNeighbours
import lightgbm as lgb

from shared.data import load_classifier_data
from shared.eval import evaluate_proba, report

Xt, yt, Xv, yv = load_classifier_data()

# Apply ENN undersampling on training data
# k=3 neighbors, mode selection (majority class in neighborhood)
enn = EditedNearestNeighbours(n_neighbors=3, kind_sel='mode')
X_resampled, y_resampled = enn.fit_resample(Xt, yt)

removed_count = len(Xt) - len(X_resampled)
print(f"Removed {removed_count} majority samples via ENN (from {len(Xt)} to {len(X_resampled)})")

# Train LightGBM on cleaned training data
m = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=7,
    learning_rate=0.1,
    num_leaves=31,
    scale_pos_weight=(y_resampled == 0).sum() / (y_resampled == 1).sum(),
    random_state=42,
    verbose=-1
)
m.fit(X_resampled, y_resampled)

# Evaluate on original test set
p_train = m.predict_proba(Xt)[:, 1]
p_test = m.predict_proba(Xv)[:, 1]

metrics = evaluate_proba(p_train, p_test, yt, yv)
metrics["enn_removed_count"] = int(removed_count)
metrics["enn_original_train_size"] = int(len(Xt))
metrics["enn_resampled_train_size"] = int(len(X_resampled))

report(
    "122_enn_undersample_lgbm",
    "ENN undersampling (k=3, mode) removes majority samples with disagreeing class labels, then LGBM trains on sharpened boundaries",
    metrics
)
