"""Shadow Feature Stability Selection + LightGBM classifier.

Architecture: Boruta-style shadow feature selection without installing Boruta.
Novel mechanism:
  1. For each original feature, create a "shadow" feature by random-shuffling values.
  2. Concatenate original + shadow features → 2× width training matrix.
  3. Train a Random Forest on this expanded matrix.
  4. Compute permutation importance for each original feature.
  5. KEEP original features whose importance exceeds the MAX importance of any shadow feature.
  6. Train final LightGBM only on the KEPT features.

This is genuinely novel for the lab — no experiment uses shadow features as a selection mechanism.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from lightgbm import LGBMClassifier

from shared.data import load_classifier_data, CLASSIFIER_FEATURES
from shared.eval import evaluate_proba, report

# Load data
Xt, yt, Xv, yv = load_classifier_data()
n_features = Xt.shape[1]

# Step 1: Create shadow features by shuffling
np.random.seed(42)
shadow_features = np.zeros_like(Xt)
for i in range(n_features):
    shadow_features[:, i] = np.random.permutation(Xt[:, i])

# Step 2: Concatenate original + shadow features
X_augmented = np.hstack([Xt, shadow_features])

# Step 3: Train Random Forest on augmented matrix
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_augmented, yt)

# Step 4: Compute permutation importance on the augmented matrix
perm_importance = permutation_importance(
    rf, X_augmented, yt, n_repeats=10, random_state=42, n_jobs=-1
)
importances = perm_importance.importances_mean

# Split importances: original features (first n_features) vs shadows (next n_features)
original_importances = importances[:n_features]
shadow_importances = importances[n_features:]

# Step 5: Keep features where original importance > max(shadow importances)
max_shadow_importance = np.max(shadow_importances)
kept_mask = original_importances > max_shadow_importance
kept_indices = np.where(kept_mask)[0]

# Map back to feature names
all_feature_names = CLASSIFIER_FEATURES
kept_feature_names = [all_feature_names[i] for i in kept_indices]
rejected_feature_names = [all_feature_names[i] for i in range(n_features) if not kept_mask[i]]

# Step 6: Train final LightGBM on kept features only
X_kept = Xt[:, kept_indices]
Xv_kept = Xv[:, kept_indices]

lgbm = LGBMClassifier(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
    is_unbalanced=True
)

lgbm.fit(X_kept, yt)

# Evaluate
metrics = evaluate_proba(
    lgbm.predict_proba(X_kept)[:, 1],
    lgbm.predict_proba(Xv_kept)[:, 1],
    yt, yv
)

# Report with feature selection details
report(
    "106_shadow_feature_selection_lgbm",
    "Shadow feature stability selection + LightGBM; selects features by comparing original vs shuffled importance",
    metrics,
    extra={
        "max_shadow_importance": float(max_shadow_importance),
        "kept_features": kept_feature_names,
        "rejected_features": rejected_feature_names,
        "n_kept": len(kept_feature_names),
        "n_rejected": len(rejected_feature_names),
    }
)
