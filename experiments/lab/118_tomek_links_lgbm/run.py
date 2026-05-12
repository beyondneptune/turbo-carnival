"""Tomek-links undersampling + LightGBM.

Tomek links identify pairs of opposite-class nearest neighbors. Removing the
majority-class member of each pair cleans the overlap zone by deleting noisy
boundary points. This is pure UNDERSAMPLING (opposite of SMOTE/ADASYN).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imblearn.under_sampling import TomekLinks
import lightgbm as lgb

from shared.data import load_classifier_data
from shared.eval import evaluate_proba, report

Xt, yt, Xv, yv = load_classifier_data()

# Apply Tomek-links undersampling to training set
tl = TomekLinks(sampling_strategy='majority')
X_cleaned, y_cleaned = tl.fit_resample(Xt, yt)
tomek_pairs_removed = len(Xt) - len(X_cleaned)
print(f"Removed {tomek_pairs_removed} majority-class samples via Tomek-link cleaning")

# Train LightGBM on cleaned training set
m = lgb.LGBMClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    num_leaves=31,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
m.fit(X_cleaned, y_cleaned)

# Evaluate on original train and test sets
metrics = evaluate_proba(
    m.predict_proba(Xt)[:, 1],
    m.predict_proba(Xv)[:, 1],
    yt,
    yv
)

# Add Tomek-links metadata to results
metrics["tomek_pairs_removed"] = int(tomek_pairs_removed)

report(
    "118_tomek_links_lgbm",
    "Tomek-link undersampling + LightGBM: majority-class boundary cleaning for decision boundary sharpening",
    metrics
)
