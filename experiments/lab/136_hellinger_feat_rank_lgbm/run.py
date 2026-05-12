"""Hellinger-distance feature ranking + LightGBM.

Architecture: Information-theoretic feature selection (Hellinger distance),
followed by light GBM on top-6 features. Hellinger distance is symmetric
(unlike KL divergence) and bounded in [0, 1]. Per-feature, compute Hellinger
distance between the conditional distribution of feature | y=1 and feature | y=0
using histograms with 20 quantile bins. Rank features, keep TOP 6, train LGBM.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import lightgbm as lgb

from shared.data import load_classifier_data, CLASSIFIER_FEATURES
from shared.eval import evaluate_proba, report


def hellinger_per_feature(X, y, n_bins=20):
    """For each column, compute Hellinger distance between P(x|y=1) and P(x|y=0).

    Hellinger distance H(p, q) = sqrt(0.5 * sum((sqrt(p_i) - sqrt(q_i))^2))
    is symmetric and bounded in [0, 1].

    Args:
        X: feature matrix of shape (n_samples, n_features)
        y: binary labels of shape (n_samples,)
        n_bins: number of quantile bins

    Returns:
        h: Hellinger distance per feature, shape (n_features,)
    """
    h = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        col = X[:, j]
        # Common bins from full distribution (quantile-based)
        quantiles = np.quantile(col, np.linspace(0, 1, n_bins + 1))
        # Avoid duplicate bin edges
        quantiles[0] -= 1e-9
        quantiles[-1] += 1e-9

        # Conditional histograms
        p1, _ = np.histogram(col[y == 1], bins=quantiles)
        p0, _ = np.histogram(col[y == 0], bins=quantiles)

        # Normalize to probabilities
        p1 = p1 / max(p1.sum(), 1)
        p0 = p0 / max(p0.sum(), 1)

        # Hellinger distance
        h[j] = np.sqrt(0.5 * ((np.sqrt(p1) - np.sqrt(p0)) ** 2).sum())

    return h


# Load data
X_train, y_train, X_test, y_test = load_classifier_data()

# Compute Hellinger distance for each feature
h_scores = hellinger_per_feature(X_train, y_train, n_bins=20)

# Rank features by Hellinger distance (descending)
top6_indices = np.argsort(-h_scores)[:6]

# Print feature ranking and selection
print("Hellinger distance feature ranking:")
for i, f in enumerate(CLASSIFIER_FEATURES):
    print(f"  {f:14s}  H = {h_scores[i]:.4f}")

top6_names = [CLASSIFIER_FEATURES[i] for i in top6_indices]
print(f"\nTop-6 selected features: {top6_names}")

# Train LightGBM on top-6 features
X_train_top6 = X_train[:, top6_indices]
X_test_top6 = X_test[:, top6_indices]

m = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=5,
    class_weight='balanced',
    random_state=42,
    verbose=-1
)
m.fit(X_train_top6, y_train)

# Evaluate
p_train = m.predict_proba(X_train_top6)[:, 1]
p_test = m.predict_proba(X_test_top6)[:, 1]

results = evaluate_proba(p_train, p_test, y_train, y_test)

# Store Hellinger scores in extra field
extra = {
    "top6_features": top6_names,
    "hellinger_scores": {CLASSIFIER_FEATURES[i]: float(h_scores[i]) for i in range(len(CLASSIFIER_FEATURES))},
}

report(
    "136_hellinger_feat_rank_lgbm",
    "Hellinger-distance feature ranking (top-6) + LightGBM. Symmetric, bounded [0,1] feature selection.",
    results,
    extra=extra
)
