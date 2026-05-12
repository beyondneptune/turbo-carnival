"""KMeans Cluster-then-LogReg — local-expert mixture architecture.

Novel composition: route each sample to a K-Means cluster (k=5), then use a
SEPARATE LogisticRegression per cluster. This is a local-expert gating strategy
distinct from ensemble voting or global-model approaches.

Handles tiny/single-class clusters by falling back to a global classifier.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from shared.data import load_classifier_data
from shared.eval import evaluate_proba, report

# Load data
Xt, yt, Xv, yv = load_classifier_data()

# Standardize features
scaler = StandardScaler().fit(Xt)
Xt_s = scaler.transform(Xt)
Xv_s = scaler.transform(Xv)

# Cluster training data
km = KMeans(n_clusters=5, random_state=42, n_init=10).fit(Xt_s)
cluster_train = km.predict(Xt_s)
cluster_test = km.predict(Xv_s)

# Train per-cluster LogReg classifiers
models = {}
cluster_sizes = {}
for c in range(5):
    mask = cluster_train == c
    n_samples = mask.sum()
    n_pos = (yt[mask] == 1).sum()
    n_neg = (yt[mask] == 0).sum()
    cluster_sizes[c] = int(n_samples)

    # Skip tiny or single-class clusters
    if n_samples < 50 or n_pos == 0 or n_neg == 0:
        continue

    m = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    m.fit(Xt_s[mask], yt[mask])
    models[c] = m

# Train global fallback for samples in clusters without per-cluster models
fallback = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
fallback.fit(Xt_s, yt)

# Per-cluster predict: route each sample to its cluster's classifier
p_train = np.zeros(len(Xt_s))
p_test = np.zeros(len(Xv_s))

for c in range(5):
    mask_tr = cluster_train == c
    mask_te = cluster_test == c

    if c in models:
        # Use cluster-specific model
        p_train[mask_tr] = models[c].predict_proba(Xt_s[mask_tr])[:, 1]
        p_test[mask_te] = models[c].predict_proba(Xv_s[mask_te])[:, 1]
    else:
        # Fall back to global model
        p_train[mask_tr] = fallback.predict_proba(Xt_s[mask_tr])[:, 1]
        p_test[mask_te] = fallback.predict_proba(Xv_s[mask_te])[:, 1]

# Evaluate
metrics = evaluate_proba(p_train, p_test, yt, yv)

# Add cluster analysis to extra metrics
extra = {
    "n_clusters_with_model": int(len(models)),
    "cluster_sizes": cluster_sizes,
}

report(
    "123_kmeans_cluster_then_logreg",
    "local-expert mixture: KMeans (k=5) routing + per-cluster LogReg, fallback to global classifier",
    metrics,
    extra=extra,
)
