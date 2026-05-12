"""KMeans Cluster-Distance Codes as Features + LogReg.

Architecture: Fit KMeans with k=20 clusters, use distances to all cluster
centers as a 20-dimensional "soft code" feature vector (genuine basis for
feature transformation), then feed to LogisticRegression.

This differs from 123_kmeans_cluster_then_logreg (which routes to per-cluster
classifiers) and 057_anomaly_kmeans_dist (which uses min-distance for anomaly).
Here clusters serve as a FEATURE BASIS for downstream linear classification.
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
sc = StandardScaler().fit(Xt)
Xt_scaled = sc.transform(Xt)
Xv_scaled = sc.transform(Xv)

# Fit KMeans with 20 clusters
km = KMeans(n_clusters=20, random_state=42, n_init=10)
km.fit(Xt_scaled)

# Use DISTANCES to all centroids as new features (soft codes)
# km.transform() returns distances to all k centroids: shape (n, 20)
Xt_codes = km.transform(Xt_scaled)
Xv_codes = km.transform(Xv_scaled)

# Train LogisticRegression on the cluster-distance codes
lr = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)
lr.fit(Xt_codes, yt)

# Predict probabilities
p_train = lr.predict_proba(Xt_codes)[:, 1]
p_test = lr.predict_proba(Xv_codes)[:, 1]

# Evaluate and report
results = evaluate_proba(p_train, p_test, yt, yv)
report(
    "138_kmeans_codes_logreg",
    "KMeans (k=20) cluster-distance codes as feature basis + LogReg",
    results
)
