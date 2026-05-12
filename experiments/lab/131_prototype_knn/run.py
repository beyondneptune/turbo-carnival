"""Prototype-based KNN with KMeans-derived class prototypes.

Architecture: Cluster positive and negative classes separately using KMeans
to derive class-specific centroids (prototypes). At inference, compute distance
to nearest positive prototype vs nearest negative prototype, and convert via
softmax to probability. This replaces instance-level KNN neighbors with
prototype-level centroids, reducing inference cost from O(n) to O(K) per query
while preserving local class structure through class-specific clustering.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from shared.data import load_classifier_data
from shared.eval import evaluate_proba, report

# Load data
Xt, yt, Xv, yv = load_classifier_data()

# Standardize features
sc = StandardScaler()
Xt_s = sc.fit_transform(Xt)
Xv_s = sc.transform(Xv)

# Separate positive and negative training points
pos_pts = Xt_s[yt == 1]
neg_pts = Xt_s[yt == 0]

# Fit class-specific prototypes using KMeans
K_pos = 50  # number of positive class prototypes
K_neg = 50  # number of negative class prototypes

km_pos = KMeans(n_clusters=K_pos, random_state=42, n_init=10)
km_neg = KMeans(n_clusters=K_neg, random_state=42, n_init=10)

km_pos.fit(pos_pts)
km_neg.fit(neg_pts)

# Define prediction function: compute softmax-style probability
# based on distance to nearest prototype for each class
def predict_proba(X):
    """
    Returns P(positive class) based on distances to nearest prototypes.

    For each sample:
    - d_pos = distance to nearest positive prototype
    - d_neg = distance to nearest negative prototype
    - P(positive) = exp(-d_pos) / (exp(-d_pos) + exp(-d_neg))

    This softmax-style formulation means smaller distance -> higher probability.
    """
    # transform() computes euclidean distances to centroids
    # min(axis=1) gives distance to nearest centroid for each sample
    d_pos = km_pos.transform(X).min(axis=1)
    d_neg = km_neg.transform(X).min(axis=1)

    # Softmax: P(pos) proportional to exp(-d_pos)
    exp_pos = np.exp(-d_pos)
    exp_neg = np.exp(-d_neg)
    p_pos = exp_pos / (exp_pos + exp_neg)

    return p_pos

# Generate train and test probabilities
p_train = predict_proba(Xt_s)
p_test = predict_proba(Xv_s)

# Evaluate and report
report(
    "131_prototype_knn",
    "KMeans-derived class prototypes (K_pos=50, K_neg=50); distance to nearest prototype determines class probability via softmax",
    evaluate_proba(p_train, p_test, yt, yv)
)
