"""LightGBM Leaf-Index Encoding -> Logistic Regression Meta-Learner.

Architecture: Train a small LightGBM on raw features, extract the leaf index
that each sample falls into across all sequential gradient-boosted trees, one-hot
encode those leaf indices, then train a LogisticRegression on the resulting
sparse leaf-index features.

This differs from 105_tree_embedding_logreg (RandomForest) in a key way:
GBM trees are sequential and fit residuals from prior trees, so each tree's
leaf partition captures gradient-boosted path information, which is qualitatively
different from RF's parallel, independent tree partitions.

Decision rule: GBM leaf indices (discrete, path-dependent) + linear boundary.
Feature treatment: raw -> LGBM leaves -> one-hot -> linear.
Algorithm family: boosting (LGBM) composed with linear meta-learner (stacking).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

from shared.data import load_classifier_data
from shared.eval import evaluate_proba, report

# Load data
Xt, yt, Xv, yv = load_classifier_data()

# Train a small LightGBM classifier to define leaf structure
# Use small n_estimators and max_depth to keep the leaf space compact
# num_leaves controls max leaves per tree (2^max_depth - 1 by default)
gbm = lgb.LGBMClassifier(
    n_estimators=80,
    max_depth=4,
    num_leaves=15,
    class_weight='balanced',
    random_state=42,
    verbose=-1
)
gbm.fit(Xt, yt)

# Extract leaf indices: (n_samples, n_estimators)
# Each entry is the leaf node index that the sample fell into for that tree
# pred_leaf=True returns the leaf indices instead of predictions
leaves_train = gbm.predict(Xt, pred_leaf=True)
leaves_test = gbm.predict(Xv, pred_leaf=True)

# One-hot encode the leaf indices
# Each tree's leaves become a sparse categorical feature
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
ohe.fit(leaves_train)
X_train_emb = ohe.transform(leaves_train)
X_test_emb = ohe.transform(leaves_test)

# Train logistic regression on the embedded features
# Use heavy regularization (low C) since we have many sparse features
lr = LogisticRegression(
    class_weight='balanced',
    max_iter=2000,
    solver='lbfgs',
    C=0.1,  # heavy regularization to combat overfitting on sparse features
    random_state=42
)
lr.fit(X_train_emb, yt)

# Predict and evaluate
p_train = lr.predict_proba(X_train_emb)[:, 1]
p_test = lr.predict_proba(X_test_emb)[:, 1]

# Compute total leaf count
total_leaf_count = gbm.n_estimators * gbm.num_leaves

report(
    "134_lgbm_leaf_encoding_to_logreg",
    "LightGBM leaf-index embeddings (80 trees, max_depth=4, num_leaves=15) -> one-hot -> LogReg; "
    "feature treatment: raw -> sequential GBM leaves -> sparse one-hot -> linear; "
    f"total leaves: {total_leaf_count}",
    evaluate_proba(p_train, p_test, yt, yv),
    extra={"total_leaf_count": total_leaf_count}
)
