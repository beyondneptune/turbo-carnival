"""Tree Embedding + Logistic Regression — use RandomForest leaf indices as features.

Architecture: Train a small RandomForest on raw features, extract the leaf index
that each sample falls into across all trees, one-hot encode those leaf indices,
then train a LogisticRegression on the resulting sparse leaf-index features.

This is a classic technique from "Practical Lessons from Predicting Clicks on Ads"
(Facebook 2014) and is genuinely novel because it treats the tree as a feature
transformer (not a classifier). None of 001-104 use leaf-index embeddings.

Decision rule: tree leaf indices (discrete, interpretable) + linear boundary.
Feature treatment: raw -> tree leaves -> one-hot -> linear.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

from shared.data import load_classifier_data
from shared.eval import evaluate_proba, report

# Load data
Xt, yt, Xv, yv = load_classifier_data()

# Train a shallow random forest to define leaf structure
# Use small n_estimators and max_depth to keep the leaf space compact
rf = RandomForestClassifier(
    n_estimators=50,
    max_depth=4,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)
rf.fit(Xt, yt)

# Extract leaf indices: (n_samples, n_estimators)
# Each entry is the leaf node index that the sample fell into for that tree
leaves_train = rf.apply(Xt)
leaves_test = rf.apply(Xv)

# One-hot encode the leaf indices
# Each tree's leaves become a sparse categorical feature
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
ohe.fit(leaves_train)
X_train_emb = ohe.transform(leaves_train)
X_test_emb = ohe.transform(leaves_test)

# Train logistic regression on the embedded features
lr = LogisticRegression(
    class_weight='balanced',
    max_iter=2000,
    solver='lbfgs',
    random_state=42
)
lr.fit(X_train_emb, yt)

# Predict and evaluate
p_train = lr.predict_proba(X_train_emb)[:, 1]
p_test = lr.predict_proba(X_test_emb)[:, 1]

report(
    "105_tree_embedding_logreg",
    "RandomForest leaf-index embeddings (50 trees, depth=4) -> one-hot -> LogReg; "
    "feature treatment: raw -> discrete tree leaves -> sparse one-hot -> linear",
    evaluate_proba(p_train, p_test, yt, yv)
)
