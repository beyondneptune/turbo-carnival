"""BART-style additive ensemble — Bayesian Additive Regression Trees approximation.

Architecture: Sequential additive regression trees with residual fitting and shrinkage.
- Train 50 shallow trees (max_depth=3) sequentially on residuals of previous trees.
- After each tree, apply shrinkage lambda=0.1.
- Final prediction = sigmoid(sum of all tree outputs).
- Uses sklearn DecisionTreeRegressor for each tree.

This differs from GBM because:
(a) Uses regression trees, not classification trees
(b) Explicit residual fitting with manual shrinkage control
(c) Sum-then-sigmoid at end (BART-style logit additivity)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from scipy.special import expit  # sigmoid function

from shared.data import load_classifier_data
from shared.eval import evaluate_proba, report


class BARTStyleAdditiveEnsemble:
    """BART-style additive regression tree ensemble."""

    def __init__(self, n_trees=50, max_depth=3, shrinkage=0.1, random_state=42):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.shrinkage = shrinkage
        self.random_state = random_state
        self.trees = []
        self.scaler = StandardScaler()
        self.tree_outputs_train = None
        self.tree_outputs_test = None

    def fit(self, X, y):
        """Fit ensemble sequentially on residuals."""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Initialize residuals to y - 0.5 (center the target)
        residuals = y.astype(np.float64) - 0.5

        self.trees = []

        for tree_idx in range(self.n_trees):
            # Fit tree to residuals
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                random_state=self.random_state + tree_idx,
                min_samples_leaf=1
            )
            tree.fit(X_scaled, residuals)
            self.trees.append(tree)

            # Get predictions from this tree
            tree_pred = tree.predict(X_scaled)

            # Apply shrinkage
            tree_pred = self.shrinkage * tree_pred

            # Update residuals for next iteration
            residuals = residuals - tree_pred

        return self

    def predict_proba(self, X):
        """Predict probability: sigmoid of sum of all tree outputs."""
        X_scaled = self.scaler.transform(X)

        # Sum all tree outputs with shrinkage
        sum_predictions = np.zeros(len(X), dtype=np.float64)
        for tree in self.trees:
            tree_pred = tree.predict(X_scaled)
            sum_predictions += self.shrinkage * tree_pred

        # Apply sigmoid to get probability
        proba = expit(sum_predictions)

        return proba

    def predict(self, X):
        """Predict binary class."""
        return (self.predict_proba(X) > 0.5).astype(int)


# Load data
Xt, yt, Xv, yv = load_classifier_data()

# Create and fit model
model = BARTStyleAdditiveEnsemble(n_trees=50, max_depth=3, shrinkage=0.1, random_state=42)
model.fit(Xt, yt)

# Get predictions
p_train = model.predict_proba(Xt)
p_test = model.predict_proba(Xv)

# Evaluate and report
results = evaluate_proba(p_train, p_test, yt, yv)
report(
    "107_bart_style_additive",
    "BART-style additive ensemble: 50 shallow regression trees (depth=3) with shrinkage=0.1, sigmoid final",
    results
)
