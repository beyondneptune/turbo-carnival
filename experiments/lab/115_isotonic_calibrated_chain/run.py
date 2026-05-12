"""Per-feature Isotonic Regression Stacked into a Classifier.

Architecture: Non-parametric monotonic feature transformation via isotonic regression,
stacked with a linear meta-learner.

Genuinely novel — fits a monotonic isotonic regression per feature to capture the
individual monotonic relationship between each feature and the target (without
assuming any parametric form), then uses those transformed features as input to
a logistic regression meta-learner. This is feature-wise non-parametric monotonic
transformation — no existing experiment uses per-feature isotonic regression.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from shared.data import load_classifier_data
from shared.eval import evaluate_proba, report

# Load data
Xt, yt, Xv, yv = load_classifier_data()

# Step 1: Fit per-feature isotonic regression on training data
# Each isotonic model captures the monotonic relationship between one feature and y
iso_models = []
for j in range(Xt.shape[1]):
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(Xt[:, j], yt)
    iso_models.append(iso)

# Step 2: Transform features via isotonic regression
X_train_iso = np.column_stack([iso_models[j].predict(Xt[:, j]) for j in range(Xt.shape[1])])
X_test_iso = np.column_stack([iso_models[j].predict(Xv[:, j]) for j in range(Xv.shape[1])])

# Step 3: Fit meta-learner (logistic regression) on isotonic-transformed features
meta = LogisticRegression(class_weight='balanced', max_iter=1000)
meta.fit(X_train_iso, yt)

# Step 4: Predict and evaluate
p_train = meta.predict_proba(X_train_iso)[:, 1]
p_test = meta.predict_proba(X_test_iso)[:, 1]

report(
    "115_isotonic_calibrated_chain",
    "per-feature isotonic regression (non-parametric monotonic feature transform) stacked with logistic regression meta-learner",
    evaluate_proba(p_train, p_test, yt, yv)
)
