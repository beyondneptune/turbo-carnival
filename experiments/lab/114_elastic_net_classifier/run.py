"""Elastic-Net Logistic Regression — combined L1 + L2 penalty.

Architecture: elastic-net regularization (l1_ratio=0.5) combines feature selection
(L1 sparsity) with smooth shrinkage (L2 ridge). This differs from pure L1 (031)
and pure L2 (001) by doing both simultaneously.

The saga solver supports elasticnet penalty and allows iterative training.
Extra field tracks feature sparsity: which coefficients are exactly zero.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from shared.data import load_classifier_data, CLASSIFIER_FEATURES
from shared.eval import evaluate_proba, report

Xt, yt, Xv, yv = load_classifier_data()

# Elastic-net with l1_ratio=0.5 (50% L1, 50% L2)
# saga solver supports elasticnet penalty
m = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(
        penalty='elasticnet',
        l1_ratio=0.5,
        solver='saga',
        max_iter=3000,
        class_weight='balanced',
        random_state=42
    ))
])

m.fit(Xt, yt)

# Evaluate on train and test
metrics = evaluate_proba(m.predict_proba(Xt)[:, 1], m.predict_proba(Xv)[:, 1], yt, yv)

# Track feature sparsity: count zero vs nonzero coefficients
clf = m.named_steps['clf']
coef = clf.coef_[0]
zero_mask = coef == 0.0
num_zero = int(zero_mask.sum())
num_nonzero = len(coef) - num_zero

extra = {
    "num_features": len(CLASSIFIER_FEATURES),
    "num_zero_coefficients": num_zero,
    "num_nonzero_coefficients": num_nonzero,
    "zero_features": [CLASSIFIER_FEATURES[i] for i in range(len(CLASSIFIER_FEATURES)) if zero_mask[i]],
    "nonzero_features": [CLASSIFIER_FEATURES[i] for i in range(len(CLASSIFIER_FEATURES)) if not zero_mask[i]],
    "l1_ratio": 0.5,
    "solver": "saga",
}

report(
    "114_elastic_net_classifier",
    "elastic-net logistic regression: l1_ratio=0.5 combines feature selection (L1) with shrinkage (L2), saga solver",
    metrics,
    extra=extra
)
