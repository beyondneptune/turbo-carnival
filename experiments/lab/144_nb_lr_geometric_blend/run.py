"""Geometric blend of GaussianNB and LogReg.

Architecture: Ensemble via geometric mean of probabilistic classifiers.
- Train GaussianNB and LogisticRegression independently.
- Compute geometric mean of class-1 probabilities: p_final = sqrt(p_nb * p_lr).
- Penalizes disagreement; both models must be confident for the blend to be confident.
- Differs from 094 (arithmetic median) and 099 (log-odds stack).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from shared.data import load_classifier_data
from shared.eval import evaluate_proba, report

# Load data
Xt, yt, Xv, yv = load_classifier_data()

# Train GaussianNB pipeline
nb = Pipeline([
    ('s', StandardScaler()),
    ('c', GaussianNB())
])
nb.fit(Xt, yt)

# Train LogisticRegression pipeline
lr = Pipeline([
    ('s', StandardScaler()),
    ('c', LogisticRegression(class_weight='balanced', max_iter=2000))
])
lr.fit(Xt, yt)

# Get class-1 probabilities
p_nb_tr = nb.predict_proba(Xt)[:, 1].clip(1e-6, 1 - 1e-6)
p_lr_tr = lr.predict_proba(Xt)[:, 1].clip(1e-6, 1 - 1e-6)
p_nb_te = nb.predict_proba(Xv)[:, 1].clip(1e-6, 1 - 1e-6)
p_lr_te = lr.predict_proba(Xv)[:, 1].clip(1e-6, 1 - 1e-6)

# Geometric mean of probabilities
p_train = np.sqrt(p_nb_tr * p_lr_tr)
p_test = np.sqrt(p_nb_te * p_lr_te)

# Evaluate and report
report(
    "144_nb_lr_geometric_blend",
    "geometric mean blend of GaussianNB and LogisticRegression; both scaled; balanced LR",
    evaluate_proba(p_train, p_test, yt, yv)
)
