"""Gradient-boosted LogReg stack — boost a sequence of linear regressors on residuals.

Architecture: Iteratively fit Ridge regressors to the log-loss gradient,
accumulate predictions with shrinkage. Weak learner is LINEAR (not trees).
This is genuinely novel — standard gradient boosting uses tree weak learners;
here we use Ridge regression to model residuals in log-odds space.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.special import expit, logit

from shared.data import load_classifier_data
from shared.eval import evaluate_proba, report

# Load data
Xt, yt, Xv, yv = load_classifier_data()

# Standardize features
scaler = StandardScaler()
scaler.fit(Xt)
Xt_scaled = scaler.transform(Xt)
Xv_scaled = scaler.transform(Xv)

# Initialize with log-odds of class prior
prior = float(yt.mean())
bias = logit(prior)
F_train = np.full(len(Xt_scaled), bias, dtype=np.float64)
F_test = np.full(len(Xv_scaled), bias, dtype=np.float64)

# Hyperparameters
learning_rate = 0.1
n_rounds = 20
ridge_alpha = 1.0

# Boosting loop: fit Ridge to log-loss gradient
boosters = []
for k in range(n_rounds):
    # Probability from current log-odds prediction
    p_train = expit(F_train)

    # Gradient of log-loss: y - p
    grad_train = yt - p_train

    # Fit Ridge regressor on gradient (linear weak learner, not classifier)
    booster = Ridge(alpha=ridge_alpha)
    booster.fit(Xt_scaled, grad_train)
    boosters.append(booster)

    # Update log-odds predictions with scaled weak learner output
    F_train += learning_rate * booster.predict(Xt_scaled)
    F_test += learning_rate * booster.predict(Xv_scaled)

# Convert final log-odds to probabilities
p_train_final = expit(F_train)
p_test_final = expit(F_test)

# Evaluate and report
metrics = evaluate_proba(p_train_final, p_test_final, yt, yv)
report(
    "132_gradient_boosted_logreg_stack",
    "Linear gradient boosting: Ridge regressors on log-loss gradient, "
    f"20 rounds, lr={learning_rate}, alpha={ridge_alpha}",
    metrics
)
