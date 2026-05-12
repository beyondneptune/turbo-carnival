"""Bayesian Ridge with sigmoid mapping for classification.

Architecture: Bayesian linear regression (conjugate Gaussian-Gaussian prior)
fitted on continuous y∈{0,1}, predictions mapped through sigmoid for probabilities.

This is genuinely different from LogReg: uses Bayes-optimal MAP estimation under
a Gaussian prior over coefficients (conjugate prior) rather than maximum likelihood.
BayesianRidge also provides posterior uncertainty, though we use point predictions here.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.special import expit  # sigmoid function

from shared.data import load_classifier_data
from shared.eval import evaluate_proba, report

# Load data
Xt, yt, Xv, yv = load_classifier_data()

# Build pipeline: standardize features, fit Bayesian Ridge
m = Pipeline([
    ("scaler", StandardScaler()),
    ("reg", BayesianRidge(max_iter=500, tol=1e-3, alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6))
])

# Fit on continuous y (0.0 and 1.0)
m.fit(Xt, yt.astype(float))

# Get regression predictions (continuous outputs)
y_train_pred = m.predict(Xt)
y_test_pred = m.predict(Xv)

# Map through sigmoid with steepness factor
# Center around 0.5 and apply sigmoid: expit(5 * (pred - 0.5))
# This maps regression outputs [0,1] → probabilities via sigmoid steepness
p_train = expit(5 * (y_train_pred - 0.5))
p_test = expit(5 * (y_test_pred - 0.5))

# Extract model parameters for extra field
br_model = m.named_steps["reg"]
extra_dict = {
    "alpha_": float(br_model.alpha_),
    "lambda_": float(br_model.lambda_),
    "sigma_": float(br_model.sigma_[0, 0]) if hasattr(br_model, "sigma_") else None,
}

# Evaluate and report
results = evaluate_proba(p_train, p_test, yt, yv)
report(
    "120_bayesian_ridge_sigmoid",
    "Bayesian linear regression (conjugate Gaussian-Gaussian prior) on y∈{0,1}, sigmoid-mapped predictions",
    results,
    extra=extra_dict
)
