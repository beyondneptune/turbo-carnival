"""Quantile Regression Meta-Classifier — novel architecture using LightGBM quantile regression.

Architecture: Train three LightGBM regressors in QUANTILE mode (alpha=0.25, 0.5, 0.75)
to model the conditional distribution of binary targets. Use the 50th percentile (median)
prediction as a point estimate, then calibrate via sigmoid to derive a binary probability.

Novel elements:
  1. Loss function: quantile regression (non-standard for binary classification)
  2. Decision rule: distribution-based (median of conditional quantiles)
  3. Calibration: sigmoid on quantile predictions (p = sigmoid(5 * (q50 - 0.5)))

This has not been attempted in the existing architecture pool.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.special import expit  # sigmoid
import lightgbm as lgb

from shared.data import load_classifier_data
from shared.eval import evaluate_proba, report

# Load data
Xt, yt, Xv, yv = load_classifier_data()

# Step 1: Convert binary labels to {0.0, 1.0} regression targets
yt_reg = yt.astype(np.float32)
yv_reg = yv.astype(np.float32)

# Step 2: Train THREE LightGBM regressors with quantile regression
print("Training quantile regressors (q=0.25, 0.5, 0.75)...")

params_base = {
    "objective": "quantile",
    "metric": "quantile",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 200,
    "verbose": -1,
}

# Q1: 25th percentile
model_q25 = lgb.LGBMRegressor(**params_base, alpha=0.25)
model_q25.fit(Xt, yt_reg, eval_set=[(Xv, yv_reg)], eval_metric="quantile")

# Q2: 50th percentile (median)
model_q50 = lgb.LGBMRegressor(**params_base, alpha=0.50)
model_q50.fit(Xt, yt_reg, eval_set=[(Xv, yv_reg)], eval_metric="quantile")

# Q3: 75th percentile
model_q75 = lgb.LGBMRegressor(**params_base, alpha=0.75)
model_q75.fit(Xt, yt_reg, eval_set=[(Xv, yv_reg)], eval_metric="quantile")

# Step 3: Get predictions on train and test sets
print("Making quantile predictions...")
q25_train = model_q25.predict(Xt)
q50_train = model_q50.predict(Xt)
q75_train = model_q75.predict(Xt)

q25_test = model_q25.predict(Xv)
q50_test = model_q50.predict(Xv)
q75_test = model_q75.predict(Xv)

# Step 4: Calibrate to probability via sigmoid
# p = sigmoid(steepness * (q50 - 0.5))
# Steepness = 5.0 is tunable; it controls how aggressively the quantile maps to probability.
steepness = 5.0
p_train = expit(steepness * (q50_train - 0.5))
p_test = expit(steepness * (q50_test - 0.5))

# Clip to valid probability range [0, 1]
p_train = np.clip(p_train, 1e-7, 1.0 - 1e-7)
p_test = np.clip(p_test, 1e-7, 1.0 - 1e-7)

# Step 5: Evaluate and report
print("Evaluating...")
metrics = evaluate_proba(p_train, p_test, yt, yv)

# Add quantile-specific diagnostics
extra = {
    "q50_median_train": float(np.median(q50_train)),
    "q50_median_test": float(np.median(q50_test)),
    "q25_median": float(np.median(q25_test)),
    "q75_median": float(np.median(q75_test)),
    "steepness_param": steepness,
    "quantile_spread_test": float(np.median(q75_test - q25_test)),
}

report(
    "110_quantile_regression_meta",
    "LightGBM quantile regressors (Q25/Q50/Q75) on binary targets, "
    "median prediction calibrated via sigmoid(5*(q50-0.5))",
    metrics,
    extra=extra
)
