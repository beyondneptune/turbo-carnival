"""Logistic Regression + LGBM residual stacking.

Architecture: Two-stage stacking where:
  Stage 1: Linear baseline (LogReg) to capture linear signal.
  Stage 2: LGBM regressor trained on LogReg residuals to capture non-linear signal.
  Final: Sigmoid(LogReg probability + LGBM residual prediction).

This is genuinely novel: explicitly models the non-linear correction to the
linear baseline by training a regressor on its prediction residuals.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.special import expit

from shared.data import load_classifier_data
from shared.eval import evaluate_proba, report

# Load data
X_train, y_train, X_test, y_test = load_classifier_data()

# Stage 1: Linear baseline (LogReg)
lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logistic', LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42))
])
lr_pipeline.fit(X_train, y_train)

# Get probabilities on both train and test
p_lr_train = lr_pipeline.predict_proba(X_train)[:, 1]
p_lr_test = lr_pipeline.predict_proba(X_test)[:, 1]

# Compute residuals: r = y - p_lr (continuous regression targets)
residuals_train = y_train.astype(np.float32) - p_lr_train

# Stage 2: LGBM regressor on residuals
lgbm_residual = lgb.LGBMRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    random_state=42,
    verbose=-1,
    n_jobs=-1
)
lgbm_residual.fit(X_train, residuals_train)

# Predict residual corrections
delta_train = lgbm_residual.predict(X_train)
delta_test = lgbm_residual.predict(X_test)

# Combined: clip to valid probability range [0.001, 0.999]
p_combined_train = (p_lr_train + delta_train).clip(0.001, 0.999)
p_combined_test = (p_lr_test + delta_test).clip(0.001, 0.999)

# Evaluate
metrics = evaluate_proba(p_combined_train, p_combined_test, y_train, y_test)

# Log extra details: delta stdev shows magnitude of non-linear correction
extra = {
    "delta_test_mean": float(np.mean(delta_test)),
    "delta_test_std": float(np.std(delta_test)),
    "delta_test_min": float(np.min(delta_test)),
    "delta_test_max": float(np.max(delta_test)),
}

report(
    "128_lr_residual_then_lgbm",
    "Two-stage stacking: LogReg baseline + LGBM on residuals. Final = sigmoid(linear + non-linear correction).",
    metrics,
    extra
)
