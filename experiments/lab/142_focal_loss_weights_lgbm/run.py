"""Focal-loss-style per-example sample weights for LGBM.

Focal loss (Lin et al. 2017) downweights easy examples and focuses on hard ones.
This architecture approximates focal loss via two-stage iterative reweighting:

1. Train baseline LGBM with balanced class weight, get probabilities.
2. Compute per-sample focal weights: w_i = alpha * (1 - p_t_i)^gamma
   where p_t = p if y=1, else (1-p), with alpha=0.25, gamma=2.
3. Retrain LGBM using these focal weights.

This differs from flat class_weight='balanced' (129) and asymmetric ratios (e.g. 3:1)
by making weights EXAMPLE-LEVEL and based on current model CONFIDENCE.

Architecture categories:
- Loss function: focal loss approximation via sample weighting
- Feature treatment: raw
- Composition: two-stage (baseline → focal reweighting)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import lightgbm as lgb

from shared.data import load_classifier_data
from shared.eval import evaluate_proba, report

# Load data
X_train, y_train, X_test, y_test = load_classifier_data()

# Stage 1: Train baseline LGBM with balanced class weight
m1 = lgb.LGBMClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42,
    verbose=-1
)
m1.fit(X_train, y_train)

# Get baseline probabilities on training set
p1 = m1.predict_proba(X_train)[:, 1].clip(1e-3, 1 - 1e-3)

# Stage 2: Compute focal weights per sample
# p_t = probability of the true class
# For y=1, p_t = p(y=1)
# For y=0, p_t = 1 - p(y=1)
p_t = np.where(y_train == 1, p1, 1 - p1)

# Focal weight formula: w_i = alpha * (1 - p_t_i)^gamma
alpha = 0.25
gamma = 2.0
focal_weights = alpha * (1 - p_t) ** gamma + 0.05  # +0.05 floor so easy examples not zeroed

# Stage 3: Retrain LGBM with focal weights
m2 = lgb.LGBMClassifier(
    n_estimators=200,
    random_state=42,
    verbose=-1
)
m2.fit(X_train, y_train, sample_weight=focal_weights)

# Predict on train and test
p_train = m2.predict_proba(X_train)[:, 1]
p_test = m2.predict_proba(X_test)[:, 1]

# Evaluate and report
metrics = evaluate_proba(p_train, p_test, y_train, y_test)
report(
    "142_focal_loss_weights_lgbm",
    "Focal-loss-style per-example sample weights: two-stage baseline→focal reweighting. "
    "Stage 1: LGBM with class_weight='balanced'. "
    "Stage 2: Compute focal weights w_i=0.25*(1-p_t_i)^2+0.05 on baseline predictions. "
    "Stage 3: Retrain LGBM with focal weights.",
    metrics
)
