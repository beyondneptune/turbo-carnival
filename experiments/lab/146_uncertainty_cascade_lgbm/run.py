"""Uncertainty-Cascade LGBM — two-stage routing by confidence.

Architecture: Cascade classifier that routes uncertain samples (0.3 < p < 0.7)
from a small, fast LGBM (Stage 1) to a larger, more powerful LGBM (Stage 2).

Stage 1: Small LGBM (n_estimators=50, max_depth=3) — fast baseline.
Stage 2: Large LGBM (n_estimators=400, max_depth=8) — only for uncertain samples.

Unlike 028_cascade_reject which rejected low-prob samples, this one IDENTIFIES
uncertain samples and routes them to a more powerful model for refinement.
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

# Stage 1: Small, fast LGBM
print("Training Stage 1 (small LGBM)...")
stage1 = lgb.LGBMClassifier(
    n_estimators=50,
    max_depth=3,
    class_weight='balanced',
    random_state=42,
    verbose=-1
)
stage1.fit(X_train, y_train)

# Get Stage 1 predictions
p1_train = stage1.predict_proba(X_train)[:, 1]
p1_test = stage1.predict_proba(X_test)[:, 1]

# Identify uncertain samples: 0.3 < p < 0.7
unc_train = (p1_train > 0.3) & (p1_train < 0.7)
unc_test = (p1_test > 0.3) & (p1_test < 0.7)

n_unc_train = unc_train.sum()
n_unc_test = unc_test.sum()
print(f"Uncertain bars: train {n_unc_train}/{len(X_train)} ({100*n_unc_train/len(X_train):.1f}%), "
      f"test {n_unc_test}/{len(X_test)} ({100*n_unc_test/len(X_test):.1f}%)")

# Stage 2: Large LGBM trained only on uncertain samples from Stage 1
print("Training Stage 2 (large LGBM on uncertain subset)...")
stage2 = lgb.LGBMClassifier(
    n_estimators=400,
    max_depth=8,
    class_weight='balanced',
    random_state=42,
    verbose=-1
)

if n_unc_train > 0:
    stage2.fit(X_train[unc_train], y_train[unc_train])
    print(f"Stage 2 trained on {n_unc_train} uncertain samples")
else:
    print("No uncertain samples in training set; Stage 2 will not be used")

# Combine predictions: use Stage 2 for uncertain, Stage 1 otherwise
p_train = p1_train.copy()
p_test = p1_test.copy()

if n_unc_train > 0:
    p2_train = stage2.predict_proba(X_train[unc_train])[:, 1]
    p_train[unc_train] = p2_train

if n_unc_test > 0:
    p2_test = stage2.predict_proba(X_test[unc_test])[:, 1]
    p_test[unc_test] = p2_test

# Evaluate
metrics = evaluate_proba(p_train, p_test, y_train, y_test)

# Add cascade-specific metrics
metrics['stage1_routed_count_train'] = int(n_unc_train)
metrics['stage1_routed_count_test'] = int(n_unc_test)
metrics['cascade_routed_pct_train'] = float(100 * n_unc_train / len(X_train))
metrics['cascade_routed_pct_test'] = float(100 * n_unc_test / len(X_test))

# Report
report(
    "146_uncertainty_cascade_lgbm",
    "Cascade classifier: Stage 1 (small LGBM) routes uncertain samples (0.3 < p < 0.7) to Stage 2 (large LGBM)",
    metrics
)
