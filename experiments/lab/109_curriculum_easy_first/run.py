"""Curriculum Learning — easy bars first, hard bars later.

Architecture: Adaptive training curriculum based on sample difficulty.

Procedure:
1. Train LogReg on ALL training data to get baseline difficulty scores:
   difficulty[i] = |predict_proba[i] - 0.5| (higher = easier, far from decision boundary)
2. Sort training samples from EASIEST to HARDEST
3. Train fresh LightGBM in 3 stages with warm-start:
   - Stage 1: train on easiest 33% only (100 estimators)
   - Stage 2: warm-start, train on easiest 66% (150 additional estimators)
   - Stage 3: warm-start, train on ALL data (200 additional estimators)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from shared.data import load_classifier_data
from shared.eval import evaluate_proba, report

# Load data
Xt, yt, Xv, yv = load_classifier_data()

# Step 1: Train LogReg baseline to get difficulty scores
scaler = StandardScaler()
Xt_scaled = scaler.fit_transform(Xt)
lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
lr.fit(Xt_scaled, yt)

# Get predicted probabilities and compute difficulty scores
# difficulty = |prob - 0.5| means:
#   - Near 0.5 (boundary) = low difficulty = hard sample
#   - Near 0 or 1 = high difficulty = easy sample
p_train = lr.predict_proba(Xt_scaled)[:, 1]
difficulty = np.abs(p_train - 0.5)

# Step 2: Sort training samples by difficulty (easiest first)
difficulty_order = np.argsort(-difficulty)  # descending (easiest first)
Xt_curriculum = Xt[difficulty_order]
yt_curriculum = yt[difficulty_order]

# Compute stage boundaries
n_train = len(Xt_curriculum)
stage1_cutoff = int(n_train * 0.33)
stage2_cutoff = int(n_train * 0.66)

# Stage 1: Train on easiest 33%
print(f"Stage 1: Training on easiest {stage1_cutoff} samples (33%)")
m1 = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.05,
    num_leaves=31,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
m1.fit(Xt_curriculum[:stage1_cutoff], yt_curriculum[:stage1_cutoff])

# Stage 2: Warm-start from Stage 1, train on easiest 66%
print(f"Stage 2: Warm-start, training on easiest {stage2_cutoff} samples (66%)")
m2 = lgb.LGBMClassifier(
    n_estimators=250,  # 100 + 150 additional
    max_depth=6,
    learning_rate=0.05,
    num_leaves=31,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    verbose=-1,
    warm_start=True
)
# Initialize with Stage 1 booster
m2._Booster = m1.booster_
m2.fit(Xt_curriculum[:stage2_cutoff], yt_curriculum[:stage2_cutoff])

# Stage 3: Warm-start from Stage 2, train on ALL data
print(f"Stage 3: Warm-start, training on all {n_train} samples (100%)")
m3 = lgb.LGBMClassifier(
    n_estimators=450,  # 100 + 150 + 200 additional
    max_depth=6,
    learning_rate=0.05,
    num_leaves=31,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    verbose=-1,
    warm_start=True
)
# Initialize with Stage 2 booster
m3._Booster = m2.booster_
m3.fit(Xt_curriculum, yt_curriculum)

# Evaluate on original (non-curriculum) train/test split
report(
    "109_curriculum_easy_first",
    "Curriculum learning: 3-stage easy-first training on LightGBM (100+150+200 estimators)",
    evaluate_proba(m3.predict_proba(Xt)[:, 1], m3.predict_proba(Xv)[:, 1], yt, yv)
)
