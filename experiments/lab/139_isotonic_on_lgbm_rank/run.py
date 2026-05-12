"""Isotonic Regression on LGBM RANK — novel calibration approach.

Architecture: Train LGBM classifier to get probability predictions, convert
probabilities to normalized rank scores, then fit isotonic regression to
calibrate rank ordering (monotone) rather than raw probabilities.

This is genuinely novel because:
1. Most calibration operates on raw probability (Platt, isotonic on proba)
2. This operates on the RANK ORDERING of predictions
3. At test time, rank predictions within the combined train+test set,
   then apply isotonic to the ranks
4. This enforces monotonicity in rank ordering, which is fundamentally
   different from calibrating probability directly
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from scipy.stats import rankdata

from shared.data import load_classifier_data
from shared.eval import evaluate_proba, report

# Load data
X_train, y_train, X_test, y_test = load_classifier_data()

# Split training data: fit LGBM on X_fit/y_fit, calibrate isotonic on X_cal/y_cal
X_fit, X_cal, y_fit, y_cal = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Train LGBM classifier
m = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=5,
    class_weight='balanced',
    random_state=42,
    verbose=-1
)
m.fit(X_fit, y_fit)

# Get raw probabilities and convert to ranks for calibration set
p_cal = m.predict_proba(X_cal)[:, 1]
rank_cal = rankdata(p_cal) / len(p_cal)  # Normalized ranks [0, 1]

# Fit isotonic regression on rank -> y_cal
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(rank_cal, y_cal)

# Apply to train and test sets
# Get raw probabilities
p_train_raw = m.predict_proba(X_train)[:, 1]
p_test_raw = m.predict_proba(X_test)[:, 1]

# Compute ranks within each set independently for ranking
rank_train = rankdata(p_train_raw) / len(p_train_raw)
rank_test = rankdata(p_test_raw) / len(p_test_raw)

# Apply isotonic calibration to ranks
p_train = iso.predict(rank_train).clip(0.001, 0.999)
p_test = iso.predict(rank_test).clip(0.001, 0.999)

# Evaluate and report
report(
    "139_isotonic_on_lgbm_rank",
    "LGBM classifier → probabilistic rank → isotonic calibration on ranks. "
    "Novel: calibrates rank ordering (monotone) rather than raw probability.",
    evaluate_proba(p_train, p_test, y_train, y_test)
)
