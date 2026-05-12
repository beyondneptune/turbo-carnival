"""Co-training with disjoint feature views — semi-supervised ensemble.

Architecture: two base classifiers (LogReg + LightGBM) trained on disjoint
feature subsets. Pseudo-labels from one classifier's confident predictions
augment the training signal for the other, iteratively. Final ensemble averages
both classifiers' probability estimates.

This is genuinely novel for the lab: semi-supervised co-training with
heterogeneous feature views and asymmetric base models.

Features split:
- View A (price): ret1, ret3, ret10, ret1_z, vwap_dist (5 features)
- View B (volume+time): range_pct, volume_z, vov, tod_sin, tod_cos (5 features)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

from shared.data import load_classifier_data
from shared.eval import evaluate_proba, report

# Feature views: disjoint split
VIEW_A_INDICES = [0, 2, 4, 5, 7]  # ret1, ret10, ret1_z, vwap_dist, (positions in feature list)
VIEW_B_INDICES = [3, 1, 9, 8, 6]  # range_pct, ret3, vov, tod_cos, tod_sin (reordered for clarity)

# Verify indices match spec
CLASSIFIER_FEATURES = [
    "ret1", "ret3", "ret10", "range_pct", "volume_z", "ret1_z",
    "vwap_dist", "tod_sin", "tod_cos", "vov",
]
# Map feature names to indices
VIEW_A_NAMES = ["ret1", "ret3", "ret10", "ret1_z", "vwap_dist"]
VIEW_B_NAMES = ["range_pct", "volume_z", "vov", "tod_sin", "tod_cos"]
VIEW_A_INDICES = [CLASSIFIER_FEATURES.index(f) for f in VIEW_A_NAMES]
VIEW_B_INDICES = [CLASSIFIER_FEATURES.index(f) for f in VIEW_B_NAMES]


def cotraining_step(X_train: np.ndarray, y_train: np.ndarray,
                    X_unlabeled: np.ndarray,
                    clf_A, clf_B,
                    view_a_idx, view_b_idx,
                    top_k: int = 50) -> tuple:
    """One iteration of co-training.

    Train both classifiers on labeled data, get confident pseudo-labels
    from each, add to the other's training set.

    Returns: updated (clf_A, clf_B, pseudo_y_from_A, pseudo_y_from_B)
    """
    # Train classifiers on current labeled set
    clf_A.fit(X_train[:, view_a_idx], y_train)
    clf_B.fit(X_train[:, view_b_idx], y_train)

    # Get confidence predictions on full training set
    proba_A = clf_A.predict_proba(X_train[:, view_a_idx])[:, 1]
    proba_B = clf_B.predict_proba(X_train[:, view_b_idx])[:, 1]

    # Select high-confidence positives (class 1) from A
    top_pos_A = np.argsort(proba_A)[-top_k:]
    # Select high-confidence negatives (class 0) from A
    top_neg_A = np.argsort(proba_A)[:top_k]

    # Select high-confidence positives from B
    top_pos_B = np.argsort(proba_B)[-top_k:]
    # Select high-confidence negatives from B
    top_neg_B = np.argsort(proba_B)[:top_k]

    pseudo_indices_from_A = np.concatenate([top_pos_A, top_neg_A])
    pseudo_labels_from_A = np.concatenate([np.ones(top_k), np.zeros(top_k)]).astype(np.int8)

    pseudo_indices_from_B = np.concatenate([top_pos_B, top_neg_B])
    pseudo_labels_from_B = np.concatenate([np.ones(top_k), np.zeros(top_k)]).astype(np.int8)

    return (
        clf_A, clf_B,
        pseudo_indices_from_A, pseudo_labels_from_A,
        pseudo_indices_from_B, pseudo_labels_from_B
    )


# Load data
X_train, y_train, X_test, y_test = load_classifier_data()

# Initialize classifiers
scaler_A = StandardScaler()
scaler_B = StandardScaler()
clf_A = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
clf_B = LGBMClassifier(n_estimators=100, max_depth=7, num_leaves=31, random_state=42, verbose=-1)

# Scale the feature views
X_train_A_scaled = scaler_A.fit_transform(X_train[:, VIEW_A_INDICES])
X_train_B_scaled = scaler_B.fit_transform(X_train[:, VIEW_B_INDICES])

# Track pseudo-labeled data across iterations
pseudo_X_A = []  # Data to add to A from B's confidences
pseudo_y_A = []
pseudo_X_B = []  # Data to add to B from A's confidences
pseudo_y_B = []

# Co-training loop: 3 iterations
num_cotraining_iterations = 3
for iteration in range(num_cotraining_iterations):
    # Current labeled set
    if iteration == 0:
        # First iteration: use original labeled data only
        cur_X_A = X_train_A_scaled
        cur_y_A = y_train
        cur_X_B = X_train_B_scaled
        cur_y_B = y_train
    else:
        # Subsequent iterations: add pseudo-labels from the other view
        if len(pseudo_X_A) > 0:
            cur_X_A = np.vstack([X_train_A_scaled, np.array(pseudo_X_A)])
            cur_y_A = np.concatenate([y_train, np.array(pseudo_y_A)])
        else:
            cur_X_A = X_train_A_scaled
            cur_y_A = y_train

        if len(pseudo_X_B) > 0:
            cur_X_B = np.vstack([X_train_B_scaled, np.array(pseudo_X_B)])
            cur_y_B = np.concatenate([y_train, np.array(pseudo_y_B)])
        else:
            cur_X_B = X_train_B_scaled
            cur_y_B = y_train

    # Train classifiers
    clf_A.fit(cur_X_A, cur_y_A)
    clf_B.fit(cur_X_B, cur_y_B)

    # Get confident predictions from each on original labeled set
    proba_A = clf_A.predict_proba(X_train_A_scaled)[:, 1]
    proba_B = clf_B.predict_proba(X_train_B_scaled)[:, 1]

    # A's high-confidence samples → pseudo-labels for B
    top_pos_A_idx = np.argsort(proba_A)[-50:]
    top_neg_A_idx = np.argsort(proba_A)[:50]

    new_pseudo_X_B_from_A = X_train_B_scaled[np.concatenate([top_pos_A_idx, top_neg_A_idx])]
    new_pseudo_y_B_from_A = np.concatenate([np.ones(50), np.zeros(50)]).astype(np.int8)

    # B's high-confidence samples → pseudo-labels for A
    top_pos_B_idx = np.argsort(proba_B)[-50:]
    top_neg_B_idx = np.argsort(proba_B)[:50]

    new_pseudo_X_A_from_B = X_train_A_scaled[np.concatenate([top_pos_B_idx, top_neg_B_idx])]
    new_pseudo_y_A_from_B = np.concatenate([np.ones(50), np.zeros(50)]).astype(np.int8)

    # Update pseudo-label pools for next iteration
    pseudo_X_A = new_pseudo_X_A_from_B
    pseudo_y_A = new_pseudo_y_A_from_B
    pseudo_X_B = new_pseudo_X_B_from_A
    pseudo_y_B = new_pseudo_y_B_from_A


# Final training on labeled + all accumulated pseudo-labels
final_X_A = X_train_A_scaled
final_y_A = y_train
if len(pseudo_X_A) > 0:
    final_X_A = np.vstack([final_X_A, np.array(pseudo_X_A)])
    final_y_A = np.concatenate([final_y_A, np.array(pseudo_y_A)])

final_X_B = X_train_B_scaled
final_y_B = y_train
if len(pseudo_X_B) > 0:
    final_X_B = np.vstack([final_X_B, np.array(pseudo_X_B)])
    final_y_B = np.concatenate([final_y_B, np.array(pseudo_y_B)])

clf_A.fit(final_X_A, final_y_A)
clf_B.fit(final_X_B, final_y_B)

# Predictions on test set
X_test_A_scaled = scaler_A.transform(X_test[:, VIEW_A_INDICES])
X_test_B_scaled = scaler_B.transform(X_test[:, VIEW_B_INDICES])

proba_A_train = clf_A.predict_proba(X_train_A_scaled)[:, 1]
proba_B_train = clf_B.predict_proba(X_train_B_scaled)[:, 1]

proba_A_test = clf_A.predict_proba(X_test_A_scaled)[:, 1]
proba_B_test = clf_B.predict_proba(X_test_B_scaled)[:, 1]

# Ensemble: average probabilities
proba_train_ensemble = (proba_A_train + proba_B_train) / 2.0
proba_test_ensemble = (proba_A_test + proba_B_test) / 2.0

# Evaluate
metrics = evaluate_proba(proba_train_ensemble, proba_test_ensemble, y_train, y_test)

report(
    "108_cotraining_disjoint_views",
    "Co-training with disjoint feature views (price vs volume+time). LogReg on View A (price), "
    "LightGBM on View B (vol+time). 3 rounds of pseudo-label augmentation. Final ensemble averages both.",
    metrics
)
