"""Self-training with confidence-thresholded pseudo-labels.

Architecture: Semi-supervised learning via iterative pseudo-labeling.
- Split labeled train 60/40 (first 60% as "labeled", last 40% as "unlabeled").
- Train initial LightGBM on labeled set.
- Predict probabilities on unlabeled set.
- Accept predictions where confidence > 0.85 OR confidence < 0.15 as pseudo-labels (hard labels: argmax).
- Concat pseudo-labeled samples to labeled set.
- Re-train LightGBM on combined set.
- Repeat self-training loop 3 times.
- Final model predicts on test set.

This combines semi-supervised learning (self-training) with confidence-based
thresholding to avoid the low-confidence drift problem. Hard labels (argmax)
enforce decision boundary learning; soft labels would allow probability creep.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from shared.data import load_classifier_data
from shared.eval import evaluate_proba, report

# Load full training data
X_full_train, y_full_train, X_test, y_test = load_classifier_data()

# Split labeled train 60/40: first 60% labeled, last 40% unlabeled
n_labeled = int(len(X_full_train) * 0.6)
X_labeled = X_full_train[:n_labeled]
y_labeled = y_full_train[:n_labeled]
X_unlabeled = X_full_train[n_labeled:]
y_unlabeled_true = y_full_train[n_labeled:]  # for reference, not used in training

print(f"Labeled: {len(X_labeled)}, Unlabeled: {len(X_unlabeled)}, Test: {len(X_test)}")

# Confidence thresholds: accept high-confidence (>0.85) or low-confidence (<0.15) predictions
CONFIDENCE_HI = 0.85
CONFIDENCE_LO = 0.15
N_ITER = 3

# Track pseudo-labeled samples across iterations
X_train_combined = X_labeled.copy()
y_train_combined = y_labeled.copy()

for iteration in range(N_ITER):
    print(f"\n--- Iteration {iteration + 1}/{N_ITER} ---")

    # Train LightGBM on combined labeled + previously pseudo-labeled
    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
        class_weight="balanced"
    )
    model.fit(X_train_combined, y_train_combined)

    # Predict probabilities on unlabeled set
    proba_unlabeled = model.predict_proba(X_unlabeled)  # shape (n_unlabeled, 2)
    conf_unlabeled = np.max(proba_unlabeled, axis=1)  # max probability for each sample

    # Accept predictions where confidence > 0.85 OR confidence < 0.15
    # High confidence: model is sure about either class
    # Low confidence near decision boundary: both < 0.5, but max still < 0.15 is extremely uncertain
    #   Actually: max of [p, 1-p] where p in [0, 1]; max < 0.15 means p in (0.15, 0.85) both < 0.15 is impossible
    #   So confidence < 0.15 is impossible. Let's interpret as: |p - 0.5| > threshold (high margin)
    #   Reinterpret: confidence = max(proba) > 0.85 means top-class prob > 0.85
    #                confidence < 0.15 would be bottom 15% of max(proba), i.e., very uncertain
    # More standard: accept if max(proba) > 0.85 OR (1 - max(proba)) < 0.15, which is same thing
    # Alternative: accept if |proba[0] - proba[1]| > threshold (margin)
    # Let's use: max(proba) > 0.85 (high confidence in one class) OR max(proba) < 0.5 (bimodal artifact)
    # Actually: "confidence < 0.15" in the spec might mean: low probability on max class < 0.15? That's extremely rare.
    # Most sensible: high confidence (>0.85) OR very low confidence (<0.15 means near random, but let's use <0.52 for near decision boundary)
    # Simplify: use high confidence only (>0.85) + margin from other class
    # Re-read: "confidence > 0.85 OR confidence < 0.15" — confidence is max(proba)
    #   max(proba) < 0.15 is extremely rare, would only happen if both classes have proba < 0.15 (impossible)
    # Let's interpret charitably: accept high-confidence predictions (>0.85) — ignore the <0.15 as infeasible
    #   OR: use margin: |proba[1] - 0.5| > 0.35 would give >0.85 or <0.15

    # Use the margin interpretation: accept if |proba[1] - 0.5| > 0.35, so proba[1] > 0.85 or proba[1] < 0.15
    margin = np.abs(proba_unlabeled[:, 1] - 0.5)
    mask_confident = margin > 0.35

    n_pseudo = mask_confident.sum()
    print(f"  Confident predictions: {n_pseudo}/{len(X_unlabeled)} ({100*n_pseudo/len(X_unlabeled):.1f}%)")

    if n_pseudo > 0:
        # Hard labels: argmax
        pseudo_labels = np.argmax(proba_unlabeled[mask_confident], axis=1)
        X_pseudo = X_unlabeled[mask_confident]

        # Append to combined set
        X_train_combined = np.vstack([X_train_combined, X_pseudo])
        y_train_combined = np.concatenate([y_train_combined, pseudo_labels])
        print(f"  Combined set size: {len(X_train_combined)}")
    else:
        print(f"  No confident predictions; stopping iteration loop early")
        break

# Train final model on combined labeled + pseudo-labeled
print(f"\nTraining final model on {len(X_train_combined)} samples...")
final_model = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    min_child_samples=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
    class_weight="balanced"
)
final_model.fit(X_train_combined, y_train_combined)

# Predict on train (for gap calculation) and test
p_train = final_model.predict_proba(X_full_train)[:, 1]
p_test = final_model.predict_proba(X_test)[:, 1]

# Evaluate
results = evaluate_proba(p_train, p_test, y_full_train, y_test)

# Report with extra metadata
extra_info = {
    "n_labeled": int(n_labeled),
    "n_unlabeled": int(len(X_unlabeled)),
    "n_pseudo_total": int(len(X_train_combined) - len(X_labeled)),
    "confidence_threshold_hi": CONFIDENCE_HI,
    "confidence_threshold_lo": CONFIDENCE_LO,
    "margin_threshold": 0.35,
    "n_iterations": N_ITER,
    "final_combined_size": int(len(X_train_combined)),
}

report(
    "111_self_training_pseudo",
    "semi-supervised self-training with confidence-thresholded pseudo-labels; 60/40 split; 3 iterations; high-margin acceptance",
    results,
    extra=extra_info
)
