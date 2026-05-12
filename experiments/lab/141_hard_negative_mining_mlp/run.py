"""Hard-negative mining MLP — two-stage training with model-driven data curation.

Architecture: Stage 1 trains initial MLP on full training set. Stage 2 identifies
hard negatives (those misclassified as positive with high confidence) and trains
a second MLP on a curated dataset: all positives + top 30% hardest negatives +
random 20% of remaining negatives. This focuses model capacity on the decision
boundary rather than easy negatives.

This is genuinely novel — uses model feedback to select training examples,
different from SMOTE/Tomek/ENN which manipulate dataset before training.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from shared.data import load_classifier_data
from shared.eval import evaluate_proba, report

# Load data
Xt, yt, Xv, yv = load_classifier_data()

# Stage 1: Train initial MLP on full training set
m1 = Pipeline([
    ("s", StandardScaler()),
    ("c", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200,
                        random_state=42, early_stopping=True))
])
m1.fit(Xt, yt)

# Get predictions from stage 1 on training data
p1_train = m1.predict_proba(Xt)[:, 1]

# Stage 2: Identify hard negatives
neg_mask = yt == 0
pos_mask = yt == 1
neg_p = p1_train[neg_mask]
neg_idx = np.where(neg_mask)[0]

# Top 30% hardest negatives = those with highest predicted-positive probability
hard_thresh = np.quantile(neg_p, 0.7)
hard_neg_idx = neg_idx[neg_p >= hard_thresh]

# Random 20% of remaining negatives
remaining_neg_idx = neg_idx[neg_p < hard_thresh]
rng = np.random.default_rng(42)
random_neg_idx = rng.choice(remaining_neg_idx,
                             size=int(0.2 * len(remaining_neg_idx)),
                             replace=False)

# Curated set: all positives + hard negatives + random negatives
keep_idx = np.concatenate([np.where(pos_mask)[0], hard_neg_idx, random_neg_idx])
print(f"Curated training set: {len(keep_idx)} samples "
      f"({pos_mask.sum()} pos + {len(hard_neg_idx)} hard neg + {len(random_neg_idx)} random neg)")

# Stage 2: Train second MLP on curated data
m2 = Pipeline([
    ("s", StandardScaler()),
    ("c", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200,
                        random_state=123, early_stopping=True))
])
m2.fit(Xt[keep_idx], yt[keep_idx])

# Final predictions from stage 2
p_train = m2.predict_proba(Xt)[:, 1]
p_test = m2.predict_proba(Xv)[:, 1]

# Report results
report(
    "141_hard_negative_mining_mlp",
    "Two-stage MLP: initial training + hard-negative mining + retraining on curated set",
    evaluate_proba(p_train, p_test, yt, yv),
    extra={
        "curated_set_size": len(keep_idx),
        "positives": int(pos_mask.sum()),
        "hard_negatives": int(len(hard_neg_idx)),
        "random_negatives": int(len(random_neg_idx)),
        "reduction_ratio": len(keep_idx) / len(Xt),
    }
)
