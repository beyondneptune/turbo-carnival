"""LDA-projected features + LGBM. Architecture: supervised dimensionality reduction as feature augmentation.

LDA projects the 10-dim feature space into a 1-dim decision-aware space (binary classification),
then concatenates this projection with the original features (11-dim total).
LGBM is trained on the augmented feature set.

Differs from:
- 012_lda: LDA used as the final classifier
- 089_kernel_pca_then_logreg: unsupervised projection
- 086_pca_then_lgbm: unsupervised projection
This is supervised projection (LDA) as an auxiliary feature.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import lightgbm as lgb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from shared.data import load_classifier_data
from shared.eval import evaluate_proba, report

# Load data
Xt, yt, Xv, yv = load_classifier_data()

# Standardize features
sc = StandardScaler().fit(Xt)
Xt_scaled = sc.transform(Xt)
Xv_scaled = sc.transform(Xv)

# Fit LDA (supervised projection to 1-dim for binary classification)
lda = LinearDiscriminantAnalysis(n_components=1)
lda.fit(Xt_scaled, yt)

# Project features to LDA space
lda_train = lda.transform(Xt_scaled)  # shape: (n_train, 1)
lda_test = lda.transform(Xv_scaled)   # shape: (n_test, 1)

# Augment original features with LDA projection
Xt_augmented = np.hstack([Xt_scaled, lda_train])  # (n_train, 11)
Xv_augmented = np.hstack([Xv_scaled, lda_test])   # (n_test, 11)

# Train LGBM on augmented features
model = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=5,
    class_weight='balanced',
    random_state=42,
    verbose=-1
)
model.fit(Xt_augmented, yt)

# Generate predictions
p_train = model.predict_proba(Xt_augmented)[:, 1]
p_test = model.predict_proba(Xv_augmented)[:, 1]

# Evaluate and report
metrics = evaluate_proba(p_train, p_test, yt, yv)

# Extract LDA coefficients for the discriminant
lda_coef = lda.coef_[0]  # shape: (10,)

# Add extra info: LDA coefficients and augmented feature dimension
extra = {
    "lda_coefficients": lda_coef.tolist(),
    "lda_explained_variance_ratio": float(lda.explained_variance_ratio_[0]),
    "augmented_feature_dim": 11,
    "original_feature_dim": 10,
}

report(
    "140_lda_features_lgbm",
    "LDA (supervised 1-dim projection) concatenated with original features; LGBM on 11-dim augmented set",
    metrics,
    extra=extra
)
