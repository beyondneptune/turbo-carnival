"""BorderlineSMOTE + LightGBM — boundary-aware oversampling + gradient boosting.

Architecture: BorderlineSMOTE identifies minority points near the decision boundary
(those whose K nearest neighbors are mostly from the OTHER class), then oversamples
only those points. This is more selective than uniform SMOTE and less density-dependent
than ADASYN. Then LightGBM (gradient boosting) trains on the resampled data.

Key novelty: Combines boundary-aware synthetic oversampling (class imbalance handling)
with gradient boosting decision rules. Different from:
- 104 (SMOTE uniform) — BorderlineSMOTE is selective to boundary points
- 112 (ADASYN density-adaptive) — BorderlineSMOTE uses K-neighbor composition
- 007 (LightGBM alone) — This one resamples first
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imblearn.over_sampling import BorderlineSMOTE
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from shared.data import load_classifier_data
from shared.eval import evaluate_proba, report

# Load data
Xt, yt, Xv, yv = load_classifier_data()

# Step 1: Scale training features
scaler = StandardScaler()
Xt_scaled = scaler.fit_transform(Xt)
Xv_scaled = scaler.transform(Xv)

# Step 2: Apply BorderlineSMOTE to training data only
# kind='borderline-1': use the more conservative variant (only those K neighbors whose majority is other class)
# k_neighbors=5: standard neighborhood size for imbalanced-learn
bs = BorderlineSMOTE(random_state=42, kind='borderline-1', k_neighbors=5)
Xt_resampled, yt_resampled = bs.fit_resample(Xt_scaled, yt)

# Step 3: Train LightGBM on resampled data
# Use num_leaves and learning_rate typical for this dataset size
lgbm_model = lgb.LGBMClassifier(
    n_estimators=200,
    num_leaves=31,
    learning_rate=0.05,
    random_state=42,
    verbose=-1,
    is_unbalanced=False,  # False because we already resampled
)
lgbm_model.fit(Xt_resampled, yt_resampled)

# Step 4: Predict on original (non-resampled) test/train
p_train = lgbm_model.predict_proba(Xt_scaled)[:, 1]
p_test = lgbm_model.predict_proba(Xv_scaled)[:, 1]

# Step 5: Evaluate
metrics = evaluate_proba(p_train, p_test, yt, yv)
report(
    "117_borderline_smote_lgbm",
    "BorderlineSMOTE (boundary-aware oversampling) + LightGBM (gradient boosting); "
    "selectively oversamples minority points near decision boundary, then trains GBDT",
    metrics
)
