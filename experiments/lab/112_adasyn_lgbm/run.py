"""ADASYN adaptive synthetic oversampling + LightGBM classifier.

Architecture: Adaptive Synthetic Sampling (ADASYN) in a pipeline with LightGBM.
This handles class imbalance by generating MORE synthetic minority examples for
instances near the decision boundary (harder to classify), then applies tree-based
gradient boosting for classification.

Novel in the lab: ADASYN differs fundamentally from SMOTE (experiment 104) by
adapting synthetic generation density to decision boundary proximity. SMOTE generates
uniform synthetic samples; ADASYN generates MORE samples for harder (near-boundary)
minority examples. This is genuinely novel architecture.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

from shared.data import load_classifier_data
from shared.eval import evaluate_proba, report

Xt, yt, Xv, yv = load_classifier_data()

# imblearn.pipeline.Pipeline supports transformers (like ADASYN) alongside scikit-learn estimators
m = ImbPipeline([
    ("scaler", StandardScaler()),
    ("adasyn", ADASYN(random_state=42, n_neighbors=5)),
    ("lgbm", LGBMClassifier(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
        is_unbalanced=True
    ))
])

m.fit(Xt, yt)
report("112_adasyn_lgbm",
       "ADASYN adaptive oversampling + LightGBM; generates more samples for hard-to-classify minority examples near decision boundary",
       evaluate_proba(m.predict_proba(Xt)[:, 1], m.predict_proba(Xv)[:, 1], yt, yv))
