"""SMOTE oversampling + LightGBM classifier.

Architecture: synthetic minority oversampling (SMOTE) in a pipeline with LightGBM.
This handles class imbalance by generating synthetic minority examples during training,
then applies tree-based gradient boosting for classification.

Novel in the lab: imbalanced-learn's SMOTE is a dedicated class-imbalance handler
not yet used in experiments 001-103.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

from shared.data import load_classifier_data
from shared.eval import evaluate_proba, report

Xt, yt, Xv, yv = load_classifier_data()

# imblearn.pipeline.Pipeline supports transformers (like SMOTE) alongside scikit-learn estimators
m = ImbPipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
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
report("104_imblearn_smote_lgbm",
       "SMOTE oversampling + LightGBM; handles class imbalance via synthetic minority generation",
       evaluate_proba(m.predict_proba(Xt)[:, 1], m.predict_proba(Xv)[:, 1], yt, yv))
