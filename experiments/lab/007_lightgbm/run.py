"""LightGBM — leaf-wise growth, GOSS sampling, exclusive feature bundling."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
from shared.data import load_classifier_data
from shared.eval import evaluate_proba, report

Xt, yt, Xv, yv = load_classifier_data()
m = lgb.LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                        num_leaves=31, class_weight="balanced", random_state=42, n_jobs=-1,
                        verbose=-1)
m.fit(Xt, yt)
report("007_lightgbm", "LightGBM leaf-wise, 300 trees, 31 leaves",
        evaluate_proba(m.predict_proba(Xt)[:, 1], m.predict_proba(Xv)[:, 1], yt, yv))
