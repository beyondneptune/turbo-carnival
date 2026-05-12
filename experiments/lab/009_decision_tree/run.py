"""Single Decision Tree — depth-controlled, interpretable baseline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.tree import DecisionTreeClassifier
from shared.data import load_classifier_data
from shared.eval import evaluate_proba, report

Xt, yt, Xv, yv = load_classifier_data()
m = DecisionTreeClassifier(max_depth=7, class_weight="balanced", random_state=42)
m.fit(Xt, yt)
report("009_decision_tree", "single decision tree depth-7, balanced class weight",
        evaluate_proba(m.predict_proba(Xt)[:, 1], m.predict_proba(Xv)[:, 1], yt, yv))
