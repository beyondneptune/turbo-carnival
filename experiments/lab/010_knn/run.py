"""K-Nearest Neighbors — instance-based, non-parametric."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from shared.data import load_classifier_data
from shared.eval import evaluate_proba, report

Xt, yt, Xv, yv = load_classifier_data()
m = Pipeline([("s", StandardScaler()), ("c", KNeighborsClassifier(n_neighbors=50, n_jobs=-1))])
m.fit(Xt, yt)
report("010_knn", "KNN k=50, Euclidean distance, scaled features",
        evaluate_proba(m.predict_proba(Xt)[:, 1], m.predict_proba(Xv)[:, 1], yt, yv))
