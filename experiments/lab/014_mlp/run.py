"""Multi-Layer Perceptron — small NN, ReLU activations, Adam optimizer."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from shared.data import load_classifier_data
from shared.eval import evaluate_proba, report

Xt, yt, Xv, yv = load_classifier_data()
m = Pipeline([("s", StandardScaler()),
              ("c", MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=300,
                                   activation="relu", solver="adam", learning_rate_init=0.001,
                                   random_state=42, early_stopping=True, validation_fraction=0.1))])
m.fit(Xt, yt)
report("014_mlp", "MLP 64-32-16, ReLU, Adam, early stopping on 10% val",
        evaluate_proba(m.predict_proba(Xt)[:, 1], m.predict_proba(Xv)[:, 1], yt, yv))
