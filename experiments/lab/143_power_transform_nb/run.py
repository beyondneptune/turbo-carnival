"""Yeo-Johnson Power Transform + Gaussian Naive Bayes.

Architecture: Power-transforms make features more Gaussian, which improves the
GaussianNB independence + Gaussian-likelihood assumption. Differs from
011_gaussian_nb (raw features) and 052_nb_bernoulli (binarized).

Feature treatment: Yeo-Johnson transform (standardized) before probabilistic
classifier. Decision rule: Gaussian density-based conditional probability.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.preprocessing import PowerTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from shared.data import load_classifier_data
from shared.eval import evaluate_proba, report

Xt, yt, Xv, yv = load_classifier_data()
m = Pipeline([
    ('power', PowerTransformer(method='yeo-johnson', standardize=True)),
    ('nb', GaussianNB())
])
m.fit(Xt, yt)
p_train = m.predict_proba(Xt)[:, 1]
p_test = m.predict_proba(Xv)[:, 1]
report("143_power_transform_nb",
       "Yeo-Johnson power transform + Gaussian Naive Bayes; Gaussian-assumption-friendly features",
       evaluate_proba(p_train, p_test, yt, yv))
