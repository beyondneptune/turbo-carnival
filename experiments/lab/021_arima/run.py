"""ARIMA(2,1,2) per symbol — classical statistical forecasting baseline.

Fits a small ARIMA per symbol on the train half, generates one-step-ahead
forecasts on the test half, and uses |forecast - actual| as an "anomaly"
score. Comparing how well ARIMA predicts vs random is the question.
"""
import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from shared.data import load_raw_bars
from shared.eval import evaluate_anomaly, report

df = load_raw_bars()
dates = sorted(df["date"].unique())
n_train = int(len(dates) * 0.7)
train_dates = set(dates[:n_train])

scores_train, ys_train = [], []
scores_test, ys_test = [], []
for sym, g in df.groupby("symbol"):
    g = g.sort_values("t").reset_index(drop=True)
    g_tr = g[g["date"].isin(train_dates)]
    g_te = g[~g["date"].isin(train_dates)]
    if len(g_tr) < 200 or len(g_te) < 50:
        continue
    try:
        # Fit on first 1000 train bars for speed
        fit = ARIMA(g_tr["ret1"].fillna(0).values[:1000], order=(2, 0, 2)).fit()
        # One-step forecasts on test
        history = list(g_tr["ret1"].fillna(0).values[-50:])
        for i, r in enumerate(g_te["ret1"].fillna(0).values):
            if i < 5:
                history.append(r); continue
            try:
                fc = ARIMA(history[-100:], order=(2, 0, 2)).fit().forecast(steps=1)[0]
                err = abs(r - fc) * 10000  # bps
            except Exception:
                err = 0
            scores_test.append(err)
            ys_test.append(int(g_te["y"].iloc[i]))
            history.append(r)
        # Resid on train (same distribution)
        resid = fit.resid[-len(scores_test):] if len(fit.resid) >= len(scores_test) else fit.resid
        ts_train = np.abs(np.array(resid)) * 10000
        for s, y in zip(ts_train, g_tr["y"].iloc[-len(ts_train):].values):
            scores_train.append(s); ys_train.append(int(y))
    except Exception:
        continue

if not scores_test:
    report("021_arima", "FAILED — ARIMA fitting failed for all symbols",
            {"error": "no symbols converged", "passes_overfit_gate": False, "test_auc": 0.5})
else:
    report("021_arima",
            "ARIMA(2,0,2) per-symbol; |forecast residual| as anomaly score",
            evaluate_anomaly(np.array(scores_train), np.array(scores_test),
                              np.array(ys_train), np.array(ys_test)))
