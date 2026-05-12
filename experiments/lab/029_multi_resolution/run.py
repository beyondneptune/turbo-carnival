"""Custom: Multi-Resolution Temporal Features — fast + medium + slow rhythms.

My own design. Conventional ML uses one fixed feature window. This computes
THREE temporal resolutions on the same bar:
  - Fast (1-3 bar): immediate momentum, micro-volatility
  - Medium (5-15 bar): trend, vwap-distance, range
  - Slow (30-60 bar): regime, recent vol of vol

Concatenates all three into a 22-dim feature vector and feeds to LightGBM.
The novelty: explicit multi-scale temporal awareness without convolutions.
Captures patterns that single-window approaches miss (e.g. fast move
against slow trend = real signal vs random noise).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
import numpy as np
import pandas as pd

from shared.data import _enrich, BARS, FORWARD_BARS, THRESHOLD_BPS
from shared.eval import evaluate_proba, report


def make_multires(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    out = pd.DataFrame(index=df.index)
    g = df.groupby("symbol")
    # FAST (1-3 bar)
    out["fast_ret1"] = df["ret1"]
    out["fast_ret3"] = df["ret3"]
    out["fast_abs_ret1"] = df["ret1"].abs()
    out["fast_range_pct"] = df["range_pct"]
    # MEDIUM (5-15 bar)
    out["med_ret5"] = g["close"].pct_change(5).fillna(0)
    out["med_ret10"] = df["ret10"]
    out["med_ret15"] = g["close"].pct_change(15).fillna(0)
    out["med_vol_5"] = g["ret1"].transform(lambda s: s.rolling(5, min_periods=3).std()).fillna(0)
    out["med_vol_15"] = g["ret1"].transform(lambda s: s.rolling(15, min_periods=5).std()).fillna(0)
    out["med_range_avg_10"] = g["range_pct"].transform(lambda s: s.rolling(10, min_periods=3).mean()).fillna(0)
    out["med_vwap_dist"] = df["vwap_dist"]
    # SLOW (30-60 bar)
    out["slow_ret30"] = g["close"].pct_change(30).fillna(0)
    out["slow_ret60"] = g["close"].pct_change(60).fillna(0)
    out["slow_vol_30"] = g["ret1"].transform(lambda s: s.rolling(30, min_periods=10).std()).fillna(0)
    out["slow_vol_60"] = g["ret1"].transform(lambda s: s.rolling(60, min_periods=20).std()).fillna(0)
    out["slow_vov"] = df["vov"]
    out["slow_range_p60"] = g["range_pct"].transform(lambda s: s.rolling(60, min_periods=20).quantile(0.6)).fillna(0)
    # Cross-resolution interaction features (the secret sauce)
    out["fast_x_slow_sign"] = np.sign(out["fast_ret1"]) * np.sign(out["slow_ret30"])
    out["fast_vs_slow_vol_ratio"] = (out["med_vol_5"] / (out["slow_vol_30"] + 1e-9)).clip(0, 10)
    out["range_now_vs_avg"] = (out["fast_range_pct"] / (out["med_range_avg_10"] + 1e-9)).clip(0, 10)
    out["tod_sin"] = df["tod_sin"]
    out["tod_cos"] = df["tod_cos"]
    return out


df = _enrich(pd.read_parquet(BARS))
df = df.dropna(subset=["y"]).reset_index(drop=True)
F = make_multires(df)
F["y"] = df["y"]
F["date"] = df["date"]
F = F.dropna().reset_index(drop=True)

feat_cols = [c for c in F.columns if c not in ("y", "date")]
dates = sorted(F["date"].unique())
n_train = int(len(dates) * 0.7)
train_dates = set(dates[:n_train])
train_mask = F["date"].isin(train_dates)
Xt, yt = F.loc[train_mask, feat_cols].values, F.loc[train_mask, "y"].values
Xv, yv = F.loc[~train_mask, feat_cols].values, F.loc[~train_mask, "y"].values

print(f"Multi-resolution features: {len(feat_cols)} columns "
      f"({sum(1 for c in feat_cols if c.startswith('fast'))} fast, "
      f"{sum(1 for c in feat_cols if c.startswith('med'))} med, "
      f"{sum(1 for c in feat_cols if c.startswith('slow'))} slow, "
      f"{sum(1 for c in feat_cols if 'x' in c)} interaction)")

m = lgb.LGBMClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                        class_weight="balanced", random_state=42, verbose=-1)
m.fit(Xt, yt)
p_tr = m.predict_proba(Xt)[:, 1]
p_te = m.predict_proba(Xv)[:, 1]
report("029_multi_resolution",
        "Custom: 22-dim multi-resolution features (fast 1-3 / med 5-15 / slow 30-60 bar) + LightGBM",
        evaluate_proba(p_tr, p_te, yt, yv),
        extra={"n_features": len(feat_cols),
               "top_feature_importances": dict(zip(feat_cols,
                                                     m.feature_importances_.astype(int).tolist()))})
