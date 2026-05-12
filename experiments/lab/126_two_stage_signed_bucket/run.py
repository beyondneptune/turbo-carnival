"""Two-stage classifier: predict signed-return-bucket then re-bin to regime.

Stage 1: 4-class classifier predicting which forward-return bucket:
  - class 0: large-down (ret < -30 bps)
  - class 1: small-down (-30 <= ret < 0)
  - class 2: small-up (0 <= ret < 30 bps)
  - class 3: large-up (ret >= 30 bps)

Stage 2: Map the 4-class predictions to binary regime label
  - regime=1 iff abs(ret) > 30 bps (i.e., large-down OR large-up)
  - P(regime=1) = P(bucket=0) + P(bucket=3)
"""
import numpy as np
import lightgbm as lgb
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.data import _enrich, BARS, CLASSIFIER_FEATURES, FORWARD_BARS
from shared.eval import evaluate_proba, report

# Load with raw bars to get the signed forward return for bucketing
df = _enrich(pd.read_parquet(BARS))
df = df.dropna(subset=CLASSIFIER_FEATURES + ['y']).reset_index(drop=True)

# Compute signed forward return per symbol (in basis points)
df['fwd_signed_ret_bps'] = (
    df.groupby('symbol')['close']
    .pct_change(FORWARD_BARS)
    .shift(-FORWARD_BARS) * 10000
)
df = df.dropna(subset=['fwd_signed_ret_bps']).reset_index(drop=True)

# 4-class bucket: 0=large-down, 1=small-down, 2=small-up, 3=large-up
df['bucket'] = 0
df.loc[(df['fwd_signed_ret_bps'] >= -30) & (df['fwd_signed_ret_bps'] < 0), 'bucket'] = 1
df.loc[(df['fwd_signed_ret_bps'] >= 0) & (df['fwd_signed_ret_bps'] < 30), 'bucket'] = 2
df.loc[df['fwd_signed_ret_bps'] >= 30, 'bucket'] = 3

# Same 70/30 day split as shared.data
dates = sorted(df['date'].unique())
n_train = int(len(dates) * 0.7)
tr_dates = set(dates[:n_train])
tr_mask = df['date'].isin(tr_dates)
te_mask = ~tr_mask

Xt = df.loc[tr_mask, CLASSIFIER_FEATURES].values
yt = df.loc[tr_mask, 'y'].values
bucket_t = df.loc[tr_mask, 'bucket'].values

Xv = df.loc[te_mask, CLASSIFIER_FEATURES].values
yv = df.loc[te_mask, 'y'].values

# Stage 1: 4-class LightGBM classifier
m = lgb.LGBMClassifier(
    n_estimators=300,
    max_depth=6,
    class_weight='balanced',
    random_state=42,
    verbose=-1,
    objective='multiclass',
    num_class=4
)
m.fit(Xt, bucket_t)

# Get class probabilities
proba_tr = m.predict_proba(Xt)  # shape: (n_train, 4)
proba_te = m.predict_proba(Xv)  # shape: (n_test, 4)

# Stage 2: Map to regime probability
# P(regime) = P(bucket=0 | large-down) + P(bucket=3 | large-up)
p_tr = proba_tr[:, 0] + proba_tr[:, 3]
p_te = proba_te[:, 0] + proba_te[:, 3]

report(
    '126_two_stage_signed_bucket',
    'Stage 1: 4-class signed bucket (large-down/-30|small-down|small-up/30|large-up). '
    'Stage 2: P(regime) = P(large-down) + P(large-up)',
    evaluate_proba(p_tr, p_te, yt, yv)
)
