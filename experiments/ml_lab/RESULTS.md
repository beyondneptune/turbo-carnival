# ML Lab — first-pass results (2026-05-11)

Three experiments on cached 21-day minute bars (54,322 rows × 7 symbols)
to assess whether ML can add value beyond rule-based heuristics in the
**post-freeze** cycle.

## Eval gate (anti-overfit)

A model passes only if BOTH:
- `test_auc > 0.55` (signal exists)
- `|train_auc − test_auc| < 0.05` (not just memorising train)

## Summary

| # | Model | Test AUC | Train-test gap | Precision@top1% | Gate |
|---|---|---:|---:|---:|---|
| 01-a | Logistic regression (regime) | 0.799 | +0.006 | n/a | **PASS** |
| 01-b | HistGradientBoosting (regime) | **0.866** | +0.070 | 76.5% | **FAIL** (overfit) |
| 02 | GBM slippage (synthetic data) | r² 0.564 | r² gap 0.287 | n/a | scaffold only |
| 03 | Autoencoder anomaly | 0.775 | **−0.001** | 68.7% | **PASS** |

## Findings

### 01 — Regime classifier (target: next 5-min `|ret| > 30 bps`)

The logistic regression baseline alone hits **0.799 test AUC with essentially zero
overfit**. That's a strong, honest signal: lined-up features already explain
most of the variance.

The boosted tree squeezes another 7 points of test AUC out (0.866) but
**overfits by 7 points** — i.e. it's partly memorising the train regime.
That's the classic financial-ML trap. Recommendation: deploy the **logreg**
as the additional regime gate; revisit GBM after we have 60+ days.

Top features (permutation importance):
1. `range_pct` (today's bar high-low) — +0.077 AUC drop when shuffled
2. `ret1` — +0.056
3. `tod_cos` (time of day, cosine) — +0.043
4. `ret1_z` — +0.042

Worth noting: **`vov` (vol-of-vol) had near-zero permutation importance**.
Opus max #9's top-recommended filter doesn't actually carry information
once `range_pct` and `ret1_z` are in the feature set. That's a real
finding worth flagging — we may be double-counting risk.

### 02 — Slippage model (scaffold)

Synthetic-data run shows the pipeline works: GBM cuts MAE from 7.1 bps
(constant 8 bps baseline) to 4.0 bps. **Re-run after `slippage_logger.py`
has ≥200 real fills.**

### 03 — Autoencoder anomaly detector

180-dim windows (30 bars × 6 features) → 90 → 45 → 90 → 180 sklearn
MLPRegressor. Trained ONLY on normal bars.

**Test AUC 0.775 with literally zero overfit** — `train_auc 0.774,
test_auc 0.775`. Precision @ top-1% reconstruction error = 68.7%.

This is the **most deploy-ready** model of the three. It's unsupervised
(robust to label shift), it generalises perfectly across the 14→7 day
split, and "high reconstruction error" is an interpretable failure mode
(unusual bar). Strong candidate for a data-driven supplement to
`news_blackout` in the next cycle.

## Post-freeze deployment proposal

For the cycle starting 2026-08-10:

1. **Add logreg regime gate to `regime_filters.py`** — block entries when
   `predict_proba > 0.7`. Falls back to existing rule-based filters if the
   model fails to load (graceful degradation).
2. **Add AE anomaly gate** — block entries when reconstruction error exceeds
   the 99th percentile of training-distribution errors.
3. **Replace constant slippage** with the trained GBM (after ≥1 week of real
   fills). Use predicted slip *as a fee*, scaling position size down when
   predicted slip is high.

None of these require model retraining inside the live cycle — they're
frozen artifacts (`.joblib` pickles) loaded at host startup. No mid-cycle
adaptation.

## Reproducibility

```bash
.venv/bin/python experiments/ml_lab/01_regime_classifier.py
.venv/bin/python experiments/ml_lab/02_slippage_model.py
.venv/bin/python experiments/ml_lab/03_autoencoder_anomaly.py
```

Results written to `experiments/ml_lab/runs/*.json`.
