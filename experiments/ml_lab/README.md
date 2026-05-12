# ML Lab — research only, not for the live cycle

These notebooks/scripts produce **proposals** for the next 90-day cycle.
They DO NOT touch `scripts/alpaca_max_hft_host.py` or any spec in
`data/strategy_specs/baseline/` while the freeze (2026-05-11 → 2026-08-09)
is active.

## Why we're doing this

The 60-model swarm (NIM × 20 + Claude × 30 + Opus × 10) verdict: 0/60
voted to deploy real money on rule-based heuristics. ML can help in two
narrow places:

1. **Regime classification** — "is the next 5 min volatile?" used as an
   *additional gate* on top of existing rule filters. Not a directional
   prediction.
2. **Slippage modeling** — replace the flat 8 bps default with a per-ticker
   per-hour distribution, so the simulator's verdict generalises beyond the
   single regime we observed.
3. **Anomaly detection** — flag bars where the autoencoder reconstruction
   error is high, as a data-driven supplement to `news_blackout`.

## Why NOT deep nets

Dataset = 54,322 minute bars × 7 symbols × 21 trading days. That's
plenty for gradient-boosted trees and small MLPs. Anything LSTM- or
transformer-sized will overfit and report fake AUCs.

## Files

| File | Goal | Status |
|---|---|---|
| `01_regime_classifier.py` | HistGradientBoosting + LR baseline; target = next-5-min `|ret| > 30 bps` | runnable now |
| `02_slippage_model.py` | GBM on `slippage_log.parquet` once we have ≥1 week of live fills | scaffold + sample data fallback |
| `03_autoencoder_anomaly.py` | sklearn MLPRegressor autoencoder on 30-bar windows | runnable now |

## Eval protocol (all three)

- Walk-forward: train days 1–14 (2026-04-13 → 2026-05-01), test days 15–21
- Report: AUC, calibration (Brier score), precision @ k=top-1% riskiest bars
- Anti-overfit gate: a model passes only if `test_auc > 0.55` AND
  `test_auc > train_auc - 0.05` (no more than 5 pts of train-test gap)

## Going live (post-freeze checklist)

For any ML model to enter the next cycle's `alpaca_max_hft_host.py`:

1. ≥30 days of held-out walk-forward eval, not 7
2. Brier score < 0.20
3. Permutation feature importance not dominated by a single proxy feature
4. A "what if it's wrong?" failure mode that's bounded (regime gate fails
   open → falls back to rule-based; slippage model fails high → trades
   smaller, not nothing)
