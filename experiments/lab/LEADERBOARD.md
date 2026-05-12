# ML Lab Leaderboard

Generated 2026-05-12T00:18:04.532695+00:00. Each experiment lives in `experiments/lab/<NNN_name>/run.py` and writes its own `results.json`. Same train/test split for all classifier/anomaly experiments (70/30 day-aware). Strategy experiments report net P&L at 8 bps slippage.

## Eval gate (anti-overfit)
Classifier passes if `test_auc > 0.55` AND `|train_test_gap| < 0.05`.
Strategy passes if `net_pnl_8bps > 0` AND `n_trades >= 5`.

## Classifier / Anomaly leaderboard (22 models)

| Rank | Experiment | Architecture | Test AUC | Train-test gap | Pass? |
|---:|---|---|---:|---:|:---:|
| 1 | `006_hist_gbm` | histogram-binned GBM, 300 iters, L2=0.5 | 0.866 | +0.057 | **fail** |
| 2 | `005_gradient_boosting` | sklearn GBM, 200 trees, depth-4, lr=0.05 | 0.865 | +0.039 | **PASS** |
| 3 | `025_stacking_ensemble` | Stack: LogReg + LightGBM + HistGBM, meta: LogReg on 3-fold O | 0.864 | +0.069 | **fail** |
| 4 | `007_lightgbm` | LightGBM leaf-wise, 300 trees, 31 leaves | 0.863 | +0.092 | **fail** |
| 5 | `008_xgboost` | XGBoost hist, 300 trees, depth-6, scale_pos_weight balanced | 0.858 | +0.111 | **fail** |
| 6 | `014_mlp` | MLP 64-32-16, ReLU, Adam, early stopping on 10% val | 0.855 | +0.029 | **PASS** |
| 7 | `002_random_forest` | 300 bagged trees, depth-8, sqrt(n_features) per split | 0.855 | +0.058 | **fail** |
| 8 | `004_adaboost` | 200 boosted stumps (depth-1), AdaBoost.R2 | 0.852 | +0.017 | **PASS** |
| 9 | `013_qda` | QDA — per-class covariance with 0.1 shrinkage regularization | 0.836 | +0.004 | **PASS** |
| 10 | `010_knn` | KNN k=50, Euclidean distance, scaled features | 0.834 | +0.053 | **fail** |
| 11 | `003_extra_trees` | 300 extra-random trees, random split selection per node | 0.829 | +0.032 | **PASS** |
| 12 | `011_gaussian_nb` | Gaussian NB — independent Gaussian likelihoods per feature | 0.828 | -0.000 | **PASS** |
| 13 | `009_decision_tree` | single decision tree depth-7, balanced class weight | 0.824 | +0.053 | **fail** |
| 14 | `018_elliptic_envelope` | robust Gaussian envelope, contamination=0.1 | 0.821 | -0.005 | **PASS** |
| 15 | `012_lda` | LDA — Bayes-optimal linear under Gaussian-with-shared-cov as | 0.802 | +0.004 | **PASS** |
| 16 | `001_logreg` | linear decision boundary, scaled features, balanced class we | 0.799 | +0.007 | **PASS** |
| 17 | `015_isolation_forest` | 300 isolation trees, contamination=auto | 0.793 | -0.006 | **PASS** |
| 18 | `019_autoencoder` | MLP-AE 180->90->45->90->180, trained on normal windows only | 0.776 | -0.001 | **PASS** |
| 19 | `016_local_outlier_factor` | LOF k=30 novelty mode, scaled features | 0.703 | -0.011 | **PASS** |
| 20 | `017_one_class_svm` | OCSVM RBF kernel, trained on 5k subsample, nu=0.1 | 0.689 | -0.033 | **PASS** |
| 21 | `021_arima` |  | 0.500 | +0.000 | **fail** |
| 22 | `020_hmm_regime` | 3-state Gaussian HMM with diagonal covariance | 0.408 | -0.008 | **fail** |

## Strategy leaderboard (3 variants)

| Rank | Experiment | Architecture | # Trades | Net P&L 8bps | Win rate | Pass? |
|---:|---|---|---:|---:|---:|:---:|
| 1 | `022_pairs_zscore` | SPY/QQQ z-score pairs, |z|>2 entry, |z|<0.5 exit, 60-bar loo | 147 | $-3,500 | 0.7% | **fail** |
| 2 | `024_donchian_breakout` | 30-bar Donchian breakout, opposite-channel stop, 20-bar time | 1624 | $-13,036 | 34.1% | **fail** |
| 3 | `023_bollinger_meanrev` | Bollinger 2σ mean-rev, entry at band touch, exit at SMA cros | 1939 | $-21,009 | 39.6% | **fail** |
