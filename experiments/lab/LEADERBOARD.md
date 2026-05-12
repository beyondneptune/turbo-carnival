# ML Lab Leaderboard

Generated 2026-05-12T01:13:20.896763+00:00. Each experiment lives in `experiments/lab/<NNN_name>/run.py` and writes its own `results.json`. Same train/test split for all classifier/anomaly experiments (70/30 day-aware). Strategy experiments report net P&L at 8 bps slippage.

## Eval gate (anti-overfit)
Classifier passes if `test_auc > 0.55` AND `|train_test_gap| < 0.05`.
Strategy passes if `net_pnl_8bps > 0` AND `n_trades >= 5`.

## Classifier / Anomaly leaderboard (108 models)

| Rank | Experiment | Architecture | Test AUC | Train-test gap | Pass? |
|---:|---|---|---:|---:|:---:|
| 1 | `029_multi_resolution` | Custom: 22-dim multi-resolution features (fast 1-3 / med 5-1 | 0.873 | +0.109 | **fail** |
| 2 | `047_gbm_exponential` | Sklearn GBM with exponential loss (AdaBoost-style) | 0.868 | +0.034 | **PASS** |
| 3 | `094_meta_quantile_median` | Median across 5 LGBMs with different seeds | 0.868 | +0.078 | **fail** |
| 4 | `106_shadow_feature_selection_lgbm` | Shadow feature stability selection + LightGBM; selects featu | 0.868 | +0.079 | **fail** |
| 5 | `093_meta_symmetric_avg` | Average prob from model trained on y and model on 1-y | 0.866 | +0.096 | **fail** |
| 6 | `006_hist_gbm` | histogram-binned GBM, 300 iters, L2=0.5 | 0.866 | +0.057 | **fail** |
| 7 | `091_meta_focal_lgbm` | LGBM with high L1 regularization — sparse boosting | 0.865 | +0.107 | **fail** |
| 8 | `005_gradient_boosting` | sklearn GBM, 200 trees, depth-4, lr=0.05 | 0.865 | +0.039 | **PASS** |
| 9 | `077_stack_meta_logreg` | Stack: 3 bases → LogReg meta-learner | 0.865 | +0.088 | **fail** |
| 10 | `097_meta_rank_avg` | Rank-averaged 5-seed LGBM ensemble | 0.864 | +0.094 | **fail** |
| 11 | `100_meta_bma` | Bayesian model averaging via NLL-weighted posterior | 0.864 | +0.094 | **fail** |
| 12 | `025_stacking_ensemble` | Stack: LogReg + LightGBM + HistGBM, meta: LogReg on 3-fold O | 0.864 | +0.069 | **fail** |
| 13 | `079_stack_passthrough` | Stack with passthrough — meta sees both predictions AND orig | 0.864 | +0.087 | **fail** |
| 14 | `085_custom_conformal_band` | Conformal-prediction calibrated LGBM with q=0.9 band | 0.864 | +0.098 | **fail** |
| 15 | `082_custom_adversarial_validation` | Adversarial-validation-weighted training (sample weighting b | 0.863 | +0.105 | **fail** |
| 16 | `042_lgbm_dart` | LightGBM DART — dropout-meets-additive-regression-trees | 0.863 | +0.052 | **fail** |
| 17 | `007_lightgbm` | LightGBM leaf-wise, 300 trees, 31 leaves | 0.863 | +0.092 | **fail** |
| 18 | `049_bagging_decision_tree` | Bagging with deep decision tree base | 0.863 | +0.137 | **fail** |
| 19 | `111_self_training_pseudo` | semi-supervised self-training with confidence-thresholded ps | 0.863 | +0.059 | **fail** |
| 20 | `096_meta_prior_recalibrate` | LGBM with class-prior recalibration of probabilities | 0.863 | +0.116 | **fail** |
| 21 | `098_meta_youden_threshold` | LGBM with Youden's J optimal threshold (proxy) | 0.863 | +0.116 | **fail** |
| 22 | `043_lgbm_goss` | LightGBM GOSS — gradient-based one-side sampling | 0.862 | +0.086 | **fail** |
| 23 | `092_meta_monotonic_lgbm` | LGBM with monotonic constraints on price-momentum features | 0.861 | +0.108 | **fail** |
| 24 | `084_custom_mixture_of_experts` | Gated mixture: linear expert + tree expert + LGBM gate | 0.861 | +0.086 | **fail** |
| 25 | `078_stack_meta_lgbm` | Stack: 3 bases → LightGBM meta-learner (nonlinear stacking) | 0.860 | +0.091 | **fail** |
| 26 | `109_curriculum_easy_first` | Curriculum learning: 3-stage easy-first training on LightGBM | 0.860 | +0.111 | **fail** |
| 27 | `067_nn_funnel` | MLP funnel topology 128→64→32→16 | 0.860 | +0.049 | **PASS** |
| 28 | `076_voting_soft_arith` | Soft-vote ensemble: arithmetic mean of probabilities | 0.859 | +0.116 | **fail** |
| 29 | `099_meta_log_odds_stack` | Log-odds stacking of LGBM and LogReg | 0.859 | +0.080 | **fail** |
| 30 | `107_bart_style_additive` | BART-style additive ensemble: 50 shallow regression trees (d | 0.859 | +0.024 | **PASS** |
| 31 | `104_imblearn_smote_lgbm` | SMOTE oversampling + LightGBM; handles class imbalance via s | 0.859 | +0.065 | **fail** |
| 32 | `008_xgboost` | XGBoost hist, 300 trees, depth-6, scale_pos_weight balanced | 0.858 | +0.111 | **fail** |
| 33 | `080_custom_per_symbol_ensemble` | Per-symbol specialist LightGBMs, predictions concatenated | 0.857 | +0.142 | **fail** |
| 34 | `014_mlp` | MLP 64-32-16, ReLU, Adam, early stopping on 10% val | 0.855 | +0.029 | **PASS** |
| 35 | `002_random_forest` | 300 bagged trees, depth-8, sqrt(n_features) per split | 0.855 | +0.058 | **fail** |
| 36 | `070_nn_tanh` | MLP with tanh activation — different non-linearity family | 0.852 | +0.014 | **PASS** |
| 37 | `004_adaboost` | 200 boosted stumps (depth-1), AdaBoost.R2 | 0.852 | +0.017 | **PASS** |
| 38 | `026_nn_scaling_curve` | Single hidden layer MLP swept across sizes 2-1024 | 0.850 | +0.012 | **PASS** |
| 39 | `069_nn_pyramid` | MLP inverse pyramid 16→32→64→128 | 0.850 | +0.034 | **PASS** |
| 40 | `044_lgbm_rf` | LightGBM in random-forest mode (no boosting) | 0.849 | +0.027 | **PASS** |
| 41 | `045_xgb_dart` | XGBoost DART — dropout regularization on boosting | 0.841 | +0.153 | **fail** |
| 42 | `068_nn_bottleneck` | MLP bottleneck 64→8→64 (autoencoder-like inside classifier) | 0.838 | +0.010 | **PASS** |
| 43 | `103_nystroem_logreg` | Nystroem method kernel approximation + LogReg | 0.837 | +0.010 | **PASS** |
| 44 | `013_qda` | QDA — per-class covariance with 0.1 shrinkage regularization | 0.836 | +0.004 | **PASS** |
| 45 | `027_athena_stack` | Custom: 3 bases on disjoint features (price/volume/time), ge | 0.835 | +0.018 | **PASS** |
| 46 | `010_knn` | KNN k=50, Euclidean distance, scaled features | 0.834 | +0.053 | **fail** |
| 47 | `105_tree_embedding_logreg` | RandomForest leaf-index embeddings (50 trees, depth=4) -> on | 0.834 | +0.071 | **fail** |
| 48 | `081_custom_time_of_day_specialist` | 4 time-of-day specialists (open/morning/midday/close) | 0.834 | +0.164 | **fail** |
| 49 | `028_cascade_reject` | Custom: 3-stage cascade (NB→LR→LGBM), each trained on prior- | 0.833 | +0.056 | **fail** |
| 50 | `003_extra_trees` | 300 extra-random trees, random split selection per node | 0.829 | +0.032 | **PASS** |
| 51 | `011_gaussian_nb` | Gaussian NB — independent Gaussian likelihoods per feature | 0.828 | -0.000 | **PASS** |
| 52 | `009_decision_tree` | single decision tree depth-7, balanced class weight | 0.824 | +0.053 | **fail** |
| 53 | `018_elliptic_envelope` | robust Gaussian envelope, contamination=0.1 | 0.821 | -0.005 | **PASS** |
| 54 | `090_random_projection_lgbm` | Gaussian random projection (5 dims) → LightGBM | 0.819 | +0.101 | **fail** |
| 55 | `087_ica_then_lgbm` | ICA-5 → LightGBM (independent components → tree) | 0.818 | +0.106 | **fail** |
| 56 | `086_pca_then_lgbm` | PCA-5 → LightGBM (linear features → tree) | 0.818 | +0.103 | **fail** |
| 57 | `074_nn_lbfgs_solver` | MLP trained with L-BFGS solver — batch optimization | 0.817 | +0.127 | **fail** |
| 58 | `089_kernel_pca_then_logreg` | Kernel PCA with RBF (8 components) → LogReg | 0.813 | +0.001 | **PASS** |
| 59 | `101_interaction_poly_logreg` | Pairwise polynomial interactions + LogReg | 0.808 | +0.014 | **PASS** |
| 60 | `108_cotraining_disjoint_views` | Co-training with disjoint feature views (price vs volume+tim | 0.807 | +0.101 | **fail** |
| 61 | `030_feature_count_scaling` | LogReg with feature subset size 1→10 added in importance-ran | 0.804 | -0.006 | **PASS** |
| 62 | `012_lda` | LDA — Bayes-optimal linear under Gaussian-with-shared-cov as | 0.802 | +0.004 | **PASS** |
| 63 | `054_lda_shrinkage` | LDA with Ledoit-Wolf shrinkage estimator | 0.802 | +0.004 | **PASS** |
| 64 | `031_logreg_l1` | L1-penalized logistic regression — sparse linear | 0.799 | +0.007 | **PASS** |
| 65 | `046_xgb_gblinear` | XGBoost with linear base learner — boosted linear model | 0.799 | +0.007 | **PASS** |
| 66 | `001_logreg` | linear decision boundary, scaled features, balanced class we | 0.799 | +0.007 | **PASS** |
| 67 | `041_linear_svc` | LinearSVC — efficient linear SVM (separate from sgd_hinge) | 0.796 | +0.007 | **PASS** |
| 68 | `015_isolation_forest` | 300 isolation trees, contamination=auto | 0.793 | -0.006 | **PASS** |
| 69 | `072_nn_identity` | MLP with identity activation — equivalent to linear stacking | 0.788 | +0.007 | **PASS** |
| 70 | `071_nn_logistic` | MLP with logistic activation — classic NN approach | 0.785 | +0.006 | **PASS** |
| 71 | `088_truncated_svd_logreg` | Truncated SVD-5 → Logistic Regression | 0.782 | -0.002 | **PASS** |
| 72 | `032_ridge_clf` | Ridge classifier — squared loss, not log loss | 0.782 | +0.005 | **PASS** |
| 73 | `019_autoencoder` | MLP-AE 180->90->45->90->180, trained on normal windows only | 0.776 | -0.001 | **PASS** |
| 74 | `051_nb_bernoulli` | Bernoulli NB on binarized features | 0.765 | +0.011 | **PASS** |
| 75 | `048_extra_tree_single` | Single ExtraTree — single tree with random splits | 0.765 | +0.059 | **fail** |
| 76 | `033_sgd_hinge` | SGD with hinge loss — linear SVM via SGD | 0.764 | +0.004 | **PASS** |
| 77 | `052_nb_complement` | Complement NB — Multinomial variant for imbalanced data | 0.755 | -0.015 | **PASS** |
| 78 | `053_nb_multinomial` | Multinomial NB on min-max scaled features | 0.755 | -0.015 | **PASS** |
| 79 | `075_voting_hard` | Hard-vote ensemble: LogReg + LGBM + XGB — majority class | 0.754 | +0.135 | **fail** |
| 80 | `073_nn_sgd_solver` | MLP trained with SGD solver instead of Adam | 0.752 | +0.001 | **PASS** |
| 81 | `102_rbf_sampler_logreg` | RBF kernel approximation (64 features) + LogReg | 0.739 | +0.025 | **PASS** |
| 82 | `021_arima` | ARIMA(2,0,2) per-symbol; |forecast residual| as anomaly scor | 0.733 | -0.202 | **fail** |
| 83 | `055_calibrated_isotonic` | Calibrated linear SVM with isotonic calibration | 0.721 | +0.009 | **PASS** |
| 84 | `050_bagging_logreg` | Bagging with logistic regression base — pasted linear ensemb | 0.713 | -0.001 | **PASS** |
| 85 | `056_calibrated_sigmoid` | Calibrated SGD with Platt-sigmoid calibration | 0.706 | -0.006 | **PASS** |
| 86 | `016_local_outlier_factor` | LOF k=30 novelty mode, scaled features | 0.703 | -0.011 | **PASS** |
| 87 | `034_sgd_modified_huber` | SGD with modified Huber — robust to outliers | 0.692 | -0.004 | **PASS** |
| 88 | `017_one_class_svm` | OCSVM RBF kernel, trained on 5k subsample, nu=0.1 | 0.689 | -0.033 | **PASS** |
| 89 | `039_svm_sigmoid` | SVM with sigmoid kernel on 5k subsample | 0.679 | +0.020 | **PASS** |
| 90 | `036_passive_aggressive` | Passive-Aggressive online learner | 0.657 | -0.009 | **PASS** |
| 91 | `110_quantile_regression_meta` | LightGBM quantile regressors (Q25/Q50/Q75) on binary targets | 0.641 | -0.003 | **PASS** |
| 92 | `038_svm_poly` | SVM with polynomial kernel (degree 3) on 5k subsample | 0.630 | -0.019 | **PASS** |
| 93 | `035_perceptron` | Perceptron — classic linear, no probability calibration | 0.624 | -0.006 | **PASS** |
| 94 | `037_svm_rbf` | SVM with RBF kernel on 5k subsample | 0.605 | +0.039 | **PASS** |
| 95 | `040_nu_svc` | Nu-SVC — alternative SVM parameterization with bound on erro | 0.500 | +0.000 | **fail** |
| 96 | `057_anomaly_kmeans_dist` | K-means: distance to nearest of 5 centroids as anomaly score | 0.500 | +0.000 | **fail** |
| 97 | `058_anomaly_gmm_loglik` | GMM 3-component: negative log-likelihood as anomaly score | 0.500 | +0.000 | **fail** |
| 98 | `059_anomaly_pca_recon` | PCA reconstruction error from 5 components | 0.500 | +0.000 | **fail** |
| 99 | `060_anomaly_ica_recon` | Independent Component Analysis reconstruction error | 0.500 | +0.000 | **fail** |
| 100 | `061_anomaly_mahalanobis` | Empirical-covariance Mahalanobis distance | 0.500 | +0.000 | **fail** |
| 101 | `062_anomaly_mcd_robust` | Robust Mahalanobis via Minimum Covariance Determinant | 0.500 | +0.000 | **fail** |
| 102 | `063_hmm_2state_full` | 2-state HMM with full covariance — regime switching | 0.500 | +0.000 | **fail** |
| 103 | `064_hmm_4state_diag` | 4-state HMM with diagonal covariance | 0.500 | +0.000 | **fail** |
| 104 | `065_hmm_5state_spherical` | 5-state HMM with spherical covariance | 0.500 | +0.000 | **fail** |
| 105 | `066_ewma_deviation` | EWMA span-30 deviation as anomaly score | 0.500 | +0.000 | **fail** |
| 106 | `083_custom_self_distillation` | Self-distillation: teacher LGBM → soft labels → GB regressor | 0.500 | +0.000 | **fail** |
| 107 | `020_hmm_regime` | 3-state Gaussian HMM with diagonal covariance | 0.408 | -0.008 | **fail** |
| 108 | `095_meta_hard_case_boost` | Second model specialized on uncertain (0.3<p<0.7) cases | 0.182 | +0.109 | **fail** |

## Strategy leaderboard (3 variants)

| Rank | Experiment | Architecture | # Trades | Net P&L 8bps | Win rate | Pass? |
|---:|---|---|---:|---:|---:|:---:|
| 1 | `022_pairs_zscore` | SPY/QQQ z-score pairs, |z|>2 entry, |z|<0.5 exit, 60-bar loo | 147 | $-3,500 | 0.7% | **fail** |
| 2 | `024_donchian_breakout` | 30-bar Donchian breakout, opposite-channel stop, 20-bar time | 1624 | $-13,036 | 34.1% | **fail** |
| 3 | `023_bollinger_meanrev` | Bollinger 2σ mean-rev, entry at band touch, exit at SMA cros | 1939 | $-21,009 | 39.6% | **fail** |
