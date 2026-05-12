# ARCHITECTURE_SPEC — what counts as a "new architecture"

Used by the overnight infinite-agentic-loop to generate genuinely novel
architectures, not parameter variants. Adapted from the
`disler/infinite-agentic-loop` pattern (see `/tmp/iag/.claude/commands/infinite.md`).

## Task

Generate ML / NN / statistical / hybrid architectures for the regime
classification task: predict P(next-5-min |ret| > 30 bps) on SPY/QQQ/AAPL/
AMD/META/NVDA/TSLA 1-minute bars.

Training data is loaded by `experiments/lab/shared/data.py` —
`load_classifier_data()` returns `X_train, y_train, X_test, y_test`. 10
features per row. ~38k train, ~16k test. Same 70/30 day-aware split for
every experiment so the leaderboard is apples-to-apples.

Evaluation is by `experiments/lab/shared/eval.py` — `evaluate_proba()` and
`report()` write standardized `results.json`.

## What counts as a NEW architecture

A new architecture differs from existing ones in **at least one** of:

1. **Loss function / training objective** (log-loss vs hinge vs Huber vs exponential vs focal vs Brier-direct)
2. **Decision rule** (linear vs tree-based vs kernel vs distance vs density vs ensemble vote)
3. **Algorithm family** (boosting vs bagging vs stacking vs cascading vs gating)
4. **Feature treatment** (raw vs polynomial vs kernel-approx vs decomposed vs interaction)
5. **Calibration approach** (none vs Platt vs isotonic vs conformal vs Bayesian)
6. **Composition strategy** (single vs voting vs stacking vs gating vs cascade vs distillation)
7. **Domain technique** (regime routing, per-symbol specialist, time-of-day specialist, slippage-aware loss)

## What does NOT count as a new architecture

- Different hyperparameter values (different C, different n_estimators, different max_depth) of an existing architecture
- Different random seeds
- Different scalers on otherwise-identical pipelines
- Re-implementations of an architecture already in the lab

## Existing architectures (do NOT duplicate)

Numbered 001-103 in `experiments/lab/`. Categories already covered:
- Linear classifiers: LogReg (L1/L2), Ridge, SGD (hinge/log/Huber/Perceptron), PassiveAggressive
- Tree families: DecisionTree, RandomForest, ExtraTrees, AdaBoost, GBM (deviance, exponential), HistGBM, LightGBM (gbdt/dart/goss/rf), XGBoost (hist/dart/gblinear), Bagging
- Kernel/distance: SVM (rbf/poly/sigmoid/linear), NuSVC, kNN, RadiusNeighbors
- Probabilistic: GaussianNB, BernoulliNB, ComplementNB, MultinomialNB, LDA, QDA
- Anomaly: IsolationForest, LocalOutlierFactor, OneClassSVM, EllipticEnvelope, MCD, K-Means residual, GMM log-likelihood
- Time-series: HMM (2/3/4/5 state, full/diag/spherical), ARIMA, EWMA-deviation
- Neural net topologies: funnel, bottleneck, pyramid, tanh, logistic, identity, sgd-solver, lbfgs-solver
- Decomposition: PCA, ICA, KernelPCA, TruncatedSVD, RandomProjection, FastICA reconstruction
- Ensemble: VotingHard, VotingSoft, Stacking (LR meta, LGBM meta, passthrough)
- Custom (mine): ATHENA-Stack (disjoint features), Cascade-Reject (Viola-Jones-style), Multi-Resolution (3 time scales), per-symbol specialist, time-of-day specialist, mixture-of-experts, self-distillation, adversarial-validation, conformal calibration
- Meta-learning: focal-LGBM, monotonic LGBM, symmetric-avg, quantile-median, hard-case-boost, prior-recalibrate, rank-avg, log-odds-stack, BMA
- Interaction: PolynomialFeatures, RBFSampler, Nystroem

## Idea pool for genuinely NEW architectures

These haven't been done yet — pick from here OR invent something not listed:

### Trading-specific
- Slippage-aware loss (penalize confidence on high-vol bars)
- Order-flow-imbalance proxy + classifier
- Volume-weighted feature importance via MIDAS-style aggregation
- Realized-vs-implied gap classifier
- Microstructure noise filter + downstream model
- Trade-direction signed feature engineering

### Active / curriculum / semi-supervised
- Self-training with confidence threshold
- Co-training with disjoint feature views
- Tri-training (3 base models vote on pseudo-labels)
- Pseudo-labeling cycle
- Curriculum: easy bars first, hard bars second
- Hard negative mining

### Class imbalance handlers
- SMOTE oversampling + classifier
- ADASYN adaptive synthetic oversampling
- Tomek-link cleaning + classifier
- Edited-Nearest-Neighbors undersampling
- Cluster-centroid undersampling
- BorderlineSMOTE

### Calibration / uncertainty
- Beta-calibration (better than Platt for trees)
- Venn-Abers calibration
- Bayesian neural net via dropout MC sampling
- Quantile regression for prediction intervals
- Conformal regression for option spread mispricing

### Feature engineering
- AutoFeat polynomial feature search
- Genetic programming for feature combinations
- Mutual information feature selection + LGBM
- Boruta-style feature selection + LGBM
- Stability selection over bootstrap samples
- Principal Cofactor Analysis (rare alternative to PCA)
- Sparse coding dictionary learning

### Online / streaming
- Online gradient boosting (river)
- Streaming Naive Bayes
- Bandit-based threshold tuning

### Information-theoretic
- Mutual-information neural estimation (MINE-style)
- Information bottleneck classifier

### Bayesian
- Bayesian linear regression with conjugate prior
- Bayesian additive regression trees (BART)
- Gaussian process classifier on subsample
- Variational Bayesian GMM

### Hybrid
- LGBM + LogReg meta with attention-weighted concatenation
- Forest-of-stumps + perceptron correction
- Tree-embedding (use tree leaf indices as features) + LogReg
- Random forest leaf-index → embedding lookup → MLP

### Trading-strategy-as-classifier
- Bollinger-z-as-feature LogReg
- Donchian-position-as-feature LGBM
- VWAP-distance-binned classifier
- Time-of-day × symbol interaction LGBM

## Output requirements

Each generation MUST:
1. Create `experiments/lab/<NNN>_<short_name>/run.py`
2. Import from `experiments.lab.shared.data` and `experiments.lab.shared.eval`
3. Train on train split, predict on test, call `report(name, description, metrics_dict)`
4. Pass at least basic syntactic correctness (compile + run without exception)
5. If it fails to converge/train, still write `results.json` with `error` field
6. <NNN> must be unique — next available 3-digit number

## Failure modes to avoid

- "Same model with different hyperparameter" — REJECTED
- "Random forest with seed 7 instead of 42" — REJECTED
- "MLP with 65 neurons instead of 64" — REJECTED
- "Re-import existing model under new name" — REJECTED
- "Architecture that requires torch/tensorflow" — REJECTED (not installed)
