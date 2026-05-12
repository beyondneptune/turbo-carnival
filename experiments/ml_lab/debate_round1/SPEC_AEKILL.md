## Specialist: AEKILL

### Core thesis (1 sentence)

Deploy the autoencoder as a hard final-veto kill-switch (recon error > p99 → block) over a sharply pruned rule-based entry stack, because the AE is the only model in the lab with literally zero overfit (train_auc 0.774 vs test_auc 0.775) and unsupervised reconstruction error is the only signal in the entire 60-model review that cannot leak forward-label information.

### Strategy decisions

1. **trade_frequency**: ~25 entries/lane/day (down from 80 baseline; today's 163/lane → 25 is an 85% cut). AE flags + tightened rule gates should reject ~70% of would-be entries.
2. **order_type**: marketable_limit with +2 bps cross offset (post limit at last±2 bps in the trade direction). Pure market = today's 12 bps round-trip slippage death. Bracket attached: TP/SL as native child orders so we are not chasing exits with a second algo.
3. **position_sizing**: AE-scaled adaptive. Base notional $15k. `size = base × clip(1 − (recon_err / p95_err), 0.4, 1.0)`. Bars with elevated (but sub-veto) reconstruction error get sized down linearly; "boring normal" bars get full size. Hard cap $20k/position, $40k/symbol gross.
4. **entry_signals**:
   - Primary: mean_reversion only (ret1_z < −1.8 long, > +1.8 short on SPY/QQQ/AAPL/META; ±2.2 on AMD/NVDA/TSLA).
   - Drop breakout entirely until we have AE-vetted backtest of it — Falcon's −$1,619 says breakout into HFT slippage is broken.
   - Threshold confirmation: `range_pct` must be in (p20, p80) of trailing 30-bar distribution (kills the "weird bar" tail before AE even sees it).
5. **exit_rules**:
   - TP: 18 bps (today's TP was too far given 12 bps slip — keep TP economically alive).
   - SL: 35 bps (≈2:1 against TP; today's stop hit 105/325 = 32% of all exits, too tight or signals too noisy → AE filter should make remaining signals cleaner so we can widen modestly).
   - Time exit: 8 minutes (down from longer holds — minute-bar mean-rev signal half-life is short).
   - **ML kill criteria (the core proposal)**:
     - `ae_recon_err > p99_train` → block entry, no exception.
     - `ae_recon_err > p95_train` → halve size (already baked into sizing rule).
     - Mid-trade: if recon error on the current bar crosses p99 while position is open, flatten at market immediately (anomaly → unknown regime → exit).
     - LogReg gets ONE vote (block if `predict_proba > 0.75`), but it is advisory; AE has the final veto. Supervised models with explicit labels are suspected of leak even at 0.006 train-test gap.
6. **symbol_universe**: SPY, QQQ, AAPL, META, NVDA. **Drop TSLA and AMD entirely.** They produced 113/163 entries and the bulk of the loss; per-symbol slippage is worst on these names and the AE was trained on a pool where they dominate the anomaly tail — running them live is paying twice (real slip + AE blocking ~40% of their bars anyway).
7. **lane_split**:
   - **Lane-1 (Aegis-AE)**: mean_reversion only, SPY/QQQ/AAPL/META, AE p99 veto, LogReg advisory, 15 entries/day.
   - **Lane-2 (Sentinel-AE)**: same signals on NVDA + META only (concentrated, lower-correlated to Lane-1), AE p97 veto (more aggressive — fewer trades, higher quality), LogReg p70 hard veto, 10 entries/day.
   - Split is **a calibration A/B** of veto tightness, not strategy diversity. Two lanes give us a 2-point curve on AE threshold sensitivity by EOD.
8. **filter_cascade** (ordered, short-circuit on first reject):
   1. `time_gate` (no first 5 min, no last 10 min, no lunch dead zone 12:00-13:00 ET)
   2. `symbol_universe` membership
   3. `abs_return_gate` (skip if last 1-min |ret| > 50 bps — already in unknown territory)
   4. `range_pct percentile` (must be in p20-p80 trailing 30 bars)
   5. **`ae_recon_err > p99_train` → HARD VETO** ← the kill switch
   6. Rule-based mean_reversion signal trigger
   7. `logreg_proba < 0.75` (advisory, Lane-2 hard)
   8. `correlation_check` (no two open positions with 30-day corr > 0.85)
   9. Position sizing (AE-scaled) → submit marketable_limit + bracket

### Expected backtest result

- Daily entries: 163 → ~25 (85% cut)
- Block rate vs today: ~85% (AE alone ~10-15%, the rest is symbol drop + range_pct + time gates)
- Predicted Δ P&L vs today on bar-close basis: **+$1,400 to +$1,900** (counterfactual estimate: LogReg counterfactual gave +$1,070 on 68.5% block; AE-veto'd entries are the worst-of-worst tail, so blocking the AE-tagged ~10% on top should add another $400-800; symbol-drop of TSLA/AMD net positive given they were today's loss leaders)
- Predicted Δ P&L at 8 bps slippage: **−$200 to +$900** (25 entries × $20k × 8 bps × 2 = ~$800 frictional cost vs the bar-close edge; honestly tight at 8 bps, properly negative at 12 bps unless TP hit-rate climbs above 35%)
- Confidence: **med**. Honest read: even with the kill-switch we may still bleed at realistic slippage; the AE's value is mostly **drawdown protection on tail bars**, not edge generation.

### Steelman against my own approach (1-2 sentences)

The AE was trained on 21 days of one symbol pool and on "non-anomalous" bars defined by a forward-return threshold — so its p99 cutoff is itself a function of today's regime, and the 68.7% precision@top1% means ~31% of vetoed bars would have been winners; we are paying real opportunity cost for a model whose veto rate (~1% of all bars) is small enough that on most days it changes essentially nothing while making the system harder to debug.

### My critique of MOMENTUM_PURE approach (1 sentence)

Any pure-momentum / breakout-first specialist is fighting the empirical evidence — Falcon (breakout) lost more per-trade than Aegis (mean-rev) today AND the simulator says breakout variants are the worst-performing class at realistic slippage, so a momentum-led proposal is choosing the demonstrably broken regime.

### Files I'd write

- `signals/ae_kill_switch.py`: Load `models/ae_anomaly.joblib` + `models/ae_scaler.joblib` at host startup; expose `is_anomalous(window_30x6) -> (block: bool, recon_err: float, size_mult: float)` returning the three-way decision (veto / size-down / pass) from p99 / p95 cutoffs persisted in the joblib metadata; graceful no-op fallback if the model fails to load.
- `data/strategy_specs/baseline/aegis_ae_lane1.json`: Spec with `ae_kill: {enabled: true, veto_percentile: 99, size_down_percentile: 95}`, symbol whitelist `[SPY, QQQ, AAPL, META]`, mean_reversion thresholds, marketable_limit order config with +2 bps cross.
- `data/strategy_specs/baseline/sentinel_ae_lane2.json`: Same shape, `veto_percentile: 97`, symbol whitelist `[NVDA, META]`, logreg hard veto enabled.
- `scripts/calibrate_ae_thresholds.py`: One-shot pre-open script — load training-set reconstruction errors from `runs/autoencoder_anomaly_results.json`, compute p95/p97/p99, write into the joblib metadata. Runs once before market open so live host has no IO dependency on the lab.
- `experiments/ml_lab/debate_round1/SPEC_AEKILL.md`: this file.
