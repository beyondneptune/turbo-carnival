## Specialist: MLGATE

### Core thesis (1 sentence)

Keep the existing rule-based architecture intact and bolt on the LogReg regime classifier (0.799 test AUC, +0.006 train-test gap) as the final entry filter — empirically the only intervention with a measured positive counterfactual on today's actual trades (+$1,070 bar-close delta from blocking 68.5% of entries).

### Strategy decisions

1. **trade_frequency**: 25 per lane per day (target). Floor of 10 to keep statistical evaluation alive; hard cap 40 if regime is unusually quiet. Derived directly from `163 × (1 − 0.685) ÷ 2 lanes ≈ 25`.
2. **order_type**: marketable_limit with +3 bps offset (buy at ask+3bps, sell at bid−3bps) on top-5 LogReg-confident entries; plain limit at midpoint+1 bps for the rest. Bracket parent for SL/TP. Skip market orders entirely — they're the slippage donor at 12 bps round-trip.
3. **position_sizing**: Adaptive, inverse to regime probability. `notional = $20k × (1 − 2 × p_vol)` clipped to `[$4k, $20k]`, where `p_vol = LogReg.predict_proba(volatile_next_5min)`. Hard block at `p_vol ≥ 0.5`. This converts the binary gate into a smooth risk dial and lets borderline-but-tradeable signals through at small size.
4. **entry_signals**: Keep mean_reversion + breakout from baseline, but require BOTH (a) `|ret1_z| > 1.5` AND (b) `range_pct` below 60th percentile (calmer bars). Drop the redundant `vol_of_vol` filter — permutation importance showed near-zero contribution once `range_pct` and `ret1_z` are present.
5. **exit_rules**: TP +25 bps (down from baseline so we actually book the 24.7% TP wins faster), SL −18 bps (tighter — losers go to full stop too often), time exit 8 min, **ML kill**: if `p_vol` recomputed at +2 min into the trade jumps above 0.65, exit immediately at marketable_limit.
6. **symbol_universe**: SPY, QQQ, AAPL, AMD, NVDA. **Drop TSLA and META.** TSLA + AMD were 69% of today's entries and the loss epicenter; we keep AMD because LogReg's regime probability discriminates well on it, but TSLA's range volatility makes the gate flap. Lift SPY/QQQ — they're the slippage-cheapest names.
7. **lane_split**: Both lanes run identical signal stack; differ ONLY in the LogReg threshold. Lane-1: `p_vol ≥ 0.50` blocks (the validated counterfactual). Lane-2: `p_vol ≥ 0.40` blocks (more conservative). This A/B's the threshold under live slippage in one day rather than guessing.
8. **filter_cascade** (ordered, fast-to-slow, cheap-to-expensive):
   1. `time_gate` (no first 5 min, no last 10 min)
   2. `news_blackout` (existing)
   3. `abs_return_gate` (`|ret1| < 80 bps`)
   4. `range_pct` < p60 (replaces vov)
   5. `correlation_check` (max 2 open positions per sector)
   6. **LogReg regime gate** — last filter, hard block at `p_vol ≥ threshold`
   7. AE anomaly score as a soft signal into sizing (not block) — multiply notional by `1 − 0.5 × ae_score_norm`

### Expected backtest result

- Daily entries: 163 → ~50 (25/lane × 2)
- Block rate vs today: 69%
- Predicted Δ P&L vs today on bar-close basis: **+$1,070** (replicated counterfactual) → conservatively **+$700** after symbol-universe trim eats some of the upside
- Predicted Δ P&L at 8 bps slippage: **+$200 to +$600** (slippage on 50 trades × $20k × 8 bps ≈ $800 vs $1,560 today; net positive if bar-close edge holds)
- Confidence: **med** — counterfactual is on bar-close not real fills, and the gate hasn't been tested at the 0.40 threshold

### Steelman against my own approach (1-2 sentences)

A single-day counterfactual on 162 trades is a sample of one; the LogReg's 0.799 AUC was measured on 21 days of cached bars but the +$1,070 delta could just be the gate accidentally avoiding TSLA's worst hours rather than capturing true regime structure. If tomorrow's regime distribution shifts (e.g. lower realised vol), the gate may block winning entries and we under-trade into a flat day with no positive selection effect.

### My critique of OPUSMAX9 approach (1 sentence)

Stacking a vol-of-vol filter on top of `range_pct` is double-counting the same risk dimension — today's permutation-importance result showed `vov` contributes near zero once `range_pct` and `ret1_z` are in, so OPUSMAX9 is paying complexity for no measured AUC gain.

### Files I'd write

- `risk/regime_gate.py`: Loads `experiments/ml_lab/runs/logreg_regime.joblib` at host start; exposes `should_block(bar_features) -> (bool, float)` returning `(p_vol >= threshold, p_vol)`. Falls back to rule-based gate if model fails to load (graceful degradation per RESULTS.md proposal).
- `data/strategy_specs/baseline/mlgate_lane1.json`: Spec with `regime_threshold: 0.50`, sizing formula, filter cascade ordering, symbol universe `[SPY, QQQ, AAPL, AMD, NVDA]`, marketable_limit order type with `+3 bps` offset.
- `data/strategy_specs/baseline/mlgate_lane2.json`: Identical to lane1 except `regime_threshold: 0.40` for live A/B.
- `scripts/alpaca_max_hft_host.py` (patch): Inject `regime_gate.should_block()` as the last call before order submission; record `p_vol` and `ae_score` to fills CSV for tomorrow's counterfactual analysis. Position sizing helper reads `p_vol` to scale notional per decision (3).
- `experiments/ml_lab/eval_mlgate_live.py`: Post-close script that joins tomorrow's fills against `p_vol` to recompute the counterfactual on REAL fills (not bar-close), so we can correct the bar-close bias before round 2 of the debate.
