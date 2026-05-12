## Specialist: REGIME

### Core thesis (1 sentence)

Don't gate trades — *route* them: use today's LogReg (0.799 AUC) as a 3-way regime classifier that dispatches each bar to the sub-strategy actually suited to it (momentum/breakout in TRENDING, mean-reversion in CHOPPY, no-trade in VOLATILE), so the 24.7% TP rate rises because we stop firing mean-reversion into trends and breakouts into chop.

### Strategy decisions

1. **trade_frequency**: ~40 entries per lane per day (80 total). Lower than baseline 80 because each regime emits signals only when its native conditions hold; higher than MLGATE's 25 because we *trade* CHOPPY/TRENDING bars instead of blocking them. Hard cap 55/lane to avoid regime-thrashing days.
2. **order_type**: marketable_limit with regime-dependent offset — TRENDING: +4 bps (chasing momentum tolerates slip), CHOPPY: +1 bps and willing to time out unfilled (mean-rev requires good entry), VOLATILE: no orders. All wrapped in bracket parents (OCO TP/SL legs). No market orders ever.
3. **position_sizing**: Regime-confidence-weighted. `notional = $20k × confidence × regime_kelly`, where `confidence = max(p_trend, p_chop) − 0.33` (margin above uniform prior), `regime_kelly = {trend: 1.0, chop: 0.7, volatile: 0.0}`. Clipped to `[$5k, $25k]`. Rebalance allocations every 30 min between active sub-strategies based on rolling 30-min P&L (capital allocator).
4. **entry_signals** (per regime):
   - **TRENDING** (`p_trend > 0.5`, autocorr(ret1, 5) > +0.1, range_pct in p40-p80): `momentum` mode with `ret5_z > 1.2` AND price > VWAP; `breakout` mode with `high20 ∧ range_pct > p60`. Long-only when autocorr > +0.15.
   - **CHOPPY** (`p_chop > 0.5`, autocorr(ret1, 5) < −0.05, range_pct < p50): `mean_reversion` mode with `|ret1_z| > 1.8` AND distance-from-VWAP > 1σ. Mean-revert both directions.
   - **VOLATILE** (`p_vol > 0.7` OR realized 5-min vol > 99th pct of trailing 30d): zero entries. Close existing positions if held > 3 min.
5. **exit_rules**: Regime-specific.
   - TRENDING: TP +35 bps (let winners run), SL −22 bps, time exit 12 min, trail-stop activates after +20 bps.
   - CHOPPY: TP +18 bps (small, fast), SL −14 bps, time exit 5 min (mean-rev should snap fast).
   - **ML kill (all regimes)**: if regime label flips during the trade OR `p_vol` recomputed crosses 0.7, exit immediately at marketable_limit.
6. **symbol_universe**: SPY, QQQ, NVDA, AMD, AAPL. Drop TSLA (range vol makes regime label flap intra-bar; 60 entries today = biggest loss generator) and META (insufficient signal density at 19 entries to justify the per-symbol regime calibration). Per-symbol regime priors loaded from the 21-day cache.
7. **lane_split**: Strategy-split, not parameter-split.
   - **Lane-1 (TREND-LANE)**: Trades only in TRENDING regime. Momentum + breakout signals. Higher avg notional ($15k), lower frequency (~25/day).
   - **Lane-2 (REV-LANE)**: Trades only in CHOPPY regime. Mean-reversion only. Higher frequency (~55/day) at smaller notional ($10k).
   - Capital allocator script reallocates idle lane capital to the active lane every 30 min if one regime dominates the morning.
8. **filter_cascade** (ordered, cheap to expensive):
   1. `time_gate` (no first 5 min, no last 10 min, no FOMC/CPI windows)
   2. `news_blackout` (existing)
   3. `abs_return_gate` (`|ret1| < 100 bps` — kill obvious noise)
   4. **LogReg regime classifier** — produces (p_trend, p_chop, p_vol); routes to sub-strategy OR blocks
   5. Regime-specific signal mode evaluation (momentum/breakout OR mean_rev)
   6. `correlation_check` (max 2 open positions correlated > 0.7)
   7. **AE anomaly score** as soft veto: if `ae_score > 0.95` in TRENDING regime, downgrade to CHOPPY rules (anomalous trends are usually reversals)
   8. Per-symbol last-trade cooldown (90 sec)

### Expected backtest result

- Daily entries: 163 → ~80 (40/lane × 2)
- Block rate vs today: ~51% (lower than MLGATE's 69% — we route rather than gate)
- Predicted Δ P&L vs today on bar-close basis: **+$1,400** (assumes regime-suited signals lift TP rate from 24.7% → ~38%; +$1,070 LogReg counterfactual is a lower bound because it's a gate, not a router)
- Predicted Δ P&L at 8 bps slippage: **+$300 to +$900** (slippage cost = 80 trades × $13k avg × 8 bps ≈ $830, offset by higher per-trade edge from regime-matched signals)
- Confidence: **med** — routing logic is well-motivated by the LogReg AUC but autocorrelation thresholds are tuned on 21-day cache, not OOS validated; regime-flip ML kill is unproven.

### Steelman against my own approach (1-2 sentences)

Routing assumes the LogReg's regime probability is calibrated, not just discriminative — a 0.799 AUC says the model ranks bars correctly but doesn't guarantee `p_trend = 0.6` actually means 60% trending; if calibration is off, we route the wrong bars to the wrong sub-strategy and compound the error rather than just over-blocking like MLGATE. Three sub-strategies also multiply the surface area for parameter overfitting versus MLGATE's single rule-set + gate.

### My critique of MLGATE approach (1 sentence)

MLGATE treats the LogReg as a binary kill switch and throws away 68.5% of entries — but the model has 3-class regime structure encoded in it (the "volatile" probability is the dangerous one), and using only `p_vol >= 0.5` discards the actionable trend/chop signal that lets us *trade the remaining bars better* rather than just trade fewer of them.

### Files I'd write

- `risk/regime_router.py`: Loads `experiments/ml_lab/runs/logreg_regime.joblib`; exposes `route(bar_features) -> {'regime': 'trend'|'chop'|'volatile', 'p_trend': f, 'p_chop': f, 'p_vol': f, 'confidence': f}`. Reads autocorrelation and range_pct from rolling buffer. Falls back to range_pct-only heuristic if model load fails.
- `signals/regime_signals.py`: Three signal generators (`trending_signal`, `choppy_signal`, no-op for volatile) each consuming the same bar features but applying regime-specific thresholds.
- `risk/capital_allocator.py`: 30-min rebalance loop. Tracks `rolling_pnl_30min` per lane and reallocates buying power: if TREND-LANE up > +0.3% and REV-LANE flat, shift 20% of REV-LANE capital to TREND-LANE for next 30 min. Capped at 70/30 max skew.
- `data/strategy_specs/baseline/regime_trend_lane.json`: TREND-LANE spec — signal modes `[momentum, breakout]`, symbol_universe `[SPY, QQQ, NVDA, AMD]`, regime_filter `p_trend > 0.5 AND autocorr5 > +0.1`, TP/SL/time `(35, 22, 12)`.
- `data/strategy_specs/baseline/regime_rev_lane.json`: REV-LANE spec — signal mode `[mean_reversion]`, symbol_universe `[SPY, QQQ, AAPL]`, regime_filter `p_chop > 0.5 AND autocorr5 < −0.05`, TP/SL/time `(18, 14, 5)`.
- `scripts/alpaca_max_hft_host.py` (patch): Replace single-signal-evaluation block with `regime = router.route(features); spec = lane_specs[regime]; signal = signal_generators[regime](features, spec)`. Add per-bar regime label to fills CSV for post-trade evaluation.
- `experiments/ml_lab/eval_regime_router.py`: Post-close analysis joining fills with regime labels; computes per-regime TP rate and per-regime P&L to validate the routing hypothesis before round 2.
