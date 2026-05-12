# ATHENA v1 — synthesis of 10-specialist debate

Designed 2026-05-11. Candidate model for deployment on 2 Alpaca paper
accounts tomorrow (2026-05-12). **This document is the decision artifact.**

## Strong consensus across 10 Opus specialists

| Item | Votes / 10 | Decision |
|---|---:|---|
| Drop TSLA from universe | **9** | Dropped |
| Drop AMD from universe | **6** | Dropped |
| Use marketable limits, NEVER market orders | **10** | Marketable_limit + IOC |
| Use LogReg regime gate (validated +$1,070 counterfactual) | **9** | Hard gate at p_vol ≥ 0.5 |
| Use AE as some form of veto (zero overfit) | **9** | Hard veto at recon > p99 |
| Time-window restriction | **10** | 10:00 ET – 15:30 ET (relaxed vs WINDOW's 14:30) |
| Bracket orders for atomic TP/SL | **8** | order_class=bracket |
| Cut trade count drastically | **10** | ~25 / lane / day (median of proposals) |

## Decisions on disputed points

### Trade frequency: 25 per lane per day (median consensus)
SPARSE's ≤5/lane is statistically sound but produces too-thin sample for any one-day evaluation. VOLTGT's ~70/lane keeps too much slippage tax. The 60-model swarm + ML lab counterfactual both point to ~25-50/lane as the sweet spot.

### Order type: Marketable_limit IOC + bracket (EXEC's design, modified)
- Entry: limit @ touch ± 2 bps in trade direction, TIF=IOC. 2 second timeout, then cancel.
- Bracket parent: server-side TP/SL legs to remove one round of slippage exposure.
- SL leg uses stop_limit (not naked stop) with 3 bps limit offset.

### Position sizing: Vol-inverse (VOLTGT's formula, capped lower than VOLTGT)
- Base: $12,000 per entry (smaller than today's $20k; smaller-per-trade = less impact)
- Multiplier: 15 / realized_vol_5min_bps
- Clamp: [$4,000, $20,000]
- Rationale: shrinks notional in costly high-vol minutes (where slippage was worst today), grows in quiet tape.

### Symbol universe: SPY, QQQ, AAPL, NVDA
Drops TSLA + AMD (loss epicenters today, widest spreads). Keeps META off Lane-1 to avoid concentration. Mid-cap META re-eligible for Lane-2 if breakout signals fire there.

### Lane split: Strategy-split (not parameter-split)
- **Lane-1 (Aegis-α)**: Mean-reversion only. Symbols SPY, QQQ, AAPL.
- **Lane-2 (Falcon-α)**: Momentum / breakout. Symbols SPY, QQQ, NVDA.
- SPY runs in both for diversification across strategies, not against itself.
- Conflict guard: if both lanes long-or-short SPY simultaneously, Lane-2 stands down (Lane-1 priority on overlapping signals).

### ML cascade: LogReg gate + AE hard veto + soft AE sizing
- LogReg `p_vol ≥ 0.5` → hard block
- AE `recon_err > p99_train` → hard veto (overrides signal)
- AE `recon_err > p95_train` → halve size (soft)
- Mid-trade: if either model flips above threshold, exit at marketable_limit

## Why we rejected the structurally different proposals

### CARRY (0DTE SPY credit spreads)
**Mathematically the strongest** (Valk's −$443 vs HFT's −$3,443 today is the cleanest evidence). Rejected for Day 2 because:
1. Requires new host code (`alpaca_0dte_credit_spread_host.py` — not yet written)
2. Multi-leg order construction + Greeks polling + assignment handling = 6-12 hours of new code with unknown bugs
3. 0DTE tail risk could blow Day 2 if SPY moves 2σ
4. Investor case study premise is 3-lane diversity; converting 2 lanes to options changes narrative
- **Tabled as Phase-3 candidate** (deploy after ATHENA v1 validates the slippage thesis)

### PAIRS (cointegrated pairs)
Theoretically elegant. Rejected because:
1. 21-day cointegration window is dangerously thin; ~5% false-positive rate on Engle-Granger
2. Cointegration breaks on regime shifts; intraday half-life is unstable
3. Orphan-leg failure is the worst failure mode and is hard to test without a live run

### REGIME (3-way routing)
Strong idea. Rejected because:
1. Requires the LogReg's regime probability to be *calibrated*, not just discriminative; 0.799 AUC doesn't guarantee well-calibrated probabilities
2. Three sub-strategies multiplies parameter surface area → overfit risk
3. ATHENA v1 already uses LogReg, just as a binary gate (lower-risk profile of the same model)

### SOLOSPY (SPY-only on both lanes)
Almost won. Rejected only because we lose lane diversity entirely — one bad SPY hour and both lanes bleed in parallel. ATHENA v1 keeps SPY central but adds 1-2 more names per lane.

## The final spec — ATHENA v1

### Lane-1: Aegis-α (mean-reversion)

```yaml
account: AEGIS_PAPER
strategy: mean_reversion_v1
symbols: [SPY, QQQ, AAPL]
max_entries_per_day: 25
order_type: marketable_limit_ioc_bracket
limit_offset_bps: 2
ioc_timeout_sec: 2
position_sizing:
  formula: vol_inverse
  base_notional_usd: 12000
  target_vol_bps: 15
  min_notional_usd: 4000
  max_notional_usd: 20000
entry_thresholds:
  ret1_z_min_abs: 1.8
  vwap_dist_min_bps: 8
  spread_max_bps: 5
exit_rules:
  tp_bps: 20
  sl_bps: 15
  sl_type: stop_limit
  sl_limit_offset_bps: 3
  time_exit_min: 10
  ml_kill:
    logreg_p_vol_exit: 0.7
    ae_recon_exit_pctile: 99
filter_cascade:
  - time_gate: { start_et: "10:00", end_et: "15:30" }
  - news_blackout: { cooldown_sec: 600 }
  - spread_gate: { max_bps: 5 }
  - abs_return_gate: { threshold_bps: 60 }
  - logreg_regime_gate: { p_vol_block: 0.50 }
  - ae_anomaly_gate: { hard_veto_pctile: 99, size_down_pctile: 95 }
  - signal_trigger: mean_reversion
  - position_correlation_check: { max_open: 2 }
```

### Lane-2: Falcon-α (momentum / breakout)

```yaml
account: FALCON_PAPER
strategy: momentum_breakout_v1
symbols: [SPY, QQQ, NVDA]
max_entries_per_day: 25
order_type: marketable_limit_ioc_bracket
limit_offset_bps: 3   # momentum tolerates slightly more chasing
ioc_timeout_sec: 3
position_sizing:
  formula: vol_inverse
  base_notional_usd: 12000
  target_vol_bps: 15
  min_notional_usd: 4000
  max_notional_usd: 20000
entry_thresholds:
  ret5_z_min: 1.4
  range_pct_min_pctile: 60
  spread_max_bps: 6
exit_rules:
  tp_bps: 30   # momentum lets winners run further
  sl_bps: 18
  sl_type: stop_limit
  sl_limit_offset_bps: 3
  time_exit_min: 15
  ml_kill:
    logreg_p_vol_exit: 0.7
    ae_recon_exit_pctile: 99
filter_cascade:
  - time_gate: { start_et: "10:00", end_et: "15:30" }
  - news_blackout: { cooldown_sec: 600 }
  - spread_gate: { max_bps: 6 }
  - abs_return_gate: { threshold_bps: 80 }
  - logreg_regime_gate: { p_vol_block: 0.50 }
  - ae_anomaly_gate: { hard_veto_pctile: 99, size_down_pctile: 95 }
  - signal_trigger: momentum_breakout
  - position_correlation_check: { max_open: 2 }
  - lane_conflict_check: { yields_to: Aegis }
```

## Expected results vs today

| Metric | Today actual | ATHENA v1 expected |
|---|---:|---:|
| Total entries (both lanes) | 163 | **~50** |
| Block rate vs today | 0% | **~69%** |
| Avg notional / entry | ~$20k | **~$12k** |
| Slippage cost / round trip | ~$12 | **~$5** (smaller size + limit vs market) |
| Daily slippage cost (total) | $3,939 | **~$500** |
| Bar-close P&L | +$496 | **+$200 to +$500** (some lost edge from blocking) |
| Net P&L at realistic slippage | **−$3,443** | **−$200 to +$600** range |
| Confidence | n/a | med (med-high on slippage reduction, low on absolute P&L) |

## What we explicitly are NOT promising

- Flipping the sign positive on Day 2. The synthesized model expects to **roughly break even** at realistic slippage; positive P&L requires the bar-close edge to actually persist out-of-sample.
- An end to drawdowns. The 90-day cumulative DD policy in `docs/RISK_POLICY.md` still applies.
- That the model is the right answer at Day 30 or Day 60. CARRY remains the structurally better play; ATHENA is the "right now, deployable" answer.

## Rollback plan

If Lane-1 or Lane-2 exceeds −1.0% intraday loss before 12:00 ET:
1. Disable the lane via `data/circuit_breaker/<lane>_paused.json`
2. Hold open positions to their existing TP/SL/time-exit
3. Do not re-enable until human operator review

If both lanes exceed −0.5% by 12:00 ET → hard stop both, flatten everything.

## Files written by this synthesis

- `data/strategy_specs/candidates/athena_v1_lane1.json` (Aegis-α spec)
- `data/strategy_specs/candidates/athena_v1_lane2.json` (Falcon-α spec)
- `scripts/athena_marketable_limit_bracket.py` (order helper)
- `scripts/athena_logreg_gate.py` (LogReg loader + predict wrapper)
- `scripts/athena_ae_veto.py` (AE loader + percentile threshold)
- `scripts/voltgt_sizer.py` (vol-inverse notional calculator)
- `experiments/ml_lab/athena_v1_backtest.py` (replays today's 163 entries through the new cascade)
- `docs/DEPLOY_ATHENA_v1_TOMORROW.md` (operator manifest)
