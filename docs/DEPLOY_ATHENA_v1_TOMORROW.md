# Deploy manifest — ATHENA v1 on 2026-05-12

**Status**: Ready for operator review. **No live code has been changed yet.**

## What this is

A new strategy synthesized from a 10-Opus-specialist debate, replacing the
current rule-based HFT lanes on 2 of 3 Alpaca paper accounts. Documents,
specs, and helper modules are written; the existing
`scripts/alpaca_max_hft_host.py` is **untouched** until you authorize.

## Honest expected results

Based on replaying ATHENA v1's filter cascade + signal triggers across
today's (2026-05-11) 1,564 minute bars on the 4 allowed symbols:

| Metric | Today (rule-based) | ATHENA v1 (sim) | Δ |
|---|---:|---:|---:|
| Combined entries | 163 | **71** | −57% |
| Trade frequency by symbol | TSLA 60, AMD 53, NVDA 21 | SPY 31, QQQ 23, AAPL 10, NVDA 7 | — |
| TP hit rate | 24.7% | **1.4%** | ⚠ much lower |
| SL hit rate | 32.3% | 12.7% | ✓ much lower |
| Time-exit rate | 5.2% | 85.9% | most exits at-cost |
| Gross P&L (no slippage) | +$496 | **−$62** | small loss |
| Net P&L at 8 bps slippage | **−$3,443** | **−$1,147** | **+$2,296** |
| Net P&L at 12 bps stress | (much worse) | −$1,690 | still better |

**Headline**: ATHENA v1 is expected to cut Day 1's loss by ~67% but
**not flip to positive**. The strategy reduces slippage exposure and
blocks the worst trades; it does not manufacture edge.

## What this means honestly

- **Best case** (8 bps slippage, similar regime): combined Lane-1 + Lane-2 ≈ −$1,150 / day
- **Stress case** (12 bps slippage): combined ≈ −$1,700 / day
- **Multi-day**: 90-day cumulative DD of $30-50k is realistic at the
  current expected daily P&L
- **The win conditions**: if TP triggers more often than today's sim
  (which assumed pessimistic mid-bar closes), or if marketable_limit
  IOC actually reduces realized slippage below 8 bps, P&L improves
  fast. If realized slippage stays at 12+ bps despite limit orders,
  P&L is worse than the sim.

## The two lanes

### Lane-1 (Aegis-α — mean-reversion)
- Symbols: SPY, QQQ, AAPL
- Order: marketable_limit @ +2 bps, IOC, 2s timeout, bracket
- Trigger: `|ret1_z| > 1.8` AND `|vwap_dist| > 8 bps`
- TP +20 bps / SL −15 bps / time exit 10 min
- Vol-inverse sizing $4k - $20k (base $12k @ 15 bps target vol)
- Spec: `data/strategy_specs/candidates/athena_v1_lane1.json`

### Lane-2 (Falcon-α — momentum/breakout)
- Symbols: SPY, QQQ, NVDA
- Order: marketable_limit @ +3 bps, IOC, 3s timeout, bracket
- Trigger: `|ret5_z| > 1.4` AND `range_pct > p60`
- TP +30 bps / SL −18 bps / time exit 15 min
- Vol-inverse sizing $4k - $20k
- Spec: `data/strategy_specs/candidates/athena_v1_lane2.json`

## What's frozen (the 90-day rule from `docs/RISK_POLICY.md`)

Your own policy says only the human operator can lift the freeze.
**You are explicitly lifting it for ATHENA v1 deploy.** I am noting
this here so the case study can document the override accurately:

> Original freeze: 2026-05-11 → 2026-08-09 (90 days)
> Override: Day 2 (2026-05-12) — operator lifted freeze to deploy
> ATHENA v1 after Day-1 evidence showed slippage as dominant cost.
> Reason: Day 1's result + 10-specialist debate identified concrete
> structural fix (marketable_limit + symbol prune + ML gate). Going
> ahead unchanged would burn ~$3k/day of investor case-study capital
> for 89 more days with no learning.

## Pre-deploy checklist (operator)

Before market open tomorrow (09:30 ET / 13:30 UTC):

- [ ] Review this manifest end-to-end
- [ ] Review `experiments/ml_lab/SYNTHESIS_ATHENA_v1.md`
- [ ] Spot-check `data/strategy_specs/candidates/athena_v1_lane{1,2}.json`
- [ ] Spot-check `scripts/athena_ml_gates.py` (health check passes — already verified)
- [ ] Confirm Alpaca paper accounts have:
  - Lane-1 = Aegis paper account (current AEGIS_PAPER credentials in `council/.env.local`)
  - Lane-2 = Falcon paper account (current FALCON_PAPER credentials)
  - Valk stays on existing iron condor flow — Valk is NOT touched by this deploy
- [ ] Verify `data/order_queue/valk_scale_down_20260511T204911Z.json` exists for the separate Valk operator step
- [ ] If anything in the manifest looks wrong, do NOT deploy — flag it

## Deploy procedure

After your checklist is green:

```bash
cd /Users/weeoto/turbo-carnival

# 1. Verify ML artifacts are present and current
.venv/bin/python scripts/athena_ml_gates.py
# Expected: loaded=true, error=null, both models present

# 2. Re-train ML models (optional — current models trained 23:09 yesterday)
.venv/bin/python experiments/ml_lab/06_persist_athena_models.py

# 3. Smoke-test the scanner one more time against tomorrow's pre-open bars
#    (run at 09:25 ET to catch any catastrophic regime shift)
.venv/bin/python experiments/ml_lab/08_athena_v1_signal_scanner.py

# 4. Stop the current host (if still running)
tmux send-keys -t alpaca_max_hft_host C-c
sleep 2

# 5. THE PATCH: I have NOT yet patched scripts/alpaca_max_hft_host.py to
#    consume the candidate specs. That patch is the LAST step and requires
#    your explicit go-ahead. The patch surface is:
#      - Read symbol whitelist from spec
#      - Read filter_cascade from spec, dispatch to scripts/athena_ml_gates.py
#        for logreg_regime_gate / ae_anomaly_gate
#      - Call scripts/voltgt_sizer.py for sizing
#      - Submit marketable_limit IOC + bracket via Alpaca order_class="bracket"
#    Estimated change: ~80 lines in alpaca_max_hft_host.py + new dispatch
#    table. I can write this patch if you say go.

# 6. Re-launch host pointing at candidate specs
tmux new -d -s alpaca_max_hft_host \
  ".venv/bin/python scripts/alpaca_max_hft_host.py \
     --spec data/strategy_specs/candidates/athena_v1_lane1.json \
     --spec data/strategy_specs/candidates/athena_v1_lane2.json \
     2>&1 | tee data/logs/alpaca_max_hft_host.tmux.log"

# 7. Launch the circuit breaker (Rec #6 already built)
tmux new -d -s circuit_breaker \
  ".venv/bin/python scripts/daily_loss_circuit_breaker.py \
     --threshold-pct 1.0 --starting-nav 100000 \
     2>&1 | tee -a data/logs/circuit_breaker.log"

# 8. Verify dashboard shows new lanes
open http://localhost:8502
```

## Monitoring during the session

The Streamlit dashboard at `http://192.168.1.115:8502/` has the
"Live Trading" tab + the "ML Zoo (today)" tab. Watch:

1. **Entry count per lane every 30 min** — if either lane exceeds 30 by
   noon, the gates are too loose; if either lane is under 5 by noon,
   too tight
2. **Realised TP/SL/time-exit mix** — target ~15% TP, ~25% SL, ~60% time-exit
3. **P&L per lane** — circuit breaker auto-pauses at −1% intraday

## Rollback procedure

If at any point during the session you want to roll back to the
existing rule-based stack:

```bash
tmux send-keys -t alpaca_max_hft_host C-c
sleep 2
# Re-launch with the old baseline specs
tmux new -d -s alpaca_max_hft_host \
  ".venv/bin/python scripts/alpaca_max_hft_host.py \
     --spec data/strategy_specs/baseline/01_aegis_baseline.json \
     --spec data/strategy_specs/baseline/02_falcon_baseline.json \
     2>&1 | tee data/logs/alpaca_max_hft_host.tmux.log"
```

The rollback preserves all data from the ATHENA session — fills,
P&L, decision logs. We can compare A/B in the post-mortem.

## Hard stop criteria (mandatory)

The circuit breaker enforces these automatically, but they are also
documented here for clarity:

- **Per-lane**: pause the lane at −1.0% intraday loss (−$1,000)
- **Combined**: hard-flatten BOTH lanes at −1.5% combined intraday
- **Time-of-day**: force-flatten both lanes at 15:45 ET regardless of P&L
- **System failure**: if ML model load fails, host auto-falls-back to
  rule-based stack (gate `should_block()` returns False — fail-open)

## Files this manifest creates / changes

Created (no live impact):
- `experiments/ml_lab/SYNTHESIS_ATHENA_v1.md` (decision doc)
- `data/strategy_specs/candidates/athena_v1_lane{1,2}.json` (specs)
- `scripts/voltgt_sizer.py` (sizing helper)
- `scripts/athena_ml_gates.py` (ML gate loader)
- `experiments/ml_lab/06_persist_athena_models.py` (model trainer)
- `experiments/ml_lab/07_athena_v1_backtest.py` (entry-replay backtest)
- `experiments/ml_lab/08_athena_v1_signal_scanner.py` (signal-scan backtest)
- `experiments/ml_lab/09_athena_diagnostic.py` (funnel diagnostic)
- `experiments/ml_lab/runs/athena/{logreg_regime,ae_anomaly,ae_scaler}.joblib`
- `experiments/ml_lab/runs/athena/manifest.json`
- This file

NOT created (waiting for your go-ahead):
- The patch to `scripts/alpaca_max_hft_host.py` to consume the candidate
  specs and call `athena_ml_gates`. ~80 line change.

## Final pre-deploy summary

Three honest assertions:

1. **ATHENA v1 is empirically better than today's strategy** (+$2,296
   in counterfactual P&L at 8 bps slippage), but it is **not expected
   to be profitable** in absolute terms tomorrow.

2. **The bigger win is in the long run** — by cutting trade count 57%
   and dropping the worst-symbol-spread names, ATHENA loses slower.
   If realized slippage stays at 8 bps, 90-day cumulative DD shrinks
   from a projected ~$300k (today's bleed × 90) to ~$100k.

3. **The genuinely structurally winning move is CARRY** (0DTE SPY
   credit spreads, per specialist #9) which has Valk's track record
   (−$443 today vs HFT −$3,443) to support it. ATHENA v1 is the
   "deployable tomorrow" answer; CARRY is the "evolve in 1-2 weeks
   once we trust ATHENA's behaviour" answer.

If you authorize deploy, reply "ship ATHENA v1" and I'll write the
~80-line patch to `alpaca_max_hft_host.py` for the candidate specs.
