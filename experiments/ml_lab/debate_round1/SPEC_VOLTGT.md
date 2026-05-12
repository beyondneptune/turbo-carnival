## Specialist: VOLTGT

### Core thesis (1 sentence)

Slippage is a linear function of realized volatility, so the single highest-leverage fix is to make notional **inversely proportional to 5-min realized vol** (target 15-20 bps/min portfolio vol), shrinking us to ~$5k in chop and growing to ~$40k in quiet tape — keep entry signals, kill the size-vs-regime mismatch that bled $3.9k today.

### Strategy decisions

1. **trade_frequency**: ~70 entries/lane/day (down from 80 baseline; vol gate prunes ~15% of attempted entries when realized_vol_5min > 80 bps or unmeasurable).
2. **order_type**: marketable_limit with vol-scaled offset = `max(2 bps, 0.25 × realized_spread_bps)`. Cap offset at 8 bps. No pure market orders — they were the per-trade $12 leak. Bracket the position with native Alpaca bracket order (TP+SL legs server-side) so a network hiccup can't strand us.
3. **position_sizing**: **THIS IS THE THESIS.** 
   - `notional = clip($20k × (17 / realized_vol_5min_bps), $5k, $40k)`
   - `realized_vol_5min_bps` = stdev(log returns of last 5 1-min bars) × sqrt(5) × 10000, computed per-symbol, refreshed every minute.
   - Floor $5k protects against vol spikes; ceiling $40k caps single-name concentration at 40% of one $100k lane.
   - When `realized_vol_5min` is NaN (first 5 min after open, or stale data), use $10k default.
   - Cross-check: portfolio-level realized vol target 15-20 bps/min. If aggregate exceeds 25 bps, halve all open size on next entry until it normalizes.
4. **entry_signals**: keep today's mean_reversion + breakout entries unchanged — *signal generation is not the bottleneck*. Add only one gate: `realized_vol_5min_bps < 100` (kill switch for fire-drill minutes; vol > 100 bps/min implies >25 bps round-trip slippage, edge gone).
5. **exit_rules**: 
   - TP: 18 bps (was hit 24.7% today — keep, but scaled by `1 + vol_ratio/4` so high-vol entries demand more profit before exit).
   - SL: 35 bps (was effectively ~25 bps after slippage — widen to absorb noise on small-size, low-vol entries; small notional means fewer dollars at risk per bps).
   - Time exit: 8 min (was implicit; explicit cap so we don't bleed theta in dead tape).
   - ML kill: LogReg (AUC 0.799, train-test gap +0.006) gates every entry — block if `p_loss > 0.55`. AE anomaly score blocks if > 99th-percentile training threshold.
6. **symbol_universe**: SPY, QQQ, AAPL, AMD, NVDA, META, TSLA — keep all 7 but **let vol-targeting do the rebalancing organically**. TSLA's higher realized vol means smaller TSLA notional, which mechanically reduces the 60-trade overconcentration we saw today without a hard symbol cap.
7. **lane_split**: 
   - **Lane-1 (VOLTGT-prime)**: full vol-scaled sizing, all 7 symbols, both signal modes.
   - **Lane-2 (VOLTGT-conservative)**: same logic, but target_vol = 12 bps (more defensive), floor $4k / ceiling $25k, SPY+QQQ+AAPL only (lowest-slippage names).
   - This is an A/B on the vol target itself, not on the thesis — both lanes embody the same idea.
8. **filter_cascade** (ordered, fail-fast):
   1. `data_freshness_check` (bar < 90s old)
   2. `realized_vol_5min < 100 bps` (vol kill)
   3. `LogReg p_loss < 0.55` (ML gate, AUC 0.799)
   4. `AE_anomaly < 99th-pct` (regime anomaly)
   5. `time_gate` (skip first 5 min + last 10 min of session)
   6. `symbol_correlation_check` (avoid stacking 3+ concurrent TSLA/AMD entries)
   7. **`vol_targeted_size_calc`** (the sizing transform — this is where my edge lives)
   8. `bracket_order_submit` (marketable_limit + native TP/SL)

### Expected backtest result

- Daily entries: 163 → ~130 (lane-aggregate; vol gate + ML gate jointly prune ~20%).
- Block rate vs today: ~20%.
- Predicted Δ P&L vs today on bar-close basis: **+$680** (sizing increase in quiet windows captures more of the +$496 underlying edge; vol gate skips ~5 worst minutes).
- Predicted Δ P&L at 8 bps slippage: **+$200 to +$1,400**. Mechanism: average realized notional drops from $20k to ~$14k (vol-weighted), cutting slippage dollars ~30% (from $3,939 → ~$2,750); plus the +$496 bar-close edge stays roughly intact; plus high-vol minutes where slippage was likely 20+ bps see size dropped to $5k. Worst case the vol estimate lags a regime shift and we re-bleed $200.
- Confidence: **med**. The math is clean and equivalent to confidence-weighted Kelly, but realized 5-min vol is a *noisy* proxy for forward 1-min slippage — could lag a regime shift by 1-2 minutes.

### Steelman against my own approach (1-2 sentences)

Vol-inverse sizing assumes vol is **persistent enough on 1-5 min horizons** to be predictive — but at HFT speed, vol can mean-revert within the holding period, meaning I'd downsize *after* the worst is over and upsize *into* a quiet period that's about to break. Worse, if slippage is driven primarily by spread (a regime variable) rather than volatility (a flow variable), I'm sizing on the wrong axis and the optimal answer is order-type discipline (REGIME's domain), not size-scaling.

### My critique of LIMITONLY approach (1 sentence)

Limit-only with mid-peg offsets risks **adverse selection on every fill** — you only get filled when the market just moved against you — and on a 24.7% TP-hit-rate strategy that already struggles with timing, missing 30-40% of intended entries because limits don't fill will gut the signal density that makes mean-reversion work in the first place; better to keep marketable_limit with tight offset and absorb 2-3 bps to guarantee fills, then attack slippage with sizing (my lever) rather than passivity.

### Files I'd write

- `/Users/weeoto/turbo-carnival/scripts/voltgt_sizer.py`: pure function `compute_notional(realized_vol_5min_bps: float, target_vol_bps: float = 17, base_notional: float = 20000, floor: float = 5000, ceiling: float = 40000) -> float`. Tested against today's 163 entries' realized-vol windows to produce a sized-trade ledger.
- `/Users/weeoto/turbo-carnival/data/strategy_specs/baseline/voltgt_lane1.json` and `voltgt_lane2.json`: full spec dicts with `position_sizing.mode = "vol_target_inverse"`, target_vol_bps = 17 / 12, plus the filter cascade above. JSON shape compatible with `scripts/alpaca_max_hft_host.py` consumer.
- `/Users/weeoto/turbo-carnival/experiments/ml_lab/voltgt_backtest.py`: replays today's 1-min bar parquet (`experiments/runs/valk_hft_demo_20260511T140548Z/minute_bars.parquet`) through the vol-scaled sizer + 8-bps slippage model, outputs per-trade P&L diff vs today's uniform-$20k baseline.
