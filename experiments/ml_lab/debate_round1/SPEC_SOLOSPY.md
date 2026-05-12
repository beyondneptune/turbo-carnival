## Specialist: SOLOSPY

### Core thesis (1 sentence)

Concentrate 100% of HFT capital into SPY only across both lanes — SPY's 0.014 bps natural spread (vs TSLA/AMD at 5-20 bps) is the single largest controllable variable, and eliminating the 113 high-cost TSLA/AMD entries removes ~$2,700 of today's $3,939 slippage gap before we change a single signal.

### Strategy decisions

1. **trade_frequency**: 50 entries/lane/day (100 combined). Down from 163 today. Same instrument both lanes means we must avoid simultaneous opposing positions — gate by lane-conflict check. SPY's depth (~$50M at NBBO) lets us trade smaller-and-more vs the same notional in TSLA.
2. **order_type**: Marketable limit with **2 bps offset** (≈ $1.40 on SPY @ 700). SPY's tick is $0.01 = 0.014 bps, so 2 bps gives ~10 ticks of cushion — virtually guaranteed fill while capping adverse selection. NO market orders (they pay full half-spread + impact). Bracket orders for TP/SL legs to avoid latency on exits.
3. **position_sizing**: Uniform **$15k notional/entry** (≈ 21 shares of SPY). Smaller-per-trade than today's ~$20k, but with 50 entries/lane = $750k gross turnover/lane (similar). Lower per-trade size → lower per-trade impact → linear-regime slippage.
4. **entry_signals**:
   - Lane-1 (mean-reversion): SPY 1-min `z_score(close, 20) < -1.8` AND `z_score > -3.0` (no falling-knife) AND VWAP-distance < -8 bps
   - Lane-2 (momentum/breakout): SPY 5-min Donchian-20 breakout AND `range_pct` in top quartile AND VIX < 22 (regime gate)
   - Both lanes: LogReg P(profitable) > 0.55 required to fire
5. **exit_rules**:
   - TP: +12 bps (Lane-1) / +18 bps (Lane-2). SPY's tight book means TP fills are reliable.
   - SL: −8 bps hard stop, bracket-attached.
   - Time exit: 12 min (Lane-1) / 20 min (Lane-2).
   - ML kill: AE anomaly score > 99th pct → skip entry. LogReg P < 0.45 → exit immediately even if not at TP/SL.
6. **symbol_universe**: **SPY ONLY**. Drop QQQ, AAPL, AMD, META, NVDA, TSLA from HFT lanes. (Valk's iron condor stays separate — different account, different strategy.)
7. **lane_split**:
   - Lane-1: SPY mean-reversion, short holding (≤12 min), z-score driven
   - Lane-2: SPY momentum/breakout, longer holding (≤20 min), Donchian driven
   - Conflict rule: if Lane-1 is short SPY and Lane-2 wants long SPY (or vice versa), Lane-2 stands down (mean-rev has priority on z-score extremes).
8. **filter_cascade** (ordered, short-circuit on fail):
   1. `time_gate` — block 09:30-09:35 and 15:55-16:00 (open/close noise)
   2. `vix_gate` — VIX > 28 → halt both lanes (regime breakdown)
   3. `lane_conflict_check` — prevent opposing SPY positions across lanes
   4. `signal_trigger` — z-score (L1) or Donchian (L2)
   5. `LogReg P(win) > 0.55` — passes overfit gate, +$1,070 counterfactual
   6. `AE anomaly < 99th pct` — 68.7% precision@top1%, zero overfit
   7. `liquidity_check` — NBBO size > 5x our order size (always true for SPY at $15k)
   8. Send marketable-limit order with 2 bps offset

### Expected backtest result

- Daily entries: 163 → **100** (50/lane)
- Block rate vs today: **39%** (ML+regime filters on top of universe collapse)
- Predicted Δ P&L vs today on bar-close basis: **+$1,400** (slightly above LogReg's $1,070 counterfactual because SPY-only also captures cleaner signal/noise on bar-close basis)
- Predicted Δ P&L at 8 bps slippage: **+$300 to +$1,800** (median ~+$900). SPY 8 bps is roughly its 95th-pct slippage day; realistic median is 2-4 bps.
- Confidence: **med-high** (SPY liquidity is the most-studied empirical fact in US equities; signal still needs validation)

### Steelman against my own approach (1-2 sentences)

Single-instrument concentration kills diversification — if SPY enters a low-volatility regime where neither mean-reversion nor breakout fires, both lanes sit idle and we generate zero alpha while still paying infrastructure cost. Worse, SPY's tight spread is partly *because* informed flow is fast there, so retail HFT may face worse adverse selection per trade even though gross slippage is lower; the real edge in TSLA/AMD was that wider spreads also reflect inventory premium we were *paying*, not capturing.

### My critique of <ANY_OTHER_SPECIALIST_BY_NAME> approach (1 sentence)

**Critique of any "diversify across all 7 symbols equal-weight" specialist**: equal-weighting SPY against TSLA is a category error — you're paying 350x the spread cost on TSLA for marginal correlation benefit, when today's data shows the correlation gain didn't even survive Day 1.

### Files I'd write

- `data/strategy_specs/baseline/solospy_lane1_meanrev.json`: SPY-only mean-rev spec, z-score(20) < -1.8 entry, 12 bps TP / 8 bps SL bracket, marketable-limit +2 bps offset, 50 max entries/day, LogReg + AE filter chain.
- `data/strategy_specs/baseline/solospy_lane2_breakout.json`: SPY-only Donchian-20 breakout spec, 18 bps TP / 8 bps SL, 20 min time-exit, VIX<22 regime gate, marketable-limit +2 bps offset.
- `scripts/lane_conflict_guard.py`: pre-trade hook for `alpaca_max_hft_host.py` that queries both Alpaca paper accounts for open SPY positions and blocks Lane-2 entries that would create offsetting exposure to Lane-1.
- `experiments/ml_lab/spy_only_backtest.py`: replays the 21-day cached `minute_bars.parquet` with SPY-only universe and the above filter cascade; produces bar-close vs 2/4/8/12 bps slippage P&L curves to validate the +$1,400 prediction before tomorrow's open.
