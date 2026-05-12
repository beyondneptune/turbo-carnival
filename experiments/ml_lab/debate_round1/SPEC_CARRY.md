## Specialist: CARRY

### Core thesis (1 sentence)

Kill both HFT lanes outright and convert them to 0DTE SPY credit-spread sellers — Valk's −$443 vs HFT's −$3,443 is the market telling us theta beats microstructure at retail latency, and 6-12 bps round-trip × 80 trades/day is a structurally unwinnable game regardless of signal quality.

### Strategy decisions

1. **trade_frequency**: 1-4 entries per lane per day (target avg 2). Hard cap 4. No re-entry after a stop within same session.
2. **order_type**: Multi-leg limit (vertical spread) at mid −$0.02 for 5 seconds, then chase mid −$0.05, then cancel. Use Alpaca `mleg` order class with `class: mleg`, `order_class: simple`, `legs: [{side: sell, ratio_qty: 1, ...}, {side: buy, ratio_qty: 1, ...}]`. No market orders on options — ever.
3. **position_sizing**: Risk-defined. Each spread risks ≤$300 (3% of $10k buying-power slice). With $100k account, max 30 spreads on the book at once but realistic load = 4-8. Credit target 25-35% of strike width. Width = $3 for 10-delta short.
4. **entry_signals**: 
   - **Lane-1 (put credit)**: Enter 09:45-15:00 ET when SPY VIX9D < 25, SPY ATR(5m) < 0.4%, and SPY is above VWAP. Short strike = 10-delta put (≈1.0-1.5% OTM on a 0DTE).
   - **Lane-2 (call credit)**: Enter 09:45-15:00 ET when VIX9D < 25, ATR(5m) < 0.4%, SPY below VWAP-thrust. Short strike = 10-delta call.
   - Skip both lanes on FOMC/CPI/NFP days (`news_blackout`).
5. **exit_rules**:
   - **TP**: Buy-to-close at 50% of credit captured (industry-standard tasty rule). 
   - **SL**: Close if short strike is breached (delta ≥ 30) OR loss ≥ 2× credit received.
   - **Time**: Hard close all legs at 15:45 ET. Never carry 0DTE into final 15min gamma cliff.
   - **ML kill**: If LogReg regime model emits regime ≤ 0.2 (trend-day probability high), don't open new spreads for 60min. AE anomaly score > 99th-pct → skip the side facing risk.
6. **symbol_universe**: SPY only. QQQ as fallback if SPY 0DTE chain liquidity drops (it won't — SPY 0DTE is the deepest options market on earth). Drop AAPL/AMD/META/NVDA/TSLA entirely — single-name options spreads are murder at retail.
7. **lane_split**:
   - **Lane-1**: Put credit spreads only (bullish-neutral premium harvester).
   - **Lane-2**: Call credit spreads only (bearish-neutral premium harvester).
   - Combined book = synthetic iron condor with independent legging, lets each side enter on its own signal rather than waiting for both.
8. **filter_cascade** (ordered, short-circuit on first fail):
   1. `news_blackout` (FOMC/CPI/NFP/earnings)
   2. `time_gate` (09:45 ≤ t ≤ 15:00 ET)
   3. `vol_regime_gate` (VIX9D < 25, term-structure not inverted)
   4. `atr_gate` (5m ATR < 0.4% of spot)
   5. `vwap_directional_filter` (per-lane)
   6. `LogReg regime score ≥ 0.5`
   7. `AE anomaly < 99th-pct` for that side
   8. `liquidity_check` (chain bid-ask ≤ $0.10 on short strike)

### Expected backtest result

- Daily entries: 163 → **~4 (2 per lane)**
- Block rate vs today: **~97.5%**
- Predicted Δ P&L vs today on bar-close basis: **+$2,800** (eliminates the $3,939 slippage hole; replaces with ~$1,100 expected daily theta − ~$0.30 × 4 spreads × $30 slip ≈ $36 slip cost)
- Predicted Δ P&L at 8 bps slippage: **+$1,500 to +$3,200** (options slippage on SPY 0DTE 10-delta is roughly $0.05-0.10/leg = $10-20 per spread, total daily cost ~$40-80 vs $3,939 today)
- Confidence: **high** on slippage math, **med** on win-rate assumption (historic 0DTE 10-delta credit-spread win rate ≈ 75-80% before tail events)

### Steelman against my own approach (1-2 sentences)

The fat-tail risk is asymmetric and ugly: one VIX-spike day where SPY moves 3σ through both short strikes can wipe a month of theta in a single afternoon, and 0DTE gamma exposure near close is unforgiving. My defined-risk wings cap the disaster at $300/spread but a correlated multi-spread loss day could still hit −$3-5k, comparable to today's HFT bleed.

### My critique of MOMO approach (1 sentence)

Any HFT-momentum specialist arguing for "tighter filters + same 80 trades/day" is ignoring that the slippage problem is **mechanical, not statistical** — you cannot filter your way out of a 12 bps × N cost structure when your edge per trade is also measured in bps; cutting trade count by 95% is the only knob that actually moves the cost line.

### Files I'd write

- `data/strategy_specs/baseline/lane1_put_credit_0dte.json`: spec for daily SPY 0DTE put-credit-spread lane with VIX9D/ATR/VWAP gates, 50% TP, 2x-credit SL, 15:45 forced close, LogReg+AE filter cascade in the order above.
- `data/strategy_specs/baseline/lane2_call_credit_0dte.json`: mirror spec for call-credit lane.
- `scripts/alpaca_0dte_credit_spread_host.py`: new host that replaces `alpaca_max_hft_host.py` for these two lanes — handles multi-leg order construction via Alpaca `mleg` order class, mid-chase logic, intraday Greeks polling for SL check, and 15:45 ET hard-flatten. Reuses existing ML model loader for LogReg regime + AE anomaly score.
- `experiments/ml_lab/debate_round1/CARRY_slippage_math.md`: one-page proof showing 4 × $0.10 × 2 legs = $0.80/day options slip vs $3,939/day equity slip, the entire argument in 10 lines.
