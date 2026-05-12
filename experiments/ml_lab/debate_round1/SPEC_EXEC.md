## Specialist: EXEC

### Core thesis (1 sentence)

Day 1's $3,939 slippage gap is not a signal problem or a sizing problem — it is a market-order tax of ~12 bps round-trip that no ML filter, no regime gate, and no symbol re-weighting can recover, so the entire strategy must be rebuilt around *liquidity-providing* order types (marketable limits at the touch, brackets for atomic TP/SL) and ruthless trade-count reduction; if I cannot pass-the-spread, I do not trade.

### Strategy decisions

1. **trade_frequency**: **20 entries/lane/day** (down from 80 baseline, down from 99 Falcon today). Each entry must clear a 6 bps expected-edge hurdle *after* assuming I pay half-spread + 2 bps adverse selection. No trade beats a bad trade.

2. **order_type**: **Marketable-limit + bracket, atomic, no naked stops.**
   - Entry: `limit` order at touch on passive side (long → bid, short → ask), `time_in_force=IOC`, `limit_price = touch ± 1 bp offset`. If unfilled in 2 seconds, **cancel and skip** (do NOT chase). Target fill rate: 60–70% of intended entries.
   - Exit: `bracket` order submitted *with* the entry (Alpaca paper supports `order_class=bracket` with `take_profit` and `stop_loss` legs). This kills the two-round-trip slippage tax — one decision, one round-trip of taker cost on the *worst* case only.
   - SL leg uses `stop_limit` with `limit_offset = 3 bps` past stop, not naked `stop` (which becomes a market at the worst possible tick).
   - Size > $10k: **fragment into 3 child orders** spaced 1.5s apart, all marketable-limit-IOC. Stops any single quote-lift from eating the whole parent.

3. **position_sizing**: **Uniform $8,000 notional/entry**. Smaller than today's ~$20k because (a) smaller orders cross fewer levels of the book, (b) Alpaca paper fill simulation penalizes size, (c) it lets me fragment without sub-$3k children that get queue-position-starved. 20 entries × $8k × 2 lanes = $320k gross turnover/day, manageable.

4. **entry_signals**: **Mean-reversion only, single mode**, threshold tightened. Specifically: 5-min z-score of (close − VWAP) / ATR(5) must be `|z| ≥ 2.2` (up from baseline ~1.5), AND the *quoted spread at signal time must be ≤ 3 bps* (kill the wide-spread trades that ate today's P&L). No momentum, no breakout, no news_fade — those modes demand taker liquidity by design and lose 12 bps before they start.

5. **exit_rules**:
   - **TP**: +18 bps (3× round-trip cost). Realistic given 24.7% TP hit rate today — need TP large enough that 1 TP pays for 3 SLs.
   - **SL**: −12 bps (1× round-trip cost; trades aren't allowed to compound losses past one full slippage tax).
   - **Time exit**: 12 minutes. Anything that hasn't moved in 12 min is dead weight occupying a bracket slot.
   - **ML kill**: if Autoencoder anomaly score > 99th percentile (0.687 prec@1%), **skip the entry entirely** — do not even submit the IOC limit. AE goes BEFORE the order, not as a post-fill exit.

6. **symbol_universe**: **SPY, QQQ, AAPL only.** Drop TSLA, AMD, NVDA, META. Justification: SPY/QQQ have the tightest spreads (typically <1 bp), AAPL is the tightest single-name. TSLA + AMD spreads are 5–8 bps and were 69% of entries / most of the loss today. A liquidity-providing strategy cannot survive on names where the spread *is* the edge you're trying to capture.

7. **lane_split**:
   - **Lane-1**: SPY + QQQ only, marketable-limit-IOC + bracket, 15 entries/day, $8k notional. The "clean" lane.
   - **Lane-2**: AAPL only, same order machinery, 20 entries/day (single-name lets me run slightly hotter), $8k notional. The "control" lane to A/B-test single-name vs ETF execution quality.
   - Different *symbols*, identical *order-type machinery*. We are not A/B-testing strategies — we are A/B-testing whether single-name execution is workable at all.

8. **filter_cascade** (ordered, short-circuit on first fail):
   1. **Spread gate**: quoted spread ≤ 3 bps. Hardest filter, cheapest to evaluate. Kills 40% of would-be entries pre-signal.
   2. **Time gate**: skip 09:30–09:35 ET and 15:55–16:00 ET. Open/close fill inflation is the #1 cited cause of paper-account slippage divergence.
   3. **LogReg regime score** ≥ 0.5 (AUC 0.799, passes overfit gate).
   4. **Autoencoder anomaly** < 99th pct. Pre-trade veto, not post-trade.
   5. **Signal**: |z| ≥ 2.2 on 5-min VWAP-distance.
   6. **Submit IOC marketable-limit + bracket**. If IOC unfilled in 2s → log "missed" and move on.

### Expected backtest result

- Daily entries: 163 → **~35** (20 Lane-2 + 15 Lane-1, after 60-70% IOC fill rate on ~55 attempted)
- Block rate vs today: **~78%**
- Predicted Δ P&L vs today on bar-close basis: **+$300 to +$600** (smaller absolute edge, but no longer paying the slippage tax)
- Predicted Δ P&L at 8 bps slippage: **+$100 to +$400** (the whole point: my P&L curve is now nearly *flat* in slippage because I'm providing, not taking)
- Confidence: **med-high** on the execution-cost direction, **low** on absolute $ — I am confident this loses *less*, not confident it makes money.

### Steelman against my own approach (1-2 sentences)

If my 60-70% IOC fill rate assumption is wrong on Alpaca paper (which simulates fills optimistically using NBBO crosses, not real queue priority), I'll get filled on 95% of orders effectively as marketable orders anyway and the spread gate is the only thing actually helping — meaning I've over-engineered the order type when I should have just cut frequency in half and used markets. Worse: 35 entries/day may be too few for any meaningful sample, so even a winning strategy looks like noise for the first week and gets killed by the kill-switch before it can prove itself.

### My critique of any signal-focused specialist (1 sentence)

Any specialist proposing "better entry signals" without first nailing order type is rearranging the labels on the deck chairs — the simulator audit shows the *same signals* are +$496 at bar-close and −$3,443 at fill, so the signal P&L space and the execution P&L space are different problems and execution is the binding constraint by a 7× margin.

### Files I'd write

- **`data/strategy_specs/exec_v1/lane1_spy_qqq.json`**: spec for Lane-1, `order_type: "bracket"`, `entry_order: {type: "limit", tif: "ioc", offset_bps: 1, side_aware: true, cancel_after_s: 2}`, `bracket: {tp_bps: 18, sl_bps: 12, sl_type: "stop_limit", sl_limit_offset_bps: 3}`, symbol whitelist `["SPY","QQQ"]`, spread_gate_bps: 3, time_blackout: ["09:30-09:35","15:55-16:00"].
- **`data/strategy_specs/exec_v1/lane2_aapl.json`**: same machinery, symbol `["AAPL"]`, max_entries: 20.
- **`scripts/exec_helpers/marketable_limit_bracket.py`**: helper that takes a signal and emits the Alpaca `submit_order` payload with `order_class="bracket"`, IOC limit entry, stop_limit SL leg, and the child-order fragmentation logic for size>$10k.
- **`experiments/ml_lab/exec_slippage_attribution.py`**: post-hoc script that joins fill prices to NBBO snapshots and reports realized slippage per order type — so tomorrow's autopsy can tell us if marketable-limit-IOC actually saved bps vs market orders.
