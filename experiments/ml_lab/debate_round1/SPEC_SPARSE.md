## Specialist: SPARSE

### Core thesis (1 sentence)

Slippage is a per-trade tax of ~12 bps round-trip — the ONLY mathematically guaranteed path to profitability tomorrow is to slash trade count by ~95% (≤5 entries/lane/day) and demand 3+ independent signal confluence so the trades we DO take carry enough information edge to overcome the structural cost.

### Strategy decisions

1. **trade_frequency**: **≤5 entries per lane per day** (hard cap 6). Combined 2 lanes → ≤10 trades/day vs today's 163. Volume cut: **94%**. Slippage cost cut: ~$3,700 → ~$220.

2. **order_type**: **Marketable limit, offset −2 bps from quote mid** (buy at mid − 2bps, sell at mid + 2bps). 10s timeout, then cancel — no fill is better than a bad fill. Reject market orders entirely. No bracket (we manage exits manually with ML kill).

3. **position_sizing**: **Adaptive, conviction-weighted**. Base = $25k notional. Multiplier = (confluence_score / 3) × (AE_precision_decile / 10). Cap $40k per name, $60k gross per lane. Fewer trades + bigger size on the BEST signals = same gross exposure as today but ~6× information density per dollar deployed.

4. **entry_signals**: Require **3-of-4 confluence**, all true simultaneously:
   - **momentum_3min** abs return ≥ 0.35% AND aligned with 15-min trend (signs match)
   - **AE_anomaly_score** in top 1% (precision 68.7% per RESULTS.md) — this is the ML edge
   - **range_pct > p70** of trailing 60-min (Opus-9's filter, NOT range_pct + vov double-count)
   - **LogReg regime probability > 0.65** (AUC 0.799, clean overfit gate)
   
   Fire only if score ≥3. Today's 163 entries would yield ~**8-12 trades** total (counterfactual: LogReg alone blocked 68.5% → 51 trades; adding AE top-1% + range_pct p70 collapses to <10).

5. **exit_rules**:
   - **TP: +0.45%** (vs ~today's tight TP that hit 24.7%) — wider so we earn more per win
   - **SL: −0.25%** (asymmetric 1.8:1 reward/risk — sparse trades deserve room)
   - **Time exit: 45 minutes** (NOT 5 min — give thesis time to play out)
   - **ML kill**: if AE_score decays below median during hold, exit at next bar
   - **Hard kill**: if lane drawdown hits −$800 intraday → halt new entries, hold existing to time exit

6. **symbol_universe**: **SPY, QQQ, NVDA, META** only. **Drop TSLA, AMD, AAPL.**
   - TSLA + AMD = 69% of today's entries AND the biggest loss generators — their spreads + volatility punish slippage hardest
   - SPY/QQQ have tightest spreads (per context) — sparse strategy NEEDS tight spreads
   - NVDA + META: liquid mega-caps with cleaner trend behavior; NVDA had only 21 entries today (the filters already self-selected better setups)
   - AAPL: too few signals historically to justify, drop for capacity reasons

7. **lane_split**:
   - **Lane-1**: SPY + QQQ only, momentum confluence, ≤5 entries
   - **Lane-2**: NVDA + META only, momentum confluence, ≤5 entries
   - Same signal logic, different symbol pools → **decorrelates lane P&L** without strategy-diversification risk (one bad strategy spec sinks both lanes)

8. **filter_cascade** (ordered, fail-fast — cheapest checks first):
   1. **time_gate**: skip first 15 min (10:00 ET open in US futures) and last 10 min (15:50–16:00 ET close) — open is noise, close is gamma
   2. **news_blackout**: ±5 min around any flagged event
   3. **correlation_check**: skip if QQQ-NVDA 5-min correlation >0.9 (no double-exposure)
   4. **range_pct > p70 gate** (rule-based, $0 cost)
   5. **momentum + trend alignment** (rule-based)
   6. **LogReg regime gate** (cheap inference, AUC 0.799)
   7. **AE anomaly score top 1%** (most expensive, last — only run on survivors)
   8. **Final confluence vote**: require ≥3 of the 4 signal layers active

### Expected backtest result

- Daily entries: 163 → **~10** (5 per lane × 2 lanes)
- Block rate vs today: **~94%**
- Predicted Δ P&L vs today on bar-close basis: **+$200 to +$600** (we lose some of today's bar-close +$496 by skipping 95% of trades, but the trades we keep are the AE/LogReg-survived ones which were positive in counterfactual)
- Predicted Δ P&L at 8 bps slippage: **+$1,500 to +$3,200** (was −$3,443; slippage tax falls from ~$3,700 to ~$220 → net swing of ~$3,500 from slippage alone, plus selection edge from confluence)
- **Confidence: med-high** (AE precision@top1% of 68.7% is real; the math of slippage reduction is deterministic; the risk is that with N=10 trades, one-day variance dominates)

### Steelman against my own approach (1-2 sentences)

With only 5 entries/lane, a single −0.25% stop-loss day across 3 names can still produce −$300 to −$500 — sample size is too small to demonstrate edge in one session, and a Bayesian observer would correctly note we're trading a wide CI prior for a narrow CI prior on a single sample. The strategy is also fragile to "no signal day": if confluence never fires, we earn $0 while paying opportunity cost and infrastructure overhead — which optically looks like a kill-switch even though it's working as designed.

### My critique of DENSE approach (1 sentence)

If DENSE proposes increasing trade count to "harvest more signal" or to "let the law of large numbers smooth slippage," that is exactly backwards — slippage is a deterministic tax that scales linearly with trade count, while signal edge has to fight that tax on every fill, so more trades at unchanged edge guarantees more loss, not less.

### Files I'd write

- **`data/strategy_specs/baseline/sparse_lane1.json`**: signal_modes=["momentum"], confluence_required=3, max_entries_per_day=5, symbols=["SPY","QQQ"], tp_pct=0.0045, sl_pct=0.0025, time_exit_min=45, order_type="marketable_limit", limit_offset_bps=2, ae_score_pctile_min=99, logreg_prob_min=0.65, range_pct_pctile_min=70, lane_dd_kill=-800

- **`data/strategy_specs/baseline/sparse_lane2.json`**: same as lane1 but symbols=["NVDA","META"]

- **`scripts/sparse_confluence_filter.py`**: standalone filter module that runs the 8-step cascade on each candidate bar; returns (proceed: bool, score: int, reasons: list[str]); imports AE + LogReg artifacts from `experiments/ml_lab/`; designed to be wrapped by `alpaca_max_hft_host.py` as a pre-trade gate

- **`experiments/ml_lab/debate_round1/SPARSE_backtest.py`**: replay today's 163 entries through the 8-step cascade and confluence vote; report (entries_taken, entries_blocked, simulated_pnl_bar_close, simulated_pnl_at_8bps, simulated_pnl_at_12bps); validates the ~10-trade prediction before tomorrow's open
