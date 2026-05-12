## Specialist: PAIRS

### Core thesis (1 sentence)

Single-name directional HFT got destroyed by 12 bps slippage because each trade pays full directional risk against 6 bps cost; switch both lanes to dollar-neutral cointegrated pairs trading z-score spread reversion on SPY/QQQ, AMD/NVDA, AAPL/META — mean-reversion of the spread is the most reliable structural alpha left after the ML lab's signal review and net beta exposure collapses, shrinking the impact of slippage relative to expected edge.

### Strategy decisions

1. **trade_frequency**: 12–18 pair entries/lane/day (24–36 leg executions/lane). One pair trade = 2 simultaneous legs. Down from baseline 80 single-name entries → ~85 % reduction in directional notional touches.
2. **order_type**: `marketable_limit` with +3 bps offset per leg, IOC. Both legs fired in parallel with a 250 ms hedge-completion deadline; if only one leg fills, immediately flatten the orphan at market (one-sided pairs is the worst failure mode). Bracket-style server-side OCO on the **spread**, not each leg — implemented client-side by monitoring z-score and sending paired close orders.
3. **position_sizing**: Dollar-neutral. Each pair leg = $15k notional (so $30k gross / pair, $0 net). Hard cap 3 concurrent pair positions per lane → $90k gross, $0 net. Per-leg size scales inversely with realized 5-min vol: `notional = 15000 * clip(0.6, 1.0, median_vol_5m / current_vol_5m)`.
4. **entry_signals**:
   - Compute rolling 60-min z-score of log-ratio `log(P_A / P_B)` per pair.
   - **Entry**: |z| > 2.0 AND z crossed the threshold within last 2 minutes AND realized correlation(A,B) over trailing 30 min > 0.55 (avoid regime breaks).
   - **No-entry gate**: skip if either leg's 5-min ATR > 1.5× its 21-day median (vol blowout = cointegration usually breaks first).
   - **ML/AE gate**: pass entry through autoencoder anomaly score → block if percentile > 95 (regime anomaly = pair likely broken, not mean-reverting). LogReg regime model gates: only enter when LogReg "calm" prob > 0.55.
5. **exit_rules**:
   - **TP**: spread z-score returns to |z| < 0.4 → close both legs.
   - **SL**: |z| > 3.25 AND spread still widening 2 min after entry → close both legs (stop = pair broke, not mean reverted).
   - **Time exit**: 75 min max hold (covers documented 30–90 min mean-reversion half-life). Force close 15:45 ET regardless.
   - **ML kill**: if intraday AE anomaly rolling 15-min mean > 0.80 → kill switch lane for the day. If LogReg "stress" prob > 0.70 → halt new entries, hold open positions to time exit.
6. **symbol_universe**: SPY/QQQ, AMD/NVDA, AAPL/META. Three pairs only. **Drop TSLA entirely** — no cointegrated partner in the 7-symbol set (it's the worst loss generator today, idiosyncratic news risk wrecks pairs). Drop standalone single-name trades.
7. **lane_split**:
   - **Lane-1 (Index pair)**: SPY/QQQ only, tighter spreads, larger size ($25k/leg), entry z > 1.8, looser ML gate. Goal: capture small-but-reliable index basis dislocations.
   - **Lane-2 (Sector pairs)**: AMD/NVDA + AAPL/META, $15k/leg, entry z > 2.2, full ML cascade. Higher z threshold compensates for wider single-name spreads.
8. **filter_cascade** (ordered, short-circuit on first fail):
   1. Time gate (skip 09:30–09:45, 15:50–16:00, FOMC days)
   2. Cointegration freshness (re-test Engle-Granger p<0.05 on trailing 5 trading days, weekly recalc)
   3. Realized correlation gate (30-min rolling corr > 0.55)
   4. Vol blowout gate (ATR ratio < 1.5×)
   5. LogReg regime gate (calm prob > 0.55)
   6. Autoencoder anomaly gate (percentile < 95)
   7. Z-score threshold + freshness (|z|>2.0 within last 2 min)
   8. Spread-direction confirmation (z derivative sign matches expected reversion direction)

### Expected backtest result

- Daily entries: 163 → **~30 pair entries combined (60 leg fills)**
- Block rate vs today: **~82 %** of yesterday's directional signals never fire
- Predicted Δ P&L vs today on bar-close basis: **+$1,400** (recovers most of $3,939 slippage gap by halving notional-touched, keeps small positive edge from spread reversion)
- Predicted Δ P&L at 8 bps slippage: **−$200 to +$900** (net-neutral structure cuts effective slippage cost ~30 % via natural cross-leg hedge)
- Confidence: **medium**. Pairs alpha is well-documented but the 21-day cointegration window is thin and intraday pair half-lives are unstable post-2024.

### Steelman against my own approach (1-2 sentences)

Pairs trading on 7 symbols with 21 days of history is dangerously under-tested — Engle-Granger p<0.05 on small samples produces ~5 % false positives, and any single regime break (Fed surprise, sector earnings) sends z-score to 4σ with no reversion, blowing through stops with double slippage on both legs. The "natural hedge" argument also evaporates if SPY/QQQ correlation collapses below 0.4 intraday, which happened 11 % of trading days in 2025.

### My critique of MOMO approach (1 sentence)

A momentum/trend specialist will recommend chasing TSLA/AMD breakouts with tighter trailing stops, but today's data shows TSLA+AMD generated 113 of 163 entries and were the **biggest loss generators** at 12 bps slippage — adding more directional speed to those names is exactly the trade the slippage math has already disproved, regardless of signal quality.

### Files I'd write

- `data/strategy_specs/pairs/spy_qqq.json`: pair definition, z-thresholds (entry=1.8, exit=0.4, stop=3.25), 60-min lookback, $25k/leg, IOC marketable limit +3 bps, Lane-1 binding.
- `data/strategy_specs/pairs/sector_pairs.json`: AMD/NVDA + AAPL/META multi-pair config, z=2.2 entry, $15k/leg, full ML cascade, Lane-2 binding.
- `scripts/alpaca_pairs_host.py`: paired-leg execution engine — submits both legs as IOC marketable_limit in parallel via asyncio.gather, 250 ms orphan-flatten timeout, rolling z-score computation from minute bars with 5-second tick refresh, kill-switch hooks reading AE/LogReg scores from `experiments/ml_lab/live_scores.json`.
- `experiments/ml_lab/pairs_cointegration_check.py`: weekly cron — re-runs Engle-Granger + Johansen tests on trailing 21 days, refreshes pair eligibility, writes results to `data/pairs_eligibility.json` consumed at host startup.
