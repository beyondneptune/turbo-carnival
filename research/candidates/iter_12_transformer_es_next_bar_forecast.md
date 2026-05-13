# Iteration 12 — Temporal Fusion Transformer Next-Bar Forecast on ES 5-Minute Bars

## 1. Thesis (one sentence)

A Temporal Fusion Transformer (TFT, Lim/Arik/Loeff/Pfister 2021) — fed a 64-bar (5.3-hour) sequence of 38 features per E-mini S&P 500 (CME: ES) front-month 5-minute bar (OHLCV + microstructure proxies + cross-asset levels: NQ, ZN, CL, DXY, VIX, GC) — produces a next-1-bar return prediction whose **directional signal**, when (a) the absolute predicted return exceeds 1.5× the model's per-sample MC-dropout prediction standard error and (b) the directional sign agrees across ≥8 of 10 MC-dropout forward passes, beats the unconditional next-bar Sharpe by ≥0.4 net of costs over a purged-walk-forward 2019-2026 out-of-sample window; the claim is falsified if, on the OOS test folds, the gated trades' annualized Sharpe net of costs is ≤0.3 OR the binary trade/no-trade gate fails to demonstrate filter power (top-decile-of-confidence has no higher hit-rate than median-of-confidence at p < 0.05) — i.e. it functions as a binary gate but its confidence is *not* a valid ranker, paralleling the C3 finding (r = −0.05 between filter confidence and realized P&L) and forcing equal-size trades behind a binary gate rather than confidence-scaled sizing.

---

## 2. Market / asset class / venue

- **Instrument:** CME E-mini S&P 500 futures, **front-month** (ESZ5, ESH6, ESM6, ESU6 …). Continuous front-month stitched with explicit roll on the 8th calendar day prior to expiry; roll-adjustment is a separate column (we model **unadjusted** intraday returns within a contract and **roll-adjusted** only for the cross-bar features that need long history, marked clearly to avoid look-ahead).
- **Symbol on IBKR:** `ES` with `secType=FUT`, `exchange=CME`, multiplier `$50/point`, tick `0.25` = `$12.50`.
- **Primary exchange:** CME Globex.
- **Broker:** Interactive Brokers (IBKR Pro) via `ib_insync` (per Lesson 8).
- **Settlement / margin:** Cash-settled. IBKR intraday day-trade margin ~$500-$1,000/contract; overnight initial ~$13,200-$14,500/contract (CME SPAN, vol-dependent). We trade **intraday-only**, force flat by 15:55 ET.
- **Tax regime (US):** Section 1256 60/40 long-term/short-term blend.

This iteration shares the **instrument** (ES) with Iter 2 (OFI rules) but the **method**, **time scale**, and **edge source** are different (5-min TFT forecast vs 1-min OFI z-score). Per Lesson 6, the two would be deployed as separate slots only if their realized correlation in shadow run is < 0.6; otherwise this candidate replaces or de-allocates against Iter 2 (see §7 "Correlation rule").

---

## 3. Time horizon

**5-minute intraday.** Input sequence: 64 5-min bars = **5.3 hours** of context (covers a full US RTH session plus an overnight Globex tail). Prediction horizon: **next 1 bar (5 min)** as the primary target; **next 3 bars (15 min)** as a secondary multi-horizon head used only for exit consistency, never for entry (avoids selection on long horizons that the academic anchors over-claim on). Holding period of an actual trade: 1-3 bars, hard exit by bar 4 (20 minutes) or at stop/target, whichever first.

---

## 4. Method family and specific algorithm

### 4.1 Family

ML / NN — sequence transformer. Specifically the **Temporal Fusion Transformer (TFT)** of Lim, Arik, Loeff, Pfister (2021, *International Journal of Forecasting*), with a smaller-footprint **Informer (Zhou et al. 2021, AAAI)** ProbSparse-attention variant as the fall-back if TFT inference latency exceeds the 5-min decision budget (negligible risk at 5-min cadence but kept as a switchable backbone).

### 4.2 Why this method fits *this* edge — the genuine argument

The edge claim is **nonlinear cross-feature interactions** in intraday microstructure that linear models (logistic regression, OLS, LightGBM on raw features) only partially capture. Specific examples:

- **Interactions** like *"OFI surge × VIX expansion × DXY direction"* — a momentum-versus-mean-reversion regime switch where the marginal effect of OFI on next-bar return flips sign as VIX crosses ~20 *and* DXY is rising. A linear or tree model with depth ≤6 misses the three-way conjunction; an attention block in a transformer can learn it.
- **Time-varying lag structure**: the lead-lag between NQ and ES varies intraday (NQ leads in the US morning when MAG-7 names dominate flow, ES leads after 14:00 ET when index-rebalance and macro flow take over). TFT's *variable selection network* + temporal multi-head attention can express that lag time-modulation explicitly; LightGBM cannot natively.
- **The Lopez de Prado caveat**: most published "transformer beats LightGBM on prices" results have weak walk-forward methodology, no embargo, leak features, and don't bootstrap CIs. The thesis here is **deliberately conservative**: we expect transformer edge over LightGBM, **if it exists at all**, to live in the cross-feature/time-varying-lag regime and to be small (Sharpe-add ~0.2-0.5, not 1.5+). If the OOS uplift over a LightGBM baseline is <0.2 Sharpe, we kill the transformer lane on cost-of-complexity grounds, not because the signal is uninvestable.

### 4.3 Specific algorithm — concrete numbers

**Backbone (default):** TFT.
- Sequence length: **64 5-min bars** (5h 20min).
- Static covariates: contract-month encoding, weekday (one-hot), is_FOMC_day, is_NFP_day, is_OPEX_day.
- Time-varying *known* future inputs: bars-into-session, scheduled macro releases in next 60 min (binary), time-to-close.
- Time-varying *observed* past inputs (38 features per bar — explicit list):
  1. log return (close-to-close)
  2. log range (high-low)/close
  3. log volume z-score (rolling 30-bar)
  4. close minus VWAP_session
  5. close minus prior bar VWAP
  6. OFI proxy: (close - open) × volume sign (signed-vol approximation)
  7. trade-count z-score (if tick-by-tick available; else 0)
  8. realized vol last 12 bars (1h)
  9. realized vol last 78 bars (1 session)
  10. RSI(14) on close
  11. distance to session high (in ATRs)
  12. distance to session low (in ATRs)
  13. distance to prior-day close
  14. distance to prior-day high
  15. distance to prior-day low
  16. NQ 5-min log return (lagged 0)
  17. NQ 5-min log return (lagged 1, i.e. last bar)
  18. NQ-ES return spread (level)
  19. ZN (10y note futures) 5-min log return
  20. ZN-ES correlation, 60-bar rolling
  21. CL (WTI crude) 5-min log return
  22. GC (gold) 5-min log return
  23. DXY (dollar index spot) 5-min log return
  24. DXY level z-score (60-bar)
  25. VIX level
  26. VIX 5-min change
  27. VIX-realized-vol gap (VIX − realized vol annualized)
  28. SKEW index level (daily, broadcast intraday)
  29. SPX cash basis (ES − SPX cash, lagged 1 min to avoid lookahead)
  30. NYSE TICK z-score (60-bar) — if feed available
  31. NYSE Advance-Decline z-score — if feed available
  32. Day-of-week sin/cos (2 features)
  33. Time-of-day sin/cos (2 features)
  34. Bars since last 1% intraday move
  35. Cumulative session signed volume
  36. Microstructure: |return| / volume (Amihud illiquidity per bar)
  37. Microstructure: rolling spread proxy (high − low + |close − open|) / close
  38. Microstructure: Roll's effective-spread proxy

Targets (multi-horizon training, single-horizon trading):
- y₁ = next-bar log return (primary, used for trading).
- y₂ = next-3-bar log return (auxiliary, regularizer only).
- Quantile heads: 0.1, 0.5, 0.9 (TFT default). The 0.5 head is the point forecast; the (0.9 − 0.1) gap is one of the uncertainty signals (cross-checked against MC-dropout SE).

Hyperparameters (frozen before any OOS evaluation; chosen on **2019-2020 training data only** with a 2018 dev split):
- Hidden size: 64 (small, deliberately).
- Attention heads: 4.
- Dropout: **0.2** (high, because dropout is also our MC-dropout uncertainty mechanism).
- Encoder/decoder layers: 1 each.
- Batch size: 128.
- Optimizer: AdamW, lr 1e-3 with cosine decay.
- Early stopping on validation quantile loss, patience 5 epochs, max 50 epochs.
- Loss: TFT quantile loss across {0.1, 0.5, 0.9} × {1-bar, 3-bar} = 6-component sum, equal weighted.

**Inference at decision time (every 5-min bar close):**
1. Build the 64×38 input window (only data ≤ bar close T).
2. Forward-pass with dropout **enabled** (MC-dropout, Gal & Ghahramani 2016): **10 stochastic forward passes**.
3. Collect 10 point forecasts ŷ₁,…,ŷ₁₀ for next-bar return.
4. Compute predicted mean μ̂ = mean(ŷᵢ), prediction SE σ̂ = std(ŷᵢ), directional vote d̂ = #{ŷᵢ > 0}/10.
5. **Trade gate (binary):**
   - LONG if μ̂ > 1.5 × σ̂ AND d̂ ≥ 0.8 AND |μ̂| > slippage_budget_in_bps AND session/regime gates pass (see §4.4).
   - SHORT if −μ̂ > 1.5 × σ̂ AND d̂ ≤ 0.2 AND |μ̂| > slippage_budget AND session/regime gates pass.
   - Else FLAT.
6. **Sizing:** equal-size 1 contract per trade by default; up to 3 contracts only if a *separate* ranking-power validation (§6.3) passes for the confidence score μ̂/σ̂. **Default = fixed size**, per Lesson 2 + C3 finding.

### 4.4 Session / regime gates

- Trade only **09:35-15:50 ET** on weekdays (skip first 5 min of RTH, last 10 min).
- Skip 5 min before and 5 min after scheduled macro releases (FOMC, NFP, CPI) — the model is trained on those bars but we refuse to *trade* them because the OOS signal-to-noise there is dominated by event jumps the model can't see.
- Skip if rolling 60-bar realized vol is in the top 2% of training-set distribution (extreme regime; model out-of-distribution).
- Skip if bar volume < 25% of 30-bar median (sleepy tape).

---

## 5. Data requirements

- **Live feeds (IBKR via `ib_insync`):**
  - ES 5-min bars + tick (CME real-time, $14-$105/mo).
  - NQ 5-min (CME, included in CME bundle).
  - ZN 5-min (CBOT, ~$10/mo addl).
  - CL, GC 5-min (NYMEX/COMEX, ~$10/mo addl).
  - VIX index level (CBOE Global Indices, ~$5/mo).
  - DXY (ICE US Futures or USD index — IBKR `DXY` cash index, $5/mo; alternatively use UUP ETF as a tradable proxy).
  - SKEW (CBOE, daily; free EOD).
- **Historical (one-time):**
  - 2019-01-01 → present 5-min OHLCV for ES, NQ, ZN, CL, GC. Source: Databento `GLBX.MDP3` schema 5-min aggregates ≈ $200-$500 one-time bulk pull.
  - 2019-01-01 → present spot VIX, DXY, SPX cash 5-min from CBOE/Refinitiv equivalent (Polygon Pro daily, or Bloomberg if accessible; for paper-research stage, **Polygon Free 5-min** under 5 rpm with overnight backfill is sufficient — matches the project CLAUDE.md "Polygon Free is allowed, optimize for resumable cache-first jobs").
- **Storage estimate:** ~9 instruments × 5-min bars × 7 years × ~80 bars/day × 252 days = ~1.3M rows × ~50 columns × 8 bytes ≈ **500 MB raw**, ~150 MB compressed Parquet. Trivial.
- **Cost estimate:** ~$40-$130/month run-rate (data subs); one-time ~$300-$500 historical bulk. Cloud cost: $0-$5/mo S3 archival.
- **Latency requirement:** **end-to-end ≤ 8 seconds** from bar close to order submission. The 5-min cadence gives 290 seconds of slack; transformer inference on a 64×38 input is <100 ms on a 2023+ M-series Mac MPS, <30 ms on a discrete GPU. We are not racing HFT; we are racing our own bar-close clock.

---

## 6. Validation plan

This section is the load-bearing critique of the published literature and the answer to Lesson 7.

### 6.1 The critique (anchor papers do this badly, we won't)

Most published "transformer-for-trading" papers — including derivative work citing TFT and Informer — share the following methodological pathologies:

- **No purge / no embargo** between train and test windows on autocorrelated minute/hour bars: a label at time `t` that uses future bars 1..3 ahead overlaps with training samples at `t+1, t+2, t+3` if no purge is applied → leakage.
- **No walk-forward**: single train/test split is selection-biased.
- **Feature leakage**: aggregating same-bar features (e.g. VWAP including the trade-bar itself) as predictors of that bar's close.
- **No deflated Sharpe**: hyperparameter sweep across dozens of architectures, no Bonferroni / DSR correction.
- **Reported Sharpe with no slippage / cost model**: removes the binding constraint.
- **Confidence used to size positions without separately validating ranking power**: directly violates Lesson 2 (C3 lesson).

This iteration explicitly avoids each one.

### 6.2 Purged walk-forward design (Lopez de Prado, *Advances in Financial Machine Learning*, ch. 7)

- **Span:** 2019-01-01 to 2026-05-12 (~7.4 years). 2019-2020 is hyperparameter tuning + dev only; **2021-01-01 onward is OOS**.
- **Walk-forward (expanding train, fixed test, quarterly retrain):**
  - Initial train window: 2019-01-01 → 2020-12-31 (2 years ≈ ~40,000 5-min bars).
  - First test fold: 2021-Q1 (Jan-Mar 2021, ~5,500 bars).
  - Step: each subsequent fold *expands* the training window by one quarter and tests the next quarter.
  - Quarterly retrain (every 3 months) — matches the spec's "quarterly retrain" requirement and bounds compute cost.
  - Total folds: 2021-Q1 through 2026-Q1 ≈ **21 OOS quarterly folds**.
- **Purge:** drop the last `H + L − 1` bars of each train window where `H` = max prediction horizon (3 bars) and `L` = input sequence length (64 bars). Purge = **66 bars** at each train/test boundary.
- **Embargo:** insert a **78-bar (1 trading day) embargo** between the purge boundary and the test fold start. Removes residual serial correlation in 5-min returns (typical AR(1) ≈ 0.05-0.1, decays within ~20 bars; 78 bars is conservative).
- **K-fold check (purged combinatorial cross-validation, de Prado):** within the 2021-2026 OOS span, additionally run purged 5-fold CV on a **rolling-origin** basis as a robustness check — but the *primary* verdict is the walk-forward one. The combinatorial test is reported alongside to surface sensitivity to which year is held out.

### 6.3 The Lesson-2 critical test: binary gate vs ranker, SEPARATELY

This is the load-bearing requirement of the iteration:

- **Test A — binary gate validation (does the gate filter trades well?):**
  - Take the binary signal `gate(t) ∈ {LONG, FLAT, SHORT}`.
  - Compare net-of-cost P&L of *gated trades* vs *all 5-min bars with naive sign-of-prediction*.
  - Threshold: gated trades must beat naive baseline by ≥0.4 annualized Sharpe and have profit factor > 1.15 net of costs.
- **Test B — ranker validation (does confidence rank realized P&L?):**
  - Compute Spearman correlation between `confidence(t) = μ̂(t) / σ̂(t)` and realized next-bar return (signed, conditional on a trade firing).
  - Compare top-decile-by-confidence trade P&L to median-decile trade P&L.
  - Threshold for "confidence is a real ranker": Spearman r ≥ 0.05 with t-stat > 2 AND top-decile mean P&L > median-decile mean P&L at bootstrap-CI 95% lower bound > 0.
- **Decision rule (matches C3 lesson):**
  - If A passes and B passes → confidence-scaled sizing allowed, capped at 3× base size at top decile.
  - If A passes and B fails → **deploy as binary gate only, fixed 1-contract sizing**, exactly as the C3 case in the user's prior work (filter saved $2,029 of losses; its confidence had r = −0.05 with P&L, so we did NOT size by it).
  - If A fails → kill the iteration regardless of B.

### 6.4 Minimum OOS volume

- Minimum OOS gated trades before verdict: **≥ 600 trades**. Expected trade rate ≈ 3-6 per RTH day × ~252 days × 5 years OOS ≈ 5,000-7,500 candidate bars passing the gate — well above 600.
- Minimum OOS 5-min bars: **≥ 80,000** in test windows aggregate.

### 6.5 Statistical thresholds

- Annualized Sharpe (net of costs) > **0.8** with stationary-block-bootstrap 95% CI lower bound > **0.3**.
- Deflated Sharpe ratio (Bailey & López de Prado 2014) > **0.6** after adjusting for ~8 architecture variants tried in dev (hidden size ∈ {32,64,128}; layers ∈ {1,2}; horizon scheme variants).
- LightGBM-baseline uplift: **TFT Sharpe − LightGBM Sharpe ≥ 0.2** on the same gated test design. If TFT does not beat LightGBM by 0.2 Sharpe, **the iteration is killed on cost-of-complexity grounds** even if TFT alone is profitable — the transformer's only justification is the nonlinear-interaction edge, and a tree model with cross-feature splits captures most of it.
- Information coefficient: Spearman(μ̂, realized next-bar return) > 0.04 across folds, with t-stat > 2.

### 6.6 What 90 days of forward shadow CAN and CANNOT falsify

- **17,000 OOS 5-min bars in 90 days** (78 RTH bars/day × ~65 RTH trading days × ~3.3 sessions/business-day inclusion of overnight gate-eligible bars when included, but we trade RTH-only → realistically ~5,000 RTH bars).
- Gate firing rate target ~3-6%, → **~150-300 trades in 90 days**.
- With 150-300 trades and per-trade Sharpe-equivalent of a 5-min return distribution: **90 days can statistically distinguish Sharpe = 1.2 from Sharpe = 0.0 with ~80% power** (rough power calc: SE of annualized Sharpe ≈ √(1/N_trades_per_year) × annualized factor ≈ 0.4 for N≈300 in 90 days). **It CANNOT** statistically distinguish Sharpe 0.6 from 1.0 in 90 days — that needs ~12-18 months of forward data.
- **What 90 days falsifies cleanly:** zero-or-negative edge (kill), broker-side latency exceeding spec, model-drift KL > threshold, slippage exceeding budget, daily-loss-cap breaches.
- **What 90 days does NOT decide:** whether realized Sharpe is 0.7 vs 1.2 — needs longer. We continue shadow rather than promote in that ambiguous regime.

---

## 7. Risk envelope

- **Account base:** $100k paper-trading notional (consistent with project convention).
- **Per-trade size (default):** **1 contract** behind the binary gate. Per Lesson 2 + C3 finding, do not size by transformer confidence unless §6.3 Test B passes OOS.
- **Maximum size if Test B passes:** 3 contracts on top-decile confidence; ramp 1→2→3 across deciles 7,8,9-10. Hard cap 3.
- **Stop / target (per Lesson 3, structural asymmetric):**
  - **Stop:** 8 ticks against entry = $100/contract risk.
  - **Target:** 12 ticks in favor = $150/contract reward. **TP:SL = 1.5:1.**
  - **Time stop:** 3 bars (15 min) — exit at market on bar close if neither stop nor target has hit.
  - Asymmetry is hard-coded; the transformer chooses entries, **not** stop/target levels.
- **Per-trade max loss:** 1 contract × $100 = **$100** (0.1% of account). At max 3 contracts, $300 (0.3%).
- **Daily loss cap:** **$1,200 = 1.2% of $100k.** Equivalent to ~12 losers at 1-contract or ~4 at 3-contract size.
- **Drawdown trigger:** trailing 5-business-day DD > 3% → 3-business-day pause + manual review.
- **Circuit breaker:** halt for the rest of the day after **3 consecutive losses** OR after 2 stop-outs in 30 minutes.
- **Correlation rule with already-deployed strategies:**
  - With Iter 2 (ES 1-min OFI rules): both trade ES. Realized correlation cap **ρ_5d < 0.6**. If exceeded, halve size and prioritize whichever has higher 90-day shadow Sharpe.
  - With Iter 5 (VIX term-structure): expected ρ < 0.4 (different time scale, different mechanism). Monitor.
  - General rule: any 20-day rolling correlation with deployed strategy > 0.6 → reduce size 50%.

---

## 8. Execution plan

- **SDK:** `ib_insync` (or its maintained fork `ib_async`).
- **Order type:** **Marketable-limit IOC**, priced 1 tick through the touch. ES is typically 1 tick wide in RTH; cross half the spread on average. If IOC unfilled, **abandon entry** — do not chase.
- **Bid-ask convention:** 1 tick = $12.50 typical RTH spread.
- **Expected slippage:**
  - Entry: 0.5 ticks = $6.25/contract.
  - Stop exit: 1.0 tick = $12.50/contract.
  - Target exit: 0.5 tick = $6.25/contract.
  - Round-trip: **~1.5-2 ticks ≈ $18.75-$25/contract**, ≈ 19-25% of $100 per-trade risk. Healthy.
- **Commission:** ~$0.85/side × 2 = $1.70/contract round-trip. Bundled into cost model.
- **Broker-rejection handling:**
  - Reject → log to `data/exec_reject_log.jsonl`, skip this bar (signal stale in 5 min), do not retry.
  - Post-fill reconciliation against `reqPositions()`: any discrepancy > 0 contracts after 5 seconds → emergency `marketOrder(close)` against broker-reported position (Lesson 4). Local cache is *never* the source of truth for closing orders.
- **Inference path:** model runs in a Python worker process; bar-close trigger from `ib_insync` 5-min bar updateEvent → feature builder reads last 64 bars from local Parquet ring buffer (updated on every tick) → torch `model.eval()` with `dropout.train()` for MC-dropout → 10 forward passes batched → gate evaluation → order placement. Total budget 8 seconds; measured target <2 seconds.

---

## 9. Capacity

- **AUM ceiling before edge degrades:**
  - At 3-contract max, notional per trade is ~$900k.
  - The edge is "nonlinear feature interactions in microstructure" — not "my own order flow." Our own order presence into ES (which trades ~$300B notional/day) at 3 contracts × ~5 trades/hour is **vanishingly small**, but the *information* in the features (especially the OFI proxy and lead-lag with NQ) starts decaying if our peers in the same trade aggregate to >0.5% of 1-min ADV.
  - Kyle's-lambda rule of thumb: order size ≤ 0.5% of average 5-min volume. ES 5-min RTH volume ≈ 25,000-75,000 contracts. 0.5% of 25,000 = 125 contracts/5min. We use 3 → **headroom ~40×** at $100k account.
  - **Stated capacity ceiling: $4M AUM** at this parameterization (3 contracts × ~40× headroom × ~$100k base ≈ $12M raw, halved to $6M for safety, halved again to $3-4M for the "edge degrades faster than slippage" effect specific to feature-based microstructure signals).
- **ADV consumption at full size:**
  - ES front-month ADV ≈ 1.5M-2M contracts/day RTH.
  - 3 contracts × ~6 round-trips/day = 36 contracts/day = **~0.002% of ADV.** Negligible.
- **Concentration risk:**
  - Single instrument (ES). Single sector (US large-cap). Single venue (CME Globex).
  - Mitigated by intraday-only holding period, daily loss cap, and the correlation rule against Iter 2.

---

## 10. Failure modes (≥ 3 named, in metric-language)

1. **Overfit to a single regime (e.g. 2020-2021 vol regime baked into hyperparameters):**
   - Symptom: walk-forward Sharpe degrades sharply quarter-over-quarter starting Q2 2022 (post-COVID regime ends).
   - Metric: rolling 60-day OOS Sharpe falls by > 0.5 between consecutive quarters.
   - Behavior: trigger immediate retrain on the most recent 2-year window; if regrade still fails, kill.

2. **Feature leakage (look-ahead bias from same-bar features or cross-asset clock skew):**
   - Symptom: IS Sharpe >> OOS Sharpe by an implausible margin (e.g. 4× gap).
   - Metric: ratio IS_Sharpe / OOS_Sharpe > 2.5 on first OOS fold.
   - Behavior: forensic audit of feature timestamps; standard suspects are VWAP_session (must exclude current bar), VIX cross-feed clock skew (must lag by 1 min minimum), SPX cash basis (must lag).

3. **Model staleness when microstructure shifts (CME algo update, new exchange tier, dominant HFT firm exits/enters):**
   - Symptom: prediction distribution drifts; the trades that fire are increasingly small-edge / random.
   - Metric (this is the formal drift detector): **Kullback-Leibler divergence between (a) the rolling 30-day live prediction distribution and (b) the training-set prediction distribution** computed in 10 bins of standardized μ̂. If **KL_30d > 0.20 nats**, flag drift; if **KL_30d > 0.40 nats**, force retrain. Computed on a daily 17:00 ET cron from the prior day's predictions.
   - Behavior: KL > 0.20 → 50% size reduction pending review. KL > 0.40 → halt new entries; quarterly retrain runs out-of-cycle.

4. **MC-dropout uncertainty mis-calibration:**
   - Symptom: σ̂ shrinks while realized error widens — model is "overconfident."
   - Metric: realized RMSE / mean(σ̂) over rolling 200 trades > 1.5 (target ≈ 1.0 by Gal-Ghahramani Bayesian interpretation).
   - Behavior: raise the gate threshold from 1.5σ̂ to 2.0σ̂ until ratio normalizes.

5. **GPU / inference latency spike:**
   - Symptom: bar-close to order-submission latency exceeds 8s budget.
   - Metric: P95 inference latency > 3s, or any single decision > 8s.
   - Behavior: degrade to Informer (lighter); if still > 3s P95, halt new entries until cause found.

6. **Broker / data outage:**
   - Symptom: `reqMktData` heartbeat gap > 10s during RTH.
   - Behavior: flatten all positions, halt for session, alarm.

7. **Confidence-ranker reversal (the C3 echo):**
   - Symptom: Test B (§6.3) flips from "weak positive ranker" in dev to "negative or zero ranker" OOS.
   - Metric: Spearman(confidence, realized P&L) on rolling 200 trades drops below 0 with t-stat < -1.
   - Behavior: automatically revert to fixed-1-contract sizing within 1 trading day, even if A still passes.

---

## 11. Falsifiable verdict criteria

### Promote to live capital (all required):
- Walk-forward OOS Sharpe > **0.8** with bootstrap 95% CI lower bound > **0.3**, ≥ 600 trades.
- LightGBM-baseline uplift ≥ **0.2 Sharpe** on identical gated design.
- DSR > **0.6**.
- 30-shadow-day forward run passes: realized Sharpe > 0.6, profit factor > 1.15, max DD < 3% of account, zero broker-truth divergence, average slippage ≤ 2 ticks round-trip, KL drift < 0.20 nats throughout, MC-dropout calibration ratio in [0.8, 1.3].
- §6.3 Test A passes (gate filters).
- §6.3 Test B explicitly evaluated; if it fails, deploy as **binary-gate fixed-size** only — not blocked from promotion, but the C3 lesson is enforced in sizing.

### Continue shadow (none of the kill conditions hit, but ambiguous):
- OOS Sharpe in [0.4, 0.8] — needs more data, not promote-worthy yet.
- Sub-600 OOS trades accumulated.
- LightGBM uplift in [0.0, 0.2] — keep both lanes running for comparison.

### Kill immediately (any one — within 90 days, per spec hard rule):
- Shadow 3-day rolling DD > 5% of account.
- Daily loss cap breach.
- Broker-side fill discrepancy > 0.5% of intended quantity on any trade.
- KL drift > 0.40 nats and no retrain has restored signal within 5 business days.
- 30-day rolling OOS IC < 0.02 across two consecutive monthly recomputations.
- §6.3 Test A fails OOS (gate has no filter power).
- LightGBM beats TFT by > 0.0 Sharpe on the same gated design over 90 days — kill the *transformer* lane (deploy LightGBM as a separate, simpler iteration).
- Round-trip slippage > 2.5 ticks over a 50-trade window.

**Falsification metric — single-number summary:** **30-day rolling OOS Sharpe of gated trades, net of costs.** First slip below 0.0 → 5-business-day confirmation window. Second consecutive month below 0.0 → strategy is dead. Worst-case time-to-falsify: ~50-60 shadow days, well inside 90.

---

## 12. One-paragraph operational risk note

The single largest operational vulnerability is **PyTorch / model-loading pipeline fragility across version updates**: a routine `pip install --upgrade torch` between deployments can change `load_state_dict` semantics (parameter renames, BatchNorm running-stats handling, lazy-init layers), causing the live model to load with silently-zeroed weights and produce garbage forecasts that still pass the type/shape sanity checks. Mitigation: pin `torch`, `pytorch-forecasting` (or HF `transformers`), and `numpy` versions exactly in a `requirements-frozen.txt`, freeze them at the moment shadow trading begins (matches the project's 90-day cycle freeze rule in CLAUDE.md), and add a **production sanity test** at every process start that runs the model on a fixed canary input window and asserts the output matches a stored fingerprint within 1e-6 — refuse to start trading if the fingerprint mismatches. Secondary ops risks: (i) **GPU choice trade-off** — local Apple Silicon MPS gives ~30-50ms inference for free but PyTorch MPS has periodic regressions (e.g. dropout on MPS has had silent broken-ness in past releases); cloud GPU (a single A10/L4 spot at ~$0.50-$1.00/hour, ~$200-700/month if always-on) is more reliable but adds latency, vendor risk, and cost — preferred path is **local MPS for decisioning** with a cloud GPU mirror for offline retrain only; (ii) **IBKR CME real-time data subscription** can silently auto-reclassify to "professional" tier and balloon from $14 to $105/month, or worse silently downgrade to delayed quotes — add the same `lastTickTime` sanity check as Iter 2; (iii) **quarterly retrain compute** must complete in < 24 hours on a single L4 GPU or local M-series — write the training loop to be checkpoint-resumable and store hyperparameters/seed alongside weights so reproducibility survives any host migration; (iv) **TWS daily auto-restart** can break the persistent socket — wrap the `ib_insync` connection in a supervised reconnect with bar-recovery from the local Parquet ring buffer so we never miss a 5-min bar to a TWS restart.

---

## Lessons compliance audit (all 10)

1. **Validate before deploy.** §6 walk-forward across 21 OOS quarterly folds + 30-day shadow specified in §11. Compliant.
2. **Binary gates, not confidence rankers.** §6.3 explicitly tests gate and ranker separately; default sizing is fixed-1-contract behind a binary gate; confidence-scaled sizing is permitted ONLY if Test B passes OOS. Cites the C3 finding (r = −0.05 between filter confidence and P&L) and inherits its sizing discipline. Compliant.
3. **TP:SL asymmetry is structural.** §7: stop 8 ticks = $100, target 12 ticks = $150, 1.5:1 hard-coded, not a hyperparameter and not chosen by the transformer. Compliant.
4. **Broker truth, not local cache.** §8: `reqPositions()` reconciliation post-fill; emergency flatten against broker-reported position on any discrepancy. Compliant.
5. **No PDT-exposed lanes on sub-$25k US equity accounts.** **N/A — futures are CFTC-regulated, not subject to FINRA Rule 4210 / PDT.** ES sub-$25k intraday is unrestricted. Lesson explicitly argued exempt.
6. **One strategy per slot.** Asset overlaps Iter 2 (ES) but method and time scale differ (transformer 5-min vs rules-based 1-min OFI). Correlation rule (§7) enforces ρ_5d < 0.6 between the two slots; if violated, halve and prioritize on shadow Sharpe. Compliant.
7. **No same-data backtest.** §6.2 purged walk-forward with 78-bar embargo + 66-bar purge; primary verdict from walk-forward, robustness from purged combinatorial 5-fold CV. Hyperparameters fixed on 2019-2020 dev only; 2021-2026 is genuinely held out. Compliant.
8. **Battle-tested broker SDKs only.** `ib_insync` at IBKR — explicitly named. Compliant.
9. **Capacity bound named upfront.** §9 states $3-4M AUM ceiling with reasoning (0.5% of 5-min ADV, halved for slippage, halved again for microstructure-feature edge decay). Compliant.
10. **Kill criteria written before deploy.** §11 lists 8 concrete kill conditions with numeric thresholds; §6 falsification metric is a single-number 30-day rolling OOS Sharpe with a 50-60 day worst-case time-to-kill. Compliant.

---

## Appendix — Why this iteration is not just "transformer brrr" hype

A short, honest list of what would make us walk away from this iteration *before* live deployment, in order of likelihood:

1. **LightGBM beats TFT** on the same gated design with the same features. Very plausible (40-60% prior) given the small feature count and 5-min cadence. If true, the LightGBM version becomes a separate iteration and this one is shelved.
2. **§6.3 Test A fails:** the gate doesn't filter trades better than the unconditional sign-of-prediction baseline. Plausible (~30% prior). Kill.
3. **§6.3 Test B fails but A passes:** identical to the C3 case in the user's prior work. Deploy as binary gate, fixed size. The C3 case shows this is still a $2,029-saved outcome and is the *expected* mode of operation here.
4. **MC-dropout uncertainty is not calibrated:** the σ̂ from 10 forward passes does not correspond to a real prediction interval. Mitigated by §10 failure mode 4; if uncalibrated, we raise the threshold or fall back to TFT's native quantile gap (0.9 − 0.1).
5. **Drift detector fires constantly:** the KL threshold of 0.20 may be too tight in practice. Tune in dev only; if it fires more than 4× in the 2021-2026 OOS span, raise to 0.30.

If 1-4 all go the wrong way, the iteration is killed. If 1 goes wrong and 2-5 are fine, LightGBM gets its own iteration and this one is parked. The only path to live-capital promotion is: TFT beats LightGBM by ≥ 0.2 Sharpe AND Test A passes AND drift behaves AND shadow confirms — a conjunction we estimate at **15-25% prior probability of clearing on first attempt**. That low prior is the iteration being honest about itself, not a bug.
