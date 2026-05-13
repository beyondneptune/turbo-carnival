# Iter 2 — ES Futures Intraday Order-Flow Imbalance

## 1. Thesis (one sentence)
Signed market-order volume aggregated over a rolling 5-minute window on the E-mini S&P 500 front-month futures contract (ES) predicts the direction of the next 1-minute return with a tradable edge, because large aggressive prints temporarily push price beyond fundamental value and a reliable mean-reversion / continuation pattern resolves within 1-5 minutes; the claim is falsified if walk-forward OOS information coefficient (Spearman rank correlation between OFI and next-bar return) falls below 0.03 across two consecutive monthly folds.

## 2. Market / asset class / venue
- **Instrument:** CME E-mini S&P 500 futures, front-month contract (ESZ5, ESH6, ESM6, ESU6 with quarterly roll on the 8th calendar day before expiry).
- **Ticker symbols:** `ES` continuous front-month via IBKR; roll explicitly on a fixed calendar rule.
- **Primary exchange:** CME Globex.
- **Broker:** Interactive Brokers (IBKR Pro) via `ib_insync`.
- **Settlement:** Cash-settled futures (mark-to-market daily, no PDT regime, Section 1256 60/40 tax treatment in US — long-term/short-term blended).
- **Margin:** Intraday day-trade margin at IBKR for ES is ~ $500-$1,000 per contract intraday (varies by broker); overnight initial margin is ~$13,200-$14,500 per contract (CME SPAN). We trade intraday-only and force flat by 15:55 ET to avoid overnight margin loading.
- **Contract multiplier:** 1 point = $50; 1 tick = 0.25 points = **$12.50**.

## 3. Time horizon
**1-minute intraday.** Bars: 1-minute OHLCV plus per-bar signed volume. Holding period 1-5 bars (60-300 seconds).

## 4. Method family and specific algorithm
- **Family:** Rules-based microstructure signal (no ML in the entry decision; ML is reserved for a later filter iteration if the rules version validates).
- **Algorithm name:** `OFI-5m → 1m mean-reversion-or-continuation rule`.
- **Core signal (Order-Flow Imbalance, OFI):**

  For each 1-minute bar `t`, compute signed volume `S_t = V_buy(t) − V_sell(t)`, where buy/sell classification uses the Lee-Ready tick rule on tick data (trades at or above the prevailing mid are buys, at or below are sells; ties broken by the last non-zero tick) — IBKR `reqTickByTickData('AllLast')` plus `reqMktData` BID/ASK gives both sides.
  
  Rolling window: `OFI_t = Σ S_{t-4..t}` (sum of signed volume over last 5 bars).
  
  Normalize to z-score against a 60-minute trailing standard deviation of `OFI`: `z_t = OFI_t / σ_60(OFI)`.

- **Entry rule:**
  - **LONG** if `z_t ≤ −2.0` AND last 1m return is negative AND bar volume > 1.5× 30-bar median (confirming the imbalance is from real prints, not a sleepy tape).
  - **SHORT** if `z_t ≥ +2.0` AND last 1m return is positive AND bar volume > 1.5× 30-bar median.
  - Direction is **counter-flow** (mean reversion). This matches Cont/Kukanov/Stoikov (2014) finding that large transient OFI overshoots revert at the 1-5 minute scale, plus the Easley-López de Prado-O'Hara VPIN observation that toxic flow clusters and is followed by normalization.

- **Holding period:** Exit at `t+3` (180 seconds from entry) OR at stop/target, whichever first.

- **Exact stop and target (in ticks, ES tick = $12.50):**
  - **Stop:** 6 ticks against entry = $75 risk per contract.
  - **Target:** 9 ticks in favor = $112.50 reward per contract.
  - **TP:SL ratio:** 1.5:1, asymmetric per Lesson 3.

- **Session restriction (exact):**
  - Trade only during **CME RTH equivalent: 09:35-15:50 ET** on weekdays.
  - **Exclude first 5 minutes** (09:30-09:34:59 ET) — opening auction noise distorts the OFI z-score.
  - **Exclude last 10 minutes** of RTH (15:50-16:00 ET) — closing auction imbalance and forced liquidation distort microstructure.
  - Optional Phase-2 extension to Globex overnight (18:00-09:25 ET) only after RTH version validates; not in scope here.
  - No new entries between 14:00-14:05 ET on FOMC days (algorithmic skip on the published FOMC calendar).

- **Why this method fits this edge:** The edge is microstructural and decays in seconds-to-minutes. ML would overfit a thin, regime-dependent signal; a transparent z-score threshold is auditable, fails loudly, and matches the academic anchor's functional form (Cont-Kukanov-Stoikov treat OFI linearly; we discretize via z-threshold).

## 5. Data requirements
- **Feeds:**
  - IBKR real-time CME Level-1 + tick-by-tick (`AllLast`, `BidAsk`) — requires CME Real-Time subscription ($14/mo IBKR market data fee for non-pro; $105/mo if pro/professional designation triggers).
  - IBKR historical 1-min bars for ES (free via TWS API, limited to ~6 months of intraday history per request — sufficient).
  - Backfill of 3 years of 1-min ES bars + tick data from a vendor: Databento (`GLBX.MDP3` schema, full L2 if needed; for this iteration L1 + signed trades is enough) — Databento ES historical tick data ~$200-$400 one-time depending on date range.
- **Storage estimate:** 1 year of ES 1-min bars ≈ 5 MB. Tick-by-tick trades for ES ≈ ~8-12 GB/year compressed Parquet. Three years tick history ≈ ~30-40 GB. Local SSD adequate.
- **Cost estimate:**
  - IBKR CME real-time data: $14-$105/month.
  - Databento one-time historical: ~$300 amortized.
  - Cloud archival (S3 Glacier of raw ticks): ~$2/month.
  - Total run-rate: **~$20/month after backfill**.
- **Latency requirement:** End-to-end signal-to-order ≤ 800 ms (1-min bars give us seconds of slack; we are NOT competing with HFT). Co-location is unnecessary.

## 6. Validation plan
- **Train/test span:** Use 2022-01-03 through 2025-12-31 (4 years), excluding the front-month roll day each quarter.
- **Walk-forward design:**
  - **Train window:** 252 trading days (~1 year of RTH minute data ≈ ~98,000 1-min bars).
  - **Test window:** 21 trading days (~1 month, ~8,200 bars) held out forward.
  - **Step:** 21 trading days (non-overlapping monthly OOS folds).
  - **Total folds:** 36 monthly OOS folds across the 4-year span.
- **Purge & embargo (Lopez de Prado standard for autocorrelated minute bars):**
  - **Purge:** Drop the last 6 bars (matching max holding + signal lookback) of each training fold to prevent label leakage at the train/test boundary.
  - **Embargo:** Insert a 60-bar embargo gap (~1 hour) between train end and test start to absorb residual serial correlation in OFI.
- **Minimum OOS trade count before verdict:** **≥ 400 trades** across the full walk-forward (target avg ~12 trades/fold × 36 folds). If fewer, sample is too sparse and we extend history backward by another year before deciding.
- **Minimum OOS minute-bar count:** **≥ 250,000 1-min bars** in aggregate test windows.
- **Statistical tests + thresholds:**
  - **Per-trade Sharpe (annualized, costs-in):** > 1.0 with 95% stationary-block-bootstrap CI lower bound > 0.4.
  - **Information coefficient (Spearman of `z_t` vs forward 1-min return):** OOS mean IC > 0.03 with t-stat > 2 across folds.
  - **Profit factor:** > 1.25 net of slippage.
  - **Hit rate × payoff:** hit rate ≥ 42% (given 1.5:1 payoff, breakeven ≈ 40%).
  - **Deflated Sharpe ratio (Bailey-López de Prado):** DSR > 0.6 after adjusting for the number of thresholds and windows scanned (`z` ∈ {1.5, 1.75, 2.0, 2.25, 2.5}, window ∈ {3, 5, 10} bars).

## 7. Risk envelope
- **Account base for sizing:** $100k notional paper-trading account (matches existing turbo-carnival convention).
- **Per-contract margin (IBKR intraday day-trade margin for ES):** ~$500-$1,000 intraday. Use **$1,000 worst-case** for sizing math.
- **Max contracts at full size:** **3 contracts** per trade. Margin load 3 × $1,000 = $3,000 (3% of account) — well under any reasonable cap. Notional exposure per contract at ES ~6000 is 6000 × $50 = $300,000, so 3 contracts = $900k notional, 9× account leverage — entirely standard for intraday futures but we cap there to limit any single-bar adverse move.
- **Per-trade max loss:** 3 contracts × $75 stop = **$225** per trade = 0.225% of account.
- **Daily loss cap (absolute):** **$1,500 = 1.5% of $100k account.** Equivalent to ~6.7 losing trades at full size, or ~20 losing trades at 1-contract size. Hit → flat all positions and stop trading for the day.
- **Drawdown trigger that halts trading:** Trailing 5-business-day drawdown > 3% ($3,000) → pause new entries for 3 business days, manual review required to resume.
- **Circuit breaker:** Halt for the rest of the day after **3 consecutive losses** OR after 2 stop-outs within 15 minutes (flow regime probably hostile).
- **Correlation rule with already-deployed strategies:** This is an intraday futures index strategy. Hermes/Atlas/Aegis are daily/EOD US equity options strategies. Daily-bar return correlation expected ρ < 0.4 (because we close flat each day; only the day's net P&L correlates with broad SPX direction, partially). Hard rule: if 20-day rolling correlation with any deployed strategy > 0.6 → reduce size 50% until correlation falls.

## 8. Execution plan
- **Broker / SDK:** Interactive Brokers via `ib_insync` (Lesson 8 — battle-tested).
- **Order type:** **Marketable limit IOC** (LMT IOC priced 1 tick through the touch). ES is so liquid that a true market order is fine in RTH, but limit-IOC gives us a hard slippage cap. If unfilled, abandon the entry — do not chase.
- **Bid-ask convention:** ES is **typically 1 tick wide ($12.50) during RTH** with sizes in the hundreds on each side. Spread occasionally widens at the open, on news, and into the close — the session restriction already handles two of those.
- **Expected slippage:**
  - **Entry:** 0.5 ticks expected ($6.25/contract) — we cross half the spread on average, since limit-IOC fills near touch.
  - **Exit (stop):** 1.0 ticks adverse ($12.50/contract) — stops trigger on prints and may slip 1 level.
  - **Exit (target):** 0.5 ticks ($6.25/contract) — limit at target.
  - **Round-trip slippage budget:** **1.5-2 ticks ≈ $18.75-$25 per contract** ≈ 25-33% of the per-trade risk. This is healthy by ES intraday standards; if realized slippage exceeds 2.5 ticks round-trip on a 50-trade rolling window we treat that as a kill signal.
  - **Commission (IBKR):** ~$0.85/side/contract = $1.70 round-trip. Already included in cost ledger.
- **Broker rejection handling:**
  - If `placeOrder` returns a reject (margin, contract halt, connection drop): log to `data/exec_reject_log.jsonl`, do NOT retry the same bar (the signal is stale within 60s anyway), wait for next signal.
  - If a fill arrives but reconciliation against `reqPositions()` shows a discrepancy > 0 contracts after 5 seconds: alarm, flatten via emergency `marketOrder(close, all)` against the broker-reported position (Lesson 4 — broker truth, not local cache).

## 9. Capacity
- **AUM ceiling before edge degrades:**
  - At 3-contract max, notional per trade is ~$900k. The OFI signal exploits *transient* imbalances; if our own orders create the imbalance, we destroy our own edge.
  - Rule of thumb (Kyle's lambda territory): keep order size ≤ 1% of average 1-minute volume.
  - ES front-month 1-minute RTH volume averages ~5,000-15,000 contracts. 1% of the slow case (5,000) = 50 contracts/minute. We trade 3. **Headroom is ~16×** at $100k account → strategy scales to roughly **~$1.5-1.6M account before order size approaches the 1%-of-1min-ADV soft limit.**
  - **Stated capacity ceiling: $1.5M AUM at this exact parameterization. Beyond that, expected slippage doubles and edge halves.**
- **ADV of underlying and what % we consume at full size:**
  - ES front-month ADV ≈ 1.5M-2M contracts/day RTH.
  - 3 contracts × ~15 round-trips/day max = 90 contracts/day = **0.005% of ADV.** Negligible.
- **Concentration risk:**
  - Single instrument (ES). Single sector (US large-cap broad index). Single venue (CME Globex).
  - This IS the concentration risk — mitigated only by tiny intraday holding period (no overnight, no event tails) and the daily loss cap.

## 10. Failure modes (≥ 3 named)
1. **Regime change to trending day (e.g. CPI/FOMC surprise, geopolitical shock):** Mean reversion fails; flow continues and stops cascade. **Metric:** consecutive-loss streak ≥ 3 with same-direction signal. **Behavior:** circuit breaker fires; daily halt.
2. **Liquidity flight / spread widening:** Bid-ask widens from 1 tick to 2-4 ticks (typical after a 30+ tick move in 1 minute). **Metric:** rolling 20-bar median bid-ask > 1 tick → suppress new entries until median returns to 1 tick.
3. **OFI signal decay (microstructure regime shift):** HFT participation changes the half-life of OFI mean reversion (e.g., faster reversion makes 1-min bars too slow). **Metric:** trailing 30-day OOS IC slips below 0.02 → enter falsification review, prepare to kill.
4. **Tick-rule misclassification at the open:** Lee-Ready breaks down when many trades occur at the mid; aggressive flow gets mislabeled. **Metric:** ratio of mid-prints to total prints > 25% on a given day → quarantine that day's data, skip new entries for the day.
5. **Broker / data outage:** IBKR TWS disconnect, CME data feed interruption. **Metric:** `reqMktData` heartbeat gap > 10 seconds → flatten all and halt for the session.

## 11. Falsifiable verdict criteria
**Promote to live capital** (all required):
- Walk-forward OOS Sharpe > 1.0 (95% bootstrap CI lower bound > 0.4) on ≥ 400 trades.
- 30 shadow-paper days at IBKR paper account with: realized Sharpe > 0.8, profit factor > 1.2, max DD < 3% of account, zero broker-truth divergence events, average round-trip slippage ≤ 2 ticks.
- IC > 0.03 over last 60 trading days of shadow.
- No more than 1 daily-loss-cap hit during shadow.

**Continue shadow** (any one, none of the kill conditions hit):
- OOS metrics pass historical thresholds but shadow Sharpe sits in [0.4, 0.8].
- Less than 400 OOS trades accumulated (need more time, no smoking gun yet).
- Slippage running between 2.0-2.5 ticks (in-range but watch).

**Kill immediately** (any one — within 90 days, per spec hard rule):
- Shadow 3-day rolling DD > 5% of account.
- Daily loss cap breach.
- Broker-side fill discrepancy > 0.5% of intended quantity on any trade.
- Trailing 30-day OOS IC < 0.03 across two consecutive monthly recomputations.
- Round-trip slippage > 2.5 ticks on a 50-trade window.
- Realized hit rate < 35% over 100 consecutive shadow trades (well below the 40% breakeven).

**Falsification metric (named, triggers ≤ 90 days):** **30-day rolling Spearman IC of `z_t` vs forward 1-min return on shadow trades.** First trips below 0.03 → enter 5-business-day confirmation window. Second consecutive month below 0.03 → strategy is dead. Worst-case time-to-falsify: ~60 shadow days, comfortably under the 90-day spec ceiling.

## 12. One-paragraph operational risk note
The single biggest operational vulnerability is the CME real-time market data subscription at IBKR: if our account gets classified as "professional" (which can happen automatically based on entity registration, employer, or trading volume thresholds), the data fee jumps from $14 to $105 per month and worse, if the subscription lapses for any reason — billing failure, account audit, classification dispute — `reqMktData` returns delayed quotes silently, our OFI z-score becomes garbage but the signal still fires, and we trade on 15-minute-old data without realizing it. Mitigation: add a `data.lastTickTime` sanity check on every bar; if `now - lastTickTime > 3 seconds` during RTH, refuse to compute new signals and emit a loud alert. Secondary ops risks: IBKR TWS/IBG requires daily restart (or the auto-restart cron) and occasional TWS upgrades break the `ib_insync` event loop in subtle ways — pin `ib_insync` and TWS versions, never auto-update mid-run, and freeze the entire dependency tree once shadow trading begins (matches the 90-day cycle freeze rule in the user's project memory).

---

### Lessons compliance audit (all 10)
1. **Validate before deploy.** Section 6 specifies walk-forward + ≥30 shadow days. Compliant.
2. **Binary gates, not confidence rankers.** OFI z-threshold is a binary gate (`|z| ≥ 2.0` AND volume confirmation AND last-bar-direction confirmation), with no confidence-based sizing. Compliant.
3. **TP:SL asymmetry is structural.** Stop 6 ticks ($75), Target 9 ticks ($112.50) = 1.5:1 asymmetry. Hard-coded, not a hyperparameter. Compliant.
4. **Broker truth, not local cache.** Section 8 mandates `reqPositions()` reconciliation post-fill, emergency flatten on > 0 discrepancy. Compliant.
5. **No PDT-exposed lanes on sub-$25k US equity accounts.** **N/A — futures are not subject to FINRA Rule 4210 / PDT.** ES is a CFTC-regulated futures contract; CFTC and CME margin rules apply, not the SEC pattern-day-trader rule. A sub-$25k futures account can intraday-trade without restriction. Lesson explicitly argued exempt.
6. **One strategy per slot.** This is the only ES intraday OFI strategy in the deployment plan; no collision with Hermes/Atlas/Aegis/TERMINATOR (all US equity options). Compliant.
7. **No same-data backtest.** Section 6 is walk-forward with purge and embargo across 36 non-overlapping monthly OOS folds. Compliant.
8. **Battle-tested broker SDKs only.** `ib_insync` at IBKR — explicitly named in the spec's allowed-list. Compliant.
9. **Capacity bound named upfront.** Section 9 states $1.5M AUM ceiling at this parameterization and the reasoning (1% of 1-min ADV). Compliant.
10. **Kill criteria written before deploy.** Section 11 lists six concrete kill conditions with numeric thresholds. Compliant.
