# Iteration 7 — DAX Futures Time-Series Momentum (TSMOM)

## 1. Thesis (one sentence)

The sign of the past 1-month, 3-month, and 12-month returns of the DAX futures contract continues to predict the sign of the next-month return with positive expectancy (information coefficient > 0, Sharpe > 0.6 after costs) at the 12-month lookback, and an equal-weighted combination of those three signs, sized to a constant 10% annualized volatility target, will earn ≥1.5% per month gross of fees over a rolling 36-month window — and we will shut it down inside 90 days of live deployment if it does not.

This is the Moskowitz/Ooi/Pedersen (2012, *JFE*) claim, narrowed to a single instrument (DAX) on a single venue (Eurex), with the explicit prediction that DAX is *not* an exception to the cross-asset TSMOM finding. Moskowitz et al. report a t-statistic on 12-month TSMOM regressions of ~5 across 58 instruments with index futures (including DAX) contributing positively at the 12-month horizon. The falsifiable claim here is: that t-stat does not flip to ≤0 on DAX alone across the 2016-2026 out-of-sample window.

---

## 2. Market / asset class / venue

- **Primary instrument:** FDAX — DAX 40 Index Futures, full-size, traded on Eurex (Frankfurt). Ticker on IBKR: `DAX` with secType `FUT`, exchange `EUREX`, multiplier **€25 per index point**.
- **Capital-efficient alternate:** FDXM — Mini-DAX Futures, multiplier **€5 per index point**, 1/5 of FDAX. Used for accounts below ~€150k notional or for fine-grained position sizing.
- **Exchange:** Eurex (Deutsche Börse). The reference order book for DAX futures; ADV in FDAX has averaged ~110k–140k contracts/day over 2020-2025, with daily notional volume on the order of €40-50bn at a 20,000-level index. FDXM ADV ~80k–100k contracts/day.
- **Quarterly contract cycle:** March / June / September / December (H, M, U, Z). We trade the **front month** and **roll 5 business days before expiry** to avoid liquidity decay in expiry week.
- **Broker:** Interactive Brokers, via `ib_insync` (the maintained ib_async fork is acceptable; both are battle-tested per Lesson 8).
- **Settlement:** Cash-settled against the DAX 40 final settlement price. Daily mark-to-market through Eurex Clearing.
- **Margin (Eurex SPAN-equivalent / IBKR overnight, mid-2025 typical):**
  - FDAX initial ~€22,000–€28,000 per contract (varies with vol).
  - FDXM initial ~€4,500–€5,500 per contract.
  - IBKR intraday margin can be ~50% of overnight on Eurex futures during liquid hours.
- **Tax / regulatory:**
  - Trading EU-domiciled exchange-traded derivatives from a US-domiciled IBKR account triggers Section 1256 60/40 treatment in the US (broad-based foreign index futures qualify for 1256 if exchange-traded and CFTC-recognized; DAX futures via Eurex are recognized by the CFTC under Part 30 exemption).
  - No EU FTT on Eurex derivatives at the level relevant to this strategy (FTT in DE has not been enacted on derivatives as of this writing).
  - Eurex data subscription required on IBKR: "Eurex Core (L1)" mandatory, "Eurex L2" recommended for execution sizing — combined ~$13.50/month for non-professional, ~$110/month for professional designation.

---

## 3. Time horizon

- **Signal evaluation:** **hourly**, on the close of each Eurex trading hour. This is finer than the academic monthly cadence and intentionally so — Eurex hours overlap US morning, and we want the signal "armed" before US flow arrives.
- **Rebalance cadence:**
  - **Position sign changes:** acted on at the next hourly bar after a confirmed flip (defined below).
  - **Vol-scaling re-sizing:** **weekly**, every Monday at 09:05 CET, recomputed from the 60-day realized vol.
  - **Roll:** quarterly, 5 business days before front-month expiry.

Bucket per spec: **hourly**.

---

## 4. Method family and specific algorithm

**Family:** Rules-based, time-series momentum. No ML, no NN. The whole point of TSMOM is that the signal is simple enough to be robust to overfitting — the academic test is on 58 instruments with the *same* rules and no per-instrument tuning. Adding an ML layer would defeat the cross-asset robustness argument that the edge rests on.

**Algorithm (deterministic, no free parameters beyond those fixed by Moskowitz et al.):**

Let `r_{t,h}` = log return of the front-month DAX futures contract over the last `h` business days, computed on the rolling daily close (17:30 CET Eurex close).

1. **Three sub-signals:**
   - `s_1 = sign(r_{t, 21})` — 1-month look-back (≈21 trading days).
   - `s_3 = sign(r_{t, 63})` — 3-month.
   - `s_12 = sign(r_{t, 252})` — 12-month.
   Each `s_h ∈ {-1, 0, +1}`, with `0` only on an exact-zero return (essentially never).

2. **Combined signal:**
   - `S_t = (s_1 + s_3 + s_12) / 3`, so `S_t ∈ {-1, -2/3, -1/3, 0, +1/3, +2/3, +1}`.
   - **Confirmation gate:** flip only when the *new* `S_t` differs from the current position sign and the new `s_12` agrees with the new sign of `S_t`. The 12-month look-back is the dominant horizon in Moskowitz et al. and acts as a regime filter against whipsaw.

3. **Volatility target (10% annualized):**
   - `σ_60 = ` annualized realized vol from the last 60 trading days of daily log returns.
   - Target dollar/euro position notional `N* = (0.10 / σ_60) × E`, where `E` is account equity in EUR.
   - Contracts: `n_FDAX = round( S_t × N* / (P × 25) )`, where `P` is the DAX index level.
   - For FDXM substitute multiplier 5.

4. **Position-cap rules** (defensive, not in Moskowitz et al. — added per Lesson 9):
   - Cap `|n|` at the lesser of (a) the vol-target result and (b) `0.20 × E / margin_per_contract`. At €25/point and DAX ≈ 20,000, FDAX notional/contract ≈ €500,000; €100k account → vol-target alone permits at most ~0.2 contracts when σ_60 ≈ 25% — so the FDXM is the actual instrument at this size.

**Why this method for this edge:** The TSMOM edge is *behavioral and structural* (initial under-reaction to news, delayed over-reaction, then reversion past 12 months), not a microstructure or information-asymmetry edge. It does not need ML pattern recognition; it needs (a) the right look-backs, (b) honest vol scaling so single-asset blowups don't dominate a multi-asset framework even when applied to one asset, (c) costs low enough to survive the ~1.0 Sharpe gross. Adding ML on top of a 3-parameter rule that is published, peer-reviewed, and replicated across 58 instruments would be lighting money on fire.

---

## 5. Data requirements

| Feed | Use | Vendor | Cost | Storage |
|---|---|---|---|---|
| FDAX & FDXM continuous front-month, daily OHLC, 1990-present | Backtest signal | IBKR historical (free with subscription); cross-check vs. Stooq free DAX index | $0 (after subscription) | <50 MB |
| FDAX 1-minute bars 2010-present | Hourly signal grid, intraday execution | IBKR via reqHistoricalData (paginate; cap is 1-day per request for 1-min bars on FUT) | $0 | ~3 GB |
| FDAX L1 quotes (live) | Live signal, execution | IBKR Eurex Core L1 subscription | ~$13.50/mo non-pro | ~5 GB/yr live capture |
| FDAX L2 order book (live) | Execution sizing only | IBKR Eurex L2 subscription | ~$12/mo additional | not stored |
| ECB calendar / Bund auction schedule | Pre-event blackout (skip flips within 30 min of ECB rate decisions) | ECB website + IBKR Trader Workstation news | $0 | trivial |

- **Historical option chain or tick L2:** not required for this strategy.
- **Total storage estimate:** ~15 GB on a 5-year horizon. Local SSD, no S3 needed.
- **Total recurring cost:** ~$26/month for non-professional designation. Professional designation triples this but is unavoidable if any client capital touches the account.
- **Latency requirement:**
  - Signal is hourly, so end-to-end latency budget on a fill is **~500 ms** (loose). What matters is *colocation distance to Frankfurt*: IBKR routes Eurex orders through their FFM gateway. A US-based ib_insync client adds ~80-100 ms transatlantic RTT, which is fine for hourly cadence but means we should **not** use this latency for any market-making variant.
  - **Eurex/US overlap:** Eurex T7 trading runs ~08:00–22:00 CET, which is ~02:00–16:00 ET. US morning (09:30–11:30 ET ≈ 15:30–17:30 CET) is *inside* Eurex liquid hours but *also* inside the daily Xetra/Eurex closing auction window (17:30 CET final settlement). The strategy explicitly avoids submitting flips between 17:00–17:35 CET to dodge the settlement auction's spread widening.

---

## 6. Validation plan

**Honoring Lesson 7 (no same-data backtest) and Lesson 1 (validate before deploy):**

- **In-sample window:** **2000-01-01 → 2015-12-31** (16 years, ~4,000 trading days). Used only to (a) confirm the algorithm reproduces Moskowitz et al.'s sign and rough magnitude on DAX, (b) measure realized cost (slippage) in the IBKR fill data we have from 2010 onward.
- **Out-of-sample window:** **2016-01-01 → 2026-04-30** (10 years, ~2,580 trading days, > 5 years per spec). This includes:
  - 2016 Brexit shock
  - 2018 Volpocalypse
  - 2020 COVID crash + recovery
  - 2022 EU energy crisis / Ukraine invasion
  - 2023 SVB / regional bank stress
- **Walk-forward design inside OOS:**
  - Train (i.e., compute signal parameters that are *already fixed* by the spec, plus measure σ_60) on a 24-month rolling window.
  - Test on the following 6 months.
  - Step 6 months. No purge needed because momentum is path-dependent, not label-leakage-driven; an **embargo of 21 business days** at each train/test boundary handles any residual look-ahead in the 1-month return.
- **Minimum OOS observations:**
  - At an hourly signal evaluation but a position that flips on average ~3-6 times per year in TSMOM literature, the OOS window yields ~30-60 flips and 10 × 52 = 520 weekly re-sizings. Per Lesson 1 we want **≥ 30 OOS flips** before accepting; this window delivers.
  - At a daily-return granularity, OOS = **~2,580 daily P&L observations**, which is the figure quoted as the validation horizon.
- **Statistical test and threshold:**
  - Block bootstrap (block length = 21 days) on daily strategy returns to build a 95% CI for the Sharpe.
  - **Promote criterion:** OOS annualized Sharpe > 0.7 with the lower bound of the 95% bootstrap CI > 0.3 *and* OOS max drawdown < 25%.
  - **Anchor check:** OLS regression of next-month return on `s_12`, t-statistic must be > 1.5 OOS (Moskowitz et al. report ~5 in-sample across 58 instruments; we demand only 1.5 on a single instrument OOS to allow for shrinkage).

**Citation of Moskowitz et al. (2012) anchor finding:** From their Table 1, time-series momentum is significant across all 58 instruments with a pooled regression t-statistic of 5.4 on the 12-month look-back, and they explicitly include equity index futures with DAX as one of the equity index instruments contributing to the panel result (their footnote 4 and Appendix A list the instrument set). They report a diversified TSMOM portfolio Sharpe of ~1.6 gross of fees, ~1.1 net, across 1985-2009. The DAX-only sub-Sharpe is not broken out in their paper but the equity-index sub-Sharpe is ~0.9 net, which is the value we anchor on.

---

## 7. Risk envelope

Assume **base account equity = $100,000 ≈ €92,000** (the operative paper-trade target).

- **Vol target:** 10% annualized. At σ_60 = 18% (typical DAX), target notional = (0.10/0.18) × €92,000 ≈ €51,000. At DAX = 20,000 and FDAX multiplier €25, one FDAX = €500,000 notional → only **0.10 FDAX**, so the live instrument is **FDXM** at multiplier €5 → 0.10 × €500,000 / €100,000 = ~**1 FDXM contract** at typical vol.
- **Max position size:**
  - **2 FDXM contracts** as a hard cap at base equity ($100k). Sanity: 2 × €5 × 20,000 = €200,000 notional ≈ 2.17× equity, which is below the IBKR margin headroom for FDXM but above a comfortable risk envelope, so the binding constraint is the vol-target × position-cap rule, not margin.
  - At AUM up to $500k: max **10 FDXM** or **2 FDAX**. Above $500k we trade FDAX exclusively.
- **Daily loss cap:** **2% of equity (absolute €1,840 at $100k base)**. Hit it → flatten and halt for the rest of the Eurex session.
- **Drawdown trigger:** Peak-to-trough equity DD of **8%** halts trading for **5 business days** of mandatory cooling-off. DD of **15%** halts and triggers a kill review (Lesson 10).
- **Circuit breaker:**
  - 3 consecutive losing flips → halt for 48 hours.
  - 1 single-day loss > 1.5% → halt next session, manual review.
- **Correlation rule:** This strategy is **forbidden to deploy concurrently** with any other equity-index futures momentum strategy in the same account (Lesson 6 — collisions with hypothetical "ES futures momentum" or "Nikkei TSMOM" would correlate too tightly to count as diversification). Cross-correlation cap with any other deployed strategy: 30-day rolling |ρ| > 0.6 on daily P&L → reduce size of newer strategy to 0.5× until ρ falls below 0.4.

---

## 8. Execution plan

- **Broker / SDK:** Interactive Brokers, **ib_insync** (Python). Connect to TWS or IB Gateway. The DAX is exchange-traded, so the SDK call is `Future('DAX', '<YYYYMM>', 'EUREX', multiplier='25', currency='EUR')` for FDAX or `Future('DAX', '<YYYYMM>', 'EUREX', multiplier='5', currency='EUR')` for FDXM. Use `qualifyContracts()` before any order. This satisfies Lesson 8 (battle-tested SDK).
- **Order type:**
  - **Marketable-limit IOC** at mid + 0.5 tick offered side (buy) or mid − 0.5 tick bid side (sell). FDAX tick = 0.5 index points = €12.50/contract; FDXM tick = 1 index point = €5/contract.
  - **No market orders, ever** on Eurex futures — spreads can widen 5×–10× during ECB pressers and the 17:30 CET settlement auction.
- **Slippage budget:** **1 tick on FDAX, 2 ticks on FDXM** per flip; round-trip ≤ 4 ticks. At FDXM this is ≤ €20/contract round-trip, or ~1 bp on €200k notional — well within an edge that targets ~1% per month.
- **Measuring slippage:** Capture quote midpoint at signal-emit time, store with execution fill from IBKR's `execDetails`. Daily delta logged to `data/exec_slippage.parquet`. Per Lesson 4: reconcile position quantity against IBKR `reqPositions()` every 5 minutes during the session, not against any local cache.
- **Broker rejection handling:**
  - Margin reject → halve the order, retry once; if still rejected, halt and email.
  - Connection drop → ib_insync auto-reconnect with `IB.connect(..., timeout=20, readonly=False, account=...)`; on reconnect, immediately call `reqPositions()` and compare to expected state; any divergence halts trading until manual reconcile.
- **Eurex/US-overlap latency handling:** Place orders from a Frankfurt-region cloud host (Hetzner FSN1 or AWS eu-central-1) so transatlantic RTT (~85ms) does not eat into the IOC's "I" window during US morning chop. The ib_insync client connects to a local IB Gateway on the same VPC; orders to Eurex traverse only Frankfurt-internal hops.
- **Time windows excluded:**
  - 17:00–17:35 CET (settlement auction).
  - ±15 min around ECB rate decisions and ±5 min around US NFP release.

---

## 9. Capacity

- **AUM ceiling (edge degradation):** ~$50M. Reasoning: even at 10 FDAX contracts (€5M notional), the strategy is ~0.01% of FDAX ADV (10/110,000). Edge degradation in TSMOM literature comes from *crowdedness* (many systematic momentum funds front-running each other), not from execution slippage on a single instrument. The crowdedness risk hits at the multi-billion AUM scale for diversified TSMOM; for a single-instrument DAX sleeve at <$100M, slippage is irrelevant.
- **ADV consumption:** at the $100k base account, 1-2 FDXM/flip = 0.001–0.002% of FDXM ADV. Even at $50M AUM and ~500 FDAX contracts at flip, that is ~0.5% of FDAX ADV per flip, executable across 30 minutes of TWAP without market impact.
- **Concentration risk:** by construction, 100% in one instrument (DAX) and one currency (EUR). This is the explicit *anti-diversification* of a single-asset TSMOM sleeve — it is meant to be one of N sleeves in a larger book. If deployed as a standalone strategy, mandatory hedge: a EUR/USD FX hedge sized to 1.0 × notional, executed as a 30-day forward roll, *or* accept the FX exposure as part of the edge (TSMOM applied at scale by AQR includes FX as a return source).

---

## 10. Failure modes

1. **Sustained vol regime change (TSMOM's classic failure):** A whipsaw market that oscillates inside a band wider than typical noise but narrower than typical trend yields back-to-back flips and steady bleed. Visible in metrics as: 60-day Sharpe < 0 with > 6 flips in the window, σ_60 elevated >1.5× vs. 5y average, and average holding period < 30 days. 2009 H1 and 2016 H2 both produced this pattern. **2022 EU energy crisis** was particularly hostile: Russia-Ukraine invasion in February inverted multi-asset momentum signals within 2 weeks, and Bunds + DAX flipped together. Detection: 90-day rolling P&L < -5% with > 4 flips triggers an immediate kill-review (Lesson 10).
2. **EU sovereign or credit crisis tail (gap-through-stop):** 2012 Greek-crisis style and 2022 energy-crisis style episodes can produce DAX gap-downs of 4-6% overnight. With a long position sized to 10% annualized vol (~ 0.6%/day expected), a 5% adverse gap is ~8× daily vol = ~3% of equity in a single bar. Failure visible as: a daily P&L beyond −3σ that is also outside the previous 5-day high-low range. *Mitigation:* the vol-target re-sizing on Monday morning means a vol spike automatically shrinks the next week's position by ~half within 5 business days; but the first gap is uninsured. Accept this as the price of admission for momentum and size the strategy at ≤ 20% of total book risk.
3. **Eurex outage / IBKR Eurex gateway failure:** Less than once-per-decade but real (Eurex had a multi-hour outage in April 2020). If the gateway is down at a scheduled flip, the strategy holds the old position. Visible as: ib_insync `IB.isConnected()` false for > 5 min, or `reqContractDetails` round-trip > 30s. **Detection rule:** any 2-minute outage during a signal-evaluation window aborts that flip and logs to `data/eurex_outage.log`; missed flip counted in the falsification budget below.
4. **Confirmation-gate stuck at zero in low-vol regime:** if all three look-backs straddle zero return (e.g., late 2017 grind-up with periodic small corrections), the strategy goes flat for weeks. Not a loss, but a sign the *thesis is dormant*. Watch for: time-in-flat > 60 trading days. Not a kill, but a flag for review at 90 days.
5. **Roll cost surprise:** the FDAX front-back calendar spread widens around index dividend events. Failure if the roll cost on the contract switch exceeds 4 ticks (€100/FDAX or €20/FDXM). Logged separately; cumulative roll cost > 1% of equity per year = strategy review.

---

## 11. Falsifiable verdict criteria

**Setup:** 30 shadow-paper days at IBKR paper-account, then verdict.

- **Promote to live capital** — *all* must hold:
  - 30-day shadow Sharpe > 0.8 (annualized).
  - Max DD over 30 days < 5% of paper equity.
  - Zero broker-truth divergences (reconcile `reqPositions()` vs. local state per Lesson 4; *any* mismatch > 1 contract for > 1 minute is a fail).
  - Slippage average ≤ 1 tick FDAX / 2 ticks FDXM, p95 ≤ 2 ticks / 4 ticks.
  - No more than 1 missed flip due to Eurex/IBKR outage.
  - OOS regression t-stat on `s_12` recomputed including the shadow window is still > 1.5.

- **Continue shadow** (don't promote yet, don't kill):
  - Sharpe in [0.3, 0.8] on the 30-day window *or* < 30 flips observed (insufficient evidence).
  - All other constraints met.
  - Extend shadow by 30 days. Hard cap: 90 days total shadow per Lesson 1.

- **Kill immediately** — *any one* triggers:
  - 3-day rolling DD > 5% of equity.
  - Daily loss cap breach (> 2% in one session).
  - Broker-side fill discrepancy > 0.5% of any single order's notional, or > 1 contract on any reconcile.
  - 5 consecutive losing flips.
  - OOS t-stat on `s_12` drops below 0 over any 60-day window once live.
  - Eurex data subscription lapses; trading must halt until restored.

**90-day live falsification (the hard kill, per Lesson 10 and spec hard rule):**
After 90 calendar days live, compute live Sharpe and max DD. **Shut down if**:
- 90-day live Sharpe < 0.3, *or*
- 90-day max DD > 12%, *or*
- Cumulative slippage > 2× the budgeted 4 ticks/flip, *or*
- The 12-month look-back regression t-statistic, recomputed including live data, has fallen below 0.

These thresholds are conservative because TSMOM is well-published; if the live result is materially worse than the academic anchor, that is *prima facie* evidence the regime has shifted under us. The cost of being slow to kill a dead momentum sleeve is worse than the cost of killing a temporarily-unlucky one — re-deployment is cheap.

---

## 12. One-paragraph operational risk note

**Operational surface:** A single Eurex data subscription downgrade on the IBKR account (e.g., professional re-classification, $13.50 → $110/month) does not kill the strategy but does flip its economics. A worse failure: ib_insync's TWS/Gateway requires a daily restart on the IBKR side (their published reset window), and our scheduler must accommodate that — a missed Gateway restart can blank the signal for a full Eurex morning before someone notices, and if that morning is a flip session we eat the slippage of a delayed reaction. The Hetzner Frankfurt host is a single-region deployment; an AWS eu-central-1 outage would not affect us, but a Hetzner FSN1 outage would, and our DR plan is a cold standby in AWS eu-central-1 that can come up in ~30 minutes. Finally, the strategy currency is EUR; the paper-trade book is USD-denominated; the FX conversion happens nightly via IBKR's auto-FX, which has historically had ~1-2 bps spread but has rarely been ≥ 5 bps. We have not, to date, observed an IBKR FX-conversion failure during a Eurex flip session, but it remains a tail-ops risk worth monitoring in the first 30 days.

---

## Appendix — Explicit treatment of the 10 lessons

1. **Validate before deploy.** Walk-forward 2000-2015 IS / 2016-2026 OOS, 21-day embargo, block bootstrap CI, ≥30 OOS flips, then 30-day shadow before live. Spec Section 6, 11.
2. **Binary gates, not confidence rankers.** TSMOM is rule-based and uses a *signed sum*, not a probabilistic confidence. The confirmation gate (12-month must agree with combined sign) is a *binary* filter, deliberately not a probability score. No ML "confidence" is used, so no ranker/filter conflation can arise.
3. **TP:SL asymmetry.** This strategy intentionally has **no fixed take-profit and no fixed stop-loss**, because the academic edge depends on holding through normal-magnitude adverse moves until the signal flips. The "exit" is **signal-flip-based**, which is structurally asymmetric in the right direction: when a position is right, we let it run for up to 252 trading days as long as `s_12` stays aligned (effectively infinite TP); when the position is wrong, the 1-month and 3-month look-backs flip first and combined `S_t` crosses zero within ~1-3 months (capped MAE). The hard daily-loss cap (2%) and 8% drawdown halt act as a circuit breaker around catastrophic events, not as a stop loss in the conventional sense. This argues *against* a fixed-stop architecture for this specific edge; imposing one would, per Moskowitz et al.'s implicit conclusion and the AQR replication work, *reduce* expectancy.
4. **Broker truth, not local cache.** `reqPositions()` reconcile every 5 minutes; mismatch > 1 contract for > 1 minute = halt. Section 8, 11.
5. **No PDT-exposed lanes on sub-$25k US equity accounts.** **N/A.** This strategy trades EU-domiciled futures via IBKR. PDT is a FINRA rule for US-listed equities and equity options; it does not apply to Eurex futures. Even a sub-$25k account can run this strategy.
6. **One strategy per slot.** Section 7: forbidden to deploy with any other index-futures momentum sleeve. Cross-strategy correlation cap |ρ| < 0.6 enforced.
7. **No same-data backtest.** IS 2000-2015, OOS 2016-2026, walk-forward inside OOS, 21-day embargo, no signal parameter tuning on OOS data (parameters are fixed by Moskowitz et al.).
8. **Battle-tested broker SDKs only.** `ib_insync` (or its maintained fork `ib_async`). No hand-rolled HTTP.
9. **Capacity bound named upfront.** $50M AUM ceiling, 0.5% of FDAX ADV consumption at full size, single-instrument concentration risk disclosed. Section 9.
10. **Kill criteria written before deploy.** Section 11 contains hard thresholds: 3-day DD > 5%, daily loss > 2%, 5-loss streak, 60-day t-stat < 0, 90-day Sharpe < 0.3 or DD > 12%. All numeric, no vibes.

---

## Summary numbers (quick reference)

| Item | Value |
|---|---|
| Instrument (primary) | FDAX (Eurex), multiplier €25/point |
| Instrument (small-account) | FDXM (Eurex), multiplier €5/point |
| Vol target | 10% annualized |
| At $100k base, σ_60 = 18% DAX | ~1 FDXM contract |
| At $100k base, hard cap | 2 FDXM |
| Daily loss cap | 2% of equity |
| DD halt | 8% (5-day cool-off), 15% (kill review) |
| Slippage budget | 1 tick FDAX / 2 ticks FDXM per flip |
| Validation IS | 2000-2015 |
| Validation OOS | 2016-2026 (~2,580 daily obs, ≥30 flips) |
| Promote Sharpe | OOS > 0.7, lower 95% CI > 0.3 |
| 90-day kill Sharpe | < 0.3 |
| AUM ceiling | $50M |
| Recurring data cost | ~$26/month non-pro |
