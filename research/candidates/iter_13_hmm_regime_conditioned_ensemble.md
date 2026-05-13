# Iteration 13 — HMM Regime-Conditioned Ensemble (SPY + Sector ETFs)

## 1. Thesis (one sentence)

A 3-state Hidden Markov Model trained on daily (SPY log return, VIX level, 50/200-DMA cross indicator) emits regime labels (Trending-Up / Trending-Down / Choppy) whose **predictive accuracy on the next day's realized regime exceeds 50% on out-of-sample data** (vs. ~33.3% random baseline and vs. a one-period-lagged persistence baseline of ~58% — the HMM must beat *the persistence baseline*, not just random), and conditioning a sub-strategy menu (TSMOM in Up, put-protection-or-flat in Down, RSI mean-reversion in Choppy) on that label produces an ensemble whose **OOS net Sharpe over 2019-2026 is ≥ 0.7 and max drawdown is < 22%** — and we will kill it within 90 days of live deployment if the HMM accuracy collapses to persistence-baseline level OR the ensemble's net P&L attribution shows regime-conditioning subtracts value vs. a single equal-weighted base strategy.

This is a **Hamilton (1989, *Econometrica*) regime-switching model** applied as a *filter* — explicitly per Lesson 2, the regime label gates *which* sub-strategy is on, not how big the position is. The HMM's posterior probability is the filter, not the ranker. **Ang & Bekaert (2002, *RFS*) "International Asset Allocation with Regime Shifts"** is the closest published precedent for using a Markov-switching layer to *condition allocation rules*, not to forecast returns directly.

---

## 2. Market / asset class / venue

- **Primary instrument:** `SPY` — SPDR S&P 500 ETF Trust, NYSE Arca. Most liquid US equity ETF (ADV ~70M shares, ~$35bn notional). Used for the regime input feature and as a fallback long instrument in the Up regime.
- **Sector ETFs (sub-strategy basket):** `XLK` (technology), `XLF` (financials), `XLE` (energy), `XLP` (consumer staples). All State Street Select Sector SPDRs, NYSE Arca, ADV $1–4bn notional each. Chosen for (a) low overlap with each other, (b) different macro betas (XLK pro-cyclical/duration-sensitive, XLF rate-sensitive, XLE commodity-cyclical, XLP defensive), (c) >20 years of clean daily history.
- **VIX input:** `^VIX` (CBOE Volatility Index, free CBOE delayed; live snapshot from IBKR `CBOE Indices Value` bundle ~$1.50/mo retail). Not a tradable instrument here, only a feature.
- **Put protection (Down regime only):** **5%-OTM SPY monthly puts**, the standard 30-45 DTE Friday expiry chain. Sized to roughly hedge the residual long-sector book (notional ≈ 50% of long sector exposure) — but the default Down posture is **flat**, with puts only used when the strategy is forced to hold sector longs through a regime transition (see §4).
- **Primary exchange:** NYSE Arca (ETFs), CBOE (^VIX feed), CBOE (SPY options).
- **Broker:** Interactive Brokers Pro via `ib_insync` (Lesson 8 — battle-tested SDK).
- **Settlement / margin:** T+1 cash settlement on ETFs; Reg-T on long positions. Long puts require option-level margin only (= premium). No PDT concern — see §10 below — we operate at ≥$25k NAV (Lesson 5) and trades are not day trades by design (overnight holds).
- **Tax regime:** US-domiciled IBKR account, 1099-B short-term capital gains on ETF turnover; SPY options are *not* Section 1256 (single-name/ETF options are 1099-B at short-term rates). Tax drag is meaningful for the Choppy mean-rev sub-strategy (high turnover) — flagged in §10.

---

## 3. Time horizon

**Daily,** with a single **intraday regime check at 10:00 ET** that can downgrade the regime confidence (but not flip the regime label) if VIX has moved >15% from the prior close. Order submission window: 15:45–15:55 ET via marketable-limit MOC-eligible orders, after the day's signal is finalized on the 15:30 ET VIX/SPY snapshot.

The 10:00 ET intraday check exists to handle the case where an overnight news event has invalidated yesterday's regime label *before* we put on or roll the day's positions. It does **not** generate trades by itself — it only forces a "stay flat" override if the intraday VIX move is large enough that the regime classifier is plausibly stale.

Bucket per spec: **daily**.

---

## 4. Method family and specific algorithm

**Family:** Hybrid — unsupervised ML (Hidden Markov Model) as a *regime classifier / filter*, with three rule-based sub-strategies sitting underneath. No NN, no LLM. The HMM does *not* trade and does *not* size positions; it only emits a label that selects which sub-strategy is live.

### 4.1 HMM regime layer

**Library choice:** `hmmlearn` (scikit-learn-style Gaussian HMM, actively maintained as of 2025). Fallback: `pomegranate` (deprecated original; v1.x rewrite on PyTorch is API-incompatible with older code — pin `hmmlearn==0.3.x`, not pomegranate). See §12 for operational risk.

**Features (exactly 3, fixed in advance per Lesson 2 to avoid feature creep into a ranker):**
1. **`r_SPY`** — daily log return of SPY adjusted close.
2. **`VIX_level`** — daily close of ^VIX (level, not change; HMM handles the scale via covariance fitting).
3. **`MA_cross`** — `+1` if SPY 50-day SMA > 200-day SMA (golden), `-1` otherwise (death). Discrete feature; the HMM treats this as a third continuous dimension but the bimodality is meaningful.

**Model:** 3-state Gaussian HMM with diagonal covariance (full covariance is harder to fit stably on only 3 features × ~250 train days per refit). States are *not* pre-labeled; we **post-hoc assign** state names by inspecting the fitted emission means:
- State with highest mean `r_SPY` and lowest mean `VIX_level` and predominantly `MA_cross = +1` → **Trending-Up**.
- State with lowest mean `r_SPY` and highest mean `VIX_level` and predominantly `MA_cross = -1` → **Trending-Down**.
- Remaining state → **Choppy**.

If post-hoc assignment is ambiguous (two states have indistinguishable means within 1σ), the model is rejected and the prior refit's model is retained — flagged in §10 as the "label-switching" failure mode.

**Inference cadence:** HMM is **refit annually** on a rolling 10-year window (2520 trading days). Predictions are produced **daily** using the fitted transition matrix and emission probabilities via the forward algorithm; `P(regime_t | observations_1..t)` is computed and the **3-day rolling mean of the posterior probability vector** is used as the smoothed regime estimate (Lesson 12 — raw HMM posteriors are jittery, smoothing is required).

**Decision rule for active regime:**
- Let `p_smooth = ` 3-day rolling mean of the posterior probability distribution.
- Active regime = `argmax(p_smooth)` **iff** `max(p_smooth) ≥ 0.60`. Otherwise active regime = **`Ambiguous`** → strategy is flat (Lesson 2 — binary gate, not a confidence ranker).
- **Regime switch cooldown:** once the active regime changes, no further switch is permitted for **3 trading days**. This is anti-whipsaw and directly addresses Lesson 2 + Lesson 6 (don't let a noisy filter chew up costs).
- **Monthly switch cap:** maximum **4 regime switches per calendar month**. The 5th attempted switch is suppressed and the strategy stays in the prior regime until month-end. Rationale: at 20 trading days/month, more than 4 switches means the HMM is hallucinating regimes and we shouldn't pay for it. (Historical base rate on 2005-2018 in-sample is ~2.1 switches/month.)

### 4.2 Sub-strategies (one and only one is live at a time)

**(a) Trending-Up sub-strategy — TSMOM on sector ETFs, long-only:**

For each of `{XLK, XLF, XLE, XLP}`, compute three sub-signals from adjusted-close daily log returns:
- `s_1 = sign(r_{t, 21})` (1-month)
- `s_3 = sign(r_{t, 63})` (3-month)
- `s_12 = sign(r_{t, 252})` (12-month)

Combined per-ETF score `S^{i}_t = (s_1 + s_3 + s_12) / 3 ∈ {-1, -2/3, ..., +1}`.

**Long-only filter:** if `S^{i}_t > 0`, include ETF `i` in the long basket; else exclude. The basket is equal-weight among included ETFs.

**Sizing:** each included ETF gets a **20% NAV slot** target, capped at **80% NAV total long exposure** (so if all 4 ETFs are positive, the basket is fully invested; if 2 are positive, NAV is 40% in those two and 60% in SGOV — see §7).

**Why TSMOM here:** Moskowitz/Ooi/Pedersen (2012) document TSMOM works best in trending regimes (their cross-asset t-stat ~5 at the 12-month horizon). Restricting TSMOM to the HMM-labeled Up state is exactly the *regime-conditioning* hypothesis. If TSMOM works as well in the Choppy regime, the conditioning adds no value (this is one of the things validation must score).

**(b) Trending-Down sub-strategy — flat + optional puts:**

**Default posture:** **100% SGOV** (cash equivalent, iShares 0-3 Month T-bill ETF, NYSE Arca). No long sector exposure.

**Put protection rule (used only as a transition handler):** if the regime *just flipped* from Up to Down with non-zero long sector positions outstanding, do not market-sell into the down-regime; instead:
1. Hold the existing sector longs for **1 trading day** to avoid panicking on a possibly-false regime flip.
2. Simultaneously buy **5%-OTM monthly SPY puts** with 30-45 DTE, in notional equal to **50% of the existing long sector $-exposure** (partial hedge — we're not trying to hedge sector basis risk, just the SPX-beta portion).
3. On day 2 after the flip, sell the sector longs and the puts together. Net cost is bid/ask + put premium decay for ~24h.

This is a transition-cost-aware Down rule. The strategy is not trying to *make money* in the Down regime — it is trying to **not lose money in transitions** while having an answer for "what do I do with the longs I had on Friday?"

**(c) Choppy sub-strategy — RSI mean-reversion on sector ETFs:**

For each of `{XLK, XLF, XLE, XLP}` independently:
- Compute `RSI(14)` on daily closes (standard Wilder smoothing).
- **Entry:** if `RSI < 30` AND no position currently → long **10% NAV** of that ETF.
- **Exit:** if `RSI > 55` (mid-band exit, not 70) → flatten.
- **Stop:** if position is held >10 trading days OR drops more than -8% from entry → flatten.
- **Reverse signal:** if `RSI > 70` AND already long → flatten (no short side — long-only on sectors per Lesson 5 PDT considerations and to keep tax treatment clean).

Max simultaneous Choppy positions: **3 of 4** sectors. If all 4 trigger long, take the 3 lowest-RSI ones.

**TP:SL asymmetry (Lesson 3):** The mean-rev exit at RSI 55 + 8% stop creates an asymmetric risk profile — typical entries at RSI 28 have ~3-4% drawdown tolerance and ~5-6% upside target, so the ratio is ~1:1.5. Documented here explicitly because Lesson 3 says stops are part of the edge.

### 4.3 Meta-learner (voting)

The "voting" is degenerate: there is exactly one sub-strategy live at any time, gated by the HMM label and the `max(p_smooth) ≥ 0.60` filter. If `max(p_smooth) < 0.60`, **no sub-strategy is live and the book is 100% SGOV**.

This is *intentionally simple* per Lesson 2 — a more sophisticated voting scheme (e.g., probability-weighted blending of sub-strategy positions) would turn the HMM posterior into a *ranker*, which Lesson 2 explicitly forbids without separate ranking-power validation.

### 4.4 Transaction cost model (MANDATORY per assignment)

Costs must be modeled or the strategy looks better than it is:

- **ETF round-trip:** **0.5 bp commission (IBKR Pro tiered) + 0.5 bp half-spread on SPY/XLK/XLF, 1 bp on XLE/XLP at midday liquidity = 2-3 bp round-trip per leg.**
- **Regime-switch cost:** flipping from Up to Choppy involves liquidating ~80% NAV of sector longs (in possibly 2-4 ETFs) and re-establishing a different basket. Assume **~6 bp NAV** per full regime switch (4 legs × 1.5 bp average half-spread + commissions).
- **Put leg cost:** SPY monthly 5% OTM puts trade at 5-10 cent spreads on a ~$2-4 contract, so ~2-3% of premium per round-trip; sized at 50% of long notional with puts at ~0.5% premium ≈ **~25 bp NAV per regime-transition use** (we expect ~6-8 of these per year).
- **Monthly switch cap (4/month × 6 bp ≈ 24 bp/month worst case)** is the budget. Annualized worst case ~290 bp = ~2.9% of NAV, which is a meaningful headwind that the gross edge has to clear.

The backtest **must** include all three of these. A backtest that nets to +6% with cost-blind execution but +2.5% with realistic costs is the Lesson 7 trap.

---

## 5. Data requirements

| Feed | Use | Vendor | Cost | Storage |
|---|---|---|---|---|
| SPY, XLK, XLF, XLE, XLP daily OHLCV adjusted | HMM features + sub-strategy signals | Stooq free / Yahoo / Polygon Free (within 5 rpm — Lesson from CLAUDE.md) | $0 | <50 MB |
| ^VIX, ^VXV daily | HMM feature | CBOE free CSV | $0 | <10 MB |
| SPY options chain (5% OTM monthly puts only) | Down-regime hedge | Polygon Free (within rate limit) or IBKR (option metadata + delayed prices free) | $0 historical; live needs IBKR US Options bundle ~$1.50/mo | <500 MB for backtest |
| SPY/XLK/etc. live quotes | Execution | IBKR US Securities Snapshot ~$10/mo | $10/mo | ~5 GB/yr live capture |
| Calendar (FOMC, CPI, NFP dates) | Pre-event blackout — disable regime flips within 24h of FOMC | FRED / public ICS feeds | $0 | trivial |

- **Total storage:** <10 GB on 5-year horizon. Local SSD; no S3 needed.
- **Total recurring cost:** ~$13/mo IBKR data subscriptions, $0 historical.
- **Latency:** signal is daily; latency budget is **~5 seconds** for the 15:45 ET trade window. Not latency-sensitive.

**Polygon Free posture (per project CLAUDE.md):** all backtest data is cached locally before any HMM fit. We do not stream Polygon during walk-forward — we extract once, cache to local Parquet, and replay. Polygon's 5 rpm limit is comfortable for this strategy.

---

## 6. Validation plan

This section does the heaviest lift in the iteration because the assignment specifically demands the HMM be scored *separately* from the ensemble P&L (Lesson 2 — regime probability is the filter, not the ranker).

### 6.1 Walk-forward windows

- **In-sample (calibration + sub-strategy parameter freezing):** **2005-01-03 → 2018-12-31** (14 years, ~3,520 trading days). Used to:
  - Choose HMM hyperparameters (3 states, 3 features, diagonal covariance, 10-year rolling refit).
  - Verify post-hoc state-to-label assignment is stable.
  - Lock the 0.60 probability threshold, 3-day smoothing, 3-day cooldown, 4-switch monthly cap.
  - Verify TSMOM look-backs (21/63/252) and RSI parameters (14, <30/>55, 8% stop, 10-day max-hold) do not need per-regime tuning beyond the defaults. **No per-regime tuning is allowed at this stage** — if a sub-strategy needs different parameters in different regimes, the whole conditioning hypothesis is suspect.

- **Out-of-sample (evidence window):** **2019-01-02 → 2026-04-30** (7 years, ~1,840 trading days, well over the 36-month minimum). Includes:
  - 2019 melt-up (Up regime test)
  - **2020 COVID crash + recovery** (regime transitions in compressed time)
  - 2021 reflation (Up regime)
  - **2022 bear market** (Down regime — the existential test)
  - **2023 banking mini-crisis** (Choppy regime test)
  - 2024 election year
  - 2025-26 tail

- **Walk-forward step:** annual refit of HMM, parameters frozen for one year forward, step = 12 months. 7 forward folds.

- **Purge & embargo:** 5 trading days between train-end and test-start on each fold (autocorrelation in VIX > 0.9 at lag 1, drops below 0.2 by lag 5). This kills the obvious leak from using a HMM trained on data that contains tomorrow's actual regime.

### 6.2 Score the HMM separately (CRITICAL per assignment + Lesson 2)

This is the part most regime-switching designs omit. We will compute **three orthogonal performance numbers** before looking at P&L:

**(i) Regime-prediction accuracy vs. random baseline.**
- Define "realized regime" using a forward-looking 21-day window oracle: at each day t, look at the next 21 days' SPY return and VIX path, label as Up if SPY return > +3% and VIX avg < 18, Down if SPY return < -3% and VIX avg > 25, Choppy otherwise. This is a *post-hoc* label, only used for HMM scoring.
- Compare HMM's day-t label to the oracle's day-t label on OOS data.
- **Random baseline = 33.3%** (3 classes).
- **Persistence baseline = "today's regime label = yesterday's HMM label"**, expected ~58-65% accuracy due to regime stickiness.
- **Pass criterion: HMM OOS accuracy ≥ persistence_baseline + 3 percentage points.** If HMM only matches persistence, it is the persistence baseline with extra steps; the strategy is just "stay in the same regime label until something changes," which is fine, but you can do that without an HMM.

**(ii) Ranking power of HMM posterior probability.**
- Compute Spearman rank correlation between `max(p_smooth)` (the regime-confidence number) and the **next-21-day sub-strategy P&L** *within* each regime.
- **Pass criterion: |Spearman r| < 0.10** within each regime. **This is a NEGATIVE test** — if the probability *ranks* P&L, we are tempted to size by it, and Lesson 2 says we explicitly tested that path and it doesn't work without separate validation. If we find ranking power we then have to *separately* validate sizing-by-probability before deploying with it. Default: do not size by probability.

**(iii) Ensemble P&L vs. unconditioned baseline.**
- Baseline strategy: 50/50 TSMOM + RSI mean-reversion at all times on the same sector ETFs, equal NAV split, no regime conditioning.
- Conditioned strategy: this iteration's HMM-routed ensemble.
- **Pass criterion:** Conditioned net Sharpe > Baseline net Sharpe by ≥ 0.25 with bootstrap 95% CI (10k draws, 20-day block).
- **CRITICAL FAILURE MODE:** if Conditioned has higher gross Sharpe but *lower* net Sharpe than Baseline, regime switching is paying for itself in costs — kill.

### 6.3 Stat tests on the ensemble

- 10k-draw stationary block bootstrap (block size 20 days, autocorrelation-aware) of OOS daily returns.
- **OOS net Sharpe ≥ 0.70** required (annualized, after all costs in §4.4).
- **OOS max DD ≤ 22%** required.
- **Bootstrap 95% CI of Sharpe excludes 0.30.** A lower CI of 0.0 is too generous given the cost headwinds; we want margin.
- **Minimum OOS regime transitions: 35** (~5/year × 7 years). If we see fewer than 35, the HMM is *too sticky* and we have not actually tested the conditioning logic on enough switches.
- **Minimum OOS sub-strategy trades: 200 in TSMOM + 150 in RSI mean-rev + 8 in put-transition.** Below this the per-leg statistics are unreliable.

### 6.4 De-overlap with iter_7 (Lesson 6)

iter_7 is DAX-futures TSMOM. This iteration uses TSMOM-on-sectors as one of three sub-strategies, on a different geography and different instrument class. The methods are not the same edge, but I flag the *overlap* of the TSMOM sub-component honestly: if iter_7 deploys, then this strategy in the Up regime is correlated with iter_7 through the global momentum factor. §7 codifies a max-correlation rule that prevents both from running at full size simultaneously.

---

## 7. Risk envelope

- **Max position per ETF:** **20% NAV** (TSMOM Up basket) or **10% NAV** (RSI Choppy). Max total long exposure: **80% NAV**.
- **Put leg max:** notional = 50% of concurrent long sector $-exposure, premium-capped at **2% NAV** per use.
- **Daily loss cap:** **-3% NAV** absolute. If realized intraday MTM hits -3% NAV, flatten everything at next print and stand down for the rest of the session.
- **Drawdown trigger:** trailing 30-day NAV drawdown ≥ 12% → strategy halted, human review required to restart.
- **Circuit breaker (Lesson 10):** at any point, **any** of the following halts trading for the day:
  - 3 consecutive negative trading days while regime-conditioned ensemble is live.
  - VIX intraday move > +30% from prior close (force regime check + freeze new entries).
  - HMM produces an `Ambiguous` label for ≥ 5 consecutive trading days (the model is broken or the world is genuinely unclassifiable; either way, stand down).
  - Realized monthly regime-switch count exceeds the cap of 4.
- **Correlation rule (Lesson 6 + Lesson 9):**
  - If iter_7 (DAX TSMOM) or iter_2 (ES intraday) is deployed: cap this strategy's size at **50% of nominal** when its 60-day return correlation with the other strategy exceeds 0.5.
  - If iter_5 (VIX term-structure) is deployed: cap this strategy at 50% when *both* are simultaneously long sector ETFs / SVXY (compounded short-vol exposure).
  - Hard-reject simultaneous deployment with any other "regime-classifier" iteration in future waves (Lesson 6, one strategy per slot — the HMM *is* the slot identity here).
- **Cash floor:** maintain ≥**20% NAV in SGOV** in Up regime (so we are never 100% in 4 ETFs); always 100% SGOV in Down or Ambiguous.

---

## 8. Execution plan

- **Broker / SDK:** Interactive Brokers Pro via `ib_insync` (Lesson 8).
- **Order types:**
  - Sector ETF entries / exits: **marketable-limit IOC** at last + 5 bps, fallback to MOC if the limit isn't filled by 15:55 ET.
  - SPY puts: **limit** at midpoint of NBBO, retry up to 3 times widening by 5 cents each time; fail-over to next-day attempt if no fill (the put leg is non-urgent — the regime flip's exposure decay happens over 1 trading day).
- **Slippage budget:** **3 bps per ETF leg, 5 bps per regime-switch event** (matches the cost model in §4.4). Slippage is *measured* daily by reconciling each fill against the 15:45 ET mid-quote snapshot stored at signal time.
- **Broker-truth reconciliation (Lesson 4):** before every order, re-query IBKR positions via `reqPositions()` and reconcile against the local model state. **Refuse to send orders if a mismatch > $50 notional or > 1 share is detected** — human review required. Local quantities are never trusted as authoritative for closing orders.
- **Rejection handling:** if IBKR rejects an order (margin, halted instrument, exchange-side outage), log the rejection with timestamp, drop the trade for that day, and **do not retry intraday**. Re-attempt at the next day's signal. The strategy is daily, so a one-day miss is acceptable; chasing fills introduces unmodeled execution risk.
- **PDT (Lesson 5):** the strategy is **not** day-trading by design — all entries are intended overnight holds. We operate at ≥$25k NAV. If account equity drops below $25k due to drawdown, the strategy halts (this is below the §7 drawdown threshold and is caught there first).

---

## 9. Capacity

- **AUM ceiling:** **$50M before slippage degradation.** Reasoning:
  - At $50M NAV, full Up-regime deployment puts ~$10M into each of 4 sector ETFs. XLE has the thinnest ADV at ~$1.5bn, so $10M = ~0.7% ADV. Acceptable.
  - XLP at ~$1.0bn ADV — $10M = 1% ADV, getting tighter. At $100M, $20M into XLP = 2% ADV, regime-switch days would push 4-8% ADV in a single window, ~10+ bps additional slippage. That's the degradation knee.
  - SPY put options at 0.5% premium scaled to 50% of $40M long exposure = $20M notional × 0.5% = $100k premium per use. ~6-8 uses/yr = ~$700k/yr in option premium. SPY options chain easily absorbs this; we'd buy ~200 contracts at 5%-OTM ~30 DTE.
- **ADV consumption at full size ($50M):** ~0.5-1% of each sector ETF's ADV. Comfortable.
- **Concentration risk:** single-ETF max 20% NAV (sized at $10M on $50M); single-sector max 20% (no doubling); equity-beta exposure max 80% NAV. The strategy is **not** market-neutral — it is a long-bias swing tactic with regime gating.
- **Capacity decays at $100M+** in the sector ETFs; further scale would require:
  - Routing into ES futures for the Up regime equity beta (capacity ~$1B+ ADV).
  - Routing into VIX futures or SVXY for Down-regime hedges.
  - Both changes would alter the operational profile substantially — flag as "scale-up redesign," not a parameter change.

---

## 10. Failure modes

At least 3 (assignment), I'll detail 6 because regime-switching strategies have a lot of surface.

1. **Regime classifier overfits to historical features.** Symptom: in-sample HMM accuracy >> OOS accuracy (e.g., 75% in-sample, 45% OOS). The HMM has memorized state transitions tied to specific VIX-level regimes that don't recur. *Metric:* the §6.2(i) HMM-accuracy test catches this directly. *Mitigation:* annual refit on rolling window keeps the model recency-anchored; if OOS accuracy drops below persistence-baseline + 3pp in **any** annual fold, halt deployment of new positions and re-evaluate. **Kill metric:** OOS accuracy < persistence baseline for 2 consecutive 6-month windows.

2. **Sub-strategy correlation in stress is higher than it looks.** Symptom: TSMOM in Up + RSI in Choppy *both* lose money in a regime-transition month because the HMM mislabels a transition and the trailing TSMOM positions get liquidated into RSI-bait that then continues falling (the "March 2020" scenario). *Metric:* monthly drawdown attribution shows the bulk of drawdown comes from *transition* days, not in-regime days. *Mitigation:* the 1-day "hold + buy puts" transition rule in §4.2(b) buys 1 day of protection. *Kill metric:* if >50% of OOS drawdown is concentrated in the 5-day window around regime switches, the conditioning is the problem and we kill.

3. **Transaction costs in regime switches eat the edge.** Symptom: gross OOS Sharpe > 1.0 but net OOS Sharpe < 0.5. *Metric:* §6.2(iii) explicitly compares Conditioned vs. Baseline on **net** Sharpe. *Mitigation:* the 4-switch/month cap and 3-day cooldown explicitly limit cost burn. *Kill metric:* net OOS Sharpe < gross OOS Sharpe by >0.4 means costs are dominant; kill.

4. **HMM hidden-state instability on retraining (label-switching).** Symptom: this year's "State 0" was Trending-Up, next year's "State 0" is Choppy — the post-hoc assignment is unstable. *Metric:* compare emission means of fitted states across consecutive annual refits; require Hungarian-algorithm matching with distance < 1σ across states. *Mitigation:* §4.1's explicit ambiguous-rejection rule (keep the prior model if labels can't be re-matched cleanly). *Kill metric:* if label-switching forces use of the prior model on >2 consecutive annual refits, the HMM is no longer converging on a stable regime structure → kill.

5. **HMM accuracy collapses to random in live deployment.** Symptom: regime labels become uncorrelated with subsequent realized regimes. *Metric:* rolling 90-day HMM accuracy on the §6.2(i) oracle dropped below 40% (random + small margin). *Mitigation:* the strategy stands down to 100% SGOV when `max(p_smooth) < 0.60`, which becomes more frequent as the HMM degrades — partial automatic graceful degradation. *Kill metric:* rolling 90-day HMM accuracy < 40% for 30 consecutive days.

6. **VIX spike from a non-equity event (FX crisis, bond market dislocation) labels regime as Down while sector ETFs are still trending.** Symptom: HMM says Down (VIX is high), but XLF/XLK realized return is +1% the next 21 days. The HMM is dominated by the VIX feature in tail events. *Metric:* days where HMM = Down but realized sector returns are >+1% over the next 21 days. *Mitigation:* the MA_cross feature (long-horizon trend) partially counteracts this — if both VIX is high *and* SPY is below its 200-DMA, then Down is correct; if VIX is high but SPY > 200-DMA, the HMM should put weight on a Choppy state, not Down. The 3-feature design specifically targets this failure mode by including a slow trend feature. *Kill metric:* >20% of Down-regime days OOS show subsequent positive 21-day sector returns → HMM is over-indexing on VIX.

---

## 11. Falsifiable verdict criteria

### Promote to live capital (after ≥30 shadow days, per Lesson 1):
- All of:
  - OOS net Sharpe ≥ 0.70 (per §6.3)
  - OOS max DD ≤ 22%
  - HMM OOS accuracy ≥ persistence baseline + 3pp (per §6.2(i))
  - Spearman |r| of `max(p_smooth)` vs. next-21-day P&L < 0.10 within each regime (per §6.2(ii) — confirming the probability is *not* a ranker, validating the binary-gate use)
  - Conditioned net Sharpe > Baseline net Sharpe by ≥ 0.25 with bootstrap 95% CI excluding 0 (per §6.2(iii))
  - ≥30 shadow days with no broker-truth divergence (Lesson 4) and no kill triggers hit
  - Total realized cost during shadow ≤ 30 bps/month

### Continue shadow (ambiguous):
- Metrics within ±1 standard error of pass thresholds, no kill triggers hit, regime label stability looks reasonable. Extend shadow another 30 days.

### Kill immediately (any one of, within 90-day live deployment window):
- 3-day net drawdown > 5% NAV.
- Daily loss cap breach (-3% NAV in a single session).
- Rolling 90-day HMM accuracy < 40% for ≥30 consecutive days.
- HMM label-switching forces prior-model fallback on >2 consecutive annual refits.
- Monthly regime switches exceed cap of 4 in any single month (suggests we calibrated the cap wrong or the world genuinely shifted regime cadence).
- Broker-truth divergence > 0.5% notional on any single reconciliation.
- Net Sharpe over the live window < 0 with ≥40 trading days elapsed.
- Sub-strategy P&L attribution shows ≥50% of total P&L came from a single regime (i.e., the diversification claim is empty — the strategy is just whichever sub-strategy is winning).

### 90-day falsification argument

Per assignment: ~60 trading days in 90 calendar days, ~2-4 regime transitions expected (historical base rate 2.1 switches/month → ~6 in 90 days, but the modal experience is 2-4 because some months have 0 switches). What 60 days CAN falsify:

- **HMM accuracy**: with 60 days and an oracle on a 21-day forward window, we get ~40 "ground truth" labels (allowing for the 21-day forward overlap). 40 trials against a 33% baseline gives a 95% CI of ±15pp on the HMM accuracy. We can detect a "no better than random" HMM with reasonable power.
- **Cost realism**: 60 days × expected ~1 switch/month = 2 switches. Each switch costs ~6 bp NAV. We can measure realized slippage vs. modeled 3 bp/leg and either confirm or reject our cost model with ±2bp resolution.
- **Drawdown discipline**: we will see at least one 5%+ daily move in 60 days (base rate ~3-5 such days). Whether the circuit breakers actually fire and whether the recovery looks like the backtest is a hard falsification of the risk framework.
- **What 60 days CANNOT falsify**: the long-run claim that regime conditioning produces +0.25 net Sharpe over the unconditioned baseline. The bootstrap CI on Sharpe needs ~250+ daily observations to be tight enough to reject the null. The 90-day window is a *risk-discipline* and *operational* falsification, not a *Sharpe* falsification. That comes from the 7-year OOS backtest pre-deployment.

This is honest: 90 days is enough to kill the strategy on operations, costs, or HMM degradation, but not enough to *promote* the strategy on Sharpe. Promotion to higher size requires 12+ months of live shadow + paper data on top of the OOS backtest.

---

## 12. One-paragraph operational risk note

The biggest non-market operational risk is **HMM library compatibility and posterior-probability jitter**. `hmmlearn==0.3.x` is the current maintained Gaussian HMM library; `pomegranate` v0.x is deprecated and `pomegranate` v1.x is a PyTorch-based rewrite with an incompatible API — a `pip install --upgrade` could silently break the entire regime layer if we don't pin versions. Beyond the library: HMM posterior probabilities are inherently jittery (the Viterbi path can flip-flop between two near-equal states across consecutive days even when the underlying observation barely changed), which is why we use a **3-day rolling mean of the posterior** (§4.1) and a **3-day regime-switch cooldown** (§4.1) and a **monthly switch cap of 4** (§4.1). If those smoothing layers are removed or accidentally disabled (e.g., a unit-test that runs with `rolling_window=1`), the strategy turns into a daily whipsaw machine that loses 20-30 bp/day to costs. Secondary ops risks: IBKR Eurex/CBOE data subscription lapses (silent failure mode where the VIX feed returns stale values, which we mitigate with a 10-minute staleness check on every signal evaluation), the post-hoc state-labeling step being non-deterministic across machines if the HMM initialization seed is not pinned (mitigated by `random_state=42` at every refit and a Hungarian-algorithm match against the previous fold's labels), and the put-protection leg's option chain data: SPY options chain pricing requires the IBKR US Options bundle ($1.50/mo additional), and the existing Polygon Free plan does have 5%-OTM SPY puts at daily granularity, but if Polygon's free tier downgrades or rate-limits, we lose backtest fidelity on the Down-transition leg — per the project's CLAUDE.md, this is the exact "synthetic prices = not valid evidence" trap the `paper_v1` policy was written to avoid, so the put leg's backtest will be run **only** against the real cached Polygon option chains (with `bid`, `ask`, `mid`, `underlying_close`, `atm_iv`) and never against synthetic/Black-Scholes fallbacks.

---

## Lesson compliance audit

| # | Lesson | How this iteration honors it |
|---|---|---|
| 1 | Validate before deploy | §6 walk-forward 2005-2018 IS / 2019-2026 OOS + ≥30 shadow days per §11. |
| 2 | Binary gates, not confidence rankers | §4.3 — regime label gates which sub-strategy is on; §6.2(ii) explicitly tests and *rejects* the use of HMM posterior as a position-sizing ranker without separate validation. The 0.60 probability threshold is a binary cutoff, not a continuous size scaler. |
| 3 | TP:SL asymmetry is structural | §4.2(c) — RSI mean-rev uses 8% stop, RSI 55 exit (~1:1.5 ratio); §7 — daily loss cap is asymmetric vs. expected daily P&L. |
| 4 | Broker truth, not local cache | §8 — `reqPositions()` reconciliation before every order, hard refusal to send if mismatch > $50/1 share. |
| 5 | No PDT-exposed lanes on sub-$25k US equity accounts | §8 — strategy runs at ≥$25k NAV; design is overnight-hold daily so PDT clock does not bind. Drawdown trigger halts before NAV crosses $25k. |
| 6 | One strategy per slot | §6.4 + §7 — hard-rejects co-deployment with any other "regime-classifier" iteration; correlation rule caps size when paired with iter_7 DAX-TSMOM or iter_5 VIX or iter_2 ES-intraday. |
| 7 | No same-data backtest | §6.1 — 2005-2018 IS / 2019-2026 OOS, walk-forward annual step with 5-day purge/embargo. |
| 8 | Battle-tested broker SDKs only | §8 — `ib_insync`. |
| 9 | Capacity bound named upfront | §9 — $50M ceiling with explicit ADV math on the tightest sector ETF (XLP), degradation knee at $100M, scale-up redesign path to ES futures. |
| 10 | Kill criteria written before deploy | §7 circuit breakers + §11 explicit kill list. |

---

## Falsifiability summary

This strategy is falsifiable on three independent axes within 90 days of live deployment:
1. **HMM signal quality**: rolling accuracy on the §6.2(i) oracle test must beat persistence baseline + 3pp. Detectable in ~40-60 trading days.
2. **Cost realism**: realized slippage on regime switches must be within 50% of the §4.4 model. Detectable in ~2-4 switches (60 days).
3. **Risk discipline**: drawdown triggers, daily caps, and broker-truth reconciliation must work as designed. One bad day (and there will be one in 60 days) tests the entire risk framework.

If any single one fails, the strategy is killed. Promotion to higher AUM requires *all three* to pass + a longer Sharpe-level falsification window of ≥12 months.
