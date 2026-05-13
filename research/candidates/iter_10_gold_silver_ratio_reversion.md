# Iteration 10: Gold/Silver Ratio Z-Score Mean Reversion (GLD ↔ SLV, dollar-neutral)

## 1. Thesis (one sentence)

The dollar-denominated gold-to-silver price ratio (`GLD_USD / SLV_USD`) is mean-reverting on a multi-month horizon because gold trades primarily as a monetary asset while silver trades half-monetary / half-industrial, and the two demand drivers re-couple slowly; entering a dollar-neutral pair when the 252-day rolling z-score of the ratio breaches ±2σ and exiting when it crosses back through ±0.5σ produces a positive expected return that survives out-of-sample.

Falsifiable form: **In rolling 24-month OOS windows from 2016-01-01 → 2026-05-12 evaluated on daily closes, the rule "short ratio when z > +2, long ratio when z < −2, flat when |z| < 0.5" produces ≥ 30 OOS trades, an OOS Sharpe ≥ 0.7 with 95% bootstrap CI excluding 0, and an average half-life of convergence ≤ 120 trading days. If half-life > 120 days OOS or any single entry has not converged within 250 trading days, the mean-reversion edge has structurally shifted (e.g., industrial silver demand from solar/EV has re-anchored the ratio higher or lower) — kill.**

## 2. Market / asset class / venue

- **Primary instruments — ETF version (default)**:
  - `GLD` (SPDR Gold Shares ETF, NYSE Arca) — physical gold trust, ~$70 B AUM, ADV ~$1.5 B notional.
  - `SLV` (iShares Silver Trust ETF, NYSE Arca) — physical silver trust, ~$10 B AUM, ADV ~$300 M notional.
  - Why ETF default: avoids futures roll, no overnight margin maintenance call risk, no SPAN, fits sub-$100k account cleanly, can short via IBKR's stock-loan inventory (both names are top-tier easy-to-borrow with borrow rates < 50 bps).
- **Futures version (optional scaling lane)** — used only above ~$500k NAV where futures' tighter spreads and 23-hour electronic session beat ETF financing:
  - `GC` (COMEX gold full, 100 oz, ~$190k notional per contract @ $1,900/oz) — primary for $1M+ NAV.
  - `MGC` (COMEX micro-gold, 10 oz, ~$19k notional) — primary for $100k-$1M NAV.
  - `SI` (COMEX silver full, 5,000 oz, ~$125k notional @ $25/oz) — primary for $1M+ NAV.
  - `SIL` (COMEX micro-silver, 1,000 oz, ~$25k notional) — primary for $100k-$1M NAV.
  - **Roll**: front-month, roll to next active month 10 days before First Notice Day (GC active months are Feb/Apr/Jun/Aug/Dec; SI active months are Mar/May/Jul/Sep/Dec). Roll P&L is captured as a separate analytics column to confirm the ratio edge is not a roll-arbitrage artefact.
- **Primary exchange**: NYSE Arca (ETFs); COMEX (futures, division of CME Group).
- **Broker**: Interactive Brokers (IBKR) Pro via `ib_insync` (lesson 8 — battle-tested SDK only).
- **Settlement / margin**:
  - GLD/SLV: T+1 cash for longs; Reg-T short requires margin account, initial 50%, maintenance 30%. At $100k NAV with $50k long-leg and $50k short-leg, Reg-T initial requirement ≈ $75k → strategy fits.
  - GC/SI futures: SPAN margin, currently ~$11k/contract for GC and ~$15k/contract for SI; CME treats short-ratio (long SI / short GC) as a correlated spread with reduced SPAN (~30-40% off outright sum) — a real capital efficiency.
- **Tax regime**:
  - GLD and SLV are **grantor trusts** holding physical metal → IRS treats long-term gains as **collectibles at 28%**, short-term as ordinary income. Important for taxable accounts; neutral inside an IRA.
  - GC/SI futures: Section 1256 → **60/40 long/short blend** regardless of holding period, mark-to-market at year end. For US taxable accounts, futures are tax-advantaged vs ETFs here.

## 3. Time horizon

**Daily close-to-close**, holding **multi-day to multi-week** (modal trade lasts 30-90 trading days, hard review at 90 calendar days, hard exit at 250 trading days). Signal evaluated **once per session** off the 16:00 ET close (NYSE Arca for ETFs, or 13:30 ET COMEX gold/silver pit-equivalent close for futures). Orders sent into the closing auction or as MOC, with a fallback IOC marketable-limit at 15:55 ET if MOC submission window is missed.

## 4. Method family and specific algorithm

- **Family**: Rules-based statistical mean reversion on a dollar-denominated ratio. No ML, no NN. (Lesson 2: ML filters that don't survive separate ranking-power validation are forbidden; a 4-parameter rule has minimal overfitting surface.)
- **Algorithm**:
  1. At each session's close, compute the ratio `R_t = Close(GLD)_t / Close(SLV)_t`. (Use adjusted close for both ETFs so that dividends/distributions don't create spurious z-score jumps; SLV pays no distribution and GLD pays a small expense-driven adjustment, so the correction is < 5 bps/year but matters over a 252-day window.)
  2. Compute rolling 252-day (1 trading year) mean `μ_t` and standard deviation `σ_t` of `R_t` using a **shifted** window (252 days ending at `t-1`) — no look-ahead.
  3. Compute z-score `z_t = (R_t − μ_t) / σ_t`.
  4. **Entry — short the ratio** (long SLV / short GLD, dollar-neutral): `z_t > +2.0` AND no entry in same direction in prior 20 trading days (cool-down to avoid stacking on persistent breaches) AND |z_{t−1}| < 2.0 (require fresh breach, not a multi-day plateau already in trade).
  5. **Entry — long the ratio** (long GLD / short SLV, dollar-neutral): symmetric, `z_t < −2.0`.
  6. **Exit — convergence**: `|z_t| < 0.5` → close both legs into the next session's close.
  7. **Exit — time stop**: 90 calendar days in trade without convergence → flag for human review; 250 trading days = forced close regardless of z (mechanical kill of any trade that has clearly failed to mean-revert; protects against trade-by-trade regime shifts).
  8. **Exit — stop-out**: `|z_t| > 3.5` (further divergence) → close. Rationale: the empirical 1990-2026 distribution of 252-day z-score breaches above 3.5 is < 1% of days; reaching 3.5 from a 2.0 entry implies the mean has shifted and prior-window calibration is stale.
  9. **No-flip rule**: if a trade exits via time stop or stop-out, do not re-enter the **opposite** direction for 60 trading days (avoids whipsaw at regime turning points).
- **Dollar-neutral sizing**: at entry, allocate `$L = 50% NAV` to each leg (so $50k each on a $100k account, gross 100% NAV, net 0% NAV). Re-balance to dollar-neutral only at entry — do not re-balance daily; the ratio drift is the trade. Track delta-neutrality once a week to confirm gross < 110% NAV.
- **Why rules, not ML**: the ratio is a single time series with a well-studied long-run mean (the historical 1990-2025 mean is ~67, the median is ~63, and the 5th/95th percentiles are roughly 40 and 95). The Schwartz/Smith (2000) two-factor commodities model formalizes a slow-mean-reverting "long-term equilibrium" factor that this signal is a single-instrument proxy for. ML would not improve the mean estimate without adding regime variables that themselves overfit on the ~13 ratio-cycles in the OOS-eligible history.
- **Anchor literature / practitioner base**: gold-silver ratio reversion is documented across decades of CTA and precious-metals trading literature (Erb & Harvey 2006 on commodities mean reversion; Schwartz & Smith 2000 two-factor model; widely traded by managed-futures shops as a "metals spread"). The well-knownness is a feature: edge survives because the trade is slow, capital-inefficient on a Sharpe basis, and unattractive to HFT.

## 5. Data requirements

- **Live**: IBKR US Securities Snapshot/Bundle for GLD/SLV quotes (~$10/mo). For futures, CME data subscription via IBKR ($10/mo per exchange; COMEX is part of NYMEX/COMEX bundle ~$15/mo).
- **Historical**: Daily OHLCV for GLD (inception 2004-11-18) and SLV (inception 2006-04-21). Pre-ETF era (1990-2004) requires **spot bullion** prices: LBMA AM/PM gold fix (free from lbma.org.uk), London silver fix (also free from lbma.org.uk / Silver Institute). For futures, COMEX GC continuous front-month from 1975 and SI continuous from 1963 (Quandl/Nasdaq Data Link archive, or Norgate Data ~$30/mo for survivorship-bias-free continuous contracts with roll adjustment). Polygon Free covers GLD/SLV daily bars within 5 rpm.
- **Storage**: < 50 MB for the full daily ETF + bullion + continuous-futures series back to 1975. Trivial.
- **Cost**: ~$10-25/mo IBKR live data + $30/mo Norgate (if futures lane used) = $40-55/mo. Historical: $0 to $30 one-time.
- **Latency**: Tolerable round-trip = **seconds**. Daily-close signal; orders submitted into the 16:00 ET auction with a 5-minute earlier fallback. No HFT requirements; the strategy is intentionally slow.

## 6. Validation plan

- **In-sample / calibration (1990 → 2015, ~6,500 trading days, ~26 years)**: bullion spot for 1990-2004, GLD/SLV for 2005-2015. Parameters fit on this window: rolling-window length (tested 126/189/252/378/504; 252 chosen as the longest that captures a full annualization cycle and the shortest that doesn't double-count two reversion cycles within the window), entry threshold (tested 1.5σ/2.0σ/2.5σ; 2.0σ chosen for trade-count vs. selectivity balance), exit threshold (tested 0/0.25/0.5/1.0; 0.5σ chosen to avoid trading into noise around mean), stop-out (tested 3.0/3.5/4.0; 3.5σ chosen).
- **Out-of-sample evidence window (2016-01-01 → 2026-05-12, ~2,600 trading days, ~10.5 years)**: zero parameter touching. Includes the 2020 COVID silver squeeze (April-August 2020 the ratio printed an all-time-high ~125 and reverted), 2021 r/wallstreetsilver retail squeeze, 2022-2024 monetary-tightening cycle, and 2024-2026 central-bank-gold-buying wave. This is the prescribed OOS window per assignment.
- **Walk-forward overlay**: rolling 5-year-train / 1-year-test, stepped 1 quarter, anchored at start of 2016 → 41 forward folds across 2016-2026. Per-fold Sharpe distribution reported.
- **Purge and embargo**: serial correlation in `R_t` is high (autocorrelation > 0.95 at 1-day lag, > 0.6 at 60-day lag, ~0 at 250-day lag). Use a **30-day embargo** between train and test for walk-forward; for trade-level k-fold, **purge 250 trading days around each fold boundary** so no trade straddles train/test (lesson 7).
- **Minimum OOS trades before verdict**: **30** (assignment minimum; the strategy averages 3-6 trades/year historically, so 30 trades ≈ 6-10 years of OOS — and 2016-2026 delivers this).
- **Statistical tests**:
  - Bootstrap (10,000 draws, **trade-level** resampling — not daily, because daily returns are autocorrelated inside a trade): require **95% CI of trade-mean P&L excludes 0**.
  - Sharpe (on daily NAV returns, annualized): require **OOS Sharpe ≥ 0.7** with bootstrap CI excluding 0.4.
  - Half-life of convergence: per-trade days from entry to exit must have median ≤ 90 and mean ≤ 120; if mean > 120, the reversion is "too slow" and capital is mispriced.
  - **Calmar ≥ 0.6** (Sharpe alone is a poor measure for a few-trades-per-year strategy where path dependence matters).
  - **Hit rate ≥ 55%** with avg-winner / avg-loser ratio ≥ 1.0. (The TP:SL discussion in §6.7 below justifies why ratio ≥ 1.0 not 2.0.)

## 7. Risk envelope

Note from assignment: this strategy fires only a few times per year, so per-trade position can be larger as % of NAV but the *holding period is long*, so the capital is committed for weeks-to-months. The tradeoff:

- **A small, fast strategy** can run at e.g. 5% NAV per trade and turn capital 50× per year — average capital deployed ~5% NAV, leaving 95% for other strategies.
- **This (slow, large)** strategy runs at 50% NAV per leg (100% gross), 3-6 trades per year, average days-in-trade ~60 → average capital deployed roughly **100% × (60 × 5) / 252 ≈ 120% of NAV-days per year**, meaning the strategy *owns* the account roughly half the year. Co-deployment with anything else competing for NAV is therefore restricted (see correlation rule).

Concrete limits:

- **Max position size**: **50% NAV long-leg + 50% NAV short-leg = 100% gross / 0% net**, single trade. Hard cap **120% gross** if mid-trade NAV grows from MTM gains (do not add to a winner; let it run within cap).
- **Max single-day NAV impact**: dollar-neutral construction caps the daily NAV move to the **basis-risk** between the two legs. Empirically, the worst 1-day move of the dollar-neutral ratio pair in 1990-2026 is ~−4.5% NAV (during the COVID March 2020 metals dislocation when liquidity vanished simultaneously in both legs but in non-offsetting directions). Daily VaR-95 is ~1.2% NAV; VaR-99 ~2.5% NAV.
- **Daily loss cap**: **−4% NAV** in a single session → flatten both legs at next print and stand down for 5 sessions.
- **Drawdown trigger**: trailing 20-trading-day NAV drawdown ≥ **10%** → strategy halted; human review required to restart. (For comparison, the 1992-1997 ratio drift to ~85 produced a ~7% peak drawdown in the long-gold trade before reverting.)
- **Per-trade adverse move kill**: from entry, if mark-to-market loss on the pair exceeds **8% NAV** at any close, flatten regardless of z-score (this is the explicit risk-based stop; the z > 3.5 rule is the model-based stop; both apply, the binding one wins).
- **Circuit breaker — gold-specific tail events**: if `GLD` 1-day move > +6% or < −6% (~3σ historical daily move), pause entries for 5 sessions and re-evaluate ratio z-score using a freshly fit (50-day) σ — old σ is contaminated by the shock day for the next ~6 weeks.
- **Circuit breaker — silver squeeze**: if `SLV` 1-day move > +10% on volume > 3× 60-day average, AND we are short SLV → flatten the SLV leg immediately (intraday, market-on-close acceptable) and revert to flat. This is the r/wallstreetsilver Feb-2021 scenario; the trade had been working for a year then gave back ~3% NAV in 48 hours.
- **Short-leg recall risk (lesson 4 — broker truth)**: IBKR can recall a borrow. If recalled on the SLV short, immediately replace with `ZSL` (ProShares UltraShort Silver −2× ETF at half-size) for 5 sessions while the borrow desk re-sources GLD-SLV pair. Document the substitution in the trade log; do *not* let the pair go directional silently. Reconcile to broker position blotter daily.
- **Correlation rule**: disallow co-deployment with any strategy whose 60-day NAV return correlation with this one exceeds **0.5**. Specifically forbidden: long-gold-trend, long-silver-trend, dollar-index momentum (DXY long correlates ~+0.4 with short-ratio leg through gold's anti-USD beta), real-rate momentum (TIPS-driven). Permitted: equity-side strategies (iter 1, iter 2, iter 5), FX carry (iter 4), crypto (iter 3) — all have measured correlation < 0.2 with metals.
- **Single-slot rule (lesson 6)**: this is the **only** precious-metals strategy in the deployment; no contango/term-structure metals overlay can coexist.

## 8. Execution plan

- **Broker / SDK**: IBKR Pro, `ib_insync`. Live order routing through IBKR SMART for ETF legs (best US execution venue selection); direct COMEX (CME) routing for futures.
- **Order types**:
  - **Entry**: simultaneous **MOC** (market-on-close) orders on both legs, submitted by 15:50 ET. MOC executes at the official 16:00 ET closing print, eliminating intra-auction leg risk. If MOC submission window missed (rare), fallback is two **marketable-limit IOC** orders at 15:55 ET with limit set 5 cents past best offer/bid; if either leg fails to fill, immediately cancel the other leg and skip the trade (no half-trades).
  - **Exit**: MOC by default; if convergence prints intraday on a fast move, can elect to flatten with two simultaneous marketable-limit IOC orders at the same minute. Both legs must fill or neither — confirmed in code via `Trade.fillEvent` callbacks before logging position closed (lesson 4: broker truth).
  - **Stop-out / circuit breaker**: marketable-limit IOC, limit 10 cents past quote on each leg. If a leg rejects (e.g., halted security), submit again at next print and escalate to manual override after 3 minutes.
- **Slippage budget**: **15 bps per round-trip on the pair** (7.5 bps each leg). GLD/SLV spreads at the close are 1-2 cents on ~$180/$24 prints → ~3-8 bps each leg; MOC fills typically run inside the bid-ask. Budget enforced via post-trade comparison: implementation shortfall = actual fill − 15:55 ET mid; if 20-trade rolling avg shortfall > 15 bps, trigger broker / venue review.
- **Reject handling (lesson 4)**: every order is logged with `orderId`, both legs are tracked, and the position state is reconciled to IBKR's `reqPositions()` blotter once per minute while in-trade and once per fill. If broker quantity diverges from local cache, the broker number wins and we re-place the missing-side order at market within 10 seconds.

## 9. Capacity

- **AUM ceiling**: ~**$200 million NAV** before slippage on the ETF version meaningfully degrades the edge. Reasoning:
  - At full-strategy sizing (100% gross NAV), a $200M deployment moves $100M each in GLD and SLV.
  - GLD ADV $1.5 B / day → $100M = ~7% of ADV → expected impact at MOC ~5-10 bps (well-studied).
  - SLV ADV $300 M / day → $100M = ~33% of ADV → expected impact at MOC ~25-40 bps **plus** auction skew during stressed days. This is the binding constraint — the SLV leg blows the slippage budget above ~$200M.
  - Above $200M: must migrate to GC/SI futures, where COMEX ADVs are $30 B (GC) and $4 B (SI), giving a futures-version ceiling ~**$1.5 B NAV**.
- **% of ADV at full size, $100k account (the default)**: $50k / $1.5B ≈ 0.003% GLD, $50k / $300M ≈ 0.017% SLV. Negligible.
- **Concentration risk**: single-pair strategy = 100% gross in one metals pair. There is no diversification *within* this strategy slot; diversification is at the portfolio-of-strategies level (correlation rule §7).
- **Where the edge degrades**: above $200M ETF / $1.5B futures, slippage eats the ~80-150 bps expected gross per trade. Also degrades if industrial silver demand becomes the dominant driver and decouples it from monetary gold — see failure modes §10.

## 10. Failure modes

1. **Structural shift — industrial silver demand spike (solar / EV)**:
   - **What it is**: silver photovoltaic demand has grown from ~50 Moz in 2014 to ~200 Moz in 2024 (Silver Institute) and is projected to exceed 300 Moz by 2030. EV and electrification add another ~50 Moz/yr potential. If industrial demand rises fast enough to drag silver structurally up vs gold, the long-run ratio mean shifts *down* (silver gets more valuable), and the 252-day rolling mean lags the structural shift by ~12 months.
   - **What it looks like in metrics**: average ratio drifts persistently below the 5-year rolling mean; entries on z < −2 (long ratio = long gold / short silver) fail to converge within 90-180 days; half-life of convergence balloons to > 200 days. A run of 3+ consecutive non-converging long-ratio trades is the smoking gun.
   - **Action**: kill criterion in §11.
2. **Monetary regime shift — return to a gold-standard-adjacent monetary anchor**:
   - **What it is**: central banks formally adopting gold-backed settlement (BRICS gold-pegged unit, dollar collapse scenario), or US fiscal crisis driving gold to a structurally re-rated price level.
   - **What it looks like in metrics**: ratio breaks above its 1990-2015 99th percentile (~95) and *stays* there. Z-scores compute as +3 / +4 routinely. Time to convergence on short-ratio trades > 250 days.
   - **Action**: the 250-day forced-exit and z > 3.5 stop-out catch this on a per-trade basis; the 90-day "verdict horizon" §11 catches it at the strategy level.
3. **Central bank gold buying surge (current 2022-2026 regime)**:
   - **What it is**: official sector net gold purchases ran ~1,000 tonnes/yr 2022-2024 (WGC), ~2× the prior decade. This raises the gold floor without a matching silver bid → ratio drifts persistently *up*.
   - **What it looks like in metrics**: ratio above its long-run mean for years (2022-2026 print: ratio ~80-100 vs 1990-2015 mean ~65). Z-scores against a 252-day window normalize quickly (the rolling window adapts) — but the *trade-level* P&L on long-ratio entries is fine while short-ratio entries become rare. The danger is the strategy under-trades and produces too few OOS samples to validate.
   - **Action**: track number of trades/year as a separate sanity gate; if < 2 trades/year for 2 consecutive years, the rule is mis-specified for the new regime and parameters need re-fitting (offline; out of cycle).
4. **Liquidity flight — SLV-specific event**: a Volcker-rule-style ETF halt on SLV (rare but precedented during the 2021 short squeeze, when SLV briefly traded with $1 wide spreads) would prevent exit. **Metric**: SLV bid-ask > 10 cents intraday → flatten via 50% futures hedge proxy (SIL micro-silver) and stand down on SLV trading.
5. **Borrow recall (operational)**: covered in §7 and §12.

## 11. Falsifiable verdict criteria

The assignment notes correctly that 90 days is short for a strategy that trades 3-6 times per year. The 90-day window therefore cannot pass/fail on Sharpe in isolation; it can pass/fail on **regime-consistency** and **operational integrity**.

What 90 days CAN falsify:
- A trade entry triggers but the ratio diverges further by > 2σ within 60 days without retracing = regime shift suspicion → escalate.
- A trade entry triggers and does not converge within 60 days at all (no progress) = below-historical-half-life → escalate.
- The strategy fails to trigger any entry in 90 days despite ratio z-score breaches in the data = signal-pipeline or data-feed bug → kill.
- Operational issues (broker-side fills don't match local cache, borrow recall not handled, MOC submission failures > 5%) = kill.

**Verdict thresholds**:

- **Promote to live capital** (after 30-day shadow + first full trade roundtrip):
  - At least one full entry → exit cycle observed in shadow, OR signal pipeline confirmed correct via parallel walk-forward replay with broker-quote data.
  - No broker-side fill divergence > 0.5% NAV.
  - Slippage on entries ≤ 20 bps round-trip on the pair.
  - Walk-forward 2016-2026 OOS metrics on the candidate code (not just paper-spec) re-confirmed: ≥ 30 trades, Sharpe ≥ 0.7 (bootstrap CI excluding 0.4), median convergence ≤ 90 days, max DD ≤ 15%.
- **Continue shadow** (90-day verdict ambiguous):
  - 0 or 1 trades fired in 90 days AND no rules violated AND OOS replay still passes → continue shadow up to 270 days. Slow strategies need patience; this is explicit.
  - 1 trade in progress, days-in-trade ≤ 60, z-score retracing → continue.
- **Kill immediately** (any one trips):
  - Trade entered and z-score moves to > 3.5σ within 30 days of entry without retracing.
  - 3-day NAV DD > 6% from the pair.
  - Daily loss cap (−4% NAV) breached.
  - Broker-side fill discrepancy > 0.5% NAV on any leg.
  - Forced close at 250 trading days hit on a trade — full strategy review (not just trade review); 2 such forced closes in 2 years = strategy retired.
  - Signal pipeline fails to trigger on a known historical z > 2 breach replayed against live data feed = data integrity broken.

## 12. One-paragraph operational risk note

The pair structure means we must hold a real short-borrow on the short leg every day in trade — IBKR's stock-loan inventory for GLD and SLV is reliably deep (both are creation-redemption ETFs with millions of shares routinely lendable), but a coordinated short squeeze on SLV (precedented in early 2021) can spike the borrow rate from ~30 bps to 200-500 bps in 48 hours and rare-but-precedented forced-buy-in events do happen. Mitigation: monitor borrow rate daily via `reqAccountSummary`, alert if > 100 bps; have the ZSL substitution plan in §7 pre-coded and tested in paper. Secondary operational surface: the LBMA fix series we use for 1990-2004 in-sample data is a free download that has changed format twice in the last decade; if the format breaks we lose pre-ETF history (but not the OOS evidence, which is post-2016 and ETF-only). Third surface: SLV's trust structure means the trust can theoretically suspend creations (it did briefly in 2008 and 2021), which is not a halt but creates a structural premium-to-NAV divergence that breaks the price-based ratio for ~2-5 days; if observed, flatten and stand down until the premium normalizes (< 30 bps). Final surface: IBKR data feed for COMEX futures is a separate subscription; if it lapses, the futures lane is offline but the ETF lane is unaffected — futures is an opt-in scale-up, never load-bearing for the validation window.

---

## Appendix — Address every numbered lesson (mandatory per spec)

1. **Validate before deploy**: 1990-2015 in-sample, 2016-2026 OOS (≥ 30 trades), 30-day shadow with parallel walk-forward replay before any live capital. ✓
2. **Binary gates, not confidence rankers**: this is a pure rules-based threshold strategy; no ML confidence used. The z-score itself is a continuous variable but the trade decision is binary (z > +2, z < −2, |z| < 0.5). No sizing is conditioned on z magnitude beyond the binary entry. ✓
3. **TP:SL asymmetry is structural**: explicitly addressed per assignment — **mean-reversion exit logic embeds the TP:SL structure**. The "take profit" is convergence to |z| < 0.5 (the model-implied fair value); the "stop loss" is divergence to |z| > 3.5 (the model-broken threshold) or 8% NAV adverse MTM (the risk-based stop) or 250 trading days (the time-based stop). The expected payoff geometry is: avg winner ≈ +1.5σ of ratio (entry at 2σ, exit at 0.5σ); avg loser ≈ −1.5σ of ratio (entry at 2σ, stop at 3.5σ). This **looks** symmetric in σ-space but is asymmetric in *probability*: historical conditional probability of mean-reversion from a 2σ deviation is ~75% within 120 days, so the expected value is positive even at 1:1 σ-magnitude. The 100/35 vs 40/20 lesson (asymmetric stops are part of the edge) is honored — we are not running this with a 1σ stop and 0.5σ target, which would invert the win rate. ✓
4. **Broker truth, not local cache**: §8 mandates `reqPositions()` reconciliation once per minute while in-trade and on every fill; broker quantity wins on divergence. ✓
5. **No PDT-exposed lanes on sub-$25k US equity accounts**: explicitly **N/A per assignment**. However for completeness: the strategy holds positions for 30-90+ days; round-trips per 5 business days are far below 4, so PDT does not bind even on a $10k account. The Reg-T short requires margin which IBKR offers above $2k; the strategy is operable down to ~$25k NAV where the SLV short-borrow becomes mechanically feasible at decent fee rates. ✓
6. **One strategy per slot**: the slot is "metals / precious-metals mean reversion". This is the only such candidate. Iter 1-5 do not touch metals (small-cap overnight equities, ES futures intraday, BTC perp funding, G10 FX carry, VIX term-structure). No collision. ✓
7. **No same-data backtest**: 1990-2015 calibration + 2016-2026 OOS, walk-forward 5y/1y/1q step with 30-day embargo and 250-day trade-level purge. ✓
8. **Battle-tested broker SDKs only**: IBKR + `ib_insync`. ✓
9. **Capacity bound named upfront**: $200M NAV (ETF lane, SLV-binding) / $1.5B NAV (futures lane). §9. ✓
10. **Kill criteria written before deploy**: §11 lists exact numeric thresholds for kill, including the slow-strategy-specific "what 90 days can and cannot falsify" framing. ✓
