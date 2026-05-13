# Iteration 24: SVI Implied-Volatility Surface Relative-Value (Single-Name + SPX/SPY Node-Deviation Arb)

## 0. Differentiation from iter_5 (VIX rotation) and iter_17 (earnings vol) — read me first

This is a **vol-of-vol surface-shape strategy**, not a vol-level rotation and not an event book. The three iterations look adjacent because they all involve implied volatility, but they trade **completely different objects** with **disjoint edge sources**:

| Axis | Iter 5 (VIX rotation) | Iter 17 (Earnings straddle) | Iter 24 (Surface RV, this) |
|---|---|---|---|
| What's traded | SVXY ETF (a single scalar exposure to front-month VIX futures roll) | One straddle on one underlier, around one scheduled event | A delta-neutral *spread* between two nodes on the same name's IV surface, away from any event |
| Object of view | Scalar: VIX/VIX3M term-structure ratio | Scalar: realized vs implied over 2-day window | Vector/surface: 5-parameter SVI fit per slice × multiple slices |
| Decision horizon | Daily, hold weeks-to-months | Event-driven, hold 1-2 days | Daily refit, hold 2-14 days |
| Edge source | VRP / roll-down of front-month VIX futures | Realized > implied around earnings on filtered universe | Cross-sectional mispricing between nodes on the *same* surface — skew arb, calendar arb, butterfly arb |
| Greek profile | No options; ETF beta to VIX futures | Long gamma, hedged delta, full vega exposure to crush | Near-zero net gamma and near-zero net vega *at the portfolio level*; long the cheap node, short the rich node, delta hedged |
| Universe count | 2 tickers (SVXY, SGOV) | ~25 names, ~4× per year per name = 100 events/yr | ~30 names + SPX/SPY, evaluated every day = ~7,500 surface fits/yr |
| Data cost | $12/month | $40-80/month (option chain pulls 2 days/event × 25 events/quarter) | **$5,000/year (ORATS Wheel or Cboe DataShop)** — by far the most data-intensive iteration after iter_22 |
| Slot rule | Filled by Iter 5 if deployed | Filled by Iter 17 if deployed | Filled by Iter 24 if deployed — none of the three conflict |

Iter 5 trades **the level** of vol. Iter 17 trades **the spike** of vol around a known event. Iter 24 trades **the shape** of the vol surface against a parametric reference, agnostic to whether the overall level is going up or down. The three are statistically near-orthogonal: a fall in VIX level (Iter 5 P&L) need not flatten 25-delta put skew (Iter 24 P&L), and a positive earnings gamma scalp on AAPL (Iter 17 P&L) need not move the SPX surface SSVI parameters at all.

Co-deployment of all three is permissible after correlation checks (see §7). The three-way correlation is what would justify running all three; the worst case is they collide on a single-name vol crush, and that's a sizing-limit problem, not a slot-collision problem.

---

## 1. Thesis (one sentence)

For a daily-refit parametric implied-volatility surface (SVI per slice, SSVI across slices) on the top 30 US single-name options underliers plus SPX and SPY, **nodes (specific strike/expiry combinations) whose market mid IV exceeds the fitted surface IV by ≥ 5 absolute volatility points or falls below it by ≥ 5 absolute volatility points, conditional on the node passing a liquidity gate (option ADV ≥ 500 contracts, bid-ask spread ≤ 3% of mid, no scheduled name-specific event in the next 21 days), revert toward the fitted surface within 2 to 14 calendar days more often than they continue to diverge** — capturing this reversion via a delta-hedged, vega-neutral spread (short the rich node, long the cheap node on the same surface) produces a residual edge that survives liquidity costs and the cost of carry on the delta hedge.

Falsifiable form: **Over a 24-month OOS window (2024-05 → 2026-05) using a frozen SVI fitter and frozen 5-vol-point trigger threshold, we observe ≥ 300 closed surface-deviation trades; the bootstrap mean per-trade P&L (gross of commissions, net of half-spread on both legs at entry and exit) is ≥ +6 bps of gross vega notional with a 95% block-bootstrap CI excluding 0; the hit rate (% of trades positive) is ≥ 53%; and no single trade exceeds -300 bps of vega notional as a tail loss. If any of those three conditions fails, the surface-shape mispricing is either (a) market-maker arbed away as desks tightened, (b) an artifact of the SVI parameterization rather than a real mispricing, or (c) dominated by gap-risk / hedge-breakdown loss that delta-hedging cannot recover — kill.**

## 2. Market / asset class / venue

- **Asset class**: Listed US single-name and index equity options. ATM-anchored vol surface trades use butterflies, risk-reversals, calendar spreads, and box-like 4-leg structures (defined per trade type in §4).
- **Underlying universe (frozen for training; refreshed quarterly in deployment)**:
  - **Tier-1 single names (n=30)**: top 30 US single-name options underliers by 90-day options ADV in dollar terms. Approximate static membership for 2024-2026 training: AAPL, MSFT, GOOG, META, AMZN, NVDA, TSLA, NFLX, AMD, AVGO, CRM, ORCL, ADBE, INTC, QCOM, MU, COIN, SHOP, SNOW, CRWD, PLTR, UBER, ABNB, BAC, JPM, XOM, CVX, GE, BA, WMT.
  - **Index leg**: SPX (cash-settled, European, Section 1256), SPY (American, share-settled), VIX options (cash-settled, European). VIX options are an explicit *node* on the broader vol-of-vol manifold; included as one source of surface fits.
- **Forbidden universe / windowing**:
  - Names with confirmed earnings inside the next **21 calendar days** — the surface will price an event-vol bump that SVI cannot fit smoothly, and the "mispricing" is a model artifact, not a tradable signal. (This is the explicit hand-off boundary with iter_17: iter_17 *trades* the event window; iter_24 *avoids* it.)
  - Names with confirmed M&A activity, FINRA OTC rumor flags, FDA Adcom dates, regulatory rulings, or any binary corporate-action date inside the next 21 calendar days.
  - Names whose **30-day ADV in *options premium* is below $5M** — surface fits become unreliable when fewer than ~80 listed strike/expiry combinations exist.
  - **VIX options** when the front VIX future is within 5 trading days of expiry — the cash-vs-futures basis dominates the surface, not the mispricing.
- **Primary exchanges**: CBOE (BZX Options, C2, EDGX), NYSE Arca Options, Nasdaq ISE / PHLX / MRX, MIAX. IBKR Smart Routing handles destination. SPX cash settlement is CBOE-only.
- **Broker**: Interactive Brokers (IBKR) Pro via `ib_insync`. Same single-broker requirement as iter_17 (lesson 8). Combo orders for multi-leg structures use IBKR's BAG/Combo contract construction.
- **Settlement / margin**:
  - SPX/VIX cash-settled European options: **Section 1256** (60/40 long-term/short-term split, mark-to-market at year end). Margin is Reg-T portfolio margin if account qualifies, or strategy-based otherwise.
  - SPY, single-name options: short-term capital gain treatment per leg. American exercise (early exercise risk on dividend-paying names if assigned through ex-div).
  - Delta-hedge leg on the underlying: Reg-T long-or-short, T+1 cash. Short locate required for any vol-trade leg that lands long a put-heavy structure; **HTB rate ceiling at entry = 50 bps/yr** (looser than iter_17 because the surface trade is multi-day and the borrow cost is amortized over the holding period).
  - **Portfolio margin strongly preferred** — strategy-based margin will refuse some butterfly+calendar combos that are obviously delta-and-vega neutral but get charged as if they were naked. **Required account equity for portfolio margin at IBKR = $110k.** This iteration thus has a **higher minimum-equity gate** than iter_5 or iter_17; argued at length in §7 and §9.
- **Tax regime**:
  - SPX/VIX: 1256, year-end mark-to-market.
  - SPY/single-names: short-term capital gain per leg.
  - Wash-sale rule across substantially-identical instruments. Wash-sale risk is meaningful here — many trades close at a loss within 30 days of opening another trade on the same underlier; tax-lot accounting is mandatory and the operator's tax preparer must be flagged.
  - No Section 988.

## 3. Time horizon

**Daily evaluation; multi-day to 2-week holding period; intraday delta-hedge tick.**

- **Surface refit**: every trading day at **16:15 ET** (15 minutes after the equity options close), using the closing NBBO mid of every quoted strike across every quoted expiry on every name in the universe. SPX uses 15:15 ET (CBOE close of regular hours, prior to extended VIX-products session).
- **Trade decision tick**: same daily refit. Trades sized and submitted **next morning at 10:00 ET** (after the open is settled and underlying spot has stopped gyrating from the cash-vs-futures basis).
- **Holding period**: minimum 2 trading days (allows for at least one overnight to clear intraday noise), maximum **14 calendar days** OR until the deviation has reverted to within ±1 vol point of the fitted surface, whichever fires first. Hard exit by day 14. A *separate* hard exit if a name in the active book has a newly-scheduled event announced inside the next 14 days (M&A pop, earnings reschedule, sudden Adcom date).
- **Delta-hedge tick**: every **30 minutes during RTH**, or whenever the underlying has moved **0.5%** since the last hedge — whichever fires first. Delta-hedge cadence is **looser** than iter_17 because the surface trade is not a gamma scalp; the gamma at the portfolio level is engineered to be small. The hedge is to neutralize unintended directional drift, not to extract gamma P&L.

## 4. Method family and specific algorithm

### 4.1 Family

**Rules-based parametric model with frozen functional form (SVI / SSVI), threshold trigger, deterministic position construction, deterministic delta hedge.** No ML in the entry decision. The fitter has parameters and the threshold has parameters, but both are *hyperparameters* (fitted on train, frozen for OOS) not free knobs in deployment.

### 4.2 The SVI model (per-slice, 5 parameters per expiry)

Gatheral's raw SVI parameterization for total implied variance `w(k) = σ²(k) × T` as a function of log-moneyness `k = log(K / F)`:

```
w_SVI(k; a, b, ρ, m, σ) = a + b × [ ρ × (k − m) + sqrt((k − m)² + σ²) ]
```

with constraints
- `a + b × σ × sqrt(1 − ρ²) ≥ 0` (positivity of total variance)
- `b ≥ 0`
- `|ρ| < 1`
- `σ > 0`
- Lee's no-arb wing constraint: `b × (1 + |ρ|) < 4/T`

Five parameters per expiry slice. Typical fit residual on liquid SPX slices is **< 0.3 vol points** at 95% of strikes within ±2 standard deviations of the at-the-money forward.

### 4.3 SSVI (Surface SVI) — Gatheral & Jacquier 2014

For the cross-slice consistency check, fit SSVI:

```
w_SSVI(k, T) = (θ_T / 2) × { 1 + ρ × φ(θ_T) × k + sqrt((φ(θ_T) × k + ρ)² + (1 − ρ²)) }
```

with `θ_T` = ATM total variance at expiry T (the "term structure of variance"), `φ(θ)` = a power-law function of θ. Three parameters control the full surface: ρ (skew), and two parameters in φ (typically `η` and `λ`). SSVI guarantees calendar-spread and butterfly no-arb by construction across all (k, T) — calibration must satisfy `0 < ρ × φ < 4`, `(ρ × φ)² ≤ 4`, and `θ × φ' / φ ≥ -1` (Gatheral-Jacquier).

We fit **SVI per slice** for the per-expiry skew/butterfly trades and **SSVI across slices** for the term-structure (calendar) trades. The two fits must agree to **within 0.5 vol points** at the ATM nodes of every slice; if they disagree, the surface is in a *unfittable* state (typically right around an earnings reschedule or M&A rumor) — **abort all trades on that name for the day**.

### 4.4 Step-by-step algorithm

1. **EOD surface refit (16:15 ET nightly)**:
   - Pull NBBO closing mid for every listed strike on every listed expiry (out to 90 DTE) for each of the 30 single names + SPX + SPY + VIX. ORATS Wheel or IBKR `reqMktData` snapshot at 16:15.
   - Compute Black-Scholes implied vol from each option mid using continuously-compounded zero rate from Treasury curve and forward dividends from IBKR `reqMktData(genericTickList="106")` (the FwdRate / DivYield tick).
   - Drop nodes whose: bid ≤ $0.05, bid-ask spread > 50% of mid, option ADV < 500 contracts, OR open interest < 200.
   - For each expiry with ≥ 7 surviving strikes, fit SVI by Levenberg-Marquardt nonlinear least squares, weighted by inverse vega × inverse spread (small-vega and wide-spread nodes contribute less to the fit). Constrain parameters to the no-arb cone.
   - Fit SSVI across all surviving slices with ≥ 30 surviving (k, T) nodes total.
   - Persist all fit parameters and residuals to a daily Parquet snapshot under `data/surface_fits/{date}/{symbol}.parquet`.

2. **Node-deviation scan (16:30 ET nightly)**:
   - For each (symbol, expiry, strike) node, compute `ΔIV = market_IV − fitted_SVI_IV` and `ΔIV_SSVI = market_IV − fitted_SSVI_IV`.
   - **Trigger candidates**:
     - **Skew arb**: a 25-delta put with `ΔIV > +5 vol points` AND the matched 25-delta call with `ΔIV < +1 vol point` (rich put, fair call) → short risk-reversal (short 25Δ put, long 25Δ call), delta-hedge.
     - **Butterfly arb**: an OTM strike with `|ΔIV| > 5 vol points` while the two adjacent listed strikes have `|ΔIV| < 2 vol points` → 3-strike butterfly trade in the opposite direction of the deviation. (Long fly if the central strike is *cheap*, short fly if *rich*.)
     - **Calendar arb (term-structure)**: an ATM strike on one expiry with `|ΔIV_SSVI| > 5 vol points` while the same strike on the adjacent expiry has `|ΔIV_SSVI| < 2 vol points` → calendar spread.
   - For each candidate, score by `|deviation| / spread_cost` where `spread_cost = sum of (½ × bid-ask) × vega per leg`. **Keep candidates with score ≥ 3.0** (expected deviation reversion ≥ 3× round-trip spread cost).
   - Hard limit: **maximum 5 new trades per day across the entire portfolio**. Pick the top 5 by score. (Limits surface-fit noise from generating a flood of correlated trades and limits intraday execution operational burden.)

3. **Position construction (10:00 ET next morning)**:
   - Per candidate, target position **vega-neutral at the portfolio level** by sizing the two legs so net vega = 0 to within ±5% of leg-1 vega. (Skew arb is natively close to vega-neutral; butterfly arb is natively close; calendar arb is sized to match ATM vegas.)
   - Per candidate, target **delta-neutral at entry** by including an underlying stock or futures hedge sized to flatten the entry delta to ±0.05 of underlying-equivalent shares.
   - Per candidate, target dollar **vega notional = min(0.5% NAV, 0.05% of options-side aggregate ADV × premium)**. The latter is the *capacity floor* — we never take more than 5 bps of the underlier's daily options premium flow in a single trade.
   - **Max gross vega across portfolio = 5% NAV per vol point** (i.e., a +/- 1 vol point parallel shift of the entire surface produces ≤ 5% NAV move). This is the central risk parameter; argued in §7.
   - **Max gross gamma across portfolio = 0.3% NAV per 1% underlying move** (the residual gamma after pairing legs is intentionally near zero, this just enforces it).
   - **Max single-name concentration = 1% NAV gross vega** on any one underlier; **max sector concentration = 2.5% NAV gross vega** by GICS.

4. **Order submission**:
   - All multi-leg structures sent as **IBKR Combo (BAG) limit orders** with one ticket, **all-or-none flag set**, limit price at NBBO mid of the combo. (Lesson 8: battle-tested SDK; `ib_insync` Combo order handling has been stable since 2019.)
   - If unfilled in 60 seconds, step the limit one tick toward the natural side; retry up to 3 times.
   - If still unfilled, **cancel and abandon the trade for the day** — partial-leg fills are forbidden under any circumstance. (Lesson 4: broker truth; never accept a partial fill that leaves naked legs.)

5. **Delta-hedge loop (every 30 min during RTH or 0.5% underlying move)**:
   - Repeatedly compute portfolio Greek delta via re-priced Black-Scholes on each leg using the live NBBO mid and live underlying spot.
   - If `|net_delta_$| > 25 bps NAV`, submit a marketable-limit IOC order in the underlying (or equity micro-future for cash-efficient hedging on SPX where ES is a tighter alternative) to flatten to zero delta.
   - Log every hedge: timestamp, pre-hedge delta, post-hedge delta, fill, slippage.

6. **Exit decision (daily, evaluated at 15:30 ET)**:
   - At each daily 15:30 ET tick after entry, re-evaluate the surface fit for the names with open trades.
   - **Exit condition A (reversion target)**: `|ΔIV|` at the rich/cheap node has fallen below 1 vol point → close the trade.
   - **Exit condition B (time stop)**: the trade has been open ≥ 14 calendar days → close regardless.
   - **Exit condition C (event eruption)**: a previously-unscheduled event (earnings reschedule, M&A announcement, FDA panel) has appeared on a name in the open book → close immediately, do not wait for next 15:30 tick.
   - **Exit condition D (model breakdown)**: SVI and SSVI fits on a name diverge by > 0.5 vol points at ATM → close the name's positions, mark the day's model as untrustworthy.
   - Close via Combo limit at NBBO mid → 3 tick steps → market order (mirror of entry). Underlying hedge flattens simultaneously.

7. **Daily reconciliation (17:00 ET)**:
   - Reconcile every leg in IBKR positions vs the local book. Reject any close-of-day where local book quantity ≠ IBKR quantity (lesson 4). Halt new trades next session until reconciliation passes.

### 4.5 Why rules, not ML

The edge thesis is **shape mispricing relative to a parametric reference**. SVI / SSVI are *the* practitioner-standard parameterizations, with explicit no-arb constraints derived from theory (Gatheral & Jacquier 2014; Roper 2010). Adding an ML overlay (e.g., gradient boost the residual after the SVI fit) is a recipe for fitting noise; the SVI residual is by construction the part of the surface that the parametric model could not capture, and there is no a-priori reason that residual is *predictably* serially correlated. Lesson 2 (binary filter, not confidence ranker) explicitly favors a threshold-based deviation trigger over a continuous "deviation score → bet size" regression. The 5-vol-point trigger is binary; sizing is fixed at the vega-neutral target.

*Anchors*: Gatheral 2004 *The Volatility Surface: A Practitioner's Guide*; Gatheral & Jacquier 2014 *Arbitrage-free SVI volatility surfaces*; Bergomi 2015 *Stochastic Volatility Modeling* (ch. 7-8 on surface arbitrage); Bakshi, Kapadia & Madan 2003 *Stock return characteristics, skew laws, and the differential pricing of individual equity options*; Carr & Wu 2009 *Variance Risk Premiums*; Roper 2010 *Arbitrage-free Implied Volatility Surfaces*; Cont & da Fonseca 2002 *Dynamics of implied volatility surfaces*.

## 5. Data requirements

### 5.1 Live data feed (deployment)

- **Options chain pulls**: Every 30 minutes during RTH for every name in the open book + every name with a candidate trigger from last night's scan. Each name pulls 6-12 expiries × 30-80 strikes per expiry = 200-800 quotes per pull. With 30 names + SPX + SPY + VIX, total quote volume = **8,000-30,000 NBBO mid quotes per 30-minute tick** during RTH.
- **EOD surface fit pulls**: Once per day at 16:15 ET, pull *every* listed strike out to 90 DTE on the full universe. **80,000-150,000 NBBO mid quotes nightly.**
- **Underlying spot**: Continuous via IBKR `reqMktData` on the 32 underlying symbols.
- **Treasury zero curve**: Daily snapshot from CME / Treasury Direct.

### 5.2 Historical data (training & OOS)

- **In-sample (training)**: 2015-01-02 → 2020-12-31. Six years. Covers low-vol grind (2015-2017), Volmageddon (Feb 2018), Q4-2018 selloff, COVID shock (Mar 2020), and post-COVID vol normalization. **Long enough vol-regime variation** as required by the assignment.
- **Out-of-sample**: 2021-01-02 → 2026-05-12. ≈ 5.4 years.
- **Required fields per option**: NBBO bid/ask, last trade, volume, open interest, exercise style, settlement type, dividend ex-date (for American options on dividend payers).
- **Source — the data-cost reality**:
  - **ORATS Wheel API**: end-of-day options data (NBBO close, IV, Greeks, surface fits) for full US options universe; **$5,000/year** retail tier, $15-25k for larger named-user tiers. ORATS publishes its own SVI fits; we will *not* use ORATS-fitted IVs and refit ourselves to control the parameterization. Used only for raw NBBO mids and metadata.
  - **Cboe DataShop**: full OPRA tape historical, ~$3,000-8,000 for the 2015-2026 historical pull for the 30-name universe; live OPRA feed costs separately. Higher quality than ORATS but materially more expensive on the live feed.
  - **IBKR options snapshot via `reqMktData`**: free with IBKR account, but is **delayed by 15 minutes** unless one pays for the relevant exchange feeds (~$5-20/month per exchange; CBOE One Bundle covers most needs at $22/month). Live IBKR options data is *good enough for live trading* but the historical archive via IBKR is thin — historical chain snapshots from IBKR are not commercially available in bulk.
  - **Polygon Options** (paid tier): $99-249/month for 1-min and EOD options bars; viable as a cheaper-than-ORATS option for the retail-scale deployment but data quality is reportedly weaker on illiquid strikes.
- **Working assumption for this iteration**:
  - **Historical (2015-2026 backtest)**: ORATS Wheel API at $5,000/year, used for closing NBBO + open interest + volume per (symbol, expiry, strike, date). **Already paying for historical** is the gate. If we cannot pay this, the iteration cannot be properly OOS-tested and must be **killed at the spec stage**.
  - **Live deployment**: IBKR live options feed + CBOE One Bundle ($22/month) for the live surface refit. ORATS Wheel API live tier ($5k/yr) as the *redundant secondary* feed for cross-validation — if IBKR live IVs disagree with ORATS by > 1 vol point at the ATM node, halt new trades and investigate.
- **Storage**: ORATS Wheel historical for 30 names × ~250 days/yr × 11 yrs × ~150 quotes/day = ~12M rows raw; compressed Parquet ≈ **15-30 GB**. SVI fit results (5 parameters × ~10 slices × 30 names × 2,700 days) ≈ **10 MB**. Total ≈ 30 GB.
- **Cost summary (live deployment, monthly)**:
  - IBKR options data (CBOE One Bundle): **$22/month**.
  - IBKR US Securities Snapshot Bundle (underlying spot): **$10/month**.
  - ORATS Wheel API: **$420/month** ($5,040/year).
  - Compute (single AWS / Hetzner box for SVI fitter): **$30-50/month**.
  - **Total: ~$480-500/month, or ~$5,800/year all-in.**
  - Iter_22 (the more data-intensive iteration referenced in the assignment, presumably full-tick-level something) is even costlier; this iteration is **the second-most data-intensive** in the entire iter_1..23 cohort, which is the operational risk flagged explicitly in §12.
- **Latency requirement**: Round-trip < 5 seconds for the 30-min hedge tick. 30-second budget for the EOD surface refit. No HFT requirement.

## 6. Validation plan

### 6.1 Walk-forward design

- **Training window**: 2015-01-02 → 2020-12-31 (six years).
  - Fit hyperparameters once on this window:
    - Deviation threshold (the assignment-specified **5 vol points**) — confirm this is the sweet spot vs. alternatives (3, 4, 5, 6, 7 vol points) by *Sharpe stability* not just Sharpe maximization; if 4 and 6 vp give Sharpe within 0.1 of the 5-vp peak, accept 5 vp as the locked threshold (a flat optimization surface is what we want, not a spike).
    - Liquidity gates (option ADV, spread, OI) — fitted once.
    - Score threshold (= 3.0 × spread cost) — fitted once.
    - Max trades/day (= 5) — fitted once.
    - Holding period max (= 14 days) — fitted once.
- **Out-of-sample**: 2021-01-02 → 2026-05-12 (5.4 years, ≈ 1,350 trading days).
- **Walk-forward refitting** (sensitivity check, not strategy update): refit hyperparameters on a 4-year rolling window quarterly through OOS; check that the locked hyperparameters from the 2015-2020 fit are still within the OOS-rolling-fit's 90% credible band. If not, the strategy's hyperparameters are unstable across vol regimes and the deployment must use the conservative envelope of the rolling fits (= a kill-criterion, see §11).
- **Purge / embargo**: 14-day embargo between train and test (matches the maximum holding period; trades opened in train should be fully closed before any test trade is initiated).
- **Cross-asset purge**: skip surface fits for any name in the universe within 21 days of a confirmed earnings event (per §2). This is the same filter applied in deployment, applied in training.

### 6.2 Minimum OOS evidence

- **Minimum closed surface-deviation trades in OOS**: **300**.
  - Justified by expected trade rate of ~80-150/year on a universe of 30 names + 3 indices with a 5-trades-per-day cap and ~50% of days producing zero triggers in placid vol regimes.
  - Over 5.4 years OOS → expected sample 400-800 trades; the 300-trade minimum is set well below to ensure even an *anemic* trade rate produces a usable evidence file.

### 6.3 Statistical tests and thresholds

- **Bootstrap (10,000 draws, block size = 14 trading days)** of per-trade P&L; require **95% CI of mean per-trade P&L excludes 0**.
- **Sharpe of the equity curve (daily mark-to-market on open positions)** ≥ **0.7** OOS with bootstrap CI excluding 0.4.
- **Hit rate** ≥ **53%** (vs. 50% null) — slim but consistent edge expected.
- **Max single-trade loss** ≤ **3% NAV** in OOS history (= 300 bps of vega notional at the 1% NAV concentration cap).
- **Max OOS drawdown** ≤ **8% NAV**.
- **No 21-day rolling stretch with > 4 consecutive losers > 1% NAV each** (avoids hidden tail bunching).

### 6.4 Pre-deployment shadow

After OOS verdict: **45 calendar-day shadow run** with paper trades to a real `ib_insync` paper account, every fit logged, every order submitted (and cancelled at zero-quantity) to verify Combo order plumbing works without partial-leg disasters. Lesson 1: no live capital before shadow.

## 7. Risk envelope

- **Account size minimum**: **$110k** (IBKR portfolio margin gate). Strategy-based margin is technically possible at lower equity but materially reduces the trade universe to debit-only structures and kills the calendar-spread leg — a substantial degradation that the team would have to accept and re-test.
- **Max position size**:
  - Per trade: gross vega notional ≤ **0.5% NAV**; underlying-equivalent share notional ≤ **2% NAV**.
  - Per name (book-wide): gross vega notional ≤ **1% NAV** across all open trades on that name.
  - Per sector: gross vega notional ≤ **2.5% NAV** by GICS.
  - **Portfolio max gross vega**: **5% NAV per 1 vol point parallel surface shift**. A 5-vol-point spike of the entire surface (rare; March 2020 saw ~+15 vp on SPX) translates to ≤ 25% NAV gross vega exposure; the *delta-and-vega-neutral construction* means the realized P&L on such a parallel shift is bounded by the residual second-order Greeks (vanna, vomma) and is far smaller than 25% — but this is the hard ceiling for sizing.
  - **Portfolio max gross gamma**: **0.3% NAV per 1% underlying move**.
- **Daily loss cap**: **2% NAV absolute** (hard halt all new trades for the rest of the day; existing positions held under the standard exit rules unless any exit condition C/D fires).
- **Drawdown trigger**: 5% NAV from any running peak → reduce gross vega budget to 50% of normal for 30 days; 8% NAV from any peak → halt new trades pending human review (kill candidate; see §11).
- **Circuit breaker — vol regime**: if **VIX > 35** intraday, no new surface-deviation trades opened; existing positions held to normal exit rules. (Surface fits get unreliable when VIX is in panic mode; the SVI residuals widen and the "mispricing" trigger fires on every node.)
- **Circuit breaker — broker-side**: any IBKR Combo order rejected with reason involving "leg unavailable" or "exchange short locate failure" → halt new trades on that name for 24 hours.
- **Correlation rule (against already-deployed strategies)**:
  - vs. iter_5 (VIX rotation): expected daily-return correlation = **low-to-moderate (|ρ| ≤ 0.35)**. Iter_5 is long the VRP on SVXY; iter_24 is *vega-neutral* at the portfolio level. They share exposure to vol-regime *level* only through the residual vol of the surface RV (vomma, vanna). Cap **combined NAV at risk = 30%** if both deployed.
  - vs. iter_17 (earnings straddle): expected daily-return correlation = **low (|ρ| ≤ 0.25)**. Iter_17 is single-event long-gamma; iter_24 explicitly excludes the 21-day pre-event window. They cannot trade the same name in the same calendar window. Cap **combined NAV at risk = 25%** if both deployed.
  - vs. iter_2 (ES intraday), iter_18 (US/HK dual-list), iter_14 (LightGBM cross-section): expected correlation **~ 0**. No portfolio cap needed beyond the per-strategy cap.

## 8. Execution plan

- **Broker / SDK**: IBKR Pro via `ib_insync`. **No third-party EMS, no hand-rolled HTTP** (lesson 8).
- **Order type**:
  - All multi-leg structures: IBKR Combo (BAG) limit, all-or-none flag set, NBBO-mid limit. Step toward the natural side every 60 seconds, max 3 steps.
  - Underlying delta hedge: marketable-limit IOC, 1 cent price improvement against the spread.
- **Expected slippage budget**:
  - Combo legs: average **5-12 bps of vega notional per leg per round trip** based on practitioner benchmarks for top-30-ADV options (Jain & Joh 1988; Easley/O'Hara on options market microstructure; recent Cboe market-quality reports). The 5-vp deviation threshold and the score-≥-3 filter exist precisely to ensure the expected mean-reversion bp gain exceeds this slippage by a comfortable margin.
  - Underlying hedge: **1-3 bps per turn** on top-30-by-ADV underliers.
  - **Measurement**: log NBBO mid at order submission and post-fill execution price for every leg; weekly slippage attribution report against the 5-12 bps budget.
- **Broker rejection handling**:
  - Combo rejection (e.g., one leg illiquid): cancel all legs, log reason, **do not retry that combo for the day**. (Partial-leg orphans are the #1 operational nightmare in surface trading.)
  - Hedge order rejection: retry once with broader limit; if still rejected, halt all new trades and pull risk down (close oldest open position first) until broker connectivity restored.

## 9. Capacity

- **AUM ceiling before edge degrades**: **$25M-50M** estimated.
  - Reasoning: the cross-sectional surface mispricings live in the wing strikes and far-month calendars where option ADV in *contracts* is 1,000-5,000/day per node. Taking 5% of ADV on each leg of a butterfly costs 50-200 contracts per leg; at $100 underlying and a 20-vega butterfly that's $1k-$4k of vega notional per trade. With 5 trades/day and ~0.5% NAV per trade, the maximum NAV that fills inside the 5%-ADV cap is roughly $25M (single-name leg-bound) to $50M (SPX-leg-bound; SPX is deeper).
  - Above this ceiling, leg slippage compounds super-linearly and the 5-vp deviation no longer comfortably exceeds spread cost.
- **% of underlying options ADV consumed**:
  - At $10M NAV, consuming **0.5-2%** of node-level options ADV per trade. Easily absorbed.
  - At $25M NAV, consuming **1.5-5%**. Approaching the ceiling.
  - At $50M NAV, consuming **3-10%**. At the ceiling; tracking error from slippage will be material; tighten the deviation threshold from 5 vp to 7 vp to compensate.
- **Concentration risk**:
  - Single-name max: 1% NAV gross vega (above).
  - Single-sector max: 2.5% NAV.
  - SPX as a single name caps at 2% (vs 1% for single-names) because SPX surface is far deeper and the GICS-sector-neutral structure of the index reduces concentration risk.

## 10. Failure modes

1. **Market-maker tightening (slow alpha decay)**. The 5-vp deviation node count falls by 50% over 18 months. *Metric*: weekly count of trigger candidates trending down >30% YoY; mean per-trade bp gain trending down with no offsetting increase in hit rate. *Response*: tighten threshold to 7 vp (fewer trades, larger per-trade edge); if even 7-vp count keeps falling, the edge is gone — kill.
2. **Surface model misspecification at the wings**. SVI cannot fit a true heavy-tailed surface (e.g., post-COVID, persistent crash-skew); model treats true tail skew as 5-vp deviation and shorts the protective put. Trade losses bunch in periods after vol shocks where the market is right and the model is wrong. *Metric*: residual at 10-delta strikes has structural bias (mean residual > 1 vol point persistently); short-skew trades have sub-50% hit rate. *Response*: add a *no-trade band* for 10-delta and below strikes; or migrate to a more flexible parameterization (Heston, rough Bergomi) at cost of more parameters and overfitting risk.
3. **Gamma risk at market gap**. The delta hedge runs every 30 minutes but a 5%-overnight gap (e.g., COVID Monday, Aug 2024 yen-carry unwind) accumulates unhedged delta. The portfolio gamma cap of 0.3% NAV per 1% move means a 5% gap = ~1.5% NAV loss on gamma alone, possibly more from cross-Greek effects. *Metric*: cumulative loss on gap-day exceeds 2× the modeled gamma loss. *Response*: VIX > 35 circuit breaker (above) should pre-emptively halt new trades before the gap; existing positions held but sized so total NAV at risk on a 7% gap ≤ 4% NAV.
4. **Earnings/M&A whiplash on a name in the open book**. A name we hold a calendar spread on announces an M&A deal mid-trade; the front-month IV explodes, the calendar gets crushed. *Metric*: exit condition C fires more than 5× in a quarter. *Response*: tighten the 21-day forward-event exclusion to 30 days; subscribe to a confirmed-event feed (Wall Street Horizon ~$200/month) and stop relying on free public earnings calendars.
5. **Data vendor IV-calculation methodology change**. ORATS or IBKR changes how IV is computed (e.g., switches to a different forward dividend assumption); historical and live IVs no longer cohere. *Metric*: SVI fit residual mean shifts step-function in a single week; ORATS publishes a methodology note that the operator didn't catch. *Response*: redundant secondary feed (CBOE DataShop or Polygon) catches the divergence within days; halt and rebuild fitter on the new methodology.
6. **Broker raises options margin requirement**. IBKR shifts the portfolio-margin treatment of butterflies/calendars in a way that 2× the strategy's margin usage overnight. *Metric*: gross margin used / NAV jumps > 30% in a single session with no position change. *Response*: forced de-leveraging to 50% gross vega; if persistent, the strategy's NAV-equity efficiency has degraded and the AUM ceiling estimate shrinks proportionally.
7. **Exchange options fee restructure**. CBOE/Nasdaq fee tiers change, raising per-contract cost from ~$0.50 to ~$0.80; round-trip cost on a 4-leg structure jumps from $4/contract to $6.40/contract. *Metric*: realized vs. budgeted slippage gap widens by 15-25%. *Response*: re-fit the score threshold (currently 3.0 × spread cost) to 4.0 × spread cost; expect ~30% drop in trade count.

## 11. Falsifiable verdict criteria

- **Promote to live capital** (all of):
  - 300+ OOS surface-deviation trades closed, bootstrap mean per-trade P&L ≥ +6 bps vega notional, 95% CI excludes 0.
  - OOS Sharpe ≥ 0.7 with bootstrap CI excluding 0.4.
  - OOS hit rate ≥ 53%.
  - Max OOS DD ≤ 8% NAV.
  - 45-day shadow run: zero broker-truth divergence events; combo order fill rate ≥ 85% of submitted; no partial-leg orphans; zero kill-criterion triggers.
  - Operational checklist passed: ORATS Wheel feed live for ≥ 30 days, IBKR-vs-ORATS cross-check agreement ≥ 99% at ATM nodes.
- **Continue shadow** (all of):
  - OOS metrics within 10% of promote threshold but not all met.
  - No kill criterion triggered.
  - Extend shadow by 45 more days; if metrics drift further from threshold, kill.
- **Kill immediately** (any one of):
  - 3-trading-day rolling DD > 5% NAV.
  - Daily loss cap breach (2% NAV intraday).
  - Any broker-truth divergence > 0.5% of position value during shadow.
  - SVI vs SSVI ATM disagreement > 0.5 vol points persists for > 3 days on > 5 names simultaneously (model-breakdown signal).
  - Mean per-trade P&L in last 50 trades flips sign (running 50-trade mean < 0) — alpha decay or model decay.
  - Vendor disagreement: IBKR live IV vs. ORATS live IV > 1 vol point at ATM persists > 5 days on > 5 names (data integrity broken).
  - **90-day kill clock**: if 90 days post-live-deployment, fewer than 50 trades have closed with the modeled-P&L distribution intact, the operational footprint or trigger rate are below modeled — kill and re-spec.
- **90-day falsification** (hard rule from assignment): if any of "fewer than 30 closed trades by day 90," "running cumulative P&L < -3% NAV by day 90," or "fitter throws an exception > 1× per week by day 90" — kill within 90 calendar days of go-live.

## 12. One-paragraph operational risk note

This iteration is **the second-most data-intensive in the entire iter_1..23 cohort** (after iter_22 referenced in the assignment). The operational fragilities are: (a) **ORATS Wheel API at $5k/year is a single-vendor dependency**; if ORATS deprecates the Wheel tier or changes the IV methodology and gives the customary 30-day notice, we must either pay up to Cboe DataShop (~5× the cost) or migrate to a more brittle stitched-together feed across IBKR + Polygon + the operator's own IV recomputation pipeline, which is a multi-week rebuild; (b) the **EOD surface refit pulls 80-150k NBBO quotes nightly** and must complete inside the ~3-hour after-hours window before 9:30 ET next morning — any single-machine compute hiccup, network outage, or rate-limit surprise will skip a day's surface refit and leave the strategy unable to scan for new triggers (existing open trades continue under exit rules); (c) **IBKR Combo (BAG) order construction is finicky** — an out-of-band exchange rule change for one leg's tick size or a broker software update can quietly break combo submissions for hours before the team notices, leaving the day's trades unfilled; (d) **portfolio margin at IBKR is a contractual relationship** that can be revoked unilaterally by IBKR risk team if the account's recent activity looks unusual (and a fast-growing surface book *will* look unusual), instantly forcing the strategy onto strategy-based margin where ~half the open positions become impossible to hold and must be closed at adverse spreads; (e) the operator-facing dashboards must distinguish between a *no-trigger day* (terminal-idle, expected) and a *broken-fitter day* (real bug) without false alarms — a stale `surface_fits/{date}/` directory is bug-or-feature depending on whether the prior session produced triggers, and the dashboard must encode that distinction explicitly per the project's `/loop` convention (`cache_complete` is terminal, stale status is only a bug if the dashboard mislabels it as hung).

## 13. Lessons compliance (all 10)

1. **Validate before deploy** — 2015-2020 in-sample, 2021-2026 OOS (300+ trades min), 45-day shadow before live (§6.4).
2. **Binary gates, not confidence rankers** — 5-vp deviation threshold is binary; score-≥-3.0 filter is binary; no ML confidence-to-size mapping (§4.5).
3. **TP:SL asymmetry is structural** — reversion target (ΔIV → 1 vp) is the take-profit; 14-day time stop + 300-bp-of-vega max-loss exit is the structural stop; tested as part of the strategy, not a hyperparameter knob (§4.4 step 6, §6.3).
4. **Broker truth, not local cache** — daily IBKR position reconcile at 17:00 ET; halt next session on mismatch (§4.4 step 7).
5. **No PDT-exposed lanes on sub-$25k accounts** — argued: the strategy's minimum equity gate is **$110k** for portfolio margin, well above the $25k PDT threshold. Most trades are held overnight (minimum 2-day hold; mean expected ~7-day hold); intraday delta hedges are equity orders on the underlying that *do* count toward PDT day-trade tally on a sub-$25k account, but the $110k minimum makes PDT structurally N/A. Even at the rare case of a same-day open-and-close on a surface trade (early reversion within hours), the per-trade frequency cap of 5/day and account equity > $110k means the FINRA day-trade rule does not bind.
6. **One strategy per slot** — explicit differentiation table in §0 vs. iter_5 (vol-level rotation) and iter_17 (event vol). Surface-shape RV is a distinct slot from level-rotation and event-vol. Cross-correlation caps in §7 enforce non-collision at the portfolio level.
7. **No same-data backtest** — 6 years in-sample (2015-2020), 5.4 years OOS (2021-2026), 14-day embargo. Walk-forward quarterly refits as sensitivity check only (§6.1).
8. **Battle-tested broker SDKs only** — IBKR via `ib_insync` exclusively; IBKR Combo (BAG) orders for multi-leg; no hand-rolled HTTP; no third-party EMS (§4.4 step 4, §8).
9. **Capacity bound named upfront** — $25M-50M AUM ceiling; 0.5-2% of node-level options ADV at $10M NAV; 3-10% at $50M (§9).
10. **Kill criteria written before deploy** — explicit §11 with 90-day falsification clock; six "kill immediately" triggers each with a metric and threshold; no vibes.

---
