# Iteration 17: Earnings ATM Straddle + Delta-Hedged Gamma Scalp (single-name US options)

## 1. Thesis (one sentence)

For a tightly filtered universe of large-cap US single names whose **prior-8-earnings realized volatility has on average exceeded the corresponding pre-announcement implied volatility by a ratio ≥ 1.10**, buying a 7-DTE ATM straddle on the trading day **before** the earnings release and **dynamically delta-hedging** the position through the announcement window — flattening by the close of the earnings day or close of the following day — extracts a positive gamma-scalping P&L because the realized path swings further than the option market is paying for, more often than not, on this conditioned subset.

Falsifiable form: **Over ≥ 200 OOS earnings events between 2023-01-01 and 2026-04-30, the per-event P&L (gross of commissions and slippage), averaged across all events that passed the universe filter at entry, must be ≥ +12 bps of underlying notional with a two-sided 95% bootstrap CI excluding 0, AND median per-event P&L ≥ 0 bps, AND maximum single-event loss ≤ 80 bps of underlying notional. If any one of those three conditions fails, the historical realized/implied edge has either decayed, was selection bias, or is dominated by gap risk that delta-hedging cannot recover — kill.**

This is **not a long-vol stat-arb on the announcement** in the naive sense; it is a *conditional* long-gamma book that only takes the trade where the historical evidence on the name says realized has been the seller's enemy. The directional component of the straddle is hedged away with the underlying so that the residual is the gamma vs theta integral over the announcement window. IV crush is therefore the **central** risk surface (see §7 and §10), not an incidental footnote.

## 2. Market / asset class / venue

- **Asset class**: US-listed single-name equity options (weekly or near-weekly chains). ATM call + ATM put at the same strike (straddle); if no ATM strike trades, use the nearer-the-money strangle (one strike OTM put + one strike OTM call, chosen so combined delta at entry is within ±0.10).
- **Underlying universe (training & deployment)**: the **top 50 US single-name underliers by 90-day options ADV** in dollar terms, intersected with **all of**:
  - Options ADV (premium notional) > **$50 M / day** measured over the 90 days ending 5 sessions before the earnings event.
  - 30-day **IV30 > 30%** at the entry timestamp (excludes ultra-low-vol names where the gamma surface is too small to scalp).
  - Underlying is a **regular C-corp** with **scheduled quarterly earnings**; ADR, REITs, BDCs, MLPs, and dual-class names with concurrent dividend ex-dates inside the holding window are excluded.
  - **Historical realized-to-implied ratio (RVI ratio) ≥ 1.10 over the prior 8 earnings releases**, where for each prior release the ratio is `realized_vol(close_T-1 → close_T+1) / implied_vol(7-DTE ATM at close_T-1)`, and the trailing mean across the 8 most recent earnings is taken.
  - **Earnings has a confirmed announcement timestamp** (Wall Street Horizon or Estimize-confirmed). No "to be confirmed" releases.
- **Example names likely to pass the filter most quarters (illustrative, not a fixed list)**: AAPL, MSFT, GOOG, META, AMZN, NVDA, TSLA, NFLX, AMD, AVGO, CRM, ORCL, ADBE, INTC, QCOM, MU, COIN, SHOP, SNOW, CRWD, PLTR, UBER, DASH, RIVN, ABNB. The filter is **recomputed every quarter** from the prior 8 earnings; names that drop below RVI ratio 1.10 fall out and are not added back until they regain the threshold for two consecutive prior earnings.
- **Forbidden universe**:
  - Anything outside the top-50-by-options-ADV cut. Bid-ask on weekly chains for thinner names blows out the slippage budget faster than the gamma compensates.
  - **M&A-rumor names** (FINRA OTC rumor flag, Bloomberg "MA" tag, unusual options activity flag from prior 5 sessions) — strangle PnL is dominated by binary corporate-action risk these names carry, which is not a vol bet.
  - Names with **ex-dividend date inside the 7-DTE window** (early-exercise risk on the short put leg... wait, we are long, but ex-div changes the option price discontinuously and confuses the delta-hedge book).
  - Names whose **most-recent earnings was an unscheduled pre-announcement, restated, or delayed**.
- **Primary exchanges**: CBOE (BZX Options), NYSE Arca Options, ISE, NASDAQ PHLX — IBKR Smart Routing handles destination selection. Underlying shares route through SMART against NYSE/NASDAQ/ARCA.
- **Broker**: Interactive Brokers (IBKR) Pro via `ib_insync`. Options and underlying single-name equities trade through the same API session, which is the single-broker requirement (lesson 8). No third-party EMS.
- **Settlement / margin**:
  - Long-only options: full premium debit, T+1 cash settlement, no margin requirement beyond posting the debit.
  - Underlying delta hedge: Reg-T long-or-short equity, T+1, requires margin if short. **The strategy will go short the underlying on delta hedges** whenever the straddle gamma builds positive delta (e.g. an upside move). Reg-T short locate must be tested at entry; if no easy locate at competitive rate (HTB > 25 bps/yr) → **abort the event**.
  - American-style exercise on single-name options — see §7 and §12.
- **Tax regime**:
  - Single-name equity options are **NOT Section 1256** (1256 covers broad-based index options + futures only). They are taxed as **short-term capital gains** on a per-leg basis. Hedge gain/loss is also short-term capital gain on the underlying. The strategy thus produces 100% short-term gain treatment — flagged but **not solved** at the strategy level; sizing assumes pre-tax bps.
  - Section 988 (foreign-currency ordinary income) does **not** apply; no FX leg.
  - Wash-sale rule applies between substantially identical instruments (e.g. closing a straddle at a loss and re-entering on a different ticker is fine; doing it on the same ticker within 30 days could disallow part of the loss). Not material at the strategy level because each event is on a different ticker per quarter, but flagged for the operator's tax preparer.

## 3. Time horizon

**Event-driven; intraday delta-hedging; total holding window 1 to 2 trading days per event.**

- Entry: at **15:30 ET on day T−1** (the trading day before the announcement), where T is the session containing the announcement (AMC reports → T is the session whose close immediately precedes the announcement; BMO reports → T is the session whose open immediately follows the announcement).
- Delta-hedge tick: every **5 minutes** during regular trading hours OR whenever **|net delta of position| crosses 25 bps of underlying notional** since the last hedge (whichever fires first).
- Exit: **close of session T** if the underlying has moved more than 1.5× the pre-announcement implied move and IV has crushed by ≥ 25%; otherwise hold one more session and **close at MOC on T+1**. Hard exit by MOC of T+1 regardless. No event held beyond T+1 close.

Decision tick (5 min) and hedge tick (5 min + delta threshold) are the **fastest of any iteration in this run that holds overnight risk** — sub-second iter 15 is meta-execution, iter 2 is 1-min ES intraday only. This is deliberate: the 5-min hedge cadence is the empirical sweet spot for single-name gamma scalping with realistic spread costs (Sinclair, *Volatility Trading*, ch. 6; practitioner BAML/Goldman vol-desk write-ups on "long gamma earnings book"). Tighter than 5 min and we eat spread on every micro-move; wider and we let the gamma slip away unhedged.

## 4. Method family and specific algorithm

- **Family**: Rules-based event-driven options strategy with deterministic Greek-driven hedging. **No ML in the entry decision**; the only quantitative model is **Black-Scholes-Merton** for live delta computation, and the universe filter is a thresholded historical statistic. ML is explicitly out of scope here (lesson 2: a binary historical filter is preferred over a confidence-ranking ML score; the realized-vs-implied filter has been shown to be a stable binary classifier in the academic literature cited below, not a noisy ranker).
- **Algorithm (step by step)**:
  1. **Universe scan (run nightly at 02:00 ET)**:
     - Pull the prior 8 earnings dates for every name in the top-50-options-ADV cut.
     - For each prior earnings date `e_i`: compute `RVR_i = realized_vol(C_{e_i - 1} → C_{e_i + 1}) / IV_7dte_atm(C_{e_i - 1})`. Realized is annualized close-to-close volatility (1 trading day = 1/252) computed on the 2-day window straddling the event; IV is the 7-DTE ATM Black-Scholes implied vol at the close of the trading day before the event, computed from the NBBO mid of ATM call + put.
     - Compute the trailing mean `RVR_mean = (1/8) Σ RVR_i`.
     - Tag the name **eligible** if `RVR_mean ≥ 1.10` AND at least 6 of the 8 prior `RVR_i ≥ 1.0` (rules out 2 huge outliers carrying the mean).
  2. **Pre-event preparation (T−1 morning, ~10:00 ET)**:
     - For each eligible name with confirmed earnings tonight or pre-open tomorrow, pull the live 7-DTE option chain.
     - Verify: bid-ask spread on ATM call AND ATM put both ≤ **2% of mid each**; option ADV (contracts) on ATM strike ≥ **1000 contracts/day**; underlying borrow rate (HTB indicator) ≤ 25 bps/yr from IBKR `reqContractDetails`.
     - If any check fails → **skip the event**. Log skip reason.
  3. **Entry (T−1, 15:30 ET)**:
     - Select strike = closest listed strike to the live underlying mid; if two strikes equidistant, prefer the lower strike (downside skew typically richer).
     - Compute target straddle size: notional position = **min(2.0% of NAV, 0.5% of options ADV)** in *underlying-equivalent notional*. Underlying equivalent = (straddle vega × IV move expected) translated into share-equivalent through the chain delta. In practice: size = floor(target_$_notional / (underlying_price × 100)) contracts per leg, with a hard cap of **20 contracts per leg per event**.
     - Submit a **combo limit order** for the straddle at NBBO mid (one ticket, both legs); if not filled in 60 seconds, step the limit one tick toward the offer side, repeat up to 3 times; if still unfilled, **cancel and abort the event** (no partial-leg fills, ever).
     - On fill, immediately compute net Greek delta (should be near zero — within ±0.05 — at entry by construction). Submit any residual delta hedge as a single equity order against IBKR SMART at marketable limit, IOC.
  4. **Delta-hedge loop (15:30 ET T−1 → close T or T+1)**:
     - Every 5 minutes during RTH: query IBKR for current position Greeks via `reqMktData(impliedVolatility=True)` or via locally re-priced Black-Scholes using current NBBO mid of each leg and the spot underlying.
     - Compute net delta in underlying-equivalent shares.
     - If `|net_delta| ≥ 25 bps of underlying notional` (= 0.0025 × position_shares_equivalent × underlying_price / underlying_price = 0.0025 × position_shares_equivalent) **or** the 5-min timer has elapsed and `|net_delta| ≥ 10 bps`: submit a **marketable-limit IOC** order in the underlying for the integer share count needed to flatten delta to zero. Underlying short sales subject to locate verification at each hedge attempt; if locate fails, attempt buy-side flatten only, do not initiate a new short.
     - Log every hedge: timestamp, pre-hedge delta, post-hedge delta, fill price, slippage vs NBBO mid.
  5. **Overnight handling (T−1 → T)**:
     - If announcement is AMC (after 16:00 ET on T−1): no overnight hedge possible; carry delta as of 16:00 ET into the open. (This is the **gap-risk window**; see §5 risk envelope and §10 failure mode #2.)
     - If announcement is BMO (before 09:30 ET on T): hold position to open, hedge at first available print (09:30:30 ET) regardless of magnitude.
  6. **Exit decision (intraday T)**:
     - At each 5-min tick on session T, evaluate: (a) `|cumulative_underlying_move|` vs **pre-announcement implied move** (= IV × √(2/252) × underlying_price); (b) current IV30 vs IV30 at entry.
     - **Exit condition A** (early): cumulative move ≥ 1.5× implied move AND IV30 has crushed by ≥ 25% from entry → close all legs at MOC of T (or immediately on marketable IOC if before 15:30 ET).
     - **Exit condition B** (default): close all legs at MOC of T+1. Hard floor.
     - Closing: combo limit order for the straddle at NBBO mid → step 3× toward bid → market order. Underlying hedge flattened at the same time via marketable IOC. Both legs of the straddle must close together; if only one leg fills, the remaining leg is closed at market immediately (zero-tolerance for naked legs across the close).
- **Why rules, not ML**: The edge thesis is structural — the *priced* IV at the 7-DTE ATM strike is a known constant at entry, and the *realized* outcome is the actual close-to-close move of the underlying; the historical conditional bias (names whose prior-8 mean RVR > 1.10) is a *binary* filter (lesson 2 — binary filter not confidence ranker). ML on top would add a failure surface without enlarging the feature set meaningfully — the option market has already digested every relevant feature into the IV print; if there is residual edge, it is the slow-moving cross-sectional bias the RVR filter is built to capture. *Anchors*: Patell & Wolfson 1979 ("Anticipated information releases reflected in call option prices") and 1981 ("The ex ante and ex post price effects of quarterly earnings announcements"); Diavatopoulos, Doran & Peterson 2008 ("The information content of implied volatility for individual stock earnings announcements"); practitioner literature (Sinclair 2013 *Volatility Trading* ch. 6; Sheldon Natenberg *Option Volatility & Pricing* ch. 13; numerous BAML/MS/GS desk pieces on "long-gamma earnings book").

## 5. Data requirements

- **Historical option chains (training & filter computation)**:
  - **Source**: Polygon.io options aggregates + chains snapshots; OptionMetrics (IvyDB) is the gold standard but $$$. Polygon **Options** developer tier ($199/mo) gives 1-min OPRA aggregates and is sufficient for ATM-strike 7-DTE backtests on top-50 names.
  - **Free tier note** (CLAUDE.md rule): Polygon free tier (5 rpm) is **insufficient** for chain-snapshot backtests across 50 names × 4 quarters × 5 years. The strategy explicitly assumes the Options Starter ($199/mo) or higher tier is available for development; for paper_v1-style cache-first runs we'd pre-fetch the relevant strikes per event into a local Parquet store (one-off cost). **No synthetic/demo chain pricing will be used at any stage of evidence generation** (CLAUDE.md hard rule).
  - **Required fields per snapshot**: `bid`, `ask`, `mid`, `underlying_close`, `atm_iv`, `delta`, `gamma`, `theta`, `vega`, `volume`, `open_interest`. If any of these are missing in the cache, augment before running evidence (CLAUDE.md rule).
- **Historical underlying (training)**: Polygon equities 1-min bars + tick-level trades for hedge-fill simulation. Free tier covers daily; intraday requires Starter tier.
- **Live chains (deployment)**: IBKR **OPRA Top of Book** subscription (~$15/mo retail) + **NASDAQ Options TotalView** if available (~$25/mo). Greeks are computed locally from NBBO mid + spot + risk-free rate; IBKR's `reqMktData` Greeks field is a fallback only.
- **Earnings calendar**: Wall Street Horizon ($) for confirmed AMC/BMO timestamps; backstop with Estimize free tier + Yahoo Finance scrape. Hardcoded blacklist of unreliable sources.
- **News / corporate-action feed**: IBKR `reqMatchingSymbols` daily check + Polygon news endpoint to flag pre-announcements, M&A rumors, restatement filings; if any flag fires within 5 sessions of entry, abort.
- **Storage**: Chain snapshots @ 1-min for top-50 × 5 years ≈ 12 GB compressed (Parquet, snappy). Earnings calendar: trivial. Underlying minute bars: ~3 GB for the universe over 5 years. **Total: ~20 GB** including a working buffer.
- **Cost**: Polygon Options Starter $199/mo, IBKR live data ~$40/mo, Wall Street Horizon ~$100/mo for retail tier (or $0 if Estimize/Yahoo are sufficient — to be re-evaluated). **Target: ~$250–340/mo**.
- **Latency requirement**: 5-min hedge cadence → tolerable round-trip seconds. The underlying hedge order send-to-ack must complete in < 2 seconds (IBKR `ib_insync` typically completes in 100–500 ms; this is not a constraint).

## 6. Validation plan

- **Total sample**: 50 names × ~4 earnings/year × 9 years (2018–2026) ≈ **1800 candidate events** before filter, **~600–800 events** after the RVR ≥ 1.10 filter (filter pass-rate empirically ~40%).
- **Train / calibration window**: **2018-01-01 → 2022-12-31** (5 years, ~400 filtered events). Used to: confirm the 1.10 RVR threshold versus alternatives in {1.00, 1.05, 1.10, 1.15, 1.20}; confirm the 7-DTE choice vs alternatives in {3-DTE, 7-DTE, 14-DTE}; confirm the 25 bps hedge threshold vs alternatives in {15, 25, 40, 60 bps}; confirm the T+1 exit vs T-only exit; confirm 5-min hedge tick vs {1-min, 15-min}.
- **Out-of-sample evidence window**: **2023-01-01 → 2026-04-30** (3.3 years, target **≥ 200 filtered events**). Names that drop in/out of the top-50-by-options-ADV are re-evaluated quarterly; the universe is **not** fixed across the OOS window — this is critical to control selection bias (see §10 failure mode #5).
- **Walk-forward design**: Train 5y → step 1 quarter → test 1 quarter, rolling forward across 2023-Q1 through 2026-Q2. ~14 forward folds, each producing 10–20 filtered events.
- **Purge & embargo**: Events are *not* serially correlated in the usual time-series sense (each event is per-name, quarterly). However, **same-name** events are correlated through the slow-moving RVR filter; we purge by ensuring no name appears in both train and test of the same fold (already handled by date-based fold cuts since earnings are quarterly).
- **Minimum OOS events for verdict**: **200 filtered events**. (Note: the 90-day live falsification window is a *separate*, smaller test — see §11.)
- **Statistical tests & thresholds**:
  - Per-event P&L denominated in bps of underlying notional at entry.
  - **Bootstrap test**: 10,000 draws (event-level resampling with replacement), require **mean P&L 95% CI excludes 0** AND **mean ≥ +12 bps**.
  - **Median test**: require **median per-event P&L ≥ 0 bps** (guards against positive mean driven by 1-2 lucky tail events).
  - **Tail-loss test**: require **maximum single-event loss ≤ 80 bps of underlying notional**. (A name that gaps 30% on guidance shock with the straddle hedge mid-flight can lose more — if any such event occurs in OOS, the strategy fails this test, full stop.)
  - **Sharpe-equivalent**: per-event Sharpe (mean / std) ≥ 0.35. Lower bar than continuous-Sharpe iterations because events are sparse — annualized Sharpe at ~50 events/yr × per-event Sharpe 0.35 ≈ annual Sharpe 2.5 under √N scaling, which is plausibly optimistic; we'll report both per-event and annualized, and the binding gate is per-event Sharpe.
  - **Sub-sample stability**: split OOS in half (2023-2024 vs 2025-2026), require both halves' mean P&L > 0 (not necessarily ≥ 12 bps each, but both positive). Guards against one-regime fluke.

## 7. Risk envelope

This is the section that does the load-bearing work for an event-driven options strategy, particularly given IV-crush risk.

- **Max position size per event**: **2.0% of NAV in underlying-equivalent notional** (= straddle delta-equivalent at entry). Equivalent debit premium ≈ 0.10–0.25% of NAV depending on IV level (e.g. 30% IV → premium ≈ 0.6% of underlying notional → 0.012% of NAV; 60% IV → ≈ 0.024% of NAV). Hard cap **3% NAV underlying-equivalent**.
- **Max concurrent events**: **3 events** open at any time. Earnings clusters (e.g. mega-cap tech week) routinely have 5+ eligible names same-week; we cap at 3 by RVR-rank (highest historical bias first) and reject the rest. Rationale: concurrent events share macro vega exposure; capping caps total vega while preserving the per-event edge.
- **Daily loss cap**: **−2.0% NAV** across the strategy. If realized MTM intraday hits −2% NAV, close all open positions at marketable IOC and stand down for the session.
- **Per-event loss cap**: **−60 bps of underlying notional** on the realized straddle + hedge book. If mid-event MTM crosses −60 bps (e.g. an after-hours gap moved against the carried delta), execute an early close at next available print regardless of where we are in the hedge schedule.
- **Drawdown trigger**: trailing 30-day strategy NAV drawdown ≥ **6%** → strategy halted, human review required to restart.
- **Circuit breakers (the IV-crush lesson, central to this strategy)**:
  - **IV-crush dominates the trade** if realized vol comes in below implied: the straddle premium crushes by typically 20–50% AMC, and the gamma scalp must have already extracted enough P&L to offset that decay. **The strategy ONLY profits if the realized path's gamma-scalp P&L exceeds the theta + IV-crush expense.**
  - **Therefore the RVR filter is non-negotiable**: if for any reason the filter cannot be computed for a name at entry (e.g. one of the prior 8 earnings is in a corporate action or data gap), the event is **automatically skipped**, not traded on weakened evidence.
  - **No-hedge protection**: if at any point the delta-hedge venue cannot fill (locate failure on a short hedge, market-wide halt, IBKR rejection), close the entire straddle at next print rather than carry naked-delta exposure. This is the IV-crush + directional-risk compounding scenario — a delta-unhedged straddle held past earnings IS the canonical "buy a straddle and lose to IV crush" trade we are explicitly avoiding.
  - **Pre-event IV gate**: if at the 15:30 ET entry tick the 7-DTE ATM IV is more than **1.5× its trailing-30-day median** for that name, the event is skipped — the market is *already* pricing a fat-tail outlier and the straddle is unlikely to underpay. (Practical example: a name where leaked numbers have run IV from 60% to 110% — abort.)
- **Gap-risk handling (AMC reports)**: any AMC report is held with the delta at the prior close — there is no overnight US equity hedging path for retail-IBKR. Mitigation: at entry, **reduce position size by 30%** for AMC reports vs BMO; AMC events thus carry 1.4% NAV vs 2.0% for BMO.
- **Correlation rule**: disallow co-deployment with any other long-vol, dispersion, or earnings-betting strategy whose 30-day return correlation with this strategy exceeds 0.5. Mostly the concern is dispersion / index-vol books that systematically buy vol heading into mega-cap earnings — those are the same trade with a different wrapper.
- **Consecutive-loss circuit breaker**: 4 consecutive negative events → reduce per-event size to 1.0% NAV for the next 8 events; restore after a 3-event positive streak.

## 8. Execution plan

- **Broker / SDK**: IBKR Pro account, `ib_insync` (lesson 8). Both option legs and the underlying hedge route through one API session and one cash account, eliminating cross-broker reconciliation.
- **Order types**:
  - **Entry (straddle)**: `ComboLegs` order at NBBO mid limit, IOC; step toward offer in 1-cent increments up to 3 times in 60 seconds; cancel and abort if unfilled. Never split into separate-leg orders at entry — single-leg fills of an intended straddle leave us with unwanted direction.
  - **Exit (straddle)**: same combo logic at NBBO mid, step toward bid; if not filled after 3 steps inside the 5-min closing window, **market order on both legs simultaneously**.
  - **Underlying hedge**: marketable-limit IOC at NBBO ± 1 cent (1 cent past the touch); fallback to MKT after 5 seconds; reconcile against `reqPositions()` after every fill.
- **Slippage budget**:
  - **Straddle entry/exit**: 4 bps of underlying notional per round-trip on each leg (combined ~8 bps round trip on the option side, expressed as fraction of underlying notional via the chain's delta-vega translation).
  - **Underlying hedge**: 2 bps per hedge transaction (large-cap names trade at sub-1-bp spread; 2 bps covers fill latency + queue position).
  - Hedge transactions per event: ~10 (5-min ticks across 1–2 RTH sessions, with delta-threshold compression on quiet stretches). Total slippage budget per event: **~28 bps**. The 12 bps minimum mean P&L is *net of* this slippage budget.
- **Order rejection handling**: IBKR rejection (margin, halt, IB-side risk) → log + retry once with the same order params + alternate limit. Second rejection on any leg → close all related legs at MKT immediately and page operator. **No silent retry loop.**
- **Broker truth reconciliation (lesson 4)**: every 30 seconds while a position is open, query `reqPositions()` and `reqOpenOrders()`; reconcile against local state. Any divergence ≥ 1 contract or ≥ 100 shares → halt all new orders, page operator. Closing orders ALWAYS use the broker's reported quantity, never the local cache.
- **Halt handling**: if the underlying or its primary option class halts during the hedge window, suspend hedging and log the halt. On reopen, recompute Greeks from the first valid post-reopen NBBO and resume.

## 9. Capacity

- **ADV ceiling per event (option side)**: target 0.5% of the option strike's daily contract ADV. ATM 7-DTE on a name like AAPL/NVDA on earnings day typically trades 10,000–40,000 contracts at the ATM strike → 0.5% = 50–200 contracts feasible per event without moving the chain. Per-leg cap of 20 contracts (§4 step 3) is well inside that.
- **AUM ceiling for the strategy**:
  - At 2% NAV per event, 20-contract cap, and assuming average underlying price $200: 20 contracts × 100 shares × $200 = $400,000 per leg = $800,000 of underlying-equivalent notional per straddle event.
  - $800,000 = 2% NAV → strategy fully scales to **~$40 M AUM** without breaching the contract cap.
  - Above $40 M, breaching the 20-contract-per-leg cap pushes us into 1%+ ADV territory on the ATM strike of mid-tier names, where the spread starts to widen during the entry/exit windows.
  - **Hard ceiling: $75 M AUM**, with the contract cap raised to 40 contracts and the per-event NAV % raised to 2.5%. Beyond $75 M the strategy is in materially worse fill territory on the smaller names of the top-50 cut.
- **Single-name concentration**: max 2% NAV underlying-equivalent in one straddle, 3 concurrent events → max **6% NAV in concurrent earnings exposure**. By design.
- **Single-sector concentration**: max 2 events concurrent in the same GICS sector (e.g. cannot hold AAPL + MSFT + GOOG simultaneously even if all 3 are eligible; sector cap of 2 reduces to a max 4% NAV per sector).

## 10. Failure modes

At least 3 named failure modes and what each looks like in metrics.

1. **IV-rich regime ends (low-realized-vol macro period dominates)**. *What it looks like*: across a quarter, mean per-event P&L drops below 0 bps; median per-event P&L drops below 0; the strategy bleeds 5–15 bps per event consistently. *Diagnostic*: 90-day rolling mean of forward-realized / pre-event implied across the **whole top-50 universe** falls below 1.0. *Mechanism*: macro vol regime change (e.g. post-2017 calm), index VIX persistently low, single-name realized doesn't escape the priced implied. *Mitigation*: pause strategy when 90-day universe-wide RVR mean < 0.95.
2. **Universe selection bias (the RVR filter was overfit on the in-sample window)**. *What it looks like*: filter still admits names but their forward realized comes in *at or below* implied in the OOS window; the historical 1.10 ratio doesn't predict forward 1.10. *Diagnostic*: regress forward `RVR_i` on trailing-8 mean `RVR_mean`; if OOS R² < 0.05 and the slope coefficient is not statistically distinguishable from 0, the filter has lost its predictive power. *Mechanism*: the underlying mechanism is name-specific (CEO communication style, sector volatility, guidance practice) and *itself* mean-reverts — the names that historically out-realized stopped doing so once they were systematically targeted by the options market. *Mitigation*: monitor filter predictive power quarterly; if R² < 0.05 for two consecutive quarters, recompute filter on a fresh feature set (e.g. macro-conditioned RVR rather than raw 8-event mean) before continuing.
3. **Gap-risk realization (M&A or guidance shock)**. *What it looks like*: a single event delivers a −80 to −150 bps loss because the underlying gapped 20–40% pre-open and we were carrying delta on an AMC report. *Diagnostic*: any single-event loss > 80 bps; check if the move was a guidance / M&A shock vs a clean vol miss. *Mechanism*: AMC announcement reveals unscheduled news (M&A, restatement, surprise CFO change) that prices the underlying through the straddle's profitable region; the long straddle should profit on the move but the delta carried into the open compounds with the lack of intraday hedging; if the gap is also accompanied by IV crush (rare but possible on a clean-news gap), the straddle premium itself loses. *Mitigation*: AMC sizing reduction (already in §7); event skip when M&A rumor / unusual options activity is flagged in prior 5 sessions.
4. **Liquidity flight on the ATM strike**. *What it looks like*: option ADV on the target strike drops 50%+ in the 5 sessions before the event; bid-ask widens to 5%+ of mid; entry combo limit goes unfilled in the 60-second window. *Diagnostic*: aborted-event rate ≥ 30% in a quarter; named on the abort log. *Mechanism*: market-maker pull on a stressed name, or low-VIX-regime liquidity withdrawal on weekly chains. *Mitigation*: liquidity check at entry already aborts; if abort rate trends above 30%, raise the option ADV gate from $50 M → $75 M.
5. **Exchange / broker outage on hedge day**. *What it looks like*: IBKR connection drop during the hedge loop; underlying or option halt; CBOE outage. *Diagnostic*: gap in hedge-log timestamps > 10 minutes; positions reconcile but with stale delta. *Mechanism*: known operational risk for any single-broker setup. *Mitigation*: secondary control channel (separate IBKR session on a separate VPS); auto-flatten everything on > 15-min outage.

## 11. Falsifiable verdict criteria

- **Promote to live capital**: after the **90-day shadow run**, with ≥ 30 filtered events traded paper-side:
  - Mean per-event P&L ≥ +10 bps (slightly below the 12 bps OOS bar to allow for paper-vs-live spread-noise margin).
  - Median per-event P&L ≥ 0 bps.
  - Max single-event loss ≤ 80 bps.
  - No broker-truth divergence > 1 contract / 100 shares in the shadow window.
  - No daily loss cap breach.
  - Slippage on entry combos within budget (mean ≤ 8 bps; max ≤ 15 bps).
  - **AND** the full OOS evidence (200+ events, 2023-04-30) has cleared §6 stats.
- **Continue shadow / extend**: metrics ambiguous (mean per-event ∈ [+5, +10] bps OR fewer than 30 filtered events in 90 days) but no kill triggers fired → extend shadow another 60 days.
- **Kill immediately**:
  - Any one of: 3-day strategy DD > 4%; daily loss cap (−2% NAV) breached; broker-side fill discrepancy > 0.5% of any position; any single-event loss > 100 bps.
  - Or any 30-event rolling subset of the shadow window shows mean P&L < −5 bps.
  - Or universe-wide 90-day RVR mean falls below 1.00 (the filter has lost its underlying economic basis).

### Falsification within 90 days of going live — the 30-event argument

The earnings calendar produces **roughly 35–45 eligible earnings events per 90-day window** when scoped to the top-50-by-options-ADV universe (each name reports once per quarter; ~50 names × 1.0 events / 13 weeks × 13 weeks = 50 candidate events, of which ~70% pass entry-time liquidity & IV gates → ~35 traded events). This is *just enough* to falsify a strong null:

- Under H0 (zero edge, true mean per-event P&L = 0, per-event std ~30 bps from the in-sample distribution), the standard error of the 30-event sample mean is **30 / √30 ≈ 5.5 bps**. A 90-day shadow showing mean per-event P&L > +10 bps is approximately **1.8 σ above H0**, p ≈ 0.04 (one-sided), borderline-significant — not promote-grade in 90 days, hence the requirement of 200+ events in the full OOS *before* the live shadow even starts.
- Under H1 (true mean = +12 bps as predicted by historical RVR > 1.10), the same 30-event window has standard error 5.5 bps → 95% CI of the sample mean spans roughly [+1, +23] bps — wide but **excludes 0 at the lower bound only in best-case realizations**. So 30 events alone *cannot promote* the strategy, but **30 events CAN falsify** decisively:
  - If mean < −5 bps over 30 events, that's >2.5 σ below H1's mean of +12 — strong evidence that either the historical edge has decayed or selection bias was at work, **kill**.
  - If max single-event loss > 100 bps, that violates the tail-risk gate independent of the mean, **kill**.
  - If broker-truth divergence occurs even once, that's an operational disqualifier, **kill**.
- So in 90 days we **cannot** confirm the edge, but we **can** falsify it. The verdict structure (promote ↔ continue shadow ↔ kill) is built around this: 90 days is a kill window, not a promote window. Full promotion requires 200+ events in the formal OOS evidence run completed before the shadow starts.

## 12. One-paragraph operational risk note

The single largest operational surface is the **delta-hedge dependency**: the strategy's gamma scalp P&L is mathematically positive only if delta is continuously hedged within the cadence we promised; an IBKR API outage, a borrow-locate failure on the underlying short hedge, a single-name halt, or a CBOE-side options halt during the hedge window all break the precondition and convert the straddle from a long-gamma book into an unhedged long-vol bet that loses to IV crush. **American-style early-exercise risk** on single-name options is the secondary surface — we are long both call and put, so we won't be assigned, but if the long leg goes deep in-the-money and we elect to close, exercise vs sell is an operator-discretion call (selling preserves time value but may have wider spreads; exercising forfeits time value but executes at intrinsic — for liquid weeklies, always sell). **Corporate actions** (special dividend, split, spinoff) inside the 7-DTE window invalidate the chain's strike grid and the Black-Scholes Greeks; these names are excluded at universe scan but a *surprise* announcement during the 1-2 day holding window will manifest as suddenly-stale Greeks — auto-flatten on any OCC adjustment memo, page operator. **Tax wrinkles** (IRS §1256 vs §988): single-name equity options are short-term gain treatment, not §1256 60/40, and not §988; we flag this for the operator's tax preparer but do not solve it at strategy level — the strategy's reported bps are pre-tax. **Data dependency**: the historical RVR filter is the entire economic basis of the strategy and depends on Polygon Options Starter ($199/mo); if Polygon downgrades, deprecates, or rate-limits the chain endpoints, the filter cannot be recomputed quarterly and the strategy must be paused, not run on stale eligibility lists. **Single-broker concentration**: IBKR Pro is the only execution path; an IBKR outage of more than 60 minutes during the hedge window of any open event triggers the auto-flatten protocol described in §7. Finally, **paper_v1 freeze rule** — per project memory, the 90-day cycle freeze blocks model/threshold/feature/risk-gate changes mid-run; any "I noticed the filter is failing" reaction during a live shadow must be deferred to the next cycle, not patched in-flight (this iteration honors that rule by writing the filter, sizing, and exit logic as code-frozen at deployment, with all human override paths going through documented circuit breakers, not parameter tweaks).

---

### Lessons compliance check

1. **Validate before deploy**: 5y in-sample 2018-2022 + 3.3y OOS 2023-2026 walk-forward + 90-day shadow (§6, §11).
2. **Binary gates, not confidence rankers**: RVR ≥ 1.10 is a binary filter, no ML confidence score sizes positions (§4).
3. **TP:SL asymmetry is structural**: per-event loss cap −60 bps vs target +12 bps mean (with positive tail unbounded by gamma); exit rule (T or T+1) is part of the edge, not a hyperparameter (§3, §7).
4. **Broker truth, not local cache**: `reqPositions()` reconcile every 30 s; closing orders always use broker quantity (§8).
5. **No PDT-exposed lanes on sub-$25k**: options + Reg-T short hedges require ≥ $25k cash account; strategy assumes ≥ $1M NAV at deployment scale (capacity §9).
6. **One strategy per slot**: this iteration is an event-driven long-gamma earnings book; correlation rule disallows co-deployment with dispersion/index-vol/other-earnings strategies (§7).
7. **No same-data backtest**: train/OOS cleanly separated by date, walk-forward with quarterly step (§6).
8. **Battle-tested broker SDKs only**: `ib_insync` for both legs (§8).
9. **Capacity bound named upfront**: $40 M soft / $75 M hard (§9).
10. **Kill criteria written before deploy**: explicit kill thresholds in §11.

