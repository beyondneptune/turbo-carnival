# iter_14_lightgbm_cross_section_russell1000

## 1. Thesis (one sentence)

A LightGBM gradient-boosted ranker, trained on a ~70-feature set spanning classical Fama-French factor primitives (value, momentum, quality, low-vol, size) and alt-data overlays (Google Trends z-score, GDELT/RavenPack news sentiment, 30-day option-skew, short-interest %), produces a monthly cross-sectional rank of Russell-1000 constituents that — long the top decile, short the bottom decile, equal-weighted, monthly-rebalanced — delivers a falsifiable claim: across 60+ out-of-sample monthly rebalances (2016-2026, walk-forward retrained every 12 months on an expanding window beginning 2000), the realized rank IC (Spearman correlation of model output rank vs. realized next-month return rank) is positive in mean with t-stat > 2.0 AND the decile-spread portfolio earns net-of-cost annualized Sharpe > 0.75, OR the strategy is killed.

## 2. Market / asset class / venue

- **Asset class (primary)**: US equity single names, large-cap and mid-cap — the **Russell 1000** constituent set (top ~1000 US stocks by float-adjusted market cap, FTSE Russell methodology, reconstituted annually in late June with a quiet-period rebalance in Q1 since 2024). Universe membership is **time-aware** (see §5 on point-in-time reconstitution).
- **Asset class (long-only fallback for cash accounts)**: Russell 1000 top-decile longs only (~100 names) plus an IWB-ETF short index hedge to neutralize beta where the user wants market-neutrality without single-name borrow. This variant is for cash-account users who cannot short single names.
- **Primary venue**: NYSE, Nasdaq, Cboe BZX, NYSE Arca — names trade on their listing venue with SIP NBBO consolidation. Strategy is venue-agnostic; broker routes via NMS.
- **Primary broker / SDK**: IBKR Pro via `ib_insync` (per Lesson 8 — battle-tested). Margin account required for the full long-short variant; cash account is fine for the long-only fallback.
- **Settlement**: T+1 (DTC US equities, effective May 2024). Dividends accrue per record date. Short-sale locate via IBKR's Stock Yield Enhancement / Securities Lending Program — most Russell 1000 names are general-collateral easy-to-borrow with negligible rebate cost (<25 bps), but the bottom-decile (low-momentum/distressed) basket is the part most likely to contain hard-to-borrow names; explicit borrow-fee budgeting in §5/§8.
- **Margin**: Reg-T 50% initial, 25% maintenance on the long side. Reg-T 150% on the short side (short proceeds + 50%). Portfolio Margin reduces total margin to ~15% gross at $125k+ equity but requires net liquidation ≥$125k and an active PM agreement — the spec **does not assume PM** in the base case.
- **Tax regime**: Section 1256 does NOT apply (these are individual equities, not 1256 contracts). Holdings of ≥1 month are short-term gains (ordinary income for US holders) — there is no LTCG optimization in the base design and we make peace with that. Wash-sale rule applies: a name that crosses from long-side to short-side basket inside 30 days could trigger wash treatment on a closed long loss; the system records this risk but does not optimize for it (the tax cost is small relative to the gross spread we are trying to capture).

## 3. Time horizon

**Daily bars, monthly rebalance, ~21 trading day holding window per cohort.** Signal computed at NYSE close on the last trading day of each calendar month (or T-1 if last day is shortened). Orders execute over the first 30 minutes of the next trading day (09:30-10:00 ET) using VWAP-targeted child slices to minimize impact. Held until the next month-end re-ranking. Multi-day swing — explicitly **not** intraday; no PDT exposure (Lesson 5 honored — a position held ~21 trading days is unambiguously not a day trade).

## 4. Method family and specific algorithm

**LightGBM gradient-boosted decision-tree ranker** (`LGBMRanker` with `lambdarank` objective, or `LGBMRegressor` with monotonic target = forward-21-day excess return — the spec uses the regressor formulation because next-month rank is naturally produced from a continuous prediction and because lambdarank's pairwise loss requires arbitrary group sizes the spec can engineer per month). Anchored to **Gu, Kelly & Xiu 2020 ("Empirical Asset Pricing via Machine Learning", *Review of Financial Studies*)** — the canonical demonstration that tree-based and shallow neural ensembles outperform linear factor models for cross-sectional next-month-return prediction by ~2x in OOS R² — and **Asness, Frazzini & Pedersen 2019 ("Quality Minus Junk", *Review of Accounting Studies*)** for the quality-factor construction.

Algorithm:

1. **Universe at month-end T**: point-in-time Russell 1000 constituents as of T (using historical FTSE Russell reconstitution snapshots — see §5 on survivorship).
2. **Feature matrix**: ~70 features per name per month-end. Full list in §5; categories:
   - **Momentum block (7)**: 12-1 month momentum (Jegadeesh-Titman skip-1), 6-1 month, 3-1 month, 1-month reversal, 36-month long-term reversal (De Bondt-Thaler), 52-week-high distance, intermediate-horizon (7-12 month).
   - **Value block (8)**: log book-to-market (Fama-French), log earnings-to-price, log sales-to-price, log cashflow-to-price, log EBITDA-to-EV, dividend yield, free-cashflow yield, log buybacks-to-marketcap.
   - **Quality block (10)**: gross profitability (Novy-Marx 2013, GP/A), return-on-equity, return-on-assets, asset-growth (Cooper-Gulen-Schill, negative), accruals (Sloan 1996, negative), net stock issuance (Pontiff-Woodgate, negative), gross-margin stability (5y std), interest-coverage, debt-to-equity (negative), Piotroski F-score.
   - **Low-vol / risk block (8)**: trailing 60-day idiosyncratic vol (residual to FF3), 60-day beta to market, 60-day beta to size (SMB), 60-day beta to value (HML), 252-day total vol, downside semi-deviation, 1-month max-daily-return (Bali-Cakici-Whitelaw MAX), max-drawdown last 252 days.
   - **Liquidity block (5)**: log dollar-volume 60-day avg, Amihud illiquidity, bid-ask spread 60-day avg, turnover ratio, days-since-IPO (capped at 1000).
   - **Microstructure block (5)**: 21-day return skew, 21-day return kurtosis, sign-of-return persistence (5-day autocorr), gap-overnight cumulative 60-day, post-earnings drift residual.
   - **Industry / size block (4)**: log market cap, GICS sector one-hot (collapsed to 11 sectors as separate features), industry-relative momentum residual.
   - **Alt-data block (8)** — the heart of the iteration:
     - **Google Trends z-score (5y)**: monthly search-volume z-score for the company name + ticker, lagged 1 day (Da-Engelberg-Gao 2011).
     - **News sentiment (GDELT / RavenPack)**: 30-day rolling mean sentiment polarity and 30-day sentiment momentum (delta of mean vs. prior 30d).
     - **News volume**: log of total news articles last 30 days, z-scored against own 1y history (proxies attention shocks).
     - **Option-skew (CBOE / OptionMetrics)**: 30-day at-the-money implied-vol minus 30-day 25-delta put IV — proxies tail-risk pricing (Xing-Zhang-Zhao 2010).
     - **Short interest (FINRA bi-monthly)**: short-interest as % of float, days-to-cover (short-interest / 10-day-avg-volume).
     - **13F crowding (quarterly)**: change in number of institutional holders, change in institutional ownership %.
     - **Social-media buzz (StockTwits / Reddit WSB mentions, optional)**: 30-day mention z-score — explicitly marked as the most-fragile alt feature (see §10 alpha-decay).
3. **Target**: forward 21-trading-day excess return vs. equal-weighted universe mean (`y_i = r_i[T, T+21] - mean(r_j[T, T+21])`). Cross-sectional demeaning isolates the relative-rank signal and removes month-level beta exposure from the label.
4. **Training**: monthly cross-sections stacked across an **expanding window** beginning 2000-01. Each retraining uses all months from 2000-01 through (T - 1 month - 12-month embargo if retraining now); embargo prevents the most recent year from leaking into the model that will trade the next year. Training months: ~180+ at first retrain. LightGBM hyperparameters: `num_leaves=63`, `learning_rate=0.02`, `min_data_in_leaf=200`, `feature_fraction=0.7`, `bagging_fraction=0.7`, `lambda_l2=1.0`, `n_estimators=600` with early-stopping on a held-out final 24 months of training data. Hyperparameters are **fixed before any OOS evaluation** (tuned on a pre-2000 hold or by Gu-Kelly-Xiu defaults — see Lesson 7 below).
5. **Retraining cadence**: every **12 months**, on the first trading day of each January, retrain the entire model on all data from 2000-01 through end-of-prior-year minus a 12-month embargo. The frozen model trades for the next 12 monthly rebalances.
6. **Inference at month-end T**: produce continuous predicted excess-return `ŷ_i` for every name in the active Russell 1000 universe.
7. **Rank** all `ŷ_i` descending.
8. **Long basket (top decile)**: top ~100 names by `ŷ_i`. Equal-weighted, so each name = 1.0% of long gross notional.
9. **Short basket (bottom decile)**: bottom ~100 names by `ŷ_i`. Equal-weighted, each name = 1.0% of short gross notional. Hard-rejected if a name appears on IBKR's no-locate list — substitute with the (101st, 102nd, …) name in rank order.
10. **Dollar-neutral construction**: long gross = short gross = 1.0x equity by default; net market exposure ≈ 0 (basket betas typically 0.95-1.05 on each side post-quality-tilt). Optional beta-neutral overlay: residual beta hedged with E-mini S&P 500 (ES) futures sized to the long-minus-short basket beta forecast.
11. **Hold until month-end T+1**; re-rank; trade the diff (turnover is typically 30-45% of names per month, so transaction cost matters — modeled in §6).

**Why LightGBM and not linear**: Gu-Kelly-Xiu 2020 demonstrate that the cross-sectional prediction surface has strong **nonlinear interactions** (e.g., value works well *only* when quality is also positive — i.e., cheap-but-not-junk; momentum works *better* in low-vol names; alt-data features have non-monotone interactions with size). A linear factor model captures the main effects but misses these interactions and the empirical OOS R² gap is ~50-100% in their study. Trees handle this natively, are robust to outliers (rank-based splits), and the LightGBM implementation is fast enough to retrain monthly in <10 minutes on a single laptop CPU. Per Lesson 2: the model's continuous score IS the ranking signal — this is the central feature of cross-sectional strategies — and §6 validates the score's rank information (Spearman IC) BEFORE relying on the top/bottom-decile portfolio result, which is the spec-correct ordering of evidence.

## 5. Data requirements

- **CRSP daily stock file**: Russell 1000 constituents 2000-2026, adjusted OHLCV, share-class data, distributions. ~3 GB total. Cost: $20k/yr academic license (WRDS access) — or, **for retail-feasible reproduction**, Polygon S3 daily bars ($79/mo Stocks Starter tier) + Sharadar SF1 fundamentals via Quandl ($300/mo equities bundle). Note that Polygon daily is delisting-aware back to 2003.
- **Sharadar / Compustat fundamentals (point-in-time)**: quarterly fundamentals back to 1999, **with restatement vintage** so that the data used at month-end T is the data that was actually publicly available on day T — no future restatements bleeding into features (see §10 leakage). Quarterly file ~500 MB. Cost: $300/mo Sharadar SF1 PIT bundle, OR Compustat Snapshot via WRDS.
- **FTSE Russell historical reconstitution files**: annual Russell 1000 constituent lists for every reconstitution date 2000-2026 (June reconstitution + Q1 quiet-period rebalance from 2024). Source: FTSE Russell research data ($5k/yr) or scraped from historical Russell index PDFs (Wayback Machine + manual reconstruction for older dates). **This is the survivorship-bias fix — without it, the strategy backtest is invalid.** ~10 MB total once compiled.
- **Earnings-announcement calendar**: Zacks Earnings Calendar or IEX Cloud earnings — exact announcement date and time per Russell-1000 name 2000-2026. Used for embargo, not as a feature. ~50 MB. Cost: $50/mo IEX, free academic.
- **Google Trends**: weekly search-volume index per company name + ticker, 2004-2026 (Trends data begins Jan 2004 — strategy uses zero-fill or "feature missing" indicator for 2000-2003 if needed). Scraped via `pytrends` (free, fragile to rate limits) or licensed via SerpAPI/Brightdata for stability ($50-100/mo).
- **News sentiment**: GDELT 2.0 GKG global knowledge graph (free, S3-hosted, ~5 GB/year) for sentiment polarity scored against company names, OR RavenPack News Analytics (Russell-coverage 2000-2026, $2-5k/mo institutional). Spec uses GDELT in the retail track and RavenPack in the institutional track; both produce a comparable sentiment_30d feature.
- **Option-skew**: OptionMetrics IvyDB US ($2k/yr academic, $20k/yr commercial) for daily ATM-IV and 25-delta put-IV per name. **Falls back gracefully**: if OptionMetrics unavailable, use IBKR's own option chain snapshot via `ib_insync.reqMktData()` on optionable Russell-1000 names — IBKR provides real-time IV. For backfill 2000-2026, OptionMetrics is the only practical source.
- **Short interest**: FINRA bi-monthly short-interest reports (free, public) joined with daily volume from CRSP/Polygon for days-to-cover calculation. ~10 MB.
- **13F institutional ownership**: SEC EDGAR 13F filings (free, quarterly). Parsed via `sec-api` or homebrewed scraper. ~2 GB raw, ~50 MB processed.
- **Social-media (optional)**: StockTwits firehose archive ($500/mo) or Reddit WSB scrape (free via PushShift). Spec marks this as the most-fragile feature category.
- **Storage estimate**: ~50 GB total (dominated by GDELT and tick-level fundamentals snapshot history); ~5 GB/year incremental.
- **Cost estimate** (retail track): ~$80/mo Polygon + $300/mo Sharadar SF1 + $50/mo IEX earnings + ~$100/mo Trends/SerpAPI = **~$530/mo all-in**, plus free GDELT and FINRA. Skip OptionMetrics in retail track (use IBKR live IV for forward; skip option-skew feature for backtest pre-2024 if no OptionMetrics — degrades model OOS R² by ~5-8% per Gu-Kelly-Xiu sensitivity analysis).
- **Cost estimate** (institutional track): ~$3-5k/mo including OptionMetrics + RavenPack + WRDS.
- **Latency requirement**: 5-second order ack tolerable. Monthly rebalance, no latency sensitivity. IBKR REST/`ib_insync` round-trip ~100-200 ms from US — irrelevant at this horizon.

**Survivorship-bias prevention (CRITICAL, per spec lesson 7)**: the universe at every historical month-end T is the *actual* Russell 1000 constituent set as of T, reconstructed from FTSE Russell historical files. Stocks delisted, acquired, or fallen out of the index are KEPT in the universe through the month they exited (so realized returns properly reflect deletion mechanics), and their post-deletion returns are computed at the CRSP delisting return (which is often -100% for bankruptcies and explicitly NOT NaN). Backfilling from current Russell 1000 list (the naive mistake) would inflate the OOS Sharpe by 1-3 points by removing every name that subsequently failed — the Lo-MacKinlay-Wang 1990s lesson, still relevant.

**Point-in-time fundamentals (CRITICAL)**: a name's book-value at month-end T is the book-value that was reported and publicly available on day T — typically the most-recent 10-Q filed at least 45 days before T (the standard SEC filing deadline for 10-Q is 40-45 days post-quarter-end). Sharadar SF1 PIT and Compustat Snapshot both expose the `as_of_date` and `period_end_date` columns; the strategy uses the most-recent fundamental row where `as_of_date <= T` AND `period_end_date >= T - 6 months` (filters out stale rows from companies that haven't filed in 6+ months — usually de-listings).

**Earnings embargo (CRITICAL)**: any name with a scheduled earnings announcement in the **5 trading days BEFORE** the rebalance date T or in the **first 5 trading days AFTER** T (i.e., during the early holding period) is **excluded from both long and short baskets that month**. This is the same embargo logic Gu-Kelly-Xiu use to avoid the earnings-announcement-day return spike contaminating monthly cross-sections. Spec also embargoes for the **3 trading days surrounding** any 8-K material event filing per SEC EDGAR feed (mergers, restatements) — these would otherwise leak forward information into the alt-data sentiment features.

## 6. Validation plan

- **Sample**: Russell 1000 constituents monthly, 2000-01-31 to 2026-04-30 (~316 month-ends). Pre-2000 data omitted because alt-data coverage is too thin (Google Trends starts 2004; GDELT starts 2013; OptionMetrics starts 1996 but coverage of mid-caps is patchy pre-2000).
- **Walk-forward design**:
  - **In-sample training (no OOS evaluation here, used only for hyperparameter sanity)**: 2000-01 to 2015-12 (192 months). LightGBM hyperparameters set per Gu-Kelly-Xiu defaults with light tuning on a 2012-2015 held-out tail; hyperparameters are then **frozen** before any 2016-2026 evaluation.
  - **Out-of-sample (rolling, the verdict window)**: 2016-01 to 2026-04 (124 month-ends; ~120 actionable monthly rebalances after retraining cadence). Spec requires **minimum 60 OOS monthly rebalances** before a verdict — comfortably satisfied; the full 2016-2026 window is the headline result.
  - **Retraining cadence**: every 12 months. Model in force for OOS year Y is trained on data 2000-01 through (Y - 1) - 12-month embargo. So model trading 2017 is trained on 2000-01 to 2015-12; model trading 2018 is trained on 2000-01 to 2016-12; etc. Expanding window — never shrinks.
- **Purge & embargo**:
  - **Inter-fold purge**: 12-month embargo between training tail and OOS start prevents label leakage (a name's forward-21-day return computed at month-end M leaks 21 days into month M+1; without an embargo the very next month's prediction would have seen its own label in training).
  - **Intra-month embargo**: 5-trading-day earnings embargo (above).
  - **Wash-sale awareness**: not validation-relevant but logged.
- **Cost model** (modeled in backtest, NOT optimistically assumed):
  - **Commissions**: IBKR Pro tiered pricing, ~$0.0035/share for typical mid-cap (e.g., $50/share name, 200 shares = $0.70 = 0.7 bps). Round-trip ~1.4 bps.
  - **Spread**: half-spread on liquid large-caps ~1 bp; on bottom-decile small-mid caps ~3-5 bps. Average half-spread budget: 2.5 bps per side, 5 bps round-trip.
  - **Market impact**: Almgren-Chriss schedule on 0.1-0.5% of ADV consumption. Estimated 1-3 bps per side at $5M long basket per name, 2-5 bps per side at $25M per name. Budget 2 bps per side, 4 bps round-trip.
  - **Short borrow cost**: average general-collateral rebate -25 bps annualized = -2 bps per month. Bottom-decile may include hard-to-borrow names at -50 to -300 bps annualized — explicit budget 10 bps/month average on short side, with a hard-reject if any name borrows above -500 bps annualized.
  - **Total cost stack**: ~10-15 bps round-trip per name per month, on ~35% turnover = ~4-5 bps cost drag per month = ~50-60 bps annualized for the top/bottom-decile portfolio. Conservative budget: **75 bps annualized cost drag** modeled into all OOS results.
- **Statistical thresholds (verdict)** — per spec Lesson 2, **rank IC is evaluated BEFORE portfolio Sharpe**:
  - **Primary test — Spearman rank IC**: monthly Spearman correlation of model output `ŷ_i` against realized forward-21-day excess return `y_i`, averaged across 60+ OOS months. **Pass threshold: mean IC ≥ 0.025 with t-stat > 2.0 (Newey-West 3-lag-adjusted) over the OOS window.** This is the spec-mandated test of ranking power — the strategy's economic claim is rank-based, so rank IC is the appropriate test, not hit rate. (Gu-Kelly-Xiu report OOS monthly IC of 0.03-0.05 for ML methods on the broader US universe; 0.025 is a conservative floor that buys ~3% of cross-sectional dispersion.)
  - **Secondary — decile-spread portfolio Sharpe**: long-top-decile minus short-bottom-decile, equal-weighted, net of the 75 bps cost stack — **annualized Sharpe ≥ 0.75 with 95% bootstrap CI excluding 0.35** over the 2016-2026 OOS.
  - **Tertiary — max drawdown**: < 18% over OOS.
  - **Quaternary — turnover stability**: monthly turnover should stay in 25-50% range; >65% in any month flags a model instability and triggers manual review.
  - **Robustness checks (must pass for promotion)**:
    - Quant-meltdown August 2007 simulation (no live data, but feed in the equivalent training window and check expected loss given factor crowding) — handled in §10.
    - Momentum-crash 2009 Q2 and 2020 Q2 specific months: portfolio loss in those single months should be < 8%.
    - SHAP feature-importance attribution: top 10 features should be a sensible mix of factor primitives and alt-data, NOT dominated by a single alt-data feature (sanity check against overfit).
- **Minimum OOS sample**: 60 monthly rebalances (the spec requirement). Full 2016-2026 is ~120; we use the full window as headline.

## 7. Risk envelope

- **Account assumed**: $250,000 USD funded margin account (full long-short variant), OR $100,000 cash account (long-only top-decile variant). Numbers below are for $250k margin.
- **Max position per single name**: **1.0% of NAV ($2,500 per name)** equal-weight in each decile — explicit single-name idiosyncratic-risk cap per spec requirement. 100 long names × 1.0% = 100% long gross; 100 short names × 1.0% = 100% short gross; total gross = 2.0x, net market exposure ~0%.
- **Sector concentration cap**: no single GICS sector >20% of gross long or gross short notional. If model output produces sector concentration above this (e.g., bottom-decile is 35% energy in an oil crash), the offending sector is capped and the excess weight redistributed pro-rata to the next-ranked names in other sectors. Cap is enforced as a constraint in the portfolio-construction step, not as a feature filter.
- **Daily loss cap**: 1.5% of NAV = $3,750/day. If breached, halt new orders for 24h, do not unwind existing positions (the strategy is monthly; intraday loss is usually transient).
- **Drawdown trigger**: 20-day rolling DD > 6% triggers 50% deleverage (cut gross from 2.0x to 1.0x by halving every name); > 12% triggers full flat and 30-day strategy pause + manual review.
- **Circuit breaker (strategy-level)**: 3 consecutive losing months triggers a strategy-level pause and forensic review (Lesson 10). 4 consecutive losing months auto-kill pending review. This is loose because cross-sectional ML strategies are known to have 2-4 month draw-down spans (Gu-Kelly-Xiu fig. 5).
- **Factor-crowding alert**: at each rebalance, compute correlation of the portfolio's *factor exposure profile* (HML, SMB, UMD, QMJ, BAB loadings via Fama-French factor regression on recent returns) against the Russell-1000 hedge-fund composite (publicly available 13F crowding metric from S3 Partners or proprietary). If portfolio's net QMJ + UMD + HML exposure is in the >90th percentile of historical hedge-fund crowding (the **Aug-2007 condition**), automatically reduce gross leverage by 30% for that month. This is the explicit Lesson 10 / Khandani-Lo 2007 quant-meltdown mitigation.
- **Correlation rule** with already-deployed strategies (Lesson 6): must check rolling 60-day correlation < 0.3 with any deployed US-equity long-short or factor strategy. **Specific collision check**: iter_1 (US small-cap overnight mean-rev) trades 5-min/overnight, different horizon, expected correlation ~0.0-0.1 — OK. iter_9 (Nifty cross-section momentum) different geography but shares the momentum factor — expected correlation 0.2-0.3 on momentum-crash days; flagged but acceptable. **Explicit hard-reject if a "cross-section ranking on developed-market equity" iteration already occupies a slot** — none of iter_1 through iter_13 do, so iter_14 is non-colliding (Lesson 6).
- **PDT non-applicability**: holding period is ~21 trading days; positions span overnight; no trade qualifies as a day trade under FINRA 4210; PDT does not apply (Lesson 5).

## 8. Execution plan

- **Broker / SDK**: IBKR Pro via `ib_insync`. Margin account for full variant; cash account for long-only fallback. Order placement via `IB.placeOrder()` with `Stock(symbol, 'SMART', 'USD')` contracts.
- **Order type**: marketable-limit IOC with 5 bps slippage tolerance vs. the prior-day close. Each rebalance produces ~30-45 new buys and ~30-45 new sells per side (only on the names that crossed in/out of deciles). Orders split into 3 child slices over the first 30 minutes (09:30, 09:45, 10:00 ET) — VWAP-targeting, not aggressive. **Slippage budget: 5 bps per side per child slice, measured ex-post against arrival-price benchmark.**
- **Locate logic**: prior to placing any short orders, query IBKR's `reqAvailableShares()` / margin checker for borrow availability on every short name. If a name is unavailable, substitute with the next-ranked candidate (rank 101, 102, …) until the short basket has its full 100 names. Log substitutions for post-trade analysis.
- **Broker rejection handling (Lesson 4 — broker truth, not local cache)**: every order's filled quantity is reconciled against `IB.positions()` after each trading session. If local state and broker state diverge by ≥1 share on any name, the strategy auto-halts, dumps a diff report, and waits for manual intervention. No closing order is ever issued using a local quantity — closing quantity is always read from `IB.positions()` at close time.
- **Wash-sale tagging**: every order is tagged with a strategy-ID so a tax-aware accountant can post-process for wash-sale matching later. Not enforced in pre-trade logic (the cost of enforcing wash-sale optimization in the signal would distort the rank, defeating the spec).
- **Earnings-day handling**: if a held name unexpectedly schedules an earnings announcement inside the holding period (e.g., an unscheduled mid-month announcement), the position is **left in place** (we accept the earnings event risk as a 1-day vol spike, sized to 1% NAV cap which limits the damage). Pre-rebalance embargo (§5) prevents *known* upcoming announcements from being entered.

## 9. Capacity

- **AUM ceiling before edge degrades**: estimated **$200M-$400M long-short notional** ($100-200M per side). Reasoning: at $200M per side, the per-name slug is ~$2M. The bottom-decile names by momentum are often small-mid caps in the $1-3B market-cap range with ADV $20-80M. A $2M order at 0.5% ADV consumes 25 bps of impact per Almgren-Chriss square-root scaling. Above $400M total, the bottom-decile leg starts to materially move the names we are trading, the impact eats the spread, and the realized Sharpe halves. Top-decile names skew larger-cap and could absorb more, but the strategy is bottleneck-bound by the short side.
- **Hard capacity ceiling**: $400M long-short ($800M total gross) for the base design. To scale beyond, the strategy would need to (a) drop to a quintile-spread (200 names per side, more dilution but more capacity), (b) shift to a smarter execution algorithm (e.g., 5-day VWAP rather than 30-min VWAP), or (c) reduce turnover by adding a 2-month minimum-holding constraint that costs ~15% of the alpha but doubles capacity. None of these are in the base design — Lesson 9 honored by stating the ceiling upfront.
- **ADV consumption at full size**: at $400M LS, average single-name notional $4M; average Russell-1000 ADV is ~$120M; so ~3.3% ADV per name at the average. Bottom-decile small-mid caps will be at 5-10% ADV — the upper end of slippage-acceptable. Above this, slippage curves nonlinearly upward.
- **Concentration risk**: per §7, single-name max 1.0% NAV ($4M at $400M), single-sector max 20% gross. No name can exceed these caps even if model strongly prefers it. The equal-weight construction is the explicit choice to keep concentration risk floored, trading off some signal magnitude.

## 10. Failure modes

1. **Factor crowding / quant meltdown (Aug 2007 replay)**: the value+momentum+quality+low-vol bouquet is the most-crowded systematic-equity factor set on the planet. When a large multi-strat fund (LTCM-shaped, AQR-shaped) deleverages, every quant book that loads on the same factors faces simultaneous forced selling — the Aug-6-9 2007 unwind moved long-short factor returns -15% in 3 days for many books, and the names recovered most of the loss within a month, but funds with monthly margin calls and lower vol budgets were forced out at the bottom (Khandani-Lo 2007). **Detection**: HML+UMD+QMJ portfolio loading correlation with hedge-fund composite crowding metric (S3 Partners, Goldman Quant Insight). **Trigger**: if 5-day rolling factor-portfolio P&L is < -5% AND market-cap-weighted hedge-fund composite is in <30 days drawdown, auto-cut leverage by 50%. **Mitigation in design**: 1.0% NAV per name cap, 20% sector cap, 30-day-pause-on-3-consecutive-losing-months rule.
2. **Momentum crash (March-April 2009; March-May 2020)**: when the market sharply rebounds after a sustained crash, prior losers (the strategy's short leg) outperform prior winners (the long leg) violently — the classic Daniel-Moskowitz 2016 momentum-crash signature. Realized monthly losses can be -10 to -25% in a single month for an unhedged momentum book. **Detection**: 1-month market return < -15% AND 60-day VIX above 35. **Mitigation in design**: low-vol and quality factor exposures provide partial offset (quality holds up in crashes; low-vol holds up in rebounds); the ML ranker can also down-weight momentum dynamically if alt-data sentiment momentum has flipped — empirically the Gu-Kelly-Xiu tree-based model does this somewhat automatically. **Backup**: the §7 drawdown trigger flat-out halts the strategy if losses exceed 12% rolling 20d, which would catch a momentum crash before full destruction.
3. **Alpha decay on published features**: every factor in the base 70-feature set is publicly published. McLean-Pontiff 2016 ("Does Academic Research Destroy Stock Return Predictability?") shows that the average factor's premium shrinks ~58% post-publication. The strategy's gross premium is the *composite* of nonlinear interactions — those should decay slower than any single factor — but the rate of decay is uncertain. **Detection**: rolling 36-month rank IC trend — if IC trends down monotonically by >0.005 per year across 3 successive years, the model is decaying. **Mitigation**: alt-data features (Google Trends, news sentiment, option-skew, social) are the freshest and least-arbitraged inputs; if classical factors decay, the alt-data block should pick up slack. If alt-data also decays (i.e., everyone is now scraping Trends), the strategy has no remaining edge and is killed.
4. **Alt-data vendor drift / definition change**: Google Trends has changed its sampling methodology twice (2018, 2022). GDELT 2.0 GKG occasionally re-scores historical articles. RavenPack has revised its sentiment lexicon. StockTwits API has rate-limited and changed access tiers 4 times since 2015. **Detection**: monthly distribution-shift check — for each alt-data feature, compare current month's feature distribution against trailing 12-month distribution via KS-test; if p < 0.01, flag as drift. **Mitigation**: drift triggers a forced model retrain and a 30-day shadow period before the retrained model deploys live. If two alt-data features drift in the same month, halt the strategy. Spec explicitly marks **social-media (StockTwits/WSB)** as most fragile — first to drop if vendor relationship fails.
5. **Factor regime change**: prolonged growth-stocks-only regimes (2017-2020) can crush value-tilted books; prolonged value rotation (Nov 2020 - Mar 2022) crushes growth-momentum books. The strategy is multi-factor so partially insulated, but a sustained 18-month single-regime can produce extended drawdowns. **Mitigation**: walk-forward retraining every 12 months adapts to the regime; 3-consecutive-month-loss circuit breaker catches the rest.
6. **Earnings-leak / event-risk contamination**: if the earnings calendar feed is wrong (Zacks/IEX miss a date) the embargo fails and the strategy trades into an announcement. **Detection**: cross-check earnings calendar against EDGAR 8-K filings post-rebalance; if any held name reported earnings within ±5 days of rebalance, log as embargo failure. **Mitigation**: redundant calendar source (Zacks + IEX + EDGAR scraping); >1 embargo failure per quarter is a process-bug that gets fixed before next deploy.

## 11. Falsifiable verdict criteria

**Promote to live capital** (after the validation runs AND ≥30 shadow days, per Lesson 1):
- OOS 2016-2026 mean monthly Spearman rank IC ≥ 0.025 with t-stat > 2.0 (Newey-West 3-lag) — **primary, per spec Lesson 2**.
- OOS decile-spread Sharpe ≥ 0.75 net of 75 bps annual cost stack, 95% bootstrap CI excluding 0.35.
- OOS max drawdown < 18%.
- Shadow-period (30 trading days) realized rank IC > 0 in each of the 1 in-shadow monthly rebalance (in 30 days only 1 rebalance occurs; this is a process-check not a stat test).
- No broker-truth divergence in shadow (Lesson 4).
- No data-pipeline crash, no vendor outage > 1 day, no rebalance missed.

**Continue shadow / extended monitor** (if metrics are ambiguous):
- OOS rank IC in [0.015, 0.025) range with t > 1.5 — borderline, extend shadow to 90 days.
- Sharpe in [0.50, 0.75) range — extend shadow.
- One sector cap forced-rebalance per quarter — acceptable.

**Kill immediately** (any one trigger):
- OOS rank IC < 0.015 over the full 2016-2026 sample (the strategy has no ranking power, full stop).
- OOS Sharpe < 0.35 net of costs.
- 3-month rolling drawdown > 12% during shadow or live.
- Daily loss cap breach (Lesson 10 hard line).
- Broker-side fill discrepancy > 0.5% on any rebalance (Lesson 4 hard line).
- Two alt-data feature drift flags in the same calendar month.
- Factor-crowding alert (HML+UMD+QMJ exposure at >90th-percentile crowding) sustained for 2 consecutive months without strategy auto-deleverage triggering.

**90-day live falsification (3 monthly rebalances)**:
The strategy faces 3 monthly rebalances inside 90 days. Per spec, the falsification claim is:
- If **mean realized monthly rank IC < 0.02 across all 3 rebalances** (i.e., the model has near-zero ranking power on live data), kill — this is statistically weak (3 samples) but meaningfully so given the OOS-mean of ~0.025-0.03; 3 consecutive months at <0.02 is an empirically unlikely outcome (probability <15%) under the null that the OOS performance generalizes, and is a sufficient signal to suspect model breakage (data pipeline drift, regime change, alpha decay).
- AND if the **realized decile-spread cumulative return is negative across all 3 months** AND turnover stability is breached (any single month >65% turnover), kill — joint trigger to avoid killing on pure bad luck.
- Either trigger alone triggers a manual review; both triggers together = auto-kill within 90 days. This satisfies the spec hard-rule that a 90-day-window falsification criterion be named upfront.

## 12. One-paragraph operational risk note

The biggest operational surface area is the **feature pipeline**: ~70 features sourced from 8+ vendors (CRSP/Polygon, Sharadar/Compustat PIT, FTSE Russell historical reconstitution, Zacks/IEX earnings, Google Trends, GDELT/RavenPack, OptionMetrics, FINRA short interest, SEC EDGAR 13F) — every one of those is an independent failure mode. A single broken vendor (Polygon S3 plan change, Sharadar deprecating SF1 PIT, Google Trends rate-limiting harder, GDELT moving to paid) can silently corrupt a feature column, and the LightGBM model will happily produce predictions on corrupted data without flagging it. Detection requires per-feature distribution-drift KS-tests every month, per-vendor uptime monitoring, and a hash-check on point-in-time fundamentals so a vendor restating history is caught before it leaks into next-month retraining. The **retraining pipeline** is the second risk: 12-month walk-forward retraining means an irreproducibility (a library version change in LightGBM 4.x → 5.x changing default splits, a numpy float-precision quirk, a non-deterministic GPU kernel even though we use CPU only) silently changes the model and we won't notice until OOS Sharpe diverges. Mitigation: pin every dependency in a locked virtualenv, hash the training data and the model artifact at every retrain, store model artifacts in version-controlled S3, and run a "shadow retrain" on the prior year that should produce identical SHAP values within tolerance — if it doesn't, the retraining is broken. Finally, the **alt-data block** carries the largest reputational risk: if Google or GDELT changes its data licensing terms (this has happened), the strategy could face a takedown demand that would force shutdown until alternative data is procured. Spec marks social-media features as droppable; the option-skew, news-sentiment, and Google-Trends features are also droppable individually with ~5-15% model performance degradation each (per Gu-Kelly-Xiu feature-ablation sensitivity), but losing 3+ alt-data features simultaneously would degrade the strategy below the kill thresholds.

---

## Lesson-by-lesson coverage (per spec hard rule)

1. **Validate before deploy** — §6 walk-forward 2016-2026 + §11 ≥30-day shadow before any live capital.
2. **Binary gates, not confidence rankers** — acknowledged in §4 step 11: the ranker's continuous score IS the strategy's rank because cross-section by decile is the spec'd mechanism, AND §6 primary test is rank IC (Spearman correlation of score vs. realized rank) at t>2.0 BEFORE the portfolio Sharpe test — this is the spec-correct test of ranking power.
3. **TP:SL asymmetry is structural** — N/A in the classical sense (no per-trade TP/SL); the equivalent is the decile boundary itself, which is by construction (top 10%, bottom 10%) — neither tightened nor widened. The portfolio-level drawdown triggers (§7: 6% rolling DD → 50% deleverage; 12% → flat) act as the stop logic, sized to known Daniel-Moskowitz momentum-crash magnitudes.
4. **Broker truth, not local cache** — §8 broker-rejection handling reconciles `IB.positions()` against local state every session and closes only from broker-reported quantities. Hard halt on ≥1 share divergence.
5. **No PDT-exposed lanes on sub-$25k US equity accounts** — §3 + §7: 21-day holding window has zero day-trades by FINRA definition; PDT not applicable at any account size.
6. **One strategy per slot** — §7 collision check confirms no other iter_1..13 occupies "developed-market cross-section ML ranker" slot; iter_9 (Nifty CS momentum) is a different geography and a rules-based linear model; flagged but non-colliding.
7. **No same-data backtest** — §6: 2000-2015 train, 2016-2026 OOS, walk-forward with 12-month embargo. Hyperparameters frozen before OOS.
8. **Battle-tested broker SDKs only** — IBKR `ib_insync` only.
9. **Capacity bound named upfront** — §9: $200-400M LS notional, single-name 1% NAV, single-sector 20%, ADV consumption 3-10%.
10. **Kill criteria written before deploy** — §11 promote/shadow/kill thresholds in exact numbers; 90-day 3-rebalance falsification rule.

---

## Anchor papers / references

- Gu, Kelly & Xiu (2020). "Empirical Asset Pricing via Machine Learning." *Review of Financial Studies* 33(5): 2223-2273. → ML methods (especially trees and shallow NNs) deliver 2x OOS R² vs. linear factor models for next-month cross-section.
- Asness, Frazzini & Pedersen (2019). "Quality Minus Junk." *Review of Accounting Studies* 24: 34-112. → QMJ factor construction.
- Fama & French (2015). "A Five-Factor Asset Pricing Model." *Journal of Financial Economics* 116(1): 1-22. → underlying factor primitives.
- Khandani & Lo (2007). "What Happened to the Quants in August 2007?" *Journal of Financial Markets* (working paper version 2007). → factor-crowding meltdown reference.
- Daniel & Moskowitz (2016). "Momentum Crashes." *Journal of Financial Economics* 122(2): 221-247. → momentum-crash literature, sizing of drawdown triggers.
- McLean & Pontiff (2016). "Does Academic Research Destroy Stock Return Predictability?" *Journal of Finance* 71(1): 5-32. → alpha-decay rate empirics.
- Da, Engelberg & Gao (2011). "In Search of Attention." *Journal of Finance* 66(5): 1461-1499. → Google Trends as a retail-attention proxy.
- Xing, Zhang & Zhao (2010). "What Does the Individual Option Volatility Smirk Tell Us About Future Equity Returns?" *Journal of Financial and Quantitative Analysis* 45(3): 641-662. → option-skew predictability.
