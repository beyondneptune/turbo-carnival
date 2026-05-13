# Iteration 23 — Global Risk Premia Multi-Factor Portfolio (Carry × Momentum × Value across 50+ Futures + FX)

## 0. Composition note (read first)

This is the **flagship capital-deployment iteration** of the wave. Distinguish it explicitly from the two neighbors it most resembles:

- **iter_4 (`g10_fx_carry_diversified`)** — single asset class (G10 spot FX), **single factor** (carry), monthly rebalance. Capacity ceiling $1M retail / $50-100M institutional. Single-factor Sharpe target 0.55.
- **iter_7 (`dax_futures_tsmom`)** — single instrument (FDAX), **single factor** (time-series momentum), three-horizon combined. Single-instrument Sharpe target 0.6-1.0.
- **iter_23 (this file)** — **50-60 global futures + FX instruments, three orthogonal factors (carry + momentum + value) per instrument, equal-risk-budget combined, vol-targeted at portfolio level.** This is the AQR/Pedersen-style risk-premia portfolio. Capacity ceiling **$300M-$1B AUM** (the largest in the wave). Combined Sharpe target 1.0-1.5.

Per the spec hard rule "Must not duplicate the thesis or method of an earlier iteration": this iteration **shares vocabulary with iter_4 (the word "carry") and iter_7 (the word "momentum")**, but the *thesis is different* — the edge is the **diversification gain from combining three uncorrelated factors across 50+ instruments**, not any single factor on any single instrument. The integration math (equal-risk-contribution overlay, cross-instrument vol scaling, portfolio vol target) is the novel piece. The leg-internal carry and momentum mechanics are restated below in full because they are computed differently than in iter_4 (futures roll-implied carry, not FX swap-rate carry) and iter_7 (cross-sectional + time-series momentum, not single-instrument three-horizon TSMOM).

---

## 1. Thesis (one sentence)

Across a globally diversified universe of ~50-60 liquid futures and FX instruments (equity indices, government bonds, commodities, G10 currencies), the three academically documented risk premia — **carry** (yield/roll differential), **time-series + cross-sectional momentum** (12-minus-1 month return), and **value** (5-year mean reversion in real prices) — are sufficiently independent (pairwise factor return correlation < 0.30 in the academic record) that an **equal-risk-contribution combined signal**, position-sized to a portfolio-level 10% annualized volatility target with per-instrument vol scaling, will produce a **net-of-cost annualized Sharpe between 1.0 and 1.5 over a minimum 13-year out-of-sample window (2013-2026, 156+ monthly rebalances)** — falsifiable as: if any one of (a) the combined OOS Sharpe drops below 0.7 with 95% bootstrap CI excluding 1.0, (b) the average information coefficient (IC) of each factor across instruments falls below 0.01 over three consecutive monthly rebalances live, or (c) the pairwise factor return correlation rises above 0.60 for two consecutive monthly windows, the strategy is killed inside 90 days.

---

## 2. Market / asset class / venue

### Instrument universe (54 names targeted; 50 minimum after liquidity filtering)

**Equity index futures (12)** — all front-month, rolled per §8:
- US: `ES` (S&P 500 E-mini, CME), `NQ` (Nasdaq-100 E-mini, CME), `RTY` (Russell 2000 E-mini, CME), `YM` (Dow E-mini, CBOT).
- Europe: `FDAX` (DAX 40, Eurex), `FESX` (Euro Stoxx 50, Eurex), `Z` (FTSE 100, ICE Europe), `FCE` (CAC 40, Euronext).
- Asia: `NK` (Nikkei 225, Osaka via SGX/CME), `TPX` (TOPIX, Osaka), `HSI` (Hang Seng, HKEX), `KOSPI200` (KSE).

**Government bond futures (8)** — front-month:
- US: `ZN` (10-yr Treasury Note, CBOT), `ZB` (30-yr Treasury Bond, CBOT), `ZF` (5-yr Note, CBOT), `ZT` (2-yr Note, CBOT).
- Europe: `FGBL` (Bund, Eurex), `FGBM` (Bobl, Eurex), `FGBS` (Schatz, Eurex), `R` (Long Gilt, ICE Europe).
- Asia: `JGB` (10-yr Japanese Govt Bond, Osaka) — included if IBKR routing available; otherwise dropped to 7.

**Commodity futures (15)** — front-month with active roll:
- Energy: `CL` (WTI Crude, NYMEX), `BZ` (Brent, ICE EU), `NG` (Natural Gas, NYMEX), `HO` (Heating Oil, NYMEX), `RB` (RBOB Gasoline, NYMEX).
- Metals: `GC` (Gold, COMEX), `SI` (Silver, COMEX), `HG` (Copper, COMEX), `PL` (Platinum, NYMEX).
- Agricultural: `ZS` (Soybeans, CBOT), `ZC` (Corn, CBOT), `ZW` (Wheat, CBOT), `KC` (Coffee, ICE US), `SB` (Sugar, ICE US), `CT` (Cotton, ICE US).

**FX (G10 majors, 9 pairs, all USD-quoted)** — via IBKR IDEALPRO:
- `EUR/USD`, `USD/JPY`, `GBP/USD`, `USD/CHF`, `USD/CAD`, `AUD/USD`, `NZD/USD`, `USD/NOK`, `USD/SEK`.

**Short-rate futures (10)** — for the rates carry signal:
- `SR3` (3-mo SOFR, CME), `ES3` / `EURIBOR` (3-mo Euribor, ICE EU), `L` (3-mo Short Sterling, ICE EU), `EYZ` (3-mo Euroyen TIBOR, TFX).
- Plus 6 additional country-specific short-rate proxies (AUD BAB, NZD BKBM, CAD BAX, CHF SARON, NOK NIBOR, SEK STIBOR) where IBKR provides a contract — otherwise carry is computed from spot+forward swap quotes directly per §4 below.

**Total target universe: 54 instruments.** A liquidity gate at §9 may drop 2-4 names (typically PL, SB, the JGB, and 1-2 short-rate contracts depending on IBKR data permissions) — minimum 50 names live.

### Venue and broker

- **Primary broker**: Interactive Brokers via `ib_insync`. Battle-tested per Lesson 8. Single account holds all futures + FX (this is a defining IBKR strength — no multi-broker fan-out needed).
- **Sub-venues** (routed automatically by IBKR Smart): CME Group (CME/CBOT/NYMEX/COMEX), Eurex, ICE Europe/US, Euronext, Osaka Exchange, HKEX, KSE.
- **FX execution**: IBKR IDEALPRO (institutional ECN, sub-bp spreads on majors at $1M+ ticket sizes).

### Settlement, margin, tax

- **Settlement**: each contract per its native rules (physical for energy/grains held to delivery — we never hold to delivery, always roll; cash-settled for indices and most rates; spot for FX with auto-rolled tom-next swap).
- **Margin**: IBKR portfolio-margin (PM) account required for AUM ≥ $110k — gives ~3-5x more buying power vs. Reg-T on a futures portfolio with low realized correlation. Initial margin per contract ranges from ~$1k (Schatz, small currencies) to ~$28k (FDAX, ZB during high-vol regimes).
- **Tax**: Section 1256 60/40 treatment for nearly all of these contracts in a US account (broad-based index futures, foreign-currency futures, regulated-exchange commodity futures — all qualify). FX spot via IDEALPRO is Section 988 ordinary unless explicit opt-out for 1256 is filed. Tax efficiency is a real edge contributor at full size (~150-200 bps/yr vs. ordinary income on equivalent equity strategies) but not exploited as a primary edge.

---

## 3. Time horizon

**Daily/weekly evaluation, monthly rebalance, 1-3 month typical holding period.**

- **Signal computation**: daily after each market's close, ingested into a portfolio-level state store. Per-factor signals recomputed daily for monitoring but trades are not placed off daily moves.
- **Rebalance cadence**: **monthly**, on the third Wednesday after global cash close (this avoids the calendar quirks of month-end and is a published academic convention; AQR uses it). Orders execute over a 2-day window (Wed/Thu) split across regional sessions to spread market impact.
- **Holding period**: positions persist until the next monthly rebalance. Typical sign-flip frequency per instrument per factor is 4-8 months for carry, 6-12 months for value, 3-6 months for momentum — so a 1-3 month holding period is the right characterization for the **combined** signal, with rebalances usually adjusting size rather than flipping direction.

Per the spec time-scale bucket: **daily (signal grid) with monthly rebalance / multi-day-to-multi-month holding**.

---

## 4. Method family and specific algorithm

**Family**: Rules-based, deterministic risk-premia portfolio construction. No ML, no NN, no LLM. Every parameter is pre-registered from the academic literature (Asness/Moskowitz/Pedersen 2013, Koijen/Moskowitz/Pedersen/Vrugt 2018, Pedersen 2015) — no in-sample tuning.

The algorithm runs once per month (third Wednesday after global cash close, 22:00 UTC) and produces one target notional vector across the 50+ instruments.

### Step 1 — Per-instrument factor signals

For each instrument `i` in the universe, compute three raw factor signals at month-end `t`:

**Carry signal `C_i(t)`** — defined per asset class:

- **Equity index futures**: `C_i = (S_i - F_i) / F_i × (365 / days_to_expiry)`, the annualized implied dividend yield minus financing rate embedded in the futures-spot basis. Long carry when futures trade at discount to spot (positive implied dividend yield net of rate); short carry when futures trade at strong premium.
- **Bond futures**: `C_i = yield_to_maturity_of_CTD - short_rate`, the "carry on the curve" — the yield the cheapest-to-deliver bond earns above the short-rate. Long bond carry when the curve is upward-sloping; short when inverted.
- **Commodity futures**: `C_i = -(F2 - F1) / F1 × (365 / (T2-T1))`, the annualized backwardation. **Positive when in backwardation (front above back, futures rolling down toward spot — the holder of futures earns roll yield); negative in contango.** This is the standard Koijen et al. (2018) commodity carry.
- **FX (spot)**: `C_i = i_foreign - i_USD`, the 1-month OIS differential. Same as iter_4 §4 carry definition.
- **Short rates**: `C_i = -slope of the front-end strip` (the "future short rate minus current short rate" implied by the strip).

**Momentum signal `M_i(t)`** — uniform across asset classes:

- `M_i = log(P_i(t-21)) - log(P_i(t-273))` — the "12 minus 1" month total return, i.e., return from 12 months ago up to 1 month ago. The 1-month skip is the Jegadeesh/Titman/Asness lookback-skip convention that mitigates short-term reversal contamination.
- For futures, use the back-adjusted continuous series (open-interest-based roll — see §8).
- For FX, use spot total return including swap accrual.

**Value signal `V_i(t)`** — 5-year mean reversion:

- **Equity indices**: `V_i = log(real_earnings_yield_i(t)) - log(mean(real_earnings_yield_i(t-1260 .. t)))`. Positive value = current earnings yield rich vs. 5-year mean = expected to revert up = long signal. (Real earnings yield = trailing-12m EPS / index level / GDP deflator.)
- **Bond futures**: `V_i = real_yield_i(t) - mean(real_yield_i(t-1260 .. t))`. Positive value = current real yield high vs. 5-year mean = expected to revert down (prices up) = long bond signal.
- **Commodity futures**: `V_i = -1 × (log(P_i(t)) - log(mean(P_i(t-1260 .. t))))`, the 5-year deviation from log-price mean, signed so positive = cheap vs. own history = long signal. This is the AMP 2013 commodity value definition.
- **FX**: `V_i = -1 × (log(real_FX_i(t)) - log(mean(real_FX_i(t-1260 .. t))))` where `real_FX = nominal × foreign_CPI / domestic_CPI`. PPP-anchored long-horizon mean reversion, signed so positive = undervalued = long.
- **Short rates**: `V_i = -1 × (forward_rate_i(t) - mean(forward_rate_i(t-1260 .. t)))`.

### Step 2 — Cross-sectional standardization

Within each asset-class group (equities, bonds, commodities, FX, rates), z-score each factor across instruments at time `t`:

```
z_C_i(t) = (C_i(t) - mean_group(C(t))) / stdev_group(C(t))
z_M_i(t) = (M_i(t) - mean_group(M(t))) / stdev_group(M(t))
z_V_i(t) = (V_i(t) - mean_group(V(t))) / stdev_group(V(t))
```

Z-scores are then clipped at ±2 to prevent any single instrument from dominating during a regime shift.

### Step 3 — Combined factor score

Equal-weight the three z-scores (the equal-risk-contribution choice; see §4 rationale below):

```
F_i(t) = (z_C_i(t) + z_M_i(t) + z_V_i(t)) / 3
```

`F_i` is dimensionless. Positive = long signal, negative = short signal, magnitude = signal strength.

### Step 4 — Per-instrument volatility scaling

For each instrument, target an equal volatility contribution. Let `σ_i(t)` = annualized realized vol from the trailing 60 trading days of daily log returns of the back-adjusted continuous series.

```
notional_target_i(t) = (target_vol_per_instrument / σ_i(t)) × F_i(t) × E
```

Where `E` is total account equity in USD and `target_vol_per_instrument = 0.10 / sqrt(N_effective)`. With ~50 instruments and the academic average pairwise return correlation of ~0.15, `N_effective ≈ 12`, giving `target_vol_per_instrument ≈ 2.9%` annualized per leg.

### Step 5 — Portfolio vol-target overlay

After Step 4, compute the ex-ante portfolio volatility from the historical 60-day instrument-return covariance matrix `Σ`:

```
σ_portfolio = sqrt(w^T Σ w) × sqrt(252)
where w = vector of notional_target_i / E
```

If `σ_portfolio ≠ 10%`, scale all positions by `0.10 / σ_portfolio` to hit the portfolio target. This is the **second-tier vol scaling** (per-instrument vol scaling is Step 4; portfolio-level is Step 5) and is the AQR canonical approach.

### Step 6 — Position caps

Three hard caps applied last (per Lesson 9 and §7):
- **Max single-instrument notional**: 5% of equity in absolute terms.
- **Max sector/region exposure**: 25% gross notional per asset-class group (equities, bonds, commodities, FX, rates).
- **Max gross leverage**: 4x equity (sum of |notional_i| across instruments / E ≤ 4.0).

### Why equal-risk weighting, not optimal mean-variance

The risk-premia literature (AMP 2013, KMPV 2018, Pedersen 2015 Ch. 12) is unanimous: optimal mean-variance weights are dominated out-of-sample by equal-risk-contribution weights because the covariance matrix is estimable but the return-vector estimates are noise-dominated at monthly frequency on 50 instruments. Equal-risk gives up <5% of theoretical Sharpe in-sample for ~30% better out-of-sample Sharpe stability. We do not optimize weights — we set them.

### Why three factors, not five or seven

The literature documents 5+ factors that have shown academic significance (carry, momentum, value, defensive/low-vol, quality, trend-following, seasonality). We include only **three** because:

1. The three chosen factors are the most thoroughly out-of-sample-tested (AMP 2013 walks forward to 2011; KMPV 2018 walks forward to 2012; multiple replications since).
2. Adding the 4th and 5th factor improves backtest Sharpe by ~0.1 each but the marginal OOS contribution is statistically insignificant and risk of factor crowding (see §10) is concentrated in those.
3. Pre-registered restraint per Lesson 7 — the spec demands no in-sample tuning. We stick to the three pre-registered factors and accept the lower in-sample Sharpe as the cost of OOS robustness.

---

## 5. Data requirements

| Feed | Use | Vendor | Cost | Storage |
|---|---|---|---|---|
| Daily OHLC for all 54 instruments (futures continuous + FX spot), **1990-present** | Backtest, signal computation, vol estimation | IBKR historical (`reqHistoricalData`, daily bars are unlimited and free with appropriate exchange subscriptions) + Stooq free historical for pre-2000 cross-checks + CRSP-equivalent commodity data via Norgate or CSI Data ($150-300/mo paid; or Stooq+Quandl free for academic-quality) | $0 with IBKR; up to $300/mo paid for clean pre-2000 commodity history | ~500 MB total |
| Front-month + 2nd-month futures prices (for carry signal) | Carry calculation (F2-F1 backwardation) | IBKR (per-instrument quote permissions; bundled cost ~$30-60/mo across the universe) | ~$45/mo | ~50 MB/yr live |
| Open interest series per instrument per contract month | OI-based roll trigger (see §8) | IBKR (free with quote permission) + CFTC COT reports (free) | $0 | ~10 MB/yr |
| Macro: CPI, GDP deflators per country, central-bank policy rates, OIS curves | Value signal real-yield/real-FX construction, FX carry | FRED (US, free), BIS Macro (free), IMF IFS (free), ECB SDW (free), BoJ stats (free) | $0 | ~50 MB |
| Equity index trailing-12m EPS series per index | Equity value signal | IBKR fundamental data ($10/mo subscription) + S&P Global free aggregates | $10/mo | ~10 MB |
| Live L1 quotes during rebalance window | Execution sizing | IBKR per-exchange bundles (CME ~$10/mo, Eurex ~$13/mo, ICE ~$8/mo, Osaka ~$15/mo, HKEX ~$15/mo, etc.) | ~$80-120/mo across exchanges | ~5 GB/yr live capture |

**Total monthly data cost**: $135-275/mo (varies with pro/non-pro designation; this strategy is large enough to comfortably absorb pro). **Total storage**: <10 GB. Compared to equity-microstructure strategies that need terabytes of tick data, this is light.

**Latency requirement**: 5-second order ack on every rebalance leg is sufficient. Monthly-horizon strategy; sub-second execution is not the edge. Round-trip <500ms is comfortable for all IBKR routes.

---

## 6. Validation plan

This is the most critical section. Multi-factor cross-instrument strategies are the most overfit-prone family in the literature, so the walk-forward design must be aggressive.

### Sample and window

- **Sample**: daily data, 1990-01-01 to 2026-04-30. ~36 years per instrument where data exists (some instruments have shorter histories — see asset-availability table below).
- **In-sample window**: 1990-2012 (23 years, ~276 monthly rebalances available). Used for: (a) confirming that factor returns have the documented signs and IC, (b) sanity-checking the covariance and vol estimation pipeline, (c) **NO parameter tuning** — all rule parameters are pre-registered from AMP 2013 and KMPV 2018.
- **Out-of-sample window**: 2013-2026 (13.3 years, **156+ monthly rebalances**). This is the test set. **Zero exposure to OOS during signal design or parameter selection** — verified by git-tagging the rule definition file before any OOS code is run.

### Why 1990 start

The carry premium has been documented since 1980s data and is robust across the 1990s and 2000s. The momentum premium has 1990s data weaknesses in some commodity sub-markets but the cross-asset average is strong. The value premium needs 5+ years of history just to compute the first signal, so 1990 is the practical earliest start for a clean walk-forward.

### Walk-forward design

1. **Pre-OOS sanity check (1990-2012)**:
   - Each factor's monthly IC (Spearman correlation between signal and next-month return) computed per instrument per year.
   - **Required**: median IC per factor across instruments > 0.04 (the threshold below which AMP 2013 considers the factor inactive).
   - **Required**: pairwise factor return correlation < 0.30 over the full IS window.
   - **If any required threshold fails on IS**: the strategy is dead. Do not run OOS.

2. **OOS evaluation (2013-2026, rolling)**:
   - Expanding training window: at the start of each calendar year, recompute the covariance estimator and the per-instrument vol baselines using all data through year-end Y-1; then run the next 12 months as test.
   - **Step**: 1 calendar year. By end of 2026 OOS, this gives 13 expanding-window steps.
   - **Per-month**: at each monthly rebalance, compute signal using only data available through the rebalance date (no lookahead).
   - **Aggregate metric**: net-of-cost annualized Sharpe across the full 156 OOS rebalances, with 95% bootstrap CI from 10,000 block-bootstrap resamples (block size = 12 months to preserve autocorrelation).

3. **Purge and embargo**: 1-month embargo at the IS/OOS boundary (Dec 2012). Within OOS, the expanding-window steps are non-overlapping at the year boundary so no purging within OOS is needed; the monthly signals are forward-only.

### Cost model (conservative)

- **Futures**: round-trip commission $4 per contract + 0.5 bps slippage on liquid front-months (ES, NQ, ZN, CL, GC, EUR/USD) up to 5 bps slippage on less liquid (NK, HSI, JGB, KC, SB) = blended ~2 bps/leg per monthly rebalance.
- **FX**: 1 bps round-trip on IDEALPRO at the strategy's notional sizes.
- **Annualized drag**: ~30 bps/yr at full size — well-documented industry estimate for AQR/Pedersen-style implementations.

### Statistical thresholds (verdict)

- **Pass / promote**:
  - Annualized OOS Sharpe > 1.0 with 95% bootstrap CI excluding 0.7.
  - OOS max drawdown < 18% (the academic literature reports ~12-15%; we allow some buffer).
  - Each individual factor's OOS IC > 0.02 on average across instruments — confirms the diversification thesis (no single factor carrying the portfolio).
  - Pairwise OOS factor return correlation < 0.45 (we allow some slippage from the IS 0.30 because regimes change).
- **Continue shadow / extend test**:
  - OOS Sharpe in [0.5, 1.0], no kill trigger, factor ICs at least non-negative.
- **Reject / dead**:
  - OOS Sharpe < 0.5 with CI including 0, OR
  - OOS max DD > 25%, OR
  - Any factor's average IC < 0 over the OOS window (factor is anti-edge — definition obsolescence), OR
  - Pairwise factor correlation > 0.60 over any 3-year rolling window (factors crowded together — diversification thesis broken).

### Minimum OOS sample

156 monthly rebalances is the design target (full 2013-2026). **Minimum** for any verdict: 60 OOS rebalances (5 years). Below 60, the bootstrap CIs are too wide to distinguish 1.0 from 0.5. This is why iter_23 deliberately starts the OOS at 2013 rather than later — to guarantee the 60-rebalance floor is far surpassed by deployment time.

---

## 7. Risk envelope

This iteration is sized for the largest capital deployment in the wave. Account assumed: **$5M-$50M AUM** (illustrative; risk envelope scales linearly).

### Hard limits

- **Max single-instrument gross notional**: 5% of equity. At $10M AUM = $500k per instrument absolute cap. This binds during high-conviction signals (e.g., extreme carry + momentum + value alignment on a single commodity).
- **Max sector/region exposure**: 25% gross per asset-class group (equities, bonds, commodities, FX, short-rates). Prevents any single global regime (e.g., 2008 equity rout) from sinking the portfolio.
- **Max gross leverage**: 4x equity. Total `Σ|notional_i| / E ≤ 4.0`. With portfolio vol-targeted at 10% and instrument vols averaging ~15-25%, 3-4x gross is the natural operating level after the equal-risk overlay.
- **Daily loss cap**: 2% of equity / day. Hard halt on breach for 24h with a portfolio-flat option if breach is from a single-factor blowup.
- **Monthly drawdown trigger**: rolling 30-day DD > 6% triggers a 50% deleverage at the next daily NY close; > 10% triggers a portfolio flat and a 5-business-day review before redeploy.
- **Year-to-date drawdown kill**: YTD DD > 15% triggers full strategy shutdown and external review (factor crowding suspected).
- **Circuit breaker**: 3 consecutive losing months triggers a factor-IC audit, not auto-shutdown (multi-factor strategies routinely have 3-4 losing months around regime transitions — March 2020, late 2022 rates shock — and recover).

### Correlation rule with other deployed strategies

- **Hard requirement**: pairwise 60-day-rolling daily-return correlation with **any** other deployed strategy must remain < 0.40. The combined-portfolio P&L is broad and will leak into many other things; in particular:
  - vs. iter_4 (G10 FX carry): expected correlation ~0.4-0.6 because iter_23 contains the same FX carry signal as a sub-component. **Do not deploy iter_23 alongside iter_4 — they share the FX carry sleeve.** Deploy one or the other. Lesson 6 ("one strategy per slot") binds here: iter_23 is the strict superset of iter_4's edge plus 45 more instruments and 2 more factors.
  - vs. iter_7 (DAX TSMOM): iter_23's momentum signal on FDAX is a 12-1 month, while iter_7's is a 1/3/12 combined. Expected correlation 0.5-0.7 on the FDAX sleeve alone, but the FDAX sleeve is <2% of iter_23's notional, so portfolio-level correlation should be < 0.30. **Verify with 60-day-rolling daily correlation < 0.40 before joint deployment.**
- **Soft requirement**: correlation with any equity beta strategy (S&P-long, sector-rotation, etc.) < 0.30 — iter_23 should be near-market-neutral on the equity sleeve in normal regimes.

### Margin budget

- IBKR portfolio-margin: at $10M equity, the portfolio described above (3-4x gross, vol-targeted 10%) uses ~20-30% of available PM, leaving 70-80% buffer for a 2008-style vol expansion. This is the deliberate design — the strategy is **not** margin-bound except during multi-sigma cross-asset shocks.

---

## 8. Execution plan

### Broker and SDK

- **IBKR via `ib_insync`**. Single account, all 54 instruments routed through IBKR Smart. Battle-tested per Lesson 8.

### Order types

- **Futures**: marketable-limit IOC with a 3-bps slippage tolerance vs. mid-quote at signal generation. For less-liquid contracts (NK, HSI, JGB, KC, SB), widen to 6 bps.
- **FX (IDEALPRO)**: marketable-limit IOC with 1.5 bps tolerance on majors, 3 bps on USD/NOK and USD/SEK.

### Order scheduling

- Rebalance window spans **two business days** (Wednesday + Thursday following the third Wednesday signal date) to spread market impact and align with each region's liquid hours:
  - **Wednesday 22:30 UTC** (early Asia open): Asia equity index futures (NK, TPX, HSI, KOSPI200), JGB.
  - **Thursday 07:00 UTC** (European open): European equity index futures (FDAX, FESX, Z, FCE), Bund/Bobl/Schatz, Gilt.
  - **Thursday 13:30 UTC** (US open): US equity index futures (ES, NQ, RTY, YM), US Treasury futures (ZN, ZB, ZF, ZT), commodities (CL, NG, GC, SI, HG, agriculture), short-rate futures (SR3), FX rebalance.
- Within each window, orders are submitted sequentially per instrument with 30-60 second spacing — total ~30-60 instrument orders per day spread across 2-3 hours per session.

### Slippage budget and measurement

- **Budget**: 2 bps/leg blended across the universe. Realized slippage measured ex-post as (fill price - mid quote at signal time) per leg, aggregated per asset-class.
- **Tracking**: 12-month rolling slippage per asset-class; if commodity slippage exceeds 5 bps for 2 consecutive rebalances, downgrade to a 3-day execution spread on commodity legs.

### Broker rejection handling

- **On reject**: retry once with a tighter limit, then escalate to a wider IOC marketable. After 3 failed attempts, log the leg as "skipped" and reduce remaining-rebalance notional pro-rata. Per Lesson 4, reconcile against `IB.positions()` after every fill; trust broker truth, not local cache.

### Roll handling — CRITICAL data quality piece per Lesson 7

This is the most operationally subtle part of the strategy. **Naive month-end roll is wrong.** Per the strategy requirements:

- **Roll trigger**: when the open interest of the next contract (M+1) exceeds the open interest of the front contract (M). Computed daily from IBKR's reported OI fields. Confirmed via CFTC COT reports for US futures (free, weekly).
- **Roll execution**: simultaneous sell-current + buy-next (or buy-current + sell-next) as a calendar-spread order to minimize the roll basis. IBKR supports this natively via `ComboLeg`.
- **Continuous-series adjustment**: **ratio adjustment**, not Panama (additive). The Panama method introduces a multiplicative bias when prices change by orders of magnitude (e.g., NG over the 2000s); ratio preserves total-return consistency at the cost of slight historical-volatility drift, but vol is recomputed continuously off rolling 60-day windows, so the drift is irrelevant. Documented in Marshall/Nguyen/Visaltanachoti (2012) and the de facto AQR convention.
- **Verification**: every roll's basis (the spread between old and new contract) is logged and compared against historical median; deviation > 3σ triggers a manual roll-quality review before the next rebalance.
- **Pre-roll embargo**: positions in any contract within 5 business days of expiry are flagged; if a rebalance falls inside the embargo, the rebalance for that instrument is shifted to the post-roll session.

This is the single highest-risk operational sub-system. Get it wrong on WTI in April 2020 and the strategy reports a fake +30% gain or -20% loss depending on roll convention.

### Recall: lessons-binding execution constraints

- Lesson 5 (PDT): **N/A — this is a futures+FX strategy.** PDT applies only to US securities margin accounts. Compliant by structural exemption.
- Lesson 4 (broker truth): `IB.positions()` reconciliation after every fill; portfolio NAV reconciled hourly during the rebalance window and end-of-day always.

---

## 9. Capacity

This iteration is the **largest-capacity candidate in the wave** by design.

### Per-instrument capacity (ADV-bounded)

Approximate consumption at $500M AUM (full size), instrument-by-instrument:

| Instrument | ADV (notional $) | Strategy notional at $500M | % of ADV |
|---|---|---|---|
| ES | $250-400B/day | $25M | 0.008% |
| NQ | $150B/day | $20M | 0.013% |
| ZN | $80B/day | $25M | 0.031% |
| CL | $80B/day | $20M | 0.025% |
| GC | $30B/day | $15M | 0.05% |
| EUR/USD | $400B/day (IDEALPRO + interbank) | $25M | 0.006% |
| NK | $5B/day | $10M | 0.20% — meaningful but tolerable |
| HSI | $4B/day | $8M | 0.20% — same |
| KC (Coffee) | $1B/day | $5M | 0.50% — flagged for slow execution |
| SB (Sugar) | $1.5B/day | $5M | 0.33% |

### AUM ceiling

- **Soft ceiling (no execution degradation)**: **$300M AUM**. Up to this size, every instrument consumes <0.5% of ADV and the 2-day rebalance window absorbs market impact comfortably.
- **Hard ceiling (edge erosion from impact)**: **$1B AUM**. Beyond this, the agricultural commodity sleeve (KC, SB, CT) becomes execution-bottlenecked and the realized strategy Sharpe drops 20-30 bps from the theoretical.
- **Industry-scale ceiling (sector-wide alpha decay)**: ~**$50-100B**. If this strategy is run at AQR-fund scale across all participants, the factor crowding reduces the cross-sectional spread by 30-50%. We are running at retail-large-pod scale, so this is not our binding constraint — but it is the binding constraint on the asset class as a whole, and we should monitor it (see §10).

### Concentration risk

- **Single-instrument max**: 5% of equity (per §7).
- **Single sector/region max**: 25% (per §7).
- **Implicit cross-sectional concentration**: the 9 G10 FX pairs span 5 regions, the 12 equity index futures span 4 regions, the 15 commodities span energy/metals/agri. No single geography dominates the portfolio.

---

## 10. Failure modes

The multi-factor risk-premia approach has well-documented failure modes — none of which were predicted by the original AMP 2013 paper, all of which were observed in the decade since.

1. **Factor crowding / alpha decay from imitation.** This is the existential risk. AQR, Two Sigma, AHL, Winton, Bridgewater All Weather variants, and dozens of smaller pods all run flavors of carry+momentum+value. When the trade is crowded, simultaneous deleveraging hits all books at once.
   - **Historical instance**: **August 2007 quant crisis** — the Lehman/Sowood/AQR August 2007 deleveraging event flattened factor returns for 3 days and caused 10-30% drawdowns at most factor funds. Pre-warned by Andrew Lo et al.
   - **Historical instance**: **March 2020 COVID shock** — risk-parity funds (including factor funds) deleveraged into a falling correlation regime, amplifying their losses.
   - **Historical instance**: **2020 momentum reversal** — March 2020 to November 2020, classic 12-1 momentum was severely negative on equity factor portfolios; many fund flows out of momentum-tilted books.
   - **Metric**: monthly cross-correlation between this strategy's returns and the HFRI Macro-Systematic Diversified index; if rolling 12-month correlation > 0.7, crowding is real.
   - **Metric**: cross-sectional dispersion of factor signals (stdev of `F_i` across instruments) trending down over a 24-month window — signals are converging because everyone is acting on the same data.
   - **Mitigation**: deleverage by 25% if the HFRI correlation exceeds 0.7 for 3 consecutive months. **Acceptance**: this is a popular strategy at scale and we cannot fully hedge crowding; the diversification across 50+ instruments is the structural defense.

2. **Correlation regime change in crisis.** Factors that historically have <0.30 pairwise correlation can converge to 0.7-0.9 in a system-wide liquidity event (2008 Q4, March 2020, August 2024 yen unwind). The portfolio's effective diversification collapses.
   - **Metric**: pairwise factor return correlation, computed daily over the trailing 30 days. If any pair exceeds 0.60 for 5 consecutive days, the diversification thesis is broken for that window.
   - **Mitigation**: when the correlation circuit fires, the portfolio scales gross exposure to 50% until the correlation drops back below 0.40 for 5 days. The factor most correlated with the others is the one that gets cut first (typically momentum during a sharp reversal).

3. **Sub-asset-class wipeout / black-swan instrument shock.** A single instrument can experience a regime collapse that no continuous signal captures.
   - **Historical instance**: **WTI crude futures went negative on April 20, 2020.** Front-month WTI traded at -$37.63. Any naive carry signal calculator that assumed positive prices, any vol estimator that used log returns, any back-adjusted continuous series using ratio rolls — all broke. Some commodity carry funds reported nominal "infinite" carry signals that triggered max-long positions into the crash.
   - **Metric**: any single instrument's daily return < -10% or > +10% (or any continuous-series ratio adjustment that crosses zero) triggers an instrument-level halt and a 5-day exclusion.
   - **Mitigation**: in addition to the halt, the value signal must use **arithmetic-log mean** with a floor at 1% of the 5-year-trailing minimum to prevent any signal from going to ±∞ when prices approach zero. The carry signal must use a non-negative price floor for the F1/F2 calculation.

4. **Factor definition obsolescence.** A factor that worked for 30 years can stop working because the underlying mechanism changes.
   - **Historical instance**: **G10 FX carry post-2008** — global central-bank coordination compressed the interest-rate dispersion, reducing the FX carry spread by 50%; the strategy still earned positive Sharpe but well below the pre-2008 record (this is partly the iter_4 thesis).
   - **Historical instance**: **bond carry post-2014** — global QE compressed the term premium; the slope-of-the-curve carry signal had a ~3-year period of near-zero IC.
   - **Metric**: rolling 24-month IC per factor per asset-class; if any drops below 0.01 for 24 months, the factor is on probation. Below 0 for 36 months, the factor is dead and the portfolio is rebuilt without it.
   - **Mitigation**: per-asset-class factor exclusion. If carry on bonds is dead, drop it from the bond signal — the equal-risk-contribution structure naturally upweights the remaining factors.

5. **Roll mechanics failure on physical-delivery commodities.** Beyond the negative-prices issue, the standard roll convention can fail when CFTC or exchange changes margin or position limits.
   - **Historical instance**: **CME crude margin hikes in 2014 / 2020** — temporarily made the marginal cost of holding CL too high to be efficient at the academic strategy size.
   - **Metric**: per-instrument margin-to-notional ratio > 25% (vs. typical 3-7%); pre-roll basis > 3σ of its 1-year history.
   - **Mitigation**: instrument-level kill switch, with the remaining 49 instruments continuing.

6. **Data outage on a single feed during rebalance**. IBKR has had multiple hours-long outages historically (most recently 2024).
   - **Metric**: any data feed lag > 5 minutes during the rebalance window.
   - **Mitigation**: rebalance window is 2 days, so a 24h outage on day 1 can be absorbed by completing on day 2; if both days fail, the rebalance is delayed by 1 week (acceptable — strategy is monthly-horizon).

---

## 11. Falsifiable verdict criteria

Within 90 days of live deployment:

### Promote to full live capital (after 90-day shadow + 60-day initial live at 25% size)
- Net annualized Sharpe (over the live + shadow window) > 1.0 with 95% bootstrap CI excluding 0.5.
- Max DD < 8% over the 90+60 = 150-day window.
- Realized slippage ≤ 3 bps/leg blended.
- Zero broker-truth reconciliation discrepancies > 0.5%.
- Each factor's average IC > 0.01 across all rebalances in the window.
- Pairwise factor return correlation < 0.50 across the window.

### Continue shadow / continue at 25% size
- Sharpe in [0.3, 1.0], no kill triggers, factor ICs at least non-negative.
- Pairwise factor correlation in [0.45, 0.55] — borderline.

### Kill immediately (any one)
- 30-day DD > 10%.
- Single-day loss > 2% of account (the daily cap fires).
- Broker-side fill discrepancy > 0.5% on any single rebalance (Lesson 4).
- **3 consecutive monthly rebalances with all three factors' ICs < 0.01** — the central falsification claim of the strategy. Three rebalances is the minimum sample but is *sufficient* to falsify because: (a) the IC predicted by AMP 2013 is ~0.04 across all factors; (b) three consecutive months at IC < 0.01 across all three factors implies a joint probability of < 0.5% under the AMP 2013 model; (c) this regime would indicate either factor crowding has saturated, factor definitions are obsolete, or the universe-construction is broken. Any of these three should kill the strategy promptly.
- Pairwise factor return correlation > 0.60 over any 2 consecutive months.
- Single instrument's continuous-series roll basis deviation > 5σ vs. 3-year history (signals a roll-mechanics break that, if uncaught, fakes P&L).

---

## 12. One-paragraph operational risk note

The dominant operational risk in this iteration is **data quality at scale across 50+ instruments crossing 8+ exchanges and 6+ time zones**. A subtle bug in any one continuous-series back-adjustment (e.g., using ratio rolls everywhere except CL during the April 2020 negative-prices episode, or mis-detecting an exchange holiday on Hang Seng and computing a spurious volatility for HSI during Lunar New Year) will silently corrupt one signal among 162 (54 instruments × 3 factors) and the diversification structure may mask the error for months. The mitigation is a daily data-quality dashboard that flags: (a) any instrument's daily return outside [-15%, +15%] for manual review, (b) any continuous-series jump > 5% on a non-roll day, (c) any factor IC computed on incomplete data (< 80% of expected sample), (d) any open-interest roll trigger that doesn't fire within 5 days of the next contract becoming the front, (e) any feed lag > 5 minutes during the rebalance window. The second-largest operational risk is **single-broker dependency on IBKR**: this strategy is structurally tied to IBKR because no other retail-accessible broker provides clean access to all 8 exchanges in a single account with portfolio margin. An IBKR outage during the 2-day rebalance window is recoverable by extending to a 3-day window or skipping the month. An IBKR PM-margin policy change (they have tightened terms three times in the past five years) could halve our buying power overnight and force a deleveraging at unfavorable prices. The third operational risk is **calendar bugs around contract rolls and exchange holidays** — the third-Wednesday rebalance can fall on a US, European, or Asian holiday, creating an asymmetric execution window where one region's instruments execute and another's doesn't, leaving cross-asset hedges temporarily unbalanced. Every rebalance plan is pre-flighted against a global exchange-holiday calendar with manual override.

---

## Lessons compliance

1. **Validate before deploy**: walk-forward 1990-2012 IS / 2013-2026 OOS with ≥156 OOS monthly rebalances; minimum 60 required pre-deploy; then 90-day shadow + 60-day quarter-size live before scaling. Compliant.
2. **Binary gates, not confidence rankers**: every gate (vol target, gross-leverage cap, sector cap, correlation circuit, factor-IC threshold) is a hard region rule. No ML confidence is used for sizing. The signal `F_i` is a deterministic z-score combination, not a confidence output. Compliant.
3. **TP:SL asymmetry is structural**: at monthly rebalance horizon, the analog is "don't deleverage on a 1σ pullback in any single factor" — only the joint-portfolio drawdown triggers and the correlation circuit fire. Single-factor performance noise does not adjust position sizing within a month; only the next monthly rebalance reflects updated signals. Compliant by structure.
4. **Broker truth, not local cache**: `IB.positions()` reconciliation after every fill; portfolio NAV reconciled hourly during the rebalance window. Lesson 4 binding. Compliant.
5. **No PDT-exposed lanes on sub-$25k US equity accounts**: N/A — this is a futures+FX strategy. PDT applies only to US securities margin accounts under $25k. Compliant by structural exemption.
6. **One strategy per slot**: **CRITICAL — iter_23 must not be deployed alongside iter_4 (subset overlap on G10 FX carry) or iter_16 (which references iter_4).** iter_23 contains iter_4's edge as a sub-component. Deployment of iter_23 supersedes iter_4 in any portfolio slot. Compatible with iter_7 only after pairwise correlation < 0.40 is verified (likely passes because iter_23's FDAX exposure is <2% of notional). Compliant with explicit deployment-conflict tracking.
7. **No same-data backtest**: walk-forward with expanding training window, parameters pre-registered from AMP 2013 / KMPV 2018 / Pedersen 2015 (no in-sample tuning). 1-month embargo at the IS/OOS boundary. Compliant.
8. **Battle-tested broker SDK only**: `ib_insync` for all instruments. Compliant.
9. **Capacity bound named upfront**: soft ceiling $300M AUM, hard ceiling $1B AUM, industry-scale crowding limit $50-100B (not our binding constraint). Compliant.
10. **Kill criteria written before deploy**: §11 lists exact triggers, all triggerable within 90 days. The 3-consecutive-month all-factor-IC < 0.01 kill criterion is the strategy-defining falsification. Compliant.

---

THESIS: A global multi-factor risk-premia portfolio (carry + 12-1 momentum + 5-year value across 50+ futures/FX instruments, equal-risk-contributed, vol-targeted at 10% annualized) earns OOS Sharpe 1.0-1.5 net of costs because the three factors remain pairwise <0.30 correlated and individually each delivers IC > 0.04 across asset classes.
VALIDATION HORIZON: 156+ OOS monthly rebalances (2013-2026 walk-forward).
CAPACITY CEILING: $300M soft / $1B hard AUM at the strategy level; $50-100B is the industry-wide alpha-decay limit for the asset class.
