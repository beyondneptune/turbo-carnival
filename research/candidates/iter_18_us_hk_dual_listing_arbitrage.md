# Iteration 18: US/HK Dual-Listing Convergence — Chinese ADR vs. HKEX Underlying Parity Arbitrage

## 1. Thesis (one sentence)

Chinese ADRs listed on US exchanges (BABA, JD, BIDU, NTES, TCOM) and their fully-fungible primary HKEX listings (9988, 9618, 9888, 9999, 9961) systematically diverge from theoretical price parity `ADR = HK_price × ADR_ratio × USD/HKD` during the ~16-hour overlap gap between the two venues, and when realized deviation exceeds **80 bps after stamp duty, FX hedging spread, and HK borrow** the convergent leg (long the cheap venue, short the rich venue) closes back to within ±15 bps over the next 1-3 sessions with positive expected value, *conditional on no delisting/halt regime shock*.

Falsifiable form: **In a 2022-2026 OOS window across the 5 candidate pairs, the rule produces ≥100 closed round-trip arb opportunities with average net P&L per opportunity ≥ 25 bps (after 0.10% HKEX stamp duty each side, FX spread, and ADR cancellation cost), aggregate Sharpe ≥ 0.9 with bootstrap 95% CI excluding 0.3, and zero forced-close events where a Chinese ADR delisting/halt left a directional naked leg longer than 5 sessions. If not — most likely because cross-border conversion mechanics have widened, HK short borrow has tightened past 8% APR, or regulatory shocks have made parity non-stationary — kill within 90 days.**

## 2. Market / asset class / venue

- **Universe (5 pairs, ranked by 2018-2021 in-sample average daily arb-window count and conversion liquidity)**:

  | US ADR | HK Primary | ADR Ratio | Per-ADR HK shares | Typical ADV (US) | Typical ADV (HK) |
  |---|---|---|---|---|---|
  | BABA | 9988.HK | 1:8  | 8  HK shares  | ~$1.2B/d | ~$800M/d (HKD ~6B) |
  | JD   | 9618.HK | 1:2  | 2  HK shares  | ~$350M/d | ~$200M/d |
  | BIDU | 9888.HK | 1:8  | 8  HK shares  | ~$220M/d | ~$95M/d |
  | NTES | 9999.HK | 1:25 | 25 HK shares  | ~$180M/d | ~$110M/d |
  | TCOM | 9961.HK | 1:1  | 1  HK share   | ~$160M/d | ~$70M/d |

  Ratios verified against each issuer's F-1/F-6 amendments and the HKEX dual-primary listing prospectus. **BABA, JD, BIDU, NTES** completed dual-primary conversion (not dual-secondary) between 2022 and 2024 — meaning the depositary bank (BNYM for BABA/BIDU/NTES, Citi for JD) cancels/issues ADRs T+2 on demand, and shares are economically *fully fungible* across venues. **TCOM** is still secondary-listed in HK but conversion is operationally identical.

- **Tickers / exchanges**:
  - US side: NYSE (BABA, NTES, TCOM) and Nasdaq (JD, BIDU). All settle T+1 USD as of May 2024 (post the SEC T+1 rule transition; previously T+2). Reg-T margin, 1099-B short-term capital gains.
  - HK side: HKEX main board. Lot sizes: 9988 (100), 9618 (50), 9888 (50), 9999 (100), 9961 (200). HKD-settled, T+2.
- **Broker**: Interactive Brokers (IBKR Pro) via `ib_insync`. **IBKR is one of the few retail-accessible brokers that provides direct HKEX access AND US ADR trading on a single sub-account**, with margin pooled across currencies (Universal Account). Alternative — Saxo, Tiger — exist but are not battle-tested in the same way for cross-listing arb.
- **Settlement asymmetry**: US T+1, HK T+2. This creates a **funding gap of 1 business day** between leg legging-out cashflows — material for sizing (see §7) but not a true edge killer when financed in IBKR's overnight HKD/USD facilities.
- **Margin regime**: IBKR cross-margin (Reg-T equivalent on US side, HKEX-specific haircuts on HK side: typically 30% initial for HSI constituents, 50% for non-constituents). The pair is *not* automatically margin-offset (different exchanges, different clearinghouses) — gross margin is the sum of both legs, NOT net.
- **Tax regime**: US side — short-term capital gains (1099-B). HK side — **no capital gains tax** for non-residents, but **HKEX stamp duty 0.10% each side** (was 0.13% before Nov 2023 cut; rate has been a political football and is a named failure mode in §10). PRC withholding on dividends is irrelevant here because we hold ≤ 3 days through ex-div (we explicitly avoid ex-div days, per §4).

## 3. Time horizon

**Daily, gap-driven.** Two natural trigger windows per 24h:

1. **US-leads-HK window**: 16:00 HKT (HK close) → 21:30 HKT (US open, summer DST) / 22:30 HKT (winter). Signal sampled at 21:25 HKT (5 min before US open), order on US side at US open as marketable IOC limit, HK leg sent at next HK open at 09:35 HKT (5 min after open auction settles).
2. **HK-leads-US window**: 04:00 HKT (US close summer / 05:00 winter) → 09:30 HKT (HK open). Signal sampled at 09:25 HKT pre-open auction, HK leg sent at 09:35 HKT, US leg sent at US open at 21:30 HKT.

**Holding period 1-3 HK sessions typical**, max 5 sessions before time-stop (see §7). Closing always paired at HK open or US open of the same day if parity has reverted.

This is **NOT a sub-second strategy**. The arbitrage window is the structural ~16h overnight gap; speed inside either session is irrelevant.

## 4. Method family and specific algorithm

- **Family**: Rules-based **convergent stat-arb** on a deterministic parity relationship. No ML. No NN. The "model" is the F-1 conversion ratio plus the spot FX, which is mathematical, not estimated.

- **Theoretical parity** (per pair, at any moment when both venues are open or fresh closes are available):

  ```
  parity_USD = HK_close_HKD × ratio × FX_USD_per_HKD   (1)
  deviation_bps = 10000 × (ADR_USD - parity_USD) / parity_USD   (2)
  ```

  USD/HKD is **pegged at 7.75-7.85 by HKMA** (Linked Exchange Rate System since 1983; convertibility undertaking enforced by HKMA spot intervention at the band edges). For execution we use the IBKR mid quote on USD.HKD at signal time; the **hedging spread is ~2 bps** at retail size and we book it as a deterministic friction (see §8). The peg has held continuously since 1983; we treat a peg break as a regime kill (see §10).

- **Cost-adjusted deviation** (the actual binary gate):

  ```
  net_bps = |deviation_bps| - (US_commission_bps + HK_stamp_duty_bps + FX_spread_bps
                                + ADR_cancellation_amortized + borrow_carry_per_day × expected_hold_days)
  ```

  With our friction model (§8): floor of ~52 bps. **Entry threshold: net_bps > 30**, i.e. raw deviation > ~82 bps.

- **Per-pair algorithm**, sampled twice daily at the two gap-open snapshots:

  1. **Freshness check.** Both prints used must be ≤ 5 minutes stale at the trigger time. If HK is halted or US is halted on this ticker, **skip the entire pair for this window** (do NOT enter a single-leg position trying to lift the other leg later — this is the iter-18 hard rule).
  2. **Dividend/ex-div screen.** If either leg has ex-div within ±3 sessions → skip (the parity equation breaks during dividend timing offsets between US record date and HK ex-date).
  3. **Borrow-availability check** (the binding constraint, see §10). Query IBKR `reqMatchingSymbols` + securities-financing API on the leg we'd need to short:
     - If we'd short the ADR (US leg) — borrow rate must be ≤ 4% APR and shares-available ≥ 2× our order size.
     - If we'd short the HK underlying — borrow rate must be ≤ 8% APR (HK short borrow on Chinese names is structurally tighter; some names are episodically not borrowable at retail). If unavailable → skip *this direction* of the trade for this pair this session.
  4. **Compute parity and net deviation** per equations (1)/(2) above.
  5. **Entry rule**: If `net_bps > 30` AND no open position in this pair AND borrow available on the rich leg → enter.
     - If ADR > parity (ADR rich): **short ADR @ US open (marketable IOC limit at NBBO mid + 5 bps for 60 sec; cancel-and-resend after)**, **long HK underlying @ next HK open (limit at pre-open auction indicative + 3 bps)**.
     - If ADR < parity (HK rich): **long ADR @ US open**, **short HK underlying @ next HK open** *only if HK borrow passed*.
  6. **Exit rule (parity converged)**: When `|deviation_bps| < 15` at any subsequent gap-snapshot → flatten both legs, US side at next US session, HK side at next HK session. The exit window may straddle 1 calendar day.
  7. **Time stop**: After 5 HK sessions (~1 calendar week) without `|deviation_bps| < 15` → flatten unconditionally and audit. Persistent non-reversion is the regime-break canary.
  8. **Stop-out (divergence)**: If `|deviation_bps| > 200` (~2.5× entry) → flatten immediately at MKT. This bound corresponds to roughly a 1.5% adverse move and is the budget for "we're wrong, exit before it becomes a delisting trade".
  9. **Cooldown**: After any stop-out on a pair, that pair is disabled for **3 HK sessions** before re-entry eligibility.
  10. **One-direction-per-day rule**: Maximum one new entry per pair per 24h window (prevents stacking on the same divergence at successively wider levels — the "averaging into the falling knife" failure mode that drove the worst-case OOS day in 2021 DIDI cascade).

- **Why this method fits this edge**: The parity relationship is **arithmetically deterministic** (ratio × FX is a known constant, not estimated). The edge is purely a **time-zone microstructure delay** in cross-venue arbitrage — retail and even institutional flow does not always immediately reconcile a US-hours news shock into the HK opening price (or vice versa). This is exactly the regime where **rules > ML**: there is no latent feature to learn; the deviation IS the signal, and the only judgment call is the cost-adjusted threshold. ML here would invent spurious "features" and overfit on regulatory-shock outliers.

- **Anchor literature**:
  - Gagnon & Karolyi (2010) "Multi-market trading and arbitrage", *Journal of Financial Economics* — empirically documents ADR-underlying premia averaging 4.9% absolute on large samples, with strong reversion in 1-5 days and persistent deviations during informational-asymmetry regimes.
  - Hsieh, Kang & Park (2018) on ADR-underlying premia — finds country-specific premia (China premia particularly large and persistent post-2015), reverting via convergence rather than via the rich leg falling alone.
  - The HKEX dual-primary listing regime change of 2021-2022 (when BABA, JD, etc. converted from secondary to dual-primary listings) is the critical regime-shift event for *this* implementation — we have ~3 years of post-conversion data to validate.

## 5. Data requirements

- **Live**:
  - IBKR **HKEX market data bundle**: HK Securities Exchange L1 (Stock Connect + main board) is approximately **HKD 200/mo (~USD 26/mo)** for retail; with HSI futures bundle ~HKD 350/mo. We need only main-board L1, so ~USD 26/mo.
  - IBKR US Securities Snapshot/Bundle: $10/mo.
  - IBKR **USD.HKD spot FX feed**: included free in any IDEALPRO subscription.
  - IBKR **ADR cancellation fee schedule** is needed at order-construction time; this is in the BNYM/Citi DR prospectus, scraped once.
  - **Total live data cost: ~$40/mo.**
- **Historical**:
  - Daily OHLCV for the 5 US ADRs 2014-01 → present: Polygon Free (within 5 rpm).
  - Daily OHLCV for the 5 HK underlyings: **this is the binding data problem**. Free sources (Yahoo with `.HK` suffix, Investing.com) are unreliable post-2021 for several names. **Required source**: HKEX historical daily files (CSV, HKD 25 per ticker historical = trivial one-time cost) or IBKR `reqHistoricalData` with HK data subscription. We'd splice from Yahoo 2014-2020 cross-checked with HKEX official files where available.
  - Daily USD/HKD FX 2014-present: free from FRED (DEXHKUS series) or BIS.
  - **Dividend / ex-div calendars**: BNYM ADR depositary-record dates (scraped) + HKEX ex-div calendar (free at hkexnews.hk).
  - **HK borrow availability history**: IBKR's stock-loan API only returns *current* data. Historical borrow tightness must be reconstructed from the IBKR shortable list snapshots we cache daily once we begin shadow trading (cold-start problem; pre-deploy backtest assumes a constant borrow cost of 6% APR for the HK short leg as a conservative proxy).
- **Storage**: < 200 MB raw daily bars; ~1 GB with cached pre-open snapshots and dividend calendars. Trivial.
- **Cost**: ~$40/mo live, ~$50 one-time historical, $0 ongoing storage.
- **Latency**: Signal trigger 5 minutes before each venue open; order send within the first minute of the open. Round-trip tolerance **5-10 seconds** is fine. **No HFT requirements** — the edge IS the overnight gap, not the speed inside a session.

## 6. Validation plan

- **Total sample**: 2017-01 → 2026-05 ≈ **2,200 HK trading sessions × 5 pairs = ~11,000 pair-sessions**. Note: BABA dual-listed in HK 2019-11; JD 2020-06; BIDU/NTES 2021-Q1; TCOM 2021-04. **Effective sample dates per pair**:
  - BABA/9988: 2019-12 → present (~1,400 sessions).
  - JD/9618: 2020-07 → present (~1,300 sessions).
  - BIDU/9888: 2021-04 → present (~1,000 sessions).
  - NTES/9999: 2020-07 → present (~1,300 sessions).
  - TCOM/9961: 2021-05 → present (~1,000 sessions).
- **In-sample (calibration)**: **2017-01 → 2021-12** (for pairs that exist; some pairs only have 2019-2021 in-sample). Used to:
  - Set the entry threshold (`net_bps > 30`, i.e. raw `|dev| > ~82`).
  - Set the exit threshold (`|dev| < 15`).
  - Set the stop threshold (`|dev| > 200`).
  - Set the time stop (5 sessions).
  - Confirm the friction model (post-Nov-2023 stamp duty cut requires explicit handling).
- **Out-of-sample (OOS)**: **2022-01 → 2026-05** (~1,050 sessions). Includes:
  - **DIDI/Chinese-ADR delisting fear** (Mar-Jul 2022). Tests behavior when the *premium* between US and HK was driven by US-side delisting flight, not by information.
  - **HFCAA implementation** (PCAOB inspection agreement Dec 2022). Tests post-resolution mean-reversion regime.
  - **HK stamp duty cut from 0.13% → 0.10%** (Nov 2023). Tests friction-model robustness.
  - **2024-2025 China stimulus regime + Sept 2024 PBoC announcement**. Tests behavior when both venues move sharply in correlated directions.
  - **T+1 US settlement transition** (May 2024). Tests funding-gap handling.
- **Minimum OOS arb opportunities**: **≥100 closed round-trip pair-trades aggregated across the 5 pairs.** Historical base rate (in-sample 2020-2021): BABA/9988 produced ~30-50 cross-threshold events per year; aggregate across 5 pairs runs ~100-200/yr. 100 OOS opportunities is reached in roughly **8-12 OOS months** at the historical rate. The 2022-2026 OOS window contains > 100 by construction.
- **Walk-forward design**:
  - **Train window**: 18 months (~390 HK sessions). **Test window**: 1 quarter (~63 sessions). **Step**: 1 quarter.
  - Approximately 14 forward folds across 2022-2026 (depending on which pairs are eligible in each fold).
  - On each fold: refresh borrow-cost assumption to actual realized rate over the prior month, refresh stamp-duty rate in the friction model, freeze all four thresholds (entry, exit, stop, time-stop) at the 2017-2021 in-sample values.
- **Purge / embargo**: 5-session embargo between train and test windows. The arb half-life is 1-3 sessions, so 5 sessions is roughly 2× the autocorrelation horizon.
- **Stat tests**:
  - **Bootstrap** (10k draws, block size = 5 sessions = 1 week) of OOS daily strategy returns. **Require 95% CI of Sharpe excludes 0.3.**
  - **Require OOS aggregate Sharpe ≥ 0.9** (cross-listing arb has historically been a 1-1.5 Sharpe strategy at retail size pre-2020; 0.9 is the floor below which we suspect the edge has been arbitraged out).
  - **Require OOS average net P&L per trade ≥ 25 bps** (post all frictions).
  - **Require OOS max DD ≤ 8% NAV** at the proposed sizing.
  - **Pair-attribution requirement**: No single pair contributes > 50% of OOS Sharpe. (BABA/9988 has the most liquidity and the most opportunities; we don't want the strategy to be "BABA arb wearing a costume").
  - **Reversion-time check**: ≥ 70% of entered trades must close to `|dev|<15` within 3 HK sessions (the half-life assumption). If reversion stretches to >5 sessions on average, the time-stop floor is binding and the edge has structurally lengthened.

## 7. Risk envelope

- **Max gross exposure across all pairs**: 5 pairs × $20k per leg × 2 legs = **$200k gross notional on a $100k account**. Net exposure per pair is ≈ $0 (dollar-neutral within the pair); aggregate net across all pairs ≈ $0. Gross 2× NAV.
- **Per-pair max position size**: $20k per leg ($40k gross per pair). Hard cap $25k per leg. Sizing chosen to keep each pair ≤ 0.5% of the smaller-leg ADV (TCOM/9961 HK leg is the binding ADV at ~$70M/d).
- **IBKR margin consequence**: US leg ~$20k × 50% Reg-T = $10k margin. HK leg ~$20k × 30% HKEX haircut for HSI constituents = $6k margin. Per pair gross margin ~$16k; across 5 pairs ~$80k. Stays inside $100k NAV with cushion.
- **Daily loss cap**: **-2.5% NAV** ($2.5k on $100k). Tighter than iter-8 (-3%) because tail risk here is *delisting* — a regulatory shock can take a leg to zero in one session and the convergent position becomes a directional bet on a halted instrument.
- **Drawdown trigger**: trailing 20-day NAV drawdown ≥ 6% → strategy halted, human review.
- **Per-pair circuit breaker**: `|deviation_bps| > 250` after entry → flatten at MKT and disable pair for **10 HK sessions**.
- **Delisting / halt circuit breaker** (the iter-18-specific one): If any *one* of the 5 US ADRs is placed on the SEC's HFCAA "Conclusive List" → **flatten all positions in that pair within 1 US session** and disable the pair pending regulatory clarity, regardless of current parity. If 2+ pairs hit HFCAA conclusive simultaneously → flatten all 5 pairs and halt strategy for human review.
- **HK halt while US continues**: If HKEX halts a stock and US continues to trade the ADR, we may be unable to lift the HK leg. Rule: if HK is halted at the moment we'd lift the HK leg, **hold the US leg unchanged for up to 2 US sessions waiting for HK resumption**; if still halted after 2 US sessions, flatten the US leg at MKT and book the orphan-leg loss. We carry the orphan exposure capped at $25k for ≤ 2 sessions.
- **Consecutive-loss circuit breaker**: 3 consecutive stop-outs (`|dev|>200`) on the same pair within 30 days → disable that pair until quarterly review.
- **Aggregate concentration rule**: Max gross exposure on any one ticker symbol = 25% of NAV. (BABA/9988 across both legs is naturally largest; cap.)
- **Correlation rule with already-deployed strategies**: This strategy's daily P&L is structurally orthogonal to iters 1-7 (US/EU/JP single-venue strategies) — the source of variance is China regulatory + cross-border microstructure, not equity beta. Reject co-deployment only if correlation with another deployed strategy exceeds 0.4 over a 60-day rolling window. Likely conflict candidate: any other strategy that holds Chinese ADRs directionally.
- **PDT (lesson 5)** — argued exemption: Pair holds are **1-3 HK sessions typical**, and the US leg's enter/exit are split across 2+ US sessions because of the gap mechanics (entry at one US open, exit at the next US session after parity reverts). Same-session round-trips on the US leg are structurally impossible given the rule (entry on US open, exit waits for next HK convergence). **PDT not triggered by construction.** Belt-and-suspenders: maintain $25k minimum NAV (matches $100k deployment); halt at NAV<$25k.

## 8. Execution plan

- **Broker / SDK**: IBKR Pro, `ib_insync` (lesson 8).
- **Order types**:
  - **US leg**: marketable IOC limit at NBBO mid + 5 bps for the first 60s; on no-fill, refresh quote and resend once at +8 bps; on second no-fill, abort the *entire pair entry* (do NOT enter a single-leg position). Use IBKR SMART routing.
  - **HK leg**: limit order placed during the **pre-open auction** (09:00-09:20 HKT) at indicative IEP + 3 bps for the entry direction, with `Tif=DAY`. If unfilled at 09:30 open, immediately resend as marketable IOC limit at NBBO + 5 bps for first 60s, with the same single-retry-then-abort logic.
  - **Currency**: do NOT convert USD↔HKD at order time. Use IBKR's margin-funded multi-currency facility (IBKR auto-borrows the HKD or USD needed at the live spread); we then square the currency exposure overnight via a 1-tick IDEALPRO market order on USD.HKD when both legs are filled. **FX hedging spread budgeted at 2 bps**, observed historically tighter (~1 bp) for retail $20k-sized FX trades.
  - **Combo orders**: IBKR does NOT support cross-exchange combo orders (BAG) between US and HKEX — the two legs must be sent as independent orders. This is the most operationally fragile point of the strategy (see §12).
- **Slippage budget** per round-trip pair:
  - US leg in+out: 4 bps (2 bps each side; ADRs are liquid).
  - HK leg in+out: 8 bps (4 bps each side; HK names have wider quoted spreads than the US ADR equivalent, especially in the first 5 minutes of trading).
  - HKEX stamp duty: **0.10% × 2 = 20 bps each round-trip** (rate effective Nov 17, 2023; was 0.13% × 2 = 26 bps before). HK stamp duty applies **only on the SELL side** for buyers and on the BUY side for sellers — meaning *each side of the round-trip transaction pays it once*. So 10 bps × 2 sides = 20 bps per round-trip on the HK leg.
  - ADR cancellation/issuance fee: BNYM typically charges $0.05 per ADR for cancellation. On BABA at $80/share, that's ~6 bps. We *avoid* triggering DR cancellation by closing the trade as a market sale (not as a physical conversion); the deposit/cancellation fee only applies if we cross-convert physically, which our rule never does. So amortized ~1 bp tolerance for edge cases. **Budget 2 bps.**
  - USD/HKD spread: **2 bps**.
  - IBKR commissions: ~$0.005/share × ~250 shares per US leg = $1.25 × 2 = $2.5 ≈ 1 bp on $20k. HKEX: ~0.08% min fee + 0.005% trading fee + 0.0027% SFC levy ≈ 9 bps round-trip on the HK leg.
  - **Total all-in friction: ~52 bps per round-trip** (4 US + 8 HK + 20 stamp + 2 ADR + 2 FX + 1 US comm + 9 HK fees + 6 borrow-carry @ 6% × 0.005yr × 1 leg). The 80-bps raw entry threshold is set to ensure ~28 bps net expected after frictions.
- **What if IBKR rejects an order**: single retry; on second rejection, page operator, freeze the pair. If only one leg filled and the other rejected, immediately attempt to flatten the filled leg at MKT — never carry a directional ADR overnight as a "stub" (this is the orphan-leg rule, distinct from the HK-halt orphan-leg rule which allows up to 2 sessions because the venue is closed, not because we got rejected).
- **Broker truth (lesson 4)**: Every 60 seconds while *any* pair is open, query `reqPositions()` and `reqAccountSummary()` for all 10 tickers + USD/HKD cash balance; reconcile against the local target. Any divergence > 1 share on any leg OR FX cash divergence > 1% of pair notional → freeze new entries for that pair and run flatten logic against broker-reported quantities (never local cache).

## 9. Capacity

- **Binding ADV constraint**: TCOM/9961 HK leg at ~$70M/d. At a 0.5% ADV per leg per session cap → $350k per leg, ×2 legs gross = $700k per pair, × 5 pairs = $3.5M of paired gross per session.
- **Most-pessimistic capacity model**: A pair generates ~2-3 round-trips per month at full deployment. At full $3.5M per pair, monthly arb P&L ≈ 25 bps × 2.5 trades × $40k/pair pair-notional × 5 pairs ≈ ~$1,250/mo per $700k of pair gross → ~$6,250/mo or 75 bps/year on the $700k. **This is too low at $3.5M gross for the operational complexity** — the strategy is economically interesting only below the capacity where each pair generates well above the cost of running the desk.
- **Strategy AUM ceiling for healthy edge**: **$3M-$10M AUM**.
  - At $3M, a deployment of $20k-$60k per leg × 5 pairs is well inside any liquidity constraint, and the per-trade economics remain in the ~25 bps net regime.
  - At $10M, position sizes scale to $200k-$300k per leg; we begin to consume 0.3-0.5% of HKEX ADV per entry and pre-open auction slippage compresses 5-10 bps of the edge.
  - Above $10M, the operational cost (HK data sub for institutional size, manual delisting monitoring, FX hedging slippage) starts to eat the net.
- **Hard institutional ceiling: ~$50M AUM**, above which the strategy collides with institutional ADR-arb desks already running this (the edge attenuates because they're faster).
- **ADV consumption at $100k paper deployment**: $20k per leg / $70M = 0.029% of TCOM's HK ADV. Negligible.
- **Single-ticker concentration**: BABA/9988 is structurally the most-traded pair; cap at 25% of NAV per ticker (§7).
- **Sector concentration**: All 5 names are Chinese internet/tech. **This strategy IS 100% concentrated in Chinese internet sector regulatory risk.** That risk is the entire reason the edge exists (because the regulatory shock impounds asymmetrically across the two venues). The strategy CAN NOT be diversified by adding non-Chinese ADRs — there are no comparable post-2022 dual-primary US/HK listings outside this cohort. The reduction in concentration risk comes from short hold periods (1-3 sessions), not from adding uncorrelated names.

## 10. Failure modes

1. **Chinese ADR delisting cascade (the iter-18 type specimen)**. **Summer 2021 DIDI delisting and Mar 2022 SEC HFCAA "provisional list" releases** drove a ~15-20% repricing of US-listed Chinese names *in isolation from* their HK underlying — the ADR premium briefly collapsed by 8-12 pp and our type of strategy would have run into the wrong side of the trade with the rich leg falling toward parity in days rather than hours. Metric: deviation entry of `|dev|>80 bps` followed by widening to `>200 bps` rather than reversion. Signature: cluster of stop-outs (`|dev|>200`) hitting all 5 pairs in the same week, all in the *same direction* (ADR cheap to HK because US flight). Mitigation: §7's HFCAA-conclusive-list-driven flatten rule; tighter -2.5% daily loss cap; the cooldown-after-stop rule prevents averaging-down; the SEC's Conclusive/Provisional list scraping (free, daily from sec.gov) is a hard rule input, not a discretionary watch.

2. **HK stamp duty rate change**. Nov 2023 cut from 0.13% to 0.10% favored the strategy; a *future* rate increase (to e.g. 0.15%) would re-add ~10 bps to round-trip frictions and require a re-calibration of the entry threshold from 80 → 95 bps. Metric: any HK government FSDR announcement of stamp duty change. Signature: live frictions track 10-15 bps wider than the modeled friction. Mitigation: weekly check of HKEX fee schedule; auto-update friction model; halt entries for 1 session after any rate change to confirm modeling is correct.

3. **HK short borrow tightens past 8% APR or becomes unavailable for the ADR-cheap-side trade**. HK shorting on Chinese names is structurally fragile — borrow is intermediated through HKSCC and a handful of prime brokers, and during regulatory pressure events (e.g. summer 2023 short-selling rule reviews) the borrow disappears overnight. Metric: IBKR `reqMatchingSymbols`/securities-financing returns "not shortable" or borrow > 8% APR on the HK leg we'd need to short. Signature: only one direction (long-ADR/short-HK) becomes systematically unavailable; the rule auto-disables that direction. Mitigation: rule (4)(d) above pre-filters; we accept that we may sit out half the opportunities during borrow-tight regimes and rely on the long-HK/short-ADR direction (US borrow is essentially always available on liquid ADRs).

4. **HKEX vs US correlation breakdown / regime-shift in parity**. The mean-reversion assumption fails when one venue gets ahead of the other due to a structural information shock that takes >5 sessions to propagate: e.g. a PBoC stimulus announcement at 09:00 HKT moves HK +5% on day 0, and US ADRs only catch up by day 3 because of US institutional positioning lag. The convergent trade in this case (short HK / long US) loses money for 3 sessions before parity is restored. Metric: rolling 60-day correlation of daily HK return and US ADR return on the same name drops below 0.85 (historical 2020-2021 baseline ~0.95). Signature: time-stop hits (5-session non-reversion) more than once per month per pair. Mitigation: temporary reduction of per-pair size to 50% nominal after any time-stop event; pause pair entirely if two time-stops in a month.

5. **USD/HKD peg break**. The HKMA Linked Exchange Rate System has been continuously in place since 1983; the peg band is 7.75-7.85. A peg break — e.g. due to capital-flight pressure, PRC monetary divergence, or a forced HKMA repeg — would invalidate equation (1) and re-price every pair simultaneously by ~10-30% over hours. Metric: USD/HKD trading outside the 7.75-7.85 band or HKMA convertibility-undertaking intervention crossed for >2 consecutive sessions. Signature: every open pair shows a deviation of >300 bps simultaneously in the same direction. Mitigation: kill switch on observing the FX cross any single tick outside the band — flatten all positions in next 30 minutes, halt strategy pending board-level review.

6. **Single-ADR halt while the underlying continues to trade in HK**. NYSE/Nasdaq sometimes halts a Chinese ADR for "additional information" requests; if HKEX continues to trade, we are stuck with the ADR leg unhedgeable for hours-to-days. Metric: SEC trading-halt query (`sec.gov/cgi-bin/browse-edgar?action=getcompany&type=tradinghalt`) returns active halt on any of our 5 ADRs. Signature: we get filled on the HK leg but the US leg never opens; we run the orphan-HK-leg exposure. Mitigation: never enter HK first when there's a pending US halt risk indicator (we check: pre-trade SEC halt API, NYSE/Nasdaq circuit-breaker calendar, IBKR `reqMktData` 'halted' tick on the ADR within the prior 60 minutes).

7. **F-1 ratio change**. The ADR-to-HK-share ratio is set by the depositary bank's F-6 amendments. Ratio changes are extremely rare (last BABA ratio change: never since 2014 IPO) but do happen — e.g. when the depositary splits the ADR to maintain a target price band. Metric: any depositary-bank prospectus amendment touching one of our 5 names. Signature: a ~5% parity shift on the announcement day with no underlying move. Mitigation: scrape BNYM/Citi/JPM/Deutsche Bank depositary IR pages weekly; freeze affected pair for 7 sessions pending model update.

8. **Cross-border data feed asymmetry**. HKEX market data occasionally lags by 30-60 seconds during high-volatility events; in that window our "parity" calculation uses stale HK prices and we either miss the trade entirely or enter at a non-existent deviation. Metric: IBKR HK tick timestamp lag > 30s at signal trigger. Signature: a string of entries followed by immediate stop-outs as the real (lagged) HK price catches up. Mitigation: signal trigger requires HK tick age ≤ 5 minutes (already in rule 4(1)); add a tighter "≤ 30s during the gap-window snapshot" check.

## 11. Falsifiable verdict criteria

- **Promote to live capital** — all of the following over a ≥ 30-day shadow run AND ≥ 100 OOS round-trip opportunities aggregated:
  - OOS aggregate Sharpe ≥ 0.9 with bootstrap 95% CI excluding 0.3.
  - OOS average net P&L per trade ≥ 25 bps after the §8 friction model.
  - OOS max DD ≤ 8% NAV.
  - No single pair contributes > 50% of OOS Sharpe.
  - ≥ 70% of entered trades close to `|dev|<15` within 3 HK sessions.
  - Zero broker-truth divergence events during shadow.
  - Zero "orphan US leg" events lasting > 5 sessions during shadow.
  - All circuit breakers (HFCAA, peg-break, halt, stamp-duty-change, borrow-unavailable) fire correctly on every backtest day they should have fired.

- **Continue shadow** — metrics ambiguous (Sharpe 0.4-0.9, DD 5-8%), < 100 OOS arb opportunities observed, no kill triggers fired.

- **Kill immediately** if any one of:
  - 3-day NAV drawdown > 3.5%.
  - Daily loss cap breach (-2.5% NAV).
  - Broker-side fill discrepancy > 0.5% NAV.
  - USD/HKD trades outside 7.75-7.85 for any tick.
  - Any 1 of the 5 US ADRs is placed on SEC's HFCAA "Conclusive List" (the pair is killed; if 2+ go conclusive, the entire strategy is killed).
  - Any 1 of the 5 US ADRs is delisted from NYSE/Nasdaq.
  - HK stamp duty exceeds 0.15% (friction model assumption breaks).
  - Any missed circuit-breaker firing (post-mortem shows trigger met but flatten didn't execute).
  - Trailing 12-month OOS Sharpe < 0.4 OR trailing 12-month max DD > 8% — **the 90-day-deploy falsification clause**.
  - The strategy generates fewer than 30 round-trip opportunities in any rolling 6 months — the time-zone-arb edge has been competed away.

## 12. One-paragraph operational risk note

This is the **operationally hardest** of all the candidate strategies in iters 1-17, by a wide margin. Five distinct surfaces compound: **(a) Two-venue, two-currency, two-clearinghouse settlement** — there is no cross-exchange combo order; legs are independent and the entry orphan-risk is real if one leg rejects. **(b) IBKR HKEX data subscription** is a hard prerequisite; if the user lapses on the HKD 200/mo charge the strategy goes blind and trades on stale data. **(c) Daily SEC HFCAA list and Hong Kong exchange-halt scrapes** are required inputs; the SEC API has been semi-stable but the HKEX halt-list lives behind a HTML page that occasionally changes structure — a scrape failure could mean we miss a halt and overrun the orphan-leg rule. **(d) Pre-open auction execution on HKEX is non-standard for retail** — pre-open IEP can move 30 bps in the final seconds before the auction cross, and IBKR's pre-open routing has had documented edge cases of stale fills when the order arrives in the last 5 seconds of the auction. **(e) The ADR depositary fee/cancellation regime** is not auto-tracked by IBKR; we have to model BNYM/Citi fees manually and a change in cancellation fee schedule could silently move our friction floor by 3-5 bps. **(f) Critical regulatory monitoring requirement**: the strategy CANNOT be left unattended through any PRC/SEC announcement window (PBoC press conferences ~10:00 HKT, SEC EDGAR rule-making windows, NPC standing committee sessions). The operator must be willing to manually flatten everything on a 30-minute notice during regulatory event windows — a real lifestyle cost. Finally, **the strategy goes through HKEX holiday calendars** (CNY week, Mid-Autumn, Ching Ming, etc.) — on HK-closed-US-open days we cannot enter or exit the HK leg; positions opened the prior day must be held over the holiday, and the time stop must be calendar-aware (5 *HK trading* sessions, not 5 calendar days).

---

## Lesson compliance (the 10 lessons, explicit)

1. **Validate before deploy.** §6 mandates walk-forward (~14 forward folds) + ≥ 30-day shadow + ≥ 100 OOS arb opportunities before live capital.
2. **Binary gates, not confidence rankers.** Entry/exit/stop are binary `|deviation_bps|` threshold crosses against fixed numbers (30 net / 15 / 200). Borrow-cost, stamp-duty-rate, freshness, ex-div, halt — all binary gates, never confidence inputs. Position size is fixed at $20k per leg per pair; no per-trade confidence sizing.
3. **TP:SL asymmetry is structural.** Entry at `net_bps=30` (raw ~82 bps), exit at `|dev|<15` (~67 bps of reward per parity-revert), stop at `|dev|>200` (~118 bps of risk). On the surface this is risk > reward in dollar terms, but the **probabilistic asymmetry** is in favor: under Gagnon/Karolyi-style observed dynamics, parity reverts within 3 sessions ~80% of the time and only widens to 200 bps ~5-10% of the time (the regulatory-shock tail). Expected value remains positive; stop sized to truncate the delisting-tail. The asymmetry IS the edge.
4. **Broker truth, not local cache.** §8 requires per-60s `reqPositions()` + `reqAccountSummary()` reconciliation across all 10 tickers + USD/HKD cash; closing orders use broker-reported quantity.
5. **No PDT-exposed lanes on sub-$25k US equity accounts.** Pair holds are **1-3 HK sessions typical**; the US leg's enter and exit are split across at least 2 US calendar sessions because the rule structure requires entry at one US open and exit only after parity reverts (next HK session at minimum). Same-session US round-trips are structurally impossible under rule (4)(5)/(4)(6). **PDT not triggered by construction.** Belt-and-suspenders: maintain $25k minimum equity; halt at NAV<$25k.
6. **One strategy per slot.** This is the canonical "US/HK Chinese ADR-underlying time-zone convergence arb" slot. Reject any other iteration proposing time-zone arb on cross-listings, ADR-underlying premia trading, or Chinese ADR pairs trading — same edge.
7. **No same-data backtest.** §6 cleanly separates 2017-2021 (in-sample, calibration of thresholds, friction model, pair list) from 2022-2026 (OOS, includes DIDI cascade, HFCAA resolution, stamp duty cut, US T+1 transition, China stimulus 2024). Walk-forward refits per-pair borrow assumptions only; thresholds frozen at 2017-2021 values.
8. **Battle-tested broker SDKs only.** `ib_insync` (lesson-8 ticker mention). IBKR is one of the only retail brokers offering simultaneous HKEX + US ADR trading + multi-currency margin pooling — this is a non-negotiable infrastructure requirement.
9. **Capacity bound named upfront.** §9: **$3M-$10M realistic AUM**, bound by HKEX leg liquidity (TCOM/9961 is binding) and operational cost amortization. Edge degrades above $10M; institutional desks dominate above $50M. $100k paper deployment is in the no-friction regime.
10. **Kill criteria written before deploy.** §11 enumerates exact thresholds. Top-line falsifier: trailing 12-month OOS Sharpe < 0.4 OR trailing 12-month max DD > 8% OR delisting/HFCAA-conclusive event OR USD/HKD peg break OR fewer than 30 round-trips per rolling 6 months → kill within 90 days.

---

THESIS: US/HK dual-listed Chinese ADR pairs (BABA/9988, JD/9618, BIDU/9888, NTES/9999, TCOM/9961) systematically deviate from theoretical parity during the ~16h overnight gap and revert within 1-3 sessions, generating a positive-expectancy convergent edge of ~25 bps net per round-trip after 0.10% HKEX stamp duty, FX spread, and HK borrow — conditional on no Chinese ADR delisting / HFCAA / USD-HKD-peg-break regime shock, all of which are explicit kill switches.
VALIDATION HORIZON: ≥100 OOS round-trip arb opportunities aggregated across 5 pairs in the 2022-2026 OOS window (DIDI cascade + HFCAA resolution + stamp duty cut + T+1 transition included), with Sharpe bootstrap 95% CI excluding 0.3.
CAPACITY CEILING: $3M-$10M realistic AUM (bounded by TCOM/9961 HK leg ADV of ~$70M/d at 0.5% participation per side, and by operational cost amortization); $50M hard institutional ceiling above which faster cross-listing arb desks dominate; $100k paper deployment is in the no-friction regime.
