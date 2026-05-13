# iter_9_nifty_cross_section_momentum

## 1. Thesis (one sentence)

Cross-sectional momentum on Nifty 50 constituents — long the top-decile and short the bottom-decile of the 12-1 month return ranking, weekly-rebalanced and vol-scaled — captures a documented emerging-market momentum premium that is empirically stronger than its developed-market counterpart, and is falsifiable as: 200+ out-of-sample weekly rebalances over 2019-2026 do not produce a net annualized Sharpe > 0.70 after costs, market-impact, and INR/USD friction.

## 2. Market / asset class / venue

- **Asset class (primary)**: Nifty 50 cash equities — the 50 large-cap NSE-listed names that comprise the Nifty 50 index (e.g., RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK, ITC, KOTAKBANK, LT, SBIN, BAJFINANCE, etc.). Constituent list rebalances semi-annually (March/September); strategy ingests official NSE index-constituent file at each rebalance week.
- **Asset class (hedge leg)**: Nifty 50 index futures — NIFTY current-month and next-month contracts on NSE F&O — used to construct the short leg because retail short-selling of individual cash equities in India is operationally restricted (see §12).
- **Asset class (US-accessible fallback)**: iShares MSCI India ETF (INDA), traded on Cboe BZX / NYSE Arca, plus listed INDA options (puts) for downside hedge. This is the long-only variant for US accounts that cannot get Indian PIS (Portfolio Investment Scheme) / FPI access.
- **Primary venue/broker (direct India)**: NSE (cash & F&O) via Zerodha Kite Connect (`kiteconnect` Python SDK). Backup: ICICIdirect Breeze API or Upstox Uplink — both are battle-tested per Lesson 8. NSE is open 09:15-15:30 IST (03:45-10:00 UTC).
- **Primary venue/broker (US fallback)**: IBKR via `ib_insync` for INDA + INDA options on US tape. INDA trades 09:30-16:00 ET; volume concentrated at open and last hour.
- **Settlement**: NSE cash equities T+1 (moved from T+2 in 2023). NIFTY futures cash-settled at expiry, last Thursday of the month. INDA US T+1.
- **Margin**: NSE F&O SPAN+exposure margin (~12-14% for index futures, lot size 25 = ~₹575k notional per lot at NIFTY 23000). INDA on IBKR Reg-T: 25% maintenance for the ETF, full premium for long puts.
- **Tax regime (India direct)**: STCG 15% on equity held <12 months; LTCG 10% above ₹100k/yr; STT levied on each trade (0.1% delivery, 0.025% intraday sell-side, 0.01% on F&O sell). All four taxes are modeled into the cost stack. **For non-resident US investors**, structure as FPI Cat-II via a SEBI-registered custodian, OR — vastly simpler — use INDA on US tape, taxed as long-term/short-term capital gains in the US.
- **Tax regime (INDA fallback)**: standard US Schedule D; long puts ≤1y = short-term.

## 3. Time horizon

**Daily bars, weekly rebalance, ~3-5 trading day holding window per cohort.** Signal computed at NSE close Friday (15:30 IST = 10:00 UTC). Orders execute at NSE open next Monday (09:15-09:30 IST) using a VWAP-targeted child-order schedule. Held until the next Friday-close re-ranking, generating a complete portfolio turnover roughly every 5 trading days. This is multi-day swing — explicitly **not** intraday and well clear of any pattern-day-trader regime (see Lesson 5 handling in §7).

## 4. Method family and specific algorithm

**Rules-based cross-sectional ranking — 12-1 momentum (Jegadeesh & Titman 1993; Rouwenhorst 1998 emerging-market extension; Asness/Moskowitz/Pedersen 2013 "Value and Momentum Everywhere").** No ML in the core signal; ML is explicitly excluded because the 12-1 momentum premium is a 30-year well-published anomaly with a clean linear form, and adding gradient-boost or NN sizing invites overfitting per Lesson 2.

Algorithm:

1. **Universe at week T**: official Nifty 50 constituents as of Friday close. Names added by NSE in the most-recent index reconstitution are admitted only after 30 calendar days of post-inclusion price history (avoids the inclusion-drift artifact).
2. **Lookback window**: trailing **132 trading days minus 21 (last month skipped)** = months T-12 through T-2 inclusive. The 21-day skip is the standard short-reversal control from Jegadeesh-Titman.
3. **Score per name**: `mom_i = ln(P_i[T-21] / P_i[T-132])`. Compute for all 50 constituents using adjusted prices (split + dividend + bonus adjusted via NSE/Bhavcopy `dividend_adj_close`).
4. **Filters (binary, per Lesson 2 — NOT confidence-weighted)**:
   - 60-day average daily ₹-volume > ₹500 crore (~$60M ADV equivalent at INR 83/USD) to ensure exit liquidity on the short side proxy.
   - No name within 3 trading days of declared earnings (Q1/Q2/Q3/Q4 announcement calendar from NSE corporate-actions feed).
   - No name with ex-date for dividend / bonus / split / spin-off within the upcoming 7-day holding period.
   - No name in the F&O ban list (the NSE-published "securities in ban period" list, which blocks F&O open-interest expansion).
5. **Rank** the surviving names by `mom_i` descending.
6. **Long basket (top decile)**: top 5 names by `mom_i` (5 of 50 = 10%).
7. **Short basket (bottom decile)**: bottom 5 names by `mom_i`.
8. **Position weights — equal-weighted within basket, vol-scaled at portfolio level**:
   - Each of the 5 long names: 20% of long gross notional. Each of the 5 short names: 20% of short gross notional.
   - Portfolio target ex-ante vol: 12% annualized. Compute trailing 60-day realized vol of the historical long-short spread; scale leverage `L = 0.12 / σ_LS`, capped at 2.0x and floored at 0.3x. This is portfolio-level vol-scaling, not per-name — keeps the cross-sectional signal clean.
9. **Implementation of the SHORT leg (critical India-specific design)**: India retail does not have a continuous easy-borrow stock-loan market; SLB (Securities Lending & Borrowing) exists but is illiquid for many names. The clean substitute is to **short NIFTY index futures** equivalent to the dollar-notional of the short basket. This means the strategy is mechanically: `long $X of top-5 cash equities + short $X NIFTY futures notional`. The short leg becomes a **beta hedge plus the negative-momentum spread bet**: P&L is `(top-5 return − NIFTY return) × $X`, which is the long-leg alpha against the index. To capture the **bottom-5 specific underperformance**, we ALSO take a small additional NIFTY-futures short scaled to the expected `bottom-5 minus index` historical beta-residual (typically ~0.4x of the bottom-basket notional, calibrated from training data). This is functionally a "long top, short index, short additional index slice" rather than a pure long-short cross-section. We accept the modest signal dilution (estimated 15-25% of the academic premium is lost vs. true single-name shorting) as the price of operational tractability. Documented in §10.
10. **Hold until Friday close week T+1; re-rank.**

**Why rules, not ML**: 12-1 momentum is one of the most-replicated factors in finance, with separate published evidence in India (Sehgal & Balakrishnan 2002, ITM Indian Institute of Capital Markets working papers, and the Asness 2013 global panel). The signal is monotone and the cross-section is small (50 names) — there is no information density for ML to exploit beyond the linear rank.

## 5. Data requirements

- **NSE daily OHLCV**: ~30 years of Nifty 50 constituent daily bars (1995-present). Source: Zerodha Kite historical-data endpoint (free for subscribers, 60 days/request, rate-limited 3 req/s), or NSE Bhavcopy daily files (free, official, downloadable as CSV). ~5 MB/year × 30 = 150 MB.
- **Corporate-actions feed**: NSE `corporates/actions` endpoint for splits, bonuses, dividends, ex-dates — free. ~10 MB total.
- **F&O ban list**: NSE `liveAnalysis/marketLot/SecurityWiseDP` daily file — free.
- **Earnings calendar**: NSE corporate-announcements feed — free.
- **NIFTY futures historical**: continuous-contract roll-adjusted series from Kite historical-data — ~5 MB.
- **INDA fallback data**: Polygon S3 daily bars, free tier. ~1 MB.
- **INR/USD spot**: RBI reference rate daily — free. ~1 MB.
- **Storage estimate**: <500 MB total static history; ~50 MB/year live capture.
- **Cost estimate**: ₹200/month (~$2.50) Zerodha Kite Connect subscription, free NSE Bhavcopy, free RBI rates. INDA fallback uses Polygon free tier. Effective: ~$3/month direct, $0/month for fallback.
- **Latency requirement**: 10-second order ack tolerable. Weekly rebalance is decidedly not latency-sensitive. Kite REST has ~200-400 ms round-trip from US east; we accept that and use child-order pacing to hide impact.

## 6. Validation plan

- **Sample**: NSE Nifty 50 constituent daily, 2005-01-01 to 2026-04-30 (~21 years). 2005 chosen because Nifty 50 methodology stabilized at 50 names that year and free-float weighting was adopted. Earlier data exists but pre-2005 index composition is messy.
- **Walk-forward design**:
  - **In-sample (training/parameter sanity check only — no fitting; rules are predetermined)**: 2005-01-01 to 2018-12-31 (~720 weekly rebalances). Used only to confirm sign and magnitude of the 12-1 effect on this universe before committing to OOS.
  - **Out-of-sample (rolling, the verdict window)**: 2019-01-01 to 2026-04-30 (~380 weekly rebalances available; we require **minimum 200 OOS weekly rebalances** before any verdict — comfortably satisfied).
  - **Step**: expanding-window retest each January; published only the unconditional 2019-2026 result is the verdict, but expanding-window robustness is checked as a sanity gate.
- **Purge & embargo**: 5 trading days between train-end and OOS-start (positions hold ~5 days). 1 trading day embargo around any constituent change to prevent lookahead from announced inclusions/deletions.
- **Cost model**:
  - NSE delivery STT 0.1% sell-side + brokerage ₹20/order Kite flat + SEBI fees + GST → ~12-15 bps round-trip on cash equity legs.
  - NIFTY futures: STT 0.01% sell-side + brokerage flat → ~3-4 bps round-trip per cycle.
  - Market-impact: estimate 8 bps per side on cash equities at 1% of ADV consumption; verify ex-post with Almgren-Chriss schedule.
  - Total stack: ~25-30 bps round-trip per weekly cycle. At 52 cycles/year this is 13-15% annual cost drag — sobering but real.
- **Statistical thresholds (verdict)**:
  - Net-of-cost annualized Sharpe > 0.70 with 95% bootstrap CI excluding 0.30 — passes promote.
  - Max drawdown < 22% on OOS — passes promote.
  - Hit rate (positive weeks) > 53% — confirmatory.
  - Tail check: COVID-March-2020 week, Adani-collapse Jan-2023 week, election-result-day weeks (2019, 2024) each contribute < 6% loss to portfolio — confirms vol-scaling is biting.
- **Minimum OOS sample**: 200 weekly rebalances (~3.8 years); 2019-2022 alone delivers this. Full 2019-2026 ≈ 380 rebalances.

## 7. Risk envelope

- **Account assumed**: $100,000 USD equivalent funded. Direct India route requires FPI Cat-II custodian, so retail traders most likely run INDA fallback at $100k; institutional direct route assumes ≥$500k AUM to amortize custody fees.
- **Max position per single name (long basket)**: 20% of long gross notional = ~$10,000 at $100k account with full-size 1.0x leverage. Concentration is real; the 5-name long book is the design.
- **Max NIFTY-futures short notional**: 1.2x equity = $120,000 notional, which is ~10-12 NIFTY lots at NIFTY 23,000 × ₹50 multiplier × 25 lot-size ≈ ₹2.9 crore (~$35k margin needed, well inside SPAN).
- **Daily loss cap**: 1.5% of equity = $1,500/day. Hard halt for 24h if breached.
- **Drawdown trigger**: rolling 20-day DD > 7% triggers 50% deleverage; > 12% triggers full flat and a 2-week review before re-deploy.
- **Circuit breaker (strategy-level)**: 4 consecutive losing weeks triggers a strategy-level pause and review — not auto-kill, since momentum is known to have 3-4 month negative runs (Daniel-Moskowitz 2016 momentum-crash literature).
- **Correlation rule**: must check rolling 60-day correlation < 0.3 with any deployed US-equity, EU-equity, or developed-market trend strategy. Against iter_5 (VIX term-structure) the natural correlation is ~0.1-0.2 in normal times but spikes in panics — explicit cross-portfolio check at each rebalance.
- **Indian-market-specific circuit-breaker handling (CRITICAL)**: NSE applies index-level circuit breakers at **±10%, ±15%, ±20% intraday moves** (15-min, 45-min, full-day halts respectively) and per-stock daily price bands of **±5%, ±10%, ±20%** depending on the stock's F&O eligibility (Nifty 50 names are all F&O-eligible so they trade with the ±20% band, but pre-open call-auction can lock names at the band). On a circuit-hit day: (a) any open order in a locked stock will simply not fill, (b) we DO NOT chase by widening the band, (c) we mark the leg as "skipped this rebalance" and let the position drift to next Friday, (d) NIFTY-futures hedge is unaffected since futures have their own (looser) circuit. This is the explicit Lesson 10 handling.
- **PDT (Pattern Day Trader) rule — argument for non-applicability**: PDT is a FINRA rule on US-equity margin accounts that triggers when >4 day-trades execute in 5 business days from a <$25k account. **This strategy has a 5-trading-day holding period — every position spans the overnight session at minimum, so no trade is a "day trade" by FINRA definition**. Therefore PDT does not apply regardless of account size. The INDA fallback path also clears PDT trivially since holding INDA for 3-5 days makes each round-trip a swing trade. Lesson 5 honored: no PDT-exposed lanes.
- **Currency-risk handling (INR exposure)**: direct route holds INR-denominated assets, so the USD investor carries INR/USD FX risk on the full notional. Historical USD/INR vol is ~6-8% annualized, additive to portfolio vol unless hedged. **Hedge mechanism**: short USD/INR-NDF (non-deliverable forward) via IBKR Forex window in 1-month tenor sized to the long-leg net dollar exposure, rolled monthly. Alternative: accept the INR depreciation drag (historically ~3-4%/year structural INR weakening vs USD) as a known cost and target a 0.70 Sharpe NET of this drag. The validation plan models BOTH variants and reports both Sharpes. INDA fallback already wraps INR exposure inside the ETF, so the FX risk is implicit but unhedgeable at the security level; if the user wants pure-equity exposure with no FX, they must hedge USD/INR separately on IBKR.

## 8. Execution plan

- **Broker / SDK (direct)**: Zerodha Kite Connect via `kiteconnect` Python package. Order placement: `kite.place_order(variety="regular", exchange="NSE", tradingsymbol="...", transaction_type="BUY", quantity=N, order_type="LIMIT", product="CNC", price=p)`. For futures: `exchange="NFO"`, `product="NRML"`, tradingsymbol like `NIFTY26MAYFUT`.
- **Broker / SDK (US fallback)**: IBKR `ib_insync` for INDA and INDA options.
- **Order type**: marketable-limit IOC with 3 bps slippage tolerance vs. NSE pre-open call-auction discovered price. Long-leg orders split into 5 child slices, one per minute, over the first 5 minutes of regular trading (09:15-09:20 IST) to avoid the opening-auction noise but capture early liquidity. NIFTY-futures hedge placed in a single market-marketable IOC at 09:15:30 IST after the long basket is at least partially filled (to keep dollar-neutral within the rebalance window).
- **Order schedule per rebalance**:
  - T = Monday 09:14 IST: pre-stage 5 long-leg orders.
  - 09:15:00: release all 5 long orders as limit-at-mid with 3 bps slip cap.
  - 09:16-09:20: child slice re-pricing if unfilled (one re-price per minute).
  - 09:15:30: NIFTY-futures hedge placed as marketable IOC sized to filled long notional + bottom-basket-proxy slice.
  - 09:25: reconciliation against `kite.positions()` (Lesson 4). Any unfilled long-leg → log, mark for catch-up at 14:00 IST (afternoon liquidity peak).
- **Slippage budget**: 8 bps per side on cash equity legs; 2 bps on futures. Measured ex-post by comparing fill VWAP to mid at 09:14:59 IST. Rolling 12-week slippage tracked; if it exceeds 15 bps for 2 weeks running, downgrade execution to TWAP-30min child schedule.
- **Broker rejection handling**: retry with widened limit (5 bps), then with marketable IOC. If still rejected (price-band lock, F&O ban-period escalation, or NSE feed glitch), skip that name and let the position roll to next Friday's re-rank with a logged "partial cycle" tag. Per Lesson 4, reconcile against broker truth after every order — `kite.positions()` is the source of record, never the local cache.
- **Reconciliation cadence**: every 60 seconds during the 09:15-09:25 rebalance window; once per hour during the hold period (because India has been known to produce intraday corporate-action surprises that mid-strategy edit positions); EOD reconciliation forced at 15:30 IST and 16:00 ET (for INR/USD spot impact).

## 9. Capacity

- **Nifty 50 ADV (notional)**: top 5 names individually trade $300M-$800M USD-equivalent ADV; bottom names in Nifty 50 still trade $50-100M ADV. Cumulative Nifty 50 ADV ≈ $5-6B/day.
- **NIFTY futures ADV**: ~₹40,000 crore notional (~$4.8B) per day on current-month alone; rolls deepen further into expiry week. Deep and growing.
- **AUM ceiling**:
  - **Retail/direct (Kite)**: $250,000 effective AUM. Above this, the 5-name concentration starts taking >1% of ADV on the smaller Nifty 50 names (e.g., DIVISLAB-sized rather than RELIANCE-sized), and slippage exceeds 15 bps.
  - **Institutional/direct (FPI)**: $20-50M before the long-leg child-slice schedule needs to extend across the full day, and before NIFTY-futures hedge consumes >0.5% of futures ADV on hedge-resets.
  - **INDA fallback**: INDA ADV ~$200-300M/day; the long-only ETF variant scales cleanly to ~$10M AUM before consuming >1% of INDA ADV.
- **Edge degradation point**: the academic momentum premium itself is robust to AUM; execution friction is the binding constraint. Above $50M direct, expected Sharpe drops from theoretical ~1.0 toward ~0.5 due to slippage on the bottom-half of Nifty 50 names.
- **ADV consumption at full retail size**: at $250k direct, long leg = $250k × 1.0 = $250k gross, split into 5 names → $50k each. Vs. say BHARTIARTL at $300M ADV that is 0.017% of ADV. Negligible.
- **Concentration risk**:
  - Single-name max: 20% of long basket (already in risk envelope).
  - Single-sector max: implicit cap from the cross-section, but periodically the top-5 momentum names cluster in one sector (e.g., IT post-2020, financials in some windows). We accept up to 3 of 5 long-leg names in one sector but log a flag if 4 or 5 cluster — at that point the trade becomes a sector bet, not a cross-section bet. Manual review trigger, not auto-halt.

## 10. Failure modes

At least 3 named, with metrics and mitigations:

1. **Momentum crash (Daniel-Moskowitz 2016)**: After steep drawdowns, prior losers rally violently and prior winners lag. India: 2009-Q2, 2020-Apr/May, parts of 2023-H2. The long-top-5 / short-NIFTY-futures construction is especially exposed because the short hedge is broad-index rather than the specific losers — when the losers rally hardest, the hedge underweights that rally.
   - **Metric**: long-leg basket trailing 60-day return < -25% absolute, OR cross-sectional dispersion (top-quartile − bottom-quartile of 12-1 momentum) collapses below the 5th-percentile of its historical distribution.
   - **Mitigation**: portfolio-level vol-scaling already drops gross from 1.0x toward 0.3x when realized vol spikes. **Additional rule**: if 5-day return of the long-leg basket < -8%, halve leverage at next Friday close. If 10-day return < -12%, flat the strategy for 4 weeks before rebuild. This is the explicit Lesson 3 acknowledgment that drawdown management is part of the edge.
2. **Single-name event shock inside the long basket (corporate-governance, fraud, sudden regulatory action)**: examples — Adani Group January 2023, Yes Bank March 2020, IL&FS October 2018. A long-leg position can lock down 20% per day for 5+ consecutive sessions on the price-band.
   - **Metric**: any long-leg name hits the lower circuit (-20% locked) on a single day; OR analyst-rating change > 3 notches; OR SEBI/ED action announced.
   - **Mitigation**: hard rule — on any long-leg circuit-down or SEBI investigation announcement, flatten THAT name at the next available open (might be days later if the band stays locked), and re-balance the remaining basket weights to 25% each from 20% each. Do NOT replace the gone name mid-cycle — wait for next Friday's re-rank. Lesson 10 honored: specific, pre-written kill rule.
3. **NIFTY-futures hedge basis blow-out (cost-of-carry shocks, RBI rate-decision week, monsoon/election-week premium spikes)**: futures-spot basis can move 1-2% in a session around macro events.
   - **Metric**: NIFTY-futures fair-value-basis residual > 1.5% on a daily mark.
   - **Mitigation**: prefer next-month futures over current-month in the last 5 trading days before current-month expiry (cleaner roll, less expiry-week distortion); accept the residual basis noise as a known cost-of-carry friction (modeled into the cost stack at ~5 bps/week).
4. **NSE technology / connectivity outage (rare but real — there was a full NSE halt in February 2021)**: 4-hour intraday halt with markets reopening at a re-priced level.
   - **Metric**: Kite WebSocket disconnect > 5 minutes OR `kite.positions()` returns 5xx for > 60 seconds.
   - **Mitigation**: ALL pending orders are auto-cancelled by NSE on a session-halt anyway. On reconnect, reconcile via `kite.positions()` (Lesson 4), do NOT re-fire any orders that were in-flight at halt time, and if the halt persists past 13:00 IST, skip the rebalance for that week — the missed week is logged as a "no-trade" cycle and the previous portfolio is held a second week. Falsifiable: more than 2 missed-rebalance weeks per year contributes to the kill criteria below.
5. **INR depreciation crisis (BoP shock, oil-price spike, taper-tantrum-style flight)**: ETF holders see USD return crushed by FX, direct-INR holders see imported inflation hit equities AND the FX simultaneously.
   - **Metric**: USD/INR moves > 4% in a week, OR 1-month INR-NDF implied vol > 12% annualized.
   - **Mitigation**: if running the unhedged direct variant, scale gross to 0.5x at next rebalance until INR-NDF vol drops below 9%. Hedged variant carries on, but expect the NDF hedge cost to widen 20-40 bps/year during stress (modeled).

## 11. Falsifiable verdict criteria

Exact numbers — written before any deployment per Lesson 10.

- **Promote to live capital** (all must hold simultaneously after **30 shadow days** = 6 weekly rebalances of paper trading on broker-fill simulation):
  - Net-of-cost annualized Sharpe over the 30 shadow days > 0.50 (sample is small, threshold therefore softer than the OOS 0.70 — but trend must be positive).
  - Max DD over shadow window < 6%.
  - Zero broker-truth divergence events (every `kite.positions()` reconciliation matches local intent within 1 share, all 6 weeks).
  - All 6 rebalances completed with > 80% of intended notional filled within slippage budget.
  - The OOS backtest (2019-2026) has already produced ≥ 200 weekly rebalances with net Sharpe > 0.70 and max DD < 22%.
- **Continue shadow** (any of these mid-shadow):
  - Sharpe 0.20-0.50 with no kill triggers — extend shadow another 30 days, total 60 shadow rebalances.
  - One incomplete rebalance (fill-rate < 80%) but no fundamental breakage.
- **Kill immediately** (any one of):
  - Shadow Sharpe < -0.5 after ≥ 4 weekly rebalances.
  - Any 3-day DD > 5% of notional.
  - Any broker-truth divergence > 0.5% of notional (cash equity quantity drift) OR any wrong-side NIFTY-futures contract held vs intent.
  - 2 consecutive missed rebalance cycles for ops reasons (Kite outage, NSE halt, network).
  - Realized portfolio vol > 25% annualized (vol-scaling not working).
  - INR/USD weekly move > 6% with the unhedged variant unable to scale down.
- **90-day live falsification (after promote)**: live Sharpe over the first 90 calendar days (~12 weekly rebalances) must be > 0.30. If not, kill and rerun OOS check with the additional 90 days of data appended. This is the upfront 90-day-falsifiability requirement met explicitly.

## 12. One-paragraph operational risk note

The largest operational surface is the cross-border execution stack: Zerodha Kite is reliable for India residents but **US-based retail accounts generally cannot open a Zerodha Kite account directly** — the typical path is either (a) become an FPI through a SEBI-registered custodian like HDFC or Kotak (≥$250k setup, multi-week onboarding, ongoing custody fees of 5-15 bps/year), (b) open a Zerodha NRI account if the user has Indian PAN/citizenship status, or (c) take the INDA-ETF route on IBKR which is the realistic-for-most fallback but loses the cross-sectional alpha (INDA is just the long-only index). **Retail US accounts essentially cannot short individual Indian stocks** — the SLB market is illiquid and not accessible to retail, ADRs of Indian names (INFY, WIT, HDB) are short-restricted at most US brokers, and shorting INDA itself defeats the long-only premise. This is why the design hedges via short NIFTY index futures (accessible to FPI accounts) rather than single-name shorts. If the FPI route is closed, the only operationally viable variant is **long-only top-5 momentum stocks + protective INDA puts** as a degraded but executable form. Beyond the access problem: Kite imposes a 3-req/sec rate limit and a 200 daily orders/symbol cap that we sit well inside but must monitor; the kiteconnect library has had breaking changes at major version bumps (2.x to 3.x to 4.x); NSE Bhavcopy daily file occasionally is published with 30-60 minute delays on policy-rate-decision days; the INR-NDF market sits offshore (Singapore, Dubai, London) and is not accessible to all US retail brokers — IBKR offers it as a cash forward, but at retail size the spreads are wider than at institutional desks. Currency wires from US to India for funding the direct route take 3-5 business days and have $50-200 wire fees plus FX spread, making lumpy contributions/withdrawals costly. None of this kills the strategy — but it strongly suggests starting with the INDA + INDA-puts fallback at $50-100k, proving the directional momentum signal in shadow there, then deciding whether the FPI uplift to the full long-short variant clears the custody/wire/legal overhead.

---

### Lesson compliance summary

1. **Validate before deploy**: OOS 2019-2026 with ≥200 weekly rebalances + 30 shadow days REQUIRED before live capital — §6, §11.
2. **Binary gates, not confidence rankers**: every filter (liquidity, earnings-window, F&O-ban, circuit-band) is a hard binary cutoff — §4 step 4. No ML; momentum rank is a deterministic ordering.
3. **TP:SL asymmetry**: portfolio-level vol-scaling and per-name circuit-down kill rules constitute the explicit stop logic; momentum-crash deleverage rules at -8% and -12% basket return are pre-written — §10 mode 1.
4. **Broker truth, not local cache**: `kite.positions()` reconciled every 60 sec during rebalance window, hourly during hold, EOD always — §8.
5. **No PDT-exposed lanes on sub-$25k US accounts**: 5-day holding period means zero day-trades; PDT explicitly inapplicable — §7 PDT paragraph.
6. **One strategy per slot**: this is the only India / NSE / cross-sectional-momentum design in the candidate set (checked against iter_1-5: US-smallcap, ES, BTC, FX, VIX — no overlap on asset class, geography, or method).
7. **No same-data backtest**: in-sample 2005-2018 used ONLY for parameter sanity; OOS 2019-2026 is the verdict window, walk-forward expanding — §6.
8. **Battle-tested broker SDKs only**: `kiteconnect` (Zerodha official) for direct, `ib_insync` for INDA fallback — §2, §8.
9. **Capacity bound named upfront**: $250k retail / $20-50M institutional / $10M INDA-fallback ceilings stated with reasoning — §9.
10. **Kill criteria written before deploy**: complete numeric kill table — §11 plus per-mode triggers in §10.

THESIS: Cross-sectional 12-1 momentum on Nifty 50 captures India's well-documented EM momentum premium via weekly long-top-5 / short-NIFTY-futures rebalancing, falsifiable as 200+ OOS weekly rebalances 2019-2026 failing net Sharpe > 0.70.
VALIDATION HORIZON: 380 OOS weekly rebalances available; 200 minimum required.
CAPACITY CEILING: $250k direct-retail / $20-50M institutional FPI / $10M INDA-fallback long-only.
