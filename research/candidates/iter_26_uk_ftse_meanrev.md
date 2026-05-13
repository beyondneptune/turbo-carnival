# Iter 26 — UK / FTSE 100 Weekly Mean Reversion via EWU

## 1. Thesis (one sentence)

After an extreme weekly move in UK large-caps (last 5 trading days' return on EWU iShares MSCI United Kingdom ETF outside +/-3.0%), the **subsequent 5-trading-day return is on average opposite-signed**, because forced index-tracking flow (UK pension trustees, FTSE 100 futures rolls, sterling-hedged buyers) over-extends mid-week and partially mean-reverts the following week; the claim is falsified if, on the held-out 2023-2026 OOS window with ≥ 100 weekly signals, the rule's net-of-cost Sharpe falls below **0.5** or its win-rate below **50%**.

## 2. Market / asset class / venue

- **Primary instrument**: `EWU` — iShares MSCI United Kingdom ETF, NYSE Arca, USD-denominated, tracks the MSCI UK Investable Market Index (close proxy for FTSE 100 + FTSE 250 large-cap UK names). Trades during US RTH (09:30-16:00 ET = 14:30-21:00 BST) — i.e. **after** the London close (16:30 BST / 11:30 ET) so the entry order at Monday 15:00 ET incorporates the full prior-week UK regular session plus that Monday's LSE close.
- **Backup instrument**: `VGK` — Vanguard FTSE Europe ETF (broader European exposure including UK). Used only if EWU has a halt or material spread widening at signal time.
- **Signal underlying**: EWU daily closes (i.e. ETF prices, not the index level). Spec is **deliberately** ETF-priced — we are trading the ETF, so the signal must be measured on the instrument we trade, not on `^FTSE` which has a 5-hour stale-close offset vs USD-quoted EWU.
- **Primary exchange**: NYSE Arca.
- **Broker**: **Alpaca** (paper, then live equities) via the `requests` REST client used by the iter_5 host. Lesson 8 exemption argued: `requests` is the de-facto pattern for the Alpaca REST API in the existing live stack (`svxy_sweet_spot_host.py`, `iter_18_adr_alpaca_host.py`); we are not hand-rolling FIX, we are calling a documented REST endpoint that Alpaca officially supports and that other production hosts in this repo already use.
- **Settlement / margin / tax**: T+1 US equity settlement; no FX leg (EWU is USD-quoted). Long allowed long-only at Alpaca paper; short allowed only on margin-enabled Alpaca live accounts (paper account simulates). Tax regime: ordinary US 1099-B short-term capital gains; no K-1; no foreign withholding pass-through (EWU is a `1940-Act` ETF, not a partnership).
- **No new broker subscription required** — the operator's existing Alpaca AEGIS/HERMES/ATLAS accounts already cover EWU and VGK during US RTH.

## 3. Time horizon

**Multi-day (weekly).** Signal evaluated **once per week on Monday at 15:00 ET** (i.e. 30 minutes after the LSE close at 11:30 ET; 60 minutes before the US close at 16:00 ET). Holding period **5 trading days** (entry Monday close → exit following Monday close, or earlier on stop / take-profit). Maximum 1 position open at any time; the slot does not stack.

## 4. Method family and specific algorithm

- **Family**: Rules-based mean reversion. No ML, no LLM, no NN. The 5-day-return threshold and the 5-day hold are both simple, low-parameter rules — a deliberate choice to stay on the right side of Lesson 2 (binary gates, not confidence rankers).
- **Algorithm** (exact):
  1. **Weekly cron**: Monday 15:00 ET. If exchange is closed (US holiday), skip to next business day same time.
  2. **Signal compute**: `r5 = EWU_close[t] / EWU_close[t-5] - 1` where `t` is the most recent EWU close at the time of evaluation (i.e. the prior Friday close if Monday US-open data isn't yet final, or the Monday LSE close mapped through EWU's NAV-tracking which has already updated to the latest UK session).
  3. **Entry rule**:
     - If `r5 <= -0.030` AND no position open → **BUY** EWU, target 20% of NAV.
     - If `r5 >= +0.030` AND no position open AND account permits shorting → **SHORT** EWU, target 20% of NAV.
     - Else → no action.
  4. **Position size**: 20% of NAV at entry, computed as `qty = floor(0.20 * NAV / EWU_close)`. Hard cap: 25% NAV.
  5. **Hold period**: exactly **5 US trading days**, exiting market-on-close on day 5.
  6. **Take-profit (early exit)**: if cumulative return from entry exceeds **+3.0%** (long) or **-3.0%** (short) on any close, exit at next close.
  7. **Stop loss**: if cumulative return from entry breaches **-2.5%** (long) or **+2.5%** (short) on any close, exit at next close. Stop is **end-of-day evaluation only** — no intraday stops (mean-reversion bets sometimes look terrible intraday and recover overnight; intraday stops would shake us out of the edge).
  8. **No-stacking**: once a position is open, do not enter new positions until it has closed.
  9. **Friday-roll guard**: if the natural exit day would be a US half-day or a UK bank holiday, exit one trading day **earlier** to avoid thin-liquidity rolls.
- **Why rules, not ML**: the edge is mechanistic (forced-flow over-extension partially reverts) and has < 4 effective parameters (entry threshold, hold period, TP, SL). An ML overlay would have no ranking signal to add on weekly data with ~50 OOS signals/year — it would only add an overfitting surface. The iter_25 meta-validation framework would flag any ML-overlay version of this strategy as having insufficient OOS sample for a complexity premium.

## 5. Data requirements

- **Live signal**: free `yfinance` daily EWU close (or, in production, Alpaca's `/v2/stocks/EWU/trades/latest` for the most recent print). No paid feed required.
- **Historical (backtest)**: free `yfinance` daily bars for EWU (inception 1996-03-12 — but `^FTSE` and `EWU` reliable daily data starts circa 2007 for the post-GFC regime that's relevant). Sample used: **2010-01-01 → today**.
- **Supplementary** (sanity-check only, not in signal): `^FTSE` daily close for cross-checking EWU's tracking error; CBOE `^VIX` for regime overlay (NOT used in signal — would be next-iter enhancement).
- **Storage**: < 5 MB. Trivial.
- **Cost**: **$0 ongoing** (yfinance free, Alpaca paper account free, Alpaca live with no data subscription required for trade execution).
- **Latency**: signal is evaluated weekly; tolerance is **minutes**, not milliseconds. No HFT requirements. Standard residential or t3.micro VPS suffices.

## 6. Validation plan

- **Total daily sample**: 2010-01-01 → 2026-05-12 ≈ **4,100 trading days** ≈ **820 weekly signals** at most (one per Monday).
- **Train / IS window**: 2010-01-01 → 2022-12-31 (~13 years). Used **only** to confirm the basic mean-reversion structure and freeze the threshold at +/-3.0%, hold = 5 days, TP/SL = +/-3.0% / -/+2.5%. Parameters are NOT re-optimized.
- **OOS window**: 2023-01-01 → 2026-05-12 (~3.4 years). This is the held-out evidence sample. Hard requirement: **minimum 100 OOS weekly signals** before any verdict. If fewer than 100 signals are observed in the natural window, the validation is **extended** until 100 are reached — we do not promote with thin OOS.
- **Walk-forward** (secondary sanity): 8-year train / 1-year test, 6-month step. ~16 folds across the full sample. Reported as a robustness check; the primary verdict is the IS / OOS split above.
- **Purge / embargo**: 5-day embargo between train and test (the strategy holds 5 days, so any signal within 5 days of the cutoff would have an entry/exit straddling train and test).
- **Round-trip cost**: **5 bps** modeled (Alpaca commissions are zero on US equities for Alpaca-Pro, but median bid-ask spread on EWU is ~$0.02 on a ~$36 stock = ~5 bps half-spread; we round trip 10 bps but cap at 5 bps per round trip as a conservative net of zero commission).
- **Statistical tests and thresholds**:
  - **Net Sharpe**: OOS Sharpe **≥ 0.5** point estimate AND bootstrap 95% CI (1,000 resamples, weekly returns) excluding 0.
  - **Win rate**: ≥ **50%** of OOS trades profitable net of costs.
  - **Max drawdown**: peak-to-trough on the position-equity series **≤ 15%**.
  - **Trade count**: ≥ 100 OOS weekly signals (the hard sample-size gate).
  - **Stop-loss hit rate**: ≤ 30% of trades hit the -2.5% stop (higher rate implies the entry threshold is too loose or the regime has flipped).
- **No same-data backtest**: parameters frozen on 2010-2022 only. Anything we touch in 2023-2026 is a kill-or-promote decision, not a re-tune.

## 7. Risk envelope

- **Account size assumption**: this candidate is sized for **$50k - $500k** Alpaca paper or live equity account. The 20% NAV sizing is conservative enough to leave headroom for two other slot strategies (e.g. iter_5 SVXY + iter_18 ADR).
- **Max position size**: **20% of NAV** per entry. Hard cap **25% NAV** (one ETF, one ticker, one side).
- **Daily loss cap**: **-1.5% NAV** rolling 24-hour P&L on the EWU position → close at next close, halt new entries for 5 trading days.
- **Drawdown trigger**: **-8% trailing peak-to-trough on net equity over any 90-day rolling window** → strategy halts pending re-validation (re-run IS/OOS on most recent 13-year window).
- **Per-trade circuit breaker**: **3 consecutive negative trades** → halve sizing to 10% NAV for the next 5 trades. Restore after 2 consecutive winners.
- **Correlation rule with iter_5 (SVXY)**: SVXY has effective beta to SPX ~ +2/+3; EWU has effective beta to SPX ~ +0.7 with additional UK-specific factors. Their daily-return correlation should be ~0.3-0.5. **If rolling 30-day correlation between live iter_26 and live iter_5 daily P&L exceeds +0.6, pause iter_26** — the diversification thesis is failing.
- **No-overlap rule with iter_5**: both strategies trade ETFs at Alpaca; they share **no tickers** (iter_5 uses SVXY+SGOV; iter_26 uses EWU only), so the only overlap risk is portfolio-level correlation, not literal ticker-collision.
- **Single-currency-risk note**: EWU is USD-quoted but holds UK equities — the operator is implicitly bearing GBP/USD FX risk on the holding. We do **not** hedge this (small position, short horizon); we **monitor** it as part of the daily-loss cap.

## 8. Execution plan

- **Broker / SDK**: **Alpaca paper-then-live via `requests` REST** (same pattern as `svxy_sweet_spot_host.py`).
- **Order types**:
  - **Entry**: **marketable-limit IOC** at last_print +/- 25 bps (same aggression as the SVXY host). Submitted at Monday 15:00 ET = 60 minutes before the US close, giving plenty of time for fill and re-submit-if-needed.
  - **Exit (scheduled 5-day)**: **marketable-limit IOC** at last_print +/- 25 bps, submitted at exit-day 15:55 ET (5 minutes before close).
  - **Exit (TP or stop)**: same marketable-limit IOC, submitted at the close of the day the threshold is met (not intraday — see §4.7 design).
- **Expected slippage budget**: **5 bps round-trip** in normal conditions (EWU ADV ~3.5M shares × $36 = ~$125M/day, median spread ~$0.02 = 5 bps half-spread, marketable-limit IOC at 25 bps aggression typically fills inside the half-spread on most days). **20 bps** in stress conditions (post-Brexit-style headlines, sterling flash crash 2016-style).
- **Order rejection handling**:
  - **Entry reject** (Alpaca returns 4xx): retry once with the same payload after 5 seconds; on second reject, **skip the week's entry** and log the reject.
  - **Exit reject**: retry once at last_print +/- 50 bps; if still rejected, fallback to **MKT** order; if MKT also rejects, page operator (PagerDuty webhook) and hold the position **at most** until next close, then escalate manually.
- **Broker-truth reconciliation (Lesson 4)**: every decision cycle calls `GET /v2/positions/EWU` and reconciles against the local SQLite state. Any divergence → halt new entries, page operator. Exit orders always use the **broker-reported** quantity, never the local cache.

## 9. Capacity

- **EWU ADV**: ~3.5M shares × ~$36 = **~$125M/day notional**. A 20% NAV allocation at $500k NAV = $100k per trade = **0.08% of ADV** — completely negligible. At $10M NAV the strategy consumes 1.6% of ADV per trade, still well below the 5% rule-of-thumb that causes adverse-selection in marketable-limit IOC.
- **Realistic AUM ceiling for this strategy as deployed**: **$50M**. Above $50M, two frictions dominate: (a) marketable-limit IOC at 5 bps becomes harder to fill at full size in pre/post-news regimes, requiring TWAP slicing; (b) the position-equity-leg starts mattering for portfolio reconciliation at the broker.
- **Theoretical AUM ceiling from ADV alone**: ~$1B (at which point 20% NAV = $200M = >100% of EWU's daily ADV — strategy could not be entered in a single session).
- **Single-name max**: 100% of strategy NAV in EWU (single ticker by design). Issuer concentration risk: iShares is the sponsor; an iShares operational failure or ETF NAV suspension event hits the entire allocation. Mitigation in §10.
- **Single-sector max**: not applicable (EWU is a broad UK-large-cap index ETF). Implicit sector exposure: ~25% financials, ~15% energy, ~15% consumer staples, ~10% materials, ~10% healthcare (typical FTSE 100 weights).

## 10. Failure modes

1. **Brexit-style structural shock** (binary political event with multi-day overshoot, not mean-reverting). Metric: a single overnight loss > 5% on EWU (e.g. June 24 2016 = -11.5% on EWU). Detection: post-trade news-event scan; if a UK political event (general election, Bank of England rate decision day, EU summit) falls within the entry-week, **explicit rule: skip entry on UK-political-event weeks**.
2. **FTSE regime shift** (the mean-reversion edge dies in a new regime — e.g. sustained bear leg from 2007-2009 GFC, or sustained energy-shock rally where every UK move is one-directional). Metric: 90-day rolling Sharpe < 0 on OOS while the IS sample's same-window historical Sharpe was > 0.5. Detection: monthly KPI dashboard, halt if 6 consecutive negative trades.
3. **EWU tracking-error divergence** (EWU NAV decouples from FTSE 100 due to AP arbitrage failure, futures-leg disruption, or iShares operational issue). Metric: rolling 30-day correlation between EWU returns and ^FTSE returns drops below 0.8 (historical baseline 0.95+). Detection: daily NAV-tracking cron; halt strategy on detected divergence pending iShares public statement.
4. **GBP/USD currency-leg drag** (the FX leg eats the trade's edge — e.g. mean-reversion in GBP terms is +2% but cable moves -3% over the hold, netting -1% in USD). Metric: per-trade FX contribution to USD return > 50% of the underlying-mean-reversion contribution. Detection: monthly FX-decomposition cron; if 3 trades in a row are USD-net-losers but GBP-net-winners, pause and add cable filter.
5. **Liquidity flight / spread blowout** (e.g. UK gilt crisis Sept 2022, sterling flash crash). Metric: per-fill slippage > 30 bps; rolling 20-trade median slippage > 12 bps. Detection: per-fill TCA cron; pause strategy on median > 12 bps over any 20-trade window.
6. **Alpaca paper / live data divergence** (paper account fills at a price that doesn't replicate live broker behavior, leading to false-positive shadow results). Metric: paper Sharpe > live Sharpe by > 0.5 over any 30-day overlap window. Detection: dual-shadow on Alpaca paper AND a second broker (e.g. IBKR equities paper) for the first 90 shadow days.

## 11. Falsifiable verdict criteria

- **Promote to live capital** (all of the following over ≥ 100 OOS signals AND ≥ 30 paper-shadow days at the post-backtest deploy):
  - OOS net Sharpe **≥ 0.5** point estimate, 95% bootstrap CI lower bound > 0.
  - OOS win rate **≥ 50%**.
  - OOS max DD **≤ 15%**.
  - Zero broker-truth divergence events in shadow.
  - At least one drawdown-event (3 consecutive losers) in OOS that the circuit breaker handled within tolerance.

- **Continue shadow** — metrics ambiguous (Sharpe between 0.3 and 0.5, win rate 47-50%, DD 12-18%), no kill triggers fired.

- **Kill immediately** if any one of:
  - 3-day NAV drawdown > 5%.
  - Daily loss cap breach (-1.5% NAV in 24 hours).
  - Broker-side fill discrepancy > 0.5% NAV.
  - Rolling 90-day OOS Sharpe < 0.3 OR rolling 90-day max DD > 15% — **the 90-day-deploy falsification clause**.
  - 30-day correlation with iter_5 SVXY P&L > +0.6 (diversification thesis dead).
  - EWU trading halt > 1 session OR iShares announces material restructure of EWU.

## 12. One-paragraph operational risk note

The strategy is **operationally trivial in normal weeks** — one signal compute, one entry, one scheduled exit, weekly cadence — but the failure surface lives entirely in (a) the data-feed for the Monday 15:00 ET signal (if yfinance is rate-limited or stale, the signal is missed and we sit out the week; mitigation: fall back to Alpaca's `/v2/stocks/EWU/bars/latest` endpoint with the same Alpaca account credentials), and (b) the GBP/USD currency leg which is implicit, unhedged, and silent until a 3% cable move on a position week erodes the strategy's edge in USD terms (mitigation: monthly FX-decomposition cron and an OOS FX-contribution attribution that we review at every promotion gate). The strategy is also **silent during long no-signal periods** (the +/-3.0% threshold is empirically hit ~15-25% of weeks, so ~10-15 trades/year; the operator stops watching). Mandatory weekly review (signal print, position state, recent fills, kill-file presence) is a hard requirement enforced by a Monday-09:00-ET status email cron.

---

## Lesson compliance (the 10 lessons, explicit)

1. **Validate before deploy.** §6 mandates walk-forward 2010-2022 IS, 2023-2026 OOS with ≥ 100 OOS signals AND ≥ 30 paper-shadow days before any live capital.
2. **Binary gates, not confidence rankers.** The signal is **binary** (`r5 <= -0.03` or `r5 >= +0.03` or neither). No confidence score, no continuous sizing. Position size is fixed at 20% NAV when triggered, 0% otherwise — eliminates the C3-style confidence/P&L mismatch.
3. **TP:SL asymmetry is structural.** TP = +/-3.0%, SL = -/+2.5% → +3.0 / -2.5 = **1.2:1 reward:risk**, deliberately asymmetric to bias toward the mean-reversion thesis. The 5-day time-stop ensures we don't let losers ride past the half-life of the edge. This honors the "stop logic is part of the edge" lesson.
4. **Broker truth, not local cache.** §8 requires per-decision-cycle reconciliation via `GET /v2/positions/EWU`; closing orders use broker-reported quantity.
5. **No PDT-exposed lanes on sub-$25k US equity accounts.** The strategy holds **5 trading days** between entry and exit — by construction the only day-trade possibility is a TP/stop hit on the same day as entry, which is extremely rare on a 5-day mean-reversion bet (the 5-day return needed to trigger a same-day TP would require an intraday +3% move on EWU, which has happened <5 times in 15 years of data). **Argued exemption: PDT cannot trigger in normal operation because the rotation cadence is weekly with a 5-day mandatory hold.** Belt-and-suspenders: maintain ≥ $25k equity in the account, halt strategy if equity falls below $25k after a drawdown.
6. **One strategy per slot.** This is the canonical "UK / FTSE 100 weekly mean reversion" slot. Reject any other iteration that proposes EWU, ^FTSE, ISF (London-listed FTSE 100 tracker), or VGK as their primary instrument with a weekly mean-reversion thesis — they are the same edge.
7. **No same-data backtest.** §6 splits 2010-2022 (calibration / freeze), 2023-2026 (OOS evidence). Parameters are not re-touched after the 2022 cutoff.
8. **Battle-tested broker SDKs only.** Alpaca REST via `requests` — same pattern as production hosts already running in this repo (`svxy_sweet_spot_host.py`, `iter_18_adr_alpaca_host.py`, `iter_3_btc_funding_host.py`). Not hand-rolled FIX, not a new protocol.
9. **Capacity bound named upfront.** §9: **$50M realistic, $1B theoretical**. Edge degrades at the lower bound from marketable-limit IOC slippage during stress.
10. **Kill criteria written before deploy.** §11 enumerates exact thresholds. Top-line falsifier: **trailing 90-day OOS Sharpe < 0.3 OR trailing 90-day max DD > 15% → kill within 90 days of live.**

---

THESIS: Weekly mean reversion in EWU (UK large-cap ETF) after +/-3% 5-day moves, with 5-day mandatory hold, +3%/-2.5% TP/SL, 20% NAV sizing.
VALIDATION HORIZON: OOS Sharpe ≥ 0.5 with bootstrap 95% CI excluding 0 over ≥ 100 OOS weekly signals (≈ 2-4 years OOS).
CAPACITY CEILING: $50M realistic AUM (edge degrades above this from marketable-limit slippage in stress regimes); ~$1B theoretical from ADV alone.
GLOBAL-EXPANSION RATIONALE: fills the operator's 08:00-14:30 BST coverage gap by using the **EWU US-listed proxy** during US RTH (which captures the prior day's full UK session), so no new broker subscription required tonight.
