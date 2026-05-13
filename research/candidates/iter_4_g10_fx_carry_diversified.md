# iter_4_g10_fx_carry_diversified

## 1. Thesis (one sentence)

Uncovered interest parity fails on average in G10 spot FX, so a monthly-rebalanced long-3/short-3 carry portfolio across the G10 currency cross-section, sized by interest-rate differential and de-risked by an FX-implied-volatility filter, earns a positive risk premium net of swap and transaction costs — falsifiable as: 60+ out-of-sample monthly rebalances 2011-2025 do not produce a net annualized Sharpe > 0.55 after costs.

## 2. Market / asset class / venue

- **Asset class**: G10 spot FX (24/5, Sun 22:00 UTC to Fri 22:00 UTC).
- **G10 currency universe (10 names)**: USD, EUR, JPY, GBP, CHF, CAD, AUD, NZD, NOK, SEK.
- **Traded pairs (9, all USD-quoted for unified PnL)**: EUR/USD, USD/JPY, GBP/USD, USD/CHF, USD/CAD, AUD/USD, NZD/USD, USD/NOK, USD/SEK. USD itself is the numeraire and is implicitly long/short as the residual of the other nine.
- **Primary venue/broker**: OANDA via `oanda-py` (v20 REST API) on the OANDA-Tokyo / OANDA-NY books for retail-scale execution; IBKR IDEALPRO via `ib_insync` is the production-scale fallback (lower spreads, higher minimums). Both are battle-tested SDKs per Lesson 8.
- **Settlement**: T+2 for spot, but we hold via rolling swap (tom-next) — broker auto-rolls daily; no physical settlement at retail.
- **Margin**: OANDA US retail = 2% on G10 majors (50:1); IBKR = ~2.5% blended. We plan to use ~5x effective leverage (20% utilization of available margin), well inside CFTC retail caps.
- **Tax regime**: US Section 988 ordinary income by default (or 1256 60/40 if Section 988 opt-out filed). Tracked, not exploited.

## 3. Time horizon

**Daily bars, monthly rebalance.** Signal is computed end-of-month NY close (21:00 UTC last business day); orders execute over the first 2 hours of the next month's Asia session to spread market impact. Positions are then held ~21 trading days until the next rebalance. Carry accrues continuously via daily swap.

## 4. Method family and specific algorithm

**Rules-based cross-sectional carry ranking with a volatility regime filter.** Not ML — the signal is the interest-rate differential, which is observable, monotonic, and not subject to overfitting.

Algorithm:
1. **Monthly carry signal** (last business day of each month): for each non-USD G10 currency `c`, compute `carry_c = i_c − i_USD`, where `i_c` is the 1-month OIS rate (proxy: central-bank policy rate from the BIS/IMF feed, or the broker's quoted swap-rate implied yield).
2. **Rank** the 9 non-USD currencies by `carry_c`, top to bottom.
3. **Long basket**: top 3 yielders (typically AUD, NZD, NOK in normal cycles; can rotate to GBP/USD/CAD).
4. **Short basket**: bottom 3 yielders (typically JPY, CHF, SEK in normal cycles).
5. **Weights**: vol-scaled within each basket — weight `w_c ∝ carry_c / σ_c` where `σ_c` is trailing 60-day realized log-return vol of the pair vs USD. Normalize each basket to 50% of gross notional. Equal-weight is the fallback if vol estimates are unstable.
6. **Volatility regime filter (CRITICAL — see §10)**: compute the cross-sectional average of 1-month FX-implied vol across the 9 pairs (proxy: CVIX index, or broker's option-implied ATM vol surface, or a 20-day realized-vol proxy if implied unavailable). If `IV_avg > 12%` annualized, **scale exposure linearly** down toward zero by 14% (i.e., full size at 12%, half size at 13%, flat at 14%+). This is the Menkhoff/Sarno/Schmeling/Schrimpf (2012) "global FX volatility" hedge against carry crashes.
7. **Hold to next monthly rebalance** unless the vol filter trips intra-month (re-check weekly), in which case scale down at the next Monday NY open.

**Why rules, not ML**: the carry edge is a 35-year robust premium with a clear economic mechanism (peso problem, crash risk compensation, intermediary capital constraints). Adding ML invites overfitting to a noisy signal whose linear form already works. Per Lesson 2, no confidence-weighted sizing — the vol filter is a binary-region gate, not a probability score.

## 5. Data requirements

- **Spot FX bars**: daily OHLC for the 9 pairs from 1990 (or earliest available) to present. Sources: HistData.com free 1-min CSVs (rebuild to daily), Dukascopy historical, or OANDA's historical endpoint (`/v3/instruments/{pair}/candles`). ~30 MB total.
- **Policy rate / OIS series**: monthly central-bank policy rate for each G10 country, 1990-present. Source: BIS Macro Data, IMF IFS, or FRED (free). ~5 MB.
- **Implied vol series**: CBOE EUVIX/JYVIX or the broker's ATM IV surface for the volatility filter. Free monthly proxy: realized 20-day vol if IV unavailable. ~10 MB.
- **Storage estimate**: <100 MB total — FX daily is trivial. 1 GB/year budget for daily snapshots of the OANDA quote stream during live operation.
- **Cost estimate**: $0/month for OANDA REST data feed (free with funded account); $0/month for FRED/BIS macro; IBKR FX data is bundled with the IDEALPRO subscription (~$10/mo). Total: ~$10/mo data, vs. equities feeds at $200+/mo.
- **Latency requirement**: 5-second order ack tolerable. Carry is a daily-horizon strategy; sub-second execution is unnecessary and counterproductive (slippage worse on aggressive orders). Round-trip <500 ms acceptable, OANDA US-east colo measures ~100-200 ms.

## 6. Validation plan

- **Sample**: Daily spot FX, 1990-01-01 to 2025-12-31 (~36 years, ~9,000 trading days, ~432 monthly rebalances available).
- **Walk-forward design**:
  - Train (parameter check only — no fitting; rules are predetermined): 1990-2010 (252 monthly rebalances).
  - Test (out-of-sample, rolling): 2011-2025 (180 monthly rebalances available; we require minimum 60 OOS monthly rebalances which is comfortably satisfied).
  - Step: expand training window by 12 months annually, re-evaluate.
- **Purge & embargo**: 5 business days between train and test windows (since rebalance positions are held ~21 days). Embargo prevents lookahead from the last training month into the first OOS rebalance.
- **Cost model**: 1.5 bps round-trip spot spread for each rebalance leg (conservative for G10 majors at OANDA retail); swap costs taken from historical broker tom-next quotes, or proxied by `(i_long − i_short) − transaction_friction` of ~10 bps/year drag.
- **Statistical thresholds**:
  - Net of cost annualized Sharpe > 0.55 with 95% bootstrap CI excluding 0.30 — passes.
  - Max drawdown < 25% on the OOS window — passes.
  - 2008 Q4 and 2024 Aug yen-reversal months: each loss < 8% (volatility filter must demonstrably reduce these tails).
- **Minimum OOS sample**: 60 monthly rebalances OOS (5 years) before any verdict. 2011-2015 alone gives 60; full 2011-2025 gives 180.

## 7. Risk envelope

- **Account assumed**: $50,000 USD funded at OANDA (illustrative; scales linearly).
- **Max gross notional per single currency leg**: 25% of equity = $12,500 notional. With 9 pairs and ~5x leverage cap, total gross notional capped at $125,000 (2.5x equity), well inside the 50:1 OANDA limit.
- **Max net notional**: 50% of equity = $25,000 (carry portfolios are designed to be near dollar-neutral; net drift indicates a stale signal).
- **Daily loss cap**: 2% of equity = $1,000/day. Hard halt on breach for 24h.
- **Drawdown trigger**: rolling 30-day DD > 8% triggers a 50% deleverage; > 12% triggers full flat and a weekly review before re-deploy.
- **Circuit breaker**: 3 consecutive losing months (rare for carry) triggers a strategy-level kill review — not auto-shutdown, since carry has historically gone 4-5 months negative around crashes before recovering.
- **Correlation rule**: zero overlap with any deployed US-equity or options strategy (different asset class entirely). Must check correlation < 0.3 with any deployed crypto-carry or rates-trade if introduced later.
- **FX margin budget**: 2% used; broker call hits at 50%. We sit at ~10% of margin utilization with a $50k account at full size, so margin call risk is near zero absent a 30%+ overnight gap.

## 8. Execution plan

- **Broker / SDK**: OANDA via `oanda-py` (`oandapyV20.endpoints.orders.OrderCreate`). IBKR via `ib_insync` for the > $250k AUM tier on IDEALPRO.
- **Order type**: marketable-limit IOC with a 2 bps slippage tolerance vs. the mid at signal time. Carry rebalance is non-time-critical, so we spread orders across the first 2 hours of the Asia session to minimize market impact and let the spread tighten as London/NY open approaches.
- **Order schedule**: 9 pair orders submitted sequentially, not in a single basket, with 60-second spacing to avoid all hitting the same liquidity pocket. Total rebalance window: 9 minutes.
- **Slippage budget**: 1.5 bps round-trip per pair, measured ex-post by comparing fill price to the mid-quote at signal generation time. Track a 12-month rolling slippage; if it exceeds 3 bps for 2 consecutive months, downgrade execution to IBKR IDEALPRO.
- **Broker rejection handling**: on reject, retry once with a tighter limit, then escalate to a wider IOC. If still rejected after 3 attempts, log and skip that leg for the cycle — note that residual non-zero net exposure will be liquidated to USD at the next rebalance. Per Lesson 4, reconcile against broker `/v3/accounts/{id}/positions` after every order; trust broker truth.
- **Reconciliation cadence**: every 60 seconds during the rebalance window, hourly during the hold period, end-of-day always.

## 9. Capacity

- **G10 FX is the deepest liquidity pool on earth** — global daily spot turnover ~$7.5 trillion (BIS Triennial 2022).
- **AUM ceiling for this design**:
  - **Retail / OANDA**: $1M effective AUM. Above this, OANDA spreads widen and the platform's slippage on the less-liquid SEK/NOK legs degrades.
  - **Institutional / IBKR IDEALPRO**: $50M-$100M before market impact on the NOK/SEK legs starts mattering. EBS/Refinitiv ECN tier (out of scope for this design) extends to $500M+.
  - **Edge degradation point**: the carry premium itself does not degrade with AUM up to multi-billions (academic literature confirms this); the degradation is purely execution.
- **ADV consumption at full size**: at $1M AUM with ~5x leverage = $5M gross, with the largest single leg ~$1.5M on a pair like AUD/USD that trades $200B+/day — strategy consumes 0.00075% of ADV. Negligible.
- **Concentration risk**:
  - Single-currency max: 25% of equity (already in risk envelope).
  - Single-region max: implicit cap via cross-section — typically 2 of the 3 long legs are commodity exporters (AUD/NZD/NOK/CAD), which is a known concentration; we accept it as the carry trade's structural feature.

## 10. Failure modes

The carry trade has well-documented crash dynamics. Each mode below has a metric signal and a mitigation.

1. **Carry crash / risk-off cascade (e.g., Oct 2008 GFC, Aug 2024 yen-carry unwind)**: high-yielders (AUD/NZD/EM) collapse vs. low-yielders (JPY/CHF) in days. Aug 5, 2024: yen rallied 7%+ in 24h, BTCJPY-style cascading liquidations across global carry books.
   - **Metric**: cross-sectional FX-implied vol surge > 14% annualized; or 5-day rolling realized vol of the long basket - short basket > 18% annualized.
   - **Mitigation (volatility filter rule, restated)**: scale linearly from full size at IV_avg=12% to flat at IV_avg=14%. **In live deployment, additionally: any single overnight gap > 2.5% on the long-short spread triggers immediate 50% deleverage at next London open.** This is the explicit address of Lesson 10.
2. **Regime change in monetary policy (e.g., BoJ ends NIRP 2024-2025)**: the JPY short suddenly carries positive forward bias against expectations; the entire ranking inverts over weeks.
   - **Metric**: month-over-month change in any leg's policy rate > 50 bps. Trigger a one-month soft pause (half size) to let the new ranking stabilize.
3. **Liquidity flight at month-end / quarter-end**: bank-balance-sheet windowing causes spreads to widen 3-5x on swap dates, particularly Dec 31 and quarter-ends.
   - **Metric**: realized slippage on the rebalance > 4 bps (vs. 1.5 bps budget).
   - **Mitigation**: shift rebalance to T-2 or T+2 of month-end, not exactly month-end. Skip the Dec 28-Jan 3 window entirely.
4. **Broker swap-rate drift**: OANDA's tom-next swap rates can be 30-50% worse than interbank, eroding the carry premium. If swap PnL realized is < 60% of theoretical, switch to IBKR.
   - **Metric**: monthly realized swap accrual vs. (i_long-i_short)*notional*(days/360) ratio < 0.6.
5. **FX intervention shock (rare, but real — e.g., SNB Jan 2015 EUR/CHF unpeg)**: low-probability tail event where a central bank breaks a peg or intervenes intraday.
   - **Metric**: any single-day single-pair move > 5σ relative to trailing-60d vol.
   - **Mitigation**: structural exposure cap of 25% per leg already enforces a hard loss ceiling. Acknowledge this is a genuine tail risk and budget it into the kill-criteria.

## 11. Falsifiable verdict criteria

These trigger within 90 days of live deployment.

- **Promote to full live capital after 90-day shadow + 30-day live $1k-size run**:
  - Net Sharpe (annualized) > 0.55 with 95% bootstrap CI excluding 0.30.
  - Max DD < 8% over the shadow window.
  - Realized slippage ≤ 2 bps round-trip per leg.
  - Zero broker-truth reconciliation discrepancies > 0.5%.
  - Realized swap PnL is ≥ 60% of theoretical (broker not stealing the carry).
- **Continue shadow / extend test**:
  - Sharpe between 0.20 and 0.55, no kill trigger hit, vol filter has fired at least once and behaved as designed.
- **Kill immediately within 90 days** (any one):
  - 30-day DD > 12%.
  - Single-day loss > 2% of account (the daily cap).
  - Broker-side fill discrepancy > 0.5% on any single rebalance (Lesson 4).
  - Realized slippage > 4 bps round-trip for 2 consecutive monthly rebalances (Lesson 8 implicit — execution surface is broken).
  - Volatility filter fails to engage during a >14% IV regime, allowing a >5% drawdown that the rule should have prevented (the rule itself is broken).
  - Any of the failure modes in §10 fires without the mitigation engaging.

## 12. One-paragraph operational risk note

OANDA is a single point of failure: if their v20 REST or the FXTrade backend goes down during a rebalance window, we may sit with stale or no positions — current mitigation is the IBKR IDEALPRO fallback, but switching between SDKs mid-cycle is operationally risky and untested. Time-zone bugs are the highest-probability operational error: rebalance is keyed off NY close (21:00 UTC), and a daylight-savings transition or a calendar lookup error could fire the rebalance an hour early into illiquid Asia open and eat 5+ bps of slippage. Daily swap accrual depends on the broker's internal tom-next calendar; holidays and weekend rolls (triple-swap Wednesday) need to be reconciled against the broker's posted swap statement weekly. Finally, central-bank policy-rate feeds (BIS/FRED) update with 1-3 day lags after a rate decision, so any rebalance within 5 business days of a major central-bank meeting (Fed/ECB/BoJ/BoE) should manually verify the rate inputs before submitting orders.

---

## Lessons compliance

1. **Validate before deploy**: walk-forward 1990-2010 train / 2011-2025 test, ≥60 OOS monthly rebalances, then 90-day shadow, then 30-day small-size live before scaling. Compliant.
2. **Binary gates, not confidence rankers**: the vol filter is a deterministic linear scaling between 12% and 14% IV — a region gate, not a probability output. No ML confidence is used for sizing. Compliant.
3. **TP:SL asymmetry applies even at monthly horizon**: at month-scale, the analog of "don't close winners too soon" is "don't deleverage on a normal 1σ pullback in the long-short spread" — the vol filter only fires on the system-wide IV signal, never on a single pair's drawdown. The structural asymmetry is encoded as: vol filter cuts exposure (kills the loser-regime) but no symmetric profit-taking trigger (lets the winner-regime run to next rebalance). This is the carry-horizon equivalent of the wide-TP/tight-SL principle: tight downside via the vol filter and the daily loss cap, but no upside cap before the rebalance horizon. Compliant.
4. **Broker truth not local cache**: every order reconciled against `/v3/accounts/{id}/positions` (OANDA) or `IB.positions()` (IBKR) at 60-second cadence during rebalance, hourly during hold. No local quantity used for closing logic. Compliant.
5. **No PDT-exposed lanes**: **N/A for FX.** PDT is FINRA Rule 4210, which applies only to US margin equity accounts under $25k. OANDA FX is a CFTC/NFA-regulated forex retail account, not a securities margin account. IBKR IDEALPRO is similarly outside Reg-T equity rules. Day-trading FX has no round-trip count cap. The account can sit at any size from $2k upward without PDT exposure. Compliant by structural exemption — this is a primary reason to choose FX as the asset class.
6. **One strategy per slot**: this is the unique "G10 monthly carry" slot. Must not be deployed alongside any other FX-carry, EM-carry, or rates-curve-carry strategy without an explicit correlation review (<0.3 required). Compliant.
7. **No same-data backtest**: walk-forward with expanding train window and purged 5-day embargo. Parameters (the long-3/short-3 split, the 12%/14% IV thresholds, the 60-day vol window) are pre-registered from the academic literature (Lustig-Verdelhan 2007; MSSS 2012), not tuned to the OOS sample. Compliant.
8. **Battle-tested SDK only**: `oanda-py` (oandapyV20) and `ib_insync`. No hand-rolled HTTP. Compliant.
9. **Capacity bound named upfront**: $1M retail at OANDA, $50-100M at IBKR IDEALPRO, edge premium itself is capacity-resilient to multi-billions — bottleneck is execution venue, not edge. Compliant.
10. **Kill criteria written before deploy**: §11 lists the exact numbers, all triggerable within 90 days. Compliant.

THESIS: G10 monthly long-3/short-3 carry, vol-scaled and filtered by FX-implied vol, earns a positive risk premium that walk-forward OOS Sharpe > 0.55 net of costs.
VALIDATION HORIZON: 60+ monthly rebalances OOS (2011-2025 provides 180; minimum 60 required for verdict).
CAPACITY CEILING: $1M USD at OANDA retail; $50-100M at IBKR IDEALPRO institutional tier.
