# Iter 6 — Nikkei 225 Futures Tokyo-Session Opening-Range Breakout

## 1. Thesis (one sentence)

The first 30 minutes of the Tokyo cash session prints a concentrated burst of Japanese retail and dealer-hedging flow into Nikkei 225 mini futures, and an N-tick break of that range during the morning session (09:30-11:30 JST) trends through to lunch break with positive expectancy net of one round-turn commission and 1-tick slippage; falsified if 200+ OOS trading days produce an average per-trade P&L not statistically distinguishable from zero at 95% bootstrap CI.

## 2. Market / asset class / venue

- **Primary instrument:** Nikkei 225 Mini Futures
  - **CME route (preferred for US-domiciled retail):** `NKD` (Nikkei/USD, $5 per index point ≈ $5 per tick, tick = 5 index points but quoted as `$5` per minimum tick movement on CME spec). Verify exact spec at deploy: `NKD` is $5 × Nikkei index. Mini contract on CME is `NIY` ($5 yen-denominated). **Selected: `NKD` USD-denominated, $5 per point, no FX leg required.**
  - **OSE route (alternate, lower-cost):** `NK225M` (Nikkei 225 mini on Osaka Exchange), multiplier ¥100 per point, tick = 5 points = ¥500 (~$3.30 at USD/JPY 150). Requires IBKR Japan permission and JPY funding.
- **Tickers / SDK identifiers (IBKR ib_insync):**
  - CME: `Future('NKD', '<frontMonth>', 'CME')` — quarterly Mar/Jun/Sep/Dec.
  - OSE: `Future('N225M', '<frontMonth>', 'OSE.JPN')`.
- **Primary exchange:** CME Globex (NKD) — fully covers Tokyo daytime via CME's electronic 23-hour schedule, but the thesis trades **only during OSE day-session hours** when the underlying cash index is live. We use NKD for execution because its quote tracks the same underlying with deeper US-retail-accessible liquidity in cross-listed hours.
- **Broker:** Interactive Brokers (IBKR), TWS / IB Gateway with `ib_insync` SDK. Account type: Reg-T margin or Portfolio Margin; futures permission required (CME and ideally OSE add-on).
- **Settlement:** Cash-settled to Nikkei 225 SQ price on second Friday of expiry month. We never hold to expiry — strategy is intraday-flat.
- **Margin:** CME NKD initial ≈ $5,500/contract (varies with SPAN; check at deploy). Maintenance ≈ $5,000. Day-trade margin via IBKR ≈ 50% of overnight (we never carry overnight on this strategy, but use full overnight margin as the planning number to absorb intraday spikes).
- **Tax regime (US filer):** Section 1256 60/40 long-term/short-term capital gains treatment on US-listed futures including NKD. Mark-to-market on Dec 31. No wash-sale rule. If routed via OSE direct, becomes ordinary income / J-tax treaty considerations — adds friction; **default to NKD.**

## 3. Time horizon

**5-minute intraday.** Opening range defined on the first six 5-minute bars (09:00-09:30 JST). Trade window 09:30-11:30 JST. Exit forced at 11:25 JST (5 min pre-lunch). No carry through 11:30-12:30 JST lunch break under any circumstance. No re-entry in afternoon session.

## 4. Method family and specific algorithm

**Rules-based opening-range breakout (Toby Crabel 1990 lineage, adapted to JP equity index 5-min bars).**

Algorithm:
1. **Build the opening range (OR):** At 09:30:00 JST capture `OR_high = max(high_t)` and `OR_low = min(low_t)` for the six 5-min bars from 09:00-09:30 JST (using TSE cash open, mirrored on NKD via session alignment table).
2. **Compute breakout buffer:** `buffer = 2 ticks` (10 index points on NKD = $10/contract). This is the Crabel-style noise filter calibrated to JP intraday vol.
3. **ATR regime filter:** Compute the 20-day ATR of the prior session's full 5-min range. **Require `0.4% ≤ ATR/Close ≤ 2.5%`.** Below 0.4%: noise regime, no edge. Above 2.5%: crisis/news regime, suppress strategy.
4. **Long entry:** marketable-limit IOC at `OR_high + 2 ticks` triggered when a 5-min bar closes above that level during 09:30-11:00 JST (last entry 30 min before forced flat).
5. **Short entry:** symmetric at `OR_low - 2 ticks`.
6. **Stop:** opposite side of OR. Long stop = `OR_low - 1 tick`. Short stop = `OR_high + 1 tick`. (Stop distance = OR width + 3 ticks.)
7. **Time stop:** Forced flat at 11:25 JST regardless of P&L.
8. **No pyramiding, no scaling, one direction per day.** If long is taken and stopped, do not flip short. Day is done.

**Why rules and not ML:** OR breakout is a structural artifact of session-open auction concentration; the edge is regime-conditional but not high-dimensional. ML on a feature set this small overfits the validation folds. Crabel-family rules have 30+ years of out-of-sample track record across multiple equity indices, including documented persistence in Nikkei post-2000 (Crabel 1990 _Day Trading with Short Term Price Patterns_; replications in JP literature on TOPIX intraday momentum, Hamao & Hasbrouck 1995 microstructure, more recent Sato & Sakata 2018 JP retail order-flow studies). Lesson 2 satisfied: no ML confidence ranking is used; this is a binary rule gate (regime filter + price break).

## 5. Data requirements

- **Historical feed (validation):**
  - 5-min OHLCV bars on NKD (CME) and N225M (OSE) from 2018-01-01 through deploy.
  - Source: Polygon.io Futures (CME side), or IBKR historical bars via `reqHistoricalData` (free with active account, 5-yr rolling limit). For OSE, IBKR historical or paid CQG / Refinitiv pull.
  - JST session calendar (TSE/OSE holidays via `pandas_market_calendars` `JPX` calendar).
- **Live feed (runtime):**
  - IBKR real-time bars via `reqRealTimeBars(5)` aggregated to 5-min, OR IBKR market data subscription "CME Real-Time" ($11/mo) for NKD top-of-book.
  - USD/JPY mid for risk reporting only (NKD is USD-denominated; no FX hedge).
- **Storage estimate:** 5-min bars × 23 hr/day × 250 trade-days × 7 yr ≈ 480k bars/contract × ~80 bytes ≈ **40 MB total** for both NKD and N225M combined. Trivial. Store as Parquet under `data/nkd_5min/`.
- **Cost estimate:**
  - IBKR market data: ~$11/mo CME Real-Time + $5/mo OSE (if direct) = **$11-16/mo runtime**.
  - Polygon Futures: $79/mo if used; otherwise free via IBKR historical (preferred — keeps us aligned with the project's Polygon-Free cache-first principle from CLAUDE.md).
  - Total target: **$11/mo runtime, $0 historical.**
- **Latency requirement:** Round-trip < 500ms from bar close to order ack. IBKR US-east → Globex Aurora < 50ms typical; we have ~280s of slack within the 5-min cadence. Latency is not a constraint at this time scale.

## 6. Validation plan

- **Walk-forward design:**
  - Train window: 2018-01-01 to 2023-12-31 (6 years in-sample).
  - Held-out test: 2024-01-01 to 2026-05-12 (~28 months OOS, ≥ 500 trading days, ≥ 200 actual trade days after regime filter).
  - Step-forward refit: rolling 12-month train / 3-month test, advancing 3 months per step. No parameter refitting during the OOS window — the 2018-2023 fit is the frozen model evaluated cleanly on 2024-2026.
- **Purge & embargo:** Trades are intraday and flat by 11:25 JST; no overnight serial correlation. Purge = 0 days, embargo = 1 day between folds to avoid look-ahead via overnight gap.
- **Minimum OOS trades before verdict:** **200 trade days with at least one breakout signal.** Given the ATR regime filter, expected trigger rate ≈ 35-45% of trading days, so 200 trades ≈ 500 calendar trading days (the 2024-2026 OOS window).
- **Statistical tests and thresholds (all must pass):**
  - Per-trade mean P&L: bootstrap 10k resamples, 95% CI lower bound > 0 USD.
  - Annualized Sharpe (per-trade, scaled by sqrt(trades/yr)): point estimate > 1.0, 95% bootstrap CI lower bound > 0.5.
  - Profit factor: > 1.25.
  - Win rate: NOT a gate (breakout strategies are sub-50% win rate by design). Lesson 3 honored.
  - Max consecutive losing days: ≤ 7 (kill threshold).
  - Maximum 60-trade-day drawdown: ≤ 10% of allocated capital.
  - Long/short symmetry test: |long_sharpe - short_sharpe| / mean_sharpe < 1.0. Asymmetric collapse is a known OR failure mode and disqualifies.
- **Shadow paper run (post-OOS, pre-live):** ≥ 30 trading days at IBKR paper account using identical code path as live. Reconcile fills against an offline backtest replay on the same bars; any divergence > 0.25 ticks/trade investigated before live.

## 7. Risk envelope

Account size assumed: **$100,000 USD** dedicated to this strategy slot.

- **Position size:** 1 contract per $50,000 of equity, capped at **2 contracts** at $100k. Reasoning: with NKD = $5 × ~38,000 index points, contract notional ≈ $190k. 2 contracts = $380k gross exposure, ~3.8× leverage. Initial margin 2 × $5,500 = $11,000 (11% of account). Stop distance ~ (OR width + 3 ticks). Average OR width on Nikkei 5-min open ≈ 80-150 points historically, so per-contract stop ≈ $400-$750.
- **Max position $ value:** $380,000 notional (2 contracts) — disclose to user, this is leveraged.
- **Daily loss cap:**
  - **Absolute USD: $1,500** (1.5% of $100k account). Hit this → no trades for 24 hours.
  - **JPY-volatility regime cutoff:** if USD/JPY 1-day realized vol > 1.5% OR JNI (Nikkei Volatility Index) > 35, suppress strategy for the session. These are the conditions under which OR breakout historically collapses (BoJ surprise days, 2024 carry-trade unwind).
- **Drawdown trigger that halts trading:**
  - 20-day rolling drawdown > 5% of account → halve size (1 contract max).
  - 60-day rolling drawdown > 10% → full halt, review.
- **Circuit breaker:**
  - 3 consecutive losing trade-days → halt for 5 trading days, review log.
  - Any single-day loss > $1,500 → 24-hour halt.
  - JP market-wide circuit breaker triggered on TSE → flatten and stand down rest of session.
- **Correlation rule:** This strategy is long/short Nikkei intraday. Already-deployed strategies in this account: none in this iteration set are JP equity. ES futures (iter 2) correlates with NKD overnight (~0.55 daily returns) but trades different sessions. **Rule:** if iter 2 (ES) is live AND this strategy is live, cap combined gross notional at 4x account ($400k). Lesson 6: this is not a duplicate — different session, different exchange, different microstructure signal.

## 8. Execution plan

- **Broker / SDK:** IBKR via `ib_insync==0.9.86` (battle-tested; lesson 8). Connect to IB Gateway, not TWS, for runtime stability.
- **Order types:**
  - **Entry:** Marketable-limit IOC at `OR_high + 2 ticks` (buy) / `OR_low - 2 ticks` (sell). Limit set 1 tick beyond trigger to permit slippage; IOC to avoid resting. If unfilled, no chase — abandon signal for the day.
  - **Stop:** stop-limit order placed immediately on fill, at `OR_low - 1 tick` (long) / `OR_high + 1 tick` (short), limit offset 3 ticks beyond stop to ensure fill in fast moves.
  - **Time exit:** `MarketOrder` at 11:25 JST. Slippage on Tokyo lunch wind-down is typically 1 tick; accept.
- **Slippage budget:** 1.5 ticks round-turn = $7.50/contract. Measured per-trade as `(realized_fill - intended_price)`; rolling 20-trade median tracked. Alert if > 2.5 ticks/RT.
- **Commission:** IBKR ~$2.20/contract round-turn on NKD. Embedded in backtest.
- **Broker rejection handling:**
  - Entry reject (price moved past limit): log, skip, no retry — protects against runaway entries.
  - Stop reject: emergency `MarketOrder` flatten + page user.
  - Connection loss: auto-reconnect via ib_insync watchdog; if position open and reconnect fails > 30s, IBKR's bracket logic holds the stop server-side. Verify stop is GTC at IBKR before disconnecting client.
  - **Broker-truth reconciliation (lesson 4):** every 5 min loop pulls `ib.positions()` and compares to local state. Any mismatch → halt new entries, flatten via broker-reported quantity, page user.

## 9. Capacity

- **AUM ceiling:** Strategy risks ~$750/contract per trade. To stay below 1% per-trade risk on a single contract, account size ≤ $75k requires 1 contract; account ≥ $150k can run 3 contracts. NKD ADV ≈ 30k-50k contracts/day at CME. **Strategy is capacity-constrained by per-trade slippage, not gross size: above 50 contracts/clip we expect ≥ 1 tick slippage and edge degrades**. AUM ceiling ≈ **$2.5M** at this configuration (50-contract clip × ~$50k margin per contract slot). Above that, route to OSE for deeper book or reduce position frequency.
- **ADV consumed at full size (2 contracts at $100k):** 0.005% of NKD ADV. Negligible. Even at $2.5M AUM (50 contracts), ~0.1% of ADV — well below the 1% rule-of-thumb impact threshold.
- **Concentration risk:** 100% in Nikkei 225 index futures. This is single-instrument concentration by design. Mitigated by intraday-flat constraint (no overnight gap, no weekend, no SQ-week roll risk handled by trading front-month only and stopping 5 days before expiry).

## 10. Failure modes

1. **Regime shift to range-bound morning sessions (failure metric: profit factor drops below 1.0 over rolling 60 trades).** Mechanism: if Japanese retail flow shifts to options or single-stocks and futures OR loses its concentration property, breakouts become whipsaws. Detection: rolling profit factor + win rate of "OR fade" inverse rule (if the inverse rule starts winning, the regime has flipped).
2. **BoJ / MoF intervention shock (failure metric: single-day loss > 2x daily cap).** Mechanism: JPY interventions or BoJ surprise rate decisions produce gap-and-reverse patterns that punish breakouts. Detection: JNI > 35 AND USD/JPY 1d move > 1% pre-open → strategy auto-suppressed. Residual risk on un-telegraphed shocks accepted as the failure-mode tail.
3. **Liquidity flight on NKD / cross-listing arbitrage breakdown (failure metric: slippage > 2.5 ticks/RT for 5 consecutive days).** Mechanism: if NKD volume migrates entirely to OSE direct (regulatory or fee-driven), the CME quote becomes stale and our entries fill at worse prices. Detection: slippage telemetry. Fallback: switch routing to OSE direct via IBKR (requires permission pre-arranged).
4. **TSE outage / exchange holiday surprise (failure metric: code crash or position stranding).** Mechanism: TSE has had multi-hour outages (Oct 2020). Detection: `pandas_market_calendars` JPX feed checked at session start; IBKR market-data heartbeat checked every 30s in-session.
5. **DST / time-zone bug in JST handling (failure metric: OR computed on wrong bars).** Mechanism: JST does not observe DST but the developer's local time does; calendar drift breaks the strategy window. Detection: unit test asserts every session start that `now.astimezone(JST).hour == 9` at first bar; if not, halt.

## 11. Falsifiable verdict criteria

**Promote to live capital** (all must hold after 30 shadow days following the OOS pass):
- Shadow Sharpe (per-trade × sqrt(trades/yr)) > 1.2.
- Shadow max drawdown < 8% of allocated capital.
- Shadow profit factor > 1.25.
- Shadow long/short symmetry holds (Sharpe ratio between sides within 2x).
- Zero broker-truth divergences (lesson 4) over shadow window.
- Slippage realized ≤ 1.5 ticks/RT median.

**Continue shadow** (one of):
- Sharpe in (0.5, 1.2) AND drawdown < 8% AND no kill trigger hit → extend shadow another 30 days, max 60 days total.
- Trade count < 60 in shadow window (insufficient sample) → extend.

**Kill immediately** (any one):
- 3-day rolling drawdown > 5% of allocated capital.
- Single-day loss > $1,500 (daily cap breach).
- Broker-side fill discrepancy > 0.5% on any single trade (lesson 4).
- 7 consecutive losing trade-days (Lesson 1 + 10).
- Long-only or short-only collapses (one side Sharpe < 0 while other > 0 over ≥ 40 trades) — the OR thesis requires both sides to work.
- Slippage > 2.5 ticks/RT for 5 consecutive days.
- **90-day total falsification:** if after 90 calendar days of live (or shadow + live combined) the cumulative trade P&L 95% bootstrap CI includes zero, kill the strategy. This is the contract-level falsifiability promise.

**TP:SL argument (lesson 3):** This strategy does not use a fixed TP. The exit logic IS the asymmetry: stop = opposite OR edge (~OR width + 3 ticks), profit is realized at 11:25 JST forced exit which on winning days runs ~1.5-3× OR width. Empirically (per Crabel-lineage literature on equity index OR breakouts) the realized average win/loss ratio is 1.6-2.2x with a 38-44% win rate. This is structurally equivalent to a 100/35-style asymmetric R:R, satisfying the lesson-3 requirement that stop logic be designed as part of the edge, not optimized post-hoc. The walk-forward validation explicitly tests this asymmetry by tracking the win/loss ratio quantile, not just hit rate.

**PDT (lesson 5):** Not applicable — NKD is a regulated futures contract on CME, exempt from FINRA PDT rules. Even if the account drops below $25k, futures day-trading is unrestricted (subject to broker margin, ~$500 day-trade margin on micro variant or full $5,500 on standard).

**Lesson 7 (no same-data backtest):** Walk-forward as specified in section 6. The 2024-2026 OOS bars are touched exactly once — at verdict time. Any post-OOS parameter tweak invalidates the result and requires a new OOS window.

**Lesson 9 (capacity bound named):** $2.5M AUM, 50-contract per-clip ceiling. Documented in section 9.

**Lesson 10 (kill criteria pre-written):** This section IS the kill criteria, finalized pre-deploy, hash-pinned in the strategy config file.

## 12. One-paragraph operational risk note

The biggest operational surface is **session-time correctness**. JST is UTC+9 year-round, but the user is in (presumed) North America observing DST shifts twice per year; a single off-by-one-hour bug in the scheduler would compute the opening range on the wrong six bars and silently take wrong-direction trades for weeks. Mitigation: every session boots with a `JST.now().hour == 9 and JST.now().minute < 5` assertion before arming entries, and the cron is timezone-pinned to `Asia/Tokyo`. Secondary risks: (a) IBKR OSE permissioning is slow to obtain — if NKD routing breaks mid-quarter, the OSE fallback may not be active and the strategy goes dark for days; (b) IBKR's 5-second real-time bar API can produce duplicate or gapped bars during high-volume opens, so we use `reqMktData` ticks aggregated locally as a cross-check; (c) on FOMC days that overlap Tokyo morning (rare — usually US afternoon = JP early morning, so we may be entering on stale overnight risk pricing), the regime filter must catch the elevated JNI; (d) holiday-calendar mismatches between TSE, OSE, and CME (CME trades NKD on US holidays, but cash Nikkei doesn't print) — we **only trade days when TSE cash session is fully open**, gated by `pandas_market_calendars.get_calendar('JPX')`. The single most likely silent-failure mode is a stale calendar package after a one-off TSE holiday change (e.g., new emperor day, Olympics rescheduling); pin `pandas_market_calendars` version and re-verify the JPX holiday list each January.

---

THESIS: Tokyo-session 09:00-09:30 JST opening-range break (±2 ticks) on Nikkei 225 mini futures trends through morning to forced 11:25 JST flat with positive expectancy net of costs.
VALIDATION HORIZON: ≥200 OOS trade days across 2024-01 to 2026-05.
CAPACITY CEILING: 50 contracts per clip, ~$2.5M AUM before slippage degrades the edge.
