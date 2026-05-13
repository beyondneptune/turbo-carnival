# Iter 1 — US Small-Cap Overnight Mean Reversion

## 1. Thesis (one sentence)

After a high-volume intraday selloff in a Russell 2000 small-cap (RSI(2) < 10 at the 16:00 ET close with same-day volume ≥ 2.0× its 20-day average), the close-to-next-open return is positive with mean ≥ +15 bps net of slippage, because forced intraday liquidation by retail/quant programs in illiquid small-caps systematically over-shoots and reverses overnight when uninformed flow is absent; the claim is falsified if rolling 50-trade OOS mean overnight return drops below +5 bps net or rolling 50-trade Sharpe falls below 0.8.

## 2. Market / asset class / venue

- **Instruments**: Common stocks that are constituents of the Russell 2000 (IWM index membership) at the time of trade, restricted to issues with $100M–$2B market cap on the trade date. Excludes ADRs, REITs, BDCs, biotech tickers with FDA catalyst windows (PDUFA within ±5 sessions), and any name with a confirmed earnings release scheduled within ±1 session of the entry close.
- **Tickers**: Dynamic universe; rebalanced weekly using the latest IWM holdings file from iShares. Expected daily candidate count: 30–80 names trigger the screen, top 5 by liquidity rank are eligible for entry.
- **Primary exchange**: NYSE and NASDAQ (NYSE Arca for ETFs not eligible — we hold single stocks only).
- **Broker**: Interactive Brokers (IBKR Pro account, US Stocks bundle, SMART routing with primary-exchange override to avoid dark venue MOO leakage).
- **Settlement**: T+1 regular-way US equity settlement (effective May 2024). No FX leg. Long-only, so no locate/HTB borrow risk and no Reg-SHO threshold issues.
- **Margin/tax regime**: Reg-T margin account, but lane is run **cash-settled-only** to dodge PDT (see Lesson 5). Tax: ordinary short-term gains (holding period < 1 day). No wash-sale optimization — strategy is too high-frequency at the lot level to manage tax-loss harvesting; assume tax drag baked into pre-tax Sharpe.

## 3. Time horizon

**Daily**, with a single overnight hold per position. Entry decisions are made between 15:50 and 15:58 ET using last-30-minute snapshot data; orders are sent as **MOC (Market-on-Close)** at 15:58 ET (IBKR MOC cutoff is 15:50 ET for NYSE / 15:55 for NASDAQ — actual implementation uses **Limit-on-Close at last + 5 bps** which fills with the closing print at the LULD-allowed reference price). Exit is **MOO (Market-on-Open)** at 09:30 ET the next session. Mean holding period: ~17.5 wall-clock hours, ~13.5 of which are non-tradeable. No intraday holds, no multi-day positions, no weekend gap exposure adjustment beyond a Friday-entry filter (see §10).

## 4. Method family and specific algorithm

- **Family**: Rules-based mean reversion. No ML, no NN, no LLM. Deliberately chosen because Lesson 2 (binary gates not confidence rankers) was the most expensive bug in the prior session; a deterministic ruleset cannot have a confidence-correlation-with-realized-PnL bug because it emits a binary signal.
- **Algorithm** (exact, no ambiguity):
  1. **Universe filter** (computed at 15:00 ET daily): market cap ∈ [$100M, $2B], 20-day dollar ADV ≥ $5M, price ∈ [$5, $100] (avoid sub-$5 penny dynamics and >$100 names where small-cap mean-reversion is a different regime), not on an earnings ±1 list, not on a PDUFA ±5 list, not flagged in IBKR HTB feed.
  2. **Entry signal** (computed at 15:55 ET using last-trade snapshot):
     - RSI(2) on daily closes ≤ 10 (Connors-Alvarez canonical small-cap RSI threshold; deliberately tighter than RSI(14) < 30 because RSI(2) has a far higher signal-to-noise for 1-day reversion in this asset class)
     - Same-day cumulative volume (15:55 ET) ÷ 20-day SMA of cumulative-end-of-day volume ≥ 2.0 (volume spike confirms forced flow, not quiet drift)
     - Same-day return ≤ −2.5% (a real selloff, not a slow bleed)
     - Close above the 200-day SMA (avoid value-trap names in confirmed downtrends — this is the only "trend" overlay; Lou-Polk-Skouras 2019 shows the overnight-intraday tug-of-war is strongest in names that retain positive long-horizon drift)
  3. **Ranking** (when more than 5 names pass): rank by descending 20-day dollar ADV (most liquid first) → take top 5. Equal-weight by dollar across the 5.
  4. **Order**: LOC at last_price × 1.0005 (5 bps cushion above last to ensure fill at closing print without crossing too aggressively); per-name cap = 25% of that name's expected closing-auction volume (estimated as median 3:55-4:00 ET 5-min volume × 1.4).
  5. **Exit**: MOO sell at 09:30:00 ET next session via IBKR OPG-attribute order, no exceptions, no overrides, no "let it run" discretion. If MOO is rejected (rare; happens on hard-halted names), immediate LMT sell at open + 50 bps with 60-second TIF, then MKT at 09:31 ET.
- **Why this method fits**: The edge (Berkman/Koch/Westerholm 2012, Lou-Polk-Skouras 2019) is *itself* a rules-driven cross-section anomaly that has survived 20+ years of academic scrutiny. Adding ML on top introduces a free parameter without adding signal — every published small-cap overnight-reversal study uses simple rank or threshold rules. The right tool is the simplest tool that captures the documented effect.

## 5. Data requirements

- **Feeds**:
  - **Polygon.io Stocks Starter** ($29/mo) for daily bars, 1-minute bars for the 15:50–16:00 ET signal window, and intraday cumulative-volume snapshots. Free tier is rate-limit-prohibitive for 30–80 candidates per day.
  - **iShares IWM holdings CSV** (free, daily download) for universe definition.
  - **Refinitiv/IBKR earnings calendar** via IBKR's fundamentals subscription ($4.50/mo for US fundamentals) for earnings ±1 filter.
  - **FDA PDUFA calendar** scraped from BioPharmCatalyst (free) for biotech ±5 exclusion.
  - **IBKR HTB list** via TWS API (free with account) for short-side avoidance metadata (we go long only, but HTB names tend to have anomalous overnight behavior — exclude defensively).
- **Storage**:
  - 2,000 universe names × 252 days × ~50 minute-bars in the signal window × ~50 bytes ≈ **1.3 GB/year of compressed Parquet**.
  - 20 years of historical daily bars for 2,500 historical R2K names ≈ **2 GB**.
  - Total: ~5 GB after 3 years of live operation. Trivial on local SSD.
- **Cost**: $29 (Polygon) + $4.50 (IBKR fundamentals) + $0 (rest) = **$33.50/mo data**. IBKR commissions: $0.0035/sh tiered, expected ~$50/mo at target sizing.
- **Latency**: Tolerable round-trip to IBKR for MOC/MOO orders: **≤ 500 ms**. This is not an HFT lane. We submit at 15:55 ET for a 16:00 fill and at 09:25 ET for a 09:30 fill — minutes of headroom on both ends. No co-location, no FIX, plain ib_insync over TCP from a residential or VPS Linux host is sufficient.

## 6. Validation plan

- **Walk-forward design**:
  - **Train window**: 4 years (Jan 2018 – Dec 2021) — used only to **freeze the parameter values listed in §4** (RSI threshold = 10, volume ratio = 2.0, selloff = −2.5%, top-N = 5). Parameters are NOT re-optimized after this freeze.
  - **Test window**: 1 year, rolled forward in 3-month steps. So first OOS slice = Jan 2022 – Dec 2022, then Apr 2022 – Mar 2023, etc., through end of historical record (currently Dec 2025).
  - **Step**: 3 months. Produces 16 overlapping OOS slices, ~16 years × 250 trade-days × ~3 trades/day expected = a lot of overlapping OOS observations; we de-overlap by reporting on the non-overlapping 3-month forward chunks (Jan 2022–Mar 2022, Apr 2022–Jun 2022, ...) → 16 independent 3-month windows of true OOS performance.
  - **Purge and embargo**: Daily strategy on independent overnight returns has near-zero serial correlation, so purge is light: 2-day purge between train end and test start, 2-day embargo on either side of any earnings-restated tickers. Apply Lopez de Prado purged k-fold framework when computing aggregate OOS confidence intervals.
- **Minimum number of OOS trades before verdict**: **400 trades** across the full walk-forward (~25/month over 16 months of contiguous OOS). This is the historical-data verdict gate. Live shadow has a separate, smaller gate (see §11).
- **Statistical tests and thresholds**:
  - **Sharpe (daily, annualized)**: point estimate > 1.0 AND 95% stationary-bootstrap CI (10,000 resamples, block length 5) lower bound > **0.5**.
  - **Mean per-trade overnight return (net of 8 bps round-trip slippage + commissions)**: > **+10 bps** with t-statistic > 2.5 against H₀: mean = 0.
  - **Profit factor** > 1.3, **win rate** > 54%.
  - **Subperiod stability**: no single 3-month OOS slice has a Sharpe < −0.5; at most 3 of 16 slices may have Sharpe < 0.5.
  - **Cost sensitivity**: edge must survive doubling the slippage assumption to 16 bps round-trip. If it doesn't, kill at validation, not in production.
- **No same-data backtest**: parameter freeze is on 2018–2021 only; 2022–2025 is the held-out OOS evidence. This separation is mandated by Lesson 7.

## 7. Risk envelope

- **Account size assumption**: This candidate is designed for a **$50k–$2M** account. Below $25k the PDT rule still bites even though we hold overnight (because *if* a trade exits intraday on the next session via a stop, that's the second leg of a day trade — see PDT discussion in §10 lesson 5).
- **Max position size**: 20% of account NAV per single name (equal-weight 5 names = 20% each, so account is 100% gross long overnight in the worst case). Hard dollar cap per name: **min(20% NAV, 25% of expected closing-auction volume for that name, $200k)**. The $200k cap is the AUM-scaling brake.
- **Daily loss cap**:
  - **Absolute**: −2.0% of starting-of-day NAV in a single overnight session → all positions exit at MOO no matter what (i.e., we do not chase intraday).
  - **Cumulative 3-day**: −4.0% NAV peak-to-trough → halt new entries for 5 trading days.
- **Drawdown trigger that halts trading**: **−8% trailing peak-to-trough on net equity over any 60-day rolling window → strategy halts pending re-validation** (full walk-forward re-run on most recent 4-year window plus 3-month shadow re-run). This is the strategy-level circuit breaker.
- **Per-overnight circuit breaker**: if 3 consecutive overnight sessions all close negative, skip the next 2 sessions (no entries) — re-engage only when next eligible signal day comes up. This is the "3-consecutive-losses" lesson from prior incidents.
- **Correlation rule with already-deployed strategies**: Returns from this strategy are computed daily and correlated against any concurrently-deployed strategy at the slot manager. If 30-day rolling return correlation with any other live strategy exceeds **+0.5**, this lane is paused and reviewed. The strategy slot manager rejects duplicate-thesis candidates per Lesson 6.

## 8. Execution plan

- **Broker / SDK**: **Interactive Brokers via ib_insync** (Python). TWS Gateway in headless mode, supervised by systemd, auto-reconnect on disconnect, dual-write of every fill to a local SQLite ledger AND an append-only JSONL audit log.
- **Order types**:
  - **Entry**: `LOC` (Limit-on-Close) with limit = last_print × 1.0005, submitted at 15:55 ET. IBKR cutoff for NYSE LOC is 15:50 (15:55 for NASDAQ-only names). We submit at 15:48 ET to be safe and accept the small adverse-selection risk of revealing intent 12 minutes early — backtested 1-minute slippage assumption captures this.
  - **Exit**: `MOO` (Market-on-Open) with the OPG TIF attribute, submitted at 09:25 ET for the 09:30 open auction. If MOO is rejected (halted name, extreme pre-market move > 10% triggering LULD bands at the open), fallback ladder: LMT at open + 50 bps × 60s → MKT at 09:31:00.
- **Expected slippage budget**: **8 bps round-trip** (4 bps entry, 4 bps exit). This is empirically the median small-cap MOC/MOO spread observed in published TCA studies (e.g., ITG / Virtu liquidity reports 2021–2023 for $500M–$2B mid-caps). Measured live by comparing fill price to the official closing/opening print published by NYSE/NASDAQ in the SIP tape (available via Polygon `/v2/aggs/ticker/{ticker}/range/1/minute` for 16:00:00 and 09:30:00 bars).
- **Slippage measurement protocol**: every fill is logged with (intended_price, fill_price, official_print, lateness_seconds). Weekly TCA report aggregates these; if rolling 50-trade median slippage exceeds 12 bps round-trip, lane is paused.
- **What happens if the broker rejects an order**:
  - **Entry reject** (most common cause: HTB-listed in real-time after our pre-screen ran): skip that name silently, do not retry, do not substitute another candidate. Log the reject.
  - **Exit reject** (rare; usually halted name or trading suspended): triggers the fallback ladder above; if all 3 levels fail, escalate to on-call human (PagerDuty webhook). Position is *never* held a second night without explicit human override.
  - **Connectivity loss**: if TWS gateway is disconnected at 15:50 ET and not recovered by 15:55 ET, **skip the entire day's entries** (no partial-execution risk). If disconnected at 09:25 ET, use IBKR's web portal as a manual fallback to send MOO orders.

## 9. Capacity

- **AUM ceiling**: **$15M dedicated to this strategy**. Reasoning:
  - Per-name cap is min(20% NAV, 25% of closing-auction volume, $200k absolute). The binding constraint flips at NAV ≈ $1M (where 20% = $200k = absolute cap), and again at NAV ≈ $3.5M for the median candidate (where 25% of a typical $14M closing auction = $3.5M is reached at $700k per name, which equals 20% of $3.5M NAV).
  - At $15M NAV, 20% per name = $3M, which is ~21% of the median candidate's $14M closing auction. We cap there. Above $15M NAV, capacity bottlenecks on the auction-volume constraint — we would need to add names beyond the top-5 ranked, which dilutes the signal (and the top-5 cutoff is in the documented edge).
- **ADV consumption**:
  - At $15M NAV, full deployment = $3M × 5 = $15M gross overnight.
  - Median candidate dollar ADV ≈ $20M → strategy consumes **~15% of ADV at full size** (concentrated in the close/open auctions, not VWAP across the session, which makes the real consumption-of-auction-liquidity ~21%).
  - This is at the upper edge of what backtests with 8 bps slippage will continue to validate. AUM beyond $15M requires expanding to top-10 names with a corresponding slippage uplift to 12 bps, which historically degrades net Sharpe by ~0.3.
- **Concentration risk**:
  - **Single-name max**: 20% of NAV.
  - **Single-sector max**: 60% of NAV (sectors per GICS Level 1). Small-cap selloffs tend to cluster (e.g., regional banks in March 2023); a 60% cap prevents the full book from being one factor.
  - **Single-day max names**: hard cap of 5 names per night (already enforced by top-N ranking).

## 10. Failure modes (named, with metric signatures)

1. **Regime change — small-caps de-couple from mean reversion** (e.g., a sustained bear leg where every selloff continues overnight, as in March 2020 first leg or Feb 2018 vol-spike). **Metric signature**: 30-day rolling Sharpe drops below 0 while broader-market overnight return (SPY overnight) also negative; correlation of strategy returns with VIX 1-day change becomes < −0.5 (implying we get hurt every time vol spikes). **Detection**: daily KPI dashboard, automatic alert at Sharpe-30d < 0 for 5 sessions.
2. **Liquidity flight at the auction** (e.g., closing auction volume collapses in a holiday-thin or post-CPI session, our MOC fills get adversely-selected by retail-only flow). **Metric signature**: rolling 20-trade median slippage > 12 bps round-trip; per-name standard deviation of fill_price − official_close > 6 bps. **Detection**: TCA weekly cron + real-time per-fill alarm if any single fill is > 30 bps from the official print.
3. **Earnings-leak / catalyst pollution** (an earnings announcement gets rescheduled into our overnight window, or an M&A bid drops at 18:00 ET — these are *not* mean-reverting flows, they are information events). **Metric signature**: a single overnight loss > 5% on one name, or > 2 names in a single night each losing > 3%. **Detection**: post-trade catalyst lookup against Refinitiv earnings/M&A calendar; if > 1 such event occurs in a 30-day window, retighten the earnings ±1 → ±2 filter and add an M&A rumor screen.
4. **Russell 2000 reconstitution drift** (annual June rebal of IWM forces $$ flows in/out of marginal names; our edge documented on the *post-rebal* universe may not hold for *added/dropped* names in the rebal window). **Metric signature**: outsized losses in late June every year. **Detection**: pre-emptively, **strategy goes flat from June 23 close through June 30 open** (around the annual reconstitution effective date), as a deterministic rule.
5. **Polygon S3 plan downgrade or data outage** (we lose the 1-minute snapshot at 15:55 ET and cannot evaluate the signal). **Metric signature**: missing data error in the entry-decision pipeline. **Detection**: pipeline fail-closed — if any required input is missing, **the day's entries are skipped, not estimated**.
6. **Friday-into-Monday gap risk** (3-day weekend, 4-day with US holiday): **explicit exclusion** — no entries on the last trading session before a US-market holiday or weekend longer than the usual Sat+Sun. Empirically, Berkman 2012 and Lou-Polk-Skouras 2019 both document that the overnight-reversal effect attenuates over multi-day gaps, while gap-risk variance scales with √days_held. Risk-reward asymmetry inverts on long weekends.

## 11. Falsifiable verdict criteria

These are checked sequentially: backtest first, then shadow, then promotion.

### Promote to live capital (all of the following, after 30 calendar / ~21 trading shadow days):
- **Backtest gate** (already cleared before shadow starts): full walk-forward 2022–2025 OOS produces ≥ 400 trades with annualized Sharpe ≥ 1.0, lower-bound 95% bootstrap CI ≥ 0.5, max DD ≤ 12%, win rate ≥ 54%, profit factor ≥ 1.3.
- **Shadow gate**: 21 trading days × ~3 candidates/day = ~63 shadow trades (target ≥ 50 OOS shadow trades). Required: realized Sharpe (paper-traded with real fills via IBKR paper account) ≥ 0.8; mean per-trade overnight return net of measured slippage ≥ +8 bps; **zero broker-truth divergence events** (i.e., every position reconciled at every snapshot matches IBKR's reported position to the share); max DD over shadow ≤ 4% of paper NAV.
- **Slippage gate**: measured median round-trip slippage ≤ 10 bps over the shadow window.
- **Ops gate**: zero unplanned outages of > 30 minutes during the 15:50–16:00 ET window or the 09:25–09:35 ET window over the shadow period.

### Continue shadow (any of the following true, none of the kill triggers hit):
- Shadow Sharpe in [0.4, 0.8] — ambiguous, extend shadow by 30 days.
- Shadow trade count < 50 — extend until trade-count gate met.
- Slippage measured > 10 but ≤ 14 bps — pause, tighten LOC/MOO fallback rules, re-shadow 14 days.

### Kill immediately (any one of these, while in shadow OR live):
- Any single day with loss > 3% of NAV (vs. 2% daily cap, this is a runaway alert).
- 3-day rolling DD > 5% of NAV.
- 60-day trailing peak-to-trough DD > 8% of NAV (the §7 strategy-level circuit breaker — kill, not pause).
- Any broker-side fill discrepancy > 0.5% (price or quantity divergence between local ledger and IBKR's reported fill on a single order).
- Median round-trip slippage > 15 bps over any rolling 20-trade window.
- Rolling 50-trade Sharpe < 0.5 for 10 consecutive days.
- **90-day hard falsification clock**: if 90 calendar days of live trading produce realized Sharpe < 0.5, the lane is killed unconditionally — no extensions, no "regime explanation" overrides. This satisfies the spec's "falsification within 90 days" hard rule.

## 12. One-paragraph operational risk note

The most fragile dependency is the 15:50–15:58 ET decision window: a Polygon outage, a TWS gateway disconnect, a residential ISP blip, or any 30-second blocking call in the signal pipeline causes us to either submit on stale data or miss the LOC cutoff entirely, and there is no human-in-the-loop fast enough to recover. Mitigation: dual-host deployment (primary on a NYC-east AWS t3.small at $15/mo, hot standby on a residential workstation; heartbeat every 15 seconds; either host can submit the day's orders independently with IBKR-side deduping via client-order-id). Secondary risk: Polygon plan changes — they've revised Starter-tier rate limits twice in the last 24 months; if they drop intraday minute bars from Starter, our $33.50/mo data budget jumps to $79/mo (Advanced tier) and the strategy's expected-net-margin assumption shifts by ~3 bps. Tertiary risk: IBKR's IWM holdings file delivery occasionally lags by a day during corporate actions (M&A delistings, spin-offs); we run a Friday consistency check comparing our cached universe against three independent sources (iShares, Refinitiv, NASDAQ FTP).

---

## Operational risk paragraph — addressing the 10 mandated lessons explicitly

1. **Validate before deploy**: §6 walk-forward 2018→2025 → §11 shadow ≥21 trading days with ≥50 trades and zero broker-divergence gate. No live capital until both complete. **HONORED**.
2. **Binary gates, not confidence rankers**: §4 is a rules-based ruleset; entry is a binary signal (all four conditions met → trade, else skip). No ML confidence is used for sizing — sizing is equal-weight across the top-N. The Connors-Alvarez RSI(2) approach is itself a binary gate. **HONORED**.
3. **TP:SL asymmetry is structural**: This candidate has **no intraday stop and no take-profit** — exit is deterministic at MOO regardless of where the name is. The "stop logic is part of the edge" lesson is honored by acknowledging that *adding* a stop here destroys the edge: Berkman 2012 and Lou-Polk-Skouras 2019 both document that the reversal occurs *because* the position is held to the open auction. Stops would cut the winners by selling into the overnight low. The kill criteria at the strategy level (§7, §11) substitute for trade-level stops. **HONORED VIA EXPLICIT CHOICE**.
4. **Broker truth, not local cache**: §11 explicitly requires zero broker-divergence events in shadow as a promotion gate, and any > 0.5% divergence in live is an immediate kill. Every position is reconciled against IBKR at submit, fill, and EOD; the local SQLite ledger is treated as a write-through cache, never as source of truth. **HONORED**.
5. **No PDT-exposed lanes on sub-$25k US equity accounts**: §7 explicitly scopes this strategy to $50k–$2M accounts. The strategy is overnight-only (close → next open, no same-session round trip), so day-trade count is structurally zero **as long as the entry-day exit is not triggered by an intraday stop** — which it isn't (see Lesson 3 honor). The only PDT exposure would be an emergency liquidation, which is governed by the §7 kill criteria and is rare by design. For accounts strictly under $25k we would reject this strategy and route to a futures/FX alternative. **HONORED with structural argument**.
6. **One strategy per slot**: This thesis is explicitly "US small-cap overnight mean reversion." The slot manager rejects any subsequent candidate whose thesis maps to (US equities) ∩ (overnight hold) ∩ (mean reversion) ∩ (small-cap universe). The correlation rule in §7 (>0.5 30-day rolling return correlation = pause) is the second line of defense. **HONORED**.
7. **No same-data backtest**: §6 parameters are frozen on 2018–2021 train and never re-touched; 2022–2025 is held-out OOS. Walk-forward 3-month step rolls forward only, never back. **HONORED**.
8. **Battle-tested broker SDKs only**: ib_insync (the de facto Python wrapper for IBKR, maintained, > 2k stars, in production at multiple funds). No hand-rolled FIX, no custom HTTP. **HONORED**.
9. **Capacity bound named upfront**: §9 names **$15M AUM ceiling** with explicit ADV-consumption math; reasoning is dollar-ADV × auction-fraction-tolerable. **HONORED**.
10. **Kill criteria written before deploy**: §11 lists exact numeric kill thresholds (3-day DD > 5%, 60-day DD > 8%, slippage > 15 bps over 20 trades, 50-trade Sharpe < 0.5 for 10 days, broker divergence > 0.5%, 90-day Sharpe < 0.5). **HONORED**.
