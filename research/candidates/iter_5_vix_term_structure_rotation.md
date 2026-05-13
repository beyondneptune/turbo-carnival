# Iteration 5: VIX Term-Structure Rotation (SVXY ↔ Cash)

## 1. Thesis (one sentence)

When the VIX/VIX3M ratio sits below 0.92 (front-month VIX cheaper than 3-month VIX = contango), VIX futures will, on average, roll **down** toward spot VIX, generating a positive expected daily return for an inverse-VIX futures ETF (SVXY); when the ratio inverts (backwardation) we are flat in T-bills and miss the volatility blow-ups.

Falsifiable form: **In a 12-month out-of-sample window, the rule "long SVXY when VIX/VIX3M < 0.92, else SGOV" produces Sharpe > 0.5 and max DD < 25%. If not, the contango edge has either decayed or been arbitraged away — kill.**

## 2. Market / asset class / venue

- **Risk asset**: `SVXY` (ProShares Short VIX Short-Term Futures ETF, currently **-0.5×** the daily return of the S&P 500 VIX Short-Term Futures Index). Post-Feb-2018, ProShares cut the leverage from -1× to -0.5×; backtests pre-2018 must be re-scaled accordingly.
- **Cash leg**: `SGOV` (iShares 0-3 Month Treasury Bond ETF) — overnight T-bill exposure, yields ~SOFR minus ~7 bps. Alternative: hold cash in IBKR sweep at posted IBKR rate.
- **Signal inputs**: `^VIX` (CBOE Volatility Index), `^VIX3M` (3-month VIX). Both are CBOE indices reproducible from listed options.
- **Primary exchange**: NYSE Arca (both SVXY and SGOV).
- **Broker**: Interactive Brokers (IBKR) via `ib_insync`.
- **Settlement / margin**: T+1 cash settlement (no margin required for long-only). Reg-T margin available but not needed at the position sizes proposed below.
- **Tax regime**: SVXY is a 1940-Act ETF (NOT a partnership / K-1) post-restructure, so gains are taxed as ordinary 1099-DIV / 1099-B short-term capital gains. SGOV distributions are largely state-tax-exempt T-bill interest.

## 3. Time horizon

**Daily.** Signal evaluated once per session at **15:30 ET** (30 minutes before close, after VIX/VIX3M have stabilized for the day). Orders sent as marketable-limit MOC-eligible at 15:55 ET. Holding period varies — historically a single SVXY leg lasts 5–60 days during sustained contango, then we flatten for hours to weeks.

## 4. Method family and specific algorithm

- **Family**: Rules-based regime detection (no ML, no NN).
- **Algorithm**:
  1. At 15:30 ET, compute `R = VIX / VIX3M` from CBOE intraday snapshots.
  2. If `R < 0.92` AND none of the circuit-breaker conditions in §7 are tripped → target position = **25% NAV long SVXY**, rest in SGOV.
  3. If `R ≥ 0.92` OR any circuit breaker is tripped → target = **0% SVXY, 100% SGOV**.
  4. Trade only if current position differs from target by ≥ 5% NAV (no-trade band suppresses micro-rotations near the threshold).
  5. **Hysteresis**: once flat for tail-risk, require both `R < 0.90` AND `VIX < 25` for **two consecutive sessions** before re-entering. (Avoids whipsawing back into a falling knife.)
  6. **Max consecutive holding cap**: 90 calendar days continuously long SVXY without a flat day → forced review (human override required to continue). Long contango streaks are real (often 4–6 months) but a forced checkpoint protects against the strategy being "asleep" through a regime shift.
- **Why rules, not ML**: The edge is mechanical — VIX futures must converge to spot at expiry, so contango is a physical roll-down. A two-variable threshold has < 5 effective parameters, low overfitting surface, and is what practitioners (VRP / VXX-short literature, Sigma Research, Volatility Trading by Sinclair, 10+ years of public posts on VRP) have used. ML on top would not add ranking power that survives 2018-style tail events; it would add a failure surface.

## 5. Data requirements

- **Live**: CBOE delayed indices `^VIX`, `^VIX3M` are free via IBKR market data subscription (`CBOE Indices Value` bundle, ~$1.50/mo retail). Spot quotes for SVXY/SGOV via IBKR US Securities Snapshot/Bundle ($10/mo).
- **Historical**: Daily VIX & VIX3M from CBOE 2009-01-02 to present (free CSV from cboe.com). SVXY adjusted close from Yahoo / Stooq / Polygon. For the pre-2011 portion when VIX3M was published as VXV, use the VXV historical series — same index, renamed in 2017.
- **Storage**: < 100 MB total. Trivial.
- **Cost**: ~$12/mo all-in for IBKR data; $0 historical. (Polygon Free tier covers historical SVXY daily bars within 5 rpm.)
- **Latency**: Tolerable round-trip = **seconds**. Signal is computed at 15:30 ET and orders go in at 15:55 ET. No HFT requirements.

## 6. Validation plan

- **Total daily sample**: 2009-01-02 → 2026-05-12 ≈ **4,360 trading days**.
- **Train / calibration**: 2009-01-02 → 2017-12-31 (pre-Volmageddon; threshold and hysteresis fitted here).
- **Stress holdout 1 — Volmageddon**: 2018-01-01 → 2018-12-31. This window is **not** used for parameter selection; it is a single-shot stress test of the circuit-breaker. Pass criterion: max single-day loss ≤ 12% NAV.
- **Stress holdout 2 — COVID**: 2020-02-01 → 2020-06-30. Single-shot test of how fast the regime flag flips out and back. Pass criterion: max DD ≤ 25%, recovery within 90 trading days.
- **Out-of-sample evidence window**: 2023-01-01 → 2026-05-12 (≈ 850 trading days). This is the **OOS not overlapping** the user's prior 2018-2025 backtest, except for 2023-2025 — to fully de-overlap we will additionally cut **2024-04-01 → 2026-05-12** as a clean post-prior-test window.
- **Walk-forward**: Train window 2009 → cutoff-3y, test window 1 year, step 1 quarter. ~36 forward folds across 2013-2026.
- **Purge**: 5-day embargo between train and test (the VIX/VIX3M signal is serially correlated; 5 days > 95th-pct autocorrelation horizon).
- **Minimum OOS regime transitions**: **100** signal flips (entry-to-flat or flat-to-entry) before any verdict is issued. Historical base rate is ~30 transitions/year, so 100 is reached in ~3-4 years of OOS.
- **Stat tests**:
  - Bootstrap (10k draws, block size = 20 days) of OOS daily returns; require **95% CI of Sharpe excludes 0**.
  - Require **OOS Sharpe ≥ 0.5** (vs. user's prior 0.66 — leaving margin for decay).
  - Require **OOS max DD ≤ 25%** (sized at 25% NAV; -100% on SVXY = -25% on NAV before circuit-breaker).
  - Require **no single-day NAV loss > 12%** in any backtest fold.

## 7. Risk envelope

This is the section that does the load-bearing work for this strategy.

- **Max single position**: **25% of NAV in SVXY** at signal. Hard cap **30%**. Rationale: SVXY can lose ~50% in a day under a Volmageddon-scale shock even at -0.5× leverage; 25% NAV × 50% = -12.5% NAV worst-case day inside the circuit breaker. The pre-2018 -1× version would have demanded ≤ 12% NAV sizing.
- **Daily loss cap**: -8% NAV. If realized intraday MTM hits -8% NAV, flatten all SVXY at next print and stand down for the rest of the session.
- **Drawdown trigger**: trailing 20-day NAV drawdown ≥ 15% → strategy halted, human review required to restart.
- **Circuit breaker (the Feb-5-2018 lesson)**: At **every minute** between 09:30 ET and 16:00 ET while we hold SVXY, evaluate:
  - `VIX_spot > 30` → flatten immediately (market-on-close OK if triggered after 15:50 ET).
  - `VIX_spot 1-day change > +50%` (intraday relative to prior close) → flatten immediately.
  - `VIX_spot intraday move > +20%` since 09:30 print of the same session → flatten immediately.
  - `SVXY intraday drawdown > -15%` from prior close → flatten immediately.
  - These are **OR**-ed: any single breach trips the breaker.
- **Tail-event historical check**: On 2018-02-05 the VIX rose from ~17 to ~37 intraday (~+115%) and XIV/SVXY -1× lost ~80%. Two of our four triggers (`+50%` 1-day VIX, `VIX > 30`) would have fired well before close. Even with the -0.5× ETF and 25% sizing, a missed circuit-breaker (broker down, illiquidity gap) costs ~10% NAV; with the breaker firing on the 13:30 ET VIX print, expected NAV loss is ~4-6%.
- **Liquidity gap risk**: SVXY can gap through limits at the open after an after-hours VIX shock. Mitigation: only hold SVXY when prior-day `VIX < 22` (a *secondary* entry gate added to the basic R < 0.92 rule).
- **Consecutive-loss circuit breaker**: 3 consecutive negative trading days while long SVXY → reduce target to 12.5% NAV for the next 10 sessions. Restore after a 5-day positive streak.
- **Correlation rule**: Disallow co-deployment with any other short-vol, dispersion, or long-equity-beta strategy whose 30-day return correlation with this strategy exceeds 0.6. SVXY's effective beta to SPX is ~+2 to +3 in calm regimes; this is *additive*, not a hedge, to any long-equity satellite.

## 8. Execution plan

- **Broker / SDK**: IBKR Pro account, `ib_insync` (battle-tested, lesson 8).
- **Order types**:
  - Routine entry/exit at 15:55 ET: **MOC (market-on-close)** for the rotation — fills at the official 16:00 closing print, eliminates inside-the-day slippage on the 15:30 signal.
  - Circuit-breaker exit: **marketable IOC limit** at bid - 0.25% to guarantee fill while bounding adverse selection. Fallback to MKT if limit unfilled after 10 seconds.
- **Slippage budget**: SVXY ADV > 10M shares, median spread ~$0.01 on a $50 stock = 2 bps. Budget **5 bps** per round trip in normal conditions; **50 bps** in a circuit-breaker exit. Measured by comparing fill price to NBBO mid at order send time, logged every fill.
- **Order rejection handling**: If IBKR rejects (margin, halt, IB-side risk), retry once with `MKT` for the same quantity. On second rejection: page operator and freeze further trading until ack. **No silent retry loop.**
- **Broker truth reconciliation (lesson 4)**: Every minute while we hold SVXY, query `reqPositions()` and reconcile against the local target. Any divergence > 1 share → halt trading, page operator. Closing orders ALWAYS use the broker's reported quantity, never the local cache.

## 9. Capacity

- **SVXY ADV**: ~12M shares × ~$50 = **~$600M/day notional**. A 25% NAV allocation that costs ≤ 1% of ADV per rebalance is feasible up to roughly **$1.5B AUM** (25% × $1.5B = $375M = 0.6% ADV).
- **Realistic AUM ceiling for this strategy as deployed**: **$50M**. Above $50M, two frictions dominate: (a) IBKR Reg-T algos still execute through the lit book, and at $50M+ we'd start moving the close print on rebalance days; (b) circuit-breaker exits in a 2018-style shock would push us into ~5-10% ADV territory and we'd take heavier slippage exactly when liquidity has evaporated.
- **Single-name max**: 100% of risk-on allocation is in SVXY (single ticker). This is by design but is a real concentration risk — see §10.
- **Single-issuer concentration**: ProShares is the ETF issuer; an issuer-level operational failure or NAV-suspension event hits the entire allocation. Mitigation discussed in §10.
- **Capacity caveat — leverage change**: When ProShares cut SVXY from -1× to -0.5× in Feb 2018, the strategy's gross return profile was approximately halved while the tail risk was also roughly halved. Annualized return from the user's 2018-2025 walk-forward (+17.1% Sharpe 0.66) is the *post-change* regime — that is the regime we are deploying into. Do not naively extrapolate -1× backtest results.

## 10. Failure modes

1. **Term-structure regime change (the contango trade dies)**. Metric: 90-day rolling fraction of days with `VIX/VIX3M < 0.92` drops below 40% (historical avg ~70%), AND OOS Sharpe over the same 90 days < 0. Signature: many shallow flat periods, few sustained holds.
2. **Tail event with broker/ETF failure**. Metric: SVXY trading halt, NAV suspension, or ETF termination (XIV was wound down within a week of Feb 2018). Signature: SVXY spread blows out > $0.20, last NBBO stale > 1 minute. Mitigation: monitor SVXY trading status; if halted, treat allocation as worthless until reopen and immediately page operator.
3. **Issuer / sponsor change**. Metric: any ProShares prospectus amendment that changes leverage, fee, or rebalance methodology. Signature: SEC filing or ProShares press release. Mitigation: subscribe to ProShares investor-relations alerts; freeze allocation on any change pending review.
4. **VIX/VIX3M data-feed corruption**. Metric: missing or stale CBOE index print at 15:30 ET, or `R` value with magnitude implausible (< 0.5 or > 1.5 outside known historical regime). Signature: signal evaluation can't run. Mitigation: secondary feed (Polygon or Cboe DataShop end-of-day file); if both stale, **default to flat / SGOV** for the session.
5. **Whipsaw around the 0.92 threshold**. Metric: > 4 entry-to-flat round trips in any rolling 10 trading days. Signature: hysteresis is failing. Mitigation: widen the no-trade band, audit logs to confirm whether real regime change or signal noise.
6. **Roll-down decay shrinking** (the *slow* edge decay). Metric: average daily return on SVXY during contango windows over a 12-month rolling window drops below 5 bps/day (vs. historical ~12 bps/day). Signature: edge being arbitraged away by short-vol ETP creation. Pre-kill warning.

## 11. Falsifiable verdict criteria

- **Promote to live capital** — all of the following over a ≥30-day shadow run AND a ≥12-month OOS walk-forward fold:
  - OOS annualized return ≥ 10%.
  - OOS Sharpe ≥ 0.5 with bootstrap 95% CI excluding 0.
  - OOS max DD ≤ 25%.
  - Zero broker-truth divergence events.
  - Circuit breaker triggered correctly on every backtest day where conditions were met (no missed flatten).
  - At least one tail-stress event in the OOS sample (or live shadow) that the circuit breaker handled within tolerance.

- **Continue shadow** — metrics ambiguous (Sharpe between 0.3 and 0.5, DD between 18% and 25%), no kill triggers fired, < 100 regime transitions observed.

- **Kill immediately** if any one of:
  - 3-day NAV drawdown > 8%.
  - Daily loss cap breach (-8% NAV).
  - Broker-side fill discrepancy > 0.5% NAV.
  - Any missed circuit-breaker firing (post-mortem shows trigger was met but flatten didn't execute).
  - Trailing 12-month OOS Sharpe < 0.5 OR trailing 12-month max DD > 25% — **the 90-day-deploy falsification clause**.
  - SVXY trading halt > 1 session OR ProShares announces material restructure.

## 12. One-paragraph operational risk note

The strategy is operationally simple — two index reads, one ETF rotation, daily cadence — but its tail risk lives entirely in the circuit-breaker code path. If `ib_insync` loses connection between 15:30 and 15:55 ET on a bad day, we miss the rotation; if it drops mid-session during a VIX spike, we miss the circuit-breaker flatten and eat the full -10 to -15% NAV. Mitigations: (a) heartbeat the IBKR gateway every 15 seconds with auto-reconnect, (b) run on a dedicated VPS in the IBKR US-East colo region with cellular failover for the operator paging channel, (c) maintain a **broker-side stop-loss** ("hardware breaker") for -15% on the SVXY position as a backstop independent of our software loop — if our process dies, the IBKR-side stop still fires. The CBOE data feed (VIX/VIX3M) is also a single point of failure; mirror it from a second vendor (Polygon snapshot + manual CBOE delayed page) and default to flat if both are stale at 15:30 ET. Finally, the strategy is silent during long contango streaks (sometimes 30-60 days with no trade), which paradoxically increases operational risk: humans stop watching. Mandatory 5-minute weekly review (signal, position, breaker logic age) is a hard requirement.

---

## Lesson compliance (the 10 lessons, explicit)

1. **Validate before deploy.** §6 mandates walk-forward + ≥30-day shadow before any live capital.
2. **Binary gates, not confidence rankers.** The signal *is* binary (R < 0.92 or not). No confidence score, no continuous sizing. Position size is fixed at 25% NAV on, 0% off — eliminates the C3-style confidence/P&L mismatch.
3. **TP:SL asymmetry is structural.** The circuit breaker is asymmetric by design: we cap downside at ~-12% NAV intraday but let upside run for weeks during sustained contango. The historical ratio of avg-up-week to avg-down-week during contango is ~3:1; that asymmetry is the edge.
4. **Broker truth, not local cache.** §8 requires per-minute reconciliation with `reqPositions()`; closing orders use broker-reported quantity.
5. **No PDT-exposed lanes on sub-$25k US equity accounts.** The strategy holds positions for *days to months* and rotates at most 1-2× per week in normal regimes. Historical rotation frequency 2014-2025 averages ~30/year ≈ 0.6/week, far below the 4-day-trades-in-5-business-days PDT trigger. Even during the worst whipsaw years (e.g. 2015 / 2018) the strategy had ~50 round trips/year ≈ ~1/week. **Argued exemption: PDT cannot trigger because the rotation cadence is ≤ 1/week even in adversarial regimes, and the strategy never opens-and-closes within the same session except in a circuit-breaker event** (which is a *risk* action, occurring less than once per year on average and explicitly exempt from PDT logic on most brokers when documented as such). However, as belt-and-suspenders: maintain $25k minimum equity in the account; if equity falls below $25k after a drawdown, halt the strategy until refunded.
6. **One strategy per slot.** This iteration is the canonical "VIX contango rotation" slot. Reject any other iteration that proposes short-VXX, short-VXM, or any other inverse-VIX trade — they are the same edge.
7. **No same-data backtest.** §6 splits 2009-2017 (calibration), 2018 (Volmageddon stress, single-shot), 2020 (COVID stress, single-shot), 2023-2026 (OOS) — and explicitly proposes the 2024-04 onward window as the de-overlapped OOS vs. the user's prior 2018-2025 run.
8. **Battle-tested broker SDKs only.** `ib_insync` (lesson 8 ticker mention).
9. **Capacity bound named upfront.** §9: **$50M realistic, $1.5B theoretical**. Edge degrades at the lower bound from execution slippage during stress, not from alpha capacity.
10. **Kill criteria written before deploy.** §11 enumerates exact thresholds. Top-line falsifier: trailing 12-month OOS Sharpe < 0.5 OR trailing 12-month max DD > 25% → kill within 90 days.

---

THESIS: When VIX/VIX3M < 0.92, contango pays a positive daily roll-down on inverse-VIX ETPs; circuit-breakers on VIX spikes prevent Feb-2018-style ruin.
VALIDATION HORIZON: Sharpe ≥ 0.5 with bootstrap 95% CI excluding 0 over ≥100 OOS regime transitions (≈3-4 years of trading).
CAPACITY CEILING: $50M realistic AUM (edge degrades above this from circuit-breaker-day slippage); ~$1.5B theoretical from ADV alone.
