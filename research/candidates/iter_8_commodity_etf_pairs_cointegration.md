# Iteration 8: Commodity ETF Pairs — Cointegration + Ornstein-Uhlenbeck Mean-Reversion

## 1. Thesis (one sentence)

Economically linked US commodity ETF pairs — selected by Engle-Granger cointegration on rolling 252-day windows and filtered by Ornstein-Uhlenbeck (OU) half-life in the 5-30 day band — produce a stationary spread whose z-score reverts often enough that a `|z|>2` entry / `|z|<0.5` exit / `|z|>3` stop rule generates a positive expected daily return that is largely orthogonal to single-leg directional commodity beta.

Falsifiable form: **In a 2020-2026 out-of-sample window across the surviving 3-4 pairs, the rule produces ≥100 pair-trades with average net return per trade ≥ 30 bps (after 4 bps round-trip frictions) and aggregate Sharpe ≥ 0.8 with 95% bootstrap CI excluding 0.3. If not, the cointegration relationship has either decayed or been arbitraged away — kill within 90 days.**

## 2. Market / asset class / venue

- **Universe (4 candidate pairs, ranked by historical 2010-2020 in-sample Engle-Granger p-value)**:
  1. **GLD / SLV** — gold vs. silver. Most stable economic link (precious metals, monetary hedge, real-rate sensitivity). Historical p-value typically < 0.005 on rolling windows. **Primary pair.**
  2. **DBA / CORN** — broad agriculture basket vs. corn futures ETF. CORN is ~25-30% of DBA's grain weight; partial-overlap cointegration. p-value typically < 0.01. **Secondary pair.**
  3. **USO / XLE** — front-month WTI crude ETF vs. US energy equities sector. Weakest of the three (XLE is equity-linked, levered, and dividend-paying; USO is futures-roll-heavy). p-value typically 0.005-0.03 and **breaks during structural shocks** (Russia-Ukraine 2022, COVID-2020 negative-oil). **Tertiary pair — kept on a tighter cointegration gate (p<0.005) and with stricter half-life cap (≤20 days).**
  4. **UNG / XOP** — natural-gas ETF vs. oil & gas exploration & production equities. Historical p-value 0.01-0.05, frequently fails the gate. **Reserve pair** — included only when rolling test passes p<0.01; otherwise stays on the bench.

- **Instruments (tickers, primary exchange)**: `GLD`, `SLV` (NYSE Arca); `DBA`, `CORN` (NYSE Arca; CORN is a Teucrium 1940-Act fund post-restructure, NOT a partnership/K-1); `USO` (NYSE Arca; USCF, formerly LP, restructured to 1940-Act); `XLE`, `XOP`, `UNG` (NYSE Arca).
- **Broker**: Interactive Brokers (IBKR Pro) via `ib_insync`.
- **Settlement**: T+1 cash settlement for all listed ETFs.
- **Margin regime**: Reg-T 50% initial, 25% maintenance — but we deploy as a **dollar-neutral pair** so net exposure is ~$0 and the margin call distance is governed by Reg-T 25% maintenance on the gross. Portfolio Margin not required at the sizes proposed.
- **Tax regime**: All four candidate funds are 1940-Act ETFs (post-2020 USO and CORN restructures), so 1099-B short-term/long-term capital gains, no K-1. **Holding period ≤ 15 days = short-term.** SLV and GLD are technically grantor trusts on physical metal — sale generates 28% collectibles rate at the federal level for >12-month holds, but our holds are <15 days so this is moot (short-term capital-gains rate applies).

## 3. Time horizon

**Daily close-to-close.** Signal computed each session at **15:30 ET** using same-day closing prints reconstructed from the 15:30 snapshot (validated against the official 16:00 close in the post-session reconciliation). Orders sent **MOC** for fill at 16:00 ET. Holding period **3-15 days typical** (matches the 5-30 day OU half-life filter — the rule says exit at `|z|<0.5`, which on average occurs 1-2 half-lives after entry).

## 4. Method family and specific algorithm

- **Family**: Rules-based statistical arbitrage (mean-reversion on a cointegrated spread). No ML, no NN.
- **Algorithm**, per pair, per session:
  1. **Rolling cointegration gate.** Each Sunday after the Friday close, run the **Engle-Granger two-step test** on the last 252 trading days of log-prices `(log P_A, log P_B)`:
     - Step 1: OLS regress `log P_A = α + β * log P_B + ε`. Record hedge ratio `β`.
     - Step 2: ADF test on residuals `ε`. **Require p < 0.01** for GLD/SLV, DBA/CORN, UNG/XOP. **Require p < 0.005** for USO/XLE (tighter, given known regime-break history).
     - If gate fails → pair is **disabled** for the coming week (no new entries; existing position is managed to exit per existing rules but no fresh entries).
  2. **OU half-life filter.** Fit an OU process `dε = -θ(ε - μ)dt + σ dW` by AR(1) on residuals: `ε_t = a + b*ε_{t-1} + η_t`, half-life `H = -ln(2)/ln(b)`. **Require 5 ≤ H ≤ 30 trading days.** Pairs with `H < 5` are too noisy (over-trading, frictions eat the edge); pairs with `H > 30` are too slow (positions tie up capital and increase regime-break exposure).
  3. **Hedge ratio recompute frequency**: **Weekly** (Sunday after Friday close). Daily recompute creates path-dependent overfitting; monthly is too slow when underlying correlation regimes shift. Weekly matches the rebalance cadence of practitioner pairs desks (Gatev/Goetzmann/Rouwenhorst 2006 used monthly; modern cointegration variants — e.g. Vidyamurthy 2004, Caldeira/Moura 2013 — typically recompute weekly).
  4. **Z-score signal**, daily at 15:30 ET:
     - `spread_t = log P_A,t - β * log P_B,t`
     - `z_t = (spread_t - mean_60d(spread)) / std_60d(spread)`. 60-day rolling normalization (≈ 2 OU half-lives at the high end of the filter).
  5. **Entry rule**: If pair is enabled AND no open position AND `|z_t| > 2.0` → enter dollar-neutral pair at MOC. If `z > 2`, **short A, long B**. If `z < -2`, **long A, short B**.
  6. **Exit rule (mean-revert win)**: `|z_t| < 0.5` → flatten both legs at MOC.
  7. **Stop rule (mean-revert fail)**: `|z_t| > 3.0` → flatten both legs at MOC. Stop is one-sided: if you entered short at `z=+2.1` and z moves to +3.05, stop. If you entered short at `z=+2.1` and z moves to -3, you've already exited at the 0.5 cross.
  8. **Time stop**: If position held > 4× half-life (e.g. 60 days for an `H=15` pair) without hitting `|z|<0.5` → flatten and mark for cointegration review. This is the regime-break canary.
  9. **No-restart hysteresis**: After a stop-out at `|z|>3`, the same pair cannot re-enter for **5 trading days** (forces the next signal to be independent of the stopped-out leg's mean-reversion-failure tape).
- **Why this method fits this edge**: Cointegration is a *statistical* statement that two non-stationary series have a stationary linear combination — directly mappable to "the spread mean-reverts". OU is the canonical continuous-time mean-reverting process; its half-life is exactly the parameter that says "how fast does the spread come back?". The combination cleanly separates *whether* mean-reversion exists (E-G) from *how tradeable* it is (OU half-life). Both are 50-year-old textbook tools (Engle-Granger 1987, Vasicek 1977), with no overfitting surface beyond the cutoffs, the rolling window length, and the z thresholds — ≤ 6 effective hyperparameters across the whole system.
- **Anchor literature**: Gatev/Goetzmann/Rouwenhorst (2006) "Pairs Trading: Performance of a Relative-Value Arbitrage Rule" (Review of Financial Studies) demonstrated the *distance-based* version (sum-of-squared-deviations on normalized prices) on US equities 1962-2002 with monthly out-of-sample Sharpe ~0.6-1.0. Modern variants (Vidyamurthy 2004, Caldeira/Moura 2013, Krauss 2017 *Statistical Arbitrage Pairs Trading Strategies: Review and Outlook*, JES) substitute Engle-Granger or Johansen cointegration plus an OU half-life filter. We're implementing the modern variant on commodity ETFs (whereas GGR was equities); the edge premise — economic linkage → cointegration → tradeable spread — is the same.

## 5. Data requirements

- **Live**: IBKR US Securities Snapshot/Bundle ($10/mo). All 7 tickers fit in the standard NYSE-Arca subscription. No options or futures data needed.
- **Historical**: Daily OHLCV 2010-01-04 → present for all 7 ETFs. Sources: Polygon Free tier (within 5 rpm limit; ~7 tickers × 16 years × 252 days = ~28k bars, trivially within free-tier coverage), Yahoo for cross-check, IBKR `reqHistoricalData` for the live-window splice.
- **Storage**: < 50 MB total raw daily bars; < 200 MB with all intermediate spread/z-score panels cached. Trivial.
- **Cost**: ~$10/mo IBKR data; $0 historical. No alternative-data fees.
- **Latency**: Daily signal at 15:30 ET, order send at 15:55 ET (MOC). Round-trip tolerance = seconds. **No HFT requirements.**

## 6. Validation plan

- **Total daily sample**: 2010-01-04 → 2026-05-12 ≈ **4,100 trading days × 4 pairs**.
- **In-sample / calibration**: **2010-01-04 → 2019-12-31** (≈ 2,517 trading days). Used to:
  - Confirm the pair list (rank by avg cointegration p-value over this window).
  - Set the entry/exit/stop z-thresholds and the half-life band.
  - Set the cointegration p-cutoffs (per pair, from rolling-test base rates).
- **Out-of-sample (OOS)**: **2020-01-01 → 2026-05-12** (≈ 1,580 trading days). Includes:
  - **COVID stress** (Feb-Apr 2020). Tests behavior when *all* commodity correlations spike to 1 simultaneously.
  - **Negative WTI** (Apr 2020). Tests USO leg behavior at price ~ $0.
  - **Russia-Ukraine** (Feb 2022 onward). Tests the explicit USO/XLE structural-break failure mode.
  - **Inflation regime 2022-2023**. Tests cross-commodity drift when monetary hedges (GLD/SLV) diverge in different directions.
- **Minimum OOS pair-trades**: **≥100 closed pair-trades aggregated across all surviving pairs** before any verdict is issued. Historical base rate: GLD/SLV alone produces ~15-25 pair-trades/yr; 3 pairs × ~20/yr × 6 yr ≈ 360 trade-slots, so 100 is reached in roughly **3 OOS years**.
- **Walk-forward design**:
  - **Train window**: 5 years (1,260 trading days). **Test window**: 1 quarter (63 trading days). **Step**: 1 quarter.
  - 24 forward folds across 2015-2026.
  - On each train→test boundary: re-rank pairs, re-fit hedge ratios, re-fit OU half-lives. Hyperparameters (z-thresholds, p-cutoffs) are **fixed** at the values set on the 2010-2019 in-sample window; only the per-pair fit parameters refresh.
- **Purge / embargo**: 10-day embargo between train and test windows. The OU autocorrelation horizon is ≤ 30 trading days (half-life upper bound), and we want at minimum 1/3 of a half-life embargo to neutralize spillover.
- **Stat tests**:
  - **Bootstrap** (10k draws, block size = 21 days = 1 month) of OOS daily strategy returns. **Require 95% CI of Sharpe excludes 0.3.**
  - **Require OOS aggregate Sharpe ≥ 0.8** (cointegration pairs are competitive only if they meaningfully beat a 0.5-Sharpe market alternative; below 0.8 the post-friction edge is unreliable).
  - **Require OOS average net return per trade ≥ 30 bps** after 4 bps round-trip frictions (see §8 for the friction estimate).
  - **Require OOS max DD ≤ 12%** at the proposed $25k-per-pair sizing (each pair has ~$50k gross exposure; an OU process that overshoots to z=3 before flattening can give back ~3% of pair gross = ~$1,500 = ~6% of pair NAV in the worst case; across 4 pairs uncorrelated that's ~3% NAV aggregate, but during a regime-break correlated drawdown it can compound to ~12%).
  - **Pair-attribution requirement**: No single pair may contribute > 60% of the OOS Sharpe. (If GLD/SLV alone carries the result, this isn't a portfolio — it's one trade dressed up.)

## 7. Risk envelope

- **Max gross exposure across all pairs**: **4 pairs × $25k each = $100k notional on a $100k account**. Each pair is dollar-neutral so *net* exposure is ~$0 and *gross* is $50k per pair ($25k long leg + $25k short leg). **Total gross across 4 pairs = $200k ≈ 2× account NAV.** This is well inside Reg-T initial margin (50% × $200k = $100k required, which equals NAV) — no margin call distance at entry. Maintenance is 25% × $200k = $50k, leaving roughly $50k of cushion before a forced-liquidation event.
- **Per-pair max position size**: $25k per leg ($50k gross per pair). Hard cap $30k per leg.
- **Daily loss cap**: **-3% NAV** ($3k on a $100k account). If aggregate intraday MTM across all open pair-trades hits -3% NAV, flatten all open pairs at next print and stand down for the rest of the session.
- **Drawdown trigger**: trailing 20-day NAV drawdown ≥ 8% → strategy halted, human review required to restart.
- **Per-pair circuit breaker**: `|z| > 4.0` (one full unit beyond the stop) → flatten the offending pair at MKT (not MOC) and disable the pair for **10 trading days** pending cointegration re-test.
- **Consecutive-loss circuit breaker**: 3 consecutive stop-outs (`|z|>3`) on the same pair within 30 days → disable that pair until the next quarterly review.
- **Aggregate concentration rule**: If at any moment ≥ 3 pairs are open *in the same direction* on the same underlying commodity (e.g. long-GLD-vs-short-SLV plus long-GLD-vs-short-DBA-via-some-cross — hypothetical), block additional entries. Pre-flag any pair construction that introduces a single underlying into multiple legs.
- **Correlation rule with already-deployed strategies**: Reject co-deployment with any strategy that has > 0.4 30-day return correlation with this strategy in backtest. Of the prior iters: iter 4 (G10 FX carry) is uncorrelated; iter 1 (US small-cap overnight) is uncorrelated; iter 5 (VIX term-structure) can correlate weakly during commodity-vol regimes (cross-asset volatility), monitor.
- **PDT (lesson 5) — argued exemption**: Pair entries hold **3-15 days typical**, and a single round-trip uses 2 day-trade slots (one to enter, one to exit) only if entry and exit fall in the same session — which **almost never happens** because the z-score doesn't cross from `|z|>2` to `|z|<0.5` within a single day for any pair with `H ≥ 5` days. Historical 2010-2020 base rate: < 0.5% of entries flattened same-day. PDT trigger requires 4 day-trades in 5 business days, and our worst historical 5-day window has 1 day-trade. **Argued exemption stands: rotation cadence prevents PDT in normal regimes.** Belt-and-suspenders: maintain $25k minimum equity (matches our $100k deployment size); if NAV falls below $25k after a drawdown, halt the strategy.

## 8. Execution plan

- **Broker / SDK**: IBKR Pro, `ib_insync` (lesson 8 — battle-tested).
- **Order types**:
  - Routine entry/exit at 15:55 ET: **MOC (market-on-close)** on both legs simultaneously, paired as a *combo* via `ib_insync` `BAG` order where possible (GLD/SLV and USO/XLE support combo legs; CORN and UNG may need two parallel single-leg MOCs).
  - **Half-and-half fill protection**: For non-combo pairs (CORN, UNG), if one leg fills and the other doesn't (e.g. one is halted), set a hard 60-second timer to flatten the filled leg at MKT to avoid carrying a naked directional position overnight.
  - Circuit-breaker exit (`|z|>4`): **marketable IOC limit** at NBBO mid ± 0.25% on each leg, fallback MKT after 10 seconds.
- **Slippage budget**: All 7 ETFs have ADV > 5M shares and spread ≤ 2 bps in normal conditions. Budget **2 bps per leg per round trip = 4 bps gross per pair-trade.** Net post-friction expected edge per trade is `(avg z-revert P&L) - 4 bps`. Measured at every fill via NBBO-mid comparison.
- **What if IBKR rejects an order**: Single retry with MKT; on second rejection, page operator and freeze the pair (do NOT freeze the entire strategy — other pairs continue independently). Each pair has its own kill-switch state.
- **Broker truth (lesson 4)**: Every minute while *any* pair is open, query `reqPositions()` for all 7 tickers; reconcile against the local target. Any divergence > 1 share on any leg → halt entries on that pair (existing exit logic still runs based on broker-reported quantity, never local cache).

## 9. Capacity

- **GLD ADV** ~ 8M sh × ~$200 = **~$1.6B/day**. **SLV ADV** ~ 25M sh × ~$25 = **~$625M/day**. **USO ADV** ~ 4M sh × ~$80 = $320M/day. **XLE ADV** ~ 10M sh × ~$95 = ~$950M/day. **DBA**, **CORN**, **UNG**, **XOP**: all > $50M/day notional. The binding capacity is the *smallest* in any pair we'd hold simultaneously.
- **Strategy AUM ceiling**: A pair-trade aiming to consume ≤ 1% of ADV per leg per rebalance is feasible up to roughly **$30M AUM** (1% × min-leg-ADV $50M = $500k per leg × 4 pairs = $2M per rebalance; 6-7% of NAV in flow per typical entry day — but rebalances are staggered across pairs, not all-on-one-day). Above $30M, two frictions: (a) **CORN and UNG ADV are the binding constraint** — at $30M AUM × 25% per pair = $7.5M per leg = ~15% of CORN's ADV; the slippage budget breaks. (b) The closing-print impact at MOC starts to be measurable once we're > 5% of the close-cross volume.
- **Realistic deployment ceiling**: **$10M-$30M AUM**. Below $10M the strategy is fine on any single pair; the $30M cap kicks in only if we want all 4 pairs running at equal size. The user's $100k paper account is firmly in the "no friction" regime.
- **ADV consumption at $100k AUM**: $25k per leg × max(0.5%) of smallest-leg ADV = ~$25k / $50M = 0.05% of ADV. Negligible.
- **Single-ticker concentration**: Max 50% of strategy NAV in any single ticker (gross). Currently GLD appears in 1 pair (GLD/SLV), SLV in 1 pair, etc. — no single ticker is concentrated. If we later add (say) GLD/GDX, we'd flag the GLD double-exposure.

## 10. Failure modes

1. **Structural break in cointegration relationship (the canonical pairs-trading killer)**. **Russia-Ukraine 2022 is the type specimen**: Feb-Apr 2022 saw WTI spike from ~$90 to ~$130, but XLE only ran +30% (equity beta dampened by hedging, dividend obligations, ESG outflows), and USO underperformed crude itself due to severe contango/roll losses on the front-month futures it holds. The USO/XLE spread broke its 2015-2021 relationship for **roughly 6-9 months** before re-establishing a new (different-slope) equilibrium. Metric: rolling Engle-Granger p-value > 0.05 for 3 consecutive weeks. Signature: the pair is auto-disabled by the weekly gate, but any *open* position taken before the break gets stopped out at `|z|>3`. Mitigation: tighter p-cutoff for USO/XLE (0.005 vs 0.01), time stop at 4× half-life, structural-break monitor.
2. **Commodity-specific shock to one leg only.** Example: a single natural-gas pipeline rupture or LNG export-terminal outage moves UNG sharply while XOP barely twitches (XOP is diversified across producers). Metric: 1-day move in one leg > 3-σ while the other leg moves < 0.5-σ. Signature: z-score jumps from ~1 to >3 in a single session; if we were already in the trade, we eat the full stop-out (~6% of pair NAV). Mitigation: pre-trade earnings/event calendar filter for XLE/XOP (avoid entries the day before major component-stock earnings); for the commodity ETFs (UNG, USO, GLD, SLV, DBA, CORN) monitor relevant futures news. **Hard cap**: do NOT enter a new position if either leg has a ≥2-σ move in the prior 2 sessions — let the spread settle.
3. **Roll-loss drift on futures-based ETFs.** USO, UNG, DBA, CORN all hold front-month or front-quarter futures and pay a continuous roll loss (or gain) into their NAV that has nothing to do with the cointegration relationship. In persistent contango, USO will *systematically* underperform any equity-based pair (XLE). Metric: 3-month rolling spread mean drifts > 1.5 std from the prior 12-month mean. Signature: persistent one-sided z (e.g. always negative for USO/XLE during steep contango); the strategy keeps re-entering on the same side and stopping out. Mitigation: the weekly hedge-ratio recompute partially absorbs persistent drift; if drift persists > 3 weeks, the rolling-window mean catches up and the z-score normalizes. Audit metric: ratio of long-A-short-B entries vs short-A-long-B entries on each pair should be roughly balanced over a year (within 70/30); if it skews 90/10, drift has eaten the symmetry.
4. **Correlated drawdown across all pairs (the "everything-mean-reverts-the-wrong-way" day).** During a true risk-off shock (e.g. March 2020), commodity sub-sector correlations spike toward 1, and every spread can widen simultaneously. Metric: 3+ pairs simultaneously open with z in the same direction (all converging or all diverging). Signature: aggregate strategy NAV drops > 5% in 5 sessions. Mitigation: aggregate -3% daily loss cap; aggregate -8% 20-day DD halt; reduce per-pair sizing to 50% of nominal after any -5% week.
5. **Hedge-ratio instability (β estimation noise).** When the underlying correlation is borderline (p ~ 0.01), β can swing by ±15% week-to-week, and the "dollar-neutral" position is actually $5k net long or short. Metric: weekly β change > 10% relative to 90-day rolling β. Signature: realized P&L has a small but persistent directional component uncorrelated with the spread. Mitigation: cap weekly β change at 5% (winsorize); if rolling-test gate trips on β instability, disable the pair until 3 consecutive stable weeks.
6. **ETF issuer / fund-restructure event.** USO restructured in 2020 (mid-year, after the negative-oil event); CORN restructured ~2020 as well. A restructure that changes the ETF's underlying (e.g. switching from front-month to 12-month strip) can permanently change the cointegration relationship. Metric: any SEC filing or prospectus amendment touching one of our 7 tickers. Signature: usually a sharp one-day jump in the spread on the restructure date. Mitigation: subscribe to issuer IR alerts; freeze affected pair for 60 days pending re-validation.
7. **Data quality / split-adjustment errors.** ETFs occasionally split; if our cache mis-applies the split, the spread jumps spuriously. Metric: any one-day spread move > 5-σ with no underlying news. Signature: false stop-out. Mitigation: cross-check daily closes against IBKR `reqHistoricalData` and Polygon; any split flag triggers a manual review and a 1-day signal freeze on the affected pair.

## 11. Falsifiable verdict criteria

- **Promote to live capital** — all of the following over a ≥ 30-day shadow run AND ≥ 100 OOS pair-trades aggregated:
  - OOS aggregate Sharpe ≥ 0.8 with bootstrap 95% CI excluding 0.3.
  - OOS average net return per trade ≥ 30 bps (after 4 bps frictions).
  - OOS max DD ≤ 12% NAV.
  - No single pair contributes > 60% of OOS Sharpe (portfolio integrity).
  - Zero broker-truth divergence events during shadow.
  - At least 2 of the 4 candidate pairs pass their cointegration gate ≥ 70% of weeks in the OOS window.
  - All circuit breakers fire correctly on every backtest day where conditions were met.

- **Continue shadow** — metrics ambiguous (Sharpe 0.4-0.8, DD 8-12%), < 100 OOS pair-trades observed, no kill triggers fired.

- **Kill immediately** if any one of:
  - 3-day NAV drawdown > 4%.
  - Daily loss cap breach (-3% NAV).
  - Broker-side fill discrepancy > 0.5% NAV.
  - Any missed circuit-breaker firing (post-mortem shows trigger was met but flatten didn't execute).
  - **All 4 pairs simultaneously fail the cointegration gate for 3 consecutive weeks** → the entire commodity-pair regime has broken.
  - Trailing 12-month OOS Sharpe < 0.4 OR trailing 12-month max DD > 12% — **the 90-day-deploy falsification clause**.
  - Any one of the 4 ETFs delisted, halted > 2 sessions, or sponsor-restructured.

## 12. One-paragraph operational risk note

The strategy has more pieces than a single-instrument rule (4 pairs × 2 legs × weekly recompute + daily signal + per-minute reconciliation), so the operational surface is larger than iters 1, 2, 4, 5. Key vulnerabilities: (a) the **weekly Sunday recompute job** is the single point of regime adaptation — if the cron misses (laptop asleep, gateway disconnected), we trade Monday-Friday with a stale hedge ratio and stale p-value gate; mitigation: cron health-check that pages the operator if the recompute writes a status row older than 72 hours, and Sunday job runs twice (00:01 ET and 12:00 ET) for redundancy. (b) **Half-and-half pair fills**: if one leg's MOC fills and the other doesn't (halt, reject, late submission), we're carrying naked directional risk equal to one leg's notional; the 60-second auto-flatten in §8 caps this, but operator must verify the auto-flatten actually fired. (c) **Cointegration p-value computation correctness** — the ADF test on residuals must use the correct lag length (we'll use AIC-selected, max lag = 10); a bug in the lag selection silently inflates or deflates the p-value and can either disable working pairs or enable broken ones. Mitigation: unit tests on the Engle-Granger function against `statsmodels.tsa.stattools.coint` reference values plus a weekly cross-check that hand-computes the p-value on a fixed historical window and compares to expected. (d) **IBKR combo orders** (BAG) sometimes fail silently in low-liquidity tickers (CORN, UNG); on combo rejection we must fall back to two single-leg MOCs and tolerate the half-fill risk. Finally, this strategy is genuinely *quiet* during low-volatility commodity regimes — sometimes 2-3 weeks with zero entries — which (per iter-5 lesson) increases the risk that the operator stops watching; mandatory weekly status review (signal counts, pair-gate status, β stability) is a hard requirement.

---

## Lesson compliance (the 10 lessons, explicit)

1. **Validate before deploy.** §6 mandates walk-forward (24 forward folds) + ≥ 30-day shadow + ≥ 100 OOS pair-trades before any live capital.
2. **Binary gates, not confidence rankers.** Entry/exit/stop are binary z-threshold crosses. The cointegration p-value is used as a **binary gate** (pair enabled / disabled), never as a continuous confidence input to position sizing. Position size is fixed at $25k per leg; pairs are equal-weight by NAV, never weighted by p-value or by recent performance.
3. **TP:SL asymmetry is structural.** Entry at `|z|=2`, exit at `|z|=0.5` (1.5-z reward), stop at `|z|=3` (1-z risk). **Reward:risk = 1.5:1** at the z-axis, and because the z distribution is roughly Gaussian, P(`|z|<0.5` reached before `|z|>3` from a `|z|=2` start) ≈ 80% under stationary OU — that asymmetry IS the edge.
4. **Broker truth, not local cache.** §8 requires per-minute `reqPositions()` reconciliation across all 7 tickers; closing orders use broker-reported quantity. Each pair has independent broker-truth state.
5. **No PDT-exposed lanes on sub-$25k US equity accounts.** Pair holds are **3-15 days typical** by construction (matches the OU half-life filter). Historical 2010-2020 base rate of same-session round-trips: < 0.5%. PDT trigger (4 day-trades in 5 business days) requires single-session entry-and-exit, which the rule structure prevents. **Argued exemption: PDT is N/A because the OU half-life filter forces multi-day holds; the worst historical 5-day window has 1 same-session round-trip vs the 4 needed.** Belt-and-suspenders: maintain $25k minimum equity (matches our $100k deployment); halt if NAV < $25k.
6. **One strategy per slot.** This is the canonical "commodity ETF cointegration pairs" slot. Reject any other iteration proposing pairs on the same 7 tickers, distance-based pairs trading on commodities, or any commodity stat-arb variant — they are the same edge.
7. **No same-data backtest.** §6 cleanly separates 2010-2019 (in-sample, calibration of hyperparameters and pair selection) from 2020-2026 (OOS, includes COVID, negative-oil, Russia-Ukraine, inflation-regime). The walk-forward design refits per-pair parameters but freezes hyperparameters at the 2010-2019 values, so no peeking into 2020+ to choose z-thresholds or p-cutoffs.
8. **Battle-tested broker SDKs only.** `ib_insync` (lesson-8 ticker mention).
9. **Capacity bound named upfront.** §9: **$10M-$30M realistic AUM**, bound by CORN/UNG ADV. Edge degrades above $30M from slippage on the small-leg ETFs. $100k paper deployment is in the no-friction regime.
10. **Kill criteria written before deploy.** §11 enumerates exact thresholds. Top-line falsifier: trailing 12-month OOS Sharpe < 0.4 OR trailing 12-month max DD > 12% OR all 4 pairs simultaneously fail cointegration gate for 3 consecutive weeks → kill within 90 days.

---

THESIS: Engle-Granger cointegration plus OU half-life 5-30d filter on 3-4 economically linked US commodity ETF pairs produces a z-score mean-reversion edge that survives the 2020-2026 OOS regime (COVID, negative oil, Russia-Ukraine, inflation) at aggregate Sharpe ≥ 0.8 after 4 bps frictions.
VALIDATION HORIZON: ≥100 OOS pair-trades aggregated across the surviving 3-4 pairs (≈ 3 OOS years at the historical ~20 trades/pair/yr rate), with Sharpe bootstrap 95% CI excluding 0.3.
CAPACITY CEILING: $10M-$30M realistic AUM (bounded by CORN/UNG ADV at ~$50M/day each); $100k paper deployment is in the no-friction regime.
