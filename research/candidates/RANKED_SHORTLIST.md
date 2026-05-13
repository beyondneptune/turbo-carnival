# Ranked Shortlist — 25 Trading-System Candidates

**Operator profile**: Retail/small-fund. ~$497k paper across Alpaca + IBKR. Consolidating to one live account, $100k-$500k initial. Already running iter_5-equivalent (VIX/VIX3M SVXY rotation) live, 7-yr walk-forward +17.1% / Sharpe 0.66. Synthesis goal: find the candidates that *add diversified expected return + risk* to the existing book while staying operationally runnable by one person.

Scoring rubric (1-5 per dimension, 5 = best; max 35):

| Dim | Captures |
|---|---|
| **EDGE** | Strength of falsifiable thesis and academic anchor |
| **FIT** | Match to a $100-500k single live account (capacity, PDT, retail-friendly margin) |
| **OPS** | One broker, one data vendor, no GPUs/custom infra |
| **VAL** | Free/cheap data sufficient; OOS sample large; not lookahead-prone |
| **KILL** | Specific numeric kill criteria triggered by ≤90 days of data |
| **TIME** | Time to first verdict signal (days-weeks > months-years) |
| **DIV** | Diversification from existing SVXY rotation |

Tie-break: EDGE × KILL.

---

## Scoring Table (25 rows)

| # | Candidate | EDGE | FIT | OPS | VAL | KILL | TIME | DIV | **TOTAL** | E×K |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | US small-cap overnight mean-rev (Berkman/Lou-Polk-Skouras anchor) | 5 | 4 | 5 | 5 | 5 | 5 | 4 | **33** | 25 |
| 2 | ES futures 1-min OFI rules (Cont-Kukanov-Stoikov anchor) | 4 | 4 | 4 | 4 | 5 | 5 | 4 | **30** | 20 |
| 3 | BTC perp funding cash-and-carry (Hougan/Lawant, Glassnode) | 4 | 3 | 3 | 4 | 5 | 5 | 5 | **29** | 20 |
| 4 | G10 FX carry (Lustig-Verdelhan, MSSS 2012) | 5 | 5 | 5 | 5 | 4 | 3 | 5 | **32** | 20 |
| 5 | VIX term-structure rotation (already live) | 5 | 5 | 5 | 5 | 5 | 5 | 1 | **31** | 25 |
| 6 | Nikkei 225 mini opening-range breakout (Crabel) | 3 | 4 | 4 | 4 | 5 | 5 | 4 | **29** | 15 |
| 7 | DAX futures TSMOM (Moskowitz/Ooi/Pedersen) | 5 | 4 | 5 | 5 | 4 | 3 | 4 | **30** | 20 |
| 8 | Commodity ETF pairs cointegration (Engle-Granger + OU) | 4 | 5 | 5 | 5 | 4 | 4 | 4 | **31** | 16 |
| 9 | Nifty 50 cross-section momentum (FPI/INDA fallback) | 4 | 2 | 2 | 3 | 4 | 4 | 4 | **23** | 16 |
| 10 | Gold/silver ratio z-score reversion (Schwartz-Smith) | 4 | 5 | 5 | 5 | 3 | 2 | 4 | **28** | 12 |
| 11 | LLM FOMC hawk/dove classifier on SPY+TLT | 3 | 4 | 3 | 3 | 3 | 3 | 4 | **23** | 9 |
| 12 | TFT transformer next-bar forecast on ES | 2 | 3 | 2 | 3 | 4 | 4 | 3 | **21** | 8 |
| 13 | HMM regime ensemble (sector ETF rotation) | 3 | 4 | 4 | 4 | 4 | 3 | 3 | **25** | 12 |
| 14 | LightGBM cross-section Russell 1000 (Gu-Kelly-Xiu) | 4 | 3 | 2 | 2 | 4 | 2 | 4 | **21** | 16 |
| 15 | PPO RL execution alpha (TWAP+) — meta-strategy | 3 | 2 | 2 | 3 | 4 | 3 | 2 | **19** | 12 |
| 16 | Multi-carry stack BTC funding × FX carry (combinatorial) | 4 | 2 | 2 | 4 | 4 | 3 | 5 | **24** | 16 |
| 17 | Earnings ATM straddle + delta-hedged gamma scalp | 4 | 3 | 3 | 3 | 4 | 4 | 4 | **25** | 16 |
| 18 | US/HK Chinese ADR convergence arb (BABA/9988 etc.) | 3 | 3 | 1 | 3 | 4 | 4 | 5 | **23** | 12 |
| 19 | On-chain DEX flow + perp funding composite (directional ETH/BTC) | 2 | 3 | 1 | 2 | 4 | 4 | 4 | **20** | 8 |
| 20 | LLM thematic ETF basket rotation (20 themes, monthly) | 2 | 3 | 2 | 2 | 3 | 2 | 3 | **17** | 6 |
| 21 | Multi-LLM debate consensus (Claude + GPT + Llama) on SPY/QQQ/IWM | 2 | 3 | 1 | 2 | 3 | 3 | 3 | **17** | 6 |
| 22 | End-to-end deep RL trader on ES | 1 | 2 | 1 | 2 | 4 | 3 | 3 | **16** | 4 |
| 23 | Global risk premia multi-factor (50+ futures, carry+mom+value) | 5 | 3 | 3 | 4 | 4 | 2 | 5 | **26** | 20 |
| 24 | SVI implied-vol surface relative-value | 4 | 3 | 2 | 2 | 4 | 3 | 4 | **22** | 16 |
| 25 | Meta-validation framework (registry + CI/CV) — NOT a strategy | 3 | 5 | 3 | 4 | 4 | 4 | 1 | **24** | 12 |

---

## Top 3 (the actual shortlist to research first)

### 1. iter_1 — US small-cap overnight mean reversion (Total 33, E×K 25)

Why this beats the others: deepest academic anchor (Berkman/Koch/Westerholm 2012, Lou-Polk-Skouras 2019) of any non-VIX candidate; deterministic rules-based ruleset (no ML failure surface); single broker (IBKR), single data vendor (Polygon $29/mo + IBKR fundamentals), no GPU, no custom infra; kill criteria are crisp (90-day rolling Sharpe < 0.5, 60-day DD > 8%, slippage > 15 bps); validation horizon is short — 400 OOS trades = ~5 months at 3 trades/day so a verdict is reachable inside the 90-day shadow; diversification from VIX rotation is strong because mean-reversion on intraday-selloffs is an equity-microstructure premium, not a vol-roll premium.

What needs to happen: (1) walk-forward 2018-2021 IS / 2022-2025 OOS using Polygon Stocks Starter — target Sharpe ≥1.0 with 95% CI ≥0.5, ≥400 trades; (2) MOC/MOO infrastructure on IBKR paper with the exact LOC + OPG attribute order plumbing; (3) 30 trading-day shadow with zero broker-truth divergence. First concrete deliverable: a notebook running the universe filter + RSI(2) + volume-ratio gate on 2022-2025 Russell 2000 history, with the 8 bps round-trip slippage assumption baked in.

Biggest risk: small-cap reversal regime can decouple in a sustained bear leg (March 2020 first leg, Feb 2018). The strategy's circuit-breaker (60-day DD > 8% halt) handles this, but a confluence of multiple multi-day weak Russell weeks could chew the operator's confidence before the kill criterion fires.

### 2. iter_4 — G10 FX carry diversified (Total 32, E×K 20)

Why this beats the others: 35-year well-published premium (Lustig-Verdelhan 2007, MSSS 2012); the vol-filter rule is binary-region and pre-registered (no in-sample tuning); OANDA / IBKR retail-accessible; PDT structurally exempt (FX is CFTC/NFA, not FINRA); 60+ OOS monthly rebalances easily satisfied by 2011-2025 data; max DD historically <25% with the vol filter biting at the 12-14% CVIX threshold. Diversification from VIX rotation is essentially complete — FX carry has near-zero correlation with short-vol-on-SVXY through normal regimes and gets hurt by *different* tail dynamics (carry crashes from yen rallies, not from VIX-spikes). $50k OANDA minimum, $1M retail capacity ceiling fits the operator's account profile exactly.

What needs to happen: (1) implement long-3/short-3 vol-scaled basket; (2) build the FX-implied-vol filter (CVIX or 20-day realized fallback); (3) backtest 1990-2010 calibration + 2011-2025 OOS with broker swap-rate friction baked in; (4) verify OANDA paper-trade plumbing on a $50k mock account. First concrete deliverable: a notebook that ranks G10 currencies by 1-month OIS differential, applies vol-scaling, computes monthly rebalance P&L 1990-2025, and reports the Aug 2024 yen-carry-unwind drawdown specifically (the most recent crash-tail test).

Biggest risk: carry crashes are slow-moving and can produce 4-5 negative months before the vol filter de-risks; operator must believe the 90-day "kill window" is a kill window, not a "promote in 90 days" window — the time-to-statistical-promotion is ≥12 months, but the kill triggers (12% 30-day DD, vol filter failure to engage) are 90-day-falsifiable.

### 3. iter_5 — VIX term-structure rotation (Total 31, E×K 25)

Why on the shortlist despite being already live: the operator's existing live strategy IS this. The shortlist value here is not "research it" but "harden it" — iter_5's spec contains tighter circuit-breakers (VIX_spot > 30, 1-day VIX > +50% intraday, etc.), broker-side hardware stop, dual-feed CBOE+Polygon redundancy, hysteresis rules, and an explicit 12-month OOS clause that the operator's existing 7-yr walk-forward result probably does not enforce in production code. Spending 1-2 weeks bringing the production code up to iter_5's documented circuit-breaker spec is the highest-value defensive use of operator time. DIV = 1 because it's the existing book; the action is *harden*, not *deploy new capital*.

What needs to happen: (1) audit current production code against §7 circuit-breaker list; (2) add the hardware-side -15% stop at IBKR as the backstop independent of the software loop; (3) add the dual-feed VIX/VIX3M with fail-closed default to SGOV; (4) test the §8 broker-truth reconciliation cadence (per-minute `reqPositions()`).

Biggest risk: this is not a research risk; it's a complacency risk. The strategy is silent during long contango streaks and operators stop watching. The iter_5 spec's mandatory 5-min weekly review is the right antibody.

---

## "Don't bother yet" — 5 low-priority candidates (with one-line why)

- **iter_22 (end-to-end deep RL on ES)** — Author's own prior is failure; requires GPU training infrastructure, multi-seed protocol, and a >$1,000 GPU-hour budget for an iteration whose head-to-head 1.0-Sharpe-uplift bar against iter_2 is very unlikely to clear. Defer until iter_2 or iter_12 is itself validated.
- **iter_21 (multi-LLM debate consensus on SPY/QQQ/IWM)** — Three vendor API surfaces to monitor for snapshot drift, no clear edge over a single-LLM gate, and the orthogonal-blindspot premise is empirically untested; complexity tax not justified.
- **iter_20 (LLM thematic ETF rotation)** — Monthly turnover of 60-100% × 30 bps round-trip = 360-720 bps/year cost drag that the LLM ranker has to clear; multi-vendor data stack (RavenPack + Quartr + Trends + Anthropic) creates many failure modes for a strategy whose net edge over a sentiment-baseline must be ≥0.2 IR units to justify.
- **iter_19 (on-chain DEX + perp funding composite)** — Requires Glassnode Advanced ($799/mo) or scraped The Graph + Etherscan stack, 24/7 ops shift overnight, and signal half-life is already shrinking from 24h (2021-22) to 4-8h (2024-25) per practitioner reports — the alpha-decay clock is ticking and the operator's time-to-deploy is months.
- **iter_18 (US/HK ADR convergence arb)** — Operationally hardest of all 24 strategy candidates by the author's own admission: two-venue/two-currency/two-clearinghouse settlement, no cross-exchange combo orders, HKEX data subscription required, daily SEC HFCAA list scraping, 24/7 monitoring for regulatory windows. Real edge but mismatched to a single-operator runtime.

---

## Recommended path (this week)

Start a focused research spike on **iter_1 (US small-cap overnight mean-reversion)** and **iter_4 (G10 FX carry)** in parallel — they are the two candidates that combine strong academic anchors with crisp 90-day-falsifiable kill criteria, diversify cleanly from the existing live VIX/SVXY rotation, and run inside a one-operator-on-IBKR-or-OANDA operational envelope without requiring GPUs, alt-data subscriptions above $50/mo, or multi-broker custody complexity. The first concrete deliverable is a single Jupyter notebook per candidate that runs the documented walk-forward (iter_1: 2018-2021 IS / 2022-2025 OOS with the RSI(2) + 2.0× volume + −2.5% selloff gate; iter_4: 1990-2010 IS / 2011-2025 OOS long-3/short-3 with the 12%/14% CVIX vol filter) and produces the iter-file's stated Sharpe/DD/CI statistics on the operator's own cached data — completing this notebook for both within ~10 days is enough to decide which one (or both) proceeds to a 30-day IBKR/OANDA paper shadow. Defer iter_22, iter_21, iter_20, iter_19, and iter_18 indefinitely, harden the live iter_5 deployment against its own §7/§8 circuit-breaker spec as a 1-2-day side task, and revisit iter_23 (global multi-factor risk premia) only after iter_1 or iter_4 has cleared shadow — iter_23 strictly contains iter_4 as a sub-component and is the natural scale-up path if iter_4 validates and the AUM ever reaches the $5M+ tier where the data subscription + portfolio-margin complexity is amortizable.
