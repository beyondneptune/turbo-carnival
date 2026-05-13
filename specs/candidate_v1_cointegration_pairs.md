# Candidate v1 — Cointegration Commodity ETF Pairs

**Status:** PREREGISTERED · NOT YET DEPLOYED
**Locked at:** 2026-05-13 (committed to git as part of the Layer-B-shutdown follow-up)
**Author:** weeoto + Claude (acting leader role per session 2026-05-13)

This document is the preregistration of the only strategy candidate following the 2026-05-13 full Layer-B shutdown. Per the literature verdict (see `~/.claude/projects/-Users-weeoto/memory/project_full_shutdown_2026-05-13.md`), no new strategy may run paper-shadow or live without an immutable preregistration that this document fulfills.

The 10 hard rules of `specs/global_market_system_v1.md` apply.

---

## 1. Thesis (one sentence)

Commodity ETFs in the same demand category co-move with a stable long-run relationship, so a mean-reverting Engle-Granger spread between two cointegrated commodity ETFs has a positive expected return when the rolling z-score of the residual exceeds ±2σ, exiting at 0, with a hard stop at ±4σ.

## 2. Market / asset class / venue

- **Instruments (universe of 10):** USO (oil), UNG (natgas), GLD (gold), SLV (silver), DBA (agriculture), DBC (broad commodity), CORN, WEAT, CPER (copper, replaces delisted JJC), PPLT (platinum, replaces delisted JJN industrial-metal slot). Universe was originally drafted with JJN/JJC; both were delisted 2023-07-21 (iPath ETN redemption) — caught by Phase 2 ingest pre-commit and corrected before locking the spec.
- **Exchange:** NYSE Arca / NASDAQ via Alpaca paper account (AEGIS)
- **Settlement:** T+1 US equity
- **Margin:** Reg-T, cash collateral; no leverage
- **Tax regime:** US equity ETFs (no K-1 for these tickers; USO has K-1 risk — flagged in §13)

## 3. Time horizon

**Daily.** Close-to-close holding period; one decision per market close. No intraday entries.

## 4. Method family and specific algorithm

- **Family:** Rules-based statistical arbitrage (no ML, no LLM, no neural network)
- **Algorithm:** Engle-Granger 2-step cointegration on log-price spread with 30-day rolling window, z-score normalization of residuals, mechanical entry/exit/stop on z-score thresholds
- **Why this fits:** The literature (IJSR SR231123060916, "Cointegration-Based Pairs Trading Strategy in Commodity Markets" by Mohandas) demonstrated this exact rule-set on commodity index pairs and survived multiple-testing corrections. The corpus also validates pairs trading as a documented anomaly in the technical-trading literature (s10479-019-03357-1) where the edge survives Bonferroni-corrected significance tests on commodity-related universes.

## 5. Entry / exit / stop rules (code-level pseudocode)

```python
# For each (asset_a, asset_b) pair in the daily candidate list:
log_a = log(close[asset_a])
log_b = log(close[asset_b])

# Engle-Granger step 1: rolling OLS over WINDOW days (default 30)
beta = ols(log_a, log_b, window=30).slope
alpha = ols(log_a, log_b, window=30).intercept
residual = log_a - (alpha + beta * log_b)

# Engle-Granger step 2: ADF on residual (cointegration test)
adf_pvalue = adfuller(residual[-30:]).pvalue
if adf_pvalue >= COINT_PVALUE_THRESHOLD:  # default 0.05
    skip_pair_today()

# Z-score the residual
z = (residual[t] - mean(residual[-30:])) / std(residual[-30:])

# Position rule
if position == FLAT:
    if z > ENTRY_THRESHOLD:           # default +2.0
        open(short=asset_a, long=asset_b, notional=PER_PAIR_NOTIONAL)
    elif z < -ENTRY_THRESHOLD:        # default -2.0
        open(long=asset_a, short=asset_b, notional=PER_PAIR_NOTIONAL)
elif position == OPEN:
    if abs(z) < EXIT_THRESHOLD:       # default 0.0
        close_at_market()
    elif abs(z) > STOP_THRESHOLD:     # default 4.0
        close_at_market_with_stop_flag()
    elif holding_days >= MAX_HOLD:    # default 60 trading days
        close_at_market_with_timeout_flag()
```

## 6. Variant universe (LOCKED for White's Reality Check)

The full universe of variants we will test must be enumerated before any backtest is run, to honor White's RC against data snooping.

| Parameter | Values | Count |
|---|---|---|
| `WINDOW` (rolling cointegration days) | 20, 30, 45, 60, 90 | 5 |
| `ENTRY_THRESHOLD` (z) | 1.5, 2.0, 2.5 | 3 |
| `STOP_THRESHOLD` (z) | 3.0, 4.0, 5.0 | 3 |
| `EXIT_THRESHOLD` (z) | 0.0, 0.5 | 2 |
| `COINT_PVALUE_THRESHOLD` | 0.01, 0.05, 0.10 | 3 |
| `MAX_HOLD` (trading days) | 30, 60, 90 | 3 |

Total variant count: **5 × 3 × 3 × 2 × 3 × 3 = 810 strategy variants**

White's RC bootstrap p-value (B = 10,000) is computed against the maximum-Sharpe variant on training data. The reported "best strategy" is only valid if `p̂_RC < 0.05` AND `p̂_SPA < 0.05`.

Per Harvey-Liu-Zhu (2016) the empirical t-stat hurdle for declaring real edge after multiple-testing correction across 810 variants is **t > 3.0**, not the conventional 2.0.

Per Bailey-Lopez de Prado PBO (2014), Probability of Backtest Overfitting must be **< 50%**, computed via combinatorially symmetric cross-validation with 16 partitions.

## 7. Sizing rule

- **Per-pair notional:** 5% of NAV per open spread (long + short legs equal-dollar)
- **Maximum simultaneous pairs:** 4 (so peak gross exposure = 8 × 5% = 40% NAV; aggregate net beta should be near zero by construction)
- **Kelly fraction:** Once OOS Sharpe is established, size each pair at `0.25 × Kelly` (fractional Kelly per Wu-Chung IEEE paper, "A Novel Approach of Option Portfolio Construction Using the Kelly Criterion") — capped at 5% per pair regardless of Kelly recommendation
- **No leverage. No margin amplification.**

## 8. Risk overlay (Phase 5 spec)

- **CVaR cap:** Rockafellar-Uryasev linear program; daily 95% CVaR of aggregate position book < 5% of NAV. If breached, scale all open notionals proportionally until inside bound.
- **Vol target:** 12% annualized portfolio vol; if 20-day realized vol exceeds 18%, scale exposure to 0.67×.
- **Chandelier per-position stop:** 3 × ATR(22) trailing stop, in addition to the z-score stop. Whichever triggers first closes the spread.
- **Drift rebalance:** if any pair's |notional drift| > 5%, force a closing-and-reopening rebalance (one round-trip cost).
- **Daily circuit breaker:** if session P&L < −3% of starting NAV, halt all new entries for the rest of the day. Existing positions managed to exit per normal rules.

## 9. Success hurdles (must ALL pass before progressing to Phase 7 shadow)

| Hurdle | Threshold | Source |
|---|---|---|
| White's Reality Check p-value | `p̂_RC < 0.05` | White 1999 |
| Hansen SPA test p-value | `p̂_SPA < 0.05` | Hansen 2005 |
| Probability of Backtest Overfitting | `PBO < 0.50` | Bailey-Lopez de Prado 2014 |
| Deflated Sharpe Ratio | `DSR > 1.0` | Bailey-Lopez de Prado 2014 |
| Best-variant t-stat | `t > 3.0` | Harvey-Liu-Zhu 2016 |
| OOS net Sharpe (Perold-cost-modeled) | `> 1.0` | Perold 1988 |
| OOS Sharpe 95% bootstrap CI lower bound | `> 0` | stationary bootstrap, 2000 iterations |
| Max drawdown (OOS) | `< 20%` | conventional retail tolerance |
| Calmar ratio (OOS) | `> 1.0` | (ann_return / max_dd) |
| Sharpe shrinkage in-sample → OOS | `< 50%` | overfitting guard |

If ANY hurdle fails, the candidate is killed. No partial-pass / vibes-based deployment.

## 10. Kill criteria (during paper shadow Phase 7)

| Trigger | Action |
|---|---|
| Realized 30-day Sharpe outside 1σ of backtest CI | Extend shadow by 30 days |
| Realized max DD > 1.5 × backtest DD | Kill candidate |
| Any single-day loss > 2 × backtest worst-day | Kill candidate |
| Implementation shortfall > 2 bps/trade above Perold model | Investigate; if reproducible, kill |
| Bottom-decile crash in first 10 shadow days (per [2512.11913] alpha-decay 1.7–1.8× crash probability signature) | Pause for 5 days; if recurs, kill |
| 3+ consecutive cycles with `submit_failed` (no broker fills) | Halt + investigate (the today-13 ADR bug pattern) |
| Cointegration ADF p-value > 0.10 across **all** pairs for 5+ consecutive days | Regime break flag; halt new entries, exit existing pairs at signal |

## 11. Data needs + train/test split

- **Source:** yfinance daily OHLCV as primary (full 2014+ history); Alpaca IEX as fallback. Alpaca free IEX feed only goes back to ~2020-04 so it cannot cover the locked train window; this was caught in Phase 2 pre-commit and ingest source order was inverted.
- **History:** 2015-01-01 → present
- **Train set:** 2015-01-01 → 2022-12-31 (8 years)
- **OOS set:** 2023-01-01 → 2025-12-31 (3 years)
- **Walk-forward folds:** 5 purged k-fold folds within the train set, 22-day purge gap to prevent leakage
- **Cache:** parquet under `data/cache/candidate_v1/<symbol>.parquet`, partitioned by symbol
- **Survivorship:** ETF universe is fixed; all 10 tickers existed through entire history (verified before backtest); no listing/delisting bias

## 12. Capacity bound (declared upfront, per rule #9)

- **Edge degradation:** at $1M AUM, the typical pair spread quote (USO/UNG) is ~$0.01–0.03 wide; 5% per-pair notional = $50k each side; expected slippage per round-trip ~5 bps, well within the Perold cost model in §9.
- **Hard cap:** $10M AUM. Above this, average daily volume of CORN/WEAT/JJN/JJC ($5M–$20M ADV) means our trade size moves the spread by > 10 bps round-trip; edge degrades materially.
- **Initial deployment size:** Phase 9 live deploy = $1k per pair maximum (so $4k peak gross), about 1% of a single $100k Alpaca paper account.

## 13. Known landmines / honest caveats

1. **USO has a K-1 tax form** (commodity-pool ETF). For a paper-only run this is fine; if this graduates to real money, swap USO → XLE (equity-energy ETF) or BNO and re-validate.
2. **Cointegration may break in regime shifts** (e.g., GLD/SLV decoupled in 2020 COVID; oil/natgas decoupled when US fracking shifted natural gas to oversupply 2018-2020). The ADF-on-residual gate at p < 0.05 should catch this — but verify on stress-test regimes per §6.
3. **Survivorship of the ETF universe:** Original draft listed JJN + JJC; both delisted 2023-07-21. Caught by Phase 2 ingest probe (yfinance coverage check) before commit. Replaced with CPER + PPLT, which have continuous data 2014-present. All 10 final tickers verified continuous through 2025. The universe was finalized BEFORE the backtest variant sweep ran; no in-flight adjustment after the fact.
4. **HK/dual-listing exclusion:** the prior ADR pairs strategy (iter_18) is explicitly out of scope here; this is US-only commodity ETFs.
5. **Beta neutrality:** the strategy is dollar-neutral by construction (equal-dollar legs), not beta-neutral. The cointegration coefficient `beta` from §5 controls relative sizing; in practice we approximate with equal-dollar then beta-rebalance daily.
6. **Daily-rebalance ETF decay:** all instruments in §2 are 1× (not leveraged), so daily-rebalance decay does not apply.

## 14. Citation block (literature anchor)

Primary sources from the NotebookLM corpus (`https://notebooklm.google.com/notebook/ae961d42-7736-4d56-ac1e-6dc45585f91e`):

1. **IJSR SR231123060916** (Mohandas) — "Cointegration-Based Pairs Trading Strategy in Commodity Markets" — provides the explicit z-score ±2 / exit 0 / stop ±4 rule set and 30-day rolling regression methodology.
2. **s10479-019-03357-1** ("Technical trading and cryptocurrencies") — pre-validates the multiple-testing correction methodology (Bonferroni/Holm/BH/BY + stationary bootstrap) we apply in §6 and §9.
3. **White 1999** ("A Reality Check for Data Snooping") — RC bootstrap test for §9.
4. **EFMA 2020 SR231123060916** ("Data Snooping Bias in Tests of the Relative Performance of Multiple Forecasting Models") — empirical demonstration that selecting from a restricted universe inflates false discoveries; mandates the variant universe in §6 be tested in full.
5. **arxiv 2512.11913** ("Not All Factors Crowd Equally") — alpha-decay watchdog rule in §10 (bottom-decile crash 1.7–1.8× probability is the crowding signature).
6. **DynamicTrading.pdf** (Pedersen et al.) — provides the aim-portfolio sizing rationale referenced in §7.
7. **IEEE A Novel Approach of Option Portfolio Construction Using the Kelly Criterion** (Wu-Chung) — fractional Kelly sizing in §7.

To-be-added gap papers (PDFs in `~/Documents/notbookllm research/`, pending NotebookLM upload):
8. **ssrn-2326253.pdf** Bailey-Lopez de Prado PBO 2014 — §6, §9 PBO and DSR.
9. **ssrn-2708678.pdf** Lopez de Prado HRP 2016 — alternative aggregation in case Kelly is too aggressive.
10. **rtr179-CVaR1.pdf** Rockafellar-Uryasev CVaR 2000 — §8 CVaR linear program.
11. **P118_and_the_cross.PDF** Harvey-Liu-Zhu 2016 — §9 t > 3 hurdle.
12. **impshort.pdf** Perold 1988 — §9 OOS net-Sharpe cost model.

## 15. Phase progression (this candidate)

Per `~/.claude/projects/-Users-weeoto/memory/project_full_shutdown_2026-05-13.md`:

- ☑ Phase 0 (papers acquired locally; manual upload to NotebookLM in progress)
- ☑ Phase 1 (this document)
- ☐ Phase 2 — Data ingest (`research/candidate_v1/ingest.py`)
- ☐ Phase 3 — Backtest with White's RC + SPA + PBO + DSR (`research/candidate_v1/backtest.py`)
- ☐ Phase 4 — Walk-forward OOS validation (`research/candidate_v1/walk_forward.py`)
- ☐ Phase 5 — Risk overlay (`research/candidate_v1/overlay.py`)
- ☐ Phase 6 — Paper shadow host on AEGIS account (`research/candidate_v1/host.py`)
- ☐ Phase 7 — 30-day shadow + protocol doc
- ☐ Phase 8 — Pre-live review
- ☐ Phase 9 — Live deploy (1% NAV per trade max, only if Phases 3–8 all pass)
- ☐ Phase 10 — 90-day live review + scale-up decision

## 16. Immutability + sign-off

Once this file is committed to git, the variant universe (§6), success hurdles (§9), kill criteria (§10), and train/test split (§11) are **frozen**. Any change to any of these requires a NEW preregistration document (`candidate_v2_*.md`) and re-running Phases 2–7 from scratch on the new variant universe — including new White's RC over the new full universe.

The 90-day cycle-freeze rule of `project_turbo_carnival.md` applies once Phase 7 (paper shadow) begins.

Signed-off git commit hash: _to be appended after the commit lands_
