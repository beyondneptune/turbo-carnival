# Postmortem — candidate_v1 (Cointegration Commodity ETF Pairs)

**Verdict:** KILLED at Phase 3
**Date killed:** 2026-05-13
**Time to verdict:** ~3 hours from preregistration commit to kill (vs ~6 weeks calendar time to first live $1)
**Capital at risk:** $0 (caught pre-shadow)
**Outcome:** **System worked as designed.**

---

## Hurdle summary

| Hurdle | Threshold | Observed | Pass? |
|---|---|---|---|
| White's Reality Check p-value | < 0.05 | **0.921** | ✗ |
| Probability of Backtest Overfitting | < 0.50 | **0.677** | ✗ |
| Deflated Sharpe Ratio | > 1.0 | 1.0 (formula bug, see §3) | (n/a) |
| Best-variant t-statistic | > 3.0 | **1.492** | ✗ |
| OOS net Sharpe | > 1.0 | **−0.277** | ✗ |
| OOS Sharpe 95% CI lower bound | > 0 | n/a (not run; Phase 4 gated on Phase 3) | — |
| Max DD (OOS) | < 20% | 1.89% | ✓ (only because returns are tiny) |
| Calmar (OOS) | > 1.0 | **−0.115** | ✗ |
| Sharpe shrinkage train→OOS | < 50% | (0.528 → −0.277) **152% shrinkage** | ✗ |

7 of 8 quantifiable hurdles failed. Max DD only "passed" because the strategy barely generates returns at all — the absolute ann return is ~0.8% on training and **negative 0.2%** OOS.

## 1. Sweep details

- **Variant universe:** 810 (5 windows × 3 entry × 3 stop × 2 exit × 3 coint-p × 3 max-hold)
- **Pairs:** C(10,2) = 45 commodity ETF pairs across USO, UNG, GLD, SLV, DBA, DBC, CORN, WEAT, CPER, PPLT
- **Total pair-backtests:** 36,450
- **Train:** 2015-01-02 → 2022-12-30 (2014 days)
- **OOS:** 2023-01-03 → 2026-05-12 (842 days)
- **Compute time:** 23.3 minutes wall-clock (Mac mini, single-threaded numpy)
- **Best variant:** `WINDOW=20, ENTRY=2.5, STOP=3.0, EXIT=0.0, COINT_P=0.05, MAX_HOLD=90`
- **White's RC bootstrap:** B=2000 iterations, 10-day stationary blocks

## 2. Why the strategy failed (interpretation)

The mechanical Engle-Granger pairs trading rules from the IJSR paper (Mohandas, "Cointegration-Based Pairs Trading Strategy in Commodity Markets") **did not generalize** from the GSCI commodity index pairs in the paper to the retail-accessible commodity ETF universe we tested.

Concretely:

1. **Commodity ETFs aren't pure spot proxies.** USO is exposed to contango/backwardation roll yield, UNG to the same on natural gas, DBA/DBC are basket products with rebalance schedules. The "cointegration" relationship between e.g. GLD and SLV is contaminated by basis differentials that don't mean-revert on the timescale of our z-score.

2. **Most pair-days have ADF p > 0.05.** With the 30-day rolling cointegration filter at `COINT_PVALUE_THRESHOLD = 0.05`, the strategy was flat ~90% of the time across the test, which makes the "edge" — to the extent it exists — too thin to overcome 20bps round-trip costs.

3. **Survivorship-corrected commodity universe is small.** After dropping JJN/JJC (delisted 2023-07), we have 10 ETFs / 45 pairs. The IJSR paper used the much larger 24-component GSCI universe with ~276 candidate pairs. Our 45-pair set may not contain enough cointegrated commodity pairs at any given time for the strategy to deploy meaningfully.

4. **Mean reversion in commodities is regime-dependent.** 2020 COVID supply shock, 2022 inflation regime, 2023 normalization — none of these are obviously stationary processes. The strategy may have worked in a calmer 2014-2019 commodity environment but the test window includes 3-4 distinct macro regimes.

5. **Equal-dollar legs (β=1 approximation) is wrong.** The strategy assumes equal-dollar long/short captures the cointegration relationship, but the cointegration coefficient β from the rolling OLS is not always close to 1. A proper implementation would size the legs in the ratio (1, β), not (1, 1). This is documented in prereg §13.5 as a known approximation — and may explain part of the gap, though it's unlikely to flip a 152%-shrinkage strategy into a winner.

## 3. Implementation bugs noted (for next candidate)

1. **Deflated Sharpe Ratio formula is buggy.** My implementation in `research/candidate_v1/backtest.py:deflated_sharpe()` returned 1.0 in both the quick-30 run and the full-810 run. The correct Bailey-Lopez de Prado DSR for observed Sharpe 0.528 with n_trials=810, T=2014 should be near 0 (the expected max Sharpe under the null with 810 random trials at T=2014 is ≈ 3.21 / √2014 ≈ 0.072 daily, well above the observed 0.033 daily). The formula needs a unit-conversion fix before reuse. For candidate_v1 this didn't matter — the other 6 failures were definitive — but candidate_v2 must use a fixed DSR.

2. **ADF p-value approximation may be too coarse.** The hot-loop ADF approximation discretises p to {0.005, 0.025, 0.075, 0.5}. This is fine for boundary checks but loses information for fine-grained cointegration filtering. Consider replacing with statsmodels.adfuller for the per-variant final pass on the best variant before declaring "no edge."

3. **No transaction-cost sensitivity sweep.** The cost model is fixed at 5bps/side. If the strategy passed everything else but failed only on net Sharpe, varying costs would tell us whether the edge is real but cost-limited. Here it doesn't matter — gross Sharpe is already weak — but candidate_v2 should do this.

## 4. What we learned that's reusable

- **Phase 2 ingest module** is sound. yfinance-primary + Alpaca-IEX-fallback, parquet cache, validation report. Reuse for next candidate.
- **Phase 3 backtest module** has the right structure (variant sweep → RC bootstrap → PBO → DSR → t-stat), just needs the DSR bug fix. Reuse.
- **Phase 5 overlay module** (CVaR + vol target + Chandelier) is generic, not strategy-specific. Reuse.
- **Phase 6 host module** has the broker-truth reconciliation + submit_failed alerting + STOP file pattern that the 2026-05-13 ADR bug class needs. Generalize it (lift the cointegration-specific signal computation out of `run_cycle` into a strategy-specific module) for reuse.
- **The "preregistration first, sweep second, kill if hurdles fail" discipline works.** Total cost of catching this dead strategy: 3 hours and zero broker capital. Without the literature gates we would have proceeded to a 30-day paper shadow on a strategy that already failed every multiple-testing criterion.

## 5. Next candidate

Per `specs/candidate_v1_cointegration_pairs.md` §15 and the literature-thresholds memory, the second-strongest candidate is **alt-crypto technical-rule rotation** on ETH/XRP/LTC, per `s10479-019-03357-1` ("Technical trading and cryptocurrencies"). That paper explicitly demonstrates the anomaly survives Bonferroni / Holm / Benjamini-Hochberg / Benjamini-Yekutieli multiple-testing corrections out-of-sample on those three (and explicitly fails on BTC — which we already knew).

Candidate v2 should:
- Reuse Phase 2 ingest with `ccxt` for crypto (or yfinance for ETH-USD, XRP-USD, LTC-USD)
- Replicate the s10479 rule universe (the paper enumerates the technical rules tested)
- Apply the same Phase 3 multiple-testing gates
- Fix the DSR bug noted above before running

If candidate v2 also fails, the next move is to question whether retail-feasible mechanical strategies have any edge at all post-cost in 2026 markets, OR pivot to portfolio-construction work (HRP on a diversified ETF basket per the friend's `Portfolio_Opt` work, after his repo's HRP module is added to NotebookLM).

## 6. Files frozen by this kill

```
research/candidate_v1/live/STOP_CANDIDATE_V1                (touched, reason=validation_failure)
research/candidate_v1/results/backtest_summary_1778699689.json
research/candidate_v1/results/variant_metrics.csv           (810 rows, per-variant in-sample + OOS metrics)
research/candidate_v1/results/train_returns.parquet         (per-variant daily returns)
research/candidate_v1/results/oos_returns.parquet           (per-variant daily returns)
specs/candidate_v1_cointegration_pairs.md                   (immutable per §16)
```

`research/candidate_v1/` code is not deleted — it remains as a reusable scaffold for candidate_v2 and beyond.

---

**Cost-of-discipline ledger (2026-05-13):**
- $0 of paper-trading P&L wasted on a dead strategy
- 3 hours of dev/compute spent confirming what the literature predicted
- 1 dead candidate, 1 working pipeline scaffold, 4 new memory entries
- Comparable cost if we had skipped Phase 3 hurdles and gone straight to live: ~$1,000–$5,000 paper P&L per month bleed + 30 days of operator attention before the failure became obvious
