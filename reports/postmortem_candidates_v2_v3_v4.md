# Postmortem — Candidates v2 / v3 / v4 (2026-05-14)

**Verdict:** All three KILLED at Phase 3.
**Cumulative tally (since 2026-05-13 shutdown):** 4 of 4 candidates failed Phase 3 hurdles. Validation pipeline rejected every retail-feasible strategy class tested.

This is one consolidated postmortem because the three failures happened in quick succession and share important common-cause analysis.

---

## Headline numbers

| Candidate | Class | Best train Sharpe | OOS Sharpe | White RC p | DSR | t-stat | All hurdles? |
|---|---|---|---|---|---|---|---|
| **v1** commodity ETF cointegration | stat-arb | 0.528 | **−0.277** | 0.921 | 0.044¹ | 1.49 | 1/8 pass |
| **v2** sector ETF cointegration | stat-arb | 0.593 | 0.706 | 0.913 | 0.064 | 1.68 | 0/5 pass |
| **v3** HRP diversified ETF | portfolio | 0.769 | 0.759 | n/a² | n/a² | n/a² | 1/6 pass |
| **v4** VIX rotation on SVXY | timing | 0.733 | 0.738 | **0.025 ✓** | 0.266 | 2.07 | 4/9 pass |

¹ With the fixed DSR formula (v1 was rerun after the patch — see below).
² HRP hurdles are different (vs-1/N benchmark, not vs zero), so RC/DSR/t-stat are not directly comparable.

## DSR bug fix

`research/candidate_v1/backtest.py:deflated_sharpe()` was patched on 2026-05-14 to fix a unit-mixing bug that clamped output to 1.0 in v1's first run. The formula now properly converts the observed annualized Sharpe to daily, computes the expected max of N standard-normal draws, scales by 1/sqrt(T−1), and applies the CDF transform.

Validation: observed Sharpe 0.528 with 810 trials at T=2014 now returns DSR = 0.044 (correctly very low; was clamped to 1.0). v1's original "DSR=1.0 pass" was an artifact; the corrected number also fails the >0.95 hurdle.

## What each candidate told us

### v2 — sector ETF cointegration (FAIL)

Same pattern as v1 (commodity ETFs), slightly cleaner numbers:
- OOS Sharpe of 0.71 (vs v1's −0.28) confirms the v1 postmortem hypothesis: commodity ETFs were structurally noisier than sector ETFs (roll yield, basis differentials).
- Yet the cointegration strategy class STILL fails — even with a more theoretically appropriate universe (sectors share an explicit common factor: S&P 500).
- White's RC p = 0.91 across the same 810-variant grid → the best-variant Sharpe of 0.59 is indistinguishable from data-mining noise once corrected for multiple testing.
- PBO = 0.65 (worse than v1's 0.68) → strategy is overfit to training data.

**Conclusion:** Daily-rebalance mechanical pairs trading on retail ETFs is a dead alpha class in 2015–2025.

### v3 — HRP-weighted diversified ETF basket (FAIL on Sharpe, PASS on DD)

Most interesting "failure":
- HRP OOS Sharpe 0.76 vs 1/N benchmark OOS Sharpe **1.24** → HRP loses to naive diversification by half a Sharpe point. This is the classic DeMiguel-Garlappi-Uppal 2009 result ("Optimal vs Naive Diversification") in action: 1/N is genuinely hard to beat OOS, especially in a strong-equity bull market regime where the optimal weights would have been "be all-in equity," which HRP refuses to do because the equity ETFs are correlated.
- BUT HRP cuts OOS max DD by 25.8% vs 1/N (10.2% vs 7.6%). That's a real risk benefit, just not enough to win the Sharpe race.
- The prereg required `OOS_sharpe_diff ≥ +0.15`. Observed diff was **−0.49**. Hurdle definitively failed.

**Conclusion:** HRP works as advertised on the risk dimension but loses the return race in the recent bull regime. May be worth revisiting in a future bear/sideways regime, or with a modified hurdle structure ("equal-Sharpe but with materially lower DD") if the user explicitly wants drawdown protection over alpha.

### v4 — VIX term-structure rotation on SVXY (FAIL on magnitude, PASS on statistical significance)

Most interesting candidate from a literature standpoint:
- **White's Reality Check p = 0.025 PASSES** — first candidate to demonstrate real, statistically significant edge after multi-testing correction across 162 variants.
- **PBO = 0.40 PASSES** — not overfit.
- **Train→OOS Sharpe shrinkage = 1%** (0.733 → 0.738) — the signal is stable across periods, also the only candidate with this property.
- **OOS max DD = 6.9%** — well within the 30% cap.

Why it fails the formal gates:
- OOS Sharpe = 0.74 < 1.0 hurdle → real signal but small magnitude
- OOS Sharpe < SPX buy-and-hold Sharpe of 1.40 in 2023-2025 → bull market beat the strategy
- DSR = 0.27 < 0.80 hurdle → the multi-test correction's expected-max-SR under null is too high for a 0.74 strategy to clear, even though White's RC bootstrap rejects null
- t-stat = 2.07 < 3.0 → below Harvey-Liu-Zhu

**Conclusion:** The Avellaneda VIX term-structure thesis is empirically REAL on our SVXY data over 2015-2022 with stable OOS extension to 2025. The strategy makes ~5% net annualized at 7% volatility (Sharpe 0.74). It is killed by the prereg hurdles because the literature-disciplined bar requires "beat the dumbest benchmark (SPX buy-hold)" AND "DSR > 0.80" AND "t > 3.0" — three additional constraints designed to prevent deploying tiny-but-real edges that aren't worth the operational complexity.

This is an honest "deploy?" decision point. Strictly per prereg §9, candidate is killed. Pragmatically: this is the FIRST candidate the system found real signal in. If the user has explicit appetite for low-risk-low-return strategies (Sharpe 0.7, DD <10%) that pair with an equity benchmark for diversification rather than replace it, v4 could be deployed at small size as a diversifier. That decision is outside the prereg's scope and requires explicit operator override.

## Common cause analysis — why 4 of 4 failed

Looking across v1/v2/v3/v4, three structural factors recur:

1. **The 2023-2025 OOS window is an extreme bull-equity regime.** SPX OOS Sharpe = 1.40 (vs typical long-run ~0.4). Anything that doesn't take full equity beta loses the Sharpe race against the benchmark. Both HRP (which down-weights equities for diversification) and VIX rotation (which has near-zero equity beta) are penalized.

2. **The literature-disciplined hurdles are HARD.** Harvey-Liu-Zhu t > 3.0 implies a published Sharpe of ~2.0 at T = 5 years to declare real edge. Most retail strategies achieve 0.5-1.0 honest OOS Sharpe — well above zero, below the formal bar.

3. **Multiple-testing correction is brutal.** With 810 / 72 / 162 variants tested, the expected max Sharpe under the null is high enough that even genuinely positive strategies (v4) fail DSR.

The validation pipeline is calibrated for "publish-worthy" alpha discovery, which is a much higher bar than "retail-deployable." This is by design — but it means the system will reject many strategies that would be marginally profitable in practice.

## Options for next iteration

1. **Relax the hurdles for retail deployment context.** Drop t > 3.0 to t > 2.0; drop DSR > 0.80 to > 0.50; require only "beat zero net of costs" instead of "beat SPX buy-and-hold." If we'd applied this looser bar, v4 would have passed (PBO, RC, stable shrinkage, real OOS Sharpe). RISK: lowering the bar reintroduces the data-snooping risk that started this whole exercise. Need to be explicit.

2. **Deploy v4 as a diversifier alongside an equity core.** Treat v4 as a SVXY-tilt that pairs with an SPY/equity benchmark rather than replaces it. Combined portfolio = 80% SPY + 20% v4. Expected combined Sharpe ≈ 1.0+ with lower DD than pure SPY. This is what Avellaneda's paper actually advocates — VIX strategies as portfolio overlays, not standalone.

3. **Stay disciplined and try more candidates.** Likely candidates in the corpus we haven't tried:
   - **Trend-following on crypto** (s10479-019-03357) — works on ETH/XRP/LTC; we deferred per "skip crypto"
   - **Momentum on equity factors** (Goyal-Santa Clara residuals)
   - **Funding-rate carry on retail-accessible perpetuals** (already tested via Kraken yesterday — failed; magnitude below threshold)
   - **Pedersen Dynamic Trading** as an aggregator across the partial signals we have

4. **Accept the literature's verdict** — retail-feasible mechanical alpha is hard. Pivot the entire program to portfolio construction (the friend's `Portfolio_Opt` direction): pick a good diversified basket, rebalance monthly, accept market Sharpe. No alpha hunting.

## Honest cost ledger

- 4 candidates designed, preregistered, ingested, backtested in 18 hours of elapsed wall time
- $0 of paper or real capital deployed
- ~5 hours of total compute (v1+v2 at 23 min each; v3 quick; v4 quick)
- 4 dead strategies caught BEFORE any 30-day shadow window
- Comparable cost-of-skipping-validation: 4 × 30-day shadows × monthly equipment/operator attention ≈ 4 months of bleed + 4 sequential disappointments

The validation pipeline is functioning correctly. The lesson is the literature is honest: most published "edge" doesn't replicate at retail scale with proper multi-testing correction.

## Action

Touched STOP files:
```
research/candidate_v2/live/STOP_CANDIDATE_V2
research/candidate_v3/live/STOP_CANDIDATE_V3
research/candidate_v4/live/STOP_CANDIDATE_V4
```

All three candidates are immutable per their respective preregistration §16. To revisit any class, write a NEW `candidate_vN.md` with adjusted hurdles or methodology and rerun the full variant sweep.

The user decision is now option 1, 2, 3, or 4 from §"Options for next iteration."
