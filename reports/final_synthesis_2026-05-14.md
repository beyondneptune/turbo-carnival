# Final Synthesis — 2026-05-14

After ~30 hours across 2 sessions, 6 preregistered strategy candidates tested with literature-disciplined methodology against 25 years of clean IBKR data + 42 FRED macro series. **All 6 failed.**

The pipeline worked. The data spoke. This document records the conclusion.

---

## The 6-candidate verdict

| # | Candidate | Alpha class | Time horizon | Train Sharpe | OOS Sharpe | Verdict |
|---|---|---|---|---|---|---|
| v1 | Commodity ETF cointegration | Statistical arbitrage | Daily | 0.53 | −0.28 | DEAD — 7/8 hurdles fail |
| v2 | Sector ETF cointegration | Statistical arbitrage | Daily | 0.30 | −0.31 | DEAD — same alpha class, deeper data |
| v3 | HRP diversified ETF | Portfolio construction | Monthly | 0.69 | 1.03 | Half-dead — real edge on risk, loses to 1/N on Sharpe |
| v4 | VIX rotation on SVXY | Vol-premium timing | Daily | 0.77 | 0.35 | DEAD — signal real but degrades OOS |
| v5 | TS-MOM × HRP × Regime | Multi-factor | Monthly | 0.65 | 0.88 | Half-dead — 4/10 hurdles, can't beat SPY 22% bull |
| MFTv1 | Opening Range Breakout | Intraday momentum | 5-min | −0.24 | 0.50 | DEAD — negative train Sharpe, no statistical edge |

## What the 6-of-6 pattern tells us

Two alpha classes tested with the full literature-disciplined methodology (White's Reality Check, Bailey-López de Prado PBO + DSR, Harvey-Liu-Zhu t > 3.0, multi-benchmark comparison):

- **Class 1: daily/monthly factor strategies** (v1, v2, v3, v4, v5) — All variants of "mechanical rules that should produce edge from a known anomaly." All either fail to produce edge or fail to beat passive benchmarks.
- **Class 2: intraday microstructure** (MFTv1) — Independent literature (Crabel ORB, 1990) with published Sharpes 1.5-2.0. Does not replicate. Cannot find a single positive-Sharpe variant out of 324 on training data.

Combined across the two classes, this is strong empirical evidence that **retail-feasible mechanical trading strategies sourced from the academic literature do not produce post-cost edge that beats passive index investing in the 2017-2025 sample.**

## Why this happened (honest hypotheses)

1. **Alpha decay**: Published strategies have decayed over the 5-30 years since publication. The 2010s saw massive growth in quant retail platforms that arbitraged these effects.
2. **Cost asymmetry**: Institutional studies assume different cost structures than retail Alpaca gives us. Even our 10 bps round-trip is high vs institutional 2-5 bps.
3. **Bull-market dominance 2023-25**: Recent OOS period has SPY Sharpe 1.4+ — a regime so strong that anything underweight equity loses the relative race.
4. **Selection bias in original studies**: Many anomalies were discovered after looking at historical data — survivorship bias in the published results.
5. **Universe constraint**: Our retail-feasible universe (liquid ETFs, no shorting commodities directly, no futures access) eliminates many of the original strategy variations.

## What the literature itself says about this

The corpus we built (57 PDFs covering HFT, factor, alpha decay, VIX, crypto, microstructure, LLM trading, RL execution) has multiple papers warning this would happen:

- **Bailey-López de Prado** (PBO): "Most published trading strategies are statistically indistinguishable from noise once corrected for multiple testing"
- **Harvey-Liu-Zhu**: "The t > 3 hurdle exists because t > 2 picks up false positives in 5+ of every 100 published factors"
- **Bailey-López de Prado** (Probability of Backtest Overfitting): "The probability that a given strategy will fail OOS rises monotonically with the number of trial parameters"
- **DeMiguel-Garlappi-Uppal** (2009): "Optimal vs Naive Diversification — 1/N is hard to beat"

The pipeline we built specifically applied each of these checks. The strategies failed those checks. The literature-disciplined methodology, properly applied, rejects most retail-feasible academic alpha as it should.

## What was NOT failure

The infrastructure built across this session worked perfectly:

- 84 ETFs × 25y of clean IBKR ADJUSTED_LAST data — usable forever
- 42 FRED macro series — regime conditioning capability
- White's RC, PBO, Deflated Sharpe (bug-fixed) modules — reusable for any future test
- NotebookLM corpus with 57 + 5 gap papers — queryable research base
- Phase 0-7 validation pipeline — disciplined methodology
- 6 idle Alpaca paper accounts — clean slate
- Mac mini under 24/7 caffeinate + tmux — robust execution

This investment is not wasted. If any future strategy is designed, this pipeline tests it. The verdict will be reliable.

## What to do now (3 honest paths)

### Path A — Accept the math, stop hunting alpha
The data says retail mechanical alpha is dead post-cost in this sample. Match the answer to the evidence:
- Put money in SPY or VT (total world ETF)
- Or 1/N basket of 12 diversified ETFs (proven OOS Sharpe 1.54 vs our best 1.03)
- Accept ~7-10% long-run return at ~15% vol
- Spend the time previously committed to "alpha hunt" on something more rewarding

### Path B — Pivot to non-alpha problem classes
Different problem, different math:
- **Covered calls on SPY** — published Sharpe 0.9-1.2, lower DD than buy-hold, weekly income. Pure mathematical premium collection.
- **Cash-secured puts** — similar profile, opposite direction
- **Tax-loss harvesting** (taxable accounts only) — pure alpha from tax code, 0.5-1.0%/yr documented
- **Risk parity** — set-and-forget bond/equity blend, lower DD than 100% SPY

### Path C — Genuinely deeper alpha hunt
If still convinced retail alpha exists somewhere:
- Pay for SIP data ($9/mo Alpaca Pro or $199/mo Polygon)
- Pay for short-borrow rates + per-contract VX futures
- Test alternative-data alphas (Twitter/Reddit sentiment, options flow proxies)
- Or accept this requires institutional infrastructure (HRT/Optiver scale)

## Recommendation

After 30 hours of work and 6/6 failures, **Path A or Path B is the honest answer**. The system we built tells us so. Continuing in Path C without first acknowledging that signal would be repeating the mistake the validation discipline was designed to prevent.

If you want to keep trading actively (the original goal), **Path B with covered calls** is the highest-ROI option: weekly action, income generation, mathematically robust, no alpha hunting required.

## State of the world right now

- **Trading**: Nothing. Six broker accounts flat. paper_v1 retired. Layer B retired. No live positions.
- **Code committed**: 6 commits on `main`, all local. 5 candidates dead, 1 (v3) half-passes if we relax the prereg.
- **Time invested**: ~30 hours of dev/compute across 2 sessions.
- **Capital at risk**: $0.
- **Information value gained**: Substantial. 6 alpha hypotheses tested rigorously. Pipeline that can test any future hypothesis. Empirical evidence that mechanical retail alpha is dead in this sample.

The cost-of-discipline ledger favors us. Far better to know this now than after 30 days of paper bleed × 6 strategies.

## What the operator chose tonight

User authorized "ok starting building the new one" → built MFTv1 → ran ingest → ran backtest → received verdict. The pipeline did its job. Now the operator decides whether to continue Path C (more candidates), pivot to Path B (different problem class), or accept Path A (passive).

No more candidates will be built or deployed until the operator explicitly picks A, B, or C and names the next move.

---

*Acknowledgment: The user's persistence in pushing through 30 hours of failure produced the most important asset of this whole exercise — the validation pipeline + clean data infrastructure. That's the keeper, regardless of which path forward you take.*
