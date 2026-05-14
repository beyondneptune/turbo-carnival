# Candidate v2 — Sector ETF Cointegration

**Status:** PREREGISTERED · NOT YET DEPLOYED
**Date:** 2026-05-14
**Predecessor:** v1 (cointegration commodity ETF pairs — killed at Phase 3 same day)
**Differs from v1 in:** universe only — replaces 10 commodity ETFs with 9 SPDR sector ETFs that share a known macro driver (S&P 500). Hypothesis: cointegration is more plausible across instruments sharing an explicit common factor than across commodity ETFs contaminated by roll-yield + basis differentials.

All other parameters — variant grid, hurdle thresholds, train/test split, sizing — are identical to v1 to make the comparison clean. If v2 also fails, the failure is the strategy CLASS, not the universe choice.

---

## 1. Thesis (one sentence)

SPDR sector ETFs all track the same underlying S&P 500 universe in different proportions, so a mean-reverting Engle-Granger spread between two sector ETFs has a positive expected return when the rolling z-score of the residual exceeds ±2σ, exiting at 0, with a hard stop at ±4σ.

## 2. Market / asset class / venue

- **Instruments (universe of 9):** XLE (energy), XLF (financials), XLK (technology), XLY (consumer discretionary), XLP (consumer staples), XLI (industrials), XLB (materials), XLV (health care), XLU (utilities)
- **Excluded:** XLRE (real estate, listed 2015-10) and XLC (communication services, listed 2018-06) for incomplete pre-2015 history
- **Exchange:** NYSE Arca via Alpaca paper account (AEGIS — same as v1 if shadowed)
- **Settlement:** T+1 US equity, Reg-T margin, no leverage

## 3. Time horizon

**Daily.** Close-to-close one decision per market close, identical to v1.

## 4. Method family and specific algorithm

Same as v1: Engle-Granger 2-step rolling cointegration on log-price spread, z-score normalization, mechanical entry/exit/stop. See `specs/candidate_v1_cointegration_pairs.md` §4-§5 for full pseudocode (re-used verbatim).

## 5. Variant universe (LOCKED for White's RC)

Identical to v1:

| Parameter | Values | Count |
|---|---|---|
| WINDOW | 20, 30, 45, 60, 90 | 5 |
| ENTRY_THRESHOLD | 1.5, 2.0, 2.5 | 3 |
| STOP_THRESHOLD | 3.0, 4.0, 5.0 | 3 |
| EXIT_THRESHOLD | 0.0, 0.5 | 2 |
| COINT_PVALUE_THRESHOLD | 0.01, 0.05, 0.10 | 3 |
| MAX_HOLD | 30, 60, 90 | 3 |

Total: **810 variants × C(9,2)=36 pairs = 29,160 pair-backtests** (vs 36,450 in v1; fewer pairs).

## 6. Sizing / risk overlay / hurdles / kill criteria

Identical to v1 §7-§10. Reuse the entire pipeline:
- 5% NAV per pair, max 4 simultaneous open pairs
- Hurdles: White RC p<0.05, SPA p<0.05, PBO<0.50, **DSR>0.95 (fixed formula, was buggy in v1)**, t>3.0, OOS net Sharpe>1.0, 95% CI lo>0, max DD<20%, Calmar>1.0, shrinkage<50%
- Kill: same kill criteria as v1 §10

## 7. Data needs + train/test split

Identical to v1 §11:
- yfinance primary, Alpaca IEX fallback
- 2015-01-01 → 2026-05-13
- Train 2015–2022, OOS 2023–2025
- 5 purged k-fold folds with 22-day purge gap

## 8. Phase progression

- ☐ Phase 1 (this document)
- ☐ Phase 2 — ingest 9 sector ETFs (reuse `research/candidate_v1/ingest.py` with universe swap)
- ☐ Phase 3 — backtest with fixed DSR (reuse `research/candidate_v1/backtest.py`)
- (downstream phases gated on Phase 3 pass)

## 9. Honest expectations

Per the v1 postmortem §2, the failure was attributed in part to "commodity ETFs aren't pure spot proxies" — roll yield, basis differentials, ETF rebalance schedules contaminate the cointegration relationship. Sector ETFs do not have these issues: each holds underlying equities with no roll-yield, no basis to a futures contract, no spread contamination. So mechanistically v2 has a better chance than v1.

That said: sector ETFs are highly correlated by construction (all are slices of S&P 500), which means the Engle-Granger residuals may simply track sector-rotation rotation rather than reverting to a mean. The strategy could fail in a different way than v1 (over-correlation) rather than the v1 way (under-cointegration).

Realistic prior: 30-40% probability v2 passes Phase 3. The literature has no direct paper on sector-ETF pairs trading in the corpus, so this is an extrapolation of the IJSR commodity-pair finding to a more theoretically appropriate universe.

## 10. Signed-off git commit hash

_to be appended after commit_
