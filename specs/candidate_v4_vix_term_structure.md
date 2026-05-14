# Candidate v4 — VIX Term-Structure Rotation

**Status:** PREREGISTERED · NOT YET DEPLOYED
**Date:** 2026-05-14
**Source:** Avellaneda, Li, Papanicolaou, Wang (2021) "Trading Signals in VIX Futures" (arxiv 2103.02016)
**Class:** Single-instrument timing on volatility-premium harvest

This candidate runs the Avellaneda VIX1/VIX3M term-structure rotation as a single-account daily-decision strategy on SVXY (short-vol ETF), the retail-accessible proxy for shorting front-month VIX futures vs. long mid-curve VIX futures. Phase 3 backtests on yfinance ^VIX, ^VIX3M, and SVXY daily data 2015-2025.

This is NOT the same as the prior "svxy_sweet" lane we killed yesterday. The previous lane had:
- No preregistration
- No variant universe lock
- No backtest with multi-testing correction
- No declared kill criteria
- Was running concurrent with 8 other zombies

This candidate has all of those.

---

## 1. Thesis (one sentence)

When the front-month VIX futures contract trades below the 3-month VIX (contango regime, VIX/VIX3M < 1), short-vol exposure earns the term-structure roll yield; circuit-break to flat when the regime inverts or when spot VIX exceeds a stress threshold.

## 2. Market / asset class / venue

- **Instruments:**
  - **Signal:** ^VIX, ^VIX3M (CBOE indices, yfinance free)
  - **Execution:** SVXY (ProShares Short VIX Short-Term Futures ETF, -0.5× daily reset post-2018)
- **Venue:** NYSE Arca via Alpaca paper account (ATLAS — preserved for this candidate per the 6-account map)
- **Settlement:** T+1 US equity, Reg-T margin, no leverage beyond SVXY's embedded -0.5×

## 3. Time horizon

**Daily.** One decision per US market close. Entry/exit at next morning's open. Hold until signal change.

## 4. Method family and specific algorithm

Per Avellaneda et al. (arxiv 2103.02016), the VIX term-structure rotation strategy. Adapted for retail SVXY proxy:

```python
# At each market close:
ratio = VIX / VIX3M
in_contango = ratio < CONTANGO_THRESHOLD  # default 1.0

# Signal smoothing — require N consecutive contango days to enter
contango_count = consecutive_days_below_threshold(ratio, CONTANGO_THRESHOLD)

# Stress filter
high_vol_regime = VIX > VOL_FILTER

# Position rule
target_position = 0  # flat
if in_contango and contango_count >= SIGNAL_CONFIRM_DAYS and not high_vol_regime:
    target_position = POSITION_SIZE_PCT_NAV  # default 50% NAV long SVXY

# Cooldown after stress exit
if just_exited_stress_within_last(REENTRY_COOLDOWN_DAYS):
    target_position = 0
```

## 5. Variant universe (LOCKED for White's RC)

| Parameter | Values | Count |
|---|---|---|
| CONTANGO_THRESHOLD | 1.00, 0.95, 0.90 | 3 |
| VOL_FILTER (force flat if VIX >) | 25, 30, 40 | 3 |
| POSITION_SIZE_PCT_NAV | 0.25, 0.50 | 2 |
| REENTRY_COOLDOWN_DAYS | 0, 5, 10 | 3 |
| SIGNAL_CONFIRM_DAYS | 1, 3, 5 | 3 |

Total: **3 × 3 × 2 × 3 × 3 = 162 variants.**

## 6. Sizing rule

`POSITION_SIZE_PCT_NAV` of total NAV in SVXY when signal is on, 0 otherwise. No fractional Kelly applied at Phase 3 — sizing is a Phase 5/7 overlay decision. Variant grid covers two crude sizes (25%, 50%); full Kelly tuning is a downstream tuning.

## 7. Risk overlay (Phase 5)

- **CVaR cap:** Rockafellar-Uryasev LP wrapper, daily 95% CVaR < 5% NAV
- **Vol target:** scale position by `0.12 / realized_30d_vol` if realized vol > 18% annualized
- **Daily circuit breaker:** if session loss > -5% NAV, halt new entries until next session
- **VIXmageddon protection (2018 Feb 5 lesson):** force liquidation at market close if VIX > 60 intraday, regardless of position

## 8. Success hurdles

| Hurdle | Threshold | Source |
|---|---|---|
| White's RC p-value | < 0.05 | White 1999 |
| PBO | < 0.50 | Bailey-Lopez de Prado |
| DSR | > 0.80 | Bailey-Lopez de Prado |
| Best-variant t-stat | > 3.0 | Harvey-Liu-Zhu |
| OOS net Sharpe (with 10bps round-trip costs) | > 1.0 | Avellaneda paper |
| OOS max DD | < 30% | conservative for SVXY |
| OOS Calmar | > 1.0 | risk-adjusted return |
| Sharpe shrinkage train → OOS | < 50% | overfitting guard |
| OOS Sharpe > SPY buy-and-hold Sharpe | yes | beat the dumbest benchmark |

## 9. Kill criteria (during shadow)

- Single-day loss > -15% (SVXY tail event signal)
- 5-day cumulative loss > -25%
- Realized 30-day Sharpe < 0 (signal-off)
- VIX > 50 occurred and circuit didn't fire (infra bug)
- Tracking error vs Avellaneda predicted P&L > 5σ in 5+ rebalance windows

## 10. Data needs + train/test split

- yfinance for ^VIX, ^VIX3M, SVXY, ^GSPC (SPY proxy for benchmark)
- 2015-01-01 → present
- Train 2015–2022 (includes Feb 2018 VIXmageddon stress event)
- OOS 2023–2025
- Stress regime test: 2018 Feb (XIV-mageddon), 2020 Mar (COVID), 2022 (inflation regime)

## 11. Capacity bound

SVXY ADV ~$15M, with bid/ask spread typically 1-3 cents. Retail capacity: up to $5M position size before slippage materially erodes edge. Initial deployment: $50k max per position (~$100k NAV × 50% target).

## 12. Honest caveats

1. **SVXY ≠ direct VIX futures.** Avellaneda's published Sharpe (1.0-11 gross, -1.4 to +6.6 net) is on continuous VIX1/VIX5 futures roll. SVXY is a -0.5× daily-reset ETF that approximates but does not replicate the strategy. Expect Sharpe degradation of 30-50% vs the paper.
2. **Post-2018 SVXY behaves differently than the pre-2018 fund.** Daily reset multiplier changed from -1× to -0.5×. Our backtest covers the post-2018 instrument.
3. **^VIX3M started Sep 2007**, so 2015+ is fully covered. No survivorship issues.
4. **Tax treatment:** SVXY is a 1099-equity ETF (not K-1). Clean for paper trading and live.

## 13. Phase progression

- ☐ Phase 1 (this document)
- ☐ Phase 2 — ingest ^VIX, ^VIX3M, SVXY, ^GSPC
- ☐ Phase 3 — backtest 162 variants + multi-test correction
- (downstream phases gated)

## 14. Citation block

Primary (in corpus):
- Avellaneda, M., Li, T. N., Papanicolaou, A., Wang, G. (2021) — *Trading Signals in VIX Futures* (arxiv 2103.02016)
- Spekulant "A Note on Trading the Term Structure of VIX Futures" — retail VIXY/VIXM variant validation

Multi-testing (to be added):
- Bailey-Lopez de Prado PBO (ssrn-2326253)
- Harvey-Liu-Zhu t > 3 hurdle (P118_and_the_cross.PDF)

## 15. Sign-off

_to be appended after commit_
