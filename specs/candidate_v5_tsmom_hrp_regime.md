# Candidate v5 — TS-MOM × HRP × Regime Gate

**Status:** PREREGISTERED · DEPLOY-IF-PASSES authorized
**Date:** 2026-05-14
**Class:** Multi-asset cross-sectional + time-series momentum, regime-conditional
**Predecessor:** v1-v4 all killed; v5 is the first candidate based on the corpus's *robust* academic alphas instead of edge-case ones

This candidate is fundamentally different from v1-v4 because it combines three independently-validated effects, none of which we previously tested:

1. **Time-series momentum (TS-MOM)** — Moskowitz, Ooi, Pedersen (2012) — most-replicated cross-asset anomaly in 40 years of literature; survives every multi-testing meta-analysis run against it.
2. **Hierarchical Risk Parity (HRP)** — López de Prado (2016) — already verified empirically in v3 (RC p=0.0245, 43% DD cut vs 1/N on our 25y data).
3. **Regime gating via FRED macro** — combines NFCI, yield curve, VIX, NBER recession into a binary "stress vs benign" classifier.

Three statistically independent sources of edge stacked. Each source individually published; combined explicitly here for the first time on our universe + data.

---

## 1. Thesis (one sentence)

A long-only diversified ETF basket where (a) only ETFs with positive 12-month excess return are held, (b) weights are HRP-determined inversely proportional to subcluster variance, and (c) total exposure is gated down by 50% in macro-stress regimes, produces a Sharpe ratio that beats both naive 1/N and SPY buy-and-hold OOS, with drawdowns under 20%.

## 2. Market / asset class / venue

- **Universe (12):** SPY, QQQ, IWM, EFA, EEM (equity), TLT, IEF, LQD, HYG (fixed income), GLD, DBC, VNQ (alternatives)
- **Same as v3** for direct comparability
- **Account:** AEGIS (Alpaca paper) — fresh, no overlap with retired Layer-B
- **Settlement:** T+1, long-only, no leverage

## 3. Time horizon

**Monthly rebalance**, last trading day of month, identical cadence to v3.

## 4. Method family + specific algorithm

```python
# At rebalance date t:

# Step 1: TS-MOM filter — drop any asset with negative 12m excess return
r_12m = (price[t] / price[t - LOOKBACK_DAYS]) - 1  # 12-month return
r_f_12m = (1 + r_f.mean()) ** (LOOKBACK_DAYS / 252) - 1  # 12-month risk-free
survivors = [s for s in UNIVERSE if (r_12m[s] - r_f_12m) > 0]

# Step 2: HRP weights on survivors only
if len(survivors) == 0:
    weights = {}  # everything goes to cash
else:
    weights = hrp_weights(returns[survivors].iloc[-HRP_LOOKBACK:],
                          linkage="single",
                          weight_floor=0.02,
                          weight_cap=0.25)

# Step 3: Regime gate — compute stress score from FRED signals
stress_signals = sum([
    vix[t] > VIX_STRESS_TH,          # default 30
    t10y2y[t] < CURVE_STRESS_TH,     # default 0 (inverted)
    nfci[t] > NFCI_STRESS_TH,        # default 0.5
    usrec[t] == 1,                   # NBER recession
])
in_stress = stress_signals >= REGIME_STRESS_COUNT  # default 2

# Step 4: Apply gate to total exposure
if in_stress:
    # Scale down to defensive cash position
    weights = {s: w * DEFENSIVE_EXPOSURE_PCT for s, w in weights.items()}
# else: weights as-is (sums to ~1 by HRP construction)

# Cash fills the difference
total_invested = sum(weights.values())
cash_pct = 1 - total_invested  # held in BIL or money-market equivalent
```

## 5. Variant universe (LOCKED for White's RC)

| Parameter | Values | Count |
|---|---|---|
| LOOKBACK_DAYS (TS-MOM window) | 126, 189, 252 (6/9/12mo) | 3 |
| HRP_LOOKBACK | 252, 504 | 2 |
| REGIME_STRESS_COUNT (signals needed) | 1, 2, 3 | 3 |
| DEFENSIVE_EXPOSURE_PCT | 0.0, 0.3, 0.5, 0.7 | 4 |
| VIX_STRESS_TH | 25, 30, 35 | 3 |
| REBAL_FREQ | M, Q | 2 |

Total: **3 × 2 × 3 × 4 × 3 × 2 = 432 variants**

Per Harvey-Liu-Zhu (2016), t-stat hurdle for 432 trials: **t > 3.0** (same as v1-v4).

## 6. Sizing rule

HRP weights, scaled per regime gate. No leverage. Single account. Per-asset cap of 25% (HRP `weight_cap`) prevents concentration even if HRP would assign higher.

## 7. Risk overlay (already inherent to the strategy)

- TS-MOM filter is itself a drawdown protector (excludes anything declining)
- HRP is itself a concentration protector
- Regime gate is itself a stress protector
- Additional: **daily 3% session-loss circuit breaker** (halt new entries that day if intraday loss > 3% NAV)

No separate Phase 5 overlay needed — the strategy IS the overlay.

## 8. Success hurdles (must ALL pass)

| Hurdle | Threshold | Source |
|---|---|---|
| White's Reality Check p-value | < 0.05 | White 1999 |
| PBO via combinatorial CV | < 0.50 | Bailey-Lopez de Prado 2014 |
| Deflated Sharpe Ratio | > 0.80 | Bailey-Lopez de Prado 2014 |
| Best-variant t-statistic | > 3.0 | Harvey-Liu-Zhu 2016 |
| OOS Sharpe (net of 5bps/side costs) | > 1.0 | retail viability |
| OOS Sharpe > 1/N benchmark Sharpe | by ≥ 0.10 | beats naive diversification |
| OOS Sharpe > SPY buy-and-hold Sharpe | by ≥ 0.10 | beats market |
| OOS max DD | < 25% | retail-tolerable |
| OOS Calmar | > 1.0 | risk-adjusted return |
| Sharpe shrinkage train→OOS | < 50% | overfitting guard |

This is a STRICTER bar than v3 (which only had to beat 1/N OR cut DD). v5 has to beat BOTH benchmarks AND multi-testing AND PBO AND DSR.

## 9. Kill criteria (during paper shadow)

- Realized 30-day Sharpe outside 1.5σ of backtest bootstrap CI → extend shadow
- Single-day loss > 1.5× backtest worst-day → investigate, may kill
- 5+ consecutive months with TS-MOM filter excluding ALL ETFs (universe pathological) → halt + investigate
- Regime gate fires for 6+ consecutive months at maximum defensive → strategy effectively in cash, kill if no risk-on opportunity
- `submit_failed` for 3+ consecutive cycles → infra alert, auto-STOP

## 10. Data needs

- 12 ETFs daily 2001-2026 — **already in `data/cache/candidate_v3/`** from IBKR pull
- FRED macro: VIX (VIXCLS), T10Y2Y, NFCI, USREC, DGS3MO (risk-free), DFF — **already in `data/offline_context/fred/`**
- Risk-free rate from DGS3MO for excess return calc

**No new data needed.** Train 2002-2022, OOS 2023-2025 (matching v3 split).

## 11. Capacity bound

Same as v3: ~$10M AUM before liquidity becomes binding on smallest ETFs (HYG, VNQ at 20% allocation). Initial paper shadow: 100% of AEGIS NAV ($98k as of 2026-05-13).

## 12. Honest priors

- **TS-MOM alone** on diversified ETF basket: published Sharpe 0.8–1.2 net of costs (Moskowitz-Ooi-Pedersen 2012, many replications)
- **HRP alone** on similar basket: published Sharpe 0.7–1.0, but loses to 1/N in bull markets (v3 finding)
- **Combined (TS-MOM × HRP)**: no published estimate but the literature strongly suggests TS-MOM filter helps HRP avoid bull-market-equity-trap
- **Regime gate** addition: incremental 0.1–0.2 Sharpe in stress periods (NFCI literature)
- **My realistic prior:** 70-80% probability of passing all 10 hurdles. The remaining uncertainty is mostly "does TS-MOM survive Harvey-Liu-Zhu t>3 hurdle after 432-variant correction" — that's the failure mode to watch.

## 13. Phase progression

- ☑ Phase 1 (this document)
- ☑ Phase 2 (data already ingested from v3 + FRED from other session)
- ☐ Phase 3 — backtest 432 variants + multi-testing
- ☐ Phase 4-5 collapsed into the strategy itself
- ☐ Phase 6 — paper shadow host on AEGIS
- ☐ Phase 7-10 standard funnel

## 14. Citation block

Primary references:
- **Moskowitz, Ooi, Pedersen** (2012) — "Time Series Momentum" JFE — TS-MOM canonical paper (NOT in corpus, will add)
- **López de Prado** (2016) — "Building Diversified Portfolios that Outperform OOS" (ssrn-2708678 in corpus) — HRP
- **White** (1999) — "A Reality Check for Data Snooping" — multi-testing correction
- **Bailey-Lopez de Prado** (2014) — "Probability of Backtest Overfitting" — PBO + DSR
- **Harvey-Liu-Zhu** (2016) — "…and the Cross-Section of Expected Returns" — t > 3 hurdle

Regime gate references (in corpus FRED data):
- Chicago Fed NFCI documentation
- Bollerslev-Zhou (2007 Fed paper in corpus, ssrn-equivalent) — VIX as macro indicator
- NBER USREC dating

## 15. Sign-off

Committed at git hash _to be appended_. Operator authorized 2026-05-14 — "yes v5, kill paper_v1."
