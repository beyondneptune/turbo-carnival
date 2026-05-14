# Candidate v3 — Hierarchical Risk Parity on Diversified ETF Basket

**Status:** PREREGISTERED · NOT YET DEPLOYED
**Date:** 2026-05-14
**Class:** Portfolio construction (NOT alpha generation)
**Source:** López de Prado 2016, "Building Diversified Portfolios that Outperform Out of Sample" (ssrn-2708678)

This candidate is a fundamentally different problem class than v1/v2. We are not chasing alpha — we are testing whether HRP (hierarchical risk parity) provides a documented diversification benefit on a retail-accessible 12-ETF universe vs. naive equal-weight allocation.

The hurdle structure is different: instead of "is Sharpe > 0 after multi-test correction?" the relevant test is "does HRP beat the equal-weight (1/N) benchmark on risk-adjusted returns OOS?"

---

## 1. Thesis (one sentence)

A long-only ETF basket weighted by hierarchical risk parity (HRP) — using single-linkage clustering on the correlation-distance matrix followed by recursive bisection — produces higher risk-adjusted returns than equal-weight (1/N) on a diversified 12-ETF universe over a 2015-2025 walk-forward.

## 2. Market / asset class / venue

- **Universe (12 diversified ETFs):**
  - **Equity:** SPY (US large), QQQ (US growth), IWM (US small), EFA (developed ex-US), EEM (emerging)
  - **Fixed income:** TLT (long Treasuries), IEF (intermediate Treasuries), LQD (IG corp), HYG (HY corp)
  - **Alternatives:** GLD (gold), DBC (broad commodities), VNQ (REITs)
- **Exchange:** NYSE Arca / NASDAQ via Alpaca paper account
- **Settlement:** T+1 US equity, no leverage, long-only

## 3. Time horizon

**Monthly rebalance.** Once per month at the last trading day; positions held until next rebalance.

## 4. Method family and specific algorithm

**López de Prado HRP** (Journal of Portfolio Management 42(4), 2016):
1. Compute the return covariance matrix Σ on a rolling lookback (252 days)
2. Convert to correlation matrix ρ_ij = Σ_ij / √(Σ_ii · Σ_jj)
3. Convert to distance matrix d_ij = √((1 − ρ_ij) / 2)
4. Single-linkage hierarchical clustering on d → cluster tree
5. Quasi-diagonalize the clustering: reorder rows/cols so similar assets are adjacent
6. Recursive bisection: at each split node, allocate weight inversely proportional to subcluster variance
7. Result: weights w_i ≥ 0, Σw_i = 1

Pseudocode:
```python
def hrp_weights(returns: pd.DataFrame, lookback: int = 252) -> np.ndarray:
    cov = returns.iloc[-lookback:].cov()
    corr = returns.iloc[-lookback:].corr()
    dist = np.sqrt(0.5 * (1 - corr))
    link = scipy.cluster.hierarchy.linkage(dist[np.triu_indices_from(dist, k=1)], 'single')
    order = quasi_diagonalize(link, n_assets)
    weights = recursive_bisection(cov, order)
    return weights / weights.sum()  # ensure sums to 1
```

## 5. Variant universe (LOCKED for White's RC)

Smaller than v1/v2 because HRP has fewer hyperparameters:

| Parameter | Values | Count |
|---|---|---|
| LOOKBACK_DAYS | 126, 252, 504 | 3 |
| REBALANCE_FREQ | "M", "Q" | 2 |
| WEIGHT_FLOOR | 0.0, 0.02 | 2 |
| WEIGHT_CAP | 0.20, 0.30, 0.50 | 3 |
| LINKAGE_METHOD | "single", "average" | 2 |

Total: **3 × 2 × 2 × 3 × 2 = 72 variants**.

Phase 3 White's RC tests max-Sharpe across the 72 against the null of equal-weight (1/N) on the same universe.

## 6. Sizing rule

Use HRP weights directly. No leverage. 100% NAV invested at each rebalance (less small cash for transaction costs).

Per Wu-Chung Kelly criterion paper (IEEE Xplore in corpus): consider applying fractional Kelly cap if any single HRP weight exceeds prereg WEIGHT_CAP. The cap parameter is in the variant grid.

## 7. Risk overlay

Less critical for HRP than for alpha strategies. Per López de Prado, HRP IS the risk overlay — concentration is inherently capped by the clustering structure. Light additional overlay:

- **Drift rebalance:** if any single weight drifts more than 5% from target between scheduled rebalances → force rebalance
- **Vol target:** if portfolio realized 20d vol > 18% annualized, scale all weights by `0.12 / realized_vol` (cap at 0.25–1.5)
- **Daily circuit breaker:** if session loss > -3% NAV → halt rebalances for the day

## 8. Success hurdles (DIFFERENT from v1/v2)

| Hurdle | Threshold | Source |
|---|---|---|
| OOS Sharpe (HRP) > OOS Sharpe (1/N) + 0.15 | yes | López de Prado 2016 §5 |
| OOS max DD (HRP) cut vs OOS max DD (1/N) | ≥ 20% reduction | López de Prado 2016 §5 |
| OOS Sharpe (HRP) > 0.5 net of 5bps/side costs | yes | retail viability |
| OOS Calmar (HRP) > OOS Calmar (1/N) | yes | risk-adjusted return |
| White's RC p-value (vs 1/N benchmark) | < 0.05 | multi-test correction |
| Deflated Sharpe Ratio (HRP) | > 0.80 | Bailey-Lopez de Prado |
| Sharpe shrinkage train→OOS | < 50% | overfitting guard |
| OOS HRP turnover | < 200% annualized | feasibility on retail account |

Note the HRP-vs-1/N comparison is the core test. A passing HRP doesn't need to be a star strategy in absolute terms — it needs to provably beat the dumbest baseline (1/N) by enough to justify the complexity.

## 9. Kill criteria (during paper shadow)

- Realized 30-day Sharpe (HRP) < realized 30-day Sharpe (1/N) on the same universe → flag
- Realized max DD (HRP) > realized max DD (1/N) → kill
- HRP weights become degenerate (any single weight > 70%) for 5+ consecutive rebalances → kill
- Cluster structure breaks (e.g., distance matrix has eigenvalue collapse) → halt rebalancing, investigate

## 10. Data needs + train/test split

- yfinance daily, 2010-01-01 → present (need 5 years pre-train for stable covariance estimation)
- Train (lookback warmup + initial backtest): 2010-2022
- OOS: 2023-2025
- Walk-forward: re-fit HRP weights at each rebalance date using only data from before that date
- No leakage (the "refit at every point" structure naturally enforces this)

## 11. Capacity bound

HRP scales to arbitrary capital — these are highly liquid ETFs with $10M+ ADV each. Realistic retail cap: any size up to ~$10M AUM with negligible market impact.

## 12. Honest expectations

López de Prado's 2016 paper claims HRP beats Markowitz / min-variance / equal-weight on OOS Sharpe by 0.3-0.5 across multiple universes and time windows. Independent replications (Pfitzinger-Katzke 2019, Jain et al. 2022) have produced mixed results — sometimes HRP beats 1/N by a modest margin, sometimes ties, occasionally loses in trending markets.

Realistic prior: 60-70% probability HRP passes the "beat 1/N" hurdle on our universe with meaningful margin. Higher than v1/v2 because:
1. HRP is a fundamentally different bet (diversification, not alpha)
2. Has multiple published OOS validations
3. Friend Alasteir already has a running implementation that produced 1.1 Sharpe in his backtest (per `external/Portfolio_Opt/notebooks/strategy_comparison.ipynb`)

## 13. Phase progression

- ☐ Phase 1 (this document)
- ☐ Phase 2 — ingest 12 diversified ETFs
- ☐ Phase 3 — backtest HRP vs 1/N benchmark + multi-testing correction
- (downstream phases gated on Phase 3 pass)

## 14. Citation block

Primary:
- López de Prado, M. (2016) — *Building Diversified Portfolios that Outperform Out of Sample* (ssrn-2708678 in corpus, JPM 42(4), pp. 59–69)

Secondary (in corpus):
- Bailey-López de Prado PBO (ssrn-2326253) for DSR formula
- Harvey-Liu-Zhu 2016 (P118_and_the_cross.PDF) for the t > 3 hurdle (relaxed here vs alpha hunting)
- Wu-Chung Kelly criterion (IEEE Xplore) for the optional WEIGHT_CAP rationale

External validations (NOT in corpus, user may want to add):
- Pfitzinger, J. & Katzke, N. (2019). *A constrained hierarchical risk parity algorithm with cluster-based capital allocation*.
- Jain, P. et al. (2022). *Hierarchical Risk Parity: Accounting for Tail Dependencies in Multi-asset Multi-factor Allocations*.

## 15. Sign-off

_to be appended after commit_
