# Quant Lab — 7 advanced strategies tested 2019-2026

Honest backtest results on the 7 strategies requested. Each runs walk-forward
on the longest data window we have. All scripts are at
`experiments/quant_lab/0[1-7]_*.py` with parquet/json output in `runs/`.

## Headline results

| # | Strategy | Period | Total return | Annualized | Sharpe | Max DD | Verdict |
|---|---|---|---:|---:|---:|---:|---|
| 1 | Cross-sectional mean reversion | 7 yrs (10 mega-cap) | **−87.0%** | −24.3% | −0.57 | −92.6% | ❌ broken (momentum dominates on this universe) |
| 2 | HMM regime detection (2-state) | 7 yrs (SPY+VIX) | **+53.9%** | +7.0% | 0.70 | −22.6% | ⚠ modest, but loses to SPY in absolute return |
| 3 | Kalman filter pairs trading | 7 yrs (3 pairs) | −53% to +3% | −10% to +0.5% | 0.14-0.22 | −80% (AMD/NVDA) | ❌ cointegration broke in AI boom |
| 4 | PCA factor-neutral L/S | 7 yrs (25 stocks) | **−14.6%** | −2.2% | −0.01 | −54.9% | ❌ no idiosyncratic-mean-reversion edge |
| 5 | VPIN order-flow toxicity | 21 days minute (7 syms) | — | — | r ≈ −0.004 | — | ❌ no signal (tick-rule too noisy at 1-min granularity) |
| 6 | Earnings IV crush (short ATM straddle) | 2 yrs (232 events) | **−2.9%** | — | −0.07 | — | ⚠ 64% win rate, but avg loss 2× avg win — classic short-vol asymmetry |
| 7 | Dispersion vol arbitrage | 2 yrs | — | — | — | — | ⏭ data-limited (no SPY ATM in our chain cache) |

## What this tells us

**6 of 7 classical quant strategies UNDERPERFORM or are infeasible** on our
available data. This is consistent with the academic finding that **most
documented strategies have decayed since their publication** (cross-sectional
reversion, pairs trading especially in tech, dispersion arb — all crowded).

The one strategy that worked reasonably well — **HMM regime detection** —
beat the market on risk-adjusted basis (lower drawdown) but missed the bulk
of the bull market.

## In comparison to the SVXY regime rotation strategy

Already in production-ready state (committed `6ebeb7a`):

| Strategy | Annualized | Max DD | Sharpe |
|---|---:|---:|---:|
| SVXY regime rotation | **+17.1%** | **−32.8%** | 0.66 |
| Best of the 7 (HMM) | +7.0% | −22.6% | 0.70 |
| 5 others | negative or no-signal | — | — |

SVXY remains the cleanest tested strategy in the lab. It's allocated to
**Hermes** in the 6-account plan.

## Caveats — what would change these results

- **Universe**: mega-cap tech (top 10) is uniquely momentum-driven. CSMR, PCA,
  pairs trading all work better on equal-weight S&P 500 or small caps. Re-run
  on Russell 2000 would likely produce different numbers.
- **Parameters**: each strategy has parameters we fixed at common defaults
  (lookback=5, hold=5, etc.). Grid-search optimization would change rankings
  but introduce overfit risk.
- **Transaction costs**: all backtests assume zero commission + slippage on
  daily-frequency trades. Real $1k account adds 0.1-0.5% drag per round trip.
- **Walk-forward integrity**: every strategy uses T-1 signal → T position,
  no lookahead. CSMR and PCA results in particular reflect honest live-trading
  expectations.

## Recommendation

For the 6-account portfolio, the **deployable winner remains SVXY rotation
on Hermes**. None of the 7 tested strategies would beat it on $1k-scale
deployment.

For Atlas, the proposed cointegrated-pairs strategy needs:
1. Better pair selection (sector ETFs > single-name mega-caps)
2. Larger universe to find genuinely stable cointegration

For TERMINATOR ($1k), composite C3 gate + Half-Kelly remains the best
research-validated configuration.

## Files

```
experiments/quant_lab/
├── 01_cross_sectional_mean_reversion.py + runs/01_csmr_*.{parquet,json}
├── 02_hmm_regime_detection.py        + runs/02_hmm_*.{parquet,json}
├── 03_kalman_pairs_trading.py        + runs/03_kalman_pairs_summary.json
├── 04_pca_factor_neutral.py          + runs/04_pca_factor_neutral_*.{parquet,json}
├── 05_vpin_order_flow_toxicity.py    + runs/05_vpin_summary.json
├── 06_earnings_iv_crush.py           + runs/06_earnings_iv_crush_*.{parquet,json}
└── 07_dispersion_volatility_arb.py   (incomplete — data-limited)
```
