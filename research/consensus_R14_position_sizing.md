# Consensus R14 — Position Sizing, Kelly Criterion, Drawdown Management
Date: 2026-05-12
Source: Consensus MCP (Semantic Scholar / PubMed / Scopus / ArXiv)

## Run Status

| # | Query | Status | Papers retrieved |
|---|-------|--------|------------------|
| 1 | "Kelly criterion fractional position sizing empirical" | OK | 10 |
| 2 | "drawdown control trading strategy leverage" | NOT RUN — Consensus monthly quota exhausted | 0 |
| 3 | "Sharpe ratio risk parity portfolio construction" | NOT RUN — Consensus monthly quota exhausted | 0 |
| 4 | "leverage optimization compound returns" | NOT RUN — Consensus monthly quota exhausted | 0 |

The Consensus free tier reported "1 search left this month" after Query 1 and then "You've used all 30 searches this month; limits reset on June 1st" when Query 2 was attempted. Queries 3 and 4 could not be executed. To complete this research item, either (a) re-run on/after 2026-06-01 when quotas reset, or (b) upgrade to Consensus Pro (1,000 searches/mo, 20 results per search).

Notes:
- DOIs are not directly returned by Consensus. Where I could not retrieve a DOI from the result payload, the Consensus URL is provided instead — operator can use it to land on the paper's detail page and grab the DOI for PDF retrieval. Authors / years / journal / citations are exactly as returned.

---

## Topic 1: Kelly criterion, fractional position sizing (empirical) — 10 papers

### Paper 1.1 — Size matters! How position sizing determines risk and return of technical timing strategies [1]
- Authors: Peter Scholz
- Year: 2012
- Venue: Unknown Journal (SSRN working paper)
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/ddca20e8db5a564caf5b1c01947c451a/
- Citations: 9
- Summary: Systematically studies how relative position sizing (a la Kelly) versus erratic/fixed-dollar sizing changes the risk-return profile of moving-average trading rules under simulated drift/volatility/autocorrelation regimes. Finds that no single optimal Kelly fraction exists across regimes, and that smaller-than-Kelly fractions deliver the highest risk-adjusted returns in most simulated scenarios.
- Relevance: 5/5 — directly supports a fractional-Kelly (sub-Kelly) sizing rule for a strategy like paper_v1 where edge estimates are uncertain.

### Paper 1.2 — A Novel Approach of Option Portfolio Construction Using the Kelly Criterion [2]
- Authors: Mu-En Wu et al.
- Year: 2018
- Venue: IEEE Access
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/c2d085171f765d6c8418b1cf74905114/
- Citations: 23
- Summary: Applies Kelly to multi-leg option portfolios where long/short strikes pre-lock the P&L distribution, sidestepping the unbounded-loss problem that breaks Kelly for naked options. Demonstrates feasibility on real options data and yields a practical option-portfolio sizing framework using the Kelly fraction.
- Relevance: 5/5 — direct precedent for applying Kelly to an options strategy with bounded payoff legs; closest match to paper_v1's hybrid options posture.

### Paper 1.3 — Managing Position Size Depending on Asset Price Characteristics [3]
- Authors: Peter Scholz
- Year: 2014
- Venue: Unknown Journal
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/33effb7a6d215d6f8367a235bfbcdffc/
- Citations: 0
- Summary: Companion / extension to Paper 1.1; reiterates that introducing relative position sizing has a major impact on trading results and that, contrary to canonical Kelly, no globally optimal position size exists across asset price regimes. Re-emphasizes regime-dependent fractional sizing.
- Relevance: 3/5 — adds confirmation but is largely overlapping with 1.1; low citation count.

### Paper 1.4 — Confidence Intervals for the Kelly Criterion [4]
- Authors: Euan Sinclair
- Year: 2014
- Venue: Mutual Funds (SSRN working paper)
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/3a1da47d5b5f5cab8305f77ac2b307cb/
- Citations: 0
- Summary: Derives the variance of the estimated Kelly fraction when the optimal fraction must be inferred from empirical data rather than known analytically. Provides a closed-form way to put confidence intervals around the Kelly estimate so that sizing can be cut to a sub-Kelly fraction proportional to estimation uncertainty.
- Relevance: 5/5 — directly applicable: paper_v1's edge estimates have meaningful sample error, and this gives a principled basis for choosing a sub-Kelly fraction.

### Paper 1.5 — Convert index trading to option strategies via LSTM architecture [5]
- Authors: J. Wu et al.
- Year: 2020
- Venue: Neural Computing and Applications
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/334bed0022fb50d6a3ed460c1e0eb11f/
- Citations: 10
- Summary: Converts a futures trading strategy into an options strategy specifically so that Kelly sizing becomes tractable (bounded loss + estimable win-rate via LSTM-predicted profit probability). Shows that the resulting options trades end up closer to the Kelly-optimal fraction than the original futures trades and improves overall P&L when the underlying strategy is profitable.
- Relevance: 4/5 — supports converting an underlying signal into bounded-payoff option structures specifically to make Kelly-sizing valid; methodologically analogous to paper_v1's options overlay.

### Paper 1.6 — An outperforming investment strategy under fractional Brownian motion [6]
- Authors: Qiang Liu et al.
- Year: 2018
- Venue: The North American Journal of Economics and Finance
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/71bd37662ab15015a584bb3c74dfdd5e/
- Citations: 8
- Summary: Derives a horizon-dependent Kelly fraction under fractional Brownian motion and proposes a regime-switching rule: invest at Kelly when fBM increments are positively autocorrelated, hold risk-free otherwise. Empirically beats classical Kelly under standard Brownian motion across US/HK/CN indices and selected single stocks.
- Relevance: 4/5 — motivates regime-aware (vol/autocorrelation-conditional) Kelly fractions rather than a static fraction.

### Paper 1.7 — The Skewness and Kurtosis of European Options and the Implications for Trade Sizing [7]
- Authors: Euan Sinclair et al.
- Year: 2017
- Venue: Capital Markets: Market Microstructure eJournal (SSRN)
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/33d80ac6508e573481adc273652df2f0/
- Citations: 0
- Summary: Computes analytical skewness and kurtosis for European calls and puts and uses a power-series expansion to derive a non-normal correction to the Kelly fraction. Shows that ignoring the higher moments of option returns leads to economically significant mis-sizing under standard Kelly.
- Relevance: 5/5 — paper_v1 trades options, so the higher-moment correction to Kelly is the appropriate sizing formula rather than the textbook variance-only Kelly.

### Paper 1.8 — Sizing the Risk: Kelly, VIX, and Hybrid Approaches in Put-Writing on Index Options [8]
- Authors: Maciej Wysocki
- Year: 2025
- Venue: Unknown Journal (working paper)
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/a0f604a163f95a949ae5aef6e6eaff16/
- Citations: 0
- Summary: Backtests three sizing rules — pure Kelly, VIX-regime scaling, and a Kelly+VIX hybrid — on systematic short-dated (0–5 DTE) SPX put-writing strategies. The hybrid sizing approach delivers the most consistent balance of return and drawdown control, particularly in low-vol regimes; ultra-short-dated far-OTM puts produce the best risk-adjusted returns.
- Relevance: 5/5 — most directly comparable to paper_v1's intended SPX put-writing / short-dated options posture; specifically addresses drawdown control via sizing.

### Paper 1.9 — Kelly criterion and fractional Kelly strategy for non-mutually exclusive bets [9]
- Authors: Benjamin P. Jacot et al.
- Year: 2023
- Venue: Journal of Quantitative Analysis in Sports
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/cfbd02a6da5a584582aca17b5462c76b/
- Citations: 2
- Summary: Generalises Kelly and fractional-Kelly sizing to bets that are not mutually exclusive (multiple overlapping wagers on a single outcome event). Provides a new fractional-Kelly formulation for that setting.
- Relevance: 3/5 — domain is horse racing but the mathematical generalisation applies to overlapping option trades on the same underlying / event.

### Paper 1.10 — The Kelly criterion: a closer look at how estimation errors affect portfolio performance [10]
- Authors: Adrian Halhjem Sælen
- Year: 2012
- Venue: Unknown Journal (master's thesis)
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/3ca8e43fc56050bb90b4761193667a72/
- Citations: 2
- Summary: Shows that as estimation error in the inputs (mean, variance) grows, full-Kelly sizing not only raises portfolio volatility but also reduces realised wealth growth — i.e. overbetting is doubly punished. Concludes that risky assets with poorly-estimated moments should be sized with a fractional Kelly or excluded altogether.
- Relevance: 5/5 — the canonical estimation-error argument for sub-Kelly in any system (like paper_v1) with limited training data.

---

## Topic 2: Drawdown control / trading strategy leverage
NOT RUN — Consensus monthly quota exhausted before this query could be executed. Re-run after 2026-06-01.

## Topic 3: Sharpe ratio / risk parity / portfolio construction
NOT RUN — Consensus monthly quota exhausted before this query could be executed. Re-run after 2026-06-01.

## Topic 4: Leverage optimization and compound returns
NOT RUN — Consensus monthly quota exhausted before this query could be executed. Re-run after 2026-06-01.

---

## Cited papers

[1] [Size matters! How position sizing determines risk and return of technical timing strategies](https://consensus.app/papers/details/ddca20e8db5a564caf5b1c01947c451a/?utm_source=claude_desktop) (Peter Scholz, 2012, Unknown Journal, 9 citations)
[2] [A Novel Approach of Option Portfolio Construction Using the Kelly Criterion](https://consensus.app/papers/details/c2d085171f765d6c8418b1cf74905114/?utm_source=claude_desktop) (Mu-En Wu et al., 2018, IEEE Access, 23 citations)
[3] [Managing Position Size Depending on Asset Price Characteristics](https://consensus.app/papers/details/33effb7a6d215d6f8367a235bfbcdffc/?utm_source=claude_desktop) (Peter Scholz, 2014, Unknown Journal, 0 citations)
[4] [Confidence Intervals for the Kelly Criterion](https://consensus.app/papers/details/3a1da47d5b5f5cab8305f77ac2b307cb/?utm_source=claude_desktop) (Euan Sinclair, 2014, Mutual Funds, 0 citations)
[5] [Convert index trading to option strategies via LSTM architecture](https://consensus.app/papers/details/334bed0022fb50d6a3ed460c1e0eb11f/?utm_source=claude_desktop) (J. Wu et al., 2020, Neural Computing and Applications, 10 citations)
[6] [An outperforming investment strategy under fractional Brownian motion](https://consensus.app/papers/details/71bd37662ab15015a584bb3c74dfdd5e/?utm_source=claude_desktop) (Qiang Liu et al., 2018, The North American Journal of Economics and Finance, 8 citations)
[7] [The Skewness and Kurtosis of European Options and the Implications for Trade Sizing](https://consensus.app/papers/details/33d80ac6508e573481adc273652df2f0/?utm_source=claude_desktop) (Euan Sinclair et al., 2017, Capital Markets: Market Microstructure eJournal, 0 citations)
[8] [Sizing the Risk: Kelly, VIX, and Hybrid Approaches in Put-Writing on Index Options](https://consensus.app/papers/details/a0f604a163f95a949ae5aef6e6eaff16/?utm_source=claude_desktop) (Maciej Wysocki, 2025, Unknown Journal, 0 citations)
[9] [Kelly criterion and fractional Kelly strategy for non-mutually exclusive bets](https://consensus.app/papers/details/cfbd02a6da5a584582aca17b5462c76b/?utm_source=claude_desktop) (Benjamin P. Jacot et al., 2023, Journal of Quantitative Analysis in Sports, 2 citations)
[10] [The Kelly criterion: a closer look at how estimation errors affect portfolio performance](https://consensus.app/papers/details/3ca8e43fc56050bb90b4761193667a72/?utm_source=claude_desktop) (Adrian Halhjem Sælen, 2012, Unknown Journal, 2 citations)

---

1 search left this month; resets to 30 on June 1st. Upgrade to Consensus Pro to get 1,000 searches per month, return 20 results per search instead of 10, and include more data like study design and key takeaways for every result.: https://consensus.app/pricing/?utm_source=claude_desktop
