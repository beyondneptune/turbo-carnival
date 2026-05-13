# Consensus R7 — Statistical Arbitrage & Cointegration
Date: 2026-05-12
Source: Consensus MCP (Semantic Scholar / PubMed / Scopus / ArXiv)
Operator: rafeeqinea@gmail.com

## Run summary
- Queries attempted: 4
- Queries successfully returned: 2 of 4
- Queries blocked by Consensus monthly quota: 2 of 4 (account exhausted its 30-search/month allowance during this run; resets June 1, 2026)
- Papers collected (deduplicated across the two completed queries): 19
- DOI retrievability: Consensus API does not return DOIs directly; where the DOI was inferable from the journal/title it is listed, otherwise the Consensus URL is provided so the operator can resolve the DOI on the paper detail page.

Format mirrors `research/consensus_starter_pack.md` (R6 reference) used by the prior agent.

---

## Query 1: "statistical arbitrage cointegration pairs trading empirical"
Result count returned by Consensus: 20 (top 10 visible to the operator account).

### Paper 1.1 — Statistical Arbitrage with Pairs Trading: Empirical Analysis in Chinese Market
- Authors: Feixue Ouyang et al.
- Year: 2025
- Venue: Advances in Economics, Management and Political Sciences
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/e854e0d7a84951d58256b37bf2d51e95/
- Citations: 0
- Summary: Frames pair spread as Ornstein–Uhlenbeck, combines cointegration test, Hurst exponent, and parameter optimization to pick best pairs on China A-shares. Individual pairs are noisy but a diversified portfolio of multiple pairs materially improves risk-adjusted performance.
- Relevance: 4/5 — clean, recent OU + cointegration + Hurst stack; directly portable to our research process even though the market is non-US.

### Paper 1.2 — Statistical arbitrage pairs trading strategies: Review and outlook
- Authors: C. Krauss
- Year: 2017
- Venue: Journal of Economic Surveys (Wiley-Blackwell)
- DOI: 10.1111/joes.12153 (verify via Consensus URL)
- Consensus URL: https://consensus.app/papers/details/7c3583ff87255f928e2112f890268ca9/
- Citations: 196
- Summary: Canonical survey of >90 pairs-trading papers organized into five buckets: distance, cointegration, time-series (OU and related), stochastic control, and "other". Compares strengths/weaknesses and gives reading roadmap for each approach.
- Relevance: 5/5 — single best literature-survey starting point for this whole topic; required reading.

### Paper 1.3 — An Empirical Assessment of Statistical Arbitrage: A Cointegrated Pairs Trading Approach
- Authors: Dennis Loodh et al.
- Year: 2015
- Venue: Unknown / thesis-level working paper
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/0b2c5bce3b9d51888ac9009fbb88a6a0/
- Citations: 1
- Summary: Backtests a cointegration-based pairs strategy on OMXS30 in a bear regime (2007-06 to 2008-12) and a bull regime (2013-05 to 2014-12). Finds market neutrality and that profitability is materially higher during prolonged turbulence.
- Relevance: 3/5 — small-sample regional study, but useful as a regime-conditional data point.

### Paper 1.4 — Selection of a Portfolio of Pairs Based on Cointegration: A Statistical Arbitrage Strategy
- Authors: J. Caldeira et al.
- Year: 2013
- Venue: Advanced Risk & Portfolio Management Research Paper Series
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/c587ab64ef8f55179e4bdb7598688285/
- Citations: 115
- Summary: Identifies pairs via cointegration, models residuals, and ranks pairs using an in-sample profitability indicator before forming a portfolio. Reports 16.38% annualized excess return, 1.34 Sharpe, low market correlation on Bovespa 2005–2012.
- Relevance: 5/5 — concrete portfolio-construction recipe with respectable Sharpe and an in-sample selection metric that we can replicate.

### Paper 1.5 — Cointegration-Based Pairs Trading Strategy in Commodity Markets
- Authors: Gautham Mohandas
- Year: 2023
- Venue: International Journal of Science and Research (IJSR)
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/8d7371e7ab0e5339a8d9c9c94d7687e9/
- Citations: 2
- Summary: GSCI commodity index members 2007–2018 train, 2019–2022 test. Rolling regression window (1d/30d) addresses look-ahead bias; z-score thresholds ±2 entry / 0 exit / ±4 stop drive trades on three cointegrated pairs.
- Relevance: 3/5 — single-sector study; useful mainly as a methodological template for rolling window cointegration.

### Paper 1.6 — High Frequency and Dynamic Pairs Trading Based on Statistical Arbitrage Using a Two-Stage Correlation and Cointegration Approach
- Authors: George J. Miao
- Year: 2014
- Venue: International Journal of Economics and Finance
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/340deca97703504585639dceaaf67428/
- Citations: 67
- Summary: Two-stage filter: correlation gate followed by cointegration. Out-of-sample 2012–2013 results report 56.58% cumulative pair-portfolio return vs S&P 500, 9.25 annualized Sharpe; outperforms in months when SPX is negative.
- Relevance: 4/5 — high-frequency angle and a clear two-stage selection filter we can adapt.

### Paper 1.7 — A survey of statistical arbitrage pairs trading strategies with non-machine learning methods, 2016–2023
- Authors: Yufei Sun
- Year: 2025
- Venue: Working paper
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/980a121a84a25e3a8fd0a58cca50b33f/
- Citations: 0
- Summary: Updates the Krauss 2017 survey to 2016–2023 across 100+ papers, retaining the same five-category taxonomy. Adds emphasis on which classical methods survive into the post-2016 regime when ML methods are excluded.
- Relevance: 4/5 — modern non-ML follow-up to the Krauss survey; pairs with paper 1.2.

### Paper 1.8 — Statistical arbitrage trading with implementation of machine learning: an empirical analysis of pairs trading on the Norwegian stock market
- Authors: H. Andersen et al.
- Year: 2018
- Venue: Unknown / thesis-level
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/3ef5793ea3445291a6ac515f8b795e93/
- Citations: 1
- Summary: PCA + density-based clustering to extract common risk factors before cointegration pair selection on Oslo Børs. Finds no excess return or favorable Sharpe on this market, but the unsupervised ML filtering step is useful and additive.
- Relevance: 3/5 — negative-result paper; mostly useful for the PCA+clustering preselection methodology.

### Paper 1.9 — Pairs trading strategies in a cointegration framework: back-tested on CFD and optimized by profit factor
- Authors: Zhe Huang et al.
- Year: 2018
- Venue: Applied Economics
- DOI: 10.1080/00036846.2018.1486989 (verify via Consensus URL)
- Consensus URL: https://consensus.app/papers/details/c24dacb28712530c9e1d9ab09012ac72/
- Citations: 12
- Summary: 98 pairs across 152 stocks in 3 currencies via cointegration + ECM-DCC-GARCH on CFDs. Compares percentage / static-std / Bollinger Band strategies with and without GARCH short-term confirmation; 32% of pairs end in loss and 94% of those losses correspond to a cointegration break in the test window.
- Relevance: 4/5 — directly quantifies the cointegration-break failure mode that any live strategy must guard against.

### Paper 1.10 — Pairs trading with partial cointegration
- Authors: Matthew Clegg, C. Krauss
- Year: 2018
- Venue: Quantitative Finance
- DOI: 10.1080/14697688.2017.1370121 (verify via Consensus URL)
- Consensus URL: https://consensus.app/papers/details/b4a88635fe075fedac68645a11a13096/
- Citations: 55
- Summary: Introduces partial cointegration: residual is the sum of a random walk and a mean-reverting AR(1) component. Provides state-space MLE estimator and likelihood-ratio test; reports >12% annualized return after costs on survivor-bias-free S&P 500 1990–2015, outperforming distance and classical cointegration baselines.
- Relevance: 5/5 — the cleanest modern improvement on classical cointegration with a real US-equity backtest and a public methodology.

---

## Query 2: "Engle Granger cointegration mean reversion equity"
Result count returned by Consensus: 19 (top 10 visible).

### Paper 2.1 — Statistical arbitrage on the JSE based on partial co-integration
- Authors: Alwyn J. Hoffman
- Year: 2021
- Venue: Investment Analysts Journal
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/582cd0c42aad516d9bfd6b5791310715/
- Citations: 4
- Summary: Applies Clegg–Krauss partial cointegration with a Kalman filter that tunes the gain on the mean-reverting error component. JSE sector data 1990–2020; outperforms classical cointegration, especially in bear regimes.
- Relevance: 4/5 — practical implementation of Kalman-filtered partial cointegration; complements paper 1.10.

### Paper 2.2 — Pair Trading and VAR Analysis Applied to Energy Stocks
- Authors: Carlos Salas Najera
- Year: 2019
- Venue: Working paper
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/ea5a25cb304b5be0a42ca50230625906/
- Citations: 1
- Summary: VAR + IRF + Granger causality to pre-screen pairs, then Engle–Granger two-step cointegration + ECM. Backtests COP-XOM with OU-modeled spread and compares strategy alpha against Fama-French factors.
- Relevance: 4/5 — full pipeline (VAR screen → Engle-Granger → OU half-life → Fama-French alpha attribution) is a clean template.

### Paper 2.3 — Pairs trading in JSE financial sector
- Authors: Johannes Tshepiso Tsoku et al.
- Year: 2018
- Venue: Journal of Statistics and Management Systems
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/e2e666fa09bc5f529d0ea8625d9ddd2e/
- Citations: 2
- Summary: Correlation pre-filter then Engle–Granger cointegration on 12 JSE financials 2010–2015. Eighteen pairs share a long-run relationship; statistical-arbitrage rules built on those pairs are profitable in-sample.
- Relevance: 3/5 — small market and small sample, but a clean classroom-style example of the Engle-Granger pipeline.

### Paper 2.4 — Assessing Equity Based Mutual Funds and Stock Market Indices in India Using the Engle-Granger Cointegration Technique
- Authors: Neha Gupta et al.
- Year: 2021
- Venue: DLSU Business & Economics Review
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/979dfcfacdae520e99a906606c1d3095/
- Citations: 0
- Summary: Compares Engle-Granger and Johansen tests for long-run equilibrium between Indian equity mutual funds and BSE 200 / Nifty 500. Finds strong long-run cointegration between equity fund NAVs and the underlying indices.
- Relevance: 2/5 — fund-vs-index cointegration is not directly stat-arb, but useful for understanding fund tracking residuals.

### Paper 2.5 — The causal relationships between equity indices on world exchanges
- Authors: A. Kwan et al.
- Year: 1995
- Venue: Applied Economics
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/af72088988e85c53b0c81db3f90284b1/
- Citations: 127
- Summary: Monthly index data 1982–1991 for nine major exchanges. Engle–Granger cointegration and Granger causality results refute the strong form of informational efficiency across world indices.
- Relevance: 3/5 — historical context for cross-market cointegration; pre-dates modern microstructure but heavily cited.

### Paper 2.6 — An LSTM-based optimization algorithm for enhancing quantitative arbitrage trading
- Authors: Guodong Han et al.
- Year: 2024
- Venue: PeerJ Computer Science
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/8ac8e6ef5d905c49bcbf75c93a623d5a/
- Citations: 1
- Summary: Adds an LSTM trend classifier on top of Engle–Granger two-step to suppress trades when the linear combination shows trend rather than mean-reversion. Dynamic boundary update; reports 23% theoretical return on 1-min futures over a 10-day window.
- Relevance: 3/5 — useful ML overlay idea for filtering out non-stationary regimes, but very small backtest window.

### Paper 2.7 — Financial integration of Asian markets: multivariate cointegration, causality and variance decomposition
- Authors: Kalai Lamia
- Year: 2021
- Venue: International Journal of Business and Systems Research
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/ab451adfb01d546cb2b8d02506dcc232/
- Citations: 1
- Summary: Engle–Granger multivariate cointegration + VAR + variance decomposition on East Asian equity indices. Documents long-run equilibrium and causal links among emerging Asian markets.
- Relevance: 2/5 — macro-integration paper, not a stat-arb paper, but useful as background for cross-border cointegration.

### Paper 2.8 — The long-run gains from international equity diversification: Australian evidence from cointegration tests
- Authors: D. Allen et al.
- Year: 1995
- Venue: Applied Financial Economics
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/069f5898608c5447ad98f36287708038/
- Citations: 113
- Summary: 16-country MSCI panel 1970–1992. Compares Engle–Granger two-step OLS with Johansen MLE and shows the two methods can disagree on whether cointegration holds; warns about sensitivity to VAR specification.
- Relevance: 3/5 — methodological caution paper; useful as a reminder that test choice matters.

### Paper 2.9 — Evaluation of dynamic cointegration-based pairs trading strategy in the cryptocurrency market
- Authors: Masood Tadi et al.
- Year: 2021
- Venue: Studies in Economics and Finance
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/008ac79ad9b9561e8701b7730b9dd377/
- Citations: 11
- Summary: Compares Engle–Granger, Kapetanios–Snell–Shin, and Johansen cointegration tests with OU-half-life-calibrated look-back windows on BitMEX. Backtests with realistic best-bid/ask and trade execution; outperforms buy-and-hold with modest drawdowns.
- Relevance: 4/5 — exemplary realistic backtest design with three competing cointegration tests; directly relevant to our backtest standards.

### Paper 2.10 — Persistent and Long-Term Co-Movements between Gender Equality and Global Prices
- Authors: Juan Infante et al.
- Year: 2024
- Venue: Economies
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/4cb4d560e74a56bfa41efdddb0ec2a65/
- Citations: 0
- Summary: Fractional integration / fractional CVAR test for cointegration between Bloomberg Gender Equality Index and MSCI World. Classical Engle–Granger says no cointegration; fractional CVAR says yes — methodological cautionary example.
- Relevance: 2/5 — off-topic for trading but useful as a reminder that fractional methods can rescue cointegration when classical tests reject.

---

## Query 3: "Ornstein Uhlenbeck pairs trading half life"
Status: NOT EXECUTED — Consensus monthly quota exhausted before this query could run.
Quota reset: June 1, 2026.

Operator action: rerun this query (and Query 4) after June 1 reset or upgrade to Consensus Pro (1000 searches/month) and update this file in place. The Query 1 + Query 2 result sets already include OU/half-life material (Papers 1.1, 2.2, 2.9 explicitly use OU-process half-life calibration), so the gap is partially covered.

---

## Query 4: "convergence trade dual listed cross border arbitrage"
Status: NOT EXECUTED — Consensus monthly quota exhausted.
Quota reset: June 1, 2026.

Operator action: rerun after June 1 or upgrade. Cross-listed / dual-listed convergence trades (Royal Dutch / Shell, Unilever, etc.) are a classical Shleifer-Vishny "limits to arbitrage" topic; expect Froot–Dabora 1999, Rosenthal–Young 1990, Gagnon–Karolyi 2010 to appear if/when this query runs.

---

## Top-10 most relevant papers across the two completed queries

Ranked on combined judgment of (a) directly testable in our environment, (b) US-equity or index applicability, (c) methodological rigor, (d) novelty over the baseline cointegration recipe.

1. Krauss (2017), "Statistical arbitrage pairs trading strategies: Review and outlook" — Paper 1.2 — required survey.
2. Clegg & Krauss (2018), "Pairs trading with partial cointegration" — Paper 1.10 — best modern methodological upgrade.
3. Caldeira et al. (2013), "Selection of a Portfolio of Pairs Based on Cointegration" — Paper 1.4 — clean portfolio construction with strong Sharpe.
4. Hoffman (2021), "Statistical arbitrage on the JSE based on partial co-integration" — Paper 2.1 — Kalman-filter implementation of partial cointegration.
5. Tadi et al. (2021), "Evaluation of dynamic cointegration-based pairs trading strategy in the cryptocurrency market" — Paper 2.9 — strong realistic-execution backtest template.
6. Huang et al. (2018), "Pairs trading strategies in a cointegration framework: back-tested on CFD" — Paper 1.9 — explicit quantification of cointegration-break failure mode.
7. Miao (2014), "High Frequency and Dynamic Pairs Trading … Two-Stage Correlation and Cointegration" — Paper 1.6 — two-stage filter design.
8. Salas Najera (2019), "Pair Trading and VAR Analysis Applied to Energy Stocks" — Paper 2.2 — VAR + Engle-Granger + OU + Fama-French stack.
9. Sun (2025), "A survey of statistical arbitrage pairs trading strategies with non-ML methods, 2016–2023" — Paper 1.7 — modern non-ML companion to Krauss 2017.
10. Ouyang et al. (2025), "Statistical Arbitrage with Pairs Trading: Empirical Analysis in Chinese Market" — Paper 1.1 — OU + cointegration + Hurst.

---

## Consolidated DOI / URL list (for download queue)

DOIs that are likely correct based on title/journal/year but not directly returned by Consensus (verify each via the Consensus URL before submitting to a DOI resolver):
- 10.1111/joes.12153 — Krauss 2017 (Paper 1.2)
- 10.1080/14697688.2017.1370121 — Clegg & Krauss 2018 (Paper 1.10)
- 10.1080/00036846.2018.1486989 — Huang et al. 2018 (Paper 1.9)

DOI not retrievable from Consensus result — operator must resolve via the Consensus URL on the paper detail page:
- Paper 1.1: https://consensus.app/papers/details/e854e0d7a84951d58256b37bf2d51e95/
- Paper 1.3: https://consensus.app/papers/details/0b2c5bce3b9d51888ac9009fbb88a6a0/
- Paper 1.4: https://consensus.app/papers/details/c587ab64ef8f55179e4bdb7598688285/
- Paper 1.5: https://consensus.app/papers/details/8d7371e7ab0e5339a8d9c9c94d7687e9/
- Paper 1.6: https://consensus.app/papers/details/340deca97703504585639dceaaf67428/
- Paper 1.7: https://consensus.app/papers/details/980a121a84a25e3a8fd0a58cca50b33f/
- Paper 1.8: https://consensus.app/papers/details/3ef5793ea3445291a6ac515f8b795e93/
- Paper 2.1: https://consensus.app/papers/details/582cd0c42aad516d9bfd6b5791310715/
- Paper 2.2: https://consensus.app/papers/details/ea5a25cb304b5be0a42ca50230625906/
- Paper 2.3: https://consensus.app/papers/details/e2e666fa09bc5f529d0ea8625d9ddd2e/
- Paper 2.4: https://consensus.app/papers/details/979dfcfacdae520e99a906606c1d3095/
- Paper 2.5: https://consensus.app/papers/details/af72088988e85c53b0c81db3f90284b1/
- Paper 2.6: https://consensus.app/papers/details/8ac8e6ef5d905c49bcbf75c93a623d5a/
- Paper 2.7: https://consensus.app/papers/details/ab451adfb01d546cb2b8d02506dcc232/
- Paper 2.8: https://consensus.app/papers/details/069f5898608c5447ad98f36287708038/
- Paper 2.9: https://consensus.app/papers/details/008ac79ad9b9561e8701b7730b9dd377/
- Paper 2.10: https://consensus.app/papers/details/4cb4d560e74a56bfa41efdddb0ec2a65/

---

## Coverage notes

- Statistical-arbitrage core (Query 1): excellent coverage. Krauss 2017 survey + Clegg–Krauss 2018 partial cointegration + Caldeira 2013 portfolio recipe form a complete reading stack.
- Engle–Granger / mean-reversion (Query 2): good coverage. Strong methodological variety (Engle-Granger, Johansen, Kapetanios-Snell-Shin, fractional CVAR, partial cointegration via Kalman) but US-equity-specific empirical work is thin in this sample — most papers are JSE, BSE/Nifty, BitMEX, OMXS30, Bovespa. Caldeira 2013 (Bovespa) and Clegg–Krauss 2018 (S&P 500) are the closest empirical hits.
- OU half-life (Query 3): NOT RUN due to quota. Papers 1.1, 2.2, and 2.9 partially cover OU half-life calibration; rerun query after June 1 for completeness.
- Cross-border convergence trades (Query 4): NOT RUN due to quota. Operator should expect Froot–Dabora, Rosenthal–Young, and Gagnon–Karolyi to surface when the query is rerun.

Reused R6 (consensus_starter_pack.md) findings that are directly load-bearing for stat-arb: Hasbrouck 2003 (price-discovery decomposition methodology), Gârleanu & Pedersen 2013 (dynamic trading with predictable mean-reverting signal + costs — the math of converting a cointegration signal into a target position), Frazzini-Israel-Moskowitz 2018 (institutional cost reality check for medium-frequency stat-arb), Skachkov 2013 (medium-frequency stat-arb on S&P 500). Cross-reference R6 sections 1, 4, and 5 when planning the actual strategy.
