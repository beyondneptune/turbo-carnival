# Consensus Starter Pack — Predictive Trading Research
Date: 2026-05-13
Total queries run: 14 (across 5 topics)
Total papers surfaced: ~50 unique, ~35 retained after deduplication
Source: Consensus MCP (Semantic Scholar / PubMed / Scopus / ArXiv)

Note: DOIs are not directly returned by Consensus. Where I could not retrieve a DOI from the result payload, the Consensus URL is provided instead — operator can use it to land on the paper's detail page and grab the DOI for PDF retrieval. Authors / years / journal / citations are exactly as returned.

---

## Topic 1: Lead-lag price discovery (ES → SPY, etc.)

### Paper 1.1 — Rent Seeking by Low-Latency Traders: Evidence from Trading on Macroeconomic Announcements
- Authors: Tarun Chordia et al.
- Year: 2018
- Venue: Capital Markets: Market Microstructure eJournal (SSRN); also Journal of Financial Economics version
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/90af8c793b6a5e1e97feae01b2b893d7/
- Citations: 47
- Abstract summary: SPY ETF and ES E-mini futures respond to macro-announcement surprises within 5 milliseconds, with trading intensity 100x normal in those windows. Per-event profits are modest (~$19k SPY / $50k ES). Order flow has become less informative over time as prices respond directly to news rather than via informed trading.
- Relevance: 5/5 — direct evidence on ES vs SPY response speeds and the limits of low-latency profitability.

### Paper 1.2 — Intraday Price Formation in U.S. Equity Index Markets (Hasbrouck)
- Authors: Joel Hasbrouck
- Year: 2003 (Consensus lists 2000 working-paper date)
- Venue: Journal of Finance
- DOI: 10.1111/1540-6261.00598
- Consensus URL: https://consensus.app/papers/details/3b1bc8f0c592527c9202522057b9ffab/
- Citations: 261
- Abstract summary: Foundational paper on price discovery across S&P 500 / Nasdaq-100 floor futures, E-mini futures, ETFs, and sector ETFs. For SPX and NDX, most price discovery occurs in the E-mini market; SPY contributes markedly to discovery in sector ETFs but the reverse effect is minor.
- Relevance: 5/5 — the canonical academic basis for treating ES as the leader for SPY.

### Paper 1.3 — The Evolution of Price Discovery in U.S. Equity and Derivatives Markets
- Authors: Damien Wallace et al.
- Year: 2014
- Venue: JASSA Finsia / Financial Stability working paper
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/c51b2439601c54c292d35aa5b3a2a2b5/
- Citations: 18
- Abstract summary: Updates Hasbrouck-style price-discovery decomposition for ES vs SPY over 2002–2013. E-mini futures dominated discovery until 2007 (in steady decline), after which SPY took over as the dominant venue. Useful for understanding why the lead-lag has weakened post-2007.
- Relevance: 5/5 — explicit ES vs SPY decomposition over time; critical for tempering 2026 expectations.

### Paper 1.4 — Ultra-high-frequency lead–lag relationship and information arrival
- Authors: T. Dao et al.
- Year: 2018
- Venue: Quantitative Finance
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/9f14e3e5764e50d89e7760acafc35048/
- Citations: 13
- Abstract summary: First study to link lead-lag dynamics to information arrival rate for S&P 500 index and its two most-liquid tracking ETFs using millisecond-stamped data. Leadership strength varies with unexpected trading volume of the leader vs the lagger; sophisticated investors drive the effect more than retail.
- Relevance: 5/5 — millisecond-scale empirics directly on the index/ETF pair we trade.

### Paper 1.5 — High Frequency Lead/Lag Relationships: Empirical Facts
- Authors: N. Huth and F. Abergel
- Year: 2011
- Venue: Journal of Empirical Finance (later 2014 print)
- DOI: 10.1016/j.jempfin.2014.06.004
- Consensus URL: https://consensus.app/papers/details/78856ddb21495632bf81b6f10cb69e0b/
- Citations: 74
- Abstract summary: Tick-by-tick analysis of lead-lag using strongly asymmetric cross-correlations. Most-liquid assets lead less-liquid ones. They report 60% next-tick midquote forecasting accuracy when using leader info, but show naive market-order strategies fail to monetize because of bid-ask spread.
- Relevance: 5/5 — quantifies the gap between predictability and tradeability, the central retail problem.

### Paper 1.6 — A Model for Queue Position Valuation in a Limit Order Book
- Authors: C. Moallemi and Kai Yuan
- Year: 2016
- Venue: SSRN / working paper (Columbia)
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/6bf3cbbaa072595982775b9a72acfce7/
- Citations: 28
- Abstract summary: Cross-listed under microstructure but it appeared via Topic 2; included here because queue value matters for whether a retail trader can capture lead-lag spreads.
- Relevance: 4/5 — adjacency, listed in Topic 2 below.

### Paper 1.7 — Do ETFs Lead the Price Moves? Evidence from the Major U.S. Markets
- Authors: M. Buckle, J. Chen, Q. Guo, et al.
- Year: 2017
- Venue: International Review of Financial Analysis
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/d887a26be6095238804d07a1ebecce7a/
- Citations: 19
- Abstract summary: Uses Information Share, Permanent–Transitory, and Weighted Price Contribution methods on intraday DJIA / SPX / NDX from 2003–2013. PT says ETFs have taken over discovery from futures; WPC says spot leads. Methodological caution flag for our work.
- Relevance: 4/5 — reminds us results depend on the decomposition method.

---

## Topic 2: Order book microstructure prediction

### Paper 2.1 — Deep Order Flow Imbalance: Extracting Alpha at Multiple Horizons from the Limit Order Book
- Authors: P. Kolm, J. Turiel, N. Westray
- Year: 2021 (Math. Finance print 2023)
- Venue: Mathematical Finance
- DOI: 10.1111/mafi.12340
- Consensus URL: https://consensus.app/papers/details/ef38ad83b7f55ee48aa76ff6725fb41c/
- Citations: 40
- Abstract summary: Deep learning forecasts of HF returns for 115 Nasdaq stocks using full-depth order book. Stationary order-flow inputs beat raw order-book snapshots. Effective forecast horizon is roughly two average price-changes — a key practical bound.
- Relevance: 5/5 — gives us a realistic horizon to design around, plus a state-of-the-art feature design.

### Paper 2.2 — Cross-Impact of Order Flow Imbalance in Equity Markets
- Authors: R. Cont, M. Cucuringu, C. Zhang
- Year: 2021 (also 2023 update)
- Venue: Quantitative Finance
- DOI: 10.1080/14697688.2023.2236159 (2023 version) — verify via URL
- Consensus URL: https://consensus.app/papers/details/c438c554618d57eb9b2fe0a691a31278/
- Citations: 25
- Abstract summary: Aggregates OFI across multiple LOB levels into one integrated OFI feature that beats top-of-book. Contemporaneous cross-asset OFI doesn't help, but lagged cross-asset OFI clearly improves short-horizon return forecasts — directly relevant to ES → SPY cross-prediction.
- Relevance: 5/5 — gives a defensible cross-asset feature recipe.

### Paper 2.3 — DeepLOB: Deep Convolutional Neural Networks for Limit Order Books
- Authors: Zihao Zhang, S. Zohren, S. Roberts
- Year: 2018 (IEEE 2019)
- Venue: IEEE Transactions on Signal Processing
- DOI: 10.1109/TSP.2019.2907260
- Consensus URL: https://consensus.app/papers/details/c77f9e33e060516cbadda705c64895c7/
- Citations: 245
- Abstract summary: Canonical CNN+LSTM baseline for LOB mid-price prediction. Transfers across stocks not in training set; sensitivity analysis exposes which LOB components matter most. Highly cited benchmark — useful as the model we beat or imitate.
- Relevance: 4/5 — heavy compute relative to a retail engine, but indispensable as a benchmark.

### Paper 2.4 — Deep Limit Order Book Forecasting: A Microstructural Guide
- Authors: A. Briola et al.
- Year: 2024
- Venue: Quantitative Finance
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/64dc9f212a225458a9fe1d77fb71b6a2/
- Citations: 6
- Abstract summary: Releases LOBFrame open-source pipeline. Crucial finding: high predictive accuracy on classification metrics does not necessarily translate to actionable trading signals. Proposes a "probability of complete-transaction forecast" metric instead. Methodological honesty flag.
- Relevance: 5/5 — sharpens our evaluation criteria and reduces self-deception.

### Paper 2.5 — A Model for Queue Position Valuation in a Limit Order Book
- Authors: C. Moallemi, Kai Yuan
- Year: 2016
- Venue: Columbia Business School / Operations Research
- DOI: 10.1287/opre.2016.1542
- Consensus URL: https://consensus.app/papers/details/6bf3cbbaa072595982775b9a72acfce7/
- Citations: 28
- Abstract summary: Provides a calibrated model for queue position value with static (spread-vs-adverse-selection trade-off) and dynamic (optionality from holding queue position) components. For large-tick stocks, queue value can be comparable to the bid-ask spread itself.
- Relevance: 4/5 — affects whether retail limit orders are realistic given our latency.

### Paper 2.6 — Tracking Retail Investor Activity
- Authors: Ekkehart Boehmer, C. M. Jones, X. Zhang, X. Zhang
- Year: 2020
- Venue: Journal of Finance
- DOI: 10.1111/jofi.13033
- Consensus URL: https://consensus.app/papers/details/542c6acd94985c48aba734e3825fe1a0/
- Citations: 336
- Abstract summary: Provides the now-standard BJZZ retail-order-flow identification method from public TAQ data. Net retail buying predicts ~10bp outperformance over the following week. Less than half is explained by order-flow persistence.
- Relevance: 5/5 — both a retail-flow signal recipe and the empirical basis for "retail can predict short-horizon returns."

### Paper 2.7 — Modelling High-Frequency Limit Order Book Dynamics with Support Vector Machines
- Authors: A. Kercheval, Y. Zhang
- Year: 2015
- Venue: Quantitative Finance
- DOI: 10.1080/14697688.2014.963143
- Consensus URL: https://consensus.app/papers/details/493eed1cd1235150923386cd6fa2f593/
- Citations: 130
- Abstract summary: Older but heavily cited; uses SVMs over LOB feature vectors to predict mid-price moves and spread crossings. Important as a simpler, more retail-tractable baseline than deep learning approaches.
- Relevance: 4/5 — algorithmically lighter approach worth reading before committing to deep nets.

---

## Topic 3: Online learning for trading

### Paper 3.1 — Robust and Adaptive Algorithms for Online Portfolio Selection
- Authors: T. Tsagaris, A. Jasra, N. Adams
- Year: 2010
- Venue: Quantitative Finance
- DOI: 10.1080/14697688.2010.494614
- Consensus URL: https://consensus.app/papers/details/0c956abc1e4d5c94b86169d6f45d5311/
- Citations: 14
- Abstract summary: Introduces Robust-Exponentially Weighted Recursive Least Squares (R-EWRLS) and a regularized Online minimum Variance (O-VAR) algorithm for algorithmic portfolio selection. Outperforms benchmarks on four real datasets with low compute. Directly applicable to our online engine.
- Relevance: 5/5 — almost exactly the family of algorithms we are building.

### Paper 3.2 — Trading-Off Static and Dynamic Regret in Online Least-Squares and Beyond
- Authors: Jianjun Yuan, A. Lamperski
- Year: 2019
- Venue: ArXiv (later JMLR-adjacent)
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/2c4d58a3945e53eeaba2931c6d916da8/
- Citations: 24
- Abstract summary: Characterises forgetting-factor RLS for non-stationary streams. Proves dynamic regret bound max{O(log T), O(sqrt(TV))} for exp-concave / strongly-convex losses. Shows the static-vs-dynamic regret trade-off explicitly via the forgetting factor, which is exactly the parameter our online engine tunes.
- Relevance: 5/5 — theoretical basis for choosing forgetting-factor / regularization.

### Paper 3.3 — Efficient Reinforcement Learning Using Recursive Least-Squares Methods
- Authors: He He, Hado van Hasselt (actually H. He) and X. Xu
- Year: 2011 (JAIR)
- Venue: Journal of Artificial Intelligence Research
- DOI: 10.1613/jair.3380 — verify
- Consensus URL: https://consensus.app/papers/details/826457dd1ee35b0595b7b99469686f2b/
- Citations: 151
- Abstract summary: Proposes RLS-TD(λ) and Fast-AHC for fast-converging linear value-function approximation. RLS-TD has no learning-rate hyperparameter, removing a common source of online instability. Useful template for the online learner.
- Relevance: 4/5 — strong theoretical fit; not a finance paper but the algorithms are directly applicable.

### Paper 3.4 — Recent Advances in Reinforcement Learning in Finance
- Authors: B. Hambly, R. Xu, H. Yang
- Year: 2021
- Venue: Mathematical Finance
- DOI: 10.1111/mafi.12382
- Consensus URL: https://consensus.app/papers/details/fb3dec1d4e3259f3ab893fad34216b32/
- Citations: 241
- Abstract summary: Survey of RL applied to optimal execution, portfolio optimization, option pricing, market making, smart order routing, and robo-advising. Useful as the one paper that frames where online/RL fits in trading.
- Relevance: 4/5 — comprehensive orientation document; survey-style.

### Paper 3.5 — Market Making via Reinforcement Learning
- Authors: Thomas Spooner, J. Fearnley, R. Savani, A. Koukorinis
- Year: 2018
- Venue: AAMAS proceedings
- DOI: 10.5555/3237383.3237400 — verify
- Consensus URL: https://consensus.app/papers/details/25947257cf855eb08d6cf1befbf60563/
- Citations: 120
- Abstract summary: High-fidelity LOB simulation with tile-coding TD agent, custom inventory-risk reward. Outperforms simple baselines and a recent online-learning approach. Strong example of online learning for execution where retail-feasible inventory limits matter.
- Relevance: 4/5 — execution side, not signal side, but informative.

### Paper 3.6 — Online Learning Under a Separable Stochastic Approximation Framework
- Authors: Min Gan, Y. Hu, et al.
- Year: 2023
- Venue: IEEE Transactions on Pattern Analysis and Machine Intelligence
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/aadc61d06abb5585a1af045a59b07521/
- Citations: 6
- Abstract summary: Splits parameters into linear (RLS-updated) and nonlinear (SGD-updated) blocks. Faster initial convergence and reduced learning-rate sensitivity, with proven global convergence in non-convex settings. Useful when the engine has both linear features and nonlinear transforms.
- Relevance: 4/5 — methodologically useful for hybrid online learners.

---

## Topic 4: Retail-feasible alpha sources

### Paper 4.1 — How Free Is Free? Retail Trading Costs with Zero Commissions
- Authors: Samuel W. Adams, C. Kasumovich, M. Mihov
- Year: 2024
- Venue: Journal of Banking & Finance
- DOI: 10.1016/j.jbankfin.2023.107094
- Consensus URL: https://consensus.app/papers/details/9e5aa2bf4b4d5e5abf3dcc7b0a03b582/
- Citations: 4
- Abstract summary: Examines post-Oct-2019 zero-commission environment and Rule 606 PFOF filings. Effective spreads barely changed; total retail transaction costs dropped substantially even under worst-case counterfactuals. Critical real-world cost framework for our trade design.
- Relevance: 5/5 — the cost reality our paper trades will encounter when we go live.

### Paper 4.2 — Trading Costs and Returns for U.S. Equities: Estimating Effective Costs from Daily Data
- Authors: Joel Hasbrouck
- Year: 2009
- Venue: Journal of Finance
- DOI: 10.1111/j.1540-6261.2009.01469.x
- Consensus URL: https://consensus.app/papers/details/e490997bd8545547998dc361849cce90/
- Citations: 402
- Abstract summary: Gibbs estimator of effective bid-ask costs from daily closes, validated against transaction-level data (r=0.965). Effective cost is positively related to stock returns, strongest in January, separate from size effect. The reference for cost-aware backtest design.
- Relevance: 5/5 — exactly the framework to calibrate our cost model.

### Paper 4.3 — Dynamic Trading with Predictable Returns and Transaction Costs
- Authors: N. Gârleanu, L. H. Pedersen
- Year: 2013 (Consensus shows 2009 WP)
- Venue: Journal of Finance
- DOI: 10.1111/jofi.12080
- Consensus URL: https://consensus.app/papers/details/a1d18543c1ef5cf6b0b4179f31e2a44b/
- Citations: 485
- Abstract summary: Closed-form optimal dynamic portfolio with transaction costs and signals having different mean-reversion speeds. Two principles: "aim in front of the target" and "trade partially toward the current aim." The right way to translate predictive signal into a position with costs.
- Relevance: 5/5 — exactly the math we need for converting our ES→SPY signal into target SPY weight.

### Paper 4.4 — Quantifying the High-Frequency Trading "Arms Race"
- Authors: M. Aquilina, E. Budish, P. O'Neill
- Year: 2020
- Venue: Quarterly Journal of Economics
- DOI: 10.1093/qje/qjab032
- Consensus URL: https://consensus.app/papers/details/3b517f61fc9f5cd3b45bf2d034387c72/
- Citations: 130
- Abstract summary: Stock-exchange message data reveals ~1 latency race per minute per FTSE 100 stock, modal race 5–10 microseconds, ~20% of volume. Quantifies the cost of latency-arbitrage as a ~0.5bp tax on trading. Tells us what retail latency cannot compete on.
- Relevance: 5/5 — sets a hard floor on what races we cannot win, which sharpens our strategy choices.

### Paper 4.5 — Equity Trading in the 21st Century: An Update
- Authors: J. Angel, L. Harris, C. Spatt
- Year: 2015
- Venue: Quarterly Journal of Finance
- DOI: 10.1142/S2010139215500032
- Consensus URL: https://consensus.app/papers/details/5745feb312255f9bb82bbefc4fe3c2e5/
- Citations: 123
- Abstract summary: Updates the famed 2010 paper on U.S. market quality. Transaction costs remain low, depth is high, intraday volatility below pre-electronic 1990s. Frames where the market is "good enough" for retail trading and where pain points remain.
- Relevance: 4/5 — context-setting for retail strategy feasibility.

### Paper 4.6 — Trading Costs (Frazzini, Israel, Moskowitz)
- Authors: A. Frazzini, R. Israel, T. Moskowitz
- Year: 2018
- Venue: SSRN Capital Markets / Review of Asset Pricing Studies
- DOI: 10.2139/ssrn.2294498
- Consensus URL: https://consensus.app/papers/details/1116811b1f9754f4bd788ee201a7f4e0/
- Citations: 117
- Abstract summary: 1.7 trillion dollars of live institutional executions across 21 markets, 19 years. Real-world trading costs are an order of magnitude smaller than prior literature suggested. Critical reality check — costs don't kill all medium-frequency alpha, but the assumption matters.
- Relevance: 4/5 — institutional-scale findings inform our retail expectations.

### Paper 4.7 — Statistical Arbitrage: Medium Frequency Portfolio Trading
- Authors: I. Skachkov
- Year: 2013
- Venue: SSRN working paper
- DOI: 10.2139/ssrn.2284932 — verify
- Consensus URL: https://consensus.app/papers/details/23fb34eafc705dcd8e724abe1991830d/
- Citations: 0
- Abstract summary: Defines medium-frequency as the band where microstructure analysis isn't required but market impact is non-negligible. Demonstrates statistical-arbitrage signals on S&P 500 members. Low citation but a clean conceptual reference for our regime.
- Relevance: 4/5 — frames our operating frequency band.

---

## Topic 5: The honest skeptic literature

### Paper 5.1 — Data-Snooping, Technical Trading Rule Performance, and the Bootstrap (Sullivan, Timmermann, White)
- Authors: R. Sullivan, A. Timmermann, H. White
- Year: 1999
- Venue: Journal of Finance
- DOI: 10.1111/0022-1082.00163
- Consensus URL: https://consensus.app/papers/details/d44e019194235c9e94e158ebd482f2a1/
- Citations: 1088
- Abstract summary: Applies White's Reality Check to ~7000 technical trading rules on 100 years of Dow data. Most apparent profits do not survive the data-snooping correction. The canonical paper to reread before declaring any signal real.
- Relevance: 5/5 — required reading before celebrating any backtest result.

### Paper 5.2 — A Reality Check for Data Snooping (White)
- Authors: Halbert White
- Year: 2000
- Venue: Econometrica
- DOI: 10.1111/1468-0262.00152
- Consensus URL: https://consensus.app/papers/details/f4c4bac11d4b5a55ab4f28e06a668038/
- Citations: 1795
- Abstract summary: Original Reality Check methodology for testing whether the best of N strategies has genuine predictive power vs a benchmark. Foundational tool for honest model selection.
- Relevance: 5/5 — the methodology to formally test our top strategies.

### Paper 5.3 — Data Snooping Bias in Tests of the Relative Performance of Multiple Forecasting Models
- Authors: D. Anghel
- Year: 2021
- Venue: Journal of Banking & Finance
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/326b50dbce575d2f88abad9535585fad/
- Citations: 2
- Abstract summary: Monte Carlo simulation shows false-discovery rate rises 0.72–2.5 percentage points per halving of the alternative-strategy universe. Estimates that at least 50% of published positive findings on trading-rule outperformance may be false.
- Relevance: 5/5 — humbling estimate to keep in mind for our paper trading.

### Paper 5.4 — Understanding Alpha Decay (Pénasse)
- Authors: Julien Pénasse
- Year: 2017 (Management Science 2022)
- Venue: Management Science
- DOI: 10.1287/mnsc.2021.4216
- Consensus URL: https://consensus.app/papers/details/9c865587c4bc525a81e07389f533b808/
- Citations: 13
- Abstract summary: Shows that measured alpha differs from true alpha by ~1.4% per year on common anomalies because alpha decay is non-stationary; provides a formula to correct asset-pricing tests. Tells us anomalies decay even before publication.
- Relevance: 5/5 — direct calibration for how to discount our expected alpha.

### Paper 5.5 — Not All Factors Crowd Equally: Modeling, Measuring, and Trading on Alpha Decay
- Authors: Chorok Lee
- Year: 2025
- Venue: Working paper
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/ec289d6694165af39fbeac15bf460bcf/
- Citations: 0
- Abstract summary: Derives hyperbolic decay form α(t)=K/(1+λt) from a game-theoretic equilibrium and tests it on Fama-French factors 1963–2024. Mechanical factors (momentum, reversal) fit; judgment factors (value, quality) do not. Crowding accelerated post-2015. Crowding predicts crashes, not means.
- Relevance: 4/5 — fresh paper directly relevant to our crowding-risk concerns.

### Paper 5.6 — The Expected Returns on Machine-Learning Strategies
- Authors: V. Azevedo et al.
- Year: 2024
- Venue: SSRN / forthcoming JFE-tier
- DOI: 10.2139/ssrn.4071581 — verify
- Consensus URL: https://consensus.app/papers/details/b14f51e7be805786b4181f397894849c/
- Citations: 2
- Abstract summary: Net-of-cost ML anomaly strategies still earn ~1.42% monthly out-of-sample after accounting for transaction costs, post-publication decay, and the post-decimalization era. LSTM models particularly strong. Counters the harshest claims that ML alpha is illusory.
- Relevance: 5/5 — encouraging counterpoint to the skeptic papers; tells us ML edge can persist with discipline.

### Paper 5.7 — Man versus Machine Learning Revisited
- Authors: Y. Zhang et al.
- Year: 2025
- Venue: The Review of Financial Studies
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/7062b1665db05d26b407f60941dd7e75/
- Citations: 0
- Abstract summary: Re-examines a 1.54% monthly alpha from a published ML model. Identifies look-ahead bias; removing it erases the alpha entirely. Linear models with no bias forecast equally well with superior profits. Cautionary tale for ML-vs-linear evaluations.
- Relevance: 5/5 — the most pointed warning in our skeptic pack.

### Paper 5.8 — Evaluating Technical Trading Strategies in U.S. Stocks: Insights From Data-Snooping Test
- Authors: Yifei Wang et al.
- Year: 2025
- Venue: Journal of Accounting and Finance
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/773f47330a855942aff1ae5b6d32f8e5/
- Citations: 0
- Abstract summary: 2000–2022 evaluation of technical trading strategies on AAPL/MSFT/NVDA using reality-check and stepwise tests. After correcting for data snooping, no rules consistently profitable in or out of sample; signal-survival especially weak post-Feb-2016.
- Relevance: 3/5 — single-stock focus rather than index-trading, but useful as a 2026-vintage skeptic data point.

---

## Top 10 most-relevant papers for immediate download

Ranked by combined judgment on (a) ES→SPY lead-lag relevance, (b) online-learning algorithmic fit, (c) retail-latency reality, (d) realistic alpha calibration:

1. Hasbrouck, "Intraday Price Formation in U.S. Equity Index Markets" — **DOI: 10.1111/1540-6261.00598** (Topic 1.2)
2. Chordia et al., "Rent Seeking by Low-Latency Traders: Evidence from Trading on Macroeconomic Announcements" — DOI not retrievable; URL https://consensus.app/papers/details/90af8c793b6a5e1e97feae01b2b893d7/ (Topic 1.1)
3. Wallace et al., "The Evolution of Price Discovery in U.S. Equity and Derivatives Markets" — DOI not retrievable; URL https://consensus.app/papers/details/c51b2439601c54c292d35aa5b3a2a2b5/ (Topic 1.3)
4. Dao et al., "Ultra-high-frequency lead–lag relationship and information arrival" — DOI not retrievable; URL https://consensus.app/papers/details/9f14e3e5764e50d89e7760acafc35048/ (Topic 1.4)
5. Kolm, Turiel & Westray, "Deep Order Flow Imbalance" — **DOI: 10.1111/mafi.12340** (Topic 2.1)
6. Cont, Cucuringu & Zhang, "Cross-Impact of Order Flow Imbalance in Equity Markets" — **DOI: 10.1080/14697688.2023.2236159** (Topic 2.2)
7. Gârleanu & Pedersen, "Dynamic Trading with Predictable Returns and Transaction Costs" — **DOI: 10.1111/jofi.12080** (Topic 4.3)
8. Tsagaris, Jasra & Adams, "Robust and Adaptive Algorithms for Online Portfolio Selection" — **DOI: 10.1080/14697688.2010.494614** (Topic 3.1)
9. Sullivan, Timmermann & White, "Data-Snooping, Technical Trading Rule Performance, and the Bootstrap" — **DOI: 10.1111/0022-1082.00163** (Topic 5.1)
10. Hasbrouck, "Trading Costs and Returns for U.S. Equities: Estimating Effective Costs from Daily Data" — **DOI: 10.1111/j.1540-6261.2009.01469.x** (Topic 4.2)

Honorable mentions (download if time): Boehmer et al. (BJZZ retail-flow), Aquilina/Budish/O'Neill (HFT arms race), Pénasse (alpha decay correction), Briola et al. (LOBFrame honest-evaluation), Zhang et al. (Man vs ML revisited).

---

## Coverage notes

- Topic 1 (lead-lag): excellent coverage. Multiple direct ES/SPY-on-millisecond papers.
- Topic 2 (order-book ML): excellent coverage. Strong recent deep-learning literature; lots of retail-relevant feature-engineering work.
- Topic 3 (online learning): mixed coverage. RLS-family papers are mostly from outside finance — finance-specific online learning is thinner than expected; most "online learning" results in this corpus are RL/market-making rather than pure online regression. The Tsagaris-Jasra-Adams paper is the closest hit.
- Topic 4 (retail alpha): strong on the cost side (Hasbrouck 2009, Frazzini et al., Adams et al.). Thinner on retail-specific alpha sources (Boehmer et al. is the best modern hit).
- Topic 5 (skeptic literature): excellent coverage. Pénasse, Sullivan-Timmermann-White, Zhang et al., Lee 2025, and the Azevedo et al. counterpoint give a balanced view.
