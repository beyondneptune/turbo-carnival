# R10 — News & LLM sentiment trading

Source: Consensus MCP (consensus.app) — search over Semantic Scholar, PubMed, Scopus, ArXiv.

Queries executed:
1. "news sentiment analysis stock return prediction" — 10 results returned
2. "large language model trading signal generation" — 10 results returned
3. "FOMC statement text analysis return prediction" — NOT RUN (Consensus monthly quota of 30 searches exhausted; resets June 1)
4. "earnings call sentiment intraday return drift" — NOT RUN (same quota exhaustion)

Note: queries 3 and 4 should be re-attempted after June 1, 2026 when the Consensus free-tier quota resets, or via a Consensus Pro upgrade.

DOIs/external IDs were not provided in the Consensus MCP response payload. Each paper below includes the Consensus permalink (which resolves to the canonical paper page) in lieu of a raw DOI. Relevance scores reflect fit to a paper-options / news-driven intraday-to-multi-day trading workflow.

---

## Query 1 — "news sentiment analysis stock return prediction"

### Paper 1
- Title: News vs. Sentiment: Predicting Stock Returns from News Stories
- Authors: S. Heston, Nitish Ranjan Sinha
- Year: 2017
- Journal: Financial Analysts Journal
- DOI / Link: https://consensus.app/papers/details/fa866db9094756a88c3006b24a053383/
- Citations: 135
- Summary: Uses >900,000 Thomson Reuters news stories scored by a proprietary neural-net sentiment model to show that daily news predicts returns for only 1–2 days, but weekly aggregated news predicts returns out to one quarter. Positive headlines are priced in quickly while negative headlines show a long delayed reaction concentrated around the next earnings announcement — implying a tradable underreaction to bad news.
- Relevance: 5/5

### Paper 2
- Title: News impact on stock price return via sentiment analysis
- Authors: Xiaodong Li, Haoran Xie, Li Chen, Jianping Wang, Xiaotie Deng
- Year: 2014
- Journal: Knowledge-Based Systems
- DOI / Link: https://consensus.app/papers/details/f4100fa803c85ed8873fb69a1563f254/
- Citations: 470
- Summary: Builds a prediction framework that projects financial news onto a sentiment space defined by the Harvard psychological dictionary and the Loughran–McDonald financial dictionary, then compares six models against a bag-of-words baseline on five years of HKEX data. Sentiment-augmented models beat bag-of-words at the stock, sector, and index levels, but polarity-only models add no predictive value — supporting graded sentiment scores over binary labels.
- Relevance: 4/5

### Paper 3
- Title: News sentiment and stock return: Evidence from managers' news coverages
- Authors: Yongan Xu, Jianqiong Wang, Zhonglu Chen, Chao Liang
- Year: 2022
- Journal: Finance Research Letters
- DOI / Link: https://consensus.app/papers/details/1504ec4cc8d9555294d258d8ae95a1d8/
- Citations: 24
- Summary: Constructs a monthly manager-sentiment index (S_M) from the tone of news coverage of corporate managers and shows it predicts next-month returns both in- and out-of-sample, adding information beyond standard economic predictors. The predictive power is concentrated in high-sentiment regimes and translates into measurable portfolio gains for sentiment-aware allocators.
- Relevance: 3/5

### Paper 4
- Title: The Power of Words: Predicting Stock Market Returns with Fine-Grained Sentiment Analysis and XGBoost
- Authors: Farshid Balaneji et al.
- Year: 2024
- Journal: (Unknown / conference paper)
- DOI / Link: https://consensus.app/papers/details/a1b708ecd4975e408d3040e173791a0b/
- Citations: 0
- Summary: Compares lexicon-based and deep-learning sentiment scoring at sentence, paragraph, and full-text granularity, feeding the result plus calendar/lagged-return features into XGBoost (with Bayesian hyperparameter tuning and nested CV) to classify next-day market direction on ten DJI stocks. Finds asymmetric sentiment sensitivity across sectors — petroleum reacts most strongly — suggesting sector-conditioned sentiment models.
- Relevance: 3/5

### Paper 5
- Title: News versus Sentiment: Predicting Stock Returns from News Stories (working paper version)
- Authors: S. Heston, Nitish Ranjan Sinha
- Year: 2015
- Journal: ERN: Asset Price Forecasts (SSRN working paper)
- DOI / Link: https://consensus.app/papers/details/21e3b26459d35bb1a88f9a60ca43de10/
- Citations: 93
- Summary: Working-paper precursor to Heston & Sinha (2017) with the same 900,000-story corpus and Reuters neural-net sentiment scoring. Confirms the 1–2-day daily horizon vs. quarter-long weekly horizon, and the asymmetric long-delayed reaction to negative news.
- Relevance: 4/5 (duplicate evidence — cite the FAJ version preferentially)

### Paper 6
- Title: Enhanced news sentiment analysis using deep learning methods
- Authors: W. Souma, I. Vodenska, H. Aoyama
- Year: 2019
- Journal: Journal of Computational Social Science
- DOI / Link: https://consensus.app/papers/details/aa87e336a90856e3b5a3b1acbf6c5228/
- Citations: 129
- Summary: Labels Reuters news articles by the 1-minute post-release return of the underlying stock, then trains an LSTM RNN on GloVe-Wikipedia/Gigaword embeddings over 2003–2012 DJIA-30 data and tests on 2013. Accuracy improves substantially when training is restricted to the strongest positive/negative articles rather than random samples — useful guidance for label-selection in supervised LLM fine-tuning.
- Relevance: 4/5

### Paper 7
- Title: Harvesting social media sentiment analysis to enhance stock market prediction using deep learning
- Authors: Pooja Mehta, Sharnil Pandya, Ketan Kotecha
- Year: 2021
- Journal: PeerJ Computer Science
- DOI / Link: https://consensus.app/papers/details/34473a03c7a05df7bfc092a06fa83433/
- Citations: 119
- Summary: Combines public social-media sentiment with news and historical prices to forecast stock direction using SVM, multinomial NB, linear regression, NB, and LSTM. Validates that social-media sentiment carries incremental information beyond news headlines and price history — relevant for Twitter/X/Reddit-augmented signal pipelines.
- Relevance: 3/5

### Paper 8
- Title: News vs. Sentiment: Predicting Stock Returns from News Stories (FAJ reprint, second citation cluster)
- Authors: S. Heston, Nitish Ranjan Sinha
- Year: 2017
- Journal: Financial Analysts Journal
- DOI / Link: https://consensus.app/papers/details/7b296e2454dc5234b5f7b9af7d81a843/
- Citations: 30
- Summary: Identical FAJ article surfacing under a separate Semantic Scholar fingerprint. Same headline finding: 1–2-day daily predictability, quarter-long weekly predictability, asymmetric delayed reaction to negative news. Treat as a duplicate of Paper 1 for citation purposes.
- Relevance: 4/5 (duplicate)

### Paper 9
- Title: News versus Sentiment: Predicting Stock Returns from News Stories (third cluster)
- Authors: Steven Heston, Nitish Ranjan Sinha
- Year: 2017
- Journal: Financial Analysts Journal
- DOI / Link: https://consensus.app/papers/details/201f384384b8536f9e07bd18eafa82bd/
- Citations: 41
- Summary: Same FAJ paper with explicit comparison of three sentiment methodologies — Harvard psychosocial dictionary (Tetlock-style), Loughran–McDonald financial dictionary, and a Thomson-Reuters neural net. Simpler bag-of-words approaches predict short reversible moves, while the neural-net signal predicts larger persistent returns — argues for sophisticated/learned sentiment over dictionaries.
- Relevance: 4/5 (duplicate but contains the methodology comparison in the abstract)

### Paper 10
- Title: Stock-specific sentiment and return predictability
- Authors: Guillaume Coqueret
- Year: 2020
- Journal: Quantitative Finance
- DOI / Link: https://consensus.app/papers/details/2c52b7a3d0d75b638ceff61658dc2ee7/
- Citations: 22
- Summary: Runs daily predictive regressions of next-day returns on stock-specific news sentiment for >1,000 US large-caps and finds significant t-stats for only ~7% of names; the dominant causal direction is in fact reverse — past returns drive future sentiment more than the other way around. Important sobering counterweight: argues against naive cross-sectional sentiment alphas at the individual-stock daily horizon.
- Relevance: 5/5 (negative result is critical for honest sizing)

---

## Query 2 — "large language model trading signal generation"

### Paper 11
- Title: Can Large Language Models beat wall street? Evaluating GPT-4's impact on financial decision-making with MarketSenseAI
- Authors: George Fatouros, Konstantinos Metaxas, John Soldatos, Manos Karathanassis
- Year: 2024
- Journal: Neural Computing and Applications
- DOI / Link: https://consensus.app/papers/details/42d587b0a95754968c32451159aa6262/
- Citations: 48
- Summary: Builds MarketSenseAI: a Chain-of-Thought + In-Context-Learning pipeline on GPT-4 that ingests trends, news, fundamentals, and macro to produce explained buy/sell signals, with GPT-4 also acting as a second-pass signal evaluator. Over a 15-month live test on the S&P 100 it reports 10–30% excess alpha and up to 72% cumulative return at risk comparable to the market.
- Relevance: 5/5

### Paper 12
- Title: Sentiment trading with large language models
- Authors: Kemal Kirtac, Guido Germano
- Year: 2024
- Journal: ArXiv preprint
- DOI / Link: https://consensus.app/papers/details/48cadb14c1f2521f831c11ef26c752e2/
- Citations: 59
- Summary: Scores 965,375 US financial news articles (2010-01 through 2023-06) with BERT, OPT (GPT-3-class), FinBERT, and Loughran–McDonald, then regresses next-day returns on each score. OPT wins at 74.4% directional accuracy and a long-short Sharpe of 3.05, vs FinBERT/BERT around 2.1 and L-M at 1.23 — direct empirical case that decoder-only LLMs beat dictionary methods for sentiment-driven trading.
- Relevance: 5/5

### Paper 13
- Title: The New Quant: A Survey of Large Language Models in Financial Prediction and Trading
- Authors: Weilong Fu
- Year: 2025
- Journal: (Survey preprint, unknown venue)
- DOI / Link: https://consensus.app/papers/details/03d51c36137355669813514b8a8af5fe/
- Citations: 1
- Summary: Synthesizes >50 primary studies plus prior surveys into a task-centered taxonomy (sentiment/event extraction, numerical reasoning, multimodal, RAG, time-series prompting, agentic systems) and discusses production-grade concerns: temporal leakage, hallucination, capacity, governance, interpretability. Useful map for picking an LLM signal architecture without re-deriving the design space.
- Relevance: 4/5

### Paper 14
- Title: An End-To-End LLM Enhanced Trading System
- Authors: Ziyao Zhou et al.
- Year: 2025
- Journal: Journal of Applied Business and Economics
- DOI / Link: https://consensus.app/papers/details/23b44742bd3356ec9657673b7fc7aff4/
- Citations: 3
- Summary: Engineering-oriented paper that wires FinGPT-based sentiment over live news + social-media streams to technical indicators, deployed on Kubernetes for real-time signal generation. Useful as a reference architecture for an event-driven LLM signal stack — less rigorous on out-of-sample alpha than the academic comps.
- Relevance: 4/5

### Paper 15
- Title: A hybrid approach to formulaic alpha discovery with large language model assistance
- Authors: Shuo Yu et al.
- Year: 2025
- Journal: Frontiers of Computer Science
- DOI / Link: https://consensus.app/papers/details/1b36403bd91e5904926559588e0db724/
- Citations: 0
- Summary: Uses LLMs either as the sole alpha generator or as a co-optimizer alongside a reinforcement-learning baseline to produce explicit-formula alpha factors. Reports an average IC of 0.0515 (75% over the RL-only baseline) and >2x cumulative excess return on real stock data — supports LLM-in-the-loop factor mining rather than direct LLM trading.
- Relevance: 3/5

### Paper 16
- Title: Quantifying legal risk with Large Language Models: A text-based investment signal
- Authors: James Offutt et al.
- Year: 2025
- Journal: Journal of High School Science
- DOI / Link: https://consensus.app/papers/details/f047329359db5f2fa2ff7d062e25974b/
- Citations: 0
- Summary: Has GPT-4o diff the Legal Proceedings sections of consecutive 10-Q/10-K filings and turn the directional change into a monthly long-short signal sized by tanh of historical volatility. Produces 0.471%/month alpha (5.80% annualized) after Fama–French 5 + momentum controls — concrete template for filing-diff LLM signals beyond raw sentiment.
- Relevance: 4/5

### Paper 17
- Title: QuantAgent: Seeking Holy Grail in Trading by Self-Improving Large Language Model
- Authors: Sai Wang et al.
- Year: 2024
- Journal: ArXiv preprint
- DOI / Link: https://consensus.app/papers/details/d54a41fc75e55fd79ddb34487f912061/
- Citations: 26
- Summary: Two-layer LLM agent loop: an inner loop that mines trading signals from a knowledge base and an outer loop that backtests signals and writes the learnings back to the KB. Demonstrates self-improvement convergence and reports gains in forecast accuracy — relevant to agentic / self-improving signal stacks.
- Relevance: 3/5

### Paper 18
- Title: Language Model Guided Reinforcement Learning in Quantitative Trading
- Authors: Adam Darmanin et al.
- Year: 2025
- Journal: ArXiv preprint
- DOI / Link: https://consensus.app/papers/details/804f4dc10ad75ed09485fd78991532b1/
- Citations: 0
- Summary: Couples LLM-generated high-level trading strategies with a downstream RL execution agent and evaluates both economic rationale (expert review) and statistical performance (Sharpe, max-drawdown). LLM guidance improves both return and risk metrics over unguided RL — supports a hybrid LLM-strategy / RL-execution pattern.
- Relevance: 3/5

### Paper 19
- Title: Automate Strategy Finding with LLM in Quant investment
- Authors: Zhizhuo Kou et al.
- Year: 2024
- Journal: ArXiv preprint
- DOI / Link: https://consensus.app/papers/details/1e6cbfdafb4153e1b39df85ce7e0e242/
- Citations: 21
- Summary: Three-stage risk-aware multi-agent framework: prompt-engineered LLMs generate candidate alpha factors, multimodal agents filter by predictive quality + category balance, and dynamic weight optimization adapts to market regime. Reports 53.17% cumulative return on SSE50 over Jan 2023 – Jan 2024 — relevant to automated signal-discovery for live trading.
- Relevance: 3/5

### Paper 20
- Title: Large Language Models in equity markets: applications, techniques, and insights
- Authors: Aakanksha Jadhav et al.
- Year: 2025
- Journal: Frontiers in Artificial Intelligence
- DOI / Link: https://consensus.app/papers/details/434b99d68f495a91946c64b36b4602f9/
- Citations: 9
- Summary: Comprehensive review of 84 LLM-in-equities studies (2022 – early 2025), dual-categorized by financial application (forecasting / sentiment / portfolio / algo trading) and by technique (prompting / fine-tuning / multi-agent / RL / custom architectures). Inventories datasets and contrasts general-purpose vs finance-specialized LLMs — best single starting point for picking a stack.
- Relevance: 4/5

---

## Paper count

- Total papers cataloged: 20 (10 per query for the 2 successful queries)
- Unique top-tier results (after de-duplicating Heston & Sinha papers 1/5/8/9 to a single paper): 17
- Queries 3 and 4 not run due to Consensus monthly quota exhaustion (resets June 1, 2026)
