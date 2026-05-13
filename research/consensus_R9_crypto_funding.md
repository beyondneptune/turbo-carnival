# Consensus R9 — Cryptocurrency Funding Rate Arbitrage & Perpetual Swaps
Date: 2026-05-12
Source: Consensus MCP (Semantic Scholar / PubMed / Scopus / ArXiv)
Status: PARTIAL — monthly search quota exhausted before query 4

---

## Quota Status

Consensus MCP free monthly quota (30 searches) was exhausted partway through this batch.
Queries 1–3 returned full results (10 papers each). Query 4 ("stablecoin crypto market microstructure")
returned the quota-exhausted error after multiple retries:
> "You've used all 30 searches this month; limits reset on June 1st. Upgrade to Consensus Pro to get 1,000 searches per month..."
> https://consensus.app/pricing/?utm_source=claude_desktop

Queries executed:
1. "Bitcoin perpetual futures funding rate arbitrage"  — returned 19 papers (top 10 shown)
2. "cryptocurrency cash and carry trade"               — returned 20 papers (top 10 shown)
3. "crypto basis trade perpetual spot"                 — returned 20 papers (top 10 shown)
4. "stablecoin crypto market microstructure"           — NOT EXECUTED (quota exhausted)

Total unique papers collected: 27 (after de-duplication of papers appearing in multiple queries).

---

## Relevance Scoring Rubric

Relevance 1–5 to a **BTC funding cash-and-carry strategy** (long spot BTC + short perpetual,
harvest funding rate):

- **5** — Directly studies BTC perp funding-rate cash-and-carry arbitrage with empirical P&L.
- **4** — Models perpetual no-arbitrage pricing / funding mechanism; directly informs sizing,
   replication, or risk of the strategy.
- **3** — Crypto carry / cross-exchange BTC arbitrage; informative on convenience yield,
   capital constraints, or risk premia driving funding.
- **2** — Crypto derivatives market microstructure / price discovery / volatility — useful
   context for execution and venue choice but not the carry itself.
- **1** — Tangential (general crypto trading surveys, ML price prediction, behavioral / mental
   health, alt-coin specific).

---

## Query 1 — "Bitcoin perpetual futures funding rate arbitrage"

### [1] Designing funding rates for perpetual futures in cryptocurrency markets
- Authors: Jaehyun Kim et al.
- Year: 2025
- Journal: Unknown Journal (working paper)
- DOI: not provided
- URL: https://consensus.app/papers/details/6c852ec1227255a5b2ff0dbd56758bde/?utm_source=claude_desktop
- Citations: 0
- Summary: Develops a continuous-time framework (path-dependent infinite-horizon BSDEs +
  arbitrage pricing) showing how funding-rate design can keep perpetual prices aligned with
  target spot. Also derives replicating portfolios for short-side hedgers and characterizes
  path-dependent vs standard funding-rate divergence over time.
- Relevance: **4**

### [2] Fundamentals of Perpetual Futures
- Authors: Song He et al.
- Year: 2022
- Journal: SSRN Electronic Journal
- DOI: not provided
- URL: https://consensus.app/papers/details/244451f06f9158d2b63cc3c1e9dd0484/?utm_source=claude_desktop
- Citations: 11
- Summary: Derives no-arbitrage prices for perpetual futures in frictionless markets and
  bounds under trading costs, then measures empirical deviations across crypto markets.
  Documents that the implied arbitrage strategy (cash-and-carry on perps) yields high
  Sharpe ratios; deviations comove across coins and shrink over time.
- Relevance: **5**

### [3] The Two-Tiered Structure of Cryptocurrency Funding Rate Markets
- Authors: Peter Zhivkov
- Year: 2026
- Journal: Mathematics
- DOI: not provided
- URL: https://consensus.app/papers/details/4de70fc431b6539282881a4eb6658ef1/?utm_source=claude_desktop
- Citations: 0
- Summary: 35.7M one-minute observations across 26 CEX + DEX venues show that ~17% of
  observations have ≥20bp arbitrage spreads but only 40% of top opportunities are
  profitable after transaction costs and spread reversal. Delta-neutral simulations
  show forced exits in 95% of opportunities, explaining why fragmented funding markets
  remain inefficient.
- Relevance: **5**

### [4] A comparison of machine learning and econometric models for pricing perpetual Bitcoin futures
- Authors: Avinash Malik
- Year: 2023
- Journal: Expert Systems
- DOI: not provided
- URL: https://consensus.app/papers/details/03d80da7d9fd5d6c9fc86bccb029b4a2/?utm_source=claude_desktop
- Citations: 1
- Summary: Uses EGARCH for intraday volatility and ARMA(0,0) for conditional mean of BTC
  perpetual returns, then builds a market-directional HFT algo around the funding-rate-
  adjusted no-arb price. Backtests claim 1500–8000% returns depending on fees/leverage
  with ~85% directional accuracy on best model.
- Relevance: **3**

### [5] Exploring Arbitrage Opportunities between the BTC Spot and Futures Market based on Funding Rates Mechanism
- Authors: Jiahua Zou
- Year: 2022
- Journal: ICFIED 2022 Conference Proceedings
- DOI: not provided
- URL: https://consensus.app/papers/details/ca8e735d14f45eae88c80094e3a60f75/?utm_source=claude_desktop
- Citations: 1
- Summary: Empirically tests spot-vs-perp arbitrage on Binance based on the funding-rate
  mechanism. Finds arbitrage profits are substantially larger during BTC market crashes
  than in normal periods, while the risk profile of the trade remains relatively low.
- Relevance: **5**

### [6] Arbitrage, contract design, and market structure in Bitcoin futures markets
- Authors: Riccardo De Blasis et al.
- Year: 2022
- Journal: Journal of Futures Markets
- DOI: not provided
- URL: https://consensus.app/papers/details/d39bdfdbc59154ed801c4d116938958c/?utm_source=claude_desktop
- Citations: 15
- Summary: Compares BTC quarterly futures vs perpetuals; finds quarterly contracts offer
  cash-and-carry arbitrage primarily during market dislocations, consistent with Hattori
  & Ishida (2021). Documents seasonal and "u-shaped" intraday effects in perpetuals despite
  24/7 trading, plus spillovers between perp and quarterly contracts.
- Relevance: **5**

### [7] The Impact of Derivatives on Spot Markets: Evidence from the Introduction of Bitcoin Futures Contracts
- Authors: Patrick Augustin et al.
- Year: 2023
- Journal: Management Science
- DOI: 10.1287/mnsc.2023.4900 (referenced as supplemental)
- URL: https://consensus.app/papers/details/a53cd162633953b2837014d35356c0f5/?utm_source=claude_desktop
- Citations: 23
- Summary: Uses the December 2017 CME BTC futures launch as a natural experiment and shows
  that futures introduction raised cross-exchange BTC-USD price synchronicity, reduced
  arbitrage opportunities and volatility, and improved spot market quality. Implies that
  futures relax short-sale constraints and reduce segmentation across spot venues.
- Relevance: **3**

### [8] Short Communication: A Primer on Perpetuals
- Authors: Guillermo Angeris et al.
- Year: 2023
- Journal: SIAM Journal on Financial Mathematics
- DOI: not provided
- URL: https://consensus.app/papers/details/90019fc3a4fb52668c9b5d4e2491763c/?utm_source=claude_desktop
- Citations: 2
- Summary: Derives model-free funding-rate and discount-rate expressions for two classes
  of perpetual contracts in continuous-time no-arbitrage markets, with replication
  strategies for the short side. Extends results to jump-diffusion settings and connects
  perpetuals to variance swaps and leveraged ETFs.
- Relevance: **4**

### [9] Bitcoin arbitrage
- Authors: Andrei Shynkevich
- Year: 2020
- Journal: Finance Research Letters
- DOI: not provided
- URL: https://consensus.app/papers/details/1c6814f7637b51848748c28f70bf717e/?utm_source=claude_desktop
- Citations: 17
- Summary: Documents inter-exchange arbitrage at the four BTC venues underlying the CME
  index, with large/persistent deviations in 2016–17 collapsing to small/sporadic
  opportunities post-2018 once transaction costs are included. Shows the arbitrage
  market has matured but profitable spread trades have become scarce.
- Relevance: **3**

### [10] Interest Rate Parity in Decentralized Finance
- Authors: Amit Chaudhary et al.
- Year: 2023
- Journal: SSRN Electronic Journal
- DOI: not provided
- URL: https://consensus.app/papers/details/3414e7e9972a5ca1bc2460aedde695ee/?utm_source=claude_desktop
- Citations: 6
- Summary: Links DeFi lending rates to the ETH/USDT perpetual premium and finds a
  statistically significant but economically weak relationship — i.e. DeFi rates and
  perp funding are only loosely tied. Wide no-arbitrage bounds from gas, trading costs,
  and price impact keep the two markets segmented.
- Relevance: **3**

---

## Query 2 — "cryptocurrency cash and carry trade"

### [11] Risk-Return Relation of Cryptocurrency Carry Trade
- Authors: Zhenzhen Fan et al.
- Year: 2023
- Journal: SSRN Electronic Journal
- DOI: not provided
- URL: https://consensus.app/papers/details/11d5f8242d565207913c9af629b31700/?utm_source=claude_desktop
- Citations: 0
- Summary: Documents large uncovered-interest-rate-parity violations in crypto using realistic
  borrow/lend rates; the cross-sectional crypto carry trade yields 46.71% annualized with
  Sharpe 0.77. Unlike fiat carry, the strategy was resilient through 2018 and 2021
  crashes, and geopolitical risk explains a substantial share of returns.
- Relevance: **4**

### [12] The Risk and Return of Cryptocurrency Carry Trade
- Authors: Zhenzhen Fan et al.
- Year: 2023
- Journal: SSRN Electronic Journal
- DOI: not provided
- URL: https://consensus.app/papers/details/1d130bd213685835ab6912d8b22c98a2/?utm_source=claude_desktop
- Citations: 0
- Summary: Companion / extended analysis: cross-sectional crypto carry strategy returns 49.3%
  annualized with Sharpe 0.81 and cannot be explained by standard crypto factors (market,
  size, momentum, volatility, downside, platform-collapse risk). Returns load on equity
  market volatility risk, suggesting cross-asset linkages.
- Relevance: **4**

### [13] Crypto Carry
- Authors: Maik Schmeling et al.
- Year: 2022
- Journal: SSRN Electronic Journal
- DOI: not provided
- URL: https://consensus.app/papers/details/58246d6d63bb54bea47cea0af6b3253d/?utm_source=claude_desktop
- Citations: 10
- Summary: Shows crypto futures carry (futures-spot) can reach 60% p.a. and is driven by a
  highly volatile convenience yield: retail leverage demand vs scarce arbitrage capital
  taking the cash-and-carry side. The risk to the carry trade comes from margin spikes
  and liquidation cascades during drawdowns, helping explain frequent crypto crashes.
- Relevance: **5**

### [14] Crypto Carry (Franz et al.)
- Authors: Friedrich-Carl Franz et al.
- Year: 2021
- Journal: Mutual Funds (working paper)
- DOI: not provided
- URL: https://consensus.app/papers/details/c58edabb9ac45aaa9de3cd551462c539/?utm_source=claude_desktop
- Citations: 0
- Summary: Studies carry trades in crypto and identifies the BTC/USD perpetual swap funding
  rate as a direct measure of carry; lack of arbitrage capital + heavy speculator
  leverage creates persistent carry opportunities. Finds carry signals (fear/greed +
  past returns) positively predict returns in BTC and across 51 cryptocurrencies.
- Relevance: **5**

### [15] Cryptocurrency trading: a comprehensive survey
- Authors: Fan Fang et al.
- Year: 2020
- Journal: Financial Innovation
- DOI: not provided
- URL: https://consensus.app/papers/details/f5c9df8ba69f5b65973f3d03946333e2/?utm_source=claude_desktop
- Citations: 352
- Summary: Synthesizes 146 papers on crypto trading systems, signals, strategies, portfolio
  construction, and risk management. Useful for situating the funding-rate carry within
  the broader crypto trading research landscape.
- Relevance: **2**

### [16] Trading and arbitrage in cryptocurrency markets
- Authors: I. Makarov et al.
- Year: 2018
- Journal: Capital Markets: Asset Pricing & Valuation eJournal
- DOI: not provided
- URL: https://consensus.app/papers/details/01c5c66a8f435ca7bc74a4e8ee90e243/?utm_source=claude_desktop
- Citations: 607
- Summary: Foundational empirical study of cross-exchange BTC arbitrage. Documents large,
  recurrent arbitrage spreads across countries (much larger than within country or
  between coins), driven by capital controls; common signed-volume component explains
  80% of BTC returns.
- Relevance: **3**

### [17] Herding and feedback trading in cryptocurrency markets
- Authors: T. King et al.
- Year: 2021
- Journal: Annals of Operations Research
- DOI: not provided
- URL: https://consensus.app/papers/details/76d6de706a285f2380f956d6ad1426a3/?utm_source=claude_desktop
- Citations: 83
- Summary: Tests for herding / trend-chasing vs contrarian behavior across nine major
  cryptocurrencies and finds heterogeneous strategies across markets. Relevant only as
  context for the demand side of funding-rate carry (retail leveraged longs).
- Relevance: **1**

### [18] Cryptocurrency Trading and Associated Mental Health Factors: A Scoping Review
- Authors: Lakshita Jain et al.
- Year: 2025
- Journal: Journal of Primary Care & Community Health
- DOI: not provided
- URL: https://consensus.app/papers/details/4e545d9aabf755a599ae3a3ab554732a/?utm_source=claude_desktop
- Citations: 5
- Summary: Scoping review (13 studies, 11,177 participants) on psychological factors in
  crypto trading — addiction-like behavior, overconfidence, herding via social media.
  Not relevant to quantitative cash-and-carry strategy design.
- Relevance: **1**

### [19] Technical trading and cryptocurrencies
- Authors: R. Hudson et al.
- Year: 2019
- Journal: Annals of Operations Research
- DOI: not provided
- URL: https://consensus.app/papers/details/2d56d9545ee95445af14d6deaf6f2c6e/?utm_source=claude_desktop
- Citations: 82
- Summary: Tests ~15,000 technical trading rules on BTC and three other coins, finding
  significant predictability and high breakeven costs; BTC predictability disappears
  out-of-sample. Not directly relevant to funding-rate arbitrage.
- Relevance: **1**

### [20] Forecasting and trading cryptocurrencies with machine learning under changing market conditions
- Authors: H. Sebastião et al.
- Year: 2021
- Journal: Financial Innovation
- DOI: not provided
- URL: https://consensus.app/papers/details/0f77a168f9855e7abf51a41fc4b23418/?utm_source=claude_desktop
- Citations: 191
- Summary: Validates ML-based directional trading strategies on BTC/ETH/LTC; ensembles
  reach Sharpe ~0.8–0.9 after costs on ETH and LTC. Tangential to funding-rate carry.
- Relevance: **1**

---

## Query 3 — "crypto basis trade perpetual spot"

### [21] Fundamentals of Perpetual Futures (DUPLICATE — see [2])
- DOI: not provided
- URL: https://consensus.app/papers/details/244451f06f9158d2b63cc3c1e9dd0484/?utm_source=claude_desktop
- Note: Already listed as paper [2]. Same relevance score: **5**.

### [22] A Study on the Arbitrage Trading with Bitcoin Perpetual Swap
- Authors: Junghwan Cho et al.
- Year: 2022
- Journal: Journal of the Korea Academia-Industrial cooperation Society
- DOI: not provided
- URL: https://consensus.app/papers/details/06c8c2c8041e54bbaffb7eb81ff26bd3/?utm_source=claude_desktop
- Citations: 0
- Summary: Empirical analysis of inter-exchange arbitrage using BTC perpetual swap vs spot
  price differences. Finds sign-reversals between perp and spot create arbitrage; minute
  transactions average 9% monthly (very fee-sensitive), daily transactions 38% monthly
  with positive returns less sensitive to fees.
- Relevance: **5**

### [23] A Low-Volatility Strategy based on Hedging a Quanto Perpetual Swap on BitMEX
- Authors: Daniel Atzberger et al.
- Year: 2024
- Journal: 2024 IEEE Intl. Conf. on Blockchain and Cryptocurrency (ICBC)
- DOI: not provided
- URL: https://consensus.app/papers/details/097ce71dc2d9518986afcfc272b68cf8/?utm_source=claude_desktop
- Citations: 1
- Summary: Builds a strategy combining a Quanto perpetual swap with a spot position to harvest
  the funding mechanism, comparing against long-only and linear-perp-based versions over
  2021–2022 backtests. Finds the quanto-hedged variant produces positive bull-market
  returns with lower volatility.
- Relevance: **4**

### [24] Price discovery in Bitcoin: The impact of unregulated markets
- Authors: C. Alexander et al.
- Year: 2020
- Journal: ERN: Other Econometrics, Applied Econometric Modeling in Financial Economics
- DOI: not provided
- URL: https://consensus.app/papers/details/7a19c55afb1c53c0a694621afc4ec6d7/?utm_source=claude_desktop
- Citations: 74
- Summary: Minute-level information-flow analysis showing perpetual swaps and futures on
  Huobi, OKEx, and BitMEX dominate BTC price discovery, with CME and US spot venues
  reacting slowly. Implies that funding-rate dynamics on unregulated venues lead the
  basis, with implications for hedging-venue selection.
- Relevance: **3**

### [25] A comparison of machine learning and econometric models for pricing perpetual Bitcoin futures (DUPLICATE — see [4])
- URL: https://consensus.app/papers/details/03d80da7d9fd5d6c9fc86bccb029b4a2/?utm_source=claude_desktop
- Note: Already listed as paper [4]. Same relevance score: **3**.

### [26] The Role of Binance in Bitcoin Volatility Transmission
- Authors: C. Alexander et al.
- Year: 2021
- Journal: Applied Mathematical Finance
- DOI: not provided
- URL: https://consensus.app/papers/details/f0610b56ae755a069d38fe39e1f67d96/?utm_source=claude_desktop
- Citations: 35
- Summary: High-frequency realized-volatility spillover analysis identifying the
  tether-margined Binance BTC perpetual as the dominant volatility source across CEX
  venues; cross-venue connectedness rises when Western equity markets are open.
  Important context for executing perp legs of a cash-and-carry on Binance.
- Relevance: **3**

### [27] Price discovery and microstructure in ether spot and derivative markets
- Authors: C. Alexander et al.
- Year: 2020
- Journal: ERN: Other Microeconomics: General Equilibrium & Disequilibrium Models of Financial Markets
- DOI: not provided
- URL: https://consensus.app/papers/details/55ec94900d665b0e8c32920c36a96686/?utm_source=claude_desktop
- Citations: 29
- Summary: ETH analog of paper [24]: shows BitMEX ether perpetual swap leads spot price
  discovery, with diurnal/weekly patterns indicating informed institutional flows.
  Relevant as an ETH analog if the strategy is later extended beyond BTC.
- Relevance: **2**

### [28] Trading Periodicity and Algorithmic Divide in Cryptocurrency Markets
- Authors: Andrei Shynkevich
- Year: 2026
- Journal: Journal of Futures Markets
- DOI: not provided
- URL: https://consensus.app/papers/details/92c71faa5bc0518bb3ca8e84d839c726/?utm_source=claude_desktop
- Citations: 0
- Summary: Documents subsecond periodicity in BTC/ETH spot and perpetual markets, with spot
  patterns matching agency execution algos and perpetual patterns matching proprietary
  liquidity-taking. Periodic activity surges raise volatility and costs but not price
  impact; useful for execution scheduling.
- Relevance: **2**

### [29] Yields: The Galapagos Syndrome Of Cryptofinance
- Authors: B. Meister et al.
- Year: 2022
- Journal: Unknown (book chapter)
- DOI: not provided
- URL: https://consensus.app/papers/details/97db9d4549755b718f69c9a9c7b392b4/?utm_source=claude_desktop
- Citations: 3
- Summary: Surveys yield-generation structures in crypto: lending, AMM LP fees, PoS staking,
  and the funding-rate yield from perpetuals / fixed-duration futures trading at
  premium/discount. Frames stablecoins, perpetuals, and crypto-native exchanges as
  evolutionary adaptations.
- Relevance: **3**

### [30] Towards Understanding Cryptocurrency Derivatives: A Case Study of BitMEX
- Authors: K. Soska et al.
- Year: 2021
- Journal: Proceedings of the Web Conference 2021
- DOI: not provided
- URL: https://consensus.app/papers/details/8af00dbed1ae55caa2ef7f288108f498/?utm_source=claude_desktop
- Citations: 28
- Summary: Case study of BitMEX's evolution from spot to leveraged derivatives, mapping
  settled vs perpetual contracts and using on-chain forensics + liquidation events to
  profile traders (amateur leveraged retail to professional automated). Documents how
  derivative trading drives spot price moves.
- Relevance: **3**

---

## Query 4 — "stablecoin crypto market microstructure"

NOT EXECUTED. Quota exhausted before this query could run. Suggest re-running this
single query after the June 1 quota reset, or by upgrading to Consensus Pro.

---

## Summary

- Total papers collected: **27 unique** (30 raw, minus 3 duplicates: [21] = [2], [25] = [4],
  and an internal dup between Q1/Q3 for paper [2]).
- Highest-relevance results for BTC funding cash-and-carry: [2], [3], [5], [6], [13], [14], [22].
- Note: Consensus did not return DOI strings for any of the papers in this batch; only
  Semantic-Scholar-style detail URLs are available. The Augustin et al. (2023) paper
  references DOI `10.1287/mnsc.2023.4900` in the abstract text.

## DOI List (raw)

Consensus did not return parsed DOI fields for the papers in this batch. The only
DOI-like identifier surfaced in any abstract was:

- `10.1287/mnsc.2023.4900`  — Augustin et al. (2023), Management Science, paper [7]

All other papers have a Consensus detail URL (listed under each entry) but no DOI
field was returned.
