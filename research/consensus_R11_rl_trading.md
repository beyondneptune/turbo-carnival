# Consensus R11 — Reinforcement Learning for Trading and Execution
Date: 2026-05-12
Total queries planned: 4
Total queries successfully executed: 1
Total papers retained: 10
Source: Consensus MCP (Semantic Scholar / PubMed / Scopus / ArXiv)

Note: DOIs are not directly returned by Consensus. Where a DOI is not retrievable from the Consensus result payload, the Consensus URL is provided instead — operator can use it to land on the paper's detail page and grab the DOI for PDF retrieval. Authors / years / journal / citations are exactly as returned.

## Quota status
The monthly Consensus MCP allowance (30 searches / month, resets 2026-06-01) was exhausted after the first of the four requested queries. Queries 2-4 ("deep Q-learning market making", "PPO trading agent equity portfolio", "model-free reinforcement learning financial markets") returned the rate-limit / quota error and produced zero results this run. Re-run those three queries after 2026-06-01 (or with a Consensus Pro account) and append the additional papers under Topics 2-4 below.

Queries executed this run:
1. "reinforcement learning optimal execution trading" — 10 papers returned (top 10 of 20 found)
2. "deep Q-learning market making" — quota exceeded, 0 papers returned
3. "PPO trading agent equity portfolio" — quota exceeded, 0 papers returned
4. "model-free reinforcement learning financial markets" — quota exceeded, 0 papers returned

---

## Topic 1: Reinforcement learning for optimal execution / trading

### Paper 1.1 — Optimal Execution with Reinforcement Learning
- Authors: Yadh Hafsi et al.
- Year: 2024
- Venue: ArXiv
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/ecd61409d34558ffad5c18c189b783eb/
- Citations: 4
- Abstract summary: Develops an RL agent to learn an optimal execution strategy over a finite horizon, with state features drawn from the current limit order book and operating at high frequency. Uses the ABIDES multi-agent market simulator to avoid historical-data overfitting; the trained agent beats standard execution benchmarks and offers a practical foundation for real-world deployment.
- Relevance: 5/5 — directly on the topic; recent; uses an LOB simulator approach we could mirror for paper-trading evaluation.

### Paper 1.2 — Reinforcement learning for optimized trade execution
- Authors: Yuriy Nevmyvaka et al.
- Year: 2006
- Venue: Proceedings of the 23rd International Conference on Machine Learning (ICML)
- DOI: 10.1145/1143844.1143929
- Consensus URL: https://consensus.app/papers/details/d4c8cbb1a51c5cbcbb99a5f7f7215cff/
- Citations: 287
- Abstract summary: The first large-scale empirical RL application to optimized trade execution, using 1.5 years of millisecond NASDAQ limit-order data. Introduces a "low-impact" state-space factorization that exploits structure in the market-microstructure problem.
- Relevance: 5/5 — the canonical academic citation for RL on trade execution; required reading.

### Paper 1.3 — Benchmarking deep reinforcement learning approaches to trade execution
- Authors: Isaac Tonkin et al.
- Year: 2025
- Venue: Pacific-Basin Finance Journal
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/9114b8f890235ae688b45ef6fe6d7410/
- Citations: 0
- Abstract summary: Builds a consistent benchmarking framework so different deep RL trade-execution formulations can be compared on equal terms, evaluating across two trading horizons on liquid ASX equities. DRL approaches consistently beat the benchmark strategies on a median basis with lower standard deviation; the most effective configuration is the one with the largest action-space degrees of freedom.
- Relevance: 5/5 — gives us a framework template plus the empirical lesson that unconstrained action spaces beat hand-shaped ones.

### Paper 1.4 — Towards Generalizable Reinforcement Learning for Trade Execution
- Authors: Chuheng Zhang et al.
- Year: 2023
- Venue: not specified in Consensus result
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/7077cc21eb865ff28f9ef18034c768cf/
- Citations: 8
- Abstract summary: Identifies overfitting as the main blocker to real-world deployment of RL execution policies, models the problem as offline RL with dynamic context (ORDC), and derives a generalization bound showing the issue stems from large context space with limited offline samples. Proposes learning compact context representations (prior-knowledge or end-to-end) and validates on a high-fidelity LOB simulator.
- Relevance: 5/5 — speaks directly to the train/test generalization problem central to `paper_v1`'s held-out-Year-2 policy.

### Paper 1.5 — A Deep Reinforcement Learning Framework for Optimal Trade Execution
- Authors: Siyu Lin et al.
- Year: 2021
- Venue: not specified in Consensus result
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/48279dcaa4fd5de799f1bd56bdb969d5/
- Citations: 10
- Abstract summary: Combines Double DQN, Dueling Network, and Noisy Nets to split a parent sell order into child orders over a fixed horizon, with shaped rewards and a zero-ending-inventory constraint baked into the Q-update at the terminal step. Beats TWAP, VWAP, Almgren-Chriss, and two prior DQN baselines across 14 US equities.
- Relevance: 5/5 — concrete architecture and reward-shaping recipe that translates to our setting.

### Paper 1.6 — Practical Application of Deep Reinforcement Learning to Optimal Trade Execution
- Authors: Wook-Jae Byun et al.
- Year: 2023
- Venue: FinTech
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/c9eae57ed08351eb85b1de891f32d8a0/
- Citations: 6
- Abstract summary: Uses PPO with an LSTM head and a proprietary level-3 KRX simulator to achieve generalization across 50 stocks and execution horizons of 165-380 minutes with dynamic target volume. Claims first commercialized DRL execution algorithm in the South Korean market; beats VWAP benchmark.
- Relevance: 5/5 — overlaps directly with the planned R12 / PPO query; useful precedent for PPO+LSTM at production scale.

### Paper 1.7 — Federated Reinforcement Learning in Stock Trading Execution: The FPPO Algorithm for Information Security
- Authors: Haogang Feng et al.
- Year: 2025
- Venue: IEEE Access
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/9cda5b1c8a6155ee92dc83685c1739ca/
- Citations: 0
- Abstract summary: Proposes a Federated PPO (FPPO) algorithm for trade execution that combines joint RL with privacy preservation, reporting 3-15% average return improvements across stocks. Examines sensitivity to client-node count and risk preferences.
- Relevance: 3/5 — federated angle is tangential to a single-operator paper-trading repo, but the FPPO + risk-preference framing is useful color.

### Paper 1.8 — Reinforcement Learning for Trade Execution with Market and Limit Orders
- Authors: Patrick Cheridito et al.
- Year: 2025
- Venue: not specified in Consensus result
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/02fed995a58053eeb45d200bf883c5f2/
- Citations: 1
- Abstract summary: Frames execution as dynamic allocation between market and limit orders, modeling allocations with multivariate logistic-normal distributions so the policy can be trained efficiently. Outperforms benchmarks in a simulated LOB environment with noise, tactical, and strategic traders.
- Relevance: 4/5 — order-type mix is the right modeling abstraction; not all options work the same way but the discretization idea is transferable.

### Paper 1.9 — A reinforcement learning extension to the Almgren-Chriss framework for optimal trade execution
- Authors: Dieter Hendricks et al.
- Year: 2014
- Venue: 2014 IEEE Conference on Computational Intelligence for Financial Engineering & Economics (CIFEr)
- DOI: 10.1109/CIFEr.2014.6924109
- Consensus URL: https://consensus.app/papers/details/bd8f679b9a4e5bf789fff4bfc7b89256/
- Citations: 75
- Abstract summary: Uses RL on top of the standard linear-impact Almgren-Chriss model to dynamically adapt the volume trajectory to live spread and volume conditions. Reports up to 10.3% average improvement in post-trade implementation shortfall on South African equities.
- Relevance: 5/5 — clean baseline-augmentation framing; high citation count; demonstrates how to combine a parametric execution model with an RL overlay.

### Paper 1.10 — Deep Reinforcement Learning Approach for Trading Automation in the Stock Market
- Authors: Taylan Kabbani et al.
- Year: 2022
- Venue: IEEE Access
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/c2665977d6c55544afdb49ca900ac9c0/
- Citations: 74
- Abstract summary: Formulates the equity trading problem as a POMDP under liquidity and transaction-cost constraints, and solves it with TD3 (Twin Delayed Deep Deterministic Policy Gradient), reporting a 2.68 Sharpe ratio on held-out test data. Unifies price prediction and portfolio allocation in one autonomous decision-making loop.
- Relevance: 4/5 — adjacent to portfolio-allocation RL rather than pure execution, but the POMDP + TD3 + Sharpe-2.68 result is a useful upper-bound reference.

---

## Topic 2: Deep Q-learning market making

(Query "deep Q-learning market making" failed: Consensus monthly quota exhausted. Re-run after 2026-06-01 and populate this section. Expected anchor papers based on prior knowledge — to be confirmed via Consensus on next run — include: Spooner et al. 2018 "Market Making via Reinforcement Learning", Gasperov & Kostanjcar 2022 "Market making with signals through deep reinforcement learning", Sadighian 2019 "Deep Reinforcement Learning in Cryptocurrency Market Making", and Selser et al. 2021 "Optimal market making via deep reinforcement learning". Do NOT treat the above as Consensus-confirmed; they are operator hints for the re-run only.)

---

## Topic 3: PPO trading agent equity portfolio

(Query "PPO trading agent equity portfolio" failed: Consensus monthly quota exhausted. Re-run after 2026-06-01 and populate this section. Note that Paper 1.6 above [Byun et al. 2023, PPO+LSTM on KRX] is already a strong PPO data point — once Topic 3 is populated, cross-reference and avoid duplicate listings.)

---

## Topic 4: Model-free reinforcement learning in financial markets

(Query "model-free reinforcement learning financial markets" failed: Consensus monthly quota exhausted. Re-run after 2026-06-01 and populate this section. Note that Paper 1.10 above [Kabbani et al. 2022, TD3 POMDP] is a representative model-free policy-gradient result already in the file — cross-reference on re-run.)

---

## DOI list (this run)

Confirmed DOIs from Consensus:
- 10.1145/1143844.1143929 — Nevmyvaka et al. 2006 (Paper 1.2)
- 10.1109/CIFEr.2014.6924109 — Hendricks et al. 2014 (Paper 1.9)

DOIs not retrievable from Consensus payload for: 1.1, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.10. Operator can resolve these via the Consensus URL or by author + title lookup in Crossref / Google Scholar.

---

Upgrade message (verbatim from Consensus, per server instructions):

Upgrade to Consensus Pro to return 20 results per search instead of 10, and include more data like study design and key takeaways for every result.: https://consensus.app/pricing/?utm_source=claude_desktop
