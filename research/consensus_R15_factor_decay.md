# Consensus R15 — Factor Decay & Alpha Persistence After Publication
Date: 2026-05-12
Source: Consensus MCP (Semantic Scholar / PubMed / Scopus / ArXiv)
Status: PARTIAL — monthly search quota exhausted mid-run

---

## Quota Status

The Consensus MCP free monthly quota (30 searches) was exhausted during this
run. Limits reset on June 1, 2026.

Quota error returned by the tool on the final attempt:
> "You've used all 30 searches this month; limits reset on June 1st. Upgrade to
> Consensus Pro to get 1,000 searches per month, return 20 results per search
> instead of 10, and include more data like study design and key takeaways for
> every result. https://consensus.app/pricing/?utm_source=claude_desktop"

Two of the four planned queries returned results before the quota was hit:
- Query 1 ("factor decay alpha post-publication out of sample") — 10 results
  returned (top 4 are finance-relevant; the rest are nuclear-physics alpha-decay
  and psychometric Cronbach-alpha papers triggered by literal "alpha" keyword
  collisions).
- Query 2 ("anomaly disappearance crowding quantitative") — 10 results, but
  every result was computer-vision crowd-anomaly detection (the word "crowding"
  matched video surveillance literature, not finance). Relevance to factor
  decay is effectively zero; listed below only for transparency.
- Query 3 ("TSMOM time series momentum decay post-2015") — NOT executed,
  blocked by monthly quota.
- Query 4 ("momentum value factor underperformance") — NOT executed, blocked
  by monthly quota.

---

## Papers Collected

### Query 1 — "factor decay alpha post-publication out of sample"

#### [1] Not All Factors Crowd Equally: Modeling, Measuring, and Trading on Alpha Decay
- Authors: Chorok Lee
- Year: 2025
- Journal: Unknown (working paper)
- DOI: not provided by Consensus
- URL: https://consensus.app/papers/details/ec289d6694165af39fbeac15bf460bcf/?utm_source=claude_desktop
- Citations: 0
- Relevance: 5/5
- Summary: Derives a hyperbolic alpha-decay functional form alpha(t)=K/(1+lambda*t) from a game-theoretic crowding equilibrium and tests it against linear/exponential alternatives on eight Fama-French factors (1963-2024); finds mechanical factors (momentum, reversal) crowd hyperbolically while judgment-based factors (value, quality) do not. Reports that crowding accelerated post-2015 (correlates with factor-ETF AUM growth, rho = -0.63), the model over-predicts remaining alpha out-of-sample (0.30 vs realised 0.15), and crowded reversal factors show 1.7-1.8x higher crash probability — i.e. crowding predicts tail risk rather than mean alpha. Directly addresses why turbo-carnival's momentum-flavoured factors should be expected to decay and why post-2015 backtests on mechanical signals are optimistic.

#### [2] Why and How Systematic Strategies Decay
- Authors: A. Falck, A. Rej, D. Thesmar (et al.)
- Year: 2021
- Journal: Unknown (working paper)
- DOI: not provided by Consensus
- URL: https://consensus.app/papers/details/1b9cab4f5e2e563ab43e1dedeeeea958/?utm_source=claude_desktop
- Citations: 3
- Relevance: 5/5
- Summary: Tests McLean-Pontiff style ex-ante predictors of out-of-sample Sharpe decay across a large universe of published equity anomalies; publication year alone explains 30% of cross-sectional Sharpe-decay variance (newly-published factors lose 5 ppt of Sharpe per year of vintage). Two overfitting proxies — the number of operations in the signal definition and Sharpe sensitivity to outliers — add another 15% of explanatory power, while arbitrage-capital variables contribute marginally. Strong empirical backing for "complexity + recency = expected decay" and a check on how aggressive turbo-carnival should be in interpreting in-sample Sharpe.

#### [3] AlphaAgent: LLM-Driven Alpha Mining with Regularized Exploration to Counteract Alpha Decay
- Authors: Ziyi Tang et al.
- Year: 2025
- Journal: Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2
- DOI: not provided by Consensus
- URL: https://consensus.app/papers/details/365ce696d7a2588e8fac268aa2479a51/?utm_source=claude_desktop
- Citations: 13
- Relevance: 3/5
- Summary: Frames factor crowding/homogenisation as the primary driver of alpha decay in genetic-programming and LLM-generated factor mining, and proposes AST-based similarity regularisation plus complexity constraints to slow it down. Reports out-of-sample alpha persistence on CSI 500 and S&P 500 over four years versus baselines that decay faster. Relevant to turbo-carnival as a cautionary mechanism (complexity + similarity to known factors -> faster decay) more than as a method, since the project policy bars post-evidence code changes.

#### [4] Out-of-Sample Alphas Post-Publication
- Authors: Andrei S. Goncalves et al.
- Year: 2025
- Journal: SSRN Electronic Journal
- DOI: not provided by Consensus
- URL: https://consensus.app/papers/details/a7a10b34092050d9be4009d5992946c5/?utm_source=claude_desktop
- Citations: 0
- Relevance: 4/5
- Summary: Working paper on post-publication out-of-sample alphas; Consensus did not return abstract text so the methodology is not directly verifiable from this record. Title and venue strongly suggest a direct successor / replication of McLean-Pontiff (2016), making it a candidate primary reference for the post-publication decay literature once the abstract can be pulled.

The remaining 6 results from Query 1 are off-topic (nuclear alpha-particle
decay physics papers and Cronbach's-alpha psychometric meta-analyses matched on
the keyword "alpha"). They are not finance papers and are not listed
individually.

### Query 2 — "anomaly disappearance crowding quantitative"

Every returned paper was computer-vision crowd-anomaly detection (CCTV, deep
learning on Shanghai-Tech / UCSD-Ped / Avenue datasets), not financial-anomaly
decay. The query needs to be rephrased (e.g. "Schwert anomalies disappearance
after discovery" or "McLean Pontiff publication bias anomaly") and re-run after
the quota resets on June 1, 2026.

Not finance-relevant — listed only for audit traceability:

#### [V1] Anomaly Detection and Localization in Crowded Scenes
- Authors: Wei-Xin Li et al.
- Year: 2013
- Journal: IEEE Transactions on Pattern Analysis and Machine Intelligence
- DOI: not provided by Consensus
- URL: https://consensus.app/papers/details/b890c37c80bf5ea39271e99444744722/?utm_source=claude_desktop
- Citations: 1065
- Relevance: 1/5
- Summary: Computer-vision paper on detecting anomalous behaviour in crowded video scenes using joint spatial/temporal mixtures of dynamic textures. Not financial-anomaly research; included only because it was the top Consensus hit for the literal query string.

#### [V2] Recent trends in crowd analysis: A review
- Authors: Mounir Bendali-Braham et al.
- Year: 2021
- Journal: Unknown
- DOI: not provided by Consensus
- URL: https://consensus.app/papers/details/717a39b798265004a785fe3c85d3fa22/?utm_source=claude_desktop
- Citations: 71
- Relevance: 1/5
- Summary: Survey of crowd-statistics and crowd-behaviour analysis methods for urban surveillance. Off-topic for factor decay; flagged here so the query-substitution rationale is auditable.

The other 8 results from Query 2 are similarly off-topic (urban surveillance,
deep learning on pedestrian datasets) and are not listed.

### Query 3 — "TSMOM time series momentum decay post-2015"

NOT EXECUTED. Monthly Consensus quota was exhausted before this query could be
sent. Re-run after June 1, 2026.

### Query 4 — "momentum value factor underperformance"

NOT EXECUTED. Monthly Consensus quota was exhausted before this query could be
sent. Re-run after June 1, 2026.

---

## DOI List

No DOIs were returned by Consensus for any of the finance-relevant papers in
this run. Consensus URLs (per-paper detail pages) are listed in each entry
above and reproduced here for convenience:

1. https://consensus.app/papers/details/ec289d6694165af39fbeac15bf460bcf/?utm_source=claude_desktop  (Lee, 2025 — Not All Factors Crowd Equally)
2. https://consensus.app/papers/details/1b9cab4f5e2e563ab43e1dedeeeea958/?utm_source=claude_desktop  (Falck et al., 2021 — Why and How Systematic Strategies Decay)
3. https://consensus.app/papers/details/365ce696d7a2588e8fac268aa2479a51/?utm_source=claude_desktop  (Tang et al., 2025 — AlphaAgent)
4. https://consensus.app/papers/details/a7a10b34092050d9be4009d5992946c5/?utm_source=claude_desktop  (Goncalves et al., 2025 — Out-of-Sample Alphas Post-Publication)

---

## Next Steps

1. After June 1, 2026 (or after a Consensus Pro upgrade), re-run Query 3
   ("TSMOM time series momentum decay post-2015") and Query 4 ("momentum value
   factor underperformance").
2. Rephrase Query 2 to target the finance literature directly. Suggested
   replacements:
   - "McLean Pontiff publication bias anomaly out-of-sample"
   - "Schwert anomalies disappearance"
   - "factor crowding hedge funds capacity"
3. Resolve DOIs manually via Semantic Scholar / SSRN for the four
   finance-relevant papers above once internet access permits — Consensus did
   not surface DOIs in this run.
