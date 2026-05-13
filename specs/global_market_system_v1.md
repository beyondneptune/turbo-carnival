# Spec: Global Market Trading System Candidates (v1)

## Intent

Generate candidate designs for **one live account** that targets the **entire global market** — all asset classes (equities, options, futures, FX, crypto, bonds, commodities), all geographies (US, EU, UK, Asia, EM), all time zones (24/7 where venue allows).

The market is the planet, the asset class is whatever has edge, the technique is whatever survives validation. Where the edge lives is a research question, not an assumption.

Each iteration is **one self-contained design proposal** for one approach. Running N iterations in parallel produces N candidates we can compare on expected edge per unit of risk and operational cost — then pick the winner(s) for shadow deployment.

---

## Lessons that MUST be respected in every iteration

These came from one session of live paper trading. Each iteration must either honor each lesson or argue with evidence why this candidate is exempt.

1. **Validate before deploy.** No live capital before walk-forward out-of-sample + ≥30-day shadow run.
2. **Binary gates, not confidence rankers.** Any ML filter must be tested for ranking power separately from filter power before sizing positions by its confidence (verified: composite C3 saved $2,029 as a filter, but its confidence had r=−0.05 with realized P&L).
3. **TP:SL asymmetry is structural.** Same entry with 100/35 stops won +108 bps; with 40/20 stops lost −26 bps. Stop logic is part of the edge, not a hyperparameter.
4. **Broker truth, not local cache.** Reconcile every tick against broker positions; never trust a local quantity for closing orders.
5. **No PDT-exposed lanes on sub-$25k US equity accounts.** Use cash settlement, futures, FX, crypto, or maintain ≥$25k equity.
6. **One strategy per slot.** Hard-reject configs that duplicate an already-deployed strategy (the Hermes/Atlas and Aegis/TERMINATOR collisions cost real time).
7. **No same-data backtest.** Selection-bias on the training set is not validation. Out-of-sample, walk-forward, purged k-fold.
8. **Battle-tested broker SDKs only.** IBKR (insync), ccxt, oanda-py, dxFeed — not hand-rolled HTTP.
9. **Capacity bound named upfront.** State the capital level at which the edge degrades and the dollar volume the venue can absorb without slippage blow-up.
10. **Kill criteria written before deploy.** Exact thresholds that trigger shutdown, not vibes.

---

## Per-iteration deliverable

Each iteration is **one markdown file** named: `iter_<N>_<short_name>.md` (e.g. `iter_3_jp_futures_overnight_rev.md`).

The file MUST contain every section below. Missing any section = invalid iteration.

### 1. Thesis (one sentence)
What edge is being captured, stated as a falsifiable claim.

### 2. Market / asset class / venue
- Specific instrument(s) and ticker symbols
- Primary exchange and broker
- Settlement, margin, and tax regime

### 3. Time horizon
Tick · sub-second · 1-min · 5-min · hourly · daily · multi-day. Pick one.

### 4. Method family and specific algorithm
- Rules / ML / NN / LLM / hybrid
- Name the algorithm (e.g. "LightGBM classifier on 23 microstructure features", "transformer next-bar forecast", "rules-based contango threshold")
- Why this method fits this edge (not "because ML is cool")

### 5. Data requirements
- Feeds needed (e.g. Polygon S3 minute bars, ccxt L2 orderbook, Bloomberg news)
- Storage estimate (GB/year)
- Cost estimate ($/month)
- Latency requirement (round-trip ms tolerable)

### 6. Validation plan
- Walk-forward design: train window / test window / step
- Purge and embargo (for serial-correlated data)
- Minimum number of out-of-sample trades before a verdict
- Specific statistical test and threshold (e.g. "Sharpe > 1.0 with 95% bootstrap CI excluding 0.5")

### 7. Risk envelope
- Max position size in dollars and as % of account
- Daily loss cap (absolute and %)
- Drawdown trigger that halts trading
- Circuit breaker rule (e.g. "halt for the day after 3 consecutive losses")
- Correlation rule with any already-deployed strategy

### 8. Execution plan
- Broker / SDK
- Order type (marketable-limit IOC, post-only, etc.)
- Expected slippage budget in bps and how it's measured
- What happens if the broker rejects an order

### 9. Capacity
- AUM ceiling before edge degrades (with reasoning)
- ADV of underlying and what % the strategy consumes at full size
- Concentration risk (single-name max, single-sector max)

### 10. Failure modes
At least 3 named failure modes and what each looks like in metrics. Examples: regime change, liquidity flight, spread widening, news shock, exchange outage.

### 11. Falsifiable verdict criteria
Exact numbers that decide:
- **Promote to live capital** (e.g. "after 30 shadow days: Sharpe > 1.2, max DD < 8%, no broker-truth divergence")
- **Continue shadow** (e.g. "metrics ambiguous in shadow but no kill triggers hit")
- **Kill immediately** (e.g. "any one of: 3-day DD > 5%, daily loss cap breach, broker-side fill discrepancy > 0.5%")

### 12. One-paragraph operational risk note
What can go wrong operationally (not market-wise) — e.g. "if Polygon S3 plan downgrades, we lose 5 years of historical bars". Honesty about ops surface.

---

## Innovation dimension matrix

Each sub-agent picks a distinct cell so candidates don't collide:

| Dimension | Examples |
|---|---|
| **Asset class** | US equities, US options, ES futures, JP futures (TOPIX), EU futures (DAX), spot FX (EUR/USD majors), crypto perpetuals (Binance/Bybit), spot crypto, treasuries, commodity futures |
| **Geography** | US, EU, UK, JP, HK, India, multi-region cross-listing |
| **Time scale** | Sub-second (only with colo), 1-min HFT, 5-min, hourly, daily, multi-day swing |
| **Method** | Order-flow rules, mean-reversion, breakout/momentum, stat-arb pairs, contango/term-structure rotation, news-driven, LLM-narrative classification, transformer next-period forecast, regime detection |
| **Edge source** | Microstructure imbalance, regime persistence, cross-asset lead-lag, scheduled-event drift, cross-section ranking, carry/roll, mean reversion |

---

## Hard rules (any violation = iteration discarded)

- Must be implementable with an off-the-shelf broker SDK.
- Must name a falsification criterion that would make us shut down within 90 days of going live.
- Must not claim edge without naming the specific out-of-sample test that produced the claim.
- Must address every numbered lesson above explicitly.
- Must not duplicate the thesis or method of an earlier iteration in this run.

---

## Evolution pattern across iterations

- **Iter 1-5**: Conservative — well-understood asset classes (US equities daily, large-cap futures intraday, top crypto pairs swing), well-understood methods.
- **Iter 6-10**: Add cross-asset and cross-geography. Bring in Asia and EU venues. Introduce stat-arb and pairs.
- **Iter 11-15**: Speculative method families — LLM narrative classification, transformer forecasting, regime-conditioned ensembles.
- **Iter 16+**: Combinatorial — pair a candidate with a hedge from a different geography, or a hybrid rules+LLM voting scheme.

Each wave should push one dimension further than the previous wave.

---

## What this spec is NOT

- Not a request to write trading code. Iterations are **design proposals**, not implementations.
- Not a backtest. The validation plan describes how the backtest would be run if this candidate is chosen.
- Not a paper trade. Shadow deploy comes after the spec-level review.
- Not a place to handwave with phrases like "uses AI to predict the market". Specifics or the iteration is invalid.
