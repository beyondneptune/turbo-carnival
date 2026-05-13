# Iteration 21: Multi-LLM Debate Consensus — Three-Vendor Agreement Gate on SPY + QQQ + IWM

## 0. Differentiation from iter_11 and iter_20 (read this first)

This iteration is the third LLM-family candidate in the run. To avoid the lesson-6 collision, it occupies a distinct slot from both predecessors. The disjointness is in the *mechanism of edge*, not just the calendar cadence.

| Axis | Iter 11 (FOMC LLM) | Iter 20 (Thematic LLM) | Iter 21 (Multi-LLM Consensus, this iteration) |
|---|---|---|---|
| Edge anchor | Lucca/Moench post-FOMC drift | Narrative momentum on weekly text | **Inter-model agreement as a Bayes-rule confidence amplifier on heterogeneous training corpora** |
| Number of LLMs in the live decision loop | 1 primary (Claude) + 1 fallback (Llama, only on failure) | 1 primary (Claude) + 1 fallback (Llama, only on failure) | **3 in parallel always — Claude, GPT-4-class, Llama-3-70B — all three votes count every day** |
| Decision rule | Single-classifier `confidence ≥ 0.80` | Cross-sectional ranking of a 20-theme universe | **AND-gate across three independent vendors: trade only if direction agreement is unanimous AND mean confidence > 0.7** |
| Cadence | 16-24 macro events/year | 12 monthly rebalances/year | **250 daily evaluations/year, expected ≤ 25 trades/year after the consensus gate** |
| Asset universe | SPY + TLT | 20 thematic ETFs cross-section | **SPY + QQQ + IWM — three broad US equity factor exposures** |
| Innovation | LLM extracts policy-language nuance faster than discretionary desks | LLM ingests more text than a human PM in a week | **Orthogonal-blindspot reduction via independent vendor ensemble — variance reduction in the directional signal** |
| Position-sizing input | Binary trade/no-trade × fixed 25%/15% NAV | Equal-weight basket of top-5 | **`mean_confidence × (1 / disagreement_penalty)` — sized continuously but only inside the binary unanimity gate** |
| The thing being tested | Does an LLM read Fed text better than keywords? | Does narrative attention persist? | **Does three-way LLM agreement correlate with realized P&L, or are the three models just predicting the same well-known consensus signals?** |

**The innovation here is the agreement requirement as a Bayesian filter, not any single LLM's directional read.** If three LLMs trained on substantially overlapping internet corpora *still* converge from independent prompt sessions, the posterior probability that the signal is real (rather than each model's idiosyncratic hallucination) is materially higher than any single-model gate of equivalent confidence. The lesson-2 hazard — "is the confidence a real ranker or just a filter?" — is sharper here than in iter_11 or iter_20, because we have **two** confidence-like quantities (the agreement gate, binary; and the mean confidence, continuous) and the spec demands we test them separately.

The strategy is co-deployable with iter_11 and iter_20 only after a correlation check on traded days (§7).

---

## 1. Thesis (one sentence)

When three independent LLMs from three vendors (Anthropic Claude, OpenAI GPT-4-class, and a self-hosted Meta Llama-3-70B-Instruct) each receive the same daily snapshot (5-day SPY/QQQ/IWM price action, aggregated overnight news headlines, options skew snapshot, VIX level) at 16:30 ET and independently output a structured `{direction, confidence, rationale}` decision, the conjunction of (i) **unanimous direction agreement** and (ii) **mean confidence > 0.7** is a stronger predictor of next-day SPY/QQQ/IWM open-to-close return sign than any single LLM's `confidence ≥ 0.7` filter — because three vendors with at-least-partially-disjoint training corpora and architectures are unlikely to share *all* of the same blindspots, so their unanimous votes screen out single-model hallucinations and idiosyncratic prompt artifacts.

**Falsifiable form:** *Over the 2023-01 → 2026-05 out-of-sample window (~825 trading days, of which we expect ~80-120 to clear the consensus gate based on 2018-2022 prompt-engineering base rates), the unanimous-and-high-mean-confidence subset of days generates a Sharpe ≥ 0.8 with 95% bootstrap CI excluding 0 on a per-trade basis, AND the Spearman correlation between mean LLM confidence and realized next-day P&L is positive with p < 0.05 across the OOS sample. If unanimity correlates with P&L but mean-confidence does not rank within the unanimity subset, then the gate is real but the sizing rule is noise — drop the sizing component and trade unit positions. If unanimity does not correlate with P&L either, the three models are just predicting the same well-known headlines (no orthogonal blindspots) — kill.*

---

## 2. Market / asset class / venue

- **Equity legs (three broad US equity factor exposures, daily decision per leg)**:
  - `SPY` (SPDR S&P 500 ETF) — NYSE Arca, ~$500B AUM, ~80M shares/day ADV, median spread ~0.2 bps. T+1 cash settlement.
  - `QQQ` (Invesco QQQ Trust, NASDAQ-100) — NASDAQ, ~$300B AUM, ~50M shares/day ADV, median spread ~0.2 bps. T+1.
  - `IWM` (iShares Russell 2000 ETF) — NYSE Arca, ~$70B AUM, ~30M shares/day ADV, median spread ~1 bp. T+1.
- **Why these three and not just SPY**: the three ETFs span the large-cap blend (SPY), large-cap growth/tech (QQQ), and small-cap (IWM) cross-section. The LLMs are asked for a directional opinion on the *broad US equity tape* — the three tickers let us A/B test whether the consensus is generic "risk on / risk off" or has theme-discriminating content. If all three LLMs say "long" we long all three equally; if they collectively say "long large-cap, short small-cap" via the per-ticker rationale field, we can take a paired position. **Primary spec is the three-ticker basket; per-ticker dispersion is a logged research artifact, not a tradable rule in v1.**
- **Primary exchange**: NYSE Arca (SPY, IWM), NASDAQ (QQQ).
- **Broker**: Interactive Brokers (IBKR Pro) via `ib_insync` (lesson 8). IBKR Reg-T US equity account, minimum $25k equity for short legs to avoid PDT exposure (lesson 5; we trade once per day and hold overnight, so PDT does not apply to long positions, but short legs require Reg-T short capability which IBKR provides on a $25k+ account).
- **Settlement / margin**: T+1 for all three ETFs. Long-only baseline. Short legs (if `direction = short`) use Reg-T short on the ETF; SPY and QQQ are HTB-easy with ~0 bps borrow; IWM small-cap basket is typically 0-25 bps borrow → tracked separately as a slippage adjustment.
- **Tax regime**: §1221 short-term cap gains for ≤ 1d holds. No §1256 advantage since we use ETFs not futures. (A v2 upgrade could move to ES/NQ/RTY futures for tax efficiency; out of scope for v1 because we want the simplest possible execution layer for the falsification phase.)
- **LLM venues (three independent vendors, the heart of this iteration)**:
  - **LLM-A: Anthropic API**, snapshot `claude-sonnet-4-5-20251022` (and `claude-opus-4-7-20260201` reserved for ambiguity escalation, not the daily path).
  - **LLM-B: OpenAI API**, snapshot `gpt-4.1-2025-04-14` (or the most recent pin-able GPT-4-class snapshot; we explicitly avoid GPT-4o-mini and GPT-3.5 to keep the three vendors at comparable capability levels).
  - **LLM-C: Local Llama-3-70B-Instruct** served via vLLM on a co-located A100 or rented runpod GPU. Self-hosted so the vendor cannot silently update the weights on us — see §12 ops note.
  - **Critical pinning rule**: each of the three vendor snapshot strings is part of the strategy artifact, written into a `models.lock` file alongside the prompt hash. A vendor de-pinning is a strategy change and triggers the lesson-6 re-validation cycle. (See §12 for the messy practicalities.)

## 3. Time horizon

**Daily. One evaluation per US trading day at 16:30 ET (30 minutes after equity close), one trade decision per day. Next-day open execution. Hold to close of next day.**

- **Decision tick**: 16:30 ET — 30 min after equity close, after the official 4:00 ET close has settled and the consolidated tape is final.
- **Order placement**: Next-day market-on-open (MOO) order sent to IBKR at 09:25 ET, fills at 09:30:00 NYSE auction print.
- **Flatten**: Same-day market-on-close (MOC) order at 15:55 ET, fills at 16:00:00 NYSE auction print.
- **Hold period**: ~6.5 hours of US equity session — open-to-close of the trade day (which is the day **after** the decision day). This avoids overnight gap risk because the position is opened at the open, not held through the overnight window.
- **Why open-to-close and not close-to-close**: opening-auction liquidity and pricing are good for SPY/QQQ/IWM (50-150 bps of ADV print in the opening auction alone); the LLM evaluates the previous-day-and-overnight news, then we capture the day's drift. Close-to-close was considered but adds an overnight gap risk that confounds the signal (we'd be paid for the LLM's read of the prior day overlaid on the market's overnight processing of post-16:30 news the LLM did not see).

## 4. Method family and specific algorithm

- **Family**: LLM directional classifier × 3 (ensemble), with a binary unanimity gate over the three direction votes and a confidence-based sizing rule inside the gate (lesson 2 hazard explicitly tested — see §6).
- **Algorithm**:

  **STEP A — Daily context assembly (every US trading day, 16:00 - 16:30 ET)**:
  1. **Price action snapshot, last 5 trading days**: for each of SPY, QQQ, IWM, pull OHLC for the past 5 trading days, today's close, today's intraday range, today's volume vs. 20d ADV. Format as a tight tabular text (~500 tokens total across three tickers).
  2. **Aggregated overnight + intraday news headlines**: pull headlines from a single primary vendor (Benzinga, Reuters, or RavenPack) over the past 24 hours, filtered to tickers in S&P 500 + Nasdaq-100 + Russell 2000 universe AND to macro tags (Fed, CPI, NFP, geopolitics, China). Cap at 60 headlines. Each headline ≤ 25 tokens. Total: ~1,500 tokens.
  3. **Options skew snapshot**: for each of SPY, QQQ, IWM, pull the 30-day at-the-money implied vol, the 25-delta put-vs-25-delta-call skew, and today's net put/call volume ratio. From CBOE DataShop or IBKR option-chain API. Format as 3 rows × 4 numbers (~200 tokens).
  4. **VIX level**: today's VIX close, VIX9D close, and the 5-day VIX trend (+/- absolute points). ~30 tokens.
  5. **Total daily context: ~2,200 input tokens.** Plus a static system message (cached on Anthropic and OpenAI APIs) of ~1,500 tokens defining the task, the rubric, the structured-output schema, and 3-5 dated few-shot examples from the prompt-engineering window only (2015-2017, well outside the OOS).

  **STEP B — Three parallel, independent LLM calls (between 16:30 and 16:35 ET)**:

  The SAME assembled context (Step A bytes-identical) is dispatched in parallel to the three vendors. Each vendor receives:
  - The static system message (cached where possible).
  - The day's context block.
  - The required output schema (Anthropic tool-call, OpenAI function-call, Llama JSON-mode):
    ```json
    {
      "decision_date": "YYYY-MM-DD",
      "direction": "long" | "short" | "flat",
      "confidence": 0.0,
      "rationale": "<2-4 sentence justification grounded in the input>",
      "primary_signals": ["<bullet 1>", "<bullet 2>"]
    }
    ```
  - Temperature: **0** on all three vendors (per-call determinism — see §6 on testing).
  - **Critical isolation rule**: NO LLM is told that the other two LLMs exist, what they were asked, or what they answered. No round-robin, no "debate" in the sense of multi-turn cross-talk. The "debate" framing in the candidate name refers to *post-hoc* aggregation by the trading agent, not in-context multi-agent dialogue. Each LLM is a single-shot independent classifier.

  Why no in-context multi-agent dialogue: it would create a coordination point that re-correlates the three votes via shared in-context text, defeating the orthogonal-blindspot premise. If we want orthogonality, we have to enforce statistical independence at the input boundary.

  **STEP C — Aggregation (16:35 - 16:36 ET)**:
  Once all three JSON responses are parsed and schema-validated:
  - Let `d_A, d_B, d_C` be the three directions ∈ {long, short, flat}.
  - Let `c_A, c_B, c_C` be the three confidences ∈ [0, 1].
  - Compute `direction_consensus = (d_A == d_B == d_C) AND (d_A ∈ {long, short})`. If any one of the three is `flat`, or if any pair disagrees on long-vs-short, `direction_consensus = False`.
  - Compute `mean_confidence = (c_A + c_B + c_C) / 3`.
  - Compute `disagreement_penalty = 1 + std_dev(c_A, c_B, c_C)`. (So a tight cluster of confidences gets a near-1 penalty, a spread cluster gets a >1 penalty that shrinks position size.)

  **STEP D — Trade gate (16:36 ET, lesson 2 — binary gate, confidence is sizing within the gate)**:
  - Trade ONLY if `direction_consensus == True` AND `mean_confidence > 0.7`. Otherwise: **abstain** for the next day.
  - Inside the gate: `target_NAV_fraction = mean_confidence × (1 / disagreement_penalty)`.
  - **Cap**: target NAV fraction is clipped to `[0.05, 0.15]`. At full-confidence unanimity (mean confidence = 1.0, std = 0) the position is 15% NAV. At the gate floor (mean = 0.7, std = 0.10) it is `0.7 / 1.1 = 0.636` → clipped to 0.15 if you let it run, but more importantly, in practice the floor case sizes to `0.7 / 1.1 ≈ 0.636` of cap = ~10% NAV. Cap is hit roughly when mean confidence ≥ 0.85 with tight std.
  - Split across the three tickers: equal-weight across SPY, QQQ, IWM (so a "long" decision at 15% NAV is 5% SPY + 5% QQQ + 5% IWM long).
  - Order type: **market-on-open (MOO)** at next-day 09:30 ET. Flatten at **market-on-close (MOC)** at 16:00 ET same day.

  **STEP E — Logging (continuous, append-only)**:
  Every day, regardless of whether we traded, log: (i) the three raw LLM JSON outputs with the model snapshot string, (ii) the assembled input context (hashed + retained for 90 days), (iii) the consensus / mean confidence values, (iv) the trade decision, and (v) the realized next-day open-to-close return per ticker. This log is the substrate for the lesson-2 re-validation (filter-vs-ranker) and the §6 Spearman test.

- **Why this method fits this edge**:
  1. **Disagreement-among-experts → variance reduction in the signal**: This is a textbook ensemble idea — Hansen 2007, Timmermann 2006, and the general bagging/Bayesian-model-averaging literature show that aggregation of independent (or even partially independent) classifiers shrinks the conditional error variance. The interesting empirical claim here is **how independent are the three LLMs in practice?** That is precisely what the §6 test measures.
  2. **Orthogonal-blindspot premise**: Claude (Anthropic constitutional training), GPT-4 (OpenAI RLHF), and Llama-3 (Meta supervised + open RLHF) were trained on overlapping-but-not-identical text corpora with different fine-tuning objectives. They are statistically correlated (all are pretrained on roughly the public internet + books + arxiv), but not collinear. A claim that survives all three independent vendor prompts has shed at least the *idiosyncratic-to-one-vendor* noise component.
  3. **The vast-majority abstain is the feature, not the bug**: A daily classifier that fires once a day on noisy financial text is going to look enthusiastic about something most days. The agreement gate is a discipline mechanism — it converts "always have an opinion" into "have an opinion only when the signal is loud enough that three independent readers all hear it the same way." We expect ~10-15% of trading days to pass the gate (calibrated below in §6); the other 85-90% of days the strategy holds cash. This is a feature because it concentrates capital in the high-conviction tail and avoids the failure mode of "the LLM has an opinion on everything every day, the average opinion is noise."

- **Why an LLM ensemble, not a smaller-and-faster classifier ensemble**:
  We could ensemble three transformer classifiers fine-tuned on financial text. We don't, because: (i) the assignment is to test the multi-LLM-vendor agreement hypothesis specifically; (ii) the relevant comparison is to single-LLM strategies (iter_11, iter_20), so the experimental control is "same family, different aggregation rule"; (iii) at 1-3 calls/vendor/day the cost is trivial (see §5) and we're not bottlenecked on inference. A smaller model ensemble is in the v2 backlog as a baseline comparator.

## 5. Data requirements

- **Live feeds**:
  - Polygon (or IBKR market data subscription) for OHLC + 5-day price action of SPY, QQQ, IWM. Already subscribed for other iterations.
  - Benzinga or Reuters or RavenPack news feed, filtered to S&P 500 + Nasdaq-100 + Russell 2000 + macro tags. ~$300-500/month for Benzinga, $1500+/month for RavenPack. **Spec default: Benzinga at $400/month.**
  - CBOE DataShop or IBKR option-chain for SPY/QQQ/IWM IV and skew. IBKR is sufficient; ~free with active account.
  - VIX from CBOE direct (free) or via IBKR.
- **Historical data**:
  - Polygon 1-min bars for SPY/QQQ/IWM 2015-01 → present (~5 GB, already locally cached).
  - Historical news headline archive 2015-01 → present from Benzinga (one-time $2k purchase) or Reuters archive ($3k). **Required for prompt engineering on 2015-2017 and validation on 2018-2022.**
  - Historical VIX, VIX9D, SPY/QQQ/IWM ATM IV and 25-delta skew 2015 → present from CBOE DataShop archive (~$1.5k one-time).
- **LLM API costs**:
  - **Per-day cost across three vendors**:
    - Claude Sonnet 4.5: ~3.5k input tokens (including cached system), ~300 output tokens. With Anthropic prompt-caching the marginal cost is ~$0.01-0.02/day.
    - GPT-4.1 (or comparable): ~3.5k input, ~300 output. ~$0.03-0.05/day at GPT-4.1 list pricing.
    - Llama-3-70B local: $0 marginal if on owned A100, ~$0.10/day on a rented runpod GPU.
    - **Total per day: ~$0.05-0.15** (well under the assignment's ~$0.10/call × 3 = $0.30 estimate, because caching cuts the input side).
  - **Per year**: 250 trading days × $0.15 = **~$40/year LLM cost**. The assignment's $75/year is the right order of magnitude; even doubled or tripled, this is trivial.
- **Storage**: < 10 GB including news archive, options data, and 1-min price bars.
- **One-time setup costs**: ~$3-5k for news + options historical archives. Amortizable across iter_11, iter_20, and iter_21.
- **Latency requirement**: 5 minutes from 16:30 ET to consensus + log. Each LLM round-trip is 2-15 seconds (Llama-70B local is the slowest at 10-15s). In parallel, all three complete within 20 seconds. The 5-minute buffer is for safety on retries / fallbacks. **Not a low-latency strategy** — this is a daily reflex, not a tick reflex.

## 6. Validation plan

This is the section that decides the iteration. Five distinct validation tasks:

### 6.1 Data hygiene plan — the LLMs cannot have seen the OOS news during prompt engineering

This is the *single most important* lesson-7 hazard in this iteration. We cannot prevent the three vendors from having pretrained on news articles dated 2018-2026 (they did — that text is on the public internet). But we **can** prevent the *researcher* from leaking OOS information into the prompt during prompt engineering, and we can structure the validation so that the LLM does not face the OOS context with a researcher-tweaked prompt.

- **Prompt-engineering window (researcher iterates freely, both human and LLM see the data many times)**: **2015-01 → 2017-12** (~750 trading days). Includes the 2015 China devaluation, the Brexit vote, Trump election. Rich variety, fully outside the OOS.
- **Few-shot exemplars in the prompt come ONLY from 2015-2017** and are explicitly dated in the prompt text so we never accidentally back-port a 2022 example into the few-shot block.
- **Walk-forward / calibration window (prompt is FROZEN; LLM runs once per day; outputs are paired with realized returns to set the confidence threshold)**: **2018-01 → 2022-12** (~1,250 trading days). The 0.7 confidence floor, the 5-15% NAV cap, and the disagreement penalty formula are calibrated on this window. **One pass over the window per parameter set; no peeking at 2023+.**
- **Out-of-sample window (verdict-bearing)**: **2023-01 → 2026-05** (~825 trading days). The frozen prompt and frozen aggregation rule are run once per day. We see the result once.
- **The unavoidable contamination, named honestly**: Claude Sonnet 4.5 was trained with a cutoff that includes 2023-2025 news; GPT-4.1 likewise; Llama-3-70B has a 2023 cutoff. So when we ask any of the three LLMs in 2026 to score a 2024-04-15 context, the model is implicitly capable of "recognizing" the period if the context strongly resembles its pretraining recollection. **We cannot eliminate this — it is the central LLM-OOS hazard.** We *can* mitigate it by:
  1. **Stripping dates from the context** before sending to the LLMs (the context says "5 days ago: SPY +1.2%, ...", never "On 2024-04-10 ..."). This is a binding rule in the prompt.
  2. **Anonymizing news headlines minimally**: keep ticker symbols (the LLM needs them) but strip the dateline and the publication date. Headlines that *embed* a date (rare) are passed through; we accept this as residual contamination.
  3. **Compare consensus performance to a "memorization control"**: run a separate prompt that asks each LLM directly "do you remember any specific event near 5 days after this context?" — if the LLMs can date-stamp the context with > 30% accuracy from anonymized inputs, we have a deep contamination problem and the OOS is corrupted. Run this control on a random 5% of OOS days.
  4. **Test whether the consensus gate's performance degrades as OOS dates approach the LLMs' pretraining cutoff dates**. If the strategy works on 2023 (well within all three cutoffs) and fails on 2026 (post-cutoff for two of the three), that's a memorization signature; if it works equally well across the OOS window, the edge is more likely real.

### 6.2 The lesson-2 critical test — agreement gate vs. confidence as ranker

**Two distinct quantities** to test separately:

- **Binary gate test**: Within the OOS window, partition days into {gate passes (~80-120 days)} vs. {gate fails (~700-740 days)}. Compute the trade-set's per-trade hit rate and mean return; compare to the gate-fails set (which would be hypothetically traded at the same NAV size). The gate must produce a **mean per-trade return ≥ 15 bps above the gate-fails mean return**, with 95% bootstrap CI excluding 0. This tests "is unanimity-plus-mean>0.7 a real filter?"
- **Continuous-confidence-as-ranker test**: Within the gate-passes set (~80-120 trades), regress realized next-day open-to-close return on `mean_confidence` and on `1 / disagreement_penalty` separately. Report **Spearman rank correlation** for each. Require the Spearman to be **positive with p < 0.05** for `mean_confidence`. If Spearman is ~0 or negative (the case for composite C3 in the user's earlier lesson), then **sizing on confidence is noise within the gate** and we collapse the rule to "unit position size when gate fires, zero otherwise."
- This explicitly separates "filter power" from "ranking power" per lesson 2.

### 6.3 The lesson-7 walk-forward inside the calibration window

Within 2018-2022, walk-forward in quarterly slices. Train: rolling 4-quarter window. Test: next quarter. Step: 1 quarter. **Re-calibrate only the confidence threshold and NAV cap per fold; the prompt is frozen.** ~20 folds. Report Sharpe stability across folds — std of fold Sharpe should be < 0.6 to support that the rule generalizes.

### 6.4 Statistical tests on the OOS

- Bootstrap (10k draws, block size = 5 days to handle weak serial correlation) of per-trade returns; require **95% CI of mean per-trade return excludes 0** AND **Sharpe ≥ 0.8 with 95% CI excluding 0.3**.
- **Hit rate on next-day open-to-close return sign vs. consensus direction**: ≥ 55% on ≥ 60 traded events, binomial test p < 0.05 against 50%.
- **Spearman of `mean_confidence` vs. realized P&L** across the gate-passes set: ρ > 0.10 with p < 0.05. (If ρ < 0.10 we drop the continuous sizing — see §6.2.)
- **Multi-LLM-vs-best-single test**: simulate the same gate using only LLM-A (Claude). If single-LLM "confidence > 0.7" performs within 0.2 Sharpe of the consensus gate, the multi-LLM apparatus adds insufficient value and we revert to single-LLM (lesson 6 — collapse into iter_11-family if no marginal edge).

### 6.5 Per-LLM agreement statistics (the orthogonal-blindspot test)

- **Conditional pairwise direction agreement**: across all OOS days, compute `P(d_A == d_B)`, `P(d_A == d_C)`, `P(d_B == d_C)`. If all three pairwise agreement probabilities are > 0.85 on long/short calls, the models are not adding orthogonal information; the "consensus" is largely tautological. Require **at least one pair below 0.80** to support the orthogonal-blindspot claim.
- **Unanimous-vs-majority comparison**: also log days where 2-of-3 agree (majority but not unanimous). Compare per-trade return of majority-rule (mean confidence > 0.7 among the two agreers) to unanimous-rule. If majority-rule has materially worse P&L, that's evidence the third dissenting model carries real information.

### 6.6 Minimum OOS trade count for verdict

We need **≥ 60 gate-passing trades in the 2023-2026 OOS window** to call a verdict. Base rate from 2018-2022 calibration suggests 10-15% of days clear the gate → 80-120 in the OOS → safely above 60. If the OOS gate-pass rate drops below 60 trades over the full OOS, that itself is a finding (the gate may be too restrictive in the post-2022 regime — and we'd need to extend the OOS or weaken the gate, with explicit documentation).

## 7. Risk envelope

- **Max position size per trade**: **15% NAV gross** across the three-ticker basket (5% SPY + 5% QQQ + 5% IWM at full conviction). The 15% cap is the upper end of the assignment's 5-15% range.
- **Min position size per trade**: **5% NAV gross** (the floor of the sizing rule; below this we'd be paying transaction costs for no gain).
- **Daily loss cap (intraday MTM during the open-to-close hold)**: -1.5% NAV on the combined position → flatten via 09:31-15:55 ET market orders, do NOT wait for the MOC. Accept slippage. -2.5% NAV intraday → also flatten and **halt strategy for 5 trading days** pending human review.
- **Per-trade total loss cap (close-out P&L)**: -1% NAV on the closed trade → halt strategy for 5 trading days pending review.
- **Drawdown trigger**: trailing 30-trade rolling drawdown ≥ -3.5% NAV → strategy halted, human review required. At expected 80-120 trades/year, 30 trades ≈ 3-4.5 months — meaningful timeframe.
- **Circuit breaker — model coordination failure**: If on any day **any one of the three LLMs fails to return a schema-valid response after 2 retries** (timeout, hallucinated JSON, schema violation), abstain for that day. Do NOT substitute another LLM into the third slot — that breaks the pinning. (Three's the rule; if you can't get three valid votes, you have no consensus.)
- **Circuit breaker — all three LLMs down**: If all three vendors fail (e.g., simultaneous outage of Anthropic + OpenAI + local GPU rack), abstain. This is treated as the **safe default** rather than catastrophic, because the strategy default is to hold cash; the worst case of all-three-down is the same as the modal outcome of any normal day.
- **Circuit breaker — agreement drift**: If over a rolling 30-day window the pairwise agreement probability `P(d_A == d_B)` etc. shifts by > 0.15 from the calibration baseline (e.g., one vendor's behavior changes dramatically — silent model update being the prime suspect), **halt strategy until human review confirms model identity**. This is the lesson-6/§12 ops hazard surfaced as a hard rule.
- **VIX > 35 hard cut**: do not trade when VIX > 35 at decision time (the LLMs' prompt-engineering window did not include a sustained crisis regime above 35 — out of distribution).
- **Correlation rule with other deployed strategies**:
  - **Not co-deployable with any other daily US large-cap directional strategy** with > 0.4 30-day return correlation. Specifically:
    - Iter_11 (FOMC LLM SPY/TLT): traded days might overlap on FOMC announcement days. **Suppress this iteration's trade on FOMC announcement days** (16 days/year) to avoid double-up risk. Iter_11 takes priority on those days.
    - Iter_20 (Thematic LLM ETF rotation): monthly rebalance day overlap is at most one decision per month; correlation expected < 0.3 in backtest but must be re-verified pre-deploy.
    - Iter_1 (US smallcap overnight mean-reversion): trades IWM-like names → potential collision. **Require pre-deploy correlation < 0.3 on IWM leg.**

## 8. Execution plan

- **Broker / SDK**: IBKR Pro, `ib_insync` (lesson 8).
- **Order type for entry**: **Market-on-open (MOO)** at 09:30 ET, child orders one per ticker. MOO is appropriate because (i) SPY, QQQ, IWM all have substantial opening-auction liquidity (50-150 bps of ADV prints in the open), (ii) we have a directional view, not a market-making one, (iii) MOO fills are deterministic and slippage-bounded.
- **Order type for exit**: **Market-on-close (MOC)** at 15:55:00 ET cutoff → fills at 16:00:00 NYSE auction print. Same liquidity argument as MOO.
- **Expected slippage budget**: 1.0 bps per side on SPY/QQQ (large-cap, deep auction), 1.5 bps on IWM. Round-trip per ticker: ~2.5-3.0 bps. At 15% NAV gross across three tickers, total drag per trade ≈ 0.45-0.50 bps of NAV. Measured against the auction-print mid as benchmark; we record the realized slippage every trade and compare to the budget over a rolling 20-trade window.
- **What happens if the broker rejects an order**: IBKR rejects MOO orders rarely (mostly for halted stocks, but SPY/QQQ/IWM essentially never halt). If a single-ticker MOO is rejected, the *other two* tickers still trade — the basket allocation per-ticker is treated as independent leg orders. We log the rejection, flatten the partially-filled basket at 09:35 ET, and abstain from re-entry that day.
- **Broker truth reconciliation (lesson 4)**: at 09:35 ET (5 min after market open) and at 16:01 ET (1 min after market close), reconcile IBKR's `reqPositions()` callback against the strategy's expected position. Mismatch ≥ 1 share on any leg → page operator immediately AND flatten via marketable-limit orders.

## 9. Capacity

- **AUM ceiling before edge degrades**: ~$500M.
  Reasoning: 15% NAV at $500M is $75M deployed across three tickers, i.e., $25M per ticker. Against SPY ADV of ~$50B/day, $25M is 0.05% of ADV — invisible. Against IWM ADV of ~$5B/day, $25M is 0.5% of ADV — meaningful but not market-impacting in the opening or closing auction (which alone prints $1-2B on a normal day for IWM). The MOO/MOC auction routine on SPY/QQQ is essentially infinitely scalable up to several percent of auction volume; the bind is IWM where the auction is thinner. Above ~$1B AUM, IWM slippage would degrade the trade economics meaningfully → strategy capped around $500M-$1B.
- **ADV of underlying / strategy's full-size consumption**:
  - SPY: ~80M shares × $520 ≈ $40B/day ADV. At $500M AUM, 5% NAV × 3-ticker split = ~$25M per ticker = 0.06% of SPY ADV. Negligible.
  - QQQ: ~50M shares × $480 ≈ $24B/day ADV. ~0.10% at $500M AUM. Negligible.
  - IWM: ~30M shares × $230 ≈ $7B/day ADV. ~0.36% at $500M AUM. Borderline noticeable in a thin morning, fine in the open auction.
- **Concentration risk**: Single-ticker max = 5% NAV per trade. Three tickers per trade = 15% NAV gross. Single-sector max is implicit — SPY/QQQ/IWM are diversified broad indices; no single sector exposure exceeds the underlying ETF's natural weights.
- **Position correlation within the basket**: SPY-QQQ 30-day return correlation typically 0.85+, SPY-IWM 0.75+, QQQ-IWM 0.7. The basket is effectively **a single-factor (US equity beta) bet that gets sized 3× the per-ticker size in correlation-adjusted terms**. This is intentional given the LLMs are giving us a directional broad-tape signal; if they ever diverge meaningfully on per-ticker direction (the rationales would have to say "long SPY, short IWM"), the v2 spec extends to paired positions. **v1 treats consensus as a broad-tape signal and runs the basket equal-weight.**

## 10. Failure modes

At least 5 named failure modes:

1. **Three LLMs converge on the same training data and all give the same wrong answer (no orthogonality)** — the central failure mode. Looks like: pairwise agreement > 0.9 across all three pairs in the OOS window, while consensus-day P&L is no better than the gate-fails set. This is the §6.5 test — if pairwise agreement is too high, the "ensemble" is a single classifier in a trench coat and there's no orthogonal-blindspot reduction. Diagnostic: Spearman of (consensus signal) vs. (a single-LLM signal alone) is ≥ 0.95.
2. **Prompt instability** — the three LLMs respond differently to small prompt rephrasings or order-of-context changes. Looks like: same-day re-run with header rotation produces > 10% direction-flip rate on the OOS sample. Diagnostic: monthly re-run with 5 prompt perturbations (reordering of context blocks, light paraphrase of system message); require ≤ 10% flip rate on traded days.
3. **Model deprecation / silent update** — one vendor updates a snapshot or deprecates an endpoint, and the downstream classifier behavior drifts. Looks like: a pairwise agreement statistic that was 0.75 in calibration becomes 0.65 over 30 days post-update, OR a single LLM's confidence distribution shifts mean by > 0.1. Diagnostic: daily logging of confidence histograms, weekly KS-test against the calibration baseline. **Pinning policy in §12.**
4. **All three LLMs go down simultaneously** — catastrophic for the day. Mitigated by: abstain-by-default (the strategy default is to hold cash, so total outage is operationally equivalent to a low-conviction day). The risk is that an outage during a major macro event causes us to miss a trade that we *should* have taken — but that's a loss-of-opportunity, not a loss-of-capital. Acceptable.
5. **Regime change** — the LLMs were prompt-engineered on 2015-2017 and calibrated on 2018-2022, neither of which contains a regime like the dot-com era or a 1970s-style stagflation. Out-of-distribution macro regimes could cause all three LLMs to misclassify with high confidence (a "confident wrong" — the most dangerous failure mode in any ML system). Diagnostic: VIX > 35 hard cut (§7) keeps us out of obvious crisis regimes; trailing 30-trade drawdown trigger (§7) catches a slow-rolling regime mismatch before it cumulates.
6. **News-feed contamination / vendor change**: the Benzinga or Reuters feed changes its taxonomy, deprecates a tag, or starts emitting headlines in a different format. The LLM context degrades silently. Looks like: daily input-token count drifts by > 30% from baseline. Diagnostic: log daily context size; alert on drift.

## 11. Falsifiable verdict criteria

**Promote to live capital (after the 2018-2022 calibration → 2023-2026 OOS results are inspected AND a 90-day shadow has run live):**
- OOS Sharpe ≥ 0.8 with 95% bootstrap CI excluding 0.3.
- OOS hit rate ≥ 55% on ≥ 60 gate-passing trades.
- Spearman of `mean_confidence` vs. realized P&L > 0.10 (p < 0.05) OR (if Spearman is below threshold) the rule is collapsed to unit-size-when-gate-fires and the binary-gate-only variant still meets the Sharpe ≥ 0.8 bar.
- At least one pairwise agreement probability across the three LLMs is < 0.80 (orthogonal-blindspot supported).
- Shadow run shows ≥ 15 gate-passing trades in 90 days with no broker-truth divergence and within 50% of OOS Sharpe.
- Multi-LLM consensus Sharpe is ≥ 0.2 Sharpe units above best single-LLM-gate-only variant (else collapse to single-LLM, abandon this iteration's multi-LLM premise, route to iter_11-family).

**Continue shadow (metrics ambiguous):**
- OOS Sharpe ∈ [0.4, 0.8], hit rate ∈ [52%, 55%], no kill triggers hit. Extend shadow by another 90 days. Re-evaluate.

**Kill immediately (any one of):**
- Shadow 30-trade rolling drawdown > -3.5% NAV at any point.
- Daily loss cap (-1.5% NAV intraday) breached on any shadow trade.
- Broker-side fill discrepancy > 0.5% on any trade (lesson 4).
- Spearman of `mean_confidence` vs. P&L is **negative** (not just weak — actually anti-correlated) with p < 0.05 in OOS. That would mean we are systematically sizing wrong; abandon the sizing rule AND probably the whole iteration.
- Pairwise LLM agreement probability rises above 0.92 on all three pairs in OOS (orthogonality dead, no ensemble value).
- Any one vendor silently updates its snapshot (per §12 pinning detection) and the rolling agreement drift trigger fires.

**Falsification within 90 days (the assignment's hard rule):**
At expected ~10-15% gate-pass rate, 90 calendar days × ~62 US trading days ≈ **8-12 gate-passing trades in the 90-day shadow window**. Is that enough to falsify?
- **For the "kill immediately" triggers (drawdown, daily loss cap, fill discrepancy)**: yes, any one of these can fire on a single trade, so 8-12 trades is plenty for the *kill* path.
- **For the "promote" path**: 8-12 trades is **NOT enough** to establish Sharpe ≥ 0.8 with statistical confidence. We make that promote decision on the *combined* OOS + shadow sample (OOS ~80-120 trades + shadow ~8-12 trades = ~90-130 trades). The 90-day shadow is a *consistency check* on the OOS, not a standalone verdict.
- **What 8-12 shadow trades CAN falsify**: a complete absence of edge. If 8-12 shadow trades produce a Sharpe < 0 with bootstrap CI excluding 0.3 from above, that's strong evidence the OOS Sharpe was overfit and the strategy is dead. Concretely: if the shadow Sharpe is < 0.2 the iteration is killed regardless of OOS Sharpe — the promote path requires OOS AND shadow consistency.
- **What 8-12 shadow trades CAN'T falsify**: marginal edges (Sharpe ∈ [0.4, 0.8]) — those need more trades, hence the "continue shadow" path.

**Argument that ≤ 25 trade-days in 90 days is sufficient as a kill-path falsifier**: the strategy is binary at the gate boundary (trade or don't). Of the 8-25 trades that fire, every one is a high-conviction high-confidence trade. If the system fires at high conviction and loses on even half of those trades, the calibration is wrong and the system is dead regardless of statistical-confidence levels. This is the same logic as: "if your highest-conviction picks lose, you have no edge."

## 12. Operational risk note

The single largest operational risk in this iteration is **inadvertent vendor de-pinning across three different vendors with three different snapshot-naming conventions**. Anthropic uses dated snapshot strings like `claude-sonnet-4-5-20251022` and deprecates older snapshots on a published schedule. OpenAI's GPT-4-class pinning has historically been *less* stable (silent backend updates have occurred on undated snapshot names). Llama-3-70B is self-hosted, which sounds safer until you remember that the vLLM runtime, the CUDA driver, and the model weights file each have their own update cycles. A driver upgrade that subtly changes BF16 rounding can change the model's outputs on the same prompt — we have lived this kind of "deterministic until it wasn't" bug before. The pinning strategy is: (i) write `models.lock` containing the three exact snapshot strings + the SHA256 of the local Llama weights file + the vLLM and CUDA versions; (ii) on every daily run, verify the lock against the running configuration before sending the LLM calls (a mismatch aborts the day and abstains); (iii) include the agreement-drift trigger in §7 as a behavioral cross-check on top of the static lock-file check; (iv) run a small fixed test-prompt battery every day and check that the three LLMs return byte-identical outputs on the fixed prompts (any drift on the fixed prompts is a vendor change). Secondary ops risks: (a) the Benzinga news feed could change taxonomy mid-quarter — handled by daily token-count monitoring; (b) IBKR's `reqMktData` for options skew snapshots is occasionally throttled at 16:30 ET when many strategies poll the close — mitigated by polling at 16:25 ET with a 5-min cache; (c) the local A100 hosting Llama-3 is a single point of failure — a rented runpod GPU is the secondary, but if both fail we lose the third vote and per §7 must abstain. The total operational surface is larger than iter_11 (one LLM vendor) or iter_20 (one vendor + Llama fallback used rarely); the three-vendor parallelism is by far the most expensive part of this iteration to keep healthy. The honest framing: the LLM API cost is trivial ($40-100/year), but the *human ops cost* of monitoring three vendors' snapshot lifecycles is non-trivial and is the real reason this strategy might fail in production even if the edge is real.

---

## Addressing the 10 lessons explicitly

1. **Validate before deploy.** Three-stage validation: 2015-2017 prompt-engineering (researcher tweaks); 2018-2022 frozen-prompt calibration (parameter set); 2023-2026 OOS (verdict); 90-day shadow before live capital. No live $ before all four stages. §6.
2. **Binary gates, not confidence rankers.** This iteration has TWO confidence-like quantities and tests both separately: the binary unanimity-and-mean-confidence gate (§6.2 binary-gate test), and the continuous `mean_confidence` and `1/disagreement_penalty` sizing inside the gate (§6.2 ranker test via Spearman). If the binary gate works but the continuous sizing doesn't rank within the gate, we collapse to unit-size-in-gate per §11. Lesson 2 is the most pointed lesson for this iteration and is built into the validation directly.
3. **TP:SL asymmetry is structural.** The strategy uses MOO/MOC same-day intraday holds with no intraday TP/SL within the open-to-close window (other than the §7 daily loss cap circuit breaker). The hold-to-close discipline is deliberate: we are testing the LLM's directional read, not its TP/SL timing. The asymmetric piece is the §7 -1.5% intraday flatten vs. the natural close — the flatten is a stop-loss with no take-profit counterpart, which is the lesson-3 asymmetry favoring tail-risk-management over take-profit-clipping.
4. **Broker truth, not local cache.** §8 mandates 09:35 ET and 16:01 ET broker reconciliation against the strategy's expected position; mismatch ≥ 1 share = page + flatten via marketable-limit. No trust of local quantity.
5. **No PDT-exposed lanes on sub-$25k US equity accounts.** Strategy is daily round-trip, not intraday round-trip (entry at open, exit at close = one day, not pattern-day-trading by the SEC definition since each ticker is held overnight from prior 16:30 decision to next 09:30 open — actually 17 hours overnight + 6.5 hours intraday = ~24 hour minimum hold per ticker, which is not a "day trade" for PDT purposes). However, short legs require Reg-T short which mandates $25k. Strategy disabled for accounts < $25k for the short side. **Long-only sub-variant** for sub-$25k accounts: trade only `direction = long` days (drops half the signal but stays PDT-clean and Reg-T-clean).
6. **One strategy per slot.** §0 details the disjointness from iter_11 (event-driven 3-day) and iter_20 (cross-sectional monthly). Mechanism of edge is different (agreement-as-Bayesian-filter, not narrative-extraction). Co-deployment correlation rule in §7.
7. **No same-data backtest.** Prompt-engineering (2015-2017) is strictly separated from calibration (2018-2022) which is strictly separated from OOS (2023-2026). Few-shot exemplars come ONLY from 2015-2017. The unavoidable LLM-pretraining contamination is named in §6.1 and mitigated via date-stripping and memorization controls.
8. **Battle-tested broker SDKs only.** IBKR `ib_insync` for execution (§2, §8). LLM API SDKs (Anthropic, OpenAI, vLLM) are vendor-supplied and pinned.
9. **Capacity bound named upfront.** §9: ~$500M AUM ceiling, capped by IWM auction depth at the small-cap leg. Acknowledged as a "boutique to mid-size hedge fund" capacity, not an institutional-scale strategy.
10. **Kill criteria written before deploy.** §11 specifies exact promote / continue-shadow / kill triggers with numbers. Six distinct kill triggers, three of which can fire on a single trade.

---

THESIS: Three independent LLMs from different vendors that unanimously agree on direction with mean confidence > 0.7 produce a Bayesian-stronger filter on next-day broad US equity moves than any single LLM's confidence gate of equivalent threshold, because orthogonal blindspots across vendors statistically dampen idiosyncratic per-model hallucinations.
VALIDATION HORIZON: ~90 gate-passing agreement-days across the 2023-01 → 2026-05 OOS window plus a 90-day shadow (~8-12 additional agreement-days), for a combined verdict sample of ~100 unanimous trade-days.
CAPACITY CEILING: ~$500M AUM (IWM auction-depth bound at the small-cap leg of the three-ticker basket).
