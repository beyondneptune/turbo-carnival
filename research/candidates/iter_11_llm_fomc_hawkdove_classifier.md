# Iteration 11: LLM Hawk/Dove Classifier on FOMC Statement Deltas (SPY + TLT)

## 1. Thesis (one sentence)

When an LLM (Claude Sonnet 4 via Anthropic API, with local Llama-3-70B as deterministic fallback) classifies the **delta between successive FOMC statements** as hawkish/dovish/neutral with classifier-reported confidence above 0.80, the post-statement 1-3 trading-day drift in SPY (long on dovish, short on hawkish) and TLT (long on dovish, short on hawkish) is large enough to overcome bid/ask, commissions, and the implementation lag from 14:00 ET release to the classifier's first tradeable decision around 14:05 ET.

Falsifiable form: **Over the 2024-04 → 2026-05 out-of-sample window (8 FOMC statements + 8 minutes releases ≈ 16 events) plus a 90-day forward shadow (~6 events), the LLM-flagged non-neutral high-confidence events generate a Sharpe > 0.7 with 95% bootstrap CI excluding 0 on a per-event return basis, and a hit rate > 60% on directional sign of SPY 3-day forward return. If not, the LLM either (a) has no incremental information over a simple "did the statement change at all" rules baseline, or (b) the post-FOMC drift documented in Lucca/Moench 2015 and the policy-shock effect of Bernanke/Kuttner 2005 has been arbitraged away in the algorithmic era — kill.**

## 2. Market / asset class / venue

- **Equity risk leg**: `SPY` (SPDR S&P 500 ETF) — listed NYSE Arca, $500B+ AUM, ~80M shares/day ADV, median spread $0.01 on a ~$520 stock = ~0.2 bps. T+1 cash settlement, 1099-B short-term cap gains.
- **Rates / safe-asset leg**: `TLT` (iShares 20+ Year Treasury Bond ETF) — listed NASDAQ, ~$50B AUM, ~50M shares/day ADV, median spread $0.01 on ~$90 = ~1.1 bps. T+1 cash settlement. Plays both as a directional duration bet on dovish surprises (rates fall → TLT rises) AND as an anti-correlated hedge to SPY on the hawkish leg.
- **Alternative rates expression for capital efficiency**: `ZN` (CBOT 10-year Treasury Note futures) or `Z3N` (3-year UST futures) for the IBKR-margin-constrained user. Spec'd default to TLT for spec-level simplicity; ZN noted as upgrade for ≥ $250k accounts.
- **Primary exchange**: NYSE Arca (SPY), NASDAQ (TLT), CBOT Globex (ZN if used).
- **Broker**: Interactive Brokers (IBKR Pro) via `ib_insync` (lesson 8). IBKR Reg-T US equity account.
- **Settlement / margin**: T+1 for the ETFs; SPAN margin for ZN if used. Long-only SPY/TLT positions need no margin; short legs (hawkish events going short SPY or TLT) require Reg-T short interest. Account requires $25k minimum equity for short stock trades (PDT-relevant — lesson 5, see below).
- **Tax regime**: SPY/TLT distributions and gains taxed under §1221 short-term cap gains for holds < 1y (which is always the case here). ZN futures get 60/40 §1256 treatment — better tax profile for the rates leg if used.
- **LLM venue**: Anthropic API (`claude-sonnet-4-5`, with `claude-opus-4-7` as escalation for ambiguous parses) and a co-located local Llama-3-70B-Instruct (vLLM) as the deterministic fallback. Local model is *also* the prompt-engineering oracle so we don't leak OOS data into the cloud LLM during research (see §6).

## 3. Time horizon

**Event-driven, multi-day.** 8 FOMC meetings/year + 8 minutes releases (3 weeks after each meeting) = **16 scheduled macro events/year**. Augmenting with the FOMC press conference transcript (released 14:30 ET, 30 min after the statement) gives a *second* trade window per meeting day if signal contradicts — but for spec-level simplicity, we count one decision per event (16/year).

Hold period per event: **3 trading days** (close of event day → close of event day + 3). 1-day hold tested as a sensitivity, but 3-day is the primary horizon based on the Lucca/Moench drift literature and the observation that algorithmic re-pricing of nuanced language has slowed (not eliminated) the half-life of the drift over the 2015-2024 sample.

## 4. Method family and specific algorithm

- **Family**: LLM narrative classification with a rules-baseline ablation. Hybrid LLM + binary-gate sizing (lesson 2).
- **Algorithm**:
  1. **T = 14:00:00 ET on FOMC statement-day** (or minutes-release-day, 14:00 ET): poll the FRB press-release endpoint and the official board feed. Cache the prior statement (already known) and the new statement (just released).
  2. **T + 0-30s**: compute a unified-diff of the two statements. If the diff is empty or trivial (only date/header changes) → emit `neutral` with confidence 1.0; **do not trade**.
  3. **T + 30-90s**: send the diff and both full statements to Claude Sonnet 4 with a **fixed, version-pinned prompt** (the prompt is part of the strategy artifact, hashed and stored). Prompt structure: system message states the task ("classify the policy stance shift from prior statement to current statement"), few-shot examples drawn ONLY from pre-2010 statements (so no contamination of the 2010-2020 training window or 2021-2026 OOS — see §6), then the diff. Required output: JSON `{stance: "hawkish"|"dovish"|"neutral", confidence: 0.0-1.0, primary_evidence: <quote>}`. Use Anthropic's structured-output mode (tool-call schema) so the response is parser-safe.
  4. **T + 90s**: parse JSON. Validation: schema-conformant, confidence ∈ [0, 1], primary_evidence is a verbatim substring of the new statement (LLM-hallucination guard). If validation fails → fall back to Llama-3-70B with the same prompt; if Llama also fails or disagrees → emit `neutral`, **do not trade**.
  5. **T + 2-3 minutes**: apply the **trade gate** (lesson 2 — binary, not continuous):
     - Trade only if `stance ∈ {hawkish, dovish}` AND `confidence ≥ 0.80` AND both LLM and Llama agree on stance sign.
     - All other cases → abstain.
  6. **T ≈ 14:05 ET — order placement**:
     - Dovish + confident → **+25% NAV long SPY**, **+25% NAV long TLT** (the dovish basket: stocks up, rates down → TLT up).
     - Hawkish + confident → **-15% NAV short SPY** (smaller because equity drawdown asymmetry is higher post-hawkish hikes — Bernanke/Kuttner show a ~1% S&P drop per 25bp surprise hike), **+20% NAV long TLT-inverse via TBT** OR equivalently **short TLT 20% NAV** (rates up → TLT down).
     - The asymmetry between dovish (50% gross exposure) and hawkish (35% gross exposure) is intentional: hawkish moves historically have higher realized volatility and faster mean-reversion within 3 days.
  7. **T + 3 trading days at close**: flatten all positions via MOC orders. No discretion, no stop-loss within the hold window other than the daily-loss circuit breaker in §7.
- **Why an LLM, not a hand-coded rule**: A keyword-count baseline ("hawkish" word count vs. "dovish" word count) ignores the structural rephrasings the FOMC uses (e.g. "labor market remains strong" → "labor market has softened somewhat" is a dovish shift that a bag-of-words counter would call neutral). LLMs encode the conditional semantics of policy language as a side-effect of pretraining on financial corpora. We MUST baseline against the keyword counter — the LLM is only worth its operational complexity if it dominates the baseline by ≥ 15 percentage points of directional hit rate on the OOS sample.
- **Why this method fits this edge**: The post-FOMC drift literature (Lucca/Moench 2015 *"The pre-FOMC announcement drift"*; Bernanke/Kuttner 2005 *"What explains the stock market's reaction to Federal Reserve policy?"*; Gorodnichenko/Pham/Talavera 2023 *"The voice of monetary policy"* using ML on Fed text) all document that the *content* and *tone* of Fed communication, beyond the rate-decision integer itself, predicts asset returns over multi-day horizons. The market typically prices the rate decision in 30-60 seconds but takes hours-to-days to fully digest nuanced language changes. An LLM operating in the 60-90 second window after release is positioned to extract that nuance before the slower discretionary desks have read paragraph 4.

## 5. Data requirements

- **Live event feeds**:
  - FRB press-release scraper / RSS for FOMC statements and minutes (free, public).
  - Reuters or Bloomberg headline feed as secondary trigger (a flagged FOMC headline triggers a poll of the FRB endpoint earlier than 14:00:00 ET — but the strategy NEVER acts on headlines alone; the canonical statement text is the trade input).
- **Historical statements**: Every FOMC statement 1994 → present is freely archived on federalreserve.gov. ~250 statements, < 10 MB total.
- **Historical minutes**: Minutes released since 1993, full text. ~250 documents, ~50 MB total.
- **Price data**: SPY and TLT 1-minute bars 1994 → present (Polygon S3 or local cache). ~3 GB total. 5-second bars for the 14:00 → 14:30 ET window of every FOMC day for fill-quality validation (~50 MB).
- **LLM API**:
  - Anthropic API for Claude Sonnet 4 — pinned model snapshot, e.g. `claude-sonnet-4-5-20251022`. Cost: a full FOMC statement is ~1k tokens in, ~200 tokens out. At Sonnet 4 list pricing (~$3/MTok input, ~$15/MTok output as of 2026-05) that is ~$0.006/call. Even with a 5× reflective re-prompt for ambiguous diffs, **24 events/year × $0.05 ≈ $1.20/year in LLM API cost.** This is a CHEAP strategy on the LLM side — the bottleneck is execution and validation, not inference cost.
  - Local Llama-3-70B served via vLLM on a co-located A100 (or rented runpod GPU on FOMC days only). Cost: $0 marginal if owned hardware, ~$1.50/hour rented × 1 hour/event × 24 events/year = $36/year.
- **Storage**: < 5 GB including all historical statements, minutes, and 1-min price bars. Trivial.
- **Cost summary**: ~$50/year all-in (LLM + GPU rental + IBKR data subscriptions already in place for other strategies). Compare to ~24 trades/year × ~$0.50 commission round-trip on ETFs at IBKR Pro = ~$12/year commission. **Total run cost < $100/year.**
- **Latency requirement**: The full pipeline (FRB poll → diff → LLM call → JSON parse → order) must complete in < **3 minutes** from 14:00:00 ET release. Anthropic API typical latency is 2-8 seconds for a ~1k-token prompt; local Llama-3-70B on an A100 is ~10-20 seconds. Network round-trip to IBKR order send is < 200 ms from a US-East VPS. Budget: 30s for FRB scrape, 90s for primary LLM, 60s for fallback if needed, 30s for order placement → fits in 210s = 3.5 minutes. Comfortable.

## 6. Validation plan

This is the critical section for an LLM-based strategy. The lesson-7 violation we MUST avoid is leaking OOS statement text into the prompt-engineering loop.

- **Total scheduled-event sample 1994-01 → 2026-05**: ~250 FOMC statements + ~250 minutes ≈ **500 events**.
- **Prompt-engineering window (no peeking, the strict rule)**: **1994-01 → 2009-12** (~120 statements + ~120 minutes = 240 events). The human researcher iterates the LLM prompt template, the few-shot examples, and the JSON schema *only* on this window. Few-shot examples come ONLY from this window (so the prompt's exemplars are dated). The classifier output for the 1994-2009 events may be tuned, audited, and re-prompted ad libitum.
- **Train / calibration window for downstream signal-to-trade conversion**: **2010-01 → 2020-12** (~88 statements + ~88 minutes = 176 events). The prompt is now FROZEN. The LLM runs on this window once and produces labels; the labels are paired with realized SPY/TLT 1d/3d/5d returns to set the trade-gate parameters (confidence threshold = 0.80, position sizing, hold period). One pass, no iteration.
- **True out-of-sample window (the verdict-bearing sample)**: **2021-01 → 2024-03** (~25 statements + ~25 minutes = ~50 events). The LLM has never seen the post-COVID Fed communication style during prompt engineering. **This is the sample that decides promote/kill.**
- **Held-out shadow window**: **2024-04 → 2026-05** (~16 events). Reserved for the 90-day forward live-shadow run + a final pre-deployment sanity check. Touched once.
- **Why the prompt-engineering / OOS split matters for an LLM specifically**:
  1. *Pretraining contamination*: Claude Sonnet 4 was likely trained on text data that includes some FOMC statements through its cutoff. The model has "seen" the literal statements during pretraining. We CANNOT eliminate that — but we can ensure that the *prompt and few-shot exemplars* we hand-craft do not encode our knowledge of which 2021-2026 events were dovish vs. hawkish in hindsight. The split above achieves that.
  2. *Researcher hindsight*: Without this split, a researcher would (consciously or not) tweak the prompt until the 2022 hiking cycle came out hawkish and 2024 easing came out dovish — that's hindsight bias, not skill. By freezing the prompt before looking at any 2010+ data, we eliminate it.
- **Walk-forward inside the train window**: Within 2010-2020, walk-forward in 1-year test slices stepping quarterly, train window expanding. ~40 forward folds. Used only to estimate parameter stability (confidence threshold), NOT to re-tune the prompt.
- **Purge and embargo**: 5-trading-day embargo around each event (purge 5 days before the event from training, 5 days after from training — prevents leakage from drift overlap). With monthly events this is conservative.
- **Minimum OOS events for verdict**: **50 events (2021-01 → 2024-03 sample)** before promote/kill verdict. Of those 50, the LLM is expected (based on the 2010-2020 base rate) to abstain on ~40% (low-confidence or neutral), leaving ~30 traded events. 30 is borderline — the spec's hard rule on OOS sample size demands more.
- **Augmentation strategy to reach a respectable sample**: As argued, *both* FOMC statements AND minutes are included (doubling the cadence from 8 → 16 events/year). Furthermore, the LLM classification is also generated for **ECB statements** and **BOE statements** during the OOS window as a *transfer-learning sanity check*: if the prompt generalizes to other central banks (different language, same task), that is supportive evidence that the SPY/TLT signal isn't overfit. ECB and BOE events are NOT traded in this candidate (geography is US-restricted per assignment), but their classification is logged.
- **Statistical tests**:
  - Bootstrap (10k draws, block size = 3 events) of per-event traded returns; require **95% CI of mean per-event return excludes 0**.
  - **Directional hit rate** on SPY 3-day forward return sign vs. LLM stance ≥ 60% (vs. 50% null). Binomial test p < 0.05 against 50%.
  - **Sharpe annualized** (16 traded events/year × ~3-day hold ≈ 48 trading days exposed/year, with ~25 traded events/year averaged across confidence gate) ≥ 0.7 with 95% CI excluding 0.
  - **Sharpe vs. keyword-count baseline**: must exceed baseline by ≥ 0.3 Sharpe units. If the LLM cannot beat a 50-line Python script counting "hawkish" / "dovish" word stems, **kill**.
  - **LLM-internal consistency check**: re-run the LLM on the *same* statement 5 times (with temperature 0 in the structured-output path) — require 5/5 same-class output on ≥ 90% of OOS events. Stochasticity > 10% disqualifies the classifier as a tradeable signal.

## 7. Risk envelope

- **Max gross exposure per event**: **50% NAV** (dovish basket: 25% SPY + 25% TLT) or 35% NAV (hawkish basket: 15% short SPY + 20% TLT-short via TBT). Hard cap: 60% NAV gross under any conditions.
- **Max net long-equity-beta exposure during a hold**: +0.25 (dovish event) or -0.15 (hawkish event). Net beta of TLT to SPX is roughly -0.3 to 0 — the TLT leg is partially a hedge on dovish events (positive correlation in rate-cut regimes) and partially anti-correlated on hawkish events.
- **Daily loss cap during a hold**: -3% NAV intraday MTM on the combined event position → flatten the SPY leg only (TLT stays since the rates leg has slower mean-reversion); page operator. -5% NAV → flatten everything, stand down for the remaining hold period.
- **Per-event total loss cap**: -4% NAV on the closed event position → halt strategy until human review. (At 50% NAV gross exposure, a -8% adverse move on both legs costs ~-4% NAV.)
- **Drawdown trigger**: trailing 10-event rolling drawdown ≥ -6% NAV → strategy halted, human review required. At ~16 events/year, 10 events ≈ 7 months, so this is a meaningful but not hair-trigger threshold.
- **Circuit breaker — event-day chaos**: If, between 14:00 ET and the trade send (~14:05 ET), the SPY 1-minute realized volatility exceeds the 99th percentile of all FOMC days 1994-2020 (roughly: SPY 1-min range > 1.5%), **abstain on this event regardless of LLM output**. The market is already in a chaotic re-pricing where the post-event drift is dominated by intraday noise.
- **LLM-output sanity gates**:
  - If `confidence < 0.80` → abstain. (Lesson 2: binary, not continuous-sized.)
  - If `stance = neutral` → abstain.
  - If the LLM's `primary_evidence` substring is not actually present in the new statement (verbatim) → abstain and log a model-failure event.
  - If Claude and Llama disagree on stance sign → abstain.
- **Correlation rule**: This strategy CANNOT be co-deployed with any other US-equity event-driven or rates-event-driven strategy with > 0.4 30-day return correlation. Specifically, it conflicts with any iteration that trades around FOMC days, NFP days, or CPI days on SPY or TLT.

## 8. Execution plan

- **Broker / SDK**: IBKR Pro, `ib_insync` (lesson 8).
- **Order types**:
  - Entry at 14:05 ET: **marketable-limit IOC** at NBBO ± 5 bps (so the order crosses spread but bounds adverse selection during a chaotic minute). Fallback to MKT if IOC unfilled after 10 seconds. SPY and TLT both have sub-2-bps median spreads on event days post-2020.
  - Exit at T+3 close: **MOC** orders submitted by 15:55 ET on the exit day. Fill at the official 16:00 close.
- **Slippage budget**: 10 bps round-trip per leg in normal conditions; 30 bps in a high-volatility-day exit. Measured by comparing fill to NBBO mid at order send and logged every fill. Budget breach over a rolling 12-event window → audit.
- **Order rejection handling**: If IBKR rejects (margin, halt, risk), retry once with MKT for same quantity. On second rejection: page operator and freeze further trading until ack. No silent retry loop (lesson 8 / lesson 4).
- **Broker truth reconciliation (lesson 4)**: Reconcile `reqPositions()` against the local target every 60 seconds during the hold window. Any divergence > 1 share → halt new orders, page operator. The MOC flatten at T+3 ALWAYS uses the broker-reported quantity.
- **What happens if Anthropic API is down at 14:00 ET**: see §12 operational risk note for the formal fallback. Summary: Llama-3-70B local fallback runs in parallel by default, so the strategy degrades gracefully. If BOTH are down → **abstain on this event**, no trade, log the miss as an operational outage.

## 9. Capacity

- **AUM ceiling before edge degrades**: **$100M**. Reasoning:
  - SPY ADV ~$45B notional. 25% of $100M = $25M SPY position = 0.06% of ADV. Negligible.
  - TLT ADV ~$4.5B notional. 25% of $100M = $25M TLT position = 0.6% of ADV at base case, up to ~1% on a thin event day. Still inside the execute-without-impact band.
  - Above $100M, the TLT leg starts moving the 14:05 ET print; above $500M we'd need to either fragment over the first hour or migrate to ZN futures (~$10B ADV, far more capacity).
- **Theoretical ceiling with ZN-futures migration**: $1-2B AUM.
- **% ADV consumed at $100M**: ~0.06% SPY, ~0.6% TLT. Comfortable.
- **Concentration risk**: 50% NAV in two highly correlated US risk assets during a 3-day window. Single-name: SPY or TLT (one ETF each). Single-issuer: BlackRock (TLT) + State Street (SPY). Single-sponsor failure is plausible only in extreme tail and would be caught by the ETF-halt monitoring path.

## 10. Failure modes

1. **LLM model drift (the "Claude got updated" failure)**. The Anthropic API may transparently update the underlying model weights for `claude-sonnet-4-5` between releases. If the classifier output distribution shifts (the same statement classified differently in 2025 than in 2024), our prompt-engineered confidence threshold is stale. **Metric**: weekly canary run of 20 historical statements through the live API endpoint; if > 10% of canary outputs change class or confidence > 0.1, flag for review. **Mitigation**: pin a model snapshot ID (e.g. `claude-sonnet-4-5-20251022`) in API calls — never use the un-versioned `claude-sonnet-4-5` alias. Same for Llama: pin the model weight hash. Re-validate on the 2010-2020 calibration sample on every API model version bump.
2. **Fed communication style change (the press-conference-takeover failure)**. Since Powell, the post-statement press conference (14:30 ET, 30 minutes after the statement) increasingly contains the *real* news (e.g. "transitory" comment in 2021, the 50-bp hint in 2022). If the FOMC statement becomes a placebo and all alpha moves to the press conference Q&A, our statement-text classifier is trading on stale info. **Metric**: rolling 12-event regression of "SPY 3-day return | LLM stance from statement" — if R² collapses by > 50% versus 2010-2020 baseline AND R² is high on the press-conference transcript classified separately, the edge has migrated. **Mitigation**: a *separate* sub-strategy classifying the press-conference transcript (released as PDF + audio ~30 min post-statement) — out of scope for this iteration but flagged as natural extension. For this iteration: kill if metric trips.
3. **Event-window contamination**. Multiple macro events on the same day (e.g. FOMC + ECB on the same Wednesday, or FOMC + Treasury auction). **Metric**: count of overlapping high-impact macro events ±30 min of FOMC release. **Mitigation**: pre-event check against a curated calendar; if another high-impact event within ±30 min, abstain on the FOMC trade (we cannot attribute the drift cleanly). This will cost ~1-2 abstentions per year.
4. **LLM hallucinated quote / parser failure**. LLM returns a JSON with `primary_evidence` that is not a substring of the actual statement. **Metric**: % of LLM outputs failing the verbatim-substring check. **Mitigation**: gated abstain (§7); log all failures; if > 5% failure rate over rolling 10 events, switch primary to Llama and re-prompt-engineer.
5. **Post-FOMC drift decay (the "everyone has an LLM now" failure)**. As algorithmic traders deploy similar tooling, the 1-3 day drift compresses into the first 5-10 minutes. **Metric**: average half-life of the cumulative drift on the 50-event OOS sample. If half-life < 30 minutes (vs. historical ~hours), our 3-day hold has no edge. **Mitigation**: progressively shorten the hold period as the half-life shrinks; if half-life drops below 30 minutes the strategy's economics break (fills at the chaotic minute) and we kill.
6. **API or local-model outage at 14:00 ET**. **Metric**: latency from FRB release to LLM JSON return. > 60 seconds with primary; trigger Llama fallback. > 180 seconds with both → abstain. **Mitigation**: dual-path inference (parallel calls), abstain default, no degraded trade.
7. **FOMC re-releases or corrects the statement**. Rare but happens (typo corrections). **Metric**: a second FRB endpoint payload after the first within 24 hours. **Mitigation**: re-run the classifier on the corrected statement; if the trade is still open and the stance flips, flatten immediately at next print.

## 11. Falsifiable verdict criteria

- **Promote to live capital** — all of the following over the 2021-01 → 2024-03 OOS window AND ≥ 90-day forward shadow (~6 events):
  - OOS Sharpe ≥ 0.7 with bootstrap 95% CI excluding 0.
  - OOS hit rate on SPY 3-day directional sign ≥ 60% (one-sided binomial p < 0.05 vs. 50%).
  - Sharpe gap over the keyword-count baseline ≥ 0.3.
  - Per-event mean return ≥ 30 bps net of slippage and commission.
  - Zero broker-truth divergence in the shadow run.
  - LLM internal consistency ≥ 90% (5-of-5 same class on a fresh re-run).
  - No model-version-bump-induced classification drift > 10% on canary set.

- **Continue shadow** — metrics ambiguous (Sharpe 0.4-0.7, hit rate 55-60%, < 30 traded events in OOS), no kill triggers fired.

- **Kill immediately** if any one of:
  - 3-event rolling P&L drawdown > -4% NAV.
  - Single-event loss > -4% NAV.
  - Broker-side fill discrepancy > 0.5% NAV.
  - LLM-output sanity failure (hallucinated quote, schema break) on > 5% of events in any 10-event window.
  - Trailing 12-month OOS Sharpe < 0.4 OR hit rate < 53% — **the 90-day-deploy falsification clause** (see §11.bis below for what 6 events can actually falsify).
  - Press-conference-takeover metric trips (statement R² halves AND press-conference R² rises).

### 11.bis — what 6 events in 90 days can actually falsify

A 90-day window covers ~3 FOMC statements + ~3 minutes ≈ 6 events. With the abstain gate at ~40%, expect to trade ~3-4 of those. That is too few to estimate Sharpe meaningfully. So the 90-day falsifier is **OPERATIONAL**, not statistical:

  - Did the pipeline actually fire correctly on each event? (Latency, parse, LLM up, order sent, fill quality.) **Threshold: ≥ 5/6 events handled cleanly. ≤ 3/6 → kill on operations grounds.**
  - Did any single event produce a -4% NAV loss? **One single-event blowup in 90 days = kill, regardless of cumulative P&L.**
  - Did the LLM classifications agree with a human auditor's hawk/dove rating on ≥ 5/6 events? **Disagreement on > 2 events = the LLM is judging on a different axis than the documented edge; kill.**
  - Did the canary-set classification stay stable across any Anthropic model snapshot updates? **Any > 10% drift on canary = halt and re-validate.**

In other words, the 90-day window can FALSIFY operationally but cannot CONFIRM statistically. Promote-to-larger-capital still requires the full 50-event OOS sample passing the §11 thresholds.

## 12. One-paragraph operational risk note

The dominant operational risk is the **Anthropic API ↔ FOMC release timing**. The FRB publishes at exactly 14:00:00 ET; in the first 90 seconds after release, every macro trader in the country is hammering the Bloomberg terminal and the API endpoint surface for every LLM provider. Anthropic's API has gracefully degraded under load before; a 60-second hang at 14:00 ET on FOMC day is plausible. We mitigate with a parallel Llama-3-70B served from co-located A100 (warmed up 30 minutes before the release) — the Llama path is the **deterministic fallback** and crucially, since it is run in parallel by default, we also use it as a cross-check (both must agree on stance sign, §4 step 5). If BOTH the Anthropic API and the local Llama path fail or disagree, the strategy ABSTAINS on the event — abstaining is always cheaper than a misclassified trade. Second-order risks: the FRB scraper hitting an HTML structure change (mitigated by parallel polling of FRED's official API and the RSS feed); IBKR gateway connection drop during the 14:00-14:05 ET window (mitigated by heartbeat every 5s, auto-reconnect, broker-side bracket stop placed before any FOMC day at 09:30 ET for the standing maximum-loss boundary — this "hardware breaker" of -4% NAV is the floor); model-version drift on Anthropic's side, addressed by pinning a dated snapshot ID and weekly canary runs against 20 historical statements; clock drift on the trading host (mitigated by chrony-disciplined GPS-NTP, since 1-minute drift = 5 bps of slippage at 14:00 ET on FOMC day). The strategy is "long-tail-of-rare-events" operationally — 16 trade attempts a year means a single operational failure is 6% of yearly trades; we therefore demand **zero unrecovered operational failures** in the 90-day shadow as a promote precondition.

---

## Lesson compliance (the 10 lessons, explicit)

1. **Validate before deploy.** §6 mandates the strict prompt-engineering / train / OOS / shadow split. No live capital until the 2021-2024 OOS Sharpe ≥ 0.7 plus a clean 90-day shadow.
2. **Binary gates, not confidence rankers.** §4 step 5 and §7 define the trade gate: confidence ≥ 0.80 AND non-neutral AND inter-model agreement → fixed sizing (25% NAV per leg). LLM confidence is used ONLY as an on/off filter, NEVER as a continuous position-size signal. This is precisely the C3-confidence-vs-P&L lesson — we do not assume r > 0 between LLM confidence and realized return until proven on OOS.
3. **TP:SL asymmetry is structural.** The 3-day hold with no intra-hold stops (except the daily-loss circuit breaker at -3% NAV intraday and the per-event cap at -4% NAV) lets winning events run for the full drift while bounding losses. Historical asymmetry: post-FOMC dovish-surprise 3-day SPY return distribution has 75th percentile ~+1.5%, 25th percentile ~-0.5% — a ~3:1 favorable skew. The fixed-window exit captures that skew.
4. **Broker truth, not local cache.** §8: per-60-second `reqPositions()` reconciliation during the 3-day hold; MOC flatten uses broker-reported quantity.
5. **No PDT-exposed lanes on sub-$25k US equity accounts.** The strategy trades ~24 round trips/year (one per event, ~16 events/year × 1 RT each = 16 RT; the rest are abstains). That's ~0.3/week, well below the 4-day-trades-in-5-business-days PDT threshold. AND each round trip spans 3 trading days — same-day open/close is impossible by design. PDT cannot fire. Belt-and-suspenders: maintain ≥ $25k account equity; halt if NAV drops below $25k (account suspension auto-triggered by broker anyway).
6. **One strategy per slot.** This is the canonical "LLM-classified Fed-statement event drift" slot. Reject any other iteration that does Fed-event LLM classification on US assets — they would duplicate the edge.
7. **No same-data backtest.** §6 enforces three non-overlapping windows: 1994-2009 for prompt engineering (the LLM-specific lesson-7 concern), 2010-2020 for parameter calibration (one-pass, prompt frozen), 2021-2024 for OOS verdict, 2024-04 onward for shadow. The prompt-engineering window is *forbidden* to inform any decision about 2010+ trade gates. The OOS window is *forbidden* to be looked at before all parameters are frozen. This is the *primary* validation discipline for this candidate.
8. **Battle-tested broker SDKs only.** `ib_insync` (US ETFs). Anthropic SDK (`anthropic` Python) is the official, well-tested client. vLLM is the production-standard server for Llama-3. No hand-rolled HTTP anywhere in the trading loop.
9. **Capacity bound named upfront.** §9: **$100M realistic** (TLT becomes the binding constraint at ~$25M positions); **$1-2B with ZN-futures migration**.
10. **Kill criteria written before deploy.** §11 enumerates exact thresholds. §11.bis explicitly distinguishes what 6 events in 90 days can falsify (operations + outliers) vs. what requires the full 50-event OOS sample (Sharpe, hit rate).

---

THESIS: LLM-classified hawk/dove deltas in FOMC statement text predict the post-event 1-3 day drift in SPY and TLT with enough edge to overcome bid/ask, after gating on classifier confidence ≥ 0.80 and inter-model agreement.
VALIDATION HORIZON: 50 events OOS (2021-01 → 2024-03 statements + minutes) of which ~30 are traded after the abstain gate, plus a 90-day forward shadow (~6 events) that falsifies operationally but not statistically.
CAPACITY CEILING: $100M realistic (TLT-leg ADV-constrained); $1-2B with ZN-futures migration of the rates leg.
