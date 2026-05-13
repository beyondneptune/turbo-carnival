# Iteration 20: LLM-Driven Thematic ETF Basket Rotation (Weekly Narrative Score, Monthly Rebalance)

## 0. Differentiation from iter_11 (read me first)

Iteration 11 used an LLM as an **event-driven, sub-3-minute reflex** on a single text type (FOMC statements) to take 3-trading-day directional bets on SPY and TLT. Two trades per central-bank event, ~16 events/year, no cross-section.

**Iteration 20 is the opposite end of the LLM design space:**

| Axis | Iter 11 (FOMC LLM) | Iter 20 (Thematic LLM, this iteration) |
|---|---|---|
| Decision horizon | Single event, 3-day hold | Weekly score, monthly rebalance, weeks-to-months hold |
| Text input | One ~1k-token FOMC statement | Aggregated weekly corpus: ~50-200 headlines + ~5-20 earnings-call excerpts + social-trend digest, ~30-80k input tokens/week |
| Asset universe | 2 ETFs (SPY, TLT) | 15-25 thematic ETFs cross-section |
| Decision type | Binary directional (long/short SPY+TLT) | Cross-sectional ranking → top-K / bottom-K basket |
| Latency budget | < 3 minutes from release | < 60 minutes from Sunday-evening corpus close |
| LLM cost regime | $50/year all-in | $15-50/MONTH at full context |
| Edge anchor | Lucca/Moench post-FOMC drift (minutes-to-days) | Narrative momentum + De Long/Shleifer noise-trader persistence (weeks-to-months) |
| Failure shape | Misclassified single event = -4% NAV blowup | Slow narrative decay = sustained underperformance vs. baseline |

The two strategies use the same *vendor* (Anthropic API) and the same *family of techniques* (structured-output LLM classification), but the **slot they occupy in the portfolio is disjoint**. They cannot be the same strategy under the "one strategy per slot" rule because their decision cadence, asset universe, and edge timescale do not overlap. Co-deployment is permissible after correlation checks (see §7).

---

## 1. Thesis (one sentence)

When an LLM (Claude Sonnet 4.5 with weekly aggregated financial-news headlines, recent earnings-call transcript excerpts, and a structured social-trend digest as input) produces a cross-sectional ranking of ~20 thematic ETFs by **narrative momentum**, equal-weighting the top 5 ranked themes and avoiding (or shorting) the bottom 5 will outperform an equal-weight buy-and-hold of the same universe by a Sharpe gap ≥ 0.3 over 24+ OOS monthly rebalances — because financial-press narrative attention is documented to decay slowly (weeks-to-months) and an LLM can ingest several orders of magnitude more text than a human PM in the rebalance window.

Falsifiable form: **Over the 2024-01 → 2026-05 out-of-sample window (29 monthly rebalances), the long-top-5 vs. equal-weight-universe basket generates an information-ratio ≥ 0.5 with 95% bootstrap CI excluding 0, AND a hit rate ≥ 55% of monthly rebalances where the top-5 basket outperforms the bottom-5 basket. If not, narrative momentum has either (a) been arbitraged into the price within a week of the news (in which case the weekly LLM scoring is too slow), or (b) does not actually predict cross-section returns at the theme level (in which case van Binsbergen/Han/Lopez-Lira 2023's "Man vs. Machine" gains don't translate to thematic-ETF picking) — kill.**

## 2. Market / asset class / venue

**Universe — 20 US-listed thematic ETFs (the working "ThematicBasket-20")**:

| Ticker | Theme | Sponsor | Approx AUM (2025) | Approx ADV $ |
|---|---|---|---|---|
| `BOTZ` | AI / robotics | Global X | $2.3B | $25M |
| `IBB` | Biotech (broad) | iShares | $7.0B | $200M |
| `URA` | Uranium miners | Global X | $3.5B | $120M |
| `ICLN` | Clean energy (global) | iShares | $1.6B | $50M |
| `BLOK` | Blockchain / digital econ | Amplify | $0.8B | $20M |
| `ARKK` | Disruptive innovation | ARK | $5.8B | $250M |
| `OBSY`/`IBLC` (placeholder) | GLP-1 / weight-loss exposure (e.g. `THCX` → if no clean GLP-1 ETF, use `XBI` biotech as proxy with annotated weight) | varies | varies | varies |
| `SOXX` | Semiconductors | iShares | $14B | $400M |
| `JETS` | Global airlines | US Global | $1.0B | $40M |
| `ITA` | US aerospace & defense | iShares | $7.5B | $150M |
| `COPX` | Copper miners | Global X | $2.2B | $40M |
| `LIT` | Lithium / battery tech | Global X | $1.6B | $30M |
| `KWEB` | China internet | KraneShares | $5.5B | $150M |
| `INDA` | India equities (theme proxy) | iShares | $11B | $180M |
| `XLE` | Energy sector (theme: traditional energy) | SPDR | $40B | $1.5B |
| `XBI` | Biotech (smid-cap, alt to IBB) | SPDR | $7B | $300M |
| `GDX` | Gold miners | VanEck | $14B | $400M |
| `SLV` | Silver (theme proxy) | iShares | $14B | $300M |
| `TAN` | Solar | Invesco | $1.4B | $40M |
| `HACK` | Cybersecurity | ETFMG | $2.4B | $30M |
| `DRIV` | Autonomous & EV | Global X | $0.4B | $5M |

That's a 21-ticker working set; **the live universe at any time is exactly 20 tickers** with one slot reserved for replacement when a constituent fails the liquidity floor (see §10 risk on ETF closure). Universe additions/deletions happen ONLY at the annual review (Jan 1 of each year) and are pre-registered; mid-year drift is forbidden (anti-survivorship).

- **Primary exchange**: NYSE Arca and NASDAQ.
- **Broker**: Interactive Brokers (IBKR Pro) via `ib_insync` (lesson 8).
- **Settlement / margin**: T+1 cash settlement; long-only baseline (Reg-T not required); optional short-bottom-5 sleeve requires Reg-T short, see §7.
- **Tax regime**: §1221 short-term cap gains for monthly rebalances (most positions held < 1y → short-term rate). Distributions taxed as ordinary or qualified per fund.
- **LLM venue**: Anthropic API, pinned snapshot `claude-sonnet-4-5-20251022` (with `claude-opus-4-7-20260201` for monthly audit re-runs), and a co-located Llama-3-70B (vLLM) as deterministic fallback. Local model is also the prompt-engineering oracle (no OOS data leakage — see §6).

## 3. Time horizon

**Two-tier:**
- **Weekly scoring tick**: Every Sunday 22:00 ET, LLM ingests the past 7-day corpus of aggregated text and emits a fresh ranking. The ranking is logged but does **not** trigger trades except at the monthly rebalance.
- **Monthly rebalance**: First trading day of each month, 09:45 ET. The latest weekly ranking becomes the next month's portfolio. Hold ≈ 18-22 trading days.

Hold period per position: typically one monthly cycle (one rebalance), but if the same theme stays in top-5 across consecutive months it is rolled (no flatten/refill cycle if already held). Average position turnover target: **~3-5 of 5 positions rotate per month** (60-100% turnover/month, ~700-1200%/year — moderate, see §8 transaction cost).

## 4. Method family and specific algorithm

- **Family**: LLM cross-sectional ranker + binary basket gate (lesson 2 — confidence is filter, not size).
- **Algorithm**:

  **STEP A — Weekly corpus assembly (every Sunday 18:00-22:00 ET)**:
  1. Pull the past 7 calendar days of headlines from a single primary news vendor (RavenPack, Reuters, or Benzinga) filtered to: (i) the 21 universe themes' keyword catalogs and (ii) the sector tags for each theme. Cap at 200 headlines per theme (most themes will yield 20-80; popular themes like AI may hit cap).
  2. Pull earnings-call transcript excerpts (from AlphaSense or Quartr or scraping Seeking Alpha) for the past 7 days of calls. Filter to the top 5 holdings of each universe ETF (so ~100 companies). For each call, extract the management Q&A section only and truncate to ~500 tokens per call. Total: ~5-20 calls × ~500 tokens = ~10k tokens/week.
  3. Pull a structured social-trend digest. For research stage: use Google Trends API for each theme keyword ("AI investment", "uranium stocks", "GLP-1 weight loss", "copper supply" etc.) — output is a 7-day rolling indexed series 0-100 per theme. Encode as text rows ("theme: BOTZ; Google Trends 7d delta: +18; rank in universe: 3rd"). Optionally augment with X/Twitter cashtag mention counts if budget allows. Total: ~2k tokens/week.
  4. Total weekly input: ~30-80k input tokens, depending on news cadence. Caching note: the system prompt and few-shot examples are static across weeks and use Anthropic prompt caching → marginal cost is on the ~30-80k of fresh text only.

  **STEP B — LLM prompt structure (frozen since prompt-engineering window close)**:

  System message (cached): defines the task, the rubric ("score each theme on a 0-10 narrative momentum scale where 10 = blanket positive press coverage AND positive earnings-call mentions AND rising social search interest, 5 = mixed/neutral, 0 = blanket negative coverage and falling interest"), few-shot examples drawn ONLY from 2015-2018 weekly snapshots (so no contamination of 2019-2023 train and 2024-2026 OOS), required output schema.

  User message: the assembled corpus, tagged with theme labels.

  Required structured output (Anthropic tool-call mode):

  ```json
  {
    "evaluation_week_ending": "YYYY-MM-DD",
    "theme_scores": [
      {
        "ticker": "BOTZ",
        "momentum_score": 8.2,
        "confidence": 0.83,
        "narrative_summary": "<= 50 words",
        "primary_evidence_snippet": "<verbatim quote from input corpus>",
        "regime_tag": "trending_up | stable | trending_down | crisis"
      },
      ... 19 more
    ],
    "universe_level_confidence": 0.81,
    "ambiguous_themes": ["..."],
    "ranked_top5": ["...", "...", "...", "...", "..."],
    "ranked_bottom5": ["...", "...", "...", "...", "..."]
  }
  ```

  **STEP C — Hallucination + sanity gates (lesson 2: binary)**:
  - For each `primary_evidence_snippet`: must be a substring of the input corpus. If not → mark that theme `abstain` (do not use its score this week).
  - `universe_level_confidence < 0.65` → **abstain entire week**, hold last month's portfolio (no rebalance). This is the LOW-CONFIDENCE ABSTAIN RULE.
  - Inter-model agreement: also run the SAME prompt through Llama-3-70B. If the top-5 sets from Claude and Llama overlap by < 3 of 5 names → abstain that week.
  - Output schema validation: any field missing or wrong type → re-prompt once; if still invalid → abstain.

  **STEP D — Monthly rebalance (first trading day, 09:45 ET)**:
  Use the most recent valid weekly score. If the most recent valid score is > 14 calendar days old → abstain (hold). Otherwise:
  - **Long sleeve (baseline, mandatory)**: equal-weight the 5 highest-scoring themes (20% NAV each = 100% NAV gross long).
  - **Short sleeve (optional, OFF by default for ≤ $50k accounts due to short-borrow cost and PDT)**: equal-weight short the 5 lowest-scoring themes (10% NAV each = 50% NAV gross short). When ON, gross is 150%, net is 50%, and a margin account is required.
  - Rebalance is done via MOO orders at 09:30 ET cancel-and-re-place to limit at 09:45 ET. Holdovers (themes still in top-5) are sized to target with delta orders, not flatten-and-refill (saves spread).

- **Why an LLM, not a rules-based ranker**: A keyword-frequency or sentiment-classifier baseline ranking (e.g. VADER on headlines) handles polarity but cannot disambiguate "Nvidia beats earnings → bullish for SOXX" from "Nvidia warns of AI bubble → bearish for SOXX" — these have the same surface lexicon. LLMs encode the conditional structure of financial discourse (earnings beats vs. forward warnings vs. analyst downgrades) as a pretraining byproduct. The candidate is only worth its LLM cost if it dominates a frozen sentiment-baseline ranker by ≥ 0.2 information-ratio units (lesson-2 ablation requirement).

- **Why this method fits this edge**: The relevant literature on narrative momentum (Wachter/Kahana 2024 "Memory and Markets"; Tetlock 2007 "Giving content to investor sentiment"; Bybee/Kelly/Manela/Xiu 2024 "Business news and business cycles") finds that thematic press coverage has a half-life of WEEKS not minutes, that retail flows respond to narrative on multi-week lags (consistent with De Long/Shleifer/Summers/Waldmann 1990's noise-trader persistence), and that ML models trained on text predict cross-section returns at monthly horizons. Van Binsbergen/Han/Lopez-Lira 2023 specifically show LLMs revisiting analyst forecasts improve cross-section equity predictions over a multi-week horizon. Our cadence (weekly score → monthly rebalance) matches that documented decay timescale.

## 5. Data requirements

- **News feed**: RavenPack ($500/mo entry tier) OR Benzinga newsfeed ($300/mo) OR Reuters scraping via a third-party aggregator. **Primary: RavenPack ($500/mo)** — has theme tags built-in, reducing prompt-side processing.
- **Earnings call transcripts**: AlphaSense ($500-1500/mo enterprise), Quartr (~$200/mo for limited coverage), or scrape Seeking Alpha transcripts (free but TOS-grey and unreliable). **Primary: Quartr at $200/mo for the top-200 US tickers covering our universe holdings**, with Seeking Alpha as backup.
- **Google Trends**: Free via `pytrends` (unofficial) or paid API access at minimal cost.
- **Optional social**: X/Twitter API ($200/mo for the basic tier) for cashtag mentions. **Optional, OFF by default**, ON only if walk-forward shows ≥ 0.1 IR contribution.
- **Historical data for backtest**: NewsAPI archive or RavenPack historical bundle for 2015-2026 ($2-5k one-time research fee). Earnings transcripts historical via AlphaSense bulk export (~$3k one-time). Google Trends 2015-2026 free via pytrends.
- **Storage**: ~50 GB for raw weekly corpora (12 years × 52 weeks × ~80k tokens × ~6 bytes/token ≈ 30 GB compressed + indexes ≈ 50 GB).
- **Latency requirement**: 60 minutes from Sunday 22:00 ET corpus close to LLM output. Anthropic API at ~30-80k input tokens runs in ~30-90 seconds; with retries and Llama parallel call, budget is 5 minutes. Comfortable.
- **LLM API cost — THE CRITICAL BUDGET**:
  - Claude Sonnet 4.5 pricing (2026-05): ~$3/MTok input, ~$15/MTok output. Prompt caching gives ~10% of input price on cached portions (system prompt + few-shot, ~10k tokens).
  - Per weekly run: ~70k input tokens (60k uncached + 10k cached) + ~5k output tokens = ~$3 × 0.06 + $0.3 × 0.01 + $15 × 0.005 = $0.18 + $0.003 + $0.075 ≈ **$0.26/week** for Claude Sonnet.
  - Plus Llama fallback (free if owned hardware; ~$3/run if rented A100): $0-3/week.
  - Plus monthly Opus 4.7 audit re-run: ~$2/month at higher per-token cost.
  - **Total LLM API cost: ~$15-30/month** at base case. If we 5x the corpus (more themes, more transcripts) → $75-150/month. **Budget cap: $50/month hard ceiling; abort weekly scoring if cumulative monthly spend > $50.**
  - Combined with data feeds: ~$700-900/month all-in operating cost. **The strategy must clear ~$10k/year of net P&L just to cover OpEx — requires ~$200k AUM minimum to be economic.**
- **Cost summary**: ~$800/month data + LLM + ~$2-3k/year commission and spread = ~$13k/year run-rate.

## 6. Validation plan

The lesson-7 LLM-specific concern (don't let the LLM see post-cutoff news during prompt engineering) is even sharper here than in iter_11, because the news corpus is broader and more recent.

- **Universe construction window**: 2015-01 → 2018-12. Picked the 21 thematic ETFs that existed by 2015-01 with > $50M AUM at that time (to avoid post-launch survivor bias). Of the universe in §2, a few (DRIV launched 2018, BLOK launched 2018) are flagged: for walks before their launch, those slots are vacant — explicit, no constituent imputation.
- **Prompt-engineering window (no peeking, strict rule)**: **2015-01 → 2018-12** (~200 weekly snapshots). Human researcher iterates the LLM prompt template, rubric, few-shot examples, and JSON schema ONLY on this window. Few-shot examples come ONLY from this window. Outputs may be inspected and re-prompted ad libitum.
- **Train / calibration window for downstream signal-to-trade conversion**: **2019-01 → 2023-12** (~260 weekly scoring runs; 60 monthly rebalances). Prompt is now FROZEN. LLM runs once; outputs paired with realized forward returns to set the basket size, top-K cutoff, abstain threshold, and inter-model agreement threshold. One pass, no iteration on this window.
- **True out-of-sample window (the verdict-bearing sample)**: **2024-01 → 2026-05** (~29 monthly rebalances OOS). The LLM has never seen the post-2023 news during prompt engineering. THIS IS THE PROMOTE/KILL SAMPLE.
- **Held-out 90-day forward**: 2026-06 onward as live shadow, ~3 monthly rebalances + ~12 weekly LLM evaluations — too few for statistics, but tests operations.
- **Why the split matters for an LLM specifically**:
  1. *Pretraining contamination*: Claude 4.5's training cutoff includes 2024 news. The LLM has "seen" the news in 2024 OOS during pretraining and may inadvertently anchor its narrative scoring to post-hoc outcomes ("ARKK 2022 crash was widely reported as a deflating bubble; the LLM may remember that NVDA AI rally peaked Q2 2024"). We cannot eliminate that, but we CAN ensure that our prompt-engineering choices and few-shot exemplars are calibrated ONLY on 2015-2018, before the LLM's most "memory-rich" recency window. We also do a *pretraining-contamination canary* (see §11): we score 5 made-up themes ("DLLM: distributed LLM compute ETF — hypothetical ticker, doesn't exist") and require the LLM to ABSTAIN rather than hallucinate scores. If it hallucinates, the prompt is leaky.
  2. *Researcher hindsight*: Without the split, a researcher would tweak the prompt until ARKK ranked low in 2022 and NVDA-themed ETFs ranked high in 2024 — both are after-the-fact narratives. Freezing the prompt at 2018-12 prevents this.
- **Walk-forward inside the train window**: 2019-2023 walk-forward in 6-month test slices stepping monthly, train (calibration) window expanding. ~50 forward folds for parameter stability of abstain threshold and top-K choice.
- **Purge and embargo**: 2-week embargo around each monthly rebalance (no news from t-2 to t+2 used in calibration of the next test slice). With monthly cadence this is conservative.
- **Minimum OOS rebalances for verdict**: **24 monthly rebalances** (the 2024-01 → 2025-12 portion is the hard floor; 2026-01 → 2026-05 is additional). Of those 24, expect ~3-6 abstains based on the 2019-2023 base rate, leaving ~18-21 traded rebalances. 18 is small but the cross-sectional design (long top-5 vs. bottom-5) gives ~120 long-leg-month observations and ~120 short-leg-month observations — enough for bootstrap.
- **Statistical tests**:
  - Bootstrap (10k draws, block size = 3 months) of monthly long-top-5 minus equal-weight-universe excess returns; require **95% CI of mean monthly excess return excludes 0**.
  - **Long-top vs. short-bottom hit rate**: ≥ 55% of months where top-5 > bottom-5. Binomial test p < 0.05 vs. 50%.
  - **Information ratio (long-only vs. equal-weight universe)**: ≥ 0.5 with 95% CI excluding 0.
  - **IR gap vs. frozen sentiment-baseline ranker** (VADER on headlines, no LLM): ≥ 0.2. If the LLM cannot beat a sentiment-counting script by 0.2 IR units, the LLM is not worth its complexity — kill.
  - **LLM internal consistency**: re-run on same week 5 times at temperature 0 → require 5/5 same top-5 set on ≥ 80% of weeks. < 80% disqualifies as a tradeable signal.
  - **Pretraining-contamination canary**: 10 fake themes injected each week (e.g. "Hypothetical lithium-recycling ETF: LIRX") — must produce `confidence < 0.65` (i.e. abstain on fake names) on ≥ 95% of canary injections.

## 7. Risk envelope

- **Max position size per ETF**: 20% NAV (one slot in top-5).
- **Max gross exposure**: 100% long-only baseline; 150% (100% long + 50% short) when short sleeve is enabled.
- **Single-sector / single-sponsor concentration**: top-5 cannot contain more than 2 ETFs from the same sponsor (e.g. cannot hold BOTZ + URA + ICLN + COPX + LIT — all Global X — at once; if LLM ranks 5 Global X funds top, override-drop the lowest two and replace with the next non-Global-X ranks). Single-sector cap: no more than 2 ETFs in the same broad sector tag (e.g. cannot hold all 5 in "natural resources").
- **Daily loss cap during a month**: -2% NAV intraday MTM on the basket → flatten one bottom-quartile-ranked-this-week leg; page operator. -4% NAV → flatten entire basket, stand down until next month.
- **Per-month total loss cap**: -6% NAV monthly drawdown on the basket → strategy halted, human review required.
- **Drawdown trigger**: trailing 6-month rolling drawdown ≥ -10% NAV → halt, human review.
- **Circuit breaker — universe-wide flash crash**: if equal-weight universe drops > -7% in a single day OR > -15% in 5 trading days → flatten all positions, abstain from rebalance until 3 consecutive weekly LLM `universe_level_confidence > 0.7` readings indicate stable regime.
- **LLM output sanity gates** (lesson 2):
  - `universe_level_confidence < 0.65` → abstain the WEEK (no rebalance triggered).
  - Inter-model top-5 overlap < 3/5 → abstain the WEEK.
  - Pretraining-contamination canary failure rate > 5% in any 4-week window → halt strategy.
- **Correlation rule**: This strategy CANNOT be co-deployed with any other US sector-rotation, momentum-on-ETFs, or LLM-thematic strategy with > 0.5 30-day return correlation. It is **compatible** with iter_11 (FOMC LLM) — correlation expected ~0.1 because the time-scale and asset universe are disjoint.

## 8. Execution plan

- **Broker / SDK**: IBKR Pro, `ib_insync` (lesson 8).
- **Order types**:
  - **Rebalance entry (1st trading day, 09:45 ET)**: marketable-limit IOC at NBBO ± 10 bps. Fallback to MKT if IOC unfilled after 30 seconds. Why marketable-limit not MOO: thematic ETFs (BLOK, DRIV, TAN) have wide morning auction spreads — better to wait 15 min for tighter spreads.
  - **Mid-month rebalance**: NONE. Monthly only.
  - **Exit on next rebalance**: same instrument, delta order to new target weight; full flatten only if dropping out of top-5.
- **Slippage budget**: 20 bps round-trip per position in normal conditions; 60 bps in a high-vol day; 100 bps for DRIV or smallest constituents. Measured by comparing fill to NBBO mid at order send. Budget breach over a rolling 6-rebalance window → audit and consider universe trim.
- **Transaction cost reality check**: with ~60-100% monthly turnover × ~30 bps round-trip × ~12 rebalances/year = ~360-720 bps/year of transaction cost drag. The LLM ranker needs to clear this drag. The 2019-2023 calibration must show **gross IR ≥ 0.8** to leave room for ≥ 0.5 net IR.
- **Order rejection handling**: If IBKR rejects, retry once with MKT for same quantity. On second rejection: page operator, freeze further orders until ack.
- **Broker truth reconciliation (lesson 4)**: Reconcile `reqPositions()` against the local target every 5 minutes during the rebalance window and every 30 minutes during hold. Divergence > 1 share → halt new orders, page operator.

## 9. Capacity

- **AUM ceiling before edge degrades**: **$50M**. Reasoning:
  - The binding ETF is `DRIV` at $5M ADV. 20% of $50M = $10M position, which is 200% of ADV — impossible to enter in one day without major impact. So at $50M AUM, DRIV must be excluded from the live universe.
  - Even with DRIV excluded, the next smallest (BLOK, TAN, HACK at $20-40M ADV): 20% of $50M = $10M = 25-50% of a single day's ADV. With limit-IOC fragmentation over 90 minutes, this is borderline workable at ~5-10% per 15-min slice. **$50M is the realistic ceiling without universe trim.**
  - **$10M AUM is the comfortable operating level** where every constituent is enterable in one day at <5% ADV.
- **Theoretical ceiling with universe trim**: drop the bottom-quartile-by-AUM tickers and the ceiling rises to ~$200M. Beyond that, the strategy would have to migrate to single-name baskets (5 stocks per theme instead of 1 ETF per theme) — out of scope.
- **% ADV consumed at $10M comfortable level**: ~2% on smallest, ~0.2% on largest. Comfortable.
- **Concentration risk**: 5 positions, 20% each. Single-sponsor cap = 2 (see §7). Single broad-sector cap = 2. Single-name cap = 20% NAV (one ETF).

## 10. Failure modes

1. **LLM model version drift (the "Anthropic updated the model" failure)**. The Anthropic API may transparently re-train `claude-sonnet-4-5`. Output distribution shifts → stale calibration. **Metric**: weekly canary run of 10 historical (2015-2018) weeks through the live API; if > 10% of canary themes change tier (top-5 / mid / bottom-5) → flag for review. **Mitigation**: pin a dated snapshot ID (e.g. `claude-sonnet-4-5-20251022`); re-validate on 2019-2023 calibration whenever a new snapshot ships; pin Llama weight hash for the fallback. If a snapshot bump causes > 15% tier flip on canary, kill and re-engineer.

2. **Theme universe drift (the "BOTZ was tiny in 2019, huge now" failure)**. Themes grow or shrink in AUM by orders of magnitude — early-stage themes (BOTZ in 2019 at $200M, now $2.3B) are not the same instrument economically. Constituent overlap also drifts (ICLN held different stocks in 2019 vs. 2026). **Metric**: rolling 12-month change in (a) AUM, (b) top-10 constituent overlap, (c) management fee for each ETF. **Mitigation**: annual universe review (Jan 1) replaces drifted constituents; mid-year, if any constituent's AUM drops > 50% peak-to-trough OR top-10 constituent overlap with prior year < 50%, that ticker is paused (cannot be in top-5 until next annual review). Pre-registration of the universe at Jan 1 prevents survivorship.

3. **ETF closure (the "BLOK was shut down" failure)**. Small thematic ETFs do close when AUM collapses. ARKK and BLOK have both seen huge outflows; both are plausibly closable in a tail. **Metric**: monthly AUM check; if AUM < $50M for 3 consecutive months → ticker is paused. Sponsor-issued closure notice → immediate pause and replacement at next monthly rebalance from a pre-registered reserve list (see §2 — 21 tickers with 1 reserve). **Mitigation**: pre-registered reserve. If 3+ closures in a single year, halt strategy and reassess universe.

4. **Narrative momentum break (the "ARKK 2022 crash" failure)**. Narratives can flip violently when reality fails to match the story (ARKK lost 67% in 2022). The LLM ranker may keep ARKK in top-5 for weeks as the press is slow to switch tone. **Metric**: per-position monthly drawdown; if a top-5 position drops > -15% in 5 trading days, the LLM is asked to re-evaluate that ticker immediately (off-cycle weekly run); if its score remains > 7/10 despite the drawdown, that's a regime-detection failure. **Mitigation**: each position has a -15% per-position stop within a month (flatten just that one); the strategy survives but logs the failure. Also: the regime_tag field in the LLM output is checked — if any top-5 has `regime_tag = "crisis"`, it's force-dropped to neutral weight (0) regardless of score.

5. **LLM hallucinated themes (the "the model invented an ETF" failure)**. LLM emits a ticker not in the universe, or hallucinates a theme that doesn't have an ETF. **Metric**: every output ticker is checked against the universe whitelist; out-of-universe tickers → drop and re-prompt once; if persistent → abstain. The pretraining-contamination canary (§6) also catches this class.

6. **Narrative arbitrage compression (the "everyone has an LLM now" failure)**. As more funds deploy LLM-narrative tools, the weeks-to-months persistence compresses to days. **Metric**: regression of forward 1-week / 4-week / 12-week return on LLM score; if the 4-week coefficient halves vs. 2019-2023 baseline AND 1-week coefficient rises, edge has migrated to higher frequency. **Mitigation**: progressively shorten rebalance to bi-weekly if the 12-week half-life drops below 8 weeks; if it drops below 2 weeks, the strategy's economics break (turnover doubles, costs eat IR) and we kill.

7. **News-feed contamination or outage**. RavenPack downtime, Quartr feed lag, or Google Trends API rate limit during Sunday corpus assembly. **Metric**: completeness check — for each theme, must have ≥ 5 news items in the week's corpus. **Mitigation**: secondary news source (Benzinga as backup); if any theme has < 5 items, mark that theme `insufficient_data` and the LLM is instructed to abstain on it (score withheld). If ≥ 5 themes are `insufficient_data` in a single week → abstain entire week.

## 11. Falsifiable verdict criteria

- **Promote to live capital** — all of the following over the 2024-01 → 2026-05 OOS window (24+ monthly rebalances) AND ≥ 90-day forward shadow (~3 rebalances + ~12 weekly LLM runs):
  - OOS net-of-cost IR (long-top-5 vs. equal-weight universe) ≥ 0.5 with bootstrap 95% CI excluding 0.
  - OOS monthly hit rate (top-5 > bottom-5) ≥ 55% (binomial p < 0.05 vs. 50%).
  - OOS IR gap vs. frozen sentiment baseline ≥ 0.2.
  - LLM internal consistency ≥ 80%.
  - Pretraining canary abstain rate ≥ 95%.
  - Zero broker-truth divergence in shadow run.
  - LLM API monthly spend stayed under $50.

- **Continue shadow** — metrics ambiguous (IR 0.3-0.5, hit rate 52-55%, < 24 OOS rebalances completed), no kill triggers fired.

- **Kill immediately** if any one of:
  - 3-month rolling NAV drawdown > -8%.
  - Single-month loss > -6%.
  - Broker-side fill discrepancy > 0.5% NAV.
  - LLM-output sanity failure (hallucinated quote, schema break) on > 10% of weeks in any 8-week window.
  - Trailing 12-month OOS IR < 0.3 OR hit rate < 51%.
  - Narrative-arbitrage-compression metric trips (4-week coefficient halves AND 1-week coefficient rises significantly).
  - Cumulative LLM API spend > $200 in a month (signals corpus blow-out or prompt explosion).
  - 3 universe closures in a year.

### 11.bis — what 90 days can falsify (3 monthly rebalances + 12 weekly LLM runs)

The 90-day shadow has too few monthly rebalances for statistical promote. The falsifier is **OPERATIONAL + OUTLIER**:

  - Did the pipeline fire on every Sunday? Did each rebalance complete? **Threshold: ≥ 11/12 weekly runs and 3/3 monthly rebalances clean. < 10/12 weekly = kill on operational grounds.**
  - Any single-month loss > -6%? **One blowup = kill regardless of cumulative.**
  - Did the canary contamination test stay > 95% abstain rate? **One failure = halt-and-reengineer; sustained < 95% = kill.**
  - Did LLM API spend stay within the $50/mo cap? **Persistent overrun = kill on economics grounds.**
  - Did the LLM model snapshot bump cause canary tier flips > 15%? **If yes, halt and re-validate.**
  - Did the human auditor's top-5 picks (a domain-expert reading the same weekly corpus) agree with LLM top-5 on at least 3 of 5 names in ≥ 8 of 12 weeks? **< 6 weeks of meaningful agreement = the LLM is judging on a different axis than human narrative attention; kill.**

The 90-day window can FALSIFY operationally and on outliers but cannot CONFIRM statistically. Promotion to larger capital requires the full 24+ monthly OOS sample passing §11.

## 12. One-paragraph operational risk note

The dominant operational risk is **data-feed reliability at Sunday 22:00 ET**. Unlike iter_11 (one 14:00 ET FOMC event with sub-3-min latency), this strategy has a multi-source ingest pipeline (RavenPack + Quartr + Google Trends + optional X) where any single source can fail without aborting the run but multi-source failure compounds the abstain rate. RavenPack outages have happened ~2-3 times/year historically; Quartr's coverage of mid-cap earnings calls is incomplete (~70% by ticker). We mitigate with a secondary news vendor (Benzinga) and a 5-item minimum per theme; if a theme has insufficient data, that theme abstains, not the strategy. The LLM-API stability risk is bounded by the dual-vendor (Anthropic + local Llama-3-70B) and the abstain default. The 60-minute latency budget gives ample re-try time. Monthly rebalance turnover is moderate (60-100%/mo); 12 rebalances/year × 5-10 fills per rebalance = 60-120 fills/year on liquid ETFs — well within `ib_insync` rate limits. The cost-budget operational risk is real: prompt blow-out (e.g. an unusually busy news week sending input tokens to 200k+) can spike LLM spend; we cap the cumulative monthly spend at $50 and skip the next weekly run if approached (the weekly miss is tolerable; an abstain on a week is built-in to the system). The pre-trade canary-check (10 fake themes per weekly run) is itself a tiny additional cost (~$0.02/week) and serves as a real-time integrity test. Broker-side risks (IBKR API timeout on rebalance morning) are mitigated by a 4-hour rebalance window (09:30-13:30 ET) — far more generous than iter_11's 5-minute event-driven window — so transient IBKR issues are recoverable. The strategy is "12 critical operations per year" plus "52 weekly inferences" — failure tolerance is ~1 missed rebalance per year (~8% of trade decisions) before economics suffer materially; we therefore demand **≤ 1 unrecovered operational failure / 24 OOS rebalances** and **≤ 4 weekly-scoring misses / 52 weeks** during shadow as a promote precondition.

---

## Lesson compliance (the 10 lessons, explicit)

1. **Validate before deploy.** §6 mandates the 4-window split: prompt-engineering 2015-2018, calibration 2019-2023, OOS 2024-2026, shadow 2026-06+. No live capital before 24+ OOS rebalances + clean 90-day shadow.

2. **Binary gates, not confidence rankers.** §4 and §7 define multiple binary gates: `universe_level_confidence ≥ 0.65`, inter-model top-5 overlap ≥ 3/5, per-theme `confidence ≥ 0.65` (else theme abstains), hallucination canary. The LLM scores ARE used as a ranker (cross-section), but the position sizing is **EQUAL-WEIGHT** (20% NAV each in top-5), NOT proportional to LLM score. The score determines membership, never weight. This is the strict version of "binary not continuous" applied to ranking: confidence filters inclusion, not size. We also validate the rank-confidence-to-realized-return correlation on calibration: if r(LLM score, forward return) < 0.1 within top-5 (i.e. the LLM cannot reliably distinguish #1 from #5), we use equal weight; if r > 0.2, we could consider score-weighted sizing but ONLY after passing OOS. Default is equal weight.

3. **TP:SL asymmetry is structural.** Within a monthly hold, no per-position take-profit (the rebalance IS the natural exit). Per-position stop at -15% inside a month (§10 failure mode 4). Basket-level daily loss cap -2%, monthly cap -6% (§7). The favorable asymmetry: theme momentum has heavy right tails (BOTZ +50% in a quarter is common in trending years) and the monthly hold lets winners run; the -15% per-name stop bounds losers. ~3:1 favorable skew on documented theme momentum.

4. **Broker truth, not local cache.** §8: `reqPositions()` reconciliation every 5 minutes during rebalance, every 30 min during hold. Flatten orders use broker-reported quantity.

5. **No PDT-exposed lanes on sub-$25k US equity accounts.** Default mode is long-only, monthly turnover. 12 rebalances/year × ~5 round trips = 60 round trips/year ≈ 1.2/week, all multi-day holds → PDT cannot fire (PDT is same-day open/close, not multi-day round trips). Short sleeve (§7) requires Reg-T short and a ≥ $25k account; OFF by default. **Belt-and-suspenders: halt strategy if NAV drops below $25k AND short sleeve is on.**

6. **One strategy per slot.** This is the canonical "LLM-thematic-ETF cross-section rotation" slot. Reject any other iteration that does monthly thematic-ETF rotation based on LLM narrative scoring on US ETFs. EXPLICIT NON-COLLISION with iter_11: iter_11 is event-driven SPY/TLT on FOMC days, this is monthly cross-section on 20 thematic ETFs. Different cadence, different universe, different edge timescale. Correlation expected ~0.1; both can be deployed if shadow shows correlation < 0.3.

7. **No same-data backtest.** §6 enforces four non-overlapping windows. Prompt engineering on 2015-2018 only; calibration one-pass on 2019-2023 with frozen prompt; OOS untouched on 2024-2026; shadow only on 2026-06+. The pretraining-contamination canary specifically tests for LLM-pretraining leakage of OOS news.

8. **Battle-tested broker SDKs only.** `ib_insync` (US ETFs). Anthropic SDK official Python client. vLLM for Llama-3 production. No hand-rolled HTTP in the trading loop.

9. **Capacity bound named upfront.** §9: **$10M comfortable**, **$50M realistic with constituent trim**, **$200M with universe trim**. Beyond that, would need migration to single-name basket implementation (out of scope this iteration).

10. **Kill criteria written before deploy.** §11 enumerates exact thresholds for promote/continue/kill. §11.bis distinguishes what 12 weekly + 3 monthly events can falsify (operations, outliers, canary, spend) from what requires the full OOS sample (IR, hit rate, baseline gap).

---

THESIS: LLM-ranked weekly thematic narrative momentum across 20 US-listed thematic ETFs, traded as an equal-weight top-5 long basket rebalanced monthly, captures a documented weeks-to-months noise-trader-driven cross-section drift with edge over both equal-weight buy-and-hold and a frozen sentiment-baseline ranker.

VALIDATION HORIZON: 24+ monthly OOS rebalances (2024-01 → 2026-05) covering ~120 long-leg-month and ~120 short-leg-month observations, plus a 90-day forward shadow (3 rebalances + 12 weekly LLM runs) that falsifies operationally but not statistically.

CAPACITY CEILING: $10M comfortable, $50M realistic (DRIV-ADV constrained at top end), $200M with universe trim of smallest constituents.
