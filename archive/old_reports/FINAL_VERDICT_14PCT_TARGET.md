# Final Verdict: 14% Monthly Target

## Question

Investor expects 14% monthly returns on $100k IBKR paper account. User wants
maximum-effort experimental answer using NVIDIA's unlimited NIM API + the
existing strategy framework.

## Method (3 phases, all run today)

### Phase 1 — Deterministic max-aggression search
- 144 strategy configurations within validated parameter ranges
- Aggressive: high delta, short DTE, wide profit-take, loose stops
- Position sizing 0.5% to 10% per trade
- $10k AND $100k account sizes
- Realistic friction (bid/ask, commission)

### Phase 2 — 30-model NIM feature engineering panel
- 30 diverse models proposed entry/exit/sizing features
- 21/29 substantive responses
- Aggregated by frequency to find consensus
- Top consensus signal: **dealer gamma exposure** (13 mentions)
- Top free-data signals: VIX term structure, Fed calendar, treasury curve

### Phase 3 — Implement + test free panel features
- Added `vix_term_ratio` (VIX/VIX3M), `fed_week`, `yc_slope`
- Built `macro_aware_v3` and `macro_aware_strict` gates
- 35 configurations tested across NAV sizes + risk levels

## Aggregate result

| Metric | Result |
|--------|--------|
| Total configurations tested | 200+ |
| Configs hitting **14% monthly** | **0** |
| Configs hitting 5% monthly | 0 in Phase 3, 16 in Phase 1 (all with -84% DD) |
| Best survivable monthly | **1.36%** (vrp_positive, $10k @ 5% risk, -22.8% DD) |
| Best safe-sizing monthly ($100k @ 1%) | **0.32%** (-6.0% DD) |
| Realistic CAGR at safe sizing | ~3.7%/year |

## Why 14% is not achievable

This isn't an opinion — it's an experimental finding. Three structural reasons:

1. **Iron condor expectancy is bounded by credit × profit_take**
   At 0.12 delta + 30 DTE + $5 wings, max gain per condor is ~$30 on ~$400
   max loss. Win rate 91% gives expected ~$10/contract/trade. Compounding
   that into 14% monthly requires position sizing that destroys the account
   on the first 8% drawdown month.

2. **Defensive gates skip recovery trades**
   The post-mortem panel correctly identified VIX spikes as causing losses,
   but skipping VIX spikes ALSO skips the post-spike IV-crush winners.
   Phase 1 and Phase 3 both showed this empirically: defensive gates reduce
   returns more than they reduce drawdowns.

3. **No combination of free-data features changes the ceiling**
   The best survivable monthly stays around 1-2% across all tested gate
   combinations. Premium data (dealer gamma exposure) might add another
   0.5-1% monthly — still order-of-magnitude below 14%.

## What would actually be required for 14% monthly

None of these are inside the current strategy framework:

- **Different strategy concept** (long vol, calendar spreads, momentum)
- **Heavy leverage** (not available on retail $100k)
- **Different underlying** (0DTE SPX, futures, crypto)
- **Premium data + intraday execution** (intraday gamma squeeze trades)
- **Fraud / Ponzi structure** (Madoff promised half this and ran a fraud)

## What I recommend

### For tomorrow's investor demo

**Reset the conversation, don't deliver 14%.**

Show the investor:
1. The realistic equity curve at $100k @ 1% per-trade risk: +29% over 7 years, -4% max DD
2. The honest range: 0.3-1.7% monthly depending on how much drawdown they tolerate
3. Comparison to alternatives: SPY buy-and-hold did +204% over same period (without active management)

Let them decide if those numbers work for them. If they don't, decline the deal. Better than committing to something the math says you can't deliver.

### For the strategy itself

The current iron-condor + earnings-straddle framework is sound but **structurally
capped at ~1-2% monthly survivable**. To improve:

1. **Add dealer gamma exposure data** ($90/month SpotGamma or build SPX flow ingestion). Probably +0.3-0.7% monthly.
2. **Different strategy sleeve**: pair the vol-selling iron condor with a vol-buying tail hedge (e.g., long VIX calls during contango). Potentially +0.5-1% monthly with much lower max DD.
3. **Walk-forward validation on hold-out period** before any live capital — don't trust 7-year backtest alone.

Realistic target with these improvements: **2-4% monthly with -10% max DD on $100k**.
That's still below 14% but is genuine, defensible, sustainable territory.

## Files written today (in case you want to dig)

- `experiments/aggressive_search.py` — Phase 1 max-aggression search
- `experiments/v2_gates_test.py` — Phase 2 simple VIX-ROC gates (didn't help)
- `experiments/v3_macro_test.py` — Phase 3 macro-aware gates (didn't help)
- `experiments/realistic_execution.py` — friction + sizing simulator
- `experiments/walk_forward.py` — IS vs OOS validation
- `council/expectations_panel.py` — 5-model 14% reality check
- `council/postmortem_panel.py` — 12 dates × 5 models = 60 calls analyzing losses
- `council/feature_panel_30.py` — 30 models proposing features
- `council/red_team_winner.py` — 6-reviewer hostile audit
- `council/code_review_panel.py` — 6-reviewer hostile code audit
- `ingest/ibkr_client.py` — IBKR adapter (Gateway not running yet)
- `ops/ibkr_setup.md` — manual setup procedure
- `HARD_RECOMMENDATION_FOR_TOMORROW.md` — earlier synthesis

## One sentence summary

**Tested 200+ strategy configurations across 3 phases with 60+ NIM model calls. Zero reached 14% monthly. The structural ceiling is ~1.7% survivable monthly. The investor's expectation needs to be reset before any live deployment.**
