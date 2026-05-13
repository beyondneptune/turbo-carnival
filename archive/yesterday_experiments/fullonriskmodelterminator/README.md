# fullonriskmodelterminator

Paper-trading-only experimental lab. Three trading lanes running in parallel on
$100k starting capital, comparing strategies from "conservative quant" to "let
the LLM YOLO".

The user explicitly waived the conservative-quant constraints for this directory.
It exists to actually test what happens at max aggression when you let LLMs drive.

## The three lanes

| Lane | Decision-maker | Sizing | What it picks from | Risk profile |
|------|---------------|--------|--------------------|--------------|
| **1 — Quant Realistic** | Existing deterministic strategy | 1% NAV per trade | Iron condor + earnings straddle (frozen) | Conservative |
| **2 — NIM Council** | 8 NIM models vote (majority) | 1–30% NAV (LLM picks) | Constrained menu of 10 actions | Aggressive |
| **3 — Pure-API** | Single NIM model, free-form | Whatever LLM asks for | ARBITRARY legs/strikes/DTE | Maximum agentic |

All three use **real Black-Scholes pricing** with friction modeling ($0.03 bid/ask spread + $0.65/leg commission).
This is NOT a fake yfinance-theta toy like Gemini's `gemini-carnival/`.

## Run

```bash
# All three lanes, 30-day window (good first test)
.venv/bin/python fullonriskmodelterminator/run_all_three.py --start 2025-05-07 --max-days 30

# Full Year 2 (~233 days; takes ~1-2 hours due to NIM call latency)
.venv/bin/python fullonriskmodelterminator/run_all_three.py --start 2025-05-07

# Just Lane 2 council
.venv/bin/python fullonriskmodelterminator/backtest.py --start 2025-05-07 --max-days 60

# Just Lane 3 pure-API
.venv/bin/python fullonriskmodelterminator/pure_api_trader.py --start 2025-05-07 --max-days 60
```

## Output

Each run drops a directory under `runs/`:

```
runs/three_lane_<TIMESTAMP>/
├── summary.json                  # headline metrics for all 3 lanes
├── three_lane_comparison.png     # side-by-side equity curves + monthly bar
├── lane1_nav.csv                 # Lane 1 NAV over time
├── lane1_trades.csv              # Lane 1 trade log
├── council_backtest_<TS>/        # Lane 2 sub-run
│   ├── nav_history.csv
│   ├── trades.csv
│   ├── decisions.csv             # what each LLM voted, per day
│   ├── vote_cache.json           # cached votes (re-runs are fast)
│   └── headline.json
└── pure_api_<TS>/                # Lane 3 sub-run
    ├── nav_history.csv
    ├── trades.csv
    ├── decisions.csv             # LLM's free-form intent per day
    └── headline.json
```

## Caveats (read before judging the numbers)

1. **Both LLM lanes use REAL options pricing** — Black-Scholes against actual
   historical IV. Not yfinance-daily-theta theatrics.
2. **Friction is modeled** — $0.03 spread per leg, $0.65 commission, applied
   on both entry and exit.
3. **NAV-aware sizing** compounds returns AND drawdowns.
4. **Short positions** use approximate max-loss estimates (5% adverse move for
   undefined-risk; spread width minus credit for defined-risk). Real markets
   can deliver bigger gap moves than 5%, so short-strangle Lane 2/3 results
   are slightly OPTIMISTIC for their wipeout risk.
5. **No defensive gates** — these lanes deliberately don't skip VIX spikes,
   Fed weeks, or earnings. The user wanted max aggression.
6. **Council vote cache** means re-running the same date range hits the cache.
   Delete `vote_cache.json` or use `--no-cache` to force fresh calls.

## What this is NOT for

- Real capital deployment without further validation
- Investor demos as headline numbers (use Lane 1 for that, with realistic friction)
- Replacement of the main `turbo-carnival/experiments/` framework

## When to use which lane

- **Lane 1** for investor-facing realistic projections
- **Lane 2** for "what does an aggressive committee of LLMs think we should
  do under unlimited freedom"
- **Lane 3** for "let a single LLM YOLO with full latitude"

Comparing the three side-by-side answers: does LLM-driven trading beat
deterministic quant? (Likely no, but the experiment is the value.)
