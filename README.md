# trading_system

Systematic options research platform — paper trading only, year 1.

## What This Is

A research-grade paper trading system for testing one specific options trading hypothesis at the smallest viable retail scale. Not an autonomous trading agent. Not financial advice. Not connected to real capital.

## Locked Plan

- **Path:** Hybrid — VRP harvesting core + earnings volatility satellite
- **Capital:** $10,000 paper notional (smallest amount where the hybrid mechanically runs)
- **Allocation:** 70% core ($7k SPY iron condors) / 30% satellite ($3k Russell 1000 earnings straddles)
- **Year 1:** zero live capital, paper trading only
- **Decision rule:** strategy passes only if it clears all 7 pre-registered gates at base AND +50% costs

See `metadata/strategic_lock.md` and `metadata/capital_policy.md` for full commitments.

## Architecture

```
trading_system/
├── ingest/         # data adapters (Polygon Free, yfinance, etc.) — no LLM
├── features/       # deterministic feature engineering — no LLM
├── signals/        # LightGBM models (Model A, Model B) — no LLM in decisions
├── risk/           # vega caps, scenario stress, kill switch — no LLM
├── paper_exec/     # simulated paper execution — no LLM
├── eval/           # walk-forward, DSR, PBO, baselines — no LLM in gate
├── research_llm/   # all LLM calls live here (advisory only)
│   └── prompts/    # version-controlled prompt templates
├── dashboard/      # Streamlit live view
├── metadata/       # locked decisions, prereg docs, audit trail
├── sql/            # SQLite schema and migrations
├── config/         # YAML configs (vendor keys live in .env, not here)
├── data/           # local data lake (Parquet partitioned, not in git)
├── tests/          # pytest
├── scripts/        # one-off utilities (backfill, daily cron)
└── research/       # research notebooks and synthesis docs (in git)
```

Three logical lanes operate over this codebase:

| Lane | Purpose | Constraints |
|---|---|---|
| `paper/` | Frozen version under test | No edits during 90-day cycle |
| `research/` | Experiments, candidates | Cannot write to paper/ |
| `eval/` | Read-only metrics | Independent of both |

## LLM Roles (advisory only — never in trade gate)

| Role | Model |
|---|---|
| Coder | Claude Opus 4.7 (via Claude Code) |
| Critic | GPT-5.5 |
| Summariser | Kimi K2.6 or Gemini 3 Flash |
| Analyst | Perplexity Pro |
| Comms | Claude Opus 4.6 |

Trade decisions are made by deterministic Python in `signals/` + `risk/` + `paper_exec/`. LLMs propose, critique, and explain — they cannot place, size, or approve trades.

## Status

**Phase 0** — skeleton + dummy strategy + dashboard. See `metadata/phase_status.md` for current phase.

## Setup (placeholder — populated when implementation begins)

Requires Python 3.11+. UK-based; broker integration via IBKR Pro UK paper account (planned, cycle 2+).

```bash
# (instructions to be added in next commit)
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Disclaimers

- Year 1 is paper trading only. No real capital is at risk.
- This is a research project, not financial advice.
- Results from paper trading do not guarantee live trading outcomes.
- The system is designed to be falsifiable: failure to pass pre-registered gates ends the project, not invites another framework.
