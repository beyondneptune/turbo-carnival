# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Two layers run in parallel here

This repo has grown into two distinct workloads with different rules. Don't conflate them.

### Layer A — `paper_v1` evidence run (the original)
A Year 1 train / Year 2 held-out OOS validation of a hybrid options strategy.
Lives in: `ingest/`, `features/`, `signals/`, `risk/`, `paper_exec/`, `eval/`, `experiments/`, `dashboard/app.py`.

### Layer B — Multi-strategy live paper trading (added 2026-05-13)
Five live strategies running on Alpaca paper + Kraken Futures Demo, observed via a custom dashboard. Designed as a **30-day shadow validation** before any real capital. Lives in: `research/`, `council/`, `dashboard/pro_live.py`.

---

## Hard rules (both layers)

- **No AI attribution in commits.** No `Co-Authored-By`, no AI signatures.
- **Never rewrite/amend pushed commits or force-push** unless explicitly asked.
- **No tagging `paper_v1.0-frozen`** — that tag is a CEO action.
- **Never treat synthetic/demo prices as official evidence** for `paper_v1`.
- **LLMs are read-only during `paper_v1` evidence generation** — no model, threshold, feature, or risk-gate changes once evidence is being inspected. Layer-B paper strategies do not have this freeze, but bugs found mid-run should still be flagged before patching mid-cycle.
- **Keep generated data out of git.** Runtime files under `data/`, `research/data/cache/`, and `research/live/*.sqlite3*` are local cache/output.

## Account framework (Layer B)

```
AEGIS  paper  ALPACA_AEGIS_*   3 strategies share one account (SVXY rotation, ADR arb, Nikkei gap)
HERMES paper  ALPACA_HERMES_*  Rust HFT engine (live paper)
ATLAS  paper  ALPACA_ATLAS_*   Shadow Lab — non-trading prediction logger
KRAKEN demo   KRAKEN_FUTURES_* BTC perpetual funding carry
+ idle treasury: FALCON, VALK, TERMINATOR (no strategies attached)
```

Credentials live in `council/.env.local` (gitignored). Kraken Demo's signed REST endpoints return `authenticationError` for our keys — read state from `research/live/kraken_btc_funding_decisions.sqlite3` instead.

---

## Common commands

### Tests / lint (paper_v1)
```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check ingest/ features/ signals/ risk/ paper_exec/ eval/ scripts/ dashboard/ tests/ conftest.py
```

### paper_v1 dashboard (legacy, port 8501)
```bash
.venv/bin/streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 8501
```

### Live trading dashboard (Layer B, port 8504, public via Cloudflare tunnel)
```bash
tmux new -d -s live_monitor "/Users/weeoto/turbo-carnival/.venv/bin/python /Users/weeoto/turbo-carnival/dashboard/pro_live.py"
# Tunnel — public URL is in research/live/logs/cloudflared_tunnel.log
tmux new -d -s cloudflared "/tmp/cloudflared tunnel --url http://localhost:8504"
```

### Rust HFT engine (HERMES live, ATLAS shadow planned migration)
```bash
cd research/hft-engine
~/.cargo/bin/cargo build --release
# Live (places paper orders on HERMES):
./target/release/hft-engine
# Dry-run (no orders, same predictor logic):
./target/release/hft-engine --dry-run
```

### Strategy hosts (Layer B)
Each runs in its own tmux session with `caffeinate` so macOS doesn't sleep them.
```bash
# Each host has its own kill-file path; touching the file ends the loop cleanly.
tmux capture-pane -t <session> -p     # peek at latest output
tmux kill-session   -t <session>      # force stop
```
Active sessions and their hosts:
| Session              | Host                                                      | Account  |
| -------------------- | --------------------------------------------------------- | -------- |
| `svxy_sweet`         | `research/live/svxy_sweet_spot_host.py`                   | AEGIS    |
| `adr_alpaca`         | `research/live/iter_18_adr_alpaca_host.py`                | AEGIS    |
| `nikkei_etf`         | `research/live/iter_6_nikkei_etf_host.py`                 | AEGIS    |
| `kraken_btc_funding` | `research/live/iter_3_kraken_btc_funding_host.py`         | Kraken   |
| `hft_engine`         | `research/hft-engine/target/release/hft-engine`           | HERMES   |
| `atlas_shadow`       | `research/live/atlas_shadow_hft.py` (Python; migrating)   | ATLAS    |

### NIM (NVIDIA) council
```bash
.venv/bin/python council/run_council.py list-models           # what's reachable
.venv/bin/python council/today_postmortem.py                  # 5-model panel on today's losses
```
Configured models: see `council/config.default.json`. Current default for analysis: `mistralai/mistral-large-3-675b-instruct-2512`. Avoid `qwen/qwen3-coder-480b-*` for financial reasoning — it's code-tuned.

### Long jobs
Wrap with `caffeinate -i` so the Mac doesn't sleep, and tmux so disconnections don't kill them. Long data builds should write progress to `data/paper_v1_status.json` so the dashboard can show liveness instead of guessing.

---

## Architecture orientation

### Layer A — paper_v1 pipeline
Pipeline shape: `ingest → features → signals → risk → paper_exec → eval`. Cache-first under Polygon Free's 5 rpm limit; jobs must be resumable. Required chain pricing columns: `bid`, `ask`, `mid`, `underlying_close`, `atm_iv` — augment the cache before running the final evidence verdict if any are missing.

### Layer B — multi-strategy live ops
Each strategy is **independent** and writes to its own SQLite DB under `research/live/<name>_decisions.sqlite3`. The dashboard reads broker truth from Alpaca + local DBs for Kraken/shadow data. **Do not aggregate AEGIS, HERMES, ATLAS, KRAKEN equity into one number** — they're different models on different accounts and the user has explicitly called this out as a sin.

Validation funnel (per `research/GLOBAL_EXPANSION_ROADMAP.md`):
- Day 0–30: paper shadow only (current phase)
- Day 30–60: validated strategies migrate to tiny live capital
- Day 60–120: new candidate research
- Day 120+: leveraged portfolio target

### HFT engine (Rust, in `research/hft-engine/`)
Modular: `main.rs` orchestrator, `data_feed.rs` (Yahoo ES + Alpaca SPY), `signal.rs` (online RLS predictor), `latency.rs` (continuous probe), `executor.rs` (marketable-limit IOC orders), `logger.rs` (SQLite), `nav.rs`, `config.rs`.

**Selective execution gate (4 conditions, ALL must hold to fire):**
1. `latency_p99 < 250 ms`
2. `confidence > 0.55`
3. `|predicted_bps| > 1.5`
4. `|predicted_bps| > 3 × spread_bps`

Known landmines:
- **Yahoo `range=10m` is invalid** — use `range=1d`. The engine keeps last 20 1-min bars internally.
- **Alpaca rejects sub-penny limit prices** (HTTP 422 code 42210000). Round to nearest cent in the marketable direction.
- **Latency p99 buffer is time-windowed** (5-min retention) + warmup-discard, otherwise the first slow sample anchors the gate indefinitely.

### Council (`council/`)
LLM panels for code review, post-mortem, red-team, idea generation, predeploy review. Useful in two cases: (a) catching bugs before deploy, (b) independent second-opinion when something looks broken. **Not useful for predicting markets** — no LLM has alpha. Don't deploy LLM-as-coach as if it adds insight; it mostly restates.

### Dashboard (`dashboard/pro_live.py`)
FastAPI + dark HTML + lightweight-charts (TradingView CDN). 1 Hz frontend polling, server-side TTL cache (3–60s by endpoint), stale-while-revalidate for NIM calls. Diff-based DOM updates, not innerHTML rewrites. **Never block the request handler on a NIM call** — always cached and async-refreshed.

---

## Workflow norms

- **Verify, don't generate.** For trading research, executing code that produces numbers beats writing markdown design docs. The user explicitly prefers this (memory: `feedback_verify_dont_generate`).
- **Explore first** when touching evidence, metadata, or API/cache behavior. Then patch narrowly and verify.
- **Distinguish terminal idle states from bugs.** `cache_complete` is terminal; "no-trade today" for Nikkei is correct behavior, not a hang. Only flag if the dashboard mislabels.
- **Prefer root-cause fixes** over user decisions when the issue is an implementation gap.
- **Single-account framework.** The user has corrected attempts to fan strategies out across multiple accounts. Stick to: ONE main portfolio account framework (AEGIS) + ONE experimental track per account when explicitly requested. Idle accounts are treasury, not parallel experiments.

## Layer-B current evidence policy

- 30-day paper shadow is **observation, not validation of P&L**. Day-1 P&L is noise. The metrics that matter at Day 30 are: win rate vs backtest, slippage vs spec, Sharpe stability, cross-strategy correlation.
- Bugs found during the shadow get patched (today: 3 P0 patches — HFT latency anchoring, sub-penny limits, ADR partial-fill book mismatch, Kraken poll_fill cumulative size).
- Strategies graduate to small live capital only after Day 30 metric review.
