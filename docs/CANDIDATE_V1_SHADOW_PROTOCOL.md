# Candidate v1 — 30-Day Paper Shadow Protocol

**Status:** READY (pending Phase 3/4/5 hurdle pass)
**Strategy:** Cointegration commodity ETF pairs — see `specs/candidate_v1_cointegration_pairs.md`
**Host:** `research/candidate_v1/host.py`
**Account:** AEGIS Alpaca paper
**Decisions DB:** `research/candidate_v1/live/decisions.sqlite3`

This protocol governs the Phase 7 paper-shadow stage. The shadow is **observation, not validation of P&L**. Day-1 P&L is noise. The metrics that matter at Day 30 are: realized Sharpe within 1σ of backtest CI, slippage within 1 bp of Perold model, max DD within 1.5× backtest, zero `submit_failed` storms.

---

## Pre-launch checklist

- [ ] Phase 3 backtest hurdles ALL passing: White RC p<0.05, SPA p<0.05, PBO<0.50, DSR>1.0, t>3.0
- [ ] Phase 4 walk-forward hurdles ALL passing: OOS DSR>1.0, 95% CI lo>0, shrinkage<50%, net Sharpe>1.0
- [ ] Phase 5 overlay hurdles passing: overlay-on Sharpe ≥ 0.9× overlay-off, max DD cut ≥ 30%
- [ ] Phase 6 dry-run output matches Phase 4 expectations within 1% on identical dates
- [ ] `STOP_CANDIDATE_V1` file NOT present in `research/candidate_v1/live/`
- [ ] AEGIS account flat (no leftover BIDU/TCOM shorts from 2026-05-13)
- [ ] `council/.env.local` contains `ALPACA_AEGIS_KEY` and `ALPACA_AEGIS_SECRET`
- [ ] Watchdog tmux session named `candidate_v1_shadow` running on a schedule (1 cycle per market close = once per US trading day)

## Daily checklist (every market close)

Run the cycle via cron / tmux scheduler:

```bash
cd ~/turbo-carnival && set -a && source council/.env.local && set +a && \
.venv/bin/python -m research.candidate_v1.host
```

Sequence inside one cycle:
1. STOP file check (if present, host exits cleanly)
2. Load locked variant from prior backtest `results/`
3. Pull AEGIS account state via Alpaca `/v2/account` + `/v2/positions`
4. Compute gross-exposure sanity: real US notional / NAV (the 2026-05-13 bug-fix)
5. Append today's IEX close to cached parquet history
6. For each of 45 pairs: rolling cointegration → z-score → decision
7. Apply MAX_OPEN_PAIRS=4 cap
8. Submit market orders for `open` / `close` / `stop` decisions
9. Log every decision + order to `decisions.sqlite3`
10. Increment / clear `submit_fail_counter.json`; trip STOP after 3 consecutive failed cycles

After cycle ends, append a one-line summary to `data/logs/candidate_v1_shadow.log`.

## Day-by-day timeline

| Day | Action |
|---|---|
| **1** | First live-paper cycle. Tight observation: tail log, verify Alpaca fills match decisions. Compare opened positions to Phase 4 walk-forward predicted opens for the same date. |
| **2–7** | Daily review every market close. Flag any deviation > 2σ between realized P&L and backtest expectation. |
| **8** | Weekly reconciliation #1 — query DB for fills, cross-check broker truth via `/v2/positions`, compute realized 7-day Sharpe. |
| **9–14** | Continue. Daily one-line ping with NAV + open pair count + decisions. |
| **15** | Mid-point review — compute realized 15-day Sharpe, compare to bootstrap CI from Phase 4. Decision point: continue or extend shadow window. |
| **16–21** | Continue. Weekly reconciliation #2 on day 22. |
| **22–28** | Final stretch. Watch alpha-decay watchdog: if any single day prints in the bottom-decile of backtest distribution, flag (per arxiv 2512.11913 — crowded factors crash 1.7-1.8× more often). |
| **29–30** | Final reconciliation. Freeze stats. Generate final report at `reports/candidate_v1_shadow_<date>.md`. |

## Weekly reconciliation steps

```bash
.venv/bin/python -c "
import sqlite3, json
c = sqlite3.connect('research/candidate_v1/live/decisions.sqlite3')
c.row_factory = sqlite3.Row
print('-- decisions last 7d --')
for r in c.execute(\"SELECT action, COUNT(*) FROM decisions WHERE ts_utc > datetime('now','-7 day') GROUP BY action\"):
    print(dict(r))
print()
print('-- orders last 7d --')
for r in c.execute(\"SELECT submitted_ok, COUNT(*) FROM orders WHERE ts_utc > datetime('now','-7 day') GROUP BY submitted_ok\"):
    print(dict(r))
print()
print('-- NAV trajectory --')
for r in c.execute('SELECT ts_utc, equity, n_open_pairs FROM nav_history ORDER BY rowid DESC LIMIT 10'):
    print(dict(r))
"
```

Cross-check with broker truth:

```bash
.venv/bin/python -c "
import os, requests
h={'APCA-API-KEY-ID':os.environ['ALPACA_AEGIS_KEY'],'APCA-API-SECRET-KEY':os.environ['ALPACA_AEGIS_SECRET']}
pos=requests.get('https://paper-api.alpaca.markets/v2/positions', headers=h).json()
print(len(pos),'positions')
for p in pos: print(p['symbol'], p['qty'], p['unrealized_pl'])
"
```

## Kill criteria (active during shadow)

The host self-kills (touches `STOP_CANDIDATE_V1`) when:
- 3 consecutive cycles with `submit_failed` and no `submit_success` → infra alert
- Real US gross exposure > 1.5 × NAV → exposure breach

Operator-triggered kill criteria (per prereg §10):

| Trigger | Decision |
|---|---|
| Realized 30-day Sharpe outside 1σ of backtest CI | Extend shadow by 30 days |
| Realized max DD > 1.5 × backtest DD | Kill candidate (touch STOP, flatten via `--flatten`) |
| Any single-day loss > 2 × backtest worst-day | Kill |
| Implementation shortfall > 2 bps/trade above Perold model | Investigate; if reproducible after 5 trades, kill |
| Bottom-decile day in first 10 shadow days | Pause 5 days; if recurs, kill |
| Cointegration ADF p > 0.10 across ALL 45 pairs for 5+ consecutive days | Halt new entries, exit existing pairs at signal |

## Day 30 — final pre-live review (Phase 8)

Generate `reports/candidate_v1_shadow_final.md` containing:
- Realized 30-day return, Sharpe, max DD, Calmar, turnover
- Bootstrap 95% CI on realized Sharpe (stationary bootstrap, B=2000, block=5)
- Comparison vs Phase 4 backtest prediction: each metric within 1σ?
- Implementation shortfall analysis: realized cost vs Perold model
- Council post-mortem: 5-LLM panel structured critique
- GO / NO-GO recommendation with rationale

GO requires: every Day-30 metric within 1.5σ of backtest, no kill criterion triggered, no infra incidents.

## After Day 30

- **GO:** Move to Phase 9 (live deploy, 1% NAV / $1k max per pair, AEGIS account).
- **NO-GO (extend):** add 30 more days of shadow data, re-evaluate.
- **NO-GO (kill):** touch STOP, flatten, write `reports/postmortem_candidate_v1.md`, return to `specs/global_market_system_v1.md` to pick the next candidate.

## Observability

- **Daily ping:** `data/logs/candidate_v1_shadow.log` tail
- **Dashboard tile:** TBD — single-strategy dashboard on port 8505 (NOT the old `pro_live.py` multi-strategy dashboard, which is retired)
- **Council post-mortem cadence:** weekly, on Sunday evenings

## Files this protocol touches

```
research/candidate_v1/
├── live/
│   ├── decisions.sqlite3         (host writes)
│   ├── nav_state.json            (host writes)
│   ├── submit_fail_counter.json  (host writes)
│   └── STOP_CANDIDATE_V1         (operator or host writes; host honors)
├── results/
│   ├── backtest_summary_*.json   (Phase 3)
│   ├── walk_forward_summary_*.json (Phase 4)
│   ├── overlay_summary_*.json    (Phase 5)
│   └── full_sweep.log
data/cache/candidate_v1/<sym>.parquet  (ingest writes)
data/logs/candidate_v1_shadow.log
reports/candidate_v1_shadow_<date>.md  (Day 30)
reports/postmortem_candidate_v1.md     (if killed)
```

---

The 90-day cycle freeze rule of `~/.claude/projects/-Users-weeoto/memory/project_turbo_carnival.md` applies the moment Day 1 begins. No code edits to anything under `research/candidate_v1/` during the shadow window. Bugs found mid-flight are flagged in the daily log; fixes wait for Day 30+.
