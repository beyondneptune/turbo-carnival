# Live Deploy Status — 2026-05-12

**ATHENA v2.1 FULL STACK** — 6 commits pushed today, host restarted 13:54:30 UTC with all gates active.

Commit chain:
- `d0e98c9` — ATHENA v2 (composite C3 + vol-inv + IOC entries + symbol blacklist + time gate + spread gate + threshold tightening)
- `e44279d` — initial status doc
- `62c6c8c` — dashboard label refresh
- `0f984a2` — TERMINATOR sizing cap ($30–$120 to share $1k with SVXY)
- `2199a47` — year-1 bootstrap Monte Carlo simulator
- `d61a52a` — **v2.1 safety: daily-loss circuit breaker + compounding capital sizing**
- `7bf7200` — **v2.1 signal: news sentiment + earnings calendar + OB imbalance gates**

## Active processes (5 tmux sessions)

| Component | tmux session | Fires |
|---|---|---|
| ATHENA v2.1 HFT host | `athena_v2_hft_host` | continuous from 10:00 ET (time gate) |
| Valk scale-down | `valk_scale_down_scheduled` | 09:31 ET — halves Valk 44→22 contracts |
| SVXY rotation host | `svxy_host` | 09:40 ET first decision; today's signal LONG (VIX/VIX3M = 0.884) |
| Valk condor monitor | `alpaca_monitor` | 60s poll, exit on stop/TP |
| Dashboard | `sys_dashboard` | http://192.168.1.115:8502 |

## ATHENA v2.1 entry gate chain (order of evaluation)

For each lane × symbol every 2 seconds:

1. **Circuit breaker** — if session loss > 3%, halt all entries for the lane until tomorrow
2. **Best-signal selection** — momentum/breakout/mean-rev per lane mode
3. **Regime filter** — vol-of-vol, ATR, news blackout, abs-return (existing)
4. **Symbol blacklist** — TSLA + AMD dropped (yesterday's 69% of losing trades)
5. **News sentiment** — if 1-hr weighted score < −0.30, skip ← NEW v2.1
6. **Earnings calendar** — if within ±1 day of earnings, skip ← NEW v2.1
7. **Order book imbalance** — if >0.30 magnitude opposing direction, skip ← NEW v2.1
8. **Time gate** — entries only 10:00–15:30 ET (skip auction zones)
9. **Tightened signal threshold** — require |score| ≥ lane.min_signal_bps × 1.3
10. **Spread gate** — skip if current bid-ask > 5 bps of mid
11. **Composite C3 ML gate** — union of LOF + OCSVM + EllipticEnv + RBF+LR blocks
12. **Vol-inverse sizing** — base notional × equity multiplier × vol scaling

## Capital sizing — compounding stack

| Component | Behavior |
|---|---|
| Equity-scaled base | `base_notional × (equity / session_start_equity)` |
| Drawdown gate | if `>20%` off peak → revert to baseline (no scaling) |
| Capacity cap | `equity_multiplier` capped at **5×** (acknowledges finite strategy capacity) |
| Vol-inverse | noisy bars get smaller notional, quiet bars larger |
| Circuit break | session loss `>3%` → halt entries entirely until next trading day |

Net effect: TERMINATOR ($1k) compounds linearly up to $5k base notional, then plateaus. Aegis/Falcon compound up to $60k base notional ($12k × 5). Drawdowns auto-circuit-break.

## Strategy assignments

| Account | Start | Today's strategy | Live |
|---|---:|---|---|
| Aegis | $99,286 | ATHENA v2.1 mean-rev | ✓ |
| Falcon | $97,701 | ATHENA v2.1 breakout | ✓ |
| Valk | $99,552 | iron condor, scaling 44→22 at 09:31 | ✓ |
| Hermes | $100,000 | SVXY rotation (today: LONG) at 09:40 | ✓ |
| Atlas | $100,000 | SVXY regime rotation (added at market open 09:30:29 EDT) | ✓ |
| TERMINATOR | $1,000 | ATHENA v2.1 mean-rev + SVXY (70/30 split) | ✓ |

## Ready-to-run offline (built today, not yet wired)

| Component | File | What it does |
|---|---|---|
| Polygon S3 historical pull | `scripts/polygon_s3_historical_pull.py` | 5-yr minute bars (needs POLYGON_S3 creds) |
| Triple-barrier + purged k-fold CV | `experiments/ml_lab/23_*.py` | proper C3 retrain methodology |
| LinUCB bandit shadow | `experiments/ml_lab/24_*.py` | logs would-have decisions vs C3 baseline |
| Offline RL CQL scaffold | `experiments/ml_lab/25_*.py` | CQL framework (needs 10k+ tuples) |
| Year-1 Monte Carlo | `experiments/ml_lab/22_*.py` | bootstrap projection of yearly take |

## Kill switches

| Process | Kill | Effect |
|---|---|---|
| ATHENA host | `tmux kill-session -t athena_v2_hft_host` | halts new HFT entries; positions managed out |
| SVXY host | `tmux kill-session -t svxy_host` | no new SVXY decisions |
| Valk scale-down | `tmux kill-session -t valk_scale_down_scheduled` | cancels 09:31 ET fire |
| Everything | `for s in athena_v2_hft_host svxy_host valk_scale_down_scheduled alpaca_monitor; do tmux kill-session -t $s; done` | full halt |

## Circuit breaker behavior

If `(equity − session_start) / session_start < −0.03` at any tick:
- Log: `CIRCUIT BREAKER <lane>: session loss X.XX%, halting entries today`
- Add lane to `_LANE_HALTED_TODAY` set
- Skip all entries on that lane for the rest of the day
- Existing positions continue to be managed by `evaluate_exits()`
- Auto-resets on next trading day

## Expected behavior today

| Time (ET) | Event |
|---|---|
| 09:30 | Market opens; host still in time-gate skip (entries paused until 10:00) |
| 09:31 | Valk scale-down fires; 22 contracts each leg |
| 09:40 | SVXY decision; Hermes goes LONG $30k, TERMINATOR $300 |
| 10:00 | ATHENA time gate opens; first HFT entries possible |
| 10:00–15:30 | Normal HFT operation; all 11 gates evaluated per tick |
| 15:30 | Time gate closes; no new entries; positions held to time-stop or close |
| 16:00 | Session ends; dashboard "All 6 Accounts" shows P&L |

Expected daily take (central case): **−$200 to +$700**, with circuit breaker capping max loss at ~3% of starting equity per lane (~$3,000 worst case for Aegis+Falcon, $30 for TERMINATOR).
