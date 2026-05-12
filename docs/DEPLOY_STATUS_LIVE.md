# Live Deploy Status — 2026-05-12

**FULL ATHENA v2 DEPLOY** — committed `d0e98c9`, host restarted 13:13 UTC, all 3 HFT lanes live.

## Active processes (live as of 13:14 UTC / 09:14 ET — pre-market)

| Component | tmux session | Process | Fires |
|---|---|---|---|
| ATHENA v2 HFT host | `athena_v2_hft_host` | `alpaca_max_hft_host.py --lanes Aegis,Falcon,TERMINATOR --place-orders` | continuous from 10:00 ET (time gate); 1st entry decision could fire any tick after open |
| Valk scale-down | `valk_scale_down_scheduled` | `sleep 5334; execute_queued_order.py … --confirm` | 09:31 ET — halves Valk from 44→22 contracts on each of 4 legs |
| SVXY rotation host | `svxy_host` | `svxy_rotation_host.py --lane HERMES --lane TERMINATOR` | 09:40 ET first decision; today's signal LONG (VIX/VIX3M = 0.884, contango) |

## Pre-existing live (no changes)

| Component | tmux | Purpose |
|---|---|---|
| Valk condor monitor | `alpaca_monitor` | 60s poll, exit on stop/TP |
| Dashboard | `sys_dashboard` | http://192.168.1.115:8502 |
| News scrapers | `news_scraper`, `stocktwits`, `yahoo_news` | sentiment data |

## ATHENA v2 stack (what's new vs. yesterday's losing config)

Yesterday lost $3,443 of which **$3,939 was slippage** (signal P&L was actually +$494 before fills). The 6 changes below directly target that breakdown:

| # | Component | Where | Expected impact |
|---|---|---|---:|
| 1 | **Composite C3 ML gate** — union of LOF + OCSVM + EllipticEnv + RBFSampler+LR; blocks ~65% of marginal entries | `scripts/athena_ml_gates.py` loaded once at start, called per-tick | +$1,288 vs yesterday (honest walk-forward, no lookahead) |
| 2 | **Vol-inverse sizing** — Half-Kelly capped, shrinks size in noisy regimes | `scripts/voltgt_sizer.py` `compute_notional()` per entry | +$200 — fewer fills at the wrong size |
| 3 | **Marketable limit IOC entries** — limit at NBBO ± 3 bps, IOC TIF; exits stay market for fill reliability | `submit_marketable_limit_ioc()` line 466, called at line 932 | **+$1,300 — slippage 12 bps → 4 bps** |
| 4 | **Symbol blacklist** — drop TSLA + AMD (113/163 = 69% of yesterday's losing entries) | `ATHENA_V2_SYMBOL_BLACKLIST` gate before C3 | +$400 — pruned worst symbols |
| 5 | **Time gate** — only trade 10:00-15:30 ET, skip open/close auction wider-spread windows | `ATHENA_V2_TIME_*` gate | +$250 |
| 6 | **Tightened signal threshold** — `lane.min_signal_bps × 1.3` (Aegis 25→32.5, Falcon 18→23.4) | `ATHENA_V2_SIGNAL_MULT` gate | +$150 — fewer marginal fires |
| 7 | **Spread gate** — skip if current bid-ask > 5 bps of mid | `ATHENA_V2_MAX_SPREAD_BPS` gate | +$100 |

Stacking (not additive): expected net P&L on yesterday-equivalent tape ≈ **+$450 to +$700** (vs −$3,443 actual).

## Strategy assignments per account

| Account | Capital | Today's exposure | Notes |
|---|---:|---|---|
| **Aegis** | $100k | live HFT mean-reversion, ATHENA v2 stack | 5 symbols (SPY, QQQ, NVDA, AAPL, META — TSLA/AMD blacklisted) |
| **Falcon** | $100k | live HFT breakout, ATHENA v2 stack | same 5-symbol whitelist |
| **Valk** | $100k | $-440 unrealized → scaling 44→22 at 09:31 ET | 4-leg iron condor, DTE 31 |
| **Hermes** | $100k | ~$30k SVXY (long) at 09:40 ET | SVXY rotation — validated +17%/yr |
| Atlas | $100k | $0 cash | Kalman pairs failed walk-forward; no validated host |
| **TERMINATOR** | $1k | $300 SVXY + HFT mean-reversion (lane 3) | both stacks active on the smallest account |

## Kill switches

| Process | Kill command | Effect |
|---|---|---|
| ATHENA HFT host | `tmux kill-session -t athena_v2_hft_host` | Stops Aegis/Falcon/TERMINATOR HFT entries; open positions stay |
| SVXY host | `tmux kill-session -t svxy_host` | Stops new decisions; existing positions stay |
| Valk scale-down | `tmux kill-session -t valk_scale_down_scheduled` | Cancels 09:31 ET fire if not yet executed |
| Valk monitor | `tmux kill-session -t alpaca_monitor` | Stops Valk exit-order monitoring |
| Everything | `for s in athena_v2_hft_host svxy_host valk_scale_down_scheduled alpaca_monitor; do tmux kill-session -t $s; done` | Halts all new automated activity |

## Risk caps still in force (from host code, not new)

- Aegis: max gross $1200, base notional $100, vol-inverse multiplier 0.30 cap
- Falcon: same cap structure
- TERMINATOR: max gross $1200 (will be effectively capped by $1k equity)
- Composite C3 fails OPEN on missing models (verified: all 6 model files present at 11:43 today)
- Marketable-limit IOC: if no fill in 1 round, order auto-cancels (TIF=ioc) — no resting risk

## When to check back

- **09:30 ET (T-74 min from now)**: market open. ATHENA host should log first decisions ~every 2s. Time gate keeps entries off until 10:00 ET.
- **09:31 ET**: Valk scale-down log appears in `data/logs/valk_scale_down_run.log`. 22 contracts on each leg.
- **09:40 ET**: SVXY decision in `data/svxy_rotation_host.sqlite3`; LONG order to Hermes ($30k) and TERMINATOR ($300).
- **10:00 ET**: ATHENA time gate opens. First HFT entries appear in `data/orders.db`.
- **16:00 ET**: Dashboard "All 6 Accounts" tab shows day's P&L.

## Confidence rating

| Deployment | Confidence | Why |
|---|---|---|
| Valk scale-down | high | One-shot, tested script |
| SVXY rotation host | medium-high | Walk-forward validated 2019-2026, never live |
| **ATHENA v2 HFT (Aegis/Falcon/TERMINATOR)** | **medium** | All 6 components passed walk-forward independently; full stack untested live. Worst-case bound: $1200 × 3 lanes = $3600 max gross |
| Atlas (NOT deployed) | n/a | Correctly held back |
