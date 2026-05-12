# Live Deploy Status — 2026-05-12

## Started today (live as of 13:02 UTC / 09:02 ET — pre-market)

| Component | tmux session | Process | Fires |
|---|---|---|---|
| Valk scale-down | `valk_scale_down_scheduled` | `sleep 5334; execute_queued_order.py … --confirm` | 09:31 ET — halves Valk from 44→22 contracts on each of 4 legs |
| SVXY rotation host | `svxy_host` | `svxy_rotation_host.py --lane HERMES --lane TERMINATOR` | 09:40 ET first decision daily; today's signal LONG (VIX/VIX3M = 0.884, contango) |

## Pre-existing live processes (still running, no changes)

| Component | tmux | Purpose |
|---|---|---|
| Valk condor monitor | `alpaca_monitor` | 60s poll, places exit orders on stop/TP trigger |
| Dashboard | `sys_dashboard` | http://192.168.1.115:8502 |
| News scrapers | `news_scraper`, `stocktwits`, `yahoo_news` | sentiment data collection |

## Deliberately NOT running

| Component | Reason |
|---|---|
| **`alpaca_max_hft_host.py` (Aegis + Falcon)** | OLD rule-based code lost $3,443 yesterday. ATHENA v2 patch (composite C3 + Half-Kelly + 2-stage execution) requires ~80 lines of changes to the 861-line host. Needs proper code review + dry-run testing + non-market-hours deployment window. Aegis + Falcon trade zero today. |
| Atlas pairs host | Kalman pairs underperformed in walk-forward; need better pair-selection logic first. |
| TERMINATOR full ATHENA stack | Depends on HFT host patch above. |

## Strategy assignments per account

| Account | Capital | Today's exposure | Notes |
|---|---:|---|---|
| Aegis | $100k | $0 (cash, lane dark) | OLD HFT code unsafe to restart; ATHENA v2 pending patch |
| Falcon | $100k | $0 (cash, lane dark) | Same |
| Valk | $100k | $-440 unrealized on condor → scaling to 22 contracts at 09:31 ET | 4-leg iron condor, DTE 31 |
| **Hermes** | **$100k** | **~$30k SVXY (long) at 09:40 ET** | SVXY regime rotation — validated +17%/yr walk-forward |
| Atlas | $100k | $0 (cash) | No host yet |
| **TERMINATOR** | **$1k** | **~$300 SVXY (long) at 09:40 ET** | Test bed; will add ATHENA later |

## Kill switches

| Process | Kill command | Effect |
|---|---|---|
| SVXY host | `tmux kill-session -t svxy_host` | Stops new decisions; existing positions stay (no auto-flatten) |
| Valk scale-down scheduler | `tmux kill-session -t valk_scale_down_scheduled` | Cancels the 09:31 ET fire if not yet executed |
| Valk monitor | `tmux kill-session -t alpaca_monitor` | Stops Valk exit-order monitoring (condor remains open until expiration 2026-06-12) |
| Everything | `for s in svxy_host valk_scale_down_scheduled alpaca_monitor; do tmux kill-session -t $s; done` | Stops all new automated activity |

## What `/ultrareview` should catch

Before deploying the HFT host patch, run `/ultrareview` (user action). Likely findings to address:

1. Order-routing safety in SVXY host — what happens if Alpaca rate-limits during the 09:40 decision?
2. Whether C3 gate fails open or closed on missing model files (currently fails open — entries allowed if models can't load; verify this is desired)
3. Race conditions if multiple lanes simultaneously hit account endpoints
4. Whether vol-inverse sizing handles NaN vol estimates correctly at session open

## Confidence rating

| Deployment | My confidence | Why |
|---|---|---|
| Valk scale-down | high | One-shot, executed by tested script, reduces risk |
| SVXY rotation host | medium-high | Walk-forward validated 2019-2026, but never live-tested |
| HFT host (NOT deployed) | n/a | Correctly held back |
| Atlas (NOT deployed) | n/a | Correctly held back |

## When to check back

- **09:31 ET (00:31 from now)**: Valk scale-down log appears in `data/logs/valk_scale_down_run.log`. Position should show 22 contracts each leg in dashboard.
- **09:40 ET (00:40 from now)**: SVXY host fires first decision. Look for `decision` row in `data/svxy_rotation_host.sqlite3` and order in Hermes + TERMINATOR accounts.
- **End of session 16:00 ET**: Dashboard "All 6 Accounts" tab shows equity curves for the day.
