# SVXY Sweet-Spot Paper-Shadow Deployment

Live host: `research/live/svxy_sweet_spot_host.py`

## Strategy parameters (sweet spot from `research/results/svxy_sweep_top.json`)

| Param | Value |
| --- | --- |
| Signal | `VIX / VIX3M < 1.00` (contango) |
| Position size | 25% of account NAV in SVXY |
| Cash leg | 75% of NAV (cash, or SGOV with `--sgov`) |
| Circuit breaker | Flatten SVXY for the iteration if intraday VIX > 40 |
| Hard kill (halt daemon) | SVXY intraday return <= -15% |
| Hard kill (halt daemon) | Account drawdown <= -12% from rolling peak |
| Decision time | 15:30 ET (configurable via `--decision-time-et`) |
| Summary time | 16:30 ET (configurable via `--summary-time-et`) |
| Backtest expectation | 92.81% days active, Sharpe 0.96, max DD -8.36% |

## Deployment commands

All commands assume the project venv at `/Users/weeoto/turbo-carnival/.venv`.

### First launch (recommended — interactive confirm on first 5 trades)

```bash
/Users/weeoto/turbo-carnival/.venv/bin/python \
    /Users/weeoto/turbo-carnival/research/live/svxy_sweet_spot_host.py \
    --account AEGIS --confirm
```

Run this inside `tmux` (or `caffeinate -i tmux ...`) on the trading host so it
survives terminal disconnects. The script will:

1. Load credentials from `council/.env.local`.
2. Re-run the sweep variant on the last 90 calendar days and abort if the
   recent signal-active percentage diverges from the sweep baseline by more
   than 10% on the downside.
3. Block on stdin before each of the first 5 live orders. Type `y` to send,
   anything else to skip.

### Steady-state daemon (after first 5 trades validated)

```bash
/Users/weeoto/turbo-carnival/.venv/bin/python \
    /Users/weeoto/turbo-carnival/research/live/svxy_sweet_spot_host.py \
    --account AEGIS
```

### One-shot (cron / launchd)

```bash
/Users/weeoto/turbo-carnival/.venv/bin/python \
    /Users/weeoto/turbo-carnival/research/live/svxy_sweet_spot_host.py \
    --account AEGIS --once
```

Suggested cron line (run at 15:30 ET weekdays):

```
30 15 * * 1-5 TZ=America/New_York /Users/weeoto/turbo-carnival/.venv/bin/python /Users/weeoto/turbo-carnival/research/live/svxy_sweet_spot_host.py --account AEGIS --once >> /Users/weeoto/turbo-carnival/data/logs/svxy_sweet_spot_cron.log 2>&1
```

### Dry run (signal only, no orders)

```bash
/Users/weeoto/turbo-carnival/.venv/bin/python \
    /Users/weeoto/turbo-carnival/research/live/svxy_sweet_spot_host.py \
    --account AEGIS --dry-run --once
```

### SGOV cash leg

Add `--sgov` to route the 75% cash leg into the SGOV ETF. The host first
checks the asset is tradable on the paper account and falls back to plain
cash if it is not.

## Stopping / kill switches

### Graceful halt (no flatten)

Create a kill file. The daemon notices it within ~1 minute and exits cleanly:

```bash
touch /Users/weeoto/turbo-carnival/research/live/STOP_SVXY
```

Remove the file before relaunching:

```bash
rm /Users/weeoto/turbo-carnival/research/live/STOP_SVXY
```

You can also send SIGINT/SIGTERM:

```bash
pkill -f svxy_sweet_spot_host.py
```

### Force-flatten now (sell entire SVXY position immediately)

```bash
/Users/weeoto/turbo-carnival/.venv/bin/python \
    /Users/weeoto/turbo-carnival/research/live/svxy_sweet_spot_host.py \
    --account AEGIS --flatten-now
```

This issues a marketable IOC sell for the full SVXY position and exits. Run
this BEFORE stopping the daemon if you want to be flat overnight while the
host is down.

### Automatic kills

| Condition | Action |
| --- | --- |
| `VIX > 40` intraday | Target -> flat for this iteration, daemon continues |
| `SVXY` intraday <= -15% | Flatten + halt daemon |
| Account drawdown <= -12% (rolling) | Flatten + halt daemon |
| `STOP_SVXY` file present | Halt daemon (no flatten) |

## What to do if something breaks

### "verify_90d ... FAILED — divergence exceeds tolerance"

The recent VIX regime is materially different from the sweep window. Investigate
manually before deciding to deploy. To force-launch anyway (only after a human
review), add `--skip-verify`.

### "Missing Alpaca paper credentials"

The script could not find `ALPACA_<ACCOUNT>_KEY` / `ALPACA_<ACCOUNT>_SECRET` in
`council/.env.local` or the process env. Either:

- Pick a different account: `--account HERMES` (or any nickname present in
  `.env.local`).
- Provide generic fallbacks: `export ALPACA_API_KEY=... ALPACA_API_SECRET=...`.

### "vix_unavailable" in the log

`yfinance` failed to fetch VIX or VIX3M. The script logs a skip row and tries
again next loop. Repeated failures usually mean Yahoo is rate-limiting; wait
a few minutes and re-check `/Users/weeoto/turbo-carnival/data/logs/svxy_sweet_spot_host.log`.

### "account_unavailable" or "svxy_price_unavailable"

Alpaca paper API is unreachable, or the SVXY trade endpoint returned no data
(common pre-market and very near close). The script logs a skip row and
retries next loop.

### "HARD KILL: ... — halting daemon"

The script flattened (or attempted to) and stopped. Inspect the latest rows
in `research/live/svxy_decisions.sqlite3` (table `decisions`) and the log
file at `data/logs/svxy_sweet_spot_host.log`. After investigation, manually
relaunch with `--confirm` so the first trades pause for review.

### "order_failed"

Alpaca rejected the submit. Check `data/logs/svxy_sweet_spot_host.log` for
the HTTP status and message. Common causes: market closed, account flagged,
buying-power exceeded. The script will retry on the next loop iteration.

## Inspecting decisions

```bash
sqlite3 /Users/weeoto/turbo-carnival/research/live/svxy_decisions.sqlite3 \
    "SELECT ts_et, account, ratio, target, target_alloc_pct, svxy_qty_held, svxy_qty_target, order_action, killswitch FROM decisions ORDER BY id DESC LIMIT 20"
```

Daily summaries:

```bash
sqlite3 /Users/weeoto/turbo-carnival/research/live/svxy_decisions.sqlite3 \
    "SELECT * FROM daily_summary ORDER BY date_et DESC LIMIT 30"
```

## Files

| Path | Purpose |
| --- | --- |
| `research/live/svxy_sweet_spot_host.py` | The deployment script |
| `research/live/svxy_decisions.sqlite3` | Decision + summary log (created on first run) |
| `research/live/STOP_SVXY` | Optional kill file (manual halt, no flatten) |
| `data/logs/svxy_sweet_spot_host.log` | Rolling text log |
| `research/results/svxy_sweep_top.json` | Source-of-truth for the sweet-spot parameters |
| `council/.env.local` | Alpaca credentials (gitignored) |
