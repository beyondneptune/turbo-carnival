# OFI Thompson-Sampling Bandit Host (SPY, Alpaca paper)

Experimental paper-trading host that learns the optimal `|OFI|` (Order Flow
Imbalance) threshold online using a 3-arm Thompson-sampling bandit.

- Host script: `research/live/exp_ofi_bandit_host.py`
- SQLite DB: `research/live/exp_ofi_bandit_decisions.sqlite3`
- Log: `research/live/exp_ofi_bandit_host.log`
- Persistent state: `research/live/exp_ofi_bandit_state.json`
- Kill flag: `research/live/STOP_OFI_BANDIT`
- Alpaca account: `VALK` (paper, ~$99,615)
- Symbol: `SPY`, 10 shares per trade (~$5k notional)

## Strategy at a glance

| Knob | Value |
| --- | --- |
| Polling cadence | 5 s |
| Rolling window | 60 s of trades |
| Bandit arms | thresholds `0.20`, `0.40`, `0.60` |
| Sampling | `np.random.beta(wins+1, losses+1)`, argmax sample |
| Order type | marketable-limit IOC, +/- 5 bps slip |
| Hold time | 30 s |
| Win label | net P&L > 1 bp of entry notional, after ~10 bps round-trip budget |
| Max concurrent | 1 position |
| Daily trade cap | 50 |
| Daily loss cap | -$200 (halt for the day) |
| Consec-loss kill | 5 in a row (halt for the day) |

The signal is OFI = (buy_volume - sell_volume) / total_volume in the last
60 s, where each trade is classified buy/sell by Lee-Ready:

1. Quote rule: trade price vs. latest NBBO midpoint (buy > mid, sell < mid)
2. Tick test fallback at the midpoint or when no quote is available

`|OFI|` is in `[0, 1]`. On every tick we draw a Beta sample for each arm,
pick the arm whose sample is largest, and treat its threshold as the
current decision boundary. We fire only when `|OFI| > threshold`.

## Launch

Dry run (no orders, log everything for ~30 s):

```bash
cd /Users/weeoto/turbo-carnival
.venv/bin/python research/live/exp_ofi_bandit_host.py --dry-run --duration 30
```

Live paper (writes real orders to the VALK paper account; runs indefinitely):

```bash
cd /Users/weeoto/turbo-carnival
.venv/bin/python research/live/exp_ofi_bandit_host.py
```

Recommended production launch wraps it under `caffeinate` + `tmux`:

```bash
tmux new -d -s ofi_bandit \
  'caffeinate -i /Users/weeoto/turbo-carnival/.venv/bin/python \
   /Users/weeoto/turbo-carnival/research/live/exp_ofi_bandit_host.py \
   2>&1 | tee -a /Users/weeoto/turbo-carnival/research/live/exp_ofi_bandit_host.log'
```

Re-attach with `tmux attach -t ofi_bandit`.

## Kill / pause

Hard stop (graceful next tick, flattens nothing — no open position is the
intended state after exit because IOC fills are immediate):

```bash
echo "manual stop $(date -u +%FT%TZ)" \
  > /Users/weeoto/turbo-carnival/research/live/STOP_OFI_BANDIT
```

Or send `SIGINT`/`SIGTERM` to the process (e.g. `Ctrl-C` in the tmux session).

To restart after a halt, delete the kill file:

```bash
rm /Users/weeoto/turbo-carnival/research/live/STOP_OFI_BANDIT
```

## Monitor

Tail the log:

```bash
tail -f /Users/weeoto/turbo-carnival/research/live/exp_ofi_bandit_host.log
```

Inspect the SQLite store:

```bash
sqlite3 /Users/weeoto/turbo-carnival/research/live/exp_ofi_bandit_decisions.sqlite3 \
  "SELECT ts_utc, ofi, threshold, chosen_arm, decision FROM ticks
   ORDER BY id DESC LIMIT 20;"

sqlite3 /Users/weeoto/turbo-carnival/research/live/exp_ofi_bandit_decisions.sqlite3 \
  "SELECT ts_open_utc, arm_idx, direction, entry_price, exit_price,
          pnl_usd, pnl_bps, outcome
   FROM trades ORDER BY id DESC LIMIT 50;"

sqlite3 /Users/weeoto/turbo-carnival/research/live/exp_ofi_bandit_decisions.sqlite3 \
  "SELECT arm_idx, threshold, wins, losses, trades
   FROM bandit_state ORDER BY id DESC LIMIT 3;"
```

The bandit posterior is also persisted to
`exp_ofi_bandit_state.json` after every trade, so you can `cat` that file for
a quick overview between runs.

## Schema

- `ticks`: one row per 5 s poll — raw inputs, computed OFI, the Beta sample
  for each arm, chosen arm, threshold used, and the decision label
  (`no_signal | warmup | no_data | below_threshold | fire | in_position | halted`).
- `trades`: one row per completed entry+exit pair, including the outcome
  label (`win | loss | flat`) used to update the bandit.
- `bandit_state`: periodic snapshots of `(wins, losses, trades)` per arm
  (every 30 s and after each trade).
- `kills`: any time the host stops because of a kill flag or an exception.

## Files / paths

- `research/live/exp_ofi_bandit_host.py` — host
- `research/live/exp_ofi_bandit_decisions.sqlite3` — decisions DB
- `research/live/exp_ofi_bandit_host.log` — rotating-append log
- `research/live/exp_ofi_bandit_state.json` — bandit + daily-risk state
- `research/live/STOP_OFI_BANDIT` — kill flag (presence = stop)

## Notes / scope guarantees

- `--dry-run` never POSTs to `/orders`. It logs `[DRY-RUN] would SUBMIT ...`
  and labels rows with `is_dry_run=1`.
- The host uses `ALPACA_VALK_KEY` / `ALPACA_VALK_SECRET` from
  `council/.env.local` and the paper endpoint
  `https://paper-api.alpaca.markets/v2`.
- Round-trip cost for the win/loss label is a flat 10 bps assumption (twice
  the 5 bps IOC slippage budget). Adjust `ROUND_TRIP_COST_BPS` in the host
  only between 90-day cycles, per repo policy.
- The `/v2/stocks/SPY/trades/latest` poll returns the most recent print and
  is de-duplicated by trade timestamp inside the rolling window, so we do
  not double-count a stale print across ticks.
