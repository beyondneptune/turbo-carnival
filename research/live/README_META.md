# META Validation Framework — Operator Guide

Implementation of iter_25: a strategy registry + auto-promote/auto-kill engine
that sits above every deployed paper-shadow host in `research/live/`. This is
**load-bearing infrastructure**, not a trading strategy. It ingests the other
strategies and emits state transitions.

| File | Purpose |
| --- | --- |
| `meta_framework.py` | The framework (CLI + library). |
| `meta_registry.sqlite3` | Registry + daily metrics + transition log. Created on `init`. |
| `meta_framework.log` | Rolling log of every framework action. |
| `META_GLOBAL_KILL.log` | Append-only record of every global-kill event. |
| `STOP_*` / `KILL_SWITCH` | Per-strategy stop files the framework touches when it auto-kills. |

Spec: `/Users/weeoto/turbo-carnival/research/candidates/iter_25_meta_validation_framework.md`

---

## 1. Database schema

The sqlite mirror has three tables. Git is still the source of truth for code,
but the registry sqlite is the source of truth for **state**.

### `strategies`

| column | type | notes |
| --- | --- | --- |
| `id` | TEXT PK | stable identifier, e.g. `svxy_sweet_spot` |
| `name` | TEXT | human label |
| `version_sha` | TEXT | git commit pinning the deploy script. Framework rejects on register if SHA is unreachable in the turbo-carnival repo. |
| `status` | TEXT | one of `research`, `shadow`, `tiny_live`, `live`, `killed`, `paused`, `quarantine` |
| `deployed_at` | TEXT | UTC ISO timestamp of the initial register |
| `host_path` | TEXT | absolute path to the deployment script |
| `broker` | TEXT | one of `alpaca`, `ibkr`, `binance`, `okx`, `coinbase`, `ccxt` |
| `account_id` | TEXT | broker-side identifier (paper/live tier) |
| `verdict_criteria_json` | TEXT | JSON: at minimum one of `promote_min_sharpe`, `promote_max_dd_pct`, `promote_min_ann_return_pct` |
| `capacity_ceiling_usd` | REAL | Lesson 9 — must be > 0 to register |
| `kill_criteria_json` | TEXT | JSON: must include `kill_3d_dd_pct`, `kill_daily_loss_pct`, `kill_rolling_sharpe`. Optional: `kill_60d_dd_pct`. Lesson 10 — must be present to register. |
| `stop_file_path` | TEXT | absolute path to the host's STOP file; framework touches this on auto-kill |
| `notes` | TEXT | free-form |

### `daily_metrics`

One row per strategy per trading day. The promotion / kill engines read from
here.

| column | notes |
| --- | --- |
| `strategy_id` | FK to strategies.id |
| `date` | YYYY-MM-DD (UTC trade day) |
| `ann_return_pct` | annualized return for the rolling window |
| `sharpe` | rolling Sharpe |
| `max_dd_pct` | worst peak-to-trough drawdown in window, as positive percent |
| `trades` | count of trades that day |
| `win_rate` | fraction in `[0, 1]` |
| `broker_truth_match` | 1 if broker-reported positions matched intended; 0 if a divergence happened |
| `drawdown_pct_today` | day's drawdown, positive percent — used for daily-loss-cap kill check |

PRIMARY KEY `(strategy_id, date)` — `record-daily` upserts.

### `state_transitions`

Append-only log. Every register, promote, kill, force-kill, and global-kill
writes one row. This is the audit trail.

| column | notes |
| --- | --- |
| `id` | auto-incrementing |
| `strategy_id` | FK |
| `from_status` | previous state (null on initial register) |
| `to_status` | new state |
| `reason` | human-readable string |
| `at` | UTC ISO timestamp |

---

## 2. Daily operator workflow

Run these two commands each trading day. Five minutes total, fully scriptable.

```bash
# Each `live` and `tiny_live` host should have written a row to daily_metrics
# (the host's job, not the framework's). Then:

cd /Users/weeoto/turbo-carnival
.venv/bin/python research/live/meta_framework.py kill-check
.venv/bin/python research/live/meta_framework.py promote-check
```

- `kill-check` runs first. If any strategy in `tiny_live` or `live` breaches
  its declared kill criteria, the framework flips `status -> killed`, logs a
  transition row, and **touches that strategy's STOP file** so the host
  process notices on its next loop and halts cleanly.
- `promote-check` runs second. For each `shadow` strategy with >=30 daily
  metrics it checks the verdict criteria; if all pass, it promotes to
  `tiny_live`. For each `tiny_live` strategy with >=14 daily metrics it
  checks the same criteria plus the hard `broker_truth_match` gate (must
  have NEVER been False historically) before promoting to `live`.

Weekly review (Friday or weekend):

```bash
# Snapshot for the dashboard / for human review
.venv/bin/python research/live/meta_framework.py dashboard-snapshot > /tmp/meta_snapshot.json

# Recent metrics for each strategy
for s in svxy_sweet_spot iter_18_adr iter_3_btc_funding iter_6_nikkei_gap ; do
    echo "=== $s ==="
    .venv/bin/python research/live/meta_framework.py metrics --strategy "$s" --days 14
done
```

The weekly review is where the operator looks for **why** the framework took
each action, especially any promote-check that promoted or any kill-check that
killed. Per the iter_25 spec §0, the operator's job is to confirm the
framework's automated decisions match what a human would have made.

---

## 3. Registering a new strategy

```bash
.venv/bin/python research/live/meta_framework.py register \
    --id   iter_8_pairs_coint \
    --name "iter_8 commodity ETF pairs cointegration" \
    --version-sha "$(git -C /Users/weeoto/turbo-carnival rev-parse HEAD)" \
    --status shadow \
    --host-path /Users/weeoto/turbo-carnival/research/live/iter_8_pairs_host.py \
    --broker alpaca \
    --account-id AEGIS \
    --criteria-json '{"promote_min_sharpe": 1.0, "promote_max_dd_pct": 6.0}' \
    --capacity-ceiling-usd 25000 \
    --kill-criteria-json '{"kill_3d_dd_pct": 4.0, "kill_daily_loss_pct": 2.0, "kill_rolling_sharpe": 0.0, "kill_60d_dd_pct": 8.0}' \
    --stop-file-path /Users/weeoto/turbo-carnival/research/live/STOP_PAIRS
```

Rejection paths the framework enforces at register time (none are bypassable):
- `kill_criteria_json` missing any of the three required keys -> rejection (Lesson 10).
- `version_sha` does not resolve to a real commit in the turbo-carnival repo -> rejection.
- `capacity_ceiling_usd <= 0` -> rejection (Lesson 9).
- `broker` not in the allowlist -> rejection (Lesson 8 spirit).
- `verdict_criteria_json` does not include at least one of `promote_min_sharpe`, `promote_max_dd_pct`, `promote_min_ann_return_pct` -> rejection (no promotion path = nonsense registration).

Pass `--allow-existing` to upsert an existing strategy (e.g., to bump
`version_sha` after a host re-deploy).

The host is responsible for writing daily metrics. A typical EOD hook in the
host script looks like:

```python
subprocess.run([
    sys.executable, str(LIVE_DIR / "meta_framework.py"), "record-daily",
    "--strategy", "iter_8_pairs_coint",
    "--date", trade_date,
    "--sharpe", str(rolling_sharpe),
    "--max-dd-pct", str(rolling_max_dd_pct),
    "--ann-return-pct", str(ann_return_pct),
    "--trades", str(trade_count),
    "--win-rate", str(win_rate),
    "--broker-truth-match", "true" if positions_matched else "false",
    "--drawdown-pct-today", str(today_dd_pct),
], check=True)
```

---

## 4. Interpreting state transitions

| transition | meaning |
| --- | --- |
| `(none) -> shadow` | Initial register. Strategy is paper-trading; framework does not promote until 30 days of metrics. |
| `shadow -> tiny_live` | Auto-promote: 30+ daily metrics on file, all `promote_*` criteria pass, no broker-truth divergence in window. Tiny-live tier deploys real capital at a small ceiling (operator's discretion via the host's own size). |
| `tiny_live -> live` | Auto-promote: 14+ tiny_live daily metrics on file, criteria still pass, AND broker_truth_match has NEVER been False at any point in the strategy's history. Full ceiling. |
| `* -> killed` (auto) | One of the kill criteria fired: 3-day DD breach, daily loss cap, rolling sharpe collapse, broker-truth divergence, or 60-day DD breach. Framework touches the strategy's STOP file. |
| `* -> killed` (force) | Operator invoked `force-kill`. STOP file is touched. |
| `* -> paused` (global) | `global-kill` fired. Every active strategy is paused (operator may later reverse to shadow). |

The `reason` column on each transition row records the exact metric that fired
the decision. Always read it before acting.

Reverse a kill: there is intentionally no `unkill` CLI. To put a killed
strategy back in service, register it again with `--allow-existing
--status shadow` (and ideally a new `version_sha` reflecting whatever fix you
made). This forces a re-shadow period.

---

## 5. Global kill (panic button)

```bash
.venv/bin/python research/live/meta_framework.py global-kill --confirm \
    --reason "operator: market-wide event, flatten everything"
```

What this does:
1. For every strategy whose `status` is not already `killed` or `research`:
   - Touch the `stop_file_path` (the host's STOP file). Each host script
     checks for the STOP file on every loop iteration and halts cleanly when
     it sees one. SVXY and iter_18 hosts already have this wiring; the new
     iter_3 and iter_6 hosts must implement the same convention.
   - Flip the strategy's `status` to `paused`.
2. Append a JSON record of the event to `META_GLOBAL_KILL.log`.

The framework does NOT itself send broker cancel/flatten orders — each host
owns its own flatten logic. The framework's job is to set the unmistakable
signal (STOP file present, registry status=paused) that every host must obey.

`--confirm` is mandatory; without it the command refuses to fire.

---

## 6. Lesson enforcement summary

The framework converts the 10 documented lessons (where applicable to a registry
+ auto-promote/kill scope) into hard programmatic gates:

| # | Lesson | Where enforced |
| --- | --- | --- |
| 1 | Validate before deploy | promote-check requires >=30 daily metrics before any tiny_live promotion |
| 4 | Broker truth, not local cache | tiny_live -> live blocked if `broker_truth_match` was ever False; any False day in current window also triggers auto-kill |
| 8 | Battle-tested broker SDKs only | `broker` must be in `VALID_BROKERS` allowlist |
| 9 | Capacity bound named upfront | `capacity_ceiling_usd` NOT NULL and > 0 |
| 10 | Kill criteria written before deploy | `kill_criteria_json` NOT NULL with three required keys |

The remaining lessons (2, 3, 5, 6, 7) are properties of each individual
strategy's iter file and are enforced at the strategy-host level rather than
the registry level. The framework re-affirms them by pinning each strategy to
a `version_sha` — any change to the host's code requires a new commit and a
new registration.

---

## 7. Initialization

First-time setup (creates db, registers the four surviving strategies):

```bash
cd /Users/weeoto/turbo-carnival
.venv/bin/python research/live/meta_framework.py init --with-defaults
```

This pre-registers all four with status `shadow`:
- `svxy_sweet_spot` -> alpaca, host `research/live/svxy_sweet_spot_host.py`, STOP file `STOP_SVXY`
- `iter_18_adr` -> ibkr, host `research/live/iter_18_adr_host.py`, STOP file `KILL_SWITCH`
- `iter_3_btc_funding` -> ccxt, host `research/live/iter_3_btc_funding_host.py`, STOP file `STOP_BTC_FUNDING`
- `iter_6_nikkei_gap` -> ibkr, host `research/live/iter_6_nikkei_host.py`, STOP file `STOP_NIKKEI`

Verdict and kill criteria are taken directly from each strategy's iter_N spec
§11. The `version_sha` is pinned to the current HEAD of the turbo-carnival
repo.

To re-initialize from a clean slate (DESTRUCTIVE — wipes registry state):

```bash
rm /Users/weeoto/turbo-carnival/research/live/meta_registry.sqlite3
.venv/bin/python research/live/meta_framework.py init --with-defaults
```

The transition log lives in the same db, so a wipe loses audit history.
Snapshot the db before doing this:

```bash
cp research/live/meta_registry.sqlite3 \
   research/live/meta_registry.snapshot.$(date -u +%Y%m%dT%H%M%SZ).sqlite3
```

---

## 8. Quick reference — every CLI command

```bash
.venv/bin/python research/live/meta_framework.py init [--with-defaults] [--version-sha SHA]
.venv/bin/python research/live/meta_framework.py register --id ID --name N --version-sha SHA \
    --status STATUS --host-path PATH --broker BROKER [--account-id ID] \
    --criteria-json JSON --capacity-ceiling-usd N --kill-criteria-json JSON \
    [--stop-file-path PATH] [--notes TEXT] [--allow-existing]
.venv/bin/python research/live/meta_framework.py list
.venv/bin/python research/live/meta_framework.py metrics --strategy ID --days N
.venv/bin/python research/live/meta_framework.py record-daily --strategy ID --date YYYY-MM-DD \
    [--ann-return-pct N] [--sharpe N] [--max-dd-pct N] [--trades N] [--win-rate N] \
    [--broker-truth-match true|false] [--drawdown-pct-today N]
.venv/bin/python research/live/meta_framework.py promote-check
.venv/bin/python research/live/meta_framework.py kill-check
.venv/bin/python research/live/meta_framework.py force-kill --strategy ID --reason TEXT
.venv/bin/python research/live/meta_framework.py global-kill --confirm [--reason TEXT]
.venv/bin/python research/live/meta_framework.py dashboard-snapshot
```
