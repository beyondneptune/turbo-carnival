# iter_18 ADR Dual-Listing Arbitrage — Paper Shadow Deployment

Production paper-shadow host for the cleaned-up `iter_18_us_hk_dual_listing_arbitrage`
strategy. Backed by `research/results/iter_18_cleanup_summary.json`
(PASS: mean_bps_net 65.79 over 1,543 trades, 6.36 years).

| Component | Path |
| --- | --- |
| Host script | `/Users/weeoto/turbo-carnival/research/live/iter_18_adr_host.py` |
| Backtest source | `/Users/weeoto/turbo-carnival/research/scripts/iter_18_cleanup.py` |
| Backtest verdict | `/Users/weeoto/turbo-carnival/research/results/iter_18_cleanup_summary.json` |
| Decisions DB | `/Users/weeoto/turbo-carnival/research/live/adr_decisions.sqlite3` |
| Log | `/Users/weeoto/turbo-carnival/research/live/iter_18_adr_host.log` |
| Kill flag | `/Users/weeoto/turbo-carnival/research/live/KILL_SWITCH` |
| NAV state | `/Users/weeoto/turbo-carnival/research/live/adr_nav_state.json` |
| Confirm counter | `/Users/weeoto/turbo-carnival/research/live/confirm_counter.json` |

---

## 1. Why IBKR (not Alpaca)?

Alpaca does not list HKEX equities. This strategy needs simultaneous access to
US ADRs (BABA, JD, BIDU, NTES, TCOM) **and** their HKEX ordinaries (9988,
9618, 9888, 9999, 9961). Interactive Brokers exposes all 10 instruments
through a single account with USD/HKD auto-conversion via IDEALPRO.

Account: **IBKR paper trading** (DU* account). Do NOT point this script at
live trading without a full risk review.

---

## 2. IBKR Setup

### 2.1 Install TWS or IB Gateway (paper)

1. Download IB Gateway from Interactive Brokers.
2. Log in with your paper credentials (the `DU…` account).
3. Configure API access:
   - `Configure -> Settings -> API -> Settings`
   - Enable **Active X and Socket Clients**
   - **Socket port**: 7497 (TWS paper) or 4002 (IB Gateway paper)
   - Add `127.0.0.1` to **Trusted IPs**
   - Uncheck **Read-Only API** (the host must place orders)
   - Set **Master API client ID** blank (host uses clientId=17 by default)

### 2.2 Market data subscriptions

You need entitlements for:
- US Securities Snapshot and Futures Value Bundle (or equivalent)
- Hong Kong Securities Exchange (HKEX)
- IDEALPRO FX (free with funded account)

Without HKEX data the host will log "No price tick" for the HK legs and skip
every pair. Verify before deployment by opening a quote for 9988 in TWS.

### 2.3 Environment variables

```bash
export IB_PORT=7497          # 7497 = TWS paper; 4002 = IB Gateway paper
export IB_CLIENT_ID=17       # anything 1-32 not already in use
```

Put these in `~/.zshrc` or load them from a `.env` in the cron wrapper.

### 2.4 Python deps

```bash
pip install ib_insync pandas numpy pyarrow
```

(`pyarrow` is required to read the dividend cache parquet.)

---

## 3. Strategy parameters (frozen)

These are duplicated inside `iter_18_adr_host.py` and must match the cleanup
JSON. If you change one, change the other and re-run the cleanup backtest.

| Param | Value |
| --- | --- |
| Pairs | BABA/9988 (1:8), JD/9618 (1:2), BIDU/9888 (1:8), NTES/9999 (1:5), TCOM/9961 (1:1) |
| Entry | `|deviation_bps|` > 80 |
| Exit | `|deviation_bps|` < 20 |
| Stop | `|deviation_bps|` > 250 |
| Cost budget | 30 bps round-trip |
| Ex-div blackout | ±3 sessions on EITHER leg |
| Per-pair size | 20% NAV per leg (dollar-neutral) |
| Max simultaneous pairs | 5 |

---

## 4. Deployment commands

### 4.1 First-run smoke test (dry, no orders)

```bash
cd /Users/weeoto/turbo-carnival
.venv/bin/python research/live/iter_18_adr_host.py --cycle adhoc --dry-run
```

Confirms: connection succeeds, prices arrive, decisions are logged into
SQLite without any orders being submitted. Inspect the log:

```bash
tail -100 research/live/iter_18_adr_host.log
sqlite3 research/live/adr_decisions.sqlite3 "SELECT * FROM decisions ORDER BY id DESC LIMIT 20;"
```

### 4.2 Live paper run with interactive confirmation (first 10 trades)

```bash
.venv/bin/python research/live/iter_18_adr_host.py --cycle adhoc --confirm
```

Each of the first 10 entry/close/stop actions will prompt the operator
`[y/N]:` before the broker order is submitted. After 10 confirmed trades,
the counter (`confirm_counter.json`) reaches 10 and subsequent runs proceed
non-interactively even with `--confirm` set.

### 4.3 Scheduled twice-daily cron (recommended)

The strategy runs once after HK close (~04:00 ET) and once after US close
(16:00 ET). Sample crontab (system timezone = ET):

```cron
# m h dom mon dow command
5  4  * * 1-5  /Users/weeoto/turbo-carnival/.venv/bin/python /Users/weeoto/turbo-carnival/research/live/iter_18_adr_host.py --cycle hk_close >> /Users/weeoto/turbo-carnival/research/live/cron.log 2>&1
5  16 * * 1-5  /Users/weeoto/turbo-carnival/.venv/bin/python /Users/weeoto/turbo-carnival/research/live/iter_18_adr_host.py --cycle us_close >> /Users/weeoto/turbo-carnival/research/live/cron.log 2>&1
```

If you prefer `launchd`, drop two `.plist` files in `~/Library/LaunchAgents/`
calling the same command with `StartCalendarInterval`.

### 4.4 Skip the 60-day backtest divergence check

The host re-validates the cleanup verdict on startup by comparing implied
60-day trade count vs spec rate. If the data cache is stale or you've just
moved machines, the check can refuse to start. Override:

```bash
.venv/bin/python research/live/iter_18_adr_host.py --cycle adhoc --skip-backtest-check
```

Only do this once you understand why the check failed (usually: rerun
`scripts/iter_18_cleanup.py` to refresh the cache).

---

## 5. Kill switches

### 5.1 Automatic (built-in, fail closed)

| Trigger | Action |
| --- | --- |
| Per-pair unrealised loss > 5% NAV | flatten all, write KILL_SWITCH, exit 9 |
| Total gross exposure > 100% NAV | flatten all, write KILL_SWITCH, exit 9 |
| IB disconnect > 5 minutes during market hours | flatten all if reconnect succeeds, exit 9 |
| HK leg rejected/halted after US fill | reverse the just-placed US leg |
| `KILL_SWITCH` flag file present | refuse to start (exit 4) |
| Backtest divergence > 30% | refuse to start (exit non-zero) |

All kill events are written to the `kill_events` table in
`adr_decisions.sqlite3`.

### 5.2 Manual kill (operator)

To stop the next scheduled run from doing anything:

```bash
echo '{"reason":"manual","detail":"halting for inspection"}' > /Users/weeoto/turbo-carnival/research/live/KILL_SWITCH
```

Once the flag file exists, every invocation exits 4 immediately. To resume:

```bash
rm /Users/weeoto/turbo-carnival/research/live/KILL_SWITCH
```

### 5.3 Manual flatten (now)

To close every iter_18 position right now (does not require KILL_SWITCH):

```bash
.venv/bin/python research/live/iter_18_adr_host.py --cycle adhoc --flatten
```

This will iterate every position with a symbol in our PAIRS set and submit
the offsetting market order. Use this before a scheduled corporate event,
broker maintenance window, or when you want to step out cleanly.

---

## 6. Reconciliation & monitoring

Every cycle writes to `adr_decisions.sqlite3`:

| Table | Contents |
| --- | --- |
| `decisions` | Per-pair: prices, FX, deviation, action, reason |
| `orders` | Every submitted order with fill stats |
| `positions_snapshot` | Broker portfolio snapshot at cycle start |
| `eod_summary` | NAV, open pair count, gross exposure, unrealised P&L |
| `kill_events` | Reasons for any hard kill |

Useful queries:

```sql
-- Today's decisions
SELECT ts_utc, pair, action, dev_bps, ex_div_block, reason
FROM decisions
WHERE ts_utc >= date('now')
ORDER BY id DESC;

-- Last 20 orders and their fill quality
SELECT ts_utc, pair, leg, side, qty, status, fill_qty, avg_fill_price
FROM orders
ORDER BY id DESC LIMIT 20;

-- Equity curve (one row per cycle)
SELECT ts_utc, cycle, nav_usd, open_pair_count, total_unreal_pnl
FROM eod_summary
ORDER BY id DESC LIMIT 30;
```

---

## 7. Known gotchas

- **HKEX trading halts and typhoon closures.** Black/Yellow/Red rainstorm
  signals and typhoon-8 close the cash market on no schedule. The host
  detects rejected HK fills and reverses the US leg automatically, but
  if you opened a position and HK halts intraday on a day-2 hold, the US
  leg will rebalance to no-hedge until HK reopens. Consider manual flatten.
- **ADR ratio changes.** NTES went 1:25 -> 1:5 in 2020 and the original
  iter_18 backtest baked the wrong ratio. Any future ADS-ratio amendment
  (typical 6-month notice via F-6 amendment) requires re-running the
  cleanup and editing the `PAIRS` constant in the host. Watch for SEC
  filings on each issuer.
- **Dividend cache freshness.** The ex-div blackout uses
  `research/data/cache/us_hk_pairs_dividends.parquet`. If you skip
  scripts/iter_18_cleanup.py for a quarter, the cache will miss the
  newest ex-dividends and we will trade *into* them. Re-run cleanup
  monthly at minimum.
- **Cross-currency P&L.** Per-pair P&L is computed at the *current* FX
  rate, not the entry FX. HKD/USD peg is tight (7.75-7.85) but if the
  peg ever broke we would mis-state P&L meaningfully. The 5% NAV
  per-pair kill check is conservative — it will fire sooner under FX
  shock.
- **HKEX board lots.** HK orders are rounded down to 100-share multiples
  in `compute_sizing`. For very small NAVs this can leave the dollar
  legs unbalanced (e.g., HK leg = 100 shares ~= 9% NAV while US leg
  hits 20%). Acceptable in paper, watch in real money.
- **Order submission outside market hours.** Cron at 04:00 ET runs after
  HK close and ~5.5h before US open. Market orders submitted to a closed
  exchange queue until next open. Spreads at open are wider than the
  30-bps cost budget assumes; consider whether to convert to LMT orders
  before going to real money.
- **Pacing violations.** `ib_insync` snapshot requests are slow but
  safe (one ticker every ~2s). Across 10 tickers + FX = ~22s per cycle.
  If IBKR returns "max number of tickers exceeded" we already cancel
  each subscription after reading; if it persists, reduce to 5 pairs.
- **Confirm counter is per-machine.** `confirm_counter.json` is local
  state. If you move machines, delete it to re-arm the first-10
  confirmation gate.
- **The host trusts the broker portfolio.** If you manually trade these
  symbols in the same paper account from TWS, the host will think those
  positions are its own and try to close them on next convergence. Use a
  separate paper account or a separate clientId-scoped sub-account.

---

## 8. Stop-the-world checklist

If something looks wrong (P&L blowing up, unfilled HK leg, weird FX print):

1. `touch /Users/weeoto/turbo-carnival/research/live/KILL_SWITCH` — next
   cron run will refuse to start.
2. `.venv/bin/python research/live/iter_18_adr_host.py --cycle adhoc --flatten`
   — close every iter_18 position now.
3. `tail -200 research/live/iter_18_adr_host.log` — read the recent log.
4. `sqlite3 research/live/adr_decisions.sqlite3 "SELECT * FROM kill_events ORDER BY id DESC LIMIT 5;"`
5. Fix the underlying issue, rerun `scripts/iter_18_cleanup.py` if data
   has gone stale, then `rm KILL_SWITCH` to resume.
