# iter_6 Nikkei 225 Gap-Continuation — Paper-Shadow Deployment

Production paper-shadow harness for the iter_6 Nikkei gap-continuation
strategy (daily-proxy version), driven by the iter_6 threshold sweep:
**1.0 % overnight gap** is the unambiguous Sharpe and risk-adjusted winner
(OOS Sharpe 2.51, ann return 29.1 %, max DD −5.40 %, 18.9 % days in market).

Host script: `research/live/iter_6_nikkei_host.py`

---

## 1. IBKR data subscription requirements

The script supports two execution venues. Default is CME for retail
accessibility.

| Venue            | Symbol  | Currency | IBKR data subscription needed                                 | Approx. cost |
| ---------------- | ------- | -------- | ------------------------------------------------------------- | ------------ |
| **CME (default)**| `NKD`   | USD      | "CME Real-Time" (Globex futures) — non-pro retail tier        | ~$10 / month |
| **OSE**          | `N225M` | JPY      | "OSE Top of Book" + Tokyo market data permission              | ~$11 / month |

Notes:
- yfinance (`^N225` cash index) is used only for the daily-gap **signal**;
  no IBKR subscription is needed for the signal itself.
- IBKR will also require the underlying paper account to have futures
  trading permissions (Futures + Futures Options on CME for the NKD path).
- NKD multiplier is **$5 per index point**. NK225M is **JPY 100 per point**.

---

## 2. Environment variables

| Variable        | Default     | Meaning                                       |
| --------------- | ----------- | --------------------------------------------- |
| `IB_HOST`       | `127.0.0.1` | TWS / IB Gateway host                          |
| `IB_PORT`       | `7497`      | 7497 = paper TWS, 4002 = paper IB Gateway      |
| `IB_CLIENT_ID`  | `26`        | Unique client_id (avoid collisions with other hosts) |

Override at the CLI with `--host`, `--port`, `--client-id` if needed.

---

## 3. Deployment commands

### 3.1 First-time pre-flight (no orders)
```bash
# 1. Verify yfinance + backtest divergence (no IBKR connection).
.venv/bin/python research/live/iter_6_nikkei_host.py --verify-only

# 2. Dry-run a full cycle against the paper account (orders skipped).
.venv/bin/python research/live/iter_6_nikkei_host.py --venue cme --dry-run
```

### 3.2 Manual paper trade (CME / NKD)
Run at the **US cash open** (~09:30 ET); the script reads `^N225` daily
OHLC to derive the signal from the prior Tokyo close to today's Tokyo open
(yfinance updates the `Open` column after Tokyo opens, which precedes US
hours, so the signal is settled by US open).

```bash
# First 5 live trades require the --confirm gate.
.venv/bin/python research/live/iter_6_nikkei_host.py \
    --venue cme --confirm --hold-secs 21000
```
`--hold-secs 21000` (~5h50m) covers the US regular session 09:30 → 16:00 ET.

### 3.3 Manual paper trade (OSE / N225M)
Run at **09:00 JST**; exit just before **15:00 JST**.
```bash
.venv/bin/python research/live/iter_6_nikkei_host.py \
    --venue ose --confirm --hold-secs 21000
```
The script accounts for the 11:30–12:30 JST lunch break only insofar as the
hold-secs window straddles it; no orders are placed during lunch because
the cycle is entry-at-open / exit-at-close (no intraday touchpoints in the
daily-proxy version).

### 3.4 Recommended cron (CME default)
```cron
# At 09:30 ET (13:30/14:30 UTC depending on DST) Mon-Fri.
30 13 * * 1-5 cd /Users/weeoto/turbo-carnival && \
    caffeinate -i .venv/bin/python research/live/iter_6_nikkei_host.py \
    --venue cme --confirm --hold-secs 21000 \
    >> research/live/iter_6_cron.log 2>&1
```

### 3.5 Force-flatten utility
```bash
.venv/bin/python research/live/iter_6_nikkei_host.py --venue cme --flatten-now
```

---

## 4. Kill switches

All kills fail closed (non-zero exit so a supervisor can page):

| Trigger                                       | Exit code | Where                                  |
| --------------------------------------------- | --------- | -------------------------------------- |
| `touch research/live/STOP_NIKKEI`             | 8         | checked on startup + each cycle        |
| Intraday loss > 2 % NAV                       | 7         | post-trade kill in `check_kill`        |
| 5 consecutive losing trades                   | 7         | post-trade kill in `check_kill`        |
| Daily loss cap > 1.5 % NAV                    | (no entry)| blocks new entries; existing trade exits at close |
| First 5 trades without `--confirm`            | 6         | confirm-gate enforcement               |
| Startup signal-count divergence > 30 %        | 3         | `verify_against_backtest`              |
| IBKR connection failure                       | 9         | `IBConnection.connect`                 |
| SIGINT / SIGTERM                              | 130       | graceful shutdown + flatten            |
| Unhandled exception                           | 1         | flatten attempt, then exit             |

State files:
- `research/live/nikkei_nav_state.json` — daily NAV, P&L, consecutive losses.
- `research/live/nikkei_confirm_counter.json` — first-5-trades counter.
- `research/live/nikkei_decisions.sqlite3` — full audit trail (evaluations,
  orders, fills, EOD summaries, kill events).

---

## 5. Upgrade path to 5-min real OR-breakout

The host is structured so the signal-generation function is pluggable. To
upgrade from the daily proxy to the spec'd 5-min opening-range breakout:

1. Acquire 5-min Nikkei 225 mini futures bars (IBKR direct OSE feed,
   Polygon-jp, or a paid vendor — yfinance does not deliver these for the
   pre-2024 backtest window).
2. Implement the body of `opening_range_breakout_signal(bars_5min, ...)`
   in `iter_6_nikkei_host.py` (currently `NotImplementedError`). Suggested
   logic is in the docstring: define OR over the first N 5-min bars after
   09:00 JST, enter on breakout, exit at 11:25 JST or close.
3. In `main()`, change the `CycleConfig.signal_fn` argument from
   `daily_gap_signal` to `opening_range_breakout_signal`. The host's
   order routing, risk envelope, kill switches, DB, NAV bookkeeping, and
   IBKR connection logic do not change.
4. Update the cron schedule to fire at 09:00 JST (not US open) for the
   true intraday strategy.
5. Re-run `--verify-only` against the 5-min logic before going live; the
   `verify_against_backtest` helper will need its `expected_signal_count`
   updated to reflect the 5-min OR backtest results.

---

## 6. Known gotchas

- **Tokyo holidays** — `^N225` simply has no bar on TSE holidays
  (e.g. Golden Week early May, Culture Day Nov 3, year-end Dec 31–Jan 3).
  The script will see an unchanged `Open`/`Close` from yfinance on those
  days and the signal will degenerate to gap = 0 → no trade. Manual
  verification of Tokyo holiday calendar is recommended before running on
  borderline dates.
- **CME holiday closures** — NKD on Globex still respects CME holidays
  (Christmas, July 4, Thanksgiving). On those days IBKR will reject the
  order; the script will log the error and the cycle will exit non-zero.
- **OSE lunch break (11:30–12:30 JST)** — only relevant if/when you swap
  in the 5-min OR strategy. The daily-proxy version doesn't touch the book
  intraday, so the lunch break is invisible.
- **JPY P&L on OSE leg** — the script converts NK225M P&L to USD using a
  flat USD/JPY = 150 for bookkeeping. For accurate NAV, reconcile from
  IBKR's `accountValues()` overnight rather than relying on the
  intra-cycle estimate.
- **yfinance latency** — `^N225` daily bars sometimes lag ~30–60 minutes
  after the Tokyo open. If the script runs *before* the bar has been
  published, the last-row `Open` will be stale (yesterday's close). Run
  the US-open cron no earlier than 09:30 ET to give yfinance time to
  publish the new bar.
- **NKD vs N225 cash index basis** — the signal uses cash-index gap,
  but the trade is in NKD (which prices off Globex's continuous overnight
  session). Basis is normally tight (< 0.5 %) but can blow out on FOMC /
  BoJ days; this is part of the strategy's idiosyncratic risk and was
  implicitly absorbed by the 5 bps cost assumption in the backtest.
- **Daily-proxy realism gap** — the daily proxy assumes you can hit the
  Tokyo open and Tokyo close prices exactly. In live execution on CME
  NKD, "open" and "close" are US-session waypoints, not Tokyo-session
  ones, so slippage relative to the backtest will be larger than 5 bps
  on volatile days. Monitor `nikkei_decisions.sqlite3` for fill-vs-signal
  drift in the first 30 cycles.
- **Pro / non-pro CME pricing** — IBKR will re-classify your account if
  you derive fee income from the data. Keep the script labelled as
  research/personal use only.
- **Multi-host client_id collisions** — if you also run the iter_18 ADR
  host (`client_id=27` per its README) on the same TWS, leave this script
  at the default `client_id=26` and never share IDs.
