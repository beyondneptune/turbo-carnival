# iter_6 Nikkei Gap Continuation — Japan ETF Proxy (Alpaca)

Honest, plain-English README for `iter_6_nikkei_etf_host.py`. This is the
**proxy** deployment of the iter_6 Nikkei gap-continuation strategy. The
**cleaner** deployment is `iter_6_nikkei_host.py` (IBKR -> NKD/NK225M futures);
use it instead the moment you have a real futures account.

---

## TL;DR

| Field | Value |
|---|---|
| Strategy | iter_6 gap continuation (1.0% threshold) |
| Signal source | `^N225` daily close-to-open via yfinance |
| Execution venue | Alpaca paper, US cash session |
| Default ETF | **EWJ** (iShares MSCI Japan; unhedged) |
| Hedged alternative | **DXJ** (WisdomTree Japan Hedged Equity) via `--etf DXJ` |
| Direction | long on gap > +1.0%, short on gap < -1.0% (if shortable) |
| Hold | one US trading day; exit at 15:55 ET |
| Sizing | 25% of NAV, max 1 position at a time |
| Backtest Sharpe (^N225) | 2.51 OOS |
| **Expected live Sharpe** | **~0.75 - 1.25** (30-50% of backtest, see below) |

---

## Why this is not as clean as NKD futures

The iter_6 backtest was developed on the `^N225` cash index daily series. NKD
(CME mini-Nikkei in USD) or NK225M (OSE mini-Nikkei in JPY) futures track that
index very tightly during the relevant Tokyo session — same instrument, same
hours. Trading EWJ or DXJ during the US cash session is a **proxy** and adds
three sources of slippage that did not exist in the backtest:

### 1. Currency drag (EWJ only)

`EWJ` holds Japanese equities denominated in JPY but is priced in USD. A 1%
move in the Nikkei combined with a 0.5% strengthening of USD vs JPY nets to
about a 0.5% EWJ move, not 1%. Daily JPY moves of 30-50 bps are routine, so on
any given signal the underlying signal can be diluted by 25-50% of its raw
size.

`DXJ` solves this by hedging JPY exposure to USD daily, at the cost of:

- **Hedge roll cost** — small but non-zero (~5-10 bps/year) drag from rolling
  the FX forwards.
- **Hedge basis** — daily hedge cannot perfectly match the timing of the
  underlying moves; small residual FX exposure remains.

**Pick DXJ if you want the cleanest Nikkei tracking. Pick EWJ if you want
liquidity (EWJ averages ~$70M/day vs DXJ ~$30M/day) and you are OK with the
currency factor adding noise.**

### 2. NAV vs. Nikkei tracking error

`^N225` is **price-weighted** (like the Dow). `EWJ` tracks the MSCI Japan
Index, which is **market-cap-weighted**. The two indices have ~0.90 daily
correlation but differ structurally:

- N225 over-weights high-priced stocks (Fast Retailing, Tokyo Electron,
  Advantest) regardless of market cap.
- EWJ overweights mega-caps like Toyota, Mitsubishi UFJ, Keyence.

So even before currency, a 1% Nikkei move is *not* a 1% EWJ NAV move. Expect
tracking error of 20-40 bps on a 1% signal day.

`DXJ` is also MSCI-Japan-based (with the hedge overlay), so it shares this
tracking error. Both ETFs are MSCI Japan, not Nikkei 225 — there is no
liquid US-listed Nikkei-225-replicating ETF.

### 3. US-hours vs Tokyo-hours timing mismatch

This is the most important caveat.

The original signal is: **"Tokyo close at T-1 vs Tokyo open at T"**. The
implied trade is: **"continuation during Tokyo session T"**. NKD trades 23h
on Globex and NK225M trades the Tokyo session directly — both can execute
exactly when the signal predicts.

EWJ and DXJ do not trade during the Tokyo session. They trade during US
cash hours (09:30 - 16:00 ET), which begins **after** the Tokyo session
has already closed. By the time the US open arrives:

- Tokyo session T has already happened. The signal-day return is "spent".
- The ADR overnight gap (e.g. Toyota ADR pricing the JPY ordinary close)
  has already absorbed part of the move into EWJ's pre-market.

So we are not capturing the original signal at all. We are capturing the
"next Tokyo session" (effectively T+1 in the original frame) continuation
**as traded by the US session**. Empirically this still has positive
expected value because gap-day momentum tends to persist multi-day, but
the auto-correlation is weaker than the same-day relationship the backtest
measured.

---

## Expected edge degradation

Stacking the three effects above:

| Effect | Sharpe haircut |
|---|---|
| Currency drag (EWJ only) | ~25% |
| NAV tracking error | ~15-20% |
| Hours mismatch | ~25-30% |
| Multiplicative combined (EWJ) | **~0.45 - 0.55 of backtest** |
| Multiplicative combined (DXJ) | **~0.55 - 0.70 of backtest** |

Backtest Sharpe was **2.51 OOS**. Expect:

- **EWJ live Sharpe**: ~1.1 - 1.4 (still excellent, but not 2.5)
- **DXJ live Sharpe**: ~1.4 - 1.75 (cleaner)

The win rate will degrade too — the backtest's 65.7% win rate likely lands at
55-60% in practice. The strategy is still expected to be profitable; it just
won't be the same number you saw on the sweep.

**Do not assume the backtest number.** Use this host to *measure* the real
proxy edge over a 60-90 day shadow window, then decide whether to scale up,
size down, or migrate to real futures.

---

## EWJ vs DXJ — choose one

| | EWJ | DXJ |
|---|---|---|
| Currency exposure | Long JPY | Hedged (USD) |
| Issuer | iShares (BlackRock) | WisdomTree |
| Expense ratio | 0.50% | 0.48% |
| ADV (90d) | ~$70M | ~$30M |
| Tracks N225 closely | No (JPY noise) | Closer (hedge active) |
| Spread (typical) | 1-2 bps | 2-4 bps |
| Easy to borrow | Yes | Usually |
| Best for | Liquidity, low spread | Cleaner Nikkei signal |

**Default**: EWJ. Liquid, tight spreads, cheaper. Acceptable for shadow trade
where the goal is to measure edge degradation honestly.

**Switch to DXJ via `--etf DXJ`** if you are convinced the JPY factor is
distorting your P&L attribution and you want a more direct Nikkei readout.

---

## Operations

### First run (interactive confirm on first 5 trades)

```bash
/Users/weeoto/turbo-carnival/.venv/bin/python \
    /Users/weeoto/turbo-carnival/research/live/iter_6_nikkei_etf_host.py \
    --account AEGIS --confirm
```

### Cron-style (one-shot per leg)

Open-leg (decide + enter), at 09:35 ET:
```bash
/Users/weeoto/turbo-carnival/.venv/bin/python \
    /Users/weeoto/turbo-carnival/research/live/iter_6_nikkei_etf_host.py \
    --account AEGIS --once --cycle open
```

Close-leg (exit), at 15:55 ET:
```bash
/Users/weeoto/turbo-carnival/.venv/bin/python \
    /Users/weeoto/turbo-carnival/research/live/iter_6_nikkei_etf_host.py \
    --account AEGIS --once --cycle close
```

### Daemon mode (recommended)

```bash
/Users/weeoto/turbo-carnival/.venv/bin/python \
    /Users/weeoto/turbo-carnival/research/live/iter_6_nikkei_etf_host.py \
    --account AEGIS
```

The daemon runs the open-leg once after 09:35 ET and the close-leg once after
15:55 ET each weekday. Run it under `caffeinate -i tmux new -s nikkei_etf`
for resilience.

### Dry run

```bash
... --dry-run
```
Computes signal, logs decisions, writes evaluations and EOD rows, but submits
no orders.

### Currency-hedged variant

```bash
... --etf DXJ
```

### Force flatten any open position

```bash
... --flatten-now
```

### Verify only (no orders)

```bash
... --verify-only
```
Confirms recent ^N225 data is reachable and that the live gap signal count is
within 30% of the backtest expectation.

---

## Safety / Kill Switches

| Trigger | Effect | Recovery |
|---|---|---|
| Daily loss cap 1.5% NAV | Block new entries (open positions held) | Resets next session |
| Intraday loss 2% NAV | Flatten + halt | Manual restart after review |
| 5 consecutive losses | Halt new entries | Manual restart |
| `STOP_NIKKEI_ETF` file | Halt (no flatten) | `rm` the file |
| Startup divergence > 30% | Exit non-zero | Investigate; `--skip-verify` to override |
| SIGINT / SIGTERM | Graceful flatten + exit | Restart |

Manual force-flatten:
```bash
touch /Users/weeoto/turbo-carnival/research/live/STOP_NIKKEI_ETF
# then, in a new shell:
... --flatten-now
```

---

## Logged state

- Decisions DB: `research/live/nikkei_etf_decisions.sqlite3`
  - `evaluations`, `orders`, `fills`, `eod_summaries`, `kill_events`
- NAV state: `research/live/nikkei_etf_nav_state.json`
- Confirm counter: `research/live/nikkei_etf_confirm_counter.json`
- Log file: `data/logs/iter_6_nikkei_etf_host.log`

The `venue` column in the DB is always `alpaca_etf` for runs of this host.
The `etf` column distinguishes EWJ from DXJ runs.

---

## Migration path: when you get IBKR / a real futures account

The right home for iter_6 is futures. When you can run an IBKR Pro paper or
funded account:

1. Subscribe to CME data (~$10/month) — cheapest path: NKD via CME.
2. Run `iter_6_nikkei_host.py` (already written) with `--venue cme`.
3. Compare 30-day shadow P&L between `iter_6_nikkei_host.py` (NKD) and this
   ETF host. The NKD path should show ~1.5-2x cleaner Sharpe.
4. Optionally upgrade to OSE NK225M (`--venue ose`) once a JPY-funded
   sub-account is available — that gets you the original Tokyo-session
   timing the backtest measured.
5. Migrate strategy capital from the ETF account to the futures account.
6. Leave the ETF host running with a small allocation as a comparison /
   degradation monitor. The diff in Sharpe between the two hosts measures
   the proxy cost in real money — useful evidence for future research
   iterations.

The SQLite schemas in this host and `iter_6_nikkei_host.py` differ slightly
but both records have `session_date`, `gap_pct`, `signal_dir`, `pnl_usd`, and
the `venue` distinguishes them. A simple join on `session_date` after
migration gives a side-by-side proxy-vs-real comparison.

---

## Known limitations / Gotchas

- **Holiday calendar**: This host only checks weekdays. US holidays (when
  EWJ/DXJ are closed) will fail the `get_latest_trade` lookup and the leg
  will log "no_price" — safe. Tokyo holidays affect the signal (no Tokyo
  open today means yfinance returns yesterday's row; the gap reads zero
  and no trade fires). Acceptable for now; the cron wrapper can pre-filter.
- **Daylight saving**: 09:35 ET is the cron pin. ET handles DST natively
  via `zoneinfo`, so 09:35 EST and 09:35 EDT both map to "5 min after the
  US cash open."
- **Short borrow**: EWJ and DXJ are normally easy-to-borrow on Alpaca paper.
  If a particular date sees them flagged hard-to-borrow, the short submit
  will fail with a 403/422 — logged but not fatal. Negative-gap signals
  on those days will degrade to no-trade.
- **No partial fills handled specially**: IOC orders fill or cancel. If a
  fill is partial, the close-leg will see whatever quantity Alpaca reports
  in the position and close that.
- **NAV state is local**: If you wipe `nikkei_etf_nav_state.json`, the
  consecutive-loss counter resets. The Alpaca account NAV remains the
  source of truth for sizing.
