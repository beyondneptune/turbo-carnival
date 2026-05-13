# iter_3 BTC Funding Carry — Kraken Deployment

Operator-facing guide for the Kraken port of the iter_3 BTC perpetual
funding cash-and-carry strategy.

Script: `research/live/iter_3_kraken_btc_funding_host.py`
DB:     `research/live/kraken_btc_funding_decisions.sqlite3`
Kill:   `research/live/STOP_BTC_FUNDING`

This replaces the Binance + OKX two-broker deployment with a single-broker
(Kraken) deployment. The strategy parameters are unchanged and remain frozen
under the 90-day cycle rule.

---

## 1. Why Kraken-only is simpler than Binance + OKX

| Concern                          | Binance + OKX                          | Kraken-only                                              |
|----------------------------------|----------------------------------------|----------------------------------------------------------|
| KYC processes                    | 2 separate brokers, both new           | 1 broker, already verified by the user                   |
| Counterparty exposure            | Split across two unrelated entities    | One legal entity (Payward) for both legs                 |
| Paper-trading availability       | Binance testnet + OKX simulated header | Kraken Futures demo sandbox + tiny live spot ($100 cap)  |
| FTX-style fragmentation risk     | If either side halts withdrawals, hedge breaks | Both legs settle on the same balance sheet         |
| Funding cadence                  | 8h (3x/day)                            | 1h (24x/day) — faster signal reaction                    |
| Regulatory friction (US)         | Binance.US is feature-limited, OKX is US-restricted | Kraken supports US accounts incl. Kraken Futures via Payward Ventures |

The strategy backtest verdict (annualized_carry_net_pct = 7.95%, in-position
17.45%) is venue-agnostic — it depends on the BTC funding distribution, which
is similar across venues for the same underlying.

---

## 2. Enable Kraken Futures on your existing Kraken account

Your spot account is already verified. Kraken Futures lives on a sibling
domain (futures.kraken.com) operated by Payward Ventures Inc., and uses the
same login but a separate API-key namespace.

Quick steps:
1. Sign in at <https://futures.kraken.com> using your existing Kraken
   credentials. If the futures profile does not exist yet you will be
   prompted to create one.
2. Accept the futures terms (this is typically instant for already-verified
   spot accounts).
3. (Optional, for live trading later) Fund the futures wallet by transferring
   BTC/USD from spot via Kraken's internal transfer (no withdrawal).
4. **For our purposes do NOT generate live-futures API keys** — we will use
   the demo sandbox keys instead (see section 3).

Reference: <https://www.kraken.com/features/futures>

---

## 3. Get demo-futures sandbox keys (no KYC)

The demo sandbox lives at a separate domain and requires its own signup; it
is isolated from your live Kraken account and uses funny-money balances.

1. Go to <https://demo-futures.kraken.com>.
2. Click "Sign Up" and create a sandbox account. Email-only — no KYC, no SSN,
   no document upload. Use any working email.
3. Once signed in, the demo account is pre-funded with paper USD.
4. Settings -> API -> Generate New Key. Permissions: enable Order Entry,
   Cancel Orders, Read Positions, Read Wallet. Save the key + secret pair.
5. Export them as `KRAKEN_FUTURES_API_KEY` and `KRAKEN_FUTURES_API_SECRET`
   (see section 5).

The ccxt client routes to the sandbox automatically when we call
`set_sandbox_mode(True)` — base URL becomes
`https://demo-futures.kraken.com/derivatives/api/`.

---

## 4. ccxt symbol reference for Kraken

| Leg                 | ccxt exchange ID | Unified symbol  | Native code      | Notes |
|---------------------|------------------|-----------------|------------------|-------|
| BTC perpetual short | `krakenfutures`  | `BTC/USD:USD`   | `PF_XBTUSD`      | LINEAR (USD-margined), qty in BTC |
| BTC spot long       | `kraken`         | `BTC/USD`       | `XBTUSD` (raw: `XXBTZUSD`) | spot ordermin 0.00005 BTC, costmin $0.50 |

Kraken Futures also offers an **inverse** BTC perp (`PI_XBTUSD`, ccxt
`BTC/USD:BTC`) whose quantity is in USD contracts (1 contract = $1) and
which settles in BTC. We deliberately do NOT trade the inverse product —
quantity-in-BTC sizing matches the linear PF_ contract 1:1 with the spot
leg, keeping the cash-and-carry hedge exact. `resolve_perp_symbol()`
refuses to fall back to PI_XBTUSD for this reason.

ccxt handles the Kraken-specific XBT/BTC currency-code aliasing internally.

Funding-rate fetch:
```python
ex = ccxt.krakenfutures({...})
ex.set_sandbox_mode(True)
ex.load_markets()
hist = ex.fetch_funding_rate_history("BTC/USD:USD", limit=720)  # ~30 days hourly
cur  = ex.fetch_funding_rate("BTC/USD:USD")
```

Funding is paid every hour (per Kraken docs `fundingRateCoefficient = 24`,
verified empirically: 3,600,000 ms between consecutive history rows).

---

## 5. Required environment variables (4 vars)

```bash
# Kraken Futures demo sandbox (paper money, no real-funds risk)
export KRAKEN_FUTURES_API_KEY="..."     # from demo-futures.kraken.com
export KRAKEN_FUTURES_API_SECRET="..."

# Kraken Spot (REAL account; hard-capped at $100 per spot trade in code)
export KRAKEN_SPOT_API_KEY="..."        # from kraken.com -> Security -> API
export KRAKEN_SPOT_API_SECRET="..."
```

For the Kraken Spot key permissions, the minimum set is:
- Query Funds
- Query Open Orders & Trades
- Query Closed Orders & Trades
- Create & Modify Orders
- Cancel & Close Orders

Do **not** enable Withdraw — the strategy never withdraws.

If any of the 4 vars are unset, the script logs a warning and continues in
read-only mode (signals + decisions written, no orders placed).

---

## 6. Deployment commands

All commands assume cwd is `/Users/weeoto/turbo-carnival`.

### 6.1 First run — dry-run, no orders, no money at risk

```bash
.venv/bin/python research/live/iter_3_kraken_btc_funding_host.py --dry-run
```

This pulls funding history, computes the signal, logs decisions to the SQLite
audit DB, and synthesizes fills at last-ticker. Use this to confirm the
funding-rate fetch and rolling-30 calculation are working end-to-end before
exposing real keys.

### 6.2 Paper-shadow with futures sandbox + tiny live spot

```bash
.venv/bin/python research/live/iter_3_kraken_btc_funding_host.py --confirm
```

- Perp leg runs on the demo sandbox (paper money).
- Spot leg runs on your real Kraken account but HARD-CAPPED at $100 USD
  per trade. The script automatically shrinks both legs to the
  spot-capped quantity so the hedge stays balanced.
- First 5 trades require interactive y/N confirmation.
- A background tmux/caffeinate wrap is recommended for multi-day runs:

```bash
caffeinate -i tmux new-session -d -s iter3-kraken \
  "/Users/weeoto/turbo-carnival/.venv/bin/python \
   /Users/weeoto/turbo-carnival/research/live/iter_3_kraken_btc_funding_host.py \
   --confirm 2>&1 | tee -a research/live/iter_3_kraken_btc_funding_host.console.log"
```

### 6.3 Skip the 30-day startup history check (operator override)

```bash
.venv/bin/python research/live/iter_3_kraken_btc_funding_host.py \
  --skip-startup-check --dry-run
```

### 6.4 Emergency flatten of all open positions

```bash
.venv/bin/python research/live/iter_3_kraken_btc_funding_host.py --flatten-now
```

Or, while the host is running, simply:

```bash
touch /Users/weeoto/turbo-carnival/research/live/STOP_BTC_FUNDING
```

The host polls for that file every 60 seconds and will flatten + exit with
code 4 when found.

---

## 7. Safety envelope (unchanged from Binance version)

| Kill condition                          | Threshold           |
|-----------------------------------------|---------------------|
| Perp/spot basis divergence              | > 2% (200 bps)      |
| Either Kraken endpoint unreachable      | > 10 minutes        |
| NAV drawdown from peak                  | > 8%                |
| BTC 1h move (pause new entries only)    | > 10%               |
| Kill file present                       | flatten + exit 4    |
| Spot leg per-trade notional             | hard capped at $100 |
| First 5 trades                          | require `--confirm` |

A kill always flattens both legs before exit. The host exits non-zero so a
process supervisor (e.g. tmux + launchd watchdog) can page on it.

---

## 8. Reconciliation

Hourly reconciliation rows are written to the `reconciliations` table in
`kraken_btc_funding_decisions.sqlite3`. Once per week the operator should:

1. Query the table:
   ```sql
   SELECT day_utc, funding_collected_usd
   FROM daily_summary
   WHERE day_utc >= date('now', '-7 day');
   ```
2. Cross-check against Kraken Futures' funding transfer history on
   `https://futures.kraken.com -> History -> Funding`.
3. The two columns should match within rounding (sub-$0.01).

Any sustained discrepancy is a bug; halt with `--flatten-now` and
investigate before resuming.

---

## 9. File map

| File                                           | Purpose                                  |
|------------------------------------------------|------------------------------------------|
| `iter_3_kraken_btc_funding_host.py`            | The host script (this README's subject)  |
| `kraken_btc_funding_decisions.sqlite3`         | Audit DB (decisions, orders, positions, reconciliations, daily_summary) |
| `kraken_btc_funding_nav_state.json`            | Persistent NAV/peak/drawdown state       |
| `kraken_btc_funding_confirm_counter.json`      | Tracks how many trades have used `--confirm` |
| `STOP_BTC_FUNDING`                             | Kill file (operator-created)             |
| `iter_3_kraken_btc_funding_host.log`           | Rolling log file                         |

---

## 10. Comparison vs the Binance+OKX script

The Kraken script preserves the strategy logic, kill-switch logic, atomic
two-leg coordination (perp first, spot second, perp-reverse-on-spot-fail),
and SQLite audit trail. The deltas are:

- One client constructor per leg (krakenfutures sandbox / kraken spot),
  no `--spot-exchange` argument.
- `FUND_PER_DAY = 24` (not 3) since Kraken funds hourly.
- `next_funding_window_ts()` returns top-of-hour rather than 00/08/16 UTC.
- `cap_spot_qty()` enforces the $100 spot hard cap and shrinks both legs
  symmetrically so the hedge stays exact.
- Reconciliation table columns renamed: `binance_ok/okx_ok` ->
  `futures_ok/spot_ok`.
- Reconciliation tries both `BTC` and `XBT` balance keys to handle Kraken's
  legacy XBT alias.
