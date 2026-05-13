# iter_3 BTC Perpetual Funding Cash-and-Carry — Paper Shadow Deployment

Production paper-shadow host for the `iter_3` strategy: short Binance BTC
perpetual + long OKX BTC spot, harvesting positive funding when the
30-period rolling annualized funding APR is hot (> 10%) and exiting when it
cools (< 4%).

Backed by `research/results/iter_3_summary.json` (PASS: 7.95% annualized
net carry over 1,957.7 days; in-position 17.45% APR; 11 positions cycled).

| Component | Path |
| --- | --- |
| Host script | `/Users/weeoto/turbo-carnival/research/live/iter_3_btc_funding_host.py` |
| Backtest source | `/Users/weeoto/turbo-carnival/research/scripts/iter_3_backtest.py` |
| Backtest verdict | `/Users/weeoto/turbo-carnival/research/results/iter_3_summary.json` |
| Decisions DB | `/Users/weeoto/turbo-carnival/research/live/btc_funding_decisions.sqlite3` |
| Log | `/Users/weeoto/turbo-carnival/research/live/iter_3_btc_funding_host.log` |
| Kill file | `/Users/weeoto/turbo-carnival/research/live/STOP_BTC` |
| NAV state | `/Users/weeoto/turbo-carnival/research/live/btc_funding_nav_state.json` |
| Confirm counter | `/Users/weeoto/turbo-carnival/research/live/btc_funding_confirm_counter.json` |

---

## 1. Why two paper venues?

The strategy needs:
- A USDT-settled BTC **perpetual** with funding (for the short leg). Binance
  testnet supports this natively.
- A USDT-quoted BTC **spot** book (for the long hedge). The Binance testnet
  spot environment is shared with futures and has thin fills. We therefore
  use a **separate venue** for the spot leg so cross-venue basis risk is
  modeled honestly:
  - Default: **OKX paper trading** (`x-simulated-trading: 1` header).
  - Fallback: **Bybit testnet** (`set_sandbox_mode(True)`), passed via
    `--spot-exchange bybit`.

Do not point the script at a live (real-money) account without a full risk
review. The hardcoded venue builders call `set_sandbox_mode(True)` or the
OKX simulated header; if you bypass them you are no longer in paper mode.

---

## 2. Get a Binance testnet account

1. Visit https://testnet.binance.vision/ (this is the **futures**
   testnet — confirm the URL says `vision`, not `binance.com`).
2. Sign in with GitHub.
3. Open `Account` -> `API Key` -> `Generate HMAC_SHA256 Key`.
4. Copy the API key and secret. Re-keys are free; rotate whenever you want.
5. Use the testnet wallet faucet (top of the page) to mint USDT/BTC for the
   paper account.

Note: Binance has two testnets — `testnet.binance.vision` (spot) and
`testnet.binancefuture.com` (futures). ccxt's `set_sandbox_mode(True)` with
`defaultType: 'future'` routes to the futures testnet automatically. Verify
by running with `--dry-run` first and watching the log for the futures URL
(`fapi`/`dapi`).

---

## 3. Get an OKX paper trading account

1. Sign up at https://www.okx.com/.
2. From the top-right account menu, switch to **Demo trading**.
3. In demo mode, open `API` -> `Create API V5 key`.
4. Generate API key, secret, and **passphrase** (OKX requires all three).
5. Mint demo USDT from the demo wallet.

OKX paper is routed via the `x-simulated-trading: 1` HTTP header. ccxt sets
this for you when the header is configured in the client; the host script
does this in `build_okx_paper`.

---

## 4. (Fallback) Bybit testnet

If OKX paper is unreachable:

1. Visit https://testnet.bybit.com/.
2. Sign in with email.
3. `API Management` -> `Create New Key` -> system-generated.
4. Mint test USDT from the faucet.
5. Run the host with `--spot-exchange bybit`.

---

## 5. Required environment variables

```bash
# Binance perp leg (futures testnet)
export BINANCE_TESTNET_API_KEY="..."
export BINANCE_TESTNET_API_SECRET="..."

# OKX spot leg (paper / simulated trading)
export OKX_PAPER_API_KEY="..."
export OKX_PAPER_API_SECRET="..."
export OKX_PAPER_API_PASSPHRASE="..."

# Optional fallback (only needed if --spot-exchange bybit)
export BYBIT_TESTNET_API_KEY="..."
export BYBIT_TESTNET_API_SECRET="..."
```

Put these in `~/.zshrc` or load them from a `.env` in the supervisor
wrapper. The host script logs a warning and runs read-only if any leg's
credentials are missing — it will still compute signals and write to the
decisions DB.

Python dependencies:

```bash
pip install ccxt pandas numpy
```

---

## 6. Strategy parameters (frozen, copied from `iter_3_backtest.py`)

| Param | Value |
| --- | --- |
| Perp symbol | `BTC/USDT:USDT` (Binance) |
| Spot symbol | `BTC/USDT` (OKX paper / Bybit testnet) |
| Rolling window | 30 funding cycles |
| Entry threshold | rolling-30 APR > 10% |
| Exit threshold | rolling-30 APR < 4% |
| Round-trip cost budget | 30 bps total (15 bps per side) |
| Funding schedule | 00:00 / 08:00 / 16:00 UTC |
| Position size | 20% NAV per entry |
| Max open positions | 5 (= 100% NAV) |
| Default paper NAV | $100,000 |

If you change any of these in the host, also change them in
`iter_3_backtest.py` and re-run the backtest. The host's startup
verification will refuse to launch if recent funding history is
incompatible with the backtest's signal density.

---

## 7. Deployment commands

### Smoke test (no orders)

```bash
.venv/bin/python research/live/iter_3_btc_funding_host.py --dry-run
```

This still hits Binance for funding history and tickers, but synthesizes
fills at the last ticker price instead of calling `create_order`.

### First live (paper) trade — interactive gate

```bash
.venv/bin/python research/live/iter_3_btc_funding_host.py --confirm
```

The script will prompt for `y/N` on each of the first **5 trades**.
Trade-count is persisted in `btc_funding_confirm_counter.json`; once 5 are
done the gate auto-releases.

### Long-running paper deployment (after first 5 trades)

```bash
nohup caffeinate -i .venv/bin/python \
  research/live/iter_3_btc_funding_host.py \
  >/dev/null 2>&1 &
```

Or under tmux:

```bash
tmux new -d -s btc_funding \
  ".venv/bin/python research/live/iter_3_btc_funding_host.py"
```

### Spot venue fallback

```bash
.venv/bin/python research/live/iter_3_btc_funding_host.py \
  --spot-exchange bybit
```

### Force flatten everything

```bash
.venv/bin/python research/live/iter_3_btc_funding_host.py --flatten-now
```

Sends reduce-only closes for both legs on every open position, then exits.
Safe to run after a kill-switch fire.

---

## 8. Kill switches

The host fails CLOSED — every kill path flattens both legs and exits with
a non-zero code so the supervisor can page.

1. **Kill file**: `touch research/live/STOP_BTC`. Picked up within 60s.
   Exits with code 4.
2. **SIGINT / SIGTERM**: clean shutdown, NAV state saved, positions left
   open for the next session to reconcile.
3. **Hard kills** (auto-flatten + exit 5):
   - Combined perp-vs-spot basis diverges > **2%** while in position.
   - Either exchange unreachable > **10 minutes** while in position.
   - NAV drawdown vs. peak > **8%**.
4. **Auto-pause** (no flatten, just skip new entries this cycle):
   - BTC moves > **10%** in the last 60 minutes (vol event).
5. **Operator gate**: first 5 trades require `--confirm`.

---

## 9. Reconciliation and storage

Every cycle (~1h) records:

- `decisions(ts, cycle_type, funding_rate, roll_apr, signal, n_open, nav,
  note)` — every signal evaluation.
- `orders(...)` — every order placed, including reverses on partial fills.
- `positions(...)` — open / close lifecycle per position.
- `reconciliations(ts, perp_position_btc, spot_position_btc, basis_pct,
  binance_ok, okx_ok, note)` — actual vs. expected leg sizes and venue
  health every cycle.
- `daily_summary(day_utc, funding_collected_usd, basis_drift_max_pct,
  perp_pnl_usd, spot_pnl_usd, net_pnl_usd, n_open_eod, note)` —
  written shortly after 00:00 UTC.

Query examples:

```sql
-- last 24h of signals
SELECT ts_utc, signal, roll_apr, n_open
FROM decisions
WHERE ts_utc > datetime('now','-1 day')
ORDER BY ts_utc DESC;

-- basis drift recent
SELECT ts_utc, basis_pct FROM reconciliations
WHERE ts_utc > datetime('now','-1 day')
ORDER BY ts_utc DESC;
```

---

## 10. Startup verification

On launch the host pulls the last ~30 days (90 cycles + warmup) of Binance
funding history and counts how many cycles had annualized funding > 10%.
The backtest had 2,768 of 5,874 cycles above 10% — about 47%. The host
expects roughly `0.47 * 90 = 42 +/- 30%` such cycles in the recent window.

- If the count is way **below** expectation, the host logs a soft-fail and
  continues in WATCH mode (no entries until the next cycle pushes signal
  density back into range).
- If the count is way **above** expectation, the host logs a soft-warn
  and continues (a hot regime is consistent with the strategy thesis).
- If the funding-history pull is empty or returns < 31 rows, the host
  exits with code 3 (cannot run without a baseline).

Override with `--skip-startup-check` for replay / surgery sessions.

---

## 11. Known gotchas

### Funding window timing
- Binance funding is paid at 00:00 / 08:00 / 16:00 UTC. To collect at a
  window, the perp short must be open BEFORE the window timestamp.
- The host detects "borderline" entries (rolling APR is just above 10% AND
  the next funding window is < 15 minutes away) and **skips** them. The
  rationale: a marginal entry that misses the next funding payment pays
  fees without harvesting anything.

### Two-exchange race conditions
- Atomic open: the host fills the perp leg first, then the spot leg. If
  the spot leg fails or times out within 30 seconds, the host immediately
  reverses the perp short with a reduce-only buy. Failure is logged as
  `aborted` in the `positions` table with reason `spot_fail_reversed`.
- Atomic close: symmetric — perp buy first, then spot sell. If one side
  fails the position is marked `aborted` (status `..._partial` in
  `exit_reason`) and the operator is expected to flatten manually before
  the next cycle. Reconciliation will flag the drift.

### Exchange outages
- Each cycle records `binance_ok` / `okx_ok` flags. If either venue is
  silent for > 10 minutes while a position is open, the host fires the
  outage kill switch and flattens what it can reach.
- A venue being slow but reachable (returning stale tickers) is **not**
  caught — only complete unreachability. Watch the log for
  `fetch_ticker ... failed` retries.

### Basis blowouts
- The strategy assumes a tight perp/spot basis. A 2% divergence is large
  enough that the carry math no longer holds; the host kills above that.
- During exchange-specific outages or maintenance windows the basis can
  briefly spike. If it triggers a kill, restart with `--flatten-now`
  first, then resume.

### NAV state file
- `btc_funding_nav_state.json` persists NAV and peak NAV. Delete it ONLY
  during clean-room replays — otherwise the drawdown kill switch will
  reset its baseline and may mask a real drawdown.

### Position bookkeeping
- Funding collected/paid is **not** automatically reconciled against the
  exchange's actual funding fee transfers. The daily summary uses
  `positions.funding_collected_usd`, which is updated by the host based
  on Binance's funding-rate stream times position size. For tighter
  audits, cross-check Binance's "Funding Fee History" page weekly.

### Symbol formats
- ccxt unified symbol for Binance USDT-margined perp: `BTC/USDT:USDT`.
- For Bybit testnet linear perp: also `BTC/USDT:USDT` (unified).
- For OKX paper spot: `BTC/USDT`.
- If you change exchanges, run `--dry-run` first and verify the host
  resolves the symbols against `load_markets()` without errors.

### Sandbox limits
- Binance testnet rate-limits aggressive polling. The host runs one
  decision cycle per hour, which is well under the limits. If you reduce
  `RECONCILE_SECS` for testing, expect 429s.
- OKX paper occasionally returns empty order books outside US business
  hours. The host treats empty tickers as `binance_ok=False` /
  `okx_ok=False` for that cycle; ten consecutive cycles trigger the
  outage kill.

---

## 12. Health-check checklist (before first live-paper trade)

- [ ] `BINANCE_TESTNET_API_*` and `OKX_PAPER_API_*` env vars set.
- [ ] `--dry-run` runs cleanly for at least one full cycle (1h).
- [ ] Decisions DB written; query returns at least one row in `decisions`.
- [ ] Reconcile row shows both `binance_ok=1` and `okx_ok=1`.
- [ ] Startup verification logs PASS or soft-warn (not exit 3).
- [ ] You have read sections 8 and 11 of this README.
- [ ] Kill file location is writable from your shell.
