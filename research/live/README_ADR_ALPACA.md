# iter_18 ADR Arbitrage — Alpaca-Only Paper Shadow (HK Leg Simulated)

This is a **half-real, half-simulated** paper-shadow host for the iter_18
US/HK ADR-parity arbitrage strategy. The US ADR leg is traded on the real
Alpaca paper API; the HKEX ordinary leg is **simulated in code** using
end-of-day HK Close prices from yfinance.

Use this README to set expectations before you deploy.

---

## What this setup DOES validate

1. **Signal logic, end to end.** Parity calculation, deviation threshold
   trigger, ex-dividend blackout, entry/exit/stop rules, and pair sizing
   all run on live (or near-live) market data. If the strategy stops
   generating signals — or generates them at a 30%+ different rate than
   the cleanup backtest predicted — the host bails out on startup with a
   loud divergence message.
2. **Real US execution.** Alpaca paper executes the US ADR side using
   marketable IOC limit orders (25 bps aggression). The host records:
   - Submission time, status, fill quantity, average fill price.
   - Partial fills, cancellations, rejections.
   - Reconciliation of expected vs. actual position state at each cycle.
3. **Broker-truth reconciliation.** Every cycle reads Alpaca positions and
   account equity directly from the broker and compares them to the host's
   internal state. Any drift is logged.
4. **Operational reliability.** SQLite decision log, structured logging,
   kill switches (file-based, signal-based, in-cycle automated), graceful
   shutdown, idempotent flatten command.
5. **Reconciliation tooling.** Every order writes a row to the `orders`
   table tagged with `is_simulated` (0 for real US, 1 for virtual HK), so
   you can audit which P&L came from where after a 30-day run.
6. **Confirmation gate behaviour.** First 10 trades require interactive
   `[y/N]` approval (with `--confirm`), exactly mirroring how a live
   deployment would behave.

## What this setup DOES NOT validate

1. **HKEX execution.** The simulated HK fills are perfect, instantaneous,
   and zero-slippage. In reality you should expect:
   - **10-30 bps round-trip cost** from HKEX bid/ask spread + 0.13%
     transaction levy + stamp duty.
   - **Partial fills** on illiquid HK lines (TCOM/9961, MNSO/9896, TME/1698
     are the thin ones in our universe).
   - **Rejected orders** if you accidentally cross the +/- 9% price band on
     a fast move.
   - **HKEX halts** (trading suspensions, typhoon closures — HKEX closes
     during T8 typhoon signals and above). The sim never halts.
2. **Cross-listing operational reality.** HKEX settlement is T+2 with
   different cutoffs than the US. Real HKD funding must exist in the
   account at order time, and FX conversion has friction (~5-15 bps
   depending on broker tier). None of this is in the sim.
3. **Intraday FX risk.** The sim uses the 7.80 USD/HKD peg. The peg
   actually trades inside a +/- 0.5% band (7.75 - 7.85). On a typical 80
   bps signal, a 50 bps adverse FX move while the trade is open can wipe
   62.5% of the gross edge before HK execution costs.
4. **Timing precision.** Simulated HK fills use the most recent
   **end-of-day** HK Close from yfinance. Real intraday execution at HK
   open (or via the after-hours auction) can give very different prints,
   especially around US earnings releases that occur during the HK trading
   day.
5. **Liquidity-cost combined edge.** The combined P&L number in the EOD
   summary is **structurally optimistic** by an unknown amount — likely
   20-50 bps per round-trip from HK costs alone, plus episodic tail risk
   from halts and partial fills. **DO NOT extrapolate it to expected live
   performance.**

## Why this is acceptable for a 30-day paper shadow

The single biggest unknown when going from "PASS backtest" to "live
deployment" is **signal stability + US execution + reconciliation**. That
risk surface has three failure modes that are silent killers if not
caught early:

- **Signal drift**: the realised parity-deviation distribution looks
  different from what the backtest assumed (e.g. due to a structural change
  in HK trading hours, ADR ratio change, or BABA splitting in 2025). This
  shows up as trade frequency divergence and is caught by the startup
  `verify_backtest_divergence` check.
- **US execution friction**: marketable IOC orders don't actually fill at
  the snapshot mid, BABA/JD/etc. have wide spreads pre/post-market, and
  Alpaca's paper engine has its own quirks. This is exactly what the real
  Alpaca leg measures.
- **Reconciliation bugs**: the host thinks it has X shares of BABA, but
  Alpaca says it has Y. If you can't tie this out cleanly with a paper
  account where you control all the variables, you cannot run live.

These three risks compose 80% of the operational tail. The HK leg's
execution friction is real but **secondary, well-bounded, and
quantitatively known** (10-30 bps per round-trip is the typical range
across HKEX prime brokers). It's much cheaper to characterise the HK
risk later — when you wire it up — by comparing a few weeks of real HK
fills against this sim's predictions, than to delay the entire signal
deployment waiting for an HK account.

In short: this setup pins down the **unknown** unknowns (signal stability
+ US ops) cheaply and quickly, so when you add HK execution you can
attribute any P&L divergence specifically to the HK side.

---

## Migration path to real HK execution

When you're ready to upgrade from "HK simulated" to "HK real broker,"
the swap is structured so you can do it without touching the signal,
sizing, or risk-management code.

### Option A — Interactive Brokers (recommended for retail-to-pro path)

IBKR Pro supports HKEX with native HKD funding and decent fees (0.025% +
HKD 18 minimum per side, plus stamp duty). The existing
`iter_18_adr_host.py` file in this same directory **is already** the IBKR
version of this strategy. Concrete swap-in steps:

1. Open and fund an IBKR Pro account (Universal account with USD + HKD
   sub-accounts). Use the same NAV as the Alpaca paper for clean A/B
   comparison.
2. Run `iter_18_adr_host.py --dry-run --cycle us_close` for one week to
   verify IBKR market-data subscriptions are correct (HKEX Top of Book is
   required: ~$10/month for HK retail).
3. Run `iter_18_adr_host.py --confirm --cycle us_close` for the first 10
   trades to manually approve real HK fills.
4. Compare the IBKR fill log (`adr_decisions.sqlite3.orders`) against this
   Alpaca host's `adr_alpaca_decisions.sqlite3.orders` table, **filtered
   to the HK_SIM leg**. The delta tells you the realised HK execution cost
   to subtract from your sim-derived edge expectation.

### Option B — Futu / Tiger / Hong Kong prime

Both Futu Securities and Tiger Brokers (Hong Kong entities) have REST
APIs with similar shape to Alpaca's. Swap-in steps:

1. Replace the `# --- HK leg: VIRTUAL update ---` block in
   `execute_decision()` with a real REST call to your HK broker's "submit
   order" endpoint.
2. Replace `fetch_hk_close_yf()` with the broker's intraday quote endpoint
   so you trade on real HK prices, not yfinance EOD.
3. Add an HKD account-equity reader to `fetch_nav_usd()` and combine the
   two currencies properly.
4. Delete the `VIRTUAL_HK_STATE_PATH` JSON persistence — broker-truth
   replaces it.
5. Delete the `is_simulated=True` branches in `log_order` and the warning
   banner.

### Option C — Drop the HK leg, trade US ADR only as a directional signal

If you can't get HK execution running, you can also use the parity signal
**asymmetrically**: when |dev| > 80 bps with US ADR cheaper, buy the US
ADR only (no HK short). This loses the market-neutral property and ~half
the historical Sharpe but is implementable on Alpaca alone. Backtest this
variant first (it's NOT what iter_18_cleanup tested — the cleanup
strategy is fully pair-neutral).

---

## Files this host writes

- `iter_18_adr_alpaca_host.log` — rolling log (append mode).
- `adr_alpaca_decisions.sqlite3` — all decisions, orders, positions,
  EOD summaries, kill events. Schema in the host file.
- `adr_alpaca_nav_state.json` — last known account NAV.
- `adr_alpaca_virtual_hk_positions.json` — sim HK position state. Lives
  on disk between cycles because the sim has no broker to query.
- `adr_alpaca_confirm_counter.json` — count of trades that passed the
  interactive confirm gate.
- `STOP_ADR_ALPACA` (kill file) — created automatically on hard kill;
  delete manually after investigating to re-arm the host.

## Schedule

The strategy is twice-daily (HK close + US close). Suggested cron on the
deployment host:

```
# HK close cycle (~04:00 ET = 16:00 HK in winter / 03:00 ET in summer)
0 4 * * 1-5  /Users/weeoto/turbo-carnival/.venv/bin/python \
    /Users/weeoto/turbo-carnival/research/live/iter_18_adr_alpaca_host.py \
    --account AEGIS --cycle hk_close >> ~/cron_adr_alpaca.out 2>&1

# US close cycle (~16:00 ET)
0 16 * * 1-5 /Users/weeoto/turbo-carnival/.venv/bin/python \
    /Users/weeoto/turbo-carnival/research/live/iter_18_adr_alpaca_host.py \
    --account AEGIS --cycle us_close >> ~/cron_adr_alpaca.out 2>&1
```

For paper shadow, the US close cycle is the more important one — that's
when fresh US prices and HK Closes are both available within the same
day.

## How to read the EOD summary

Each EOD row in `eod_summary` has three P&L columns:

- `real_us_pnl` — actual unrealised P&L on Alpaca, broker-reported.
  **Trust this.**
- `sim_hk_pnl` — virtual HK P&L from sim. **Treat as upper bound.**
- `total_combined_pnl` — sum of the above. **Structurally optimistic by
  ~20-50 bps per round-trip.** Use the *shape* (mean, vol, drawdown,
  Sharpe) but discount the absolute level by your assumed HK cost. To
  estimate realistic edge, subtract `0.0030 * n_trades * avg_notional`
  (30 bps round-trip x number of trades x average pair notional).

## Acceptance criteria for moving to live

Before swapping the HK sim out for a real HK broker, this paper shadow
should hit all of:

1. Run continuously for 30 days with zero unhandled exceptions.
2. Generate at least 10 closed trades on the US leg (60-day backtest rate
   predicts ~20-25 per universe-core month).
3. Zero broker-truth reconciliation drift between Alpaca-reported
   positions and the host's expected state.
4. No hard kills, no manual overrides via the kill file.
5. Realised US fill quality within 50 bps of the snapshot mid on at least
   80% of trades (i.e. the marketable IOC strategy works on this universe).
6. Backtest divergence stays below the 30% threshold throughout the run.

If all six pass, you've validated the signal + US execution + reconciliation
risk surface. Time to wire up HK.
