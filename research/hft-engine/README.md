# hft-engine (Predictive Selective Trading prototype)

NOT real high-frequency trading. We're not co-located. This is a v0.1 research
prototype in Rust that asks one specific question: can a latency-aware,
strictly-selective predictor combining ES futures lead-lag with SPY top-of-book
extract any edge on a 30-60 second horizon, against Alpaca's paper API, from
~100-200ms of round-trip latency in London?

Almost certainly the honest answer will be "no". That's a valid result. The
engine is designed to make that conclusion auditable.

## What it does

1. Continuously profiles round-trip latency to Alpaca every 5 seconds.
2. Every 30 seconds, pulls SPY top-of-book from Alpaca and the most recent ES=F
   1-minute bar from Yahoo Finance.
3. Maintains a single-feature online RLS predictor: `predict SPY 60s return given
   ES 1-minute return`. Cold-starts with zero weight; learns from every realized
   move (whether or not we traded on it).
4. Every 30 seconds, evaluates the "spaceship gravity assist" alignment:
   - `|predicted_bps| > 3 * current_spread_bps`
   - rolling 50-sample hit rate `> 0.55`
   - latency p99 `< 250 ms`
   - no NAV halt / regime pause
   When all four align, place a marketable-limit IOC; hold 60s; exit IOC.
5. Persists everything (decisions, orders, latency samples, NAV snapshots,
   regime events) to a SQLite database.

## Layout

```
src/
  main.rs        daemon, signal handlers, decision loop
  config.rs      env + YAML config
  latency.rs     /v2/clock round-trip profiler
  data_feed.rs   Alpaca SPY quotes + Yahoo ES bars
  signal.rs      online RLS predictor
  executor.rs    selective gate + order placement
  nav.rs         NAV / drawdown state (SQLite-backed)
  logger.rs      SQLite logger
config/
  default.yaml   tunable thresholds
```

## Build

```bash
cd /Users/weeoto/turbo-carnival/research/hft-engine
cargo build --release
```

If you don't already have Rust:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
```

## Run

Credentials are read from `/Users/weeoto/turbo-carnival/council/.env.local`
(`ALPACA_FALCON_KEY`, `ALPACA_FALCON_SECRET`).

```bash
# Dry run — computes signals, places NO orders.
./target/release/hft-engine --dry-run

# Live paper trading (still paper money; uses your FALCON paper account).
./target/release/hft-engine

# Live with interactive confirm for the first 5 trades.
HFT_REQUIRE_CONFIRM=1 ./target/release/hft-engine

# Custom config.
./target/release/hft-engine --dry-run --config config/default.yaml
```

## Kill paths

| Mechanism             | Effect                                             |
|-----------------------|----------------------------------------------------|
| SIGINT / SIGTERM      | Graceful shutdown; tasks aborted.                  |
| Kill file present     | `/Users/weeoto/turbo-carnival/research/live/STOP_HFT_ENGINE` triggers shutdown. |
| NAV drawdown >= 8%    | `nav.halted = true`; no further trades.            |
| 5 consecutive losses  | `nav.halted = true`; no further trades.            |
| Latency p99 > 500ms for 5 min | `nav.halted = true`; logged as regime_event. |

## SQLite

Output:
`/Users/weeoto/turbo-carnival/research/live/hft_engine_decisions.sqlite3`

Tables: `decisions`, `orders`, `latency_samples`, `nav_snapshots`,
`regime_events`, plus `nav_state` (single-row).

```bash
sqlite3 /Users/weeoto/turbo-carnival/research/live/hft_engine_decisions.sqlite3 \
  "SELECT ts, fired, predicted_bps, spread_bps, confidence FROM decisions ORDER BY id DESC LIMIT 20;"
```

## Honest constraints

- We're on a Mac in London. Round-trip to Alpaca will be ~100-200ms. The
  selectivity gate rejects trades when p99 climbs above 250ms.
- ES futures data via Yahoo Finance is delayed up to ~15 minutes outside
  RTH and at best a few seconds during. Production would use CME's
  direct feed (we cannot).
- The predictor is intentionally one-dimensional. More features without
  more samples is over-fit theatre.
- The 30-60s prediction horizon is realistic; anything sub-second is not.
- Expected outcome on Alpaca paper after a week: probably no detectable
  edge. The output should still be a clean dataset to argue from.

## v0.2 ideas (not implemented)

- Multi-feature predictor (NQ, VIX term structure, dollar index).
- Streaming SPY trades via Alpaca websocket (lower-noise mids).
- Regime classifier (VIX > 30 -> halt entries).
- Walk-forward backtest harness against the SQLite log.
