# Simulator audit — strategy_swarm_lab.py

Honest critique of the simulator at `experiments/strategy_swarm_lab.py` so
neither we nor Codex over-trust its outputs. Generated 2026-05-11.

## What it does well

- Drives signals from real cached 1-minute bars (21 days × 7 tickers, 54,322 rows)
- Supports 6 signal modes (mean_reversion, momentum, breakout, turbo,
  news_fade, neutral_carry) and 6 filter primitives sharing the same schema
  as the live host
- Configurable slippage (defaulted to 8 bps after the 12 bps stress test
  confirmed everything degrades further)
- Auto-normalises ratio thresholds → bps via `to_bps()` so spec authors
  can write 0.0025 or 25 interchangeably
- Drops in a synthetic news-blackout window from cached news JSONL files
- Outputs per-spec daily P&L, sharpe, hit rate, trade count

## What it does NOT do — known limitations

### 1. Bar-close fills only
Signals look at the last close; entries assume you fill at *that* close
plus slippage. Reality: you submit an order, the market moves while it
routes, and you fill at some quote a few ms later. **Direction of bias:**
optimistic — the simulator gives you exactly the price you wanted.

### 2. No order book or quote ladder
We don't model bid-ask spread per-symbol, depth, or whether your size eats
through multiple levels. The 8 bps slippage default is a single scalar.
Reality for low-cap, news-driven names can be 20–60 bps round-trip. **Bias:**
optimistic for higher-volatility tickers.

### 3. No per-symbol position sizing reality
We assume uniform notional per signal. Live host has per-lane caps,
per-symbol caps, and Alpaca's PDT / margin rules. **Bias:** uncertain;
mostly optimistic because we're not constrained by margin.

### 4. Look-ahead in news_fade mode
`news_fade` signals look up news within the *same bar* as the entry.
Yahoo's `providerPublishTime` precision is seconds, so if a headline
published at 14:30:25 ET we're entering on the 14:30 bar's close at 14:31:00.
That's OK in *real* time but in backtest the news is treated as "known at
14:30:00 bar open". **Bias:** optimistic for news_fade mode specifically.

### 5. Iron condor is NOT simulated
The simulator handles equity longs/shorts only. Valk's 4-leg condor with
Black-Scholes pricing, theta decay, and IV dynamics is OUT OF SCOPE here.
The `11_valk_14dte_q22.json` spec is a record-keeping placeholder until
Codex (or we) wire up an options engine.

### 6. No regime drift / overfit guard
We have 21 days of data. Any strategy that's flat-but-not-negative on this
window may simply be lucky for this 21-day regime (Fed pause, VIX 15–18,
SPY range-bound). The 60-model swarm reviewed against this same window —
their consensus that 0/60 strategies have edge is informed by a tiny sample.

### 7. Slippage is a constant, not a model
Real slippage scales with: signed order flow, time-of-day, ticker liquidity,
volatility regime. We use 8 bps everywhere. The honest version would sample
from a fitted distribution per ticker per hour. The `slippage_logger.py`
parquet file is the seed dataset for building that.

### 8. No funding / borrow cost on shorts
Short SPY for 30+ minutes incurs a hard-to-borrow fee in real life. We
charge zero. **Bias:** optimistic for any short-heavy strategy.

### 9. Synthetic news_active_until is a single broadcast
Currently we pick news events from a cached file and treat them as a
market-wide event (all tickers blocked). Real news is per-ticker, per-cluster,
and sometimes leaks. **Bias:** the blackout is *too* protective in backtest,
making baseline-without-blackout look worse than baseline-with-blackout.

### 10. No path-dependency on capital
Each spec starts at $100k. We don't compound losses into reduced position
size, or simulate Reg-T margin call cascades. **Bias:** optimistic at the
tail — a real account in drawdown loses *more* because it can hold less.

## What this means for the 60-model verdict

The simulator's tendency is **optimistic across the board**. So when 26 of
26 variants show negative returns at 8 bps slippage, that's actually a
*stronger* signal than the raw number suggests: even the most generous
honest backtest can't manufacture edge. The verdict ("none ready for real
money") survives the audit.

## Recommended fixes (for after the 90-day freeze)

P0 — must-have before any future cycle
1. Replace constant slippage with a per-ticker, per-hour fitted distribution
   from `slippage_log.parquet`.
2. Submit fills against the *next* bar's open, not the current bar's close.
3. Add a Black-Scholes condor engine (Codex handoff item C).

P1 — strongly recommended
4. Per-ticker news joining, not a single market-wide blackout.
5. Per-spec walk-forward: train on first 14 days, test on last 7.
6. Funding/borrow cost line items.

P2 — nice-to-have
7. Path-dependent capital (compound, not reset).
8. Order book ladder for size-aware slippage.

These are documented for Codex to pick up — none can land mid-cycle.
