## Specialist: WINDOW

### Core thesis (1 sentence)

Slippage is regime-dependent and the regime is *time*: trade only 10:30-14:30 ET, skip the open/close auctions and 15-min news blackouts, and the same rule-based entries that lost $3,443 today would have been mostly suppressed during the worst-spread windows where TSLA/AMD bled.

### Strategy decisions
1. trade_frequency: **~25 entries / lane / day** (down from baseline ~80). 4-hour window = 240 minutes, minimum 8-min spacing per symbol, 5 active symbols → natural cap ~30; news blackouts shave to ~25.
2. order_type: **marketable_limit, +3 bps offset** inside the window (spreads are tightest 10:30-14:30). No market orders, ever. Bracket attached: TP/SL as child limits/stops.
3. position_sizing: **uniform $20k notional per entry**, but scale to $12k in the 10:30-11:00 warm-up half-hour (still wider spreads than midday). Cap 5 concurrent positions / lane.
4. entry_signals: existing rule-based mean_reversion (Aegis) + breakout (Falcon) **unchanged in logic**, gated by `time_gate(10:30-14:30 ET)` AND `news_blackout(±15 min around scheduled releases from FRED/BLS/earnings calendar)`. Thresholds: keep today's z-score / ATR breakout levels — the filter, not the trigger, is what changes.
5. exit_rules: TP 0.35 %, SL 0.55 %, **time-exit hard 14:45 ET** (force flat 15 min before our window ends, well before MOC traffic). ML kill: if LogReg regime model (AUC 0.799) emits P(profit) < 0.40 on the bar, skip the signal. AE anomaly z > 3.0 → skip (microstructure outlier).
6. symbol_universe: **SPY, QQQ, AAPL, NVDA, META** — drop TSLA and AMD entirely. TSLA+AMD = 69 % of today's entries and the bulk of slippage cost; their intraday spreads are 2-3x SPY/QQQ. Tighter-spread symbols dominate inside the midday window.
7. lane_split:
   - **Lane-1**: Aegis mean-reversion, SPY+QQQ+AAPL only, 10:30-14:30, TP 0.30 / SL 0.45.
   - **Lane-2**: Falcon breakout, QQQ+NVDA+META, 10:30-14:30, TP 0.45 / SL 0.65.
   Different symbol sets so the lanes aren't correlated; both share the same time-window discipline.
8. filter_cascade (ordered, short-circuit):
   1. `time_gate(10:30 ≤ t ≤ 14:30 ET)` — cheapest, kills ~38 % of would-be entries
   2. `news_blackout(±15 min)` — economic calendar + earnings dates
   3. `symbol_whitelist` (drop TSLA/AMD)
   4. `abs_return_gate` (skip if last 5-min |return| > 0.6 %; tape too hot)
   5. `LogReg regime P(profit) ≥ 0.40`
   6. `AE anomaly z ≤ 3.0`
   7. signal-mode trigger (mean-rev z / breakout)
   8. position/correlation check

### Expected backtest result
- Daily entries: 163 → **~50 (25/lane × 2)**
- Block rate vs today: **~69 %** (time 38 % + symbol 20 % + news/anomaly 11 %)
- Predicted Δ P&L vs today on bar-close basis: **+$1,400 to +$1,900** (we keep most of the +$496 winners that landed in 10:30-14:30 and lose the worst TSLA/AMD open-hour entries; counterfactual on today's 162 entries: ~50 % occurred outside 10:30-14:30 on TSLA/AMD)
- Predicted Δ P&L at 8 bps slippage: **+$200 to +$900** (50 trades × ~$8 slippage = ~$400 drag, but average spread inside the window is ~30 % tighter than today's volume-weighted average)
- Confidence: **med** — the time-of-day permutation importance (tod_cos top-2) and the fact that 9:30-10:00 and 15:30-16:00 are documented widest-spread windows are robust priors; uncertainty is whether the *signal* still fires enough inside the window to clear the slippage hurdle.

### Steelman against my own approach (1-2 sentences)

The 10:30-14:30 window is also when realized vol is lowest, which kills mean-reversion edge along with the spread — I might be cutting trade count without proportionally cutting loss rate, and the surviving 50 entries could still average −$8 each at 8 bps. Also, the economic calendar I'd consume tomorrow (CPI 08:30 ET is *before* my window anyway) may not actually generate the 15-min blackouts the spec assumes are common during midday.

### My critique of any ML-only specialist approach (1 sentence)

Filtering on a 0.799-AUC LogReg without a time gate still lets the model fire during 9:30-10:00 when fills are 2-3x worse than the model was trained on (training bars were 1-min but mid-quote, not actual fills) — the model's edge gets eaten by a regime it can't see in features.

### Files I'd write
- `/Users/weeoto/turbo-carnival/data/strategy_specs/baseline/window_aegis.json`: Aegis spec with `entry_window=["10:30","14:30"]`, `force_flat_time="14:45"`, `symbols=["SPY","QQQ","AAPL"]`, `news_blackout_minutes=15`, `order_type="marketable_limit"`, `limit_offset_bps=3`, `tp_pct=0.30`, `sl_pct=0.45`.
- `/Users/weeoto/turbo-carnival/data/strategy_specs/baseline/window_falcon.json`: Falcon spec, `symbols=["QQQ","NVDA","META"]`, same time window, `tp_pct=0.45`, `sl_pct=0.65`.
- `/Users/weeoto/turbo-carnival/scripts/economic_calendar.py`: lightweight loader that returns blackout intervals for 2026-05-12 (CPI/PPI/FOMC/earnings); offline JSON in `data/calendar/2026-05-12.json` to avoid live API dependency at open.
- `/Users/weeoto/turbo-carnival/scripts/alpaca_max_hft_host.py`: patch `_should_enter()` to short-circuit on `not in_window(now)` BEFORE any signal computation; log `entries_blocked_by_window` counter for next-day reconciliation.
- `/Users/weeoto/turbo-carnival/experiments/ml_lab/window_backtest.py`: replay today's 163 entries from `minute_bars.parquet`, partition by time-of-day bucket, report P&L by 30-min bucket so we can validate the 10:30-14:30 hypothesis before committing to tomorrow's open.
