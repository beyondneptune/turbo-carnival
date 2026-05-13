# Nine-Lane Paper Plan - Claude Handoff

This file is the exact handoff for Claude Code.

Goal: tomorrow's paper deployment should run the original 3 lanes plus 6 additional high-return SPY-core strategy candidates. Total = 9 things running/logging.

Important framing:
- This is paper trading / research.
- Lane 1 is still the controlled hosted lane.
- Lanes 2-9 are research lanes that must log decisions, orders, fills, positions, and PnL separately.
- The six added strategies are proxy-backtest candidates. They are not proven live-execution strategies.

## Code Paths

- Main SPY-core proxy runner: `experiments/run_spy_core_strategy_grid.py`
- Existing 3-lane lab: `fullonriskmodelterminator/run_all_three.py`
- Lane 2 council code: `fullonriskmodelterminator/backtest.py`
- Lane 3 pure API code: `fullonriskmodelterminator/pure_api_trader.py`
- Existing 3-lane README: `fullonriskmodelterminator/README.md`
- Six-strategy runnable spec JSON: `fullonriskmodelterminator/additional_six_spy_core_specs.json`

## Result Sources To Verify

- Original high-return SPY-core context:
  - `council/lab_runs/20260510T153306Z/context.md`
- Council decision note:
  - `council/lab_runs/20260510T153306Z/decision_note.md`
- Council validated specs:
  - `council/lab_runs/20260510T153306Z/validated_spy_core_specs.json`
- Council grid result CSV:
  - `experiments/runs/council_spy_grid_20260510T153306Z/summary.csv`
- Council grid trades:
  - `experiments/runs/council_spy_grid_20260510T153306Z/trades.csv`
- Council grid daily NAV:
  - `experiments/runs/council_spy_grid_20260510T153306Z/daily_nav.csv`
- Red-team synthesis for the main council winner:
  - `council/redteam_runs/20260510T155050Z/SYNTHESIS_BY_CLAUDE.md`

## The Existing 3 Lanes

### Lane 1 - Quant Realistic

Purpose: controlled hosted paper strategy.

Source:
- `fullonriskmodelterminator/run_all_three.py`
- function: `lane_1_quant_replay`

Latest completed Lane 1-only result:
- Path: `fullonriskmodelterminator/runs/three_lane_20260510T225632Z/summary.json`
- Result: `$100,000 -> $105,522.20`
- Return: `+5.5222%`
- Monthly: `+0.4603%`
- Max DD: `-5.1001%`
- Trades: `74`

Claude instruction:
- This is the main paper deployment lane.
- Use deterministic logic.
- Log under a separate run id, e.g. `live_lane1_quant_YYYYMMDD`.

### Lane 2 - NIM Council

Purpose: 8 NIM models vote daily from a constrained menu.

Source:
- `fullonriskmodelterminator/backtest.py`
- council vote implementation: `fullonriskmodelterminator/council.py`
- action menu: `fullonriskmodelterminator/menu.py`

Known smoke result:
- Path: `fullonriskmodelterminator/runs/council_backtest_20260510T222403Z/headline.json`
- Result: `$100,000 -> $58,472.53` over 10 days
- Return: `-41.5275%`
- Max DD: `-41.9975%`
- Trades placed: `10`

Claude instruction:
- Run as research lane only.
- Log every model vote and final vote.
- Suggested run id: `live_lane2_council_YYYYMMDD`.

### Lane 3 - Pure API

Purpose: single NIM model proposes free-form option legs.

Source:
- `fullonriskmodelterminator/pure_api_trader.py`

Known smoke result:
- Path: `fullonriskmodelterminator/runs/pure_api_20260510T225714Z/headline.json`
- Result: `$100,000 -> $243,030.34` over 15 days
- Return: `+143.0303%`
- Max DD: `-26.9939%`
- Trades placed: `4`

Critical caveat:
- This was mostly short-premium / short-strangle behavior during a calm window.
- It can reverse hard on a volatility spike.

Claude instruction:
- Run as research lane only.
- If connected to paper broker, use tiny size until live paper risk is understood.
- Suggested run id: `live_lane3_pure_api_YYYYMMDD`.

## Add These 6 More Lanes

These are deterministic SPY-core strategy candidates. They should be added as Lane 4 through Lane 9.

### Lane 4 - aggressive_daily_nogate_d20_30d_w10

Exact strategy name:
- `aggressive_daily_nogate_d20_30d_w10`

Source metric:
- `council/lab_runs/20260510T153306Z/context.md`

Proxy result:
- Final NAV: `$57,036.23` from `$10,000`
- Return: `+470.3623%`
- Max DD: `-50.1279%`
- Sharpe: `1.079972`
- Trades: `1827`
- Max open trades: `11`

Spec:
- `entry_mode`: `daily_ladder`
- `gate`: `none`
- `target_delta`: `0.20`
- `target_dte`: `30`
- `wing_width`: `10.0`
- `profit_take`: `0.50`
- `stop_loss_credit_mult`: `2.00`
- `close_dte`: `21`
- `min_credit`: `0.10`

Paper lane id:
- `live_lane4_aggressive_d20_nogate_30d_w10_YYYYMMDD`

### Lane 5 - aggressive_daily_nogate_d16_30d_w10

Exact strategy name:
- `aggressive_daily_nogate_d16_30d_w10`

Source metric:
- `council/lab_runs/20260510T153306Z/context.md`

Proxy result:
- Final NAV: `$52,526.34` from `$10,000`
- Return: `+425.2634%`
- Max DD: `-51.2822%`
- Sharpe: `1.030090`
- Trades: `1827`
- Max open trades: `11`

Spec:
- `entry_mode`: `daily_ladder`
- `gate`: `none`
- `target_delta`: `0.16`
- `target_dte`: `30`
- `wing_width`: `10.0`
- `profit_take`: `0.50`
- `stop_loss_credit_mult`: `2.00`
- `close_dte`: `21`
- `min_credit`: `0.10`

Paper lane id:
- `live_lane5_aggressive_d16_nogate_30d_w10_YYYYMMDD`

### Lane 6 - daily_ladder_vrp_positive_d16_45d_w5

Exact strategy name:
- `daily_ladder_vrp_positive_d16_45d_w5`

Source metric:
- `council/lab_runs/20260510T153306Z/context.md`

Proxy result:
- Final NAV: `$45,976.76` from `$10,000`
- Return: `+359.7676%`
- Max DD: `-32.1266%`
- Sharpe: `0.997495`
- Trades: `1495`
- Max open trades: `22`

Spec:
- `entry_mode`: `daily_ladder`
- `gate`: `vrp_positive`
- `target_delta`: `0.16`
- `target_dte`: `45`
- `wing_width`: `5.0`
- `profit_take`: `0.50`
- `stop_loss_credit_mult`: `2.00`
- `close_dte`: `21`
- `min_credit`: `0.20`

Paper lane id:
- `live_lane6_vrp_d16_45d_w5_YYYYMMDD`

### Lane 7 - daily_ladder_vrp_positive_d16_30d_w5

Exact strategy name:
- `daily_ladder_vrp_positive_d16_30d_w5`

Source metric:
- `council/lab_runs/20260510T153306Z/context.md`

Proxy result:
- Final NAV: `$30,318.52` from `$10,000`
- Return: `+203.1852%`
- Max DD: `-16.3734%`
- Sharpe: `1.213017`
- Trades: `1495`
- Max open trades: `11`

Spec:
- `entry_mode`: `daily_ladder`
- `gate`: `vrp_positive`
- `target_delta`: `0.16`
- `target_dte`: `30`
- `wing_width`: `5.0`
- `profit_take`: `0.50`
- `stop_loss_credit_mult`: `2.00`
- `close_dte`: `21`
- `min_credit`: `0.20`

Paper lane id:
- `live_lane7_vrp_d16_30d_w5_YYYYMMDD`

### Lane 8 - risk_qwen_risk_designer_risktest_d12_30d_lowdelta_pt30_sl20_c10

Exact strategy name:
- `risk_qwen_risk_designer_risktest_d12_30d_lowdelta_pt30_sl20_c10`

Source metric:
- `experiments/runs/council_spy_grid_20260510T153306Z/summary.csv`
- `council/lab_runs/20260510T153306Z/decision_note.md`
- `council/lab_runs/20260510T153306Z/lab_summary.md`

Proxy result:
- Final NAV: `$36,467.17` from `$10,000`
- Return: `+264.6717%`
- Max DD: `-14.7004%`
- Sharpe: `1.338386`
- Win rate: `91.4381%`
- Avg trade: `$17.7073`
- Worst trade: `-$298.1499`
- Trades: `1495`
- Max open trades: `16`

Spec:
- `entry_mode`: `daily_ladder`
- `gate`: `vrp_positive`
- `target_delta`: `0.12`
- `target_dte`: `30`
- `wing_width`: `5.0`
- `profit_take`: `0.30`
- `stop_loss_credit_mult`: `2.00`
- `close_dte`: `10`
- `min_credit`: `0.20`

Extra validation paths:
- Walk-forward: `experiments/runs/council_spy_grid_20260510T153306Z/walk_forward/walk_forward_split_2023-01-01.json`
- Realistic friction report: `experiments/runs/council_spy_grid_20260510T153306Z/realistic/risk_qwen_risk_designer_risktest_d12_30d_lowdelta_pt30_sl20_c10.md`
- Red-team: `council/redteam_runs/20260510T155050Z/SYNTHESIS_BY_CLAUDE.md`

Paper lane id:
- `live_lane8_qwen_risktest_d12_30d_pt30_sl20_c10_YYYYMMDD`

### Lane 9 - execution_llama_allocator_designer_daily_ladder_vrp_positive_d12_30d_w5_stop_loss_1_5

Exact strategy name:
- `execution_llama_allocator_designer_daily_ladder_vrp_positive_d12_30d_w5_stop_loss_1_5`

Source metric:
- `experiments/runs/council_spy_grid_20260510T153306Z/summary.csv`
- `council/lab_runs/20260510T153306Z/decision_note.md`
- `council/lab_runs/20260510T153306Z/lab_summary.md`

Proxy result:
- Final NAV: `$33,033.51` from `$10,000`
- Return: `+230.3351%`
- Max DD: `-15.8747%`
- Sharpe: `1.250643`
- Win rate: `90.1003%`
- Avg trade: `$15.4105`
- Worst trade: `-$251.3054`
- Trades: `1495`
- Max open trades: `16`

Spec:
- `entry_mode`: `daily_ladder`
- `gate`: `vrp_positive`
- `target_delta`: `0.12`
- `target_dte`: `30`
- `wing_width`: `5.0`
- `profit_take`: `0.30`
- `stop_loss_credit_mult`: `1.50`
- `close_dte`: `10`
- `min_credit`: `0.20`

Extra validation paths:
- Walk-forward row exists in `experiments/runs/council_spy_grid_20260510T153306Z/walk_forward/walk_forward_split_2023-01-01.csv`
- Raw validated spec source: `council/lab_runs/20260510T153306Z/validated_spy_core_specs.json`

Paper lane id:
- `live_lane9_llama_d12_30d_w5_sl15_YYYYMMDD`

## How To Re-Run The Six Added Strategies In The Existing Proxy Runner

Use:

```bash
cd /Users/weeoto/turbo-carnival
.venv/bin/python experiments/run_spy_core_strategy_grid.py \
  --spec-json fullonriskmodelterminator/additional_six_spy_core_specs.json \
  --output-dir fullonriskmodelterminator/runs/additional_six_proxy_$(date -u +%Y%m%dT%H%M%SZ)
```

That produces:
- `summary.csv`
- `trades.csv`
- `daily_nav.csv`
- `equity_curves_30_strategies.html`

## Paper Deployment Interpretation

Total 9 lanes:

1. `live_lane1_quant_YYYYMMDD` - controlled deterministic hosted lane.
2. `live_lane2_council_YYYYMMDD` - NIM council vote lane.
3. `live_lane3_pure_api_YYYYMMDD` - pure API free-form lane.
4. `live_lane4_aggressive_d20_nogate_30d_w10_YYYYMMDD`
5. `live_lane5_aggressive_d16_nogate_30d_w10_YYYYMMDD`
6. `live_lane6_vrp_d16_45d_w5_YYYYMMDD`
7. `live_lane7_vrp_d16_30d_w5_YYYYMMDD`
8. `live_lane8_qwen_risktest_d12_30d_pt30_sl20_c10_YYYYMMDD`
9. `live_lane9_llama_d12_30d_w5_sl15_YYYYMMDD`

Recommended account handling:
- Use one IBKR paper account if necessary, but separate every order by lane id in local DB.
- Better: simulate separate subaccounts locally with virtual capital buckets.
- Suggested virtual capital: `$100,000` per lane in logs even if broker account is one shared paper account.

Minimum logging required per lane:
- market snapshot at decision time
- selected strikes/expiry/legs
- quote snapshot for every leg
- order request JSON
- broker response JSON
- fill events
- position snapshot
- daily mark
- exit reason
- realized PnL

Hard caveat:
- The six added strategy returns are proxy backtest returns from Black-Scholes / IV proxy, not guaranteed live broker fills.
- They are appropriate for paper testing and comparison.
- They should not be represented as live-proven performance.

