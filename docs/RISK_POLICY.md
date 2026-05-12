# Risk Policy — turbo-carnival

This document codifies the operating rules for the 3-lane Alpaca paper demo.
It is the source of truth referenced by Sonnet #03, Opus max #9, and Opus xhigh
recommendations from the 60-model review.

---

## 1. 90-day code freeze

**Rule.** Once the demo cycle starts (Day 1), no trading-logic code may change
for 90 calendar days. Day 1 = 2026-05-11, freeze ends 2026-08-09.

**What's frozen.**
- `scripts/alpaca_max_hft_host.py`
- `scripts/alpaca_three_lane_host.py`
- Entry / exit / sizing / strike-selection logic
- `data/regime_filters_config.json` thresholds (the *values*, not the schema)
- Strategy spec JSON in `data/strategy_specs/baseline/`

**What's NOT frozen.**
- Observability: logs, dashboards, attribution, slippage logger
- Tomorrow's pre-queued **Valk scale-down** (already an in-flight risk reduction
  approved by the 60-model review)
- The circuit breaker — it's pure defensive scaffolding that can only reduce
  exposure, never increase it
- Off-line research: simulator code, spec proposals in `nim_proposed/`, the
  PDF case study, this doc
- Emergency manual interventions (flatten everything, kill switch)

**Why a freeze.** The investor case study is built on the premise that the
demo runs the strategy *as advertised* for 90 days. If we patch the strategy
mid-cycle, the resulting track record is uninterpretable — was it the rules,
the patch, or luck? A clean 90-day arc, even if poor, is more useful than
a patched arc that "wins" but proves nothing.

## 2. Acceptable drawdown

| Lane    | Starting NAV | Daily loss cap | Cumulative cap (90d) | Action at cap                |
|---------|--------------|----------------|----------------------|------------------------------|
| Aegis   | $100,000     | −0.5 %         | −15 %                | Circuit breaker pauses entries |
| Falcon  | $100,000     | −0.5 %         | −10 %                | Already paused (Day 1)         |
| Valk    | $100,000     | −0.5 %         | −20 %                | Scale-down queued; then manual |
| Combined | $300,000    | −1.5 %         | −15 %                | Hard stop, full flatten        |

The daily cap is enforced by `scripts/daily_loss_circuit_breaker.py` polling
every 30s. The cumulative cap is reviewed end-of-day in the daily report
(`scripts/daily_review_report.py`).

## 3. What "hard stop" means

If the combined cumulative cap (−15 %, i.e. −$45k of $300k) is breached:

1. Cancel all open orders across all three lanes.
2. Submit market orders to flatten every position.
3. Flip every lane's `enabled` flag in `data/regime_filters_config.json` to
   false (and add a `_hard_stop_rationale` field).
4. Write a post-mortem to `reports/hard_stop_<date>.md` covering: trigger
   trade, news context (via `news_trade_joiner.py`), slippage distribution,
   regime filter decisions.
5. The demo ends. We do not "try to make it back".

## 4. Day-1 lessons (frozen)

These are recorded for the post-cycle review, not for mid-cycle patches:

- Aegis closed −$1,824 / −1.82 % — vol-of-vol gate now ON for tomorrow.
- Falcon closed −$1,619 / −1.62 % — paused before 11:00 ET; remains paused.
- Valk holds the iron condor with $19,360 max loss — scale-down to 22
  contracts queued (`data/order_queue/valk_scale_down_20260511T204911Z.json`).
- 60-model independent swarm (20 NIM + 30 Claude + 10 Opus tier sweep): 0/60
  voted to deploy real money on this stack.

## 5. Authority to override

Only the human operator can:
- Lift the freeze
- Raise any drawdown cap
- Resume Falcon
- Approve any change to a strategy_spec/baseline file

Agents and Codex can *propose* but not *enact* any of the above.
