# Mistakes & Errors — Honest Postmortem (Session 2026-05-12)

Date: 2026-05-12
Author: assistant (Claude / Sonnet 4.5) reviewing this session's actions
Scope: the day's deploy + iteration on turbo-carnival, plus errors of judgment along the way

This is an honest, evidence-cited list. Each entry has: what we did, evidence, impact, and how to prevent recurrence.

---

## TIER 1 — The costly architectural mistakes

### M1. Deployed live based on same-day overfit data
- **What:** Composite C3 winner was selected from a 60-model swarm + 14 composite designs + 100-Haiku voting — **all evaluated on the same 1-day tape (2026-05-11)**. We then deployed the "winner" live on 2026-05-12.
- **Evidence:** `reports/no_lookahead_yesterday/summary.json` documented +$2,029 (with lookahead) vs +$1,287 (without) = **$741 / 36% selection-bias premium**, BEFORE deployment. We had the warning and deployed anyway.
- **Impact:** Today's net portfolio Δ ~−$30 vs simulator's central case of +$209 to +$2,260. The strategy did not have the deployed-believed edge.
- **Fix:** No live deploy without 180-day walk-forward Sharpe ≥ 1.0 + multiple-testing correction `1.0 + 0.3·log₂(N)` where N = variants tested. With N=30 the bar is 2.5 Sharpe — C3 would not have passed.

### M2. Deployed TERMINATOR HFT on $1k when PDT requires $25k
- **What:** Designed TERMINATOR as an "aggressive HFT probe" at $1k. Alpaca enforces FINRA's Pattern Day Trader rule and blocks same-day closes on sub-$25k accounts.
- **Evidence:** First HFT entry on TERMINATOR at 14:23 EDT → all close attempts returned `403 trade denied due to pattern day trading protection`. PDT was *knowable at design time*.
- **Impact:** TERMINATOR had 4 stuck positions all day, generated stop_loss spam, and required a PDT guard fix mid-session.
- **Fix:** Binary spec-file check before deploy: PDT compliance, options level, short locate, minimum trade size. Document in a strategy spec template that must be ticked before going live.

### M3. Built simulations that confirmed the same data they were validating
- **What:** Wrote `experiments/ml_lab/22_terminator_yearly_sim.py` and `26_six_account_yearly_sim.py` that bootstrap from yesterday's 80 C3-kept entries. Projected median $1k → $1,209 (central case 20% decay) up to $2,260 (100% decay).
- **Evidence:** Both simulators use `experiments/ml_lab/runs/composite_backtest/winner_per_entry.parquet` as the bootstrap source — the exact same 1-day dataset C3 was selected against.
- **Impact:** Gave false confidence. We told the user "simulation says $1k → $2,260" without flagging that the bootstrap data was itself overfit.
- **Fix:** Never bootstrap a forward simulation from the data the strategy was selected against. Use out-of-sample data for projections, or explicitly label the projection "in-sample only — not a deploy signal."

### M4. Decided "fix it and deploy" multiple times during market hours
- **What:** 8+ host restarts between 13:13 and 17:25 UTC, each in response to live behavior we'd just observed. Every restart was a fit to the immediate prior 30 minutes of tape.
- **Evidence:** Commits `0f984a2`, `7bf7200`, `d61a52a`, `964dfaf`, `f93d393`, `5dbba06` — all parameter tweaks made AFTER seeing live behavior, mid-session.
- **Impact:** Each restart introduced its own selection bias. The Falcon "loosened gates" change made things worse (verified in `today_live_edge_test_20260512.md`). The TERMINATOR aggressive-probe variant produced 0% win rate on its 3 fills.
- **Fix:** No live parameter changes during a trading session. Observations from a session go into the NEXT pre-market dry-run; tweaks tested off-line; deployed only after explicit validation.

---

## TIER 2 — Engineering bugs (that produced wrong inferences)

### M5. Valk monitor used stale `qty=44` after manual scale-down
- **What:** `alpaca_three_lane_monitor.py` read quantity from the original opening order (44) instead of live positions (22 post-scaledown). The monitor then issued a buy-back of 22 short calls *because it thought it should be closing 44 of them*. Result: 3-leg broken structure.
- **Evidence:** Codex's investigation showed monitor logs at `qty=44` throughout the day after the 09:31 scale-down. Two extra orders fired at 13:52:56 buying back another 22 NVDA short calls.
- **Impact:** Valk briefly had a broken 3-leg condor. Codex restored the missing short call at 14:13 EDT, then patched the monitor to read live qty (commit `f93d393`).
- **Fix:** Monitor MUST read live position quantities before deciding action. Never trust stale "intended" quantities from the opening order. The fix is now in `alpaca_three_lane_monitor.py` via `live_condor_qty()`.

### M6. SQLite `datetime('now','-X seconds')` against ISO timestamps gave wrong filter results
- **What:** Queries like `WHERE ts_utc >= datetime('now','-60 seconds')` look right but actually do string comparison because timestamps stored as `2026-05-12T14:00:00+00:00` (with `T`) compare lexicographically wrong against SQLite's `2026-05-12 14:00:00` (with space).
- **Evidence:** My claim that Aegis was making "4,300 decisions in 60s" — actual proper count was ~30/60s. The query was pulling hours of rows because string-cmp put 'T' > ' '.
- **Impact:** Reported "tight-loop bug" that didn't exist. User had to push back. Codex caught it. Several minutes of wrong analysis and dashboard fix.
- **Fix:** Use Python `datetime.now(timezone.utc) - timedelta(seconds=N)` and pass the ISO string explicitly to SQL. Never rely on `datetime('now')` for ISO-string columns. Already corrected in `dashboard/system_dashboard.py` (commit `5dbba06`).

### M7. Misread `position_marks` historical rows as current live state
- **What:** Claimed "Falcon has stacked long+short positions in NVDA simultaneously" — based on querying `position_marks` without a proper recency filter. Those rows were historical exit marks, not currently held positions.
- **Evidence:** Live Alpaca `/v2/positions` showed Falcon had 1 NVDA long only. My SQL query was wrong (M6).
- **Impact:** Made user think there was a code bug netting failure. There wasn't — Falcon's actual problem was different (over-trading with tight stops).
- **Fix:** When asking "what does X hold right now," query Alpaca `/v2/positions` directly. Position_marks is a historical log, not a current-state mirror.

### M8. Local DB `pending_new` rows never reconciled with broker state
- **What:** Host writes `INSERT INTO orders (status, ...) VALUES ('pending_new', ...)` at order-submit time and never updates it after fill/cancel. Dashboard then showed "orders pending" forever.
- **Evidence:** Commit `5dbba06` separated "host submission log" from "broker actual status" in the dashboard. Live broker call showed 0 pending orders despite local DB showing dozens.
- **Impact:** False alarms about "stuck orders." Repeated incorrect inferences about Falcon being broken.
- **Fix:** Reconcile periodically: pull current order status from broker, update local DB. Or: don't trust local status column; query broker for canonical state.

### M9. Falcon overtraded with no position-netting against existing same-symbol exposure
- **What:** Falcon opened new NVDA positions while already holding one, sometimes opposite direction. The host doesn't check "already have a position in this symbol" before opening another.
- **Evidence:** Broker order log shows multiple buy/sell cycles on same symbol within minutes — `NVDA buy 90 → NVDA sell 90 → NVDA sell 25 (short open) → NVDA buy 25 (close) → repeat`. 90 fills + 47 cancels in 6 hours.
- **Impact:** Each round-trip pays spread. ~$43 of Falcon's $-65 loss is pure churn cost.
- **Fix:** Pre-entry check: `if symbol in existing_positions: skip or net`. Either don't open against existing exposure or compute net target size, not gross. Falcon decommissioned today; this needs to be fixed before redeploying any HFT lane.

### M10. ATHENA_V2_SIGNAL_MULT (1.3×) silently tightened lane thresholds
- **What:** Set a global signal multiplier of 1.3× that applied to ALL HFT lanes. Aegis was supposed to be at 25 bps but actually required 32.5; Falcon supposed to be 18 but required 23.4; TERMINATOR supposed to be 10 but required 13.
- **Evidence:** Codex's commit `f93d393` adding `lane_signal_mult()` documented this with "Lane presets are already calibrated thresholds; do not secretly tighten them."
- **Impact:** Lanes were running at thresholds we didn't think we'd configured. Per-lane test design was undermined by the global multiplier.
- **Fix:** No global multipliers on lane-specific parameters. If you want lane-specific calibration, set it on the lane.

### M11. Polygon S3 pull script had wrong date-format key
- **What:** Script generated S3 keys like `us_stocks_sip/minute_aggs_v1/2026/05/08.csv.gz` but actual keys are `us_stocks_sip/minute_aggs_v1/2026/05/2026-05-08.csv.gz` (full date in filename).
- **Evidence:** Live test showed 403 on the wrong path; list_objects_v2 revealed correct filenames.
- **Impact:** Pull would have failed every day if we'd run it overnight. Caught by testing one file first.
- **Fix:** Always test S3 fetch on one file before kicking off a batch download. Now corrected with `s.strftime("%Y/%m/%Y-%m-%d")`.

---

## TIER 3 — Process & risk-management failures

### M12. No daily-loss circuit breaker on day 1
- **What:** Deployed ATHENA v2 without a max-loss kill switch. Added v2.1's circuit breaker (3% intraday halt) retroactively after market opened.
- **Evidence:** Commit `d61a52a` added circuit breaker — but it was AFTER yesterday's lesson where we lost $3,443 and AFTER today's market open. The Tier-1 list of "missing safety features" should have been mandatory pre-deploy.
- **Impact:** Today's loss was bounded by other factors (gates blocking most entries), but on a worse tape we'd have had no auto-stop.
- **Fix:** Pre-deploy checklist: daily-loss circuit breaker installed, tested, fired in a dry-run.

### M13. No purged k-fold cross-validation before live deploy
- **What:** Built `experiments/ml_lab/23_triple_barrier_purged_kfold.py` and `scripts/polygon_s3_historical_pull.py` ON THE SAME DAY as deploying the strategy. Never ran them.
- **Evidence:** `data/cache/minute_bars/` is empty. The k-fold script has never been executed.
- **Impact:** We had the right validation tool sitting unused while deploying the unvalidated strategy.
- **Fix:** Validation infrastructure must run BEFORE deploy decision, not as a "we'll get to it" item.

### M14. Made multiple lanes that were actually the same strategy at different scale
- **What:** Originally TERMINATOR had identical mode/signal/TP/SL as Aegis, just $1k vs $99k. Hermes and Atlas had identical SVXY 30% rotation, just different accounts.
- **Evidence:** User flagged it: "what's the point of running the same thing when Hermes and Atlas gonna give the same output?" Codex fixed it with lane_policies + per-lane parameter divergence (commit `f93d393`).
- **Impact:** "6 lanes" was actually "3 strategies × different account sizes." Today's "diversification" was illusory.
- **Fix:** Each lane must test a different hypothesis. Document the hypothesis. If two lanes are identical except capital, that's one lane.

### M15. Ran 60-model swarm + 14 composites + 100-Haiku on same day's data
- **What:** Massive multiple-testing on a single day's tape. Selected the "winner" and shipped it.
- **Evidence:** `experiments/runs/swarm_lab_20260511T204106Z/summary.csv` — 60+ rows of variants all evaluated on 2026-05-11.
- **Impact:** Statistical guarantee that some variant would look good. Selected winner had inflated apparent edge.
- **Fix:** Multiple-testing correction `Sharpe ≥ 1.0 + 0.3·log₂(N_variants)`. With N=30 → bar=2.5. C3 would not have passed.

---

## TIER 4 — Communication & judgment errors (mine specifically)

### M16. Claimed Falcon was "stacking contradictory positions" without checking broker
- **What:** Read historical position_marks rows + made the wrong inference. Reported confidently that Falcon held both long and short of same symbol simultaneously.
- **Evidence:** Live broker call showed flat or single position. Stack rows were closed trades from earlier.
- **Impact:** Wrong inference about a code bug. User corrected. Cost time and trust.
- **Fix:** Verify "current state" claims against the source-of-truth (broker API), not local logs.

### M17. Said the market was "live" when it wasn't (or wasn't yet)
- **What:** Replied to user's "the markets live" without checking the clock. Earlier in session, also confused BST/EDT/UTC timezones.
- **Evidence:** Multiple exchanges where I had to be corrected on timing.
- **Impact:** User had to provide their actual local time multiple times. Erosion of confidence in my real-time observability.
- **Fix:** Always run `TZ=America/New_York date` AND `alpaca clock` before claiming market state. Don't infer from system clock alone.

### M18. Built complex visualizations during a frustrating debug session
- **What:** Spent cycles building plotly dashboards, year-1 simulators, and HTML reports while the actual strategy was bleeding live and needing fixes.
- **Evidence:** Commits `e17a419`, `c334282`, `814afeb` — plotly chart upgrades on the simulator — happened DURING market hours while Falcon was overtrading.
- **Impact:** Misallocated effort. Should have stopped Falcon first, debugged the no-trades issue second, and visualized third.
- **Fix:** Triage. If something live is bleeding, fix it before improving polish elsewhere.

### M19. Spawned the council 3-agent panel without checking usage cost
- **What:** Codex (and earlier my agent invocations) spawned 3-4 parallel sub-agents to "audit" things we'd already analyzed. Hit usage limit.
- **Evidence:** Codex's final session message: "You've hit your usage limit." User mentioned "i just bought 8 quid of extra usage."
- **Impact:** Real money cost ($8+), with marginal informational return because the sub-agents would look at the same data we already had.
- **Fix:** Sub-agents are valuable when (a) they have access to data the main agent doesn't or (b) the question genuinely needs independent perspectives. Otherwise don't spawn.

### M20. Misled with "4,300 decisions in 60s" claim
- **What:** SQL filter bug (M6) caused me to report a 100× inflated decision rate as a "loop bug." It wasn't a bug; my query was.
- **Evidence:** Codex's correction with proper Python timestamp filter showed ~30/60s actual rate.
- **Impact:** Wrong diagnosis presented confidently. Made user think the host was misbehaving when it wasn't.
- **Fix:** Sanity-check rates against configured intervals. If you're seeing 70/sec from a 2-second loop, something is wrong with the measurement, not the host.

---

## Summary by category

| Category | Mistakes | Estimated cost (real) |
|---|---:|---|
| Architectural (T1) | 4 | The bulk of the day's $-30 + uncalculable opportunity cost |
| Engineering (T2) | 7 | Wasted hours of analysis time, several wrong inferences |
| Process (T3) | 4 | Risk that wasn't bounded |
| Communication (T4) | 5 | User had to correct me multiple times; $8+ in tokens |
| **Total** | **20** | |

## What I would do differently in one sentence

**Validate offline with the right CV methodology, deploy only what passes the bar, never tweak parameters during live trading, and always verify "current state" claims against the source of truth (broker API) rather than local logs.**

## What we genuinely got right (counterbalance)

For honesty: not everything was wrong.

- The infrastructure to RUN multi-lane paper trading worked end-to-end (host + monitor + dashboard + observer + SVXY rotation)
- Aegis (the conservative configuration) was verified-positive today on its 2 trades
- Valk's iron condor + scale-down executed cleanly after the monitor fix
- SVXY rotation behaved as designed across Hermes/Atlas
- Codex's lane-divergence patch made the 6 accounts test different hypotheses
- Today's 3 verified findings (TP/SL ratio, C3-as-filter, Aegis rare-fire) are real and reproducible
- Total drawdown bounded at −$30 / −0.006% — system survived

The strategy didn't have the edge we believed. But the *system* worked, and we ended the day with a clearer evidence base for tomorrow's decisions than we started with.

---

Reproducible by reading the cited commits and reports.
