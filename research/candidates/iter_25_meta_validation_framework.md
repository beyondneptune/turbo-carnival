# Iteration 25 — Meta-Validation Framework (CI/CV Pipeline for the Multi-Strategy System)

## 0. Read me first — this is not a strategy

Iterations 1 through 24 each propose **one trading strategy** that ingests market data and emits orders. Iteration 25 ingests **the other 24 iterations** and emits *state transitions* (`research → shadow → live → killed`). Its "edge" is not market-based; it is **operational** — it prevents the multi-strategy operation from accumulating zombie strategies, from re-tuning on its own data, from leaking PnL through stale parameters, and from violating any of the 10 lessons in production.

The spec normally asks "what is the market mechanic that produces edge?" — for this iteration, the market mechanic analog is the **failure mode of multi-strategy desks**: in published surveys of multi-strategy quant funds (Khandani-Lo 2007, Lopez de Prado 2018 ch. 11, AQR 2019 "alpha decay" paper), the *single largest source of realized loss* on a multi-strategy book is not any one strategy's worst day — it is the **persistence of decayed strategies past the date their edge died**. The framework's edge is converting that persistence-loss into a measurable, auto-triggered shutdown.

This file therefore answers each numbered section in the spec by replacing "market mechanics" with "framework mechanics." Every section is present.

---

## 1. Thesis (one sentence)

A continuously-running registry + walk-forward orchestrator + shadow runner + drift monitor + auto-promotion/auto-kill engine, anchored to git as the source of truth for strategy code/parameters and to a Postgres state table for run history, can enforce all 10 documented lessons as programmatic gates such that **no candidate transitions from `research` to `shadow` to `live` without passing the validation tests it declared in its own iter file, and any deployed candidate that breaches its declared kill criteria is auto-demoted within one trading session** — the framework is falsified if, over a 90-day held-out historical re-simulation of the multi-strategy book, its automated promote/demote decisions disagree with documented human-expert decisions on more than 20% of decision events, or if any zombie strategy (defined: a deployed strategy whose rolling 90-day Sharpe is below its declared kill threshold) survives in `live` status for more than 5 trading days.

---

## 2. Market / asset class / venue (framework analog: scope and substrate)

- **"Market" served**: all 24 candidate strategies in `/Users/weeoto/turbo-carnival/research/candidates/iter_1_*.md` through `iter_24_*.md`. Designed to scale to ~50 concurrent registered candidates without architectural change (see §9).
- **"Asset class" of decisions**: state-machine transitions (`status` ∈ {`research`, `shadow`, `live`, `killed`, `paused`, `quarantine`}), parameter freezes, capacity allocations, kill events.
- **"Primary venue"**: a single Postgres 16 instance for run state (or sqlite for dev), a Prefect 2.x orchestrator for scheduled DAGs, a Grafana dashboard for human read-out, and a local-only Slack/email webhook for alerts. No cloud lock-in: everything runs on one Linux host (or one t3.large VPS) and one S3-compatible bucket for run artefacts.
- **"Settlement"**: every framework decision (promote, kill, pause) writes (a) a row to the `decisions` table, (b) a signed git tag `mvf/decision/<id>` against the candidate's pinned commit, (c) a Slack message. Three-way reconciliation between table, git tag, and Slack archive at end-of-day; any divergence pages the on-call operator.
- **"Margin / tax regime" analog (compute and storage cost)**:
  - Compute: a single 8-vCPU / 32 GB host handles 24 candidates' quarterly walk-forward (~6 hours wall-clock if parallelized 4-wide) plus continuous shadow tick reconciliation.
  - Storage: ~80 GB/year for 24 candidates' walk-forward logs and shadow tick-by-tick reconciliation diffs (compressed Parquet); ~5 GB for the registry DB.
  - Cost: ~$60/mo (VPS) + ~$5/mo (S3) + $0 software (all OSS) = **~$65/mo for the entire framework**.

## 3. Time horizon (framework analog: decision cadence)

**Continuous monitoring (1-minute heartbeat for live candidates) with quarterly heavy retest (every 90 days, all candidates re-run their walk-forward against the most recent 90 days of unseen data).** Specifically:

| Cadence | What runs |
|---|---|
| 1-minute | Heartbeat ping to every `live` candidate's broker session; if no heartbeat for 5 minutes, candidate auto-pauses |
| 5-minute | Drift monitor recomputes feature-distribution KL divergence on streaming live data for each `live` candidate |
| End-of-trading-day | Shadow reconciliation: every shadow candidate's local fill ledger is reconciled against broker truth; any tick with divergence > 0.1% halts that candidate to `quarantine` |
| Weekly | Run the auto-promote / auto-kill SQL queries (see §4); produce summary report; require human acknowledgement for any proposed `live` promotion |
| Monthly | Capacity audit: each `live` candidate's deployed-dollar against its registered ceiling; alert at 80%, hard-cap at 100% |
| Quarterly | Heavy retest: every candidate's walk-forward is re-run with the latest 90-day data appended; rank-IC, mean per-trade PnL, and Sharpe are re-estimated; any candidate whose retest Sharpe drops below 50% of its initial Sharpe is moved to `paused` for human review |

This is **not** a sub-second framework. The decisions it makes (promote/kill) operate on a time scale of days; the live-data monitoring operates on minutes. The latency budget §5 reflects this.

## 4. Method family and specific algorithm

**Family**: rules-based finite state machine over a strategy registry, backed by a relational DB and orchestrated by Prefect 2.x DAGs. Deliberately NOT ML — Lesson 2 (binary gates not confidence rankers) applies to the framework itself: promotion is a deterministic gate, not a learned classifier of which candidates "look promising." A framework that uses ML to pick which strategies to promote would itself need walk-forward validation, recursing the problem.

**Specific components and their algorithms**:

### 4.1 Strategy registry (source of truth: git, mirrored to Postgres)

Each registered candidate is one **row** in the `strategies` table AND one **directory** in `git@repo:strategies/<name>/` containing:
- `manifest.yaml` — declarative spec: name, version_sha (filled by CI), broker SDK identifier, capacity_ceiling_usd, verdict_criteria (the §11 numbers from the iter file), kill_criteria (the §11 immediate-kill numbers), pdt_compatibility (bool), tuning_data_ranges (list of date intervals — Lesson 7).
- `entrypoint.py` — must expose `run_backtest(start, end, params, data_handle) -> RunReport` and `run_live(broker_handle, params) -> EventStream`.
- `params.json` — frozen parameter values.
- `pinned_sdk_versions.txt` — exact `pip freeze` of the broker SDK used.

The registry is loaded from git on every framework boot; the Postgres mirror exists for fast queries but **git wins on conflict** (Lesson: registry corruption → use git as source of truth, see §10).

### 4.2 Walk-forward orchestrator

Prefect DAG `wf_quarterly`. For each candidate registered with `status ∈ {research, shadow, live}`:
1. Fetch latest 90 days of data via candidate's declared data feed.
2. Append to the candidate's training-and-OOS history (read-only — the historical OOS slices are immutable; only new dates are added).
3. Run the candidate's `run_backtest(start, end, params, data_handle)` over the most recent 90 days as a **pure OOS slice** (params are frozen, no re-tuning).
4. Compute: mean per-trade PnL, Sharpe, rolling rank-IC (if the candidate is an ML candidate per its manifest), drawdown, win rate, profit factor.
5. Compare against the candidate's declared §11 verdict-criteria values. Write a row to `wf_runs`.
6. If rank-IC drops below the candidate's declared rank-IC threshold (typically the original training rank-IC × 0.5), flag `rank_ic_degradation = true`.
7. If sign of rolling rank-correlation between current OOS slice and training slice flips for two consecutive quarters, flag `rank_drift = true`.

### 4.3 Shadow runner

Prefect deployment `shadow_loop_<candidate_id>` runs continuously per shadow candidate:
1. Connects to the candidate's broker via the candidate's pinned SDK (IBKR insync, ccxt, oanda-py, dxFeed — **rejected at registry time if any candidate ships hand-rolled HTTP**, Lesson 8).
2. Subscribes to broker positions, orders, fills.
3. Calls `run_live(broker_handle, params)` which emits its own intended orders and tracks its own intended positions.
4. **Broker-truth reconciliation** (Lesson 4): every 1 second, diff (broker_position, intended_position) and (broker_cash, intended_cash). If absolute divergence > 0.1% on either, the candidate immediately auto-pauses and the divergence is logged to `truth_divergences`.
5. Records every shadow tick to `shadow_ticks(candidate_id, ts, broker_truth_pos, intended_pos, broker_cash, intended_cash, divergence_bps)`.
6. After ≥ 30 trading days of contiguous shadow with zero unresolved truth_divergences and metrics meeting the candidate's promotion thresholds, the candidate becomes eligible for `live` promotion (still requires human ack — see §4.5).

### 4.4 Drift monitor

For each `live` candidate, on a 5-minute schedule:
1. Pull the last 5 minutes of feature vectors the candidate is consuming (each candidate exposes a `current_features()` method returning a dict of named features).
2. Compute KL divergence between the **rolling 30-day distribution of each feature** and the **training-window distribution of each feature** (the training distribution is frozen at registration time and stored as a histogram in `feature_baselines`).
3. If KL_divergence(feature_i) > 0.5 nats for any feature, log a `drift_warning`. If > 1.0 nats for ≥ 3 consecutive 5-minute windows, log a `drift_alert` and auto-pause the candidate (status → `quarantine`) for human review.
4. **False-alarm guard** (failure-mode §10): drift signals are not auto-kills, they are auto-pauses. Killing on drift alone would over-fire when the market simply moves regime; pausing for human review is the correct response.

### 4.5 Promotion logic (auto-promote/auto-kill SQL)

Auto-promotion is a **two-step** process — the framework does NOT autonomously move a candidate to `live`. It marks the candidate as `promotion_ready`, sends a Slack message with the metrics summary, and waits for a human operator to type `/promote <candidate_id>` in Slack (or equivalent CLI command). This is intentional — Lesson "no automated promotion before human review" (§10 of this file's failure modes).

Auto-kill, by contrast, IS autonomous — a kill is a defensive action that cannot wait for human availability.

```sql
-- Auto-mark promotion-ready (does NOT change status to live; sets a flag)
UPDATE strategies s
SET promotion_ready = true,
    promotion_ready_at = NOW()
FROM (
  SELECT candidate_id,
         COUNT(*) AS shadow_days,
         AVG(daily_pnl_pct) * 252 / NULLIF(STDDEV(daily_pnl_pct) * SQRT(252), 0) AS shadow_sharpe,
         MAX(rolling_60d_drawdown) AS max_dd_60d,
         BOOL_OR(divergence_bps > 10) AS has_truth_divergence
  FROM shadow_daily
  WHERE shadow_start_ts > NOW() - INTERVAL '60 days'
  GROUP BY candidate_id
) sd
WHERE s.id = sd.candidate_id
  AND s.status = 'shadow'
  AND sd.shadow_days >= 30
  AND sd.shadow_sharpe >= (s.verdict_criteria_json->>'promote_min_sharpe')::numeric
  AND sd.max_dd_60d <= (s.verdict_criteria_json->>'promote_max_dd')::numeric
  AND sd.has_truth_divergence = false
  AND s.wf_latest_passed = true       -- Lesson 1: must also have passing walk-forward
  AND s.lesson_checks_all_passed = true; -- all 10 lesson gates must currently pass

-- Auto-kill (this DOES autonomously demote)
UPDATE strategies s
SET status = 'killed',
    killed_at = NOW(),
    killed_reason = kr.reason
FROM (
  SELECT candidate_id,
         CASE
           WHEN MAX(rolling_3d_drawdown) > (kc->>'kill_3d_dd')::numeric THEN '3-day DD breach'
           WHEN MAX(rolling_60d_drawdown) > (kc->>'kill_60d_dd')::numeric THEN '60-day DD breach'
           WHEN BOOL_OR(daily_pnl_pct < (kc->>'kill_daily_loss')::numeric) THEN 'Daily loss cap breach'
           WHEN MAX(divergence_bps) > 50 THEN 'Broker-truth divergence > 0.5%'
           WHEN MIN(rolling_50_trade_sharpe) < (kc->>'kill_rolling_sharpe')::numeric THEN 'Rolling Sharpe collapse'
         END AS reason
  FROM live_daily ld
  JOIN strategies s2 ON s2.id = ld.candidate_id
  CROSS JOIN LATERAL (SELECT s2.kill_criteria_json AS kc) k
  WHERE ld.ts > NOW() - INTERVAL '60 days'
  GROUP BY candidate_id, kc
) kr
WHERE s.id = kr.candidate_id
  AND kr.reason IS NOT NULL
  AND s.status = 'live';
```

### 4.6 Global kill switch (critical — §spec requirement 4)

A single operator-issued command, `mvf kill-all --confirm "<reason>"`, instantly:
1. Sends `cancel_all_orders` to every broker connection of every `live` candidate.
2. Sends `flatten_all_positions` (broker-specific market-out, or limit-out with 60s TIF then market-out) to every broker.
3. Sets every `live` candidate's status to `paused`.
4. Writes a `global_kill_event` row.
5. Posts a red-banner alert to all channels.

Implementation: a single FastAPI endpoint `POST /kill-all` protected by a hardware-token (YubiKey OTP) header, plus a redundant CLI path that goes through the same code. Tested weekly on the staging environment. Failure to flatten any single position within 90 seconds escalates to PagerDuty.

Read-out: every `live` candidate independently checks a `system_state.kill_all_engaged` boolean on every order-emit; if true, the candidate refuses to emit any order. Defense in depth — the kill switch does not rely solely on broker-side cancel.

## 5. Data requirements (framework analog)

- **Inputs needed**:
  - Each candidate's declared data feed (the framework does NOT add a new market-data feed; it reuses what each candidate already needs).
  - Broker streaming positions/orders/fills via each candidate's SDK.
  - Prefect/Postgres internal state.
  - Git as the registry source of truth.
- **Storage estimate**:
  - 24 candidates × 252 trade-days × ~3,000 shadow ticks/day × 200 bytes/tick ≈ **3.6 GB/year of shadow reconciliation logs** at full registry load.
  - Walk-forward run reports: ~10 MB per candidate per quarter × 24 candidates × 4 quarters = **1 GB/year**.
  - Drift monitor histograms and KL series: ~500 MB/year.
  - Registry DB itself: <100 MB even after 5 years.
  - Total: **~6 GB/year, 30 GB after 5 years** — trivial.
- **Cost estimate**: $60/mo VPS + $5/mo S3 + $0 OSS software = **$65/mo for the framework, exclusive of each candidate's own data-feed cost**.
- **Latency requirement**:
  - 1-second tolerance for broker-truth reconciliation diffs.
  - 5-minute tolerance for drift monitor recomputation.
  - 1-hour tolerance for walk-forward DAG completion (per candidate).
  - 24-hour tolerance for quarterly heavy retest completion (across all 24 candidates).

## 6. Validation plan (framework analog: how is the framework itself validated?)

This is the meta-meta question and the hardest part. The framework's own validation plan:

### 6.1 Dogfood

Every existing candidate (iter_1 through iter_24) must register in the framework before iter_25 is itself promoted. Registration is automated via a `register_all.py` script that reads each `iter_<N>_*.md` file, extracts §11 (verdict criteria) and §7 (kill thresholds), and writes a `manifest.yaml` per candidate. Any candidate whose iter file lacks parseable §11/§7 numbers is rejected — this is *itself* a useful Lesson-10 enforcement (mandatory written kill criteria).

### 6.2 Historical re-simulation (90-day held-out)

The 90-day held-out validation: replay the period **Feb 12 2026 → May 12 2026** (the most recent 90 trading days from today's date 2026-05-12). For each day in this window:
1. Reconstruct the registry state as it would have been on that day (from git history).
2. Feed the framework the day's broker-truth and market-data tape.
3. Record every auto-promote-ready flag and every auto-kill the framework would have issued.
4. Compare against a **held-out human-expert decision log** maintained independently — the operator's actual judgement on each candidate on each day (or a synthetic expert log built by replaying §11 criteria by hand on a sampled subset of days).
5. **Validation threshold**: framework decisions agree with human-expert decisions on ≥ 80% of decision events. Below 80% → framework kept in `research` status; not deployed.

This is the framework's analog of "out-of-sample walk-forward" and is the **falsification within 90 days** §10 requirement.

### 6.3 Kill-switch fire drill

Weekly: a non-production candidate (a synthetic `mvf_test_candidate` that holds a tiny IBKR position in SPY) is killed via the global kill switch. The drill verifies (a) flatten happens within 90 seconds, (b) every `live` candidate's order-emit refuses for the duration, (c) the kill event is logged in all three sinks (DB, git tag, Slack).

### 6.4 Continuous monitoring (post-deploy validation)

Once the framework is in `live` status, its own performance is monitored:
- **False-promote rate**: of all candidates the framework promoted to `live`, how many were killed within 30 days? Threshold: < 10%. Above → review promotion gates.
- **False-kill rate**: of all candidates the framework killed, how many would the human operator have kept? Tracked via post-hoc operator review every Friday. Threshold: < 15%. Above → review kill thresholds and drift sensitivity.
- **Time-to-kill latency**: from first breach of kill criteria to status flip. Threshold: < 1 trading session (4-12h depending on candidate's primary market). Above → speed up the DAG cadence.

## 7. Risk envelope (framework analog: blast radius and operational containment)

This is **load-bearing infrastructure**. The risk envelope is not P&L drawdown — it is operational availability and decision correctness.

- **Blast radius if the framework malfunctions**:
  - Worst case: the framework hangs and stops issuing kills. Live candidates keep trading even if they breach kill criteria. **Mitigation**: every `live` candidate ALSO has an in-process kill check that does not depend on the framework (it reads its own §7 thresholds from its own manifest and self-kills if breached). The framework is a redundant outer layer, not the only kill path.
  - Second-worst case: the framework spuriously kills a healthy `live` candidate. **Mitigation**: kills are reversible by operator within 4 hours of the kill event (a soft-killed candidate's positions are flattened but its status can be moved back to `shadow` or `live` by operator command; the candidate then re-shadows for 30 days before re-promotion). This is the "drift detector false-alarm" failure mode from §10 of this file.
  - Third-worst case: registry corruption. **Mitigation**: git is the source of truth. The Postgres mirror can be rebuilt from git in < 5 minutes. Daily Postgres snapshot to S3 plus git tag on every state-changing decision.
- **Daily decision-budget cap**:
  - Auto-promote events are capped at **1 per day** across the entire registry (a "rate limit" on promotions, to give the operator time to notice and ack).
  - Auto-kill events are NOT capped — defensive actions must fire immediately. But if more than 3 candidates auto-kill in any 24-hour window, the framework escalates to "system-wide alert" and pages the operator: this pattern usually indicates a data-feed problem, not 3 strategies all dying at once.
- **Drawdown trigger for the framework itself** (operational SLO):
  - If the framework's own uptime drops below **99.0% over a rolling 30 days**, that itself is an alert. Two consecutive months below 99% triggers a re-architecture review.
  - If the framework's auto-promote disagrees with the operator's manual override more than 20% of the time over 30 days, the promotion thresholds are reviewed.
- **Correlation rule with deployed strategies**: the framework itself does not generate market correlations. But it **enforces** a cross-strategy correlation cap: any pair of `live` candidates whose rolling 30-day return correlation exceeds +0.5 triggers a `correlation_warning`; +0.7 triggers an auto-pause of the *newer* candidate pending human review. (Lesson 6: one strategy per slot.)

## 8. Execution plan (framework analog)

- **Broker / SDK**: N/A directly. The framework calls each candidate's pre-pinned SDK on the candidate's behalf. The framework does not authenticate as a trader account.
- **"Order type"**: state transitions, written as DB row updates within Postgres transactions and as git tags. Every transition is idempotent (re-running a promotion or a kill produces the same end-state, no double-execution).
- **"Slippage budget"**: latency budget. Auto-kill must execute within **30 seconds of breach detection**; global kill switch must flatten all live positions within **90 seconds of activation**. Measured weekly via the kill-switch fire drill (§6.3).
- **"What happens if the broker rejects an order"** (analog: what if a state transition fails?):
  - **Promote fails**: if the Slack ack times out (24h no operator response), `promotion_ready` flag clears, candidate stays in `shadow`, alert posted.
  - **Kill fails** (e.g., broker connection down at kill time): retry with exponential backoff (1s, 5s, 25s, 125s); after 4 failed retries, escalate to PagerDuty and attempt the global kill switch (which has its own redundant broker paths).
  - **Registry write fails**: the git tag is the authoritative record. The Postgres update is retried until success; alarm fires if Postgres is unavailable > 5 minutes. Any decision made while Postgres is down is replayed from the git log on recovery.

## 9. Capacity (framework analog: scale ceiling)

- **Concurrent candidates the framework can monitor in `live` status**: **~50** with the single-host architecture described above. Reasoning:
  - 5-minute drift recomputation × 50 candidates × ~30 features each × 5,000 historical bins per feature ≈ 0.5 GB working memory peak, well within 32 GB.
  - 1-second broker reconciliation × 50 candidates × ~3 broker queries each = 150 queries/sec, within IBKR/ccxt/oanda combined rate limits if each candidate uses a dedicated session.
  - Walk-forward quarterly = 50 candidates × 1 hour each / 4-wide parallelism = 12.5 hours wall-clock per quarter — fits in a weekend.
- **Scale beyond 50**: shard the orchestrator across multiple hosts (Prefect supports this natively); the registry remains single-source-of-truth on git, with a primary-replica Postgres. The current spec serves the 24-candidate run with ~50% headroom.
- **"ADV consumption" analog (compute consumption)**: at 24 candidates, the framework consumes ~30% of one 8-vCPU host on average, peaking to ~80% during quarterly retest weekends. No risk of starving any one candidate's live decision-loop.
- **Concentration risk** (framework analog: single-point-of-failure surface):
  - Single Postgres instance — mitigated by daily S3 snapshots and git as fallback authority.
  - Single Prefect orchestrator — mitigated by Prefect's built-in retry and by per-candidate self-kill (each candidate has an in-process check that does not depend on the orchestrator).
  - Single Slack workspace for alerts — mitigated by dual-write to email and to PagerDuty.

## 10. Failure modes (framework analog: at least 3 named operational failure modes)

### 10.1 Framework bugs let bad candidates through (REAL CONSEQUENCES)
- **What it looks like**: a candidate that should have been killed (e.g., its 3-day drawdown exceeds its declared threshold) remains in `live`. Operator notices the next morning. By then the candidate has compounded its loss.
- **Detection**: cross-check: every `live` candidate's daily summary email includes its kill thresholds AND its current rolling metrics. If a metric exceeds threshold but status is still `live`, the email banner is red and the operator pages themselves.
- **Mitigation**: the in-process per-candidate self-kill (§7 mitigation #1) is the second line of defense. Both the framework AND the candidate independently check kill criteria; either path triggers shutdown.

### 10.2 Over-engineered framework eats the human time that could have produced 5 more candidates
- **What it looks like**: the operator spends 30 hours/week debugging Prefect flows, tuning drift thresholds, fixing schema migrations. Meanwhile no new strategies enter the registry.
- **Detection**: weekly engineering-time-on-framework / engineering-time-on-candidates ratio. If > 1.0 for two consecutive weeks, framework is over-engineered.
- **Mitigation**: the framework is deliberately small (one Postgres, one Prefect, one VPS, no Kubernetes, no microservices, no service mesh). The "v1" feature set is locked at the components described in this iter file; net-new features require an operator-time-budget review. **The framework's own velocity is constrained — its purpose is to enable candidate velocity.**

### 10.3 Registry corruption (git/DB divergence)
- **What it looks like**: the Postgres mirror says candidate X is `live` but the git registry says it's `killed` (or vice versa). The candidate either keeps trading when it shouldn't, or refuses to trade when it should.
- **Detection**: hourly reconciliation script compares `git ls-files strategies/` + parsed manifests against `SELECT id, status FROM strategies`. Any diff alerts.
- **Mitigation**: **git wins on conflict** — the reconciliation script auto-corrects the Postgres mirror from git. The git registry can only be modified by signed commits; the Postgres mirror is never the authority.

### 10.4 Drift detector false-alarms cause spurious kills
- **What it looks like**: a transient market regime shift (e.g., a 1-day VIX spike) elevates feature distributions and the drift monitor pages every candidate at once. If the framework auto-killed on drift, the entire book flattens.
- **Detection**: track `drift_alert` fire rate. If > 3 alerts/day in any single 24h window, trigger anti-flood mode.
- **Mitigation**: as designed, drift alone NEVER auto-kills — it only auto-pauses (`status → quarantine`), which is reversible. Auto-kill requires a hard kill-criteria breach (3-day DD, daily loss cap, broker-truth divergence, rolling-Sharpe collapse), not a feature-distribution shift. This is intentional.

### 10.5 Automated promotion before human review
- **What it looks like**: the framework moves a candidate to `live` overnight based on shadow metrics. The candidate enters production without a human ever having signed off on the latest parameter values.
- **Detection**: as designed, this CANNOT happen — promotion sets `promotion_ready = true` but does NOT change `status` until an operator types `/promote`. The two-step gate is structural, not policy.
- **Mitigation**: status transitions from `shadow` → `live` are write-protected in Postgres via a trigger that requires a matching `operator_ack` row signed within the last 24 hours. The DB schema itself enforces the gate.

## 11. Falsifiable verdict criteria (framework analog)

### Promote framework to "load-bearing production"
ALL of:
- 90-day historical re-simulation (§6.2) shows ≥ 80% agreement with human-expert decisions on the held-out test set.
- All 24 existing candidates successfully register from their iter files without manual edits to manifest YAML (i.e., every iter file has parseable §11 and §7 numbers).
- Kill-switch fire drill (§6.3) completes successfully on 4 consecutive weekly drills with flatten-time < 90 seconds each time.
- All 10 lesson-checks pass on every registered candidate (§details below).
- Postgres / git reconciliation has zero unresolved diffs over 14 consecutive days.

### Continue in "framework-shadow" mode
- Re-simulation agreement is in [60%, 80%]: framework runs in observation mode (logs every decision, never acts) for another 90-day window. After 180 total days, decision required.
- Any of the kill-switch drills failed, but a fix is in progress.
- 1 or 2 candidates have unregisterable iter files: fix those candidates (or fix the registration parser) before promoting framework.

### Kill / re-architect immediately
ANY of:
- Re-simulation agreement < 60% with human expert.
- Global kill switch fails to flatten within 90 seconds on any drill.
- Registry corruption persists for > 24 hours.
- Framework auto-killed a healthy candidate in production AND failed to allow operator reversal within 4 hours.
- An audit reveals a `live` candidate trading without a passing walk-forward AND a passing 30-day shadow (Lesson 1 violation by the framework itself).

## 12. One-paragraph operational risk note

This is the **load-bearing infrastructure** for the entire multi-strategy operation. If the framework is down, no candidate can promote, no kill can fire, no drift alert reaches the operator, no broker-truth reconciliation runs. We pay this risk by (a) keeping the framework extremely small and OSS-only — a single VPS, a single Postgres, a single Prefect, no cloud lock-in — so the operator can rebuild it in a day from the git history if the host evaporates; (b) by making every `live` candidate self-defending — each candidate carries its own §7 kill checks in-process and self-flattens without needing the framework to tell it to; (c) by making git, not Postgres, the source of truth for the registry, so even a complete DB corruption is recoverable in < 5 minutes; (d) by daily S3 snapshots; (e) by a hardware-token-protected global kill switch with a CLI fallback that doesn't depend on any web service being up. The framework is also explicitly **not** a self-modifying system — it does not auto-tune its own thresholds. Any change to a promotion gate, a kill criterion, or a drift threshold requires a signed git commit and an operator review. This prevents the framework from drifting away from the operator's mental model of what it is enforcing.

---

## Operational risk paragraph (spec extra)

Beyond the §12 paragraph: the second-order risk is **operator complacency** — once the framework runs reliably for 6 months, the operator stops reading the daily summary emails, stops noticing the `promotion_ready` Slack messages, stops second-guessing the auto-kills. This is the same operator-failure-mode that plagues real-world alerting systems (alarm fatigue). Mitigation: weekly the framework produces a short "operator awareness" report that explicitly asks the operator to manually-review 2 randomly-sampled framework decisions from the prior week and either confirm or override. If the operator skips this review for 3 consecutive weeks, the framework downgrades its own status to "needs-review" and refuses to issue any new `live` promotions until the operator catches up. This is uncomfortable on purpose.

---

## How this iteration enforces all 10 lessons (the contract)

| # | Lesson | Framework's enforcement mechanism |
|---|---|---|
| 1 | Validate before deploy | `live` promotion blocked unless **both** `wf_latest_passed = true` AND `shadow_days >= 30` AND `shadow_metrics_pass = true`. Postgres trigger enforces this at write time; no code path can bypass. |
| 2 | Binary gates, not confidence rankers | For every candidate flagged in its manifest as `method_family ∈ {ml, nn, llm, hybrid}`, the walk-forward DAG additionally computes (a) the candidate's rank-IC and reports it separately, (b) the correlation between candidate's confidence score and realized PnL — reported as `confidence_pnl_corr`. If `|confidence_pnl_corr| < 0.05` (Lesson 2's specific finding for composite C3), the candidate is auto-flagged "binary-gate-only" — the manifest must declare it is used as a filter, not a sizer. Framework refuses to promote ML candidates whose manifest declares confidence-based sizing if the empirical confidence-PnL correlation fails this test. |
| 3 | TP:SL asymmetry is structural | For every candidate that uses stops (manifest field `uses_stops = true`), the walk-forward DAG runs a `tp_sl_sensitivity_sweep` — re-runs the backtest with TP:SL ∈ {(40bp, 20bp), (60bp, 30bp), (100bp, 35bp), (150bp, 50bp)} and reports the sweep. Promotion requires the candidate's declared TP:SL to be at or near a local optimum (within 0.3 Sharpe of best swept config). Lesson 3's actual finding (100/35 won, 40/20 lost) is built into the sweep grid. |
| 4 | Broker truth, not local cache | Shadow runner reconciles broker_position vs intended_position **every 1 second**. Any divergence > 0.1% halts the candidate. Cannot be disabled — disabling requires changing source code (signed git commit). |
| 5 | No PDT-exposed lanes on sub-$25k US accounts | Registry pre-flight check: each candidate's manifest declares `requires_pdt_room: bool` and `target_account_tier: '<25k' \| '25k-100k' \| '>100k'`. Framework refuses to promote a `requires_pdt_room=true` candidate against a `target_account_tier='<25k'` slot. Hard rejection at registration, before walk-forward even runs. |
| 6 | One strategy per slot | Hash-based duplicate detection: each candidate's manifest is reduced to a canonical (method_family, asset_class, time_horizon, entry_logic_hash) tuple; if a new candidate's tuple matches an existing `live` or `shadow` candidate's tuple, registration is rejected with a diff report. Also: rolling 30-day return-correlation cap of +0.5 between any two `live` candidates (§7), pause newer at +0.7. |
| 7 | No same-data backtest | The manifest's `tuning_data_ranges` field is mandatory; every quarterly walk-forward DAG asserts that the OOS window does not overlap any declared tuning_data_range. If a parameter file changes (detected by file hash), the candidate's `tuning_data_ranges` field MUST be updated in the same commit (CI gate). Re-tuning on previously-tested data is therefore a CI failure, not a runtime check. |
| 8 | Battle-tested broker SDKs only | The manifest's `pinned_sdk_versions.txt` is parsed at registration. Permitted SDK names are an explicit allowlist: `ib_insync`, `ccxt`, `oanda-py`, `dxFeed`, `polygon-api-client`. Any candidate importing `requests`, `httpx`, or any unlisted HTTP library AS its broker connection (detected via AST scan of `entrypoint.py`) is auto-rejected. |
| 9 | Capacity bound named upfront | `capacity_ceiling_usd` is a NOT NULL column in the `strategies` table. Cannot register without a number. Monthly capacity audit (§3) verifies deployed-dollar ≤ declared ceiling; alert at 80%, hard-cap at 100%. |
| 10 | Kill criteria written before deploy | `kill_criteria_json` is a NOT NULL JSONB column. Schema enforces the keys (`kill_3d_dd`, `kill_60d_dd`, `kill_daily_loss`, `kill_rolling_sharpe`) and that they are numeric. Cannot register without populating. Framework's auto-kill SQL (§4.5) reads directly from this column — there is no "default kill criteria" to fall back to; either the candidate declared them or the candidate cannot exist in the registry. |

This iteration **is** the enforcement mechanism for the 10 lessons. Without this framework, each candidate's iter file is a promise; with this framework, each lesson is a programmatic gate that prevents the promise from being broken in production.

---

## Specific deliverables (per spec §requirements 3)

### Strategy registry schema

```sql
CREATE TABLE strategies (
  id              SERIAL PRIMARY KEY,
  name            TEXT NOT NULL UNIQUE,
  iter_number     INT NOT NULL UNIQUE,
  version_sha     CHAR(40) NOT NULL,         -- git commit pinning the candidate code
  status          TEXT NOT NULL CHECK (status IN
                    ('research','shadow','live','killed','paused','quarantine')),
  status_changed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  method_family   TEXT NOT NULL CHECK (method_family IN
                    ('rules','ml','nn','llm','hybrid','rl')),
  asset_class     TEXT NOT NULL,
  time_horizon    TEXT NOT NULL,
  geography       TEXT NOT NULL,
  broker_sdk      TEXT NOT NULL,            -- allow-listed value
  capacity_ceiling_usd  NUMERIC NOT NULL CHECK (capacity_ceiling_usd > 0),
  verdict_criteria_json JSONB NOT NULL,     -- §11 promotion numbers
  kill_criteria_json    JSONB NOT NULL,     -- §11 kill numbers
  requires_pdt_room     BOOLEAN NOT NULL DEFAULT false,
  target_account_tier   TEXT NOT NULL CHECK (target_account_tier IN
                    ('<25k','25k-100k','>100k')),
  uses_stops      BOOLEAN NOT NULL DEFAULT false,
  tuning_data_ranges    JSONB NOT NULL,    -- list of {start,end} intervals
  duplicate_hash  TEXT NOT NULL,           -- Lesson 6 canonical hash
  promotion_ready BOOLEAN NOT NULL DEFAULT false,
  promotion_ready_at TIMESTAMPTZ,
  wf_latest_passed BOOLEAN NOT NULL DEFAULT false,
  wf_latest_at    TIMESTAMPTZ,
  lesson_checks_all_passed BOOLEAN NOT NULL DEFAULT false,
  killed_at       TIMESTAMPTZ,
  killed_reason   TEXT,
  UNIQUE (duplicate_hash)                  -- Lesson 6 enforced at row level
);

CREATE INDEX idx_strategies_status ON strategies(status);
CREATE INDEX idx_strategies_promotion_ready ON strategies(promotion_ready) WHERE promotion_ready = true;
```

### Run-log schema

```sql
CREATE TABLE wf_runs (
  id              BIGSERIAL PRIMARY KEY,
  candidate_id    INT NOT NULL REFERENCES strategies(id),
  run_started_at  TIMESTAMPTZ NOT NULL,
  run_ended_at    TIMESTAMPTZ,
  oos_start       DATE NOT NULL,
  oos_end         DATE NOT NULL,
  mean_per_trade_bps  NUMERIC,
  sharpe          NUMERIC,
  max_drawdown_pct NUMERIC,
  rank_ic         NUMERIC,
  confidence_pnl_corr NUMERIC,
  tp_sl_sweep_json    JSONB,
  passed_verdict  BOOLEAN,
  notes           TEXT
);

CREATE TABLE shadow_ticks (
  id              BIGSERIAL PRIMARY KEY,
  candidate_id    INT NOT NULL REFERENCES strategies(id),
  ts              TIMESTAMPTZ NOT NULL,
  broker_pos_json JSONB NOT NULL,
  intended_pos_json JSONB NOT NULL,
  broker_cash     NUMERIC NOT NULL,
  intended_cash   NUMERIC NOT NULL,
  divergence_bps  NUMERIC NOT NULL
);
CREATE INDEX idx_shadow_ticks_div ON shadow_ticks(candidate_id, ts)
  WHERE divergence_bps > 10;

CREATE TABLE live_daily (
  id              BIGSERIAL PRIMARY KEY,
  candidate_id    INT NOT NULL REFERENCES strategies(id),
  trade_date      DATE NOT NULL,
  daily_pnl_pct   NUMERIC NOT NULL,
  rolling_3d_drawdown NUMERIC,
  rolling_60d_drawdown NUMERIC,
  rolling_50_trade_sharpe NUMERIC,
  UNIQUE(candidate_id, trade_date)
);

CREATE TABLE decisions (
  id              BIGSERIAL PRIMARY KEY,
  candidate_id    INT NOT NULL REFERENCES strategies(id),
  decided_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  decision_type   TEXT NOT NULL CHECK (decision_type IN
                    ('register','promote_ready','promote','pause','kill','reverse_kill','quarantine')),
  decided_by      TEXT NOT NULL,           -- 'framework' or '<operator_email>'
  reason          TEXT,
  git_tag         TEXT                     -- e.g. 'mvf/decision/<id>'
);

CREATE TABLE global_kill_events (
  id              BIGSERIAL PRIMARY KEY,
  fired_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  fired_by        TEXT NOT NULL,
  reason          TEXT NOT NULL,
  candidates_affected INT[] NOT NULL,
  flatten_complete_at TIMESTAMPTZ,
  flatten_complete BOOLEAN NOT NULL DEFAULT false
);

CREATE TABLE feature_baselines (
  candidate_id    INT NOT NULL REFERENCES strategies(id),
  feature_name    TEXT NOT NULL,
  histogram_json  JSONB NOT NULL,          -- frozen at registration
  PRIMARY KEY (candidate_id, feature_name)
);

CREATE TABLE drift_observations (
  id              BIGSERIAL PRIMARY KEY,
  candidate_id    INT NOT NULL REFERENCES strategies(id),
  observed_at     TIMESTAMPTZ NOT NULL,
  feature_name    TEXT NOT NULL,
  kl_divergence   NUMERIC NOT NULL,
  alert_level     TEXT CHECK (alert_level IN ('ok','warning','alert'))
);

CREATE TABLE truth_divergences (
  id              BIGSERIAL PRIMARY KEY,
  candidate_id    INT NOT NULL REFERENCES strategies(id),
  observed_at     TIMESTAMPTZ NOT NULL,
  divergence_bps  NUMERIC NOT NULL,
  resolved_at     TIMESTAMPTZ,
  resolution      TEXT
);
```

### Dashboard endpoints (FastAPI)

```
GET  /strategies              → list all with status, last_wf, last_decision
GET  /strategies/{id}         → full detail incl. all wf_runs, drift series, last 100 shadow_ticks
GET  /strategies/{id}/lessons → 10-row report of each lesson's current pass/fail
GET  /decisions               → recent state transitions, filterable
GET  /drift                   → cross-candidate drift heatmap data
GET  /shadow/divergences      → all unresolved truth_divergences across candidates
GET  /system/health           → uptime, last-DAG-success, postgres-git-sync status
POST /promote/{id}            → operator ack (requires bearer token + 2fa header)
POST /reverse-kill/{id}       → operator reversal (requires within 4h of kill)
POST /kill-all                → global kill switch (requires hardware-token OTP header)
GET  /audit/decisions         → CSV export of all decisions for compliance
GET  /audit/lesson-violations → any historical case where a lesson check failed
```

---

THESIS: A registry + walk-forward + shadow + drift + auto-promote/auto-kill platform that enforces all 10 documented lessons as programmatic gates so no candidate goes live without validation and no decayed candidate persists in live status.
VALIDATION HORIZON: 90-day held-out re-simulation of registry decisions against human-expert decision log on the same period, requiring >=80% agreement, plus weekly kill-switch fire drills with <90s flatten time.
CAPACITY CEILING: ~50 concurrent registered candidates on the single-host architecture (24 current + ~50% headroom); shard the orchestrator across multiple hosts to scale beyond.
