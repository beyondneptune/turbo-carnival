# Pre-Registration: paper_v1

**Version:** v1.0 (frozen on tag)
**Cycle ID:** paper_v1
**Prereg authored:** 2026-05-06
**Pre-freeze amendment:** 2026-05-07 — held-out Year 2 OOS evidence policy
**Freeze tag (target):** `paper_v1.0-frozen` — set by CEO at cycle start
**Year 1 (training, in-sample):** 2024-05-07 → 2025-05-06 inclusive
**Year 2 (held-out OOS evidence):** 2025-05-07 → 2026-05-06 inclusive
**Primary evidence window:** Year 2, evaluated once after tag
**Forward confirmation:** optional 90-calendar-day paper run from tag, frozen
**Modification rule:** Once tagged, any change to this document, the strategy code, or the metadata files resets the cycle clock to day 0.

---

## 1. Hypothesis

The hybrid options strategy under test combines a **structurally-supported income floor** with a **falsifiable alpha hunt**, sized at the smallest viable retail capital ($10,000 paper notional). The core sleeve harvests the volatility risk premium via SPY iron condors; the satellite sleeve selects earnings straddles on the Russell 1000 using a twin-model A/B design. The hybrid is preferred over a single-hypothesis bet because the VRP core does not require the satellite alpha hunt to succeed for the project to demonstrate investor value — and the satellite test, being twin-model, is falsifiable on its own terms.

The two specific claims tested in cycle 1:

- **VRP core (Engine A):** A defined-risk, weekly-cadence iron condor program on SPY at 30–45 DTE produces positive net-of-cost returns sufficient to clear `max(B0..B3)` by ≥200 bps annualised at +50% costs.
- **Earnings satellite (Engine B):** A condition-aligned target `(return ÷ expected_event_risk)` selects better than a plain return target on Russell 1000 earnings straddles, demonstrated via twin-model A vs B with all three thresholds met (net excess +50 bps annualised, Sharpe delta +0.2, paired t ≥ 2.0).

## 2. Capital and Sizing

- **$10,000 paper notional** — locked for Year 1 per `capital_policy.md`
- **70 / 30 core / satellite split** — $7,000 core, $3,000 satellite
- **Cash buffer ≥20% of NAV** — floats with NAV; sleeves re-size if NAV falls
- **Per-trade sizing:** core = 1 contract per condor, max 4–6 simultaneous; satellite = 1 contract per straddle, max 2–4 simultaneous

Full per-trade rules: `metadata/capital_policy.md` § Sizing Per Trade.

## 3. Sleeve Specifications

### 3.1 Core (VRP iron condors on SPY)

- **Underlying:** SPY only (SPX out of capital reach at $10k)
- **Entry:** 30–45 DTE, weekly cadence
- **Structure:** $5-wide wings, ~10–15 delta short strikes
- **Profit target:** 50% of credit received, **OR** close at 21 DTE
- **Stop:** 2× credit received as max loss before forced exit
- **Code:** `signals/vrp_core_spy.py`

### 3.2 Satellite (earnings straddles on Russell 1000)

- **Universe:** Russell 1000 names passing the liquidity filter — trailing 30-day options ADV ≥ 5,000 contracts, OI ≥ 10,000, median ATM bid-ask ≤ 8% of mid, ≥ 8 historical earnings events with chain data (per `strategic_lock.md` § Satellite Sleeve)
- **Cap:** Top 50 names by options ADV, refreshed quarterly
- **Structure:** ATM straddle, nearest weekly expiry ≥3 calendar days post-earnings
- **Hold:** T-1 close to T+5 close, no partial exits
- **Selection:** Twin-model (Model A vs Model B) — Model B used **only** if all three twin-model thresholds met (§ 9)
- **Code:** `signals/earnings_satellite.py`, `signals/twin_model_trainer.py`, `signals/earnings_universe.py`

## 4. Risk Envelope and Kill Switch

Pre-trade risk gates run on every candidate before order submission. The kill switch is **deterministic Python code** — no LLM can override it; no human can disable it during a frozen cycle (would reset the cycle clock).

- Pre-trade risk gates: `risk/envelope.py`
- Kill switch: `risk/kill_switch.py`

**Hard limits (verbatim from `capital_policy.md` § Hard Risk Limits):**

| Limit | Trigger | Action |
|---|---|---|
| Daily intraday loss | -8% NAV | Cycle suspended, postmortem required |
| Cumulative cycle drawdown | -15% NAV | Cycle suspended, restart prereg |
| Single-position vega | >5% NAV | Position rejected by risk gate |
| Total portfolio vega | >15% NAV | New positions rejected until reduced |
| Single-name concentration | >20% NAV | Position rejected by risk gate |
| Cash buffer floor | <20% NAV | Stop opening new positions |

### 4.1 Adversarial Stress Scenarios (locked)

Implemented in `risk/envelope.py::check_scenario_stress`. The four pre-registered scenarios are evaluated for every candidate:

| Scenario | Spot move | Vol move (vol points) |
|---|---|---|
| Mild down | −5% | +10 |
| **Adversarial (Gate 5)** | **−10%** | **+30** |
| Mild up | +5% | +5 |
| Vol contraction | flat | −5 |

The worst projected scenario drop must remain < 25% NAV (Gate 5).

## 5. Cost Model

- **Commission:** $0.65 / contract IBKR Pro
- **Spread cost:** half-spread per leg
- **Slippage:** $0.01 / share default (recorded as intended − actual on every fill in `trade_legs`)
- **Cost stress:** gate must hold at **base costs AND +50% costs** (spread × 1.5)
- **+100% costs:** informational only, reported in postmortem, non-blocking

## 6. Annualisation Conventions (locked)

Mixed conventions, deliberate per the cadence each metric measures:

| Metric | Convention | Rationale |
|---|---|---|
| Strategy returns, Sharpe, max DD | 252 trading days, arithmetic annualisation | Trading positions are marked on trading days only |
| Twin-model net-excess (B vs A) | events per year = 252 ÷ 5 = 50.4 | Each satellite event is a 5-day hold |
| B0 cash accrual | 365 calendar days, geometric: `(1 + r) ** (5 / 365) - 1` | Cash sits in collateral every calendar day, not just trading days |

Source-of-truth: `eval/metrics.py` (Sharpe / DD), `signals/twin_model_trainer.py::EVENTS_PER_YEAR`, `eval/baselines.py::baseline_b0_no_trade`.

## 7. Pre-Registered Gates (7)

All seven must hold simultaneously at **Base AND +50% costs**. Verbatim from `strategic_lock.md` § Pre-Registered Gates.

| # | Gate | Threshold | Computed by |
|---|---|---|---|
| 1 | DSR | DSR ≥ 0 with t ≥ 3.84 | `eval/metrics.py::deflated_sharpe_ratio` |
| 2 | PBO | PBO < 0.5 | `eval/metrics.py::probability_backtest_overfitting` |
| 3 | Net excess vs `max(B0..B3)` | ≥ 200 bps annualised | `eval/baselines.py` + `eval/gates.py::evaluate_net_excess` |
| 4 | Max drawdown | < 15% NAV | `eval/metrics.py::max_drawdown` |
| 5 | Adversarial stress drop | < 25% relative | `risk/envelope.py::check_scenario_stress` |
| 6 | Regime stability | no HMM partition underperforming best benchmark by >5% (boundary inclusive at exactly −5%) | `eval/metrics.py::evaluate_regime_stability` |
| 7 | Sample size | ≥ 50 satellite events from the held-out Year 2 evidence run where `sleeve='satellite'`; no in-sample, synthetic, or post-result replay padding | `SELECT COUNT(*) FROM trades WHERE run_id=? AND sleeve='satellite'` |

Composite evaluator: `eval/gates.py::evaluate_full`.

### 7.1 Gate 6 HMM Specifications (locked)

- **Type:** 2-state Gaussian HMM (`hmmlearn.hmm.GaussianHMM`)
- **Input feature:** 1-D series of daily SPY benchmark returns
- **Fitting window:** the held-out Year 2 evidence window; optional forward confirmation fits on its own forward window if reported separately
- **`random_state`:** 42

## 8. Baselines

| Baseline | Definition | Code |
|---|---|---|
| B0 | No-trade (cash at risk-free rate) | `eval/baselines.py::baseline_b0_no_trade` |
| B1 | Equal-weight long straddle every event | `eval/baselines.py::baseline_b1_long_straddle` |
| B2 | Defined-risk short vol (iron butterfly) | `eval/baselines.py::baseline_b2_iron_butterfly` |
| B3 | Dumb logistic baseline (3 features) | `eval/baselines.py::baseline_b3_logistic` |

### 8.1 B0 Risk-Free Rate (locked)

- **Series:** FRED `DGS3MO` (3-month Treasury bill, secondary-market rate)
- **Snapshot date:** 2026-05-05 (most recent value at freeze authoring time)
- **Annual rate:** **3.69%** — hardcoded as `eval/baselines.py::RISK_FREE_RATE_PAPER_V1 = 0.0369`
- **Per-event return:** `(1 + 0.0369) ** (5 / 365) - 1` ≈ 4.97 bps per 5-day hold
- **Refetch policy:** **None.** The cycle does not refetch DGS3MO during the 90-day frozen window.

`max(B0..B3)` per-event provides the "easy alpha" the strategy must clear by ≥200 bps annualised (Gate 3).

## 9. Twin-Model Decision Rule (Satellite)

**Both models:** LightGBM gradient-boosted trees, identical hyperparameters (`max_depth=4`, `n_estimators=500`, `learning_rate=0.05`, `early_stopping_rounds=50`, MSE loss on respective transformed target).

- **Model A (baseline):** target = `net_5day_straddle_return_after_costs`
- **Model B (condition-aligned):** target = `return / max(expected_event_risk, risk_floor)`, where `expected_event_risk = pre_event_ATM_IV_5d × √(5/252)` and `risk_floor = 20th percentile of expected_event_risk` in training data.

| Criterion | Threshold | Computed by |
|---|---|---|
| Net excess return (B − A) | ≥ +50 bps annualised | `signals/twin_model_trainer.py::compare_twin_models` |
| Sharpe delta (B − A) | ≥ +0.2 | (same) |
| Paired t-statistic on per-event diff | ≥ 2.0 | (same) |

Model B is adopted only if **all three** thresholds are met. Otherwise the condition-aligned hypothesis is falsified for cycle 1 — the methodology (the twin-model framework itself) is not falsified, only the cycle 1 hypothesis.

## 10. Audit Data Model

The schema (`sql/schema.sql`) records every gate decision, trade, leg, and LLM activity. All rows are append-only; foreign keys are enforced.

- `runs` — cycle metadata (run_id, prereg_version, methodology_commit, status, initial_nav_usd)
- `gate_decisions` — every candidate's deterministic gate evaluation (signal pass + 6 risk gates + critic verdict + final decision)
- `trades` and `trade_legs` — placed positions with entry/exit Greeks, cost breakdown, realized P&L
- `agents_log` — LLM activity, prompt version, JSON-schema validity, cost (advisory only; never blocks the gate)

The `trades.decision_id` → `gate_decisions.decision_id` link is the canonical audit chain from "why we considered this trade" to "what filled".

## 11. LLM Roles (Advisory Only)

LLMs may be used for code generation, postmortem summaries, and analysis but **never decide trades, never see broker keys, never override the kill switch, never modify the running paper system mid-cycle**. All LLM output is JSON-schema validated; invalid output is discarded. The Critic LLM can flag issues but cannot block trades — only the deterministic risk gate blocks.

Per `strategic_lock.md` § LLM Role Constraints. Activity logged in `agents_log`.

## 12. Decision Framework Post-Cycle

| Outcome | Action |
|---|---|
| All 7 gates pass at base AND +50% costs | Promote to consideration for live capital deployment (≤1% of treasury), subject to CEO + investor written approval per `capital_policy.md` § Live Capital Policy |
| Any gate fails at +50% costs but passes at base | **Pivot** — re-examine cost assumptions, possibly tighten size/strike selection, then run paper_v2 |
| Any gate fails at base | **Stop** — the hypothesis is falsified for cycle 1; do not run paper_v2 of the same hypothesis |

Per `strategic_lock.md` § Authorization, modifications require explicit CEO override AND reset the cycle clock to day 0.

## 13. Historical Data and Simulation Use Policy

Cycle 1's primary evidence is a **frozen, one-shot out-of-sample backtest** over the second year of the available historical window. The first year is training/calibration only; the second year is held out until after `paper_v1.0-frozen` is tagged.

| Window | Role | Evidence status |
|---|---|---|
| Year 1: **2024-05-07 → 2025-05-06 inclusive** | Train Model A and Model B, fit feature transforms, set universe construction, calibrate cost assumptions | In-sample; never counts toward final gates |
| Year 2: **2025-05-07 → 2026-05-06 inclusive** | Frozen models generate predictions; deterministic gates and paper execution replay create the evidence trade log | Out-of-sample; counts toward cycle 1 gates |
| Optional 90-day forward paper run after tag | Real-market confirmation and operational monitoring | Reported separately; not required for cycle 1 verdict |

Hard evidence rules:

1. **Tag first, evaluate second.** `paper_v1.0-frozen` must be created before any Year 2 gate/evidence result is inspected.
2. **Year 2 is never used for training, model selection, threshold tuning, feature engineering, or manual strategy adjustment.** Any use of Year 2 for those purposes converts it into in-sample data and invalidates the cycle.
3. **One-shot eval.** Run the Year 2 evaluator once. Failed gates stand. Any change after viewing Year 2 results creates `paper_v2` and resets the cycle.
4. **Gate 7 counts only held-out Year 2 satellite trades produced by the frozen strategy.** It excludes Year 1 training trades, synthetic/demo rows, and any replay padding after results are known.
5. **Historical simulation remains allowed for engineering validation** only when it does not touch the held-out Year 2 evidence set.
6. **LLMs are read-only during evidence generation.** Weekly LLM analysis may summarize dashboard/results and prepare postmortems, but it may not alter models, features, thresholds, risk gates, costs, code, or metadata inside `paper_v1`.

The optional forward paper run uses the same frozen code and constants. It can strengthen confidence if it agrees with the Year 2 result, but it does not rescue a failed Year 2 gate and does not change the official cycle 1 verdict.

## 14. Out of Scope for Cycle 1

Verbatim from `strategic_lock.md` § What's Out of Scope for Cycle 1:

- Heston / Dupire / SABR calibration in critical path (research-only, may inform Phase 2 if time permits)
- CPCV-options validation (deferred to cycle 2)
- Custom LightGBM Sharpe objective (target engineering substitutes)
- Live broker integration (cycle 2+ only)
- Multi-asset expansion (focus is SPY core + Russell 1000 satellite only)
- Crypto, futures, FX (out of scope entirely for Year 1)

## 15. Sign-Off

**Effective:** This document becomes locked when tagged `paper_v1.0-frozen` in git.
**Reviewed by:** CEO (project owner)
**Locked under:** turbo-carnival project authority, 2026-05-06 (v1.0)

**v0.1 → v1.0 changes (this revision):**

1. § 4.1 (NEW): pinned the 4 adversarial stress scenarios already coded in `risk/envelope.py`.
2. § 6 (NEW): pinned the mixed annualisation conventions (252 trading days for strategy, events/year for twin-model, 365 calendar for B0 cash).
3. § 7.1 (NEW): pinned the Gate 6 HMM specification (2-state Gaussian, SPY returns, held-out evidence-window fit, `random_state=42`).
4. § 8.1 (NEW): pinned the B0 risk-free rate (FRED `DGS3MO` snapshot 2026-05-05 = 3.69%, hardcoded constant, no refetch during cycle).
5. § 7 Gate 7 wording: counts only held-out Year 2 satellite trades from the frozen evidence run.
6. § 13 (NEW): historical-data-and-simulation policy — Year 1 train/calibrate, Year 2 one-shot out-of-sample evidence, optional 90-day forward confirmation.
7. § 13 explicitly bans weekly LLM/model improvements inside `paper_v1`; LLMs are read-only until post-cycle.

---

**No further architecture debate.** The evidence run starts when the CEO tags `paper_v1.0-frozen`. After tag, run the Year 2 out-of-sample evaluator once; optional forward orchestration may follow, but no model/code/threshold improvements are permitted inside `paper_v1`.
