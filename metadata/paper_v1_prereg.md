# Pre-Registration: paper_v1

**Version:** v0.1 (pre-freeze draft)
**Cycle ID:** paper_v1
**Prereg authored:** 2026-05-06
**Freeze target:** week 11 of project timeline (post Phase 3)
**Cycle duration:** 90 calendar days, frozen
**Modification rule:** Once frozen via git tag `paper_v1.0-frozen`, any change to this document, the strategy code, or the metadata files resets the cycle clock to day 0.

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
- **Selection:** Twin-model (Model A vs Model B) — Model B used **only** if all three twin-model thresholds met (§ 8)
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

## 5. Cost Model

- **Commission:** $0.65 / contract IBKR Pro
- **Spread cost:** half-spread per leg
- **Slippage:** $0.01 / share default (recorded as intended − actual on every fill in `trade_legs`)
- **Cost stress:** gate must hold at **base costs AND +50% costs** (spread × 1.5)
- **+100% costs:** informational only, reported in postmortem, non-blocking

## 6. Pre-Registered Gates (7)

All seven must hold simultaneously at **Base AND +50% costs**. Verbatim from `strategic_lock.md` § Pre-Registered Gates.

| # | Gate | Threshold | Computed by |
|---|---|---|---|
| 1 | DSR | DSR ≥ 0 with t ≥ 3.84 | `eval/metrics.py::deflated_sharpe_ratio` |
| 2 | PBO | PBO < 0.5 | `eval/metrics.py::probability_backtest_overfitting` |
| 3 | Net excess vs `max(B0..B3)` | ≥ 200 bps annualised | `eval/baselines.py` + `eval/gates.py::evaluate_net_excess` |
| 4 | Max drawdown | < 15% NAV | `eval/metrics.py::max_drawdown` |
| 5 | Adversarial stress drop | < 25% relative | `risk/envelope.py::check_scenario_stress` |
| 6 | Regime stability | no HMM partition underperforming best benchmark by >5% | `eval/metrics.py::evaluate_regime_stability` |
| 7 | Sample size | ≥ 50 events (floor; aim ≥150 if feasibility allows) | counted from `trades` where `sleeve='satellite'` |

Composite evaluator: `eval/gates.py::evaluate_full`.

## 7. Baselines

| Baseline | Definition | Code |
|---|---|---|
| B0 | No-trade (cash at risk-free rate) | `eval/baselines.py::baseline_b0_no_trade` |
| B1 | Equal-weight long straddle every event | `eval/baselines.py::baseline_b1_long_straddle` |
| B2 | Defined-risk short vol (iron butterfly) | `eval/baselines.py::baseline_b2_iron_butterfly` |
| B3 | Dumb logistic baseline (3 features) | `eval/baselines.py::baseline_b3_logistic` |

`max(B0..B3)` per-event provides the "easy alpha" that the strategy must clear by ≥200 bps annualised (Gate 3).

## 8. Twin-Model Decision Rule (Satellite)

**Both models:** LightGBM gradient-boosted trees, identical hyperparameters (`max_depth=4`, `n_estimators=500`, `learning_rate=0.05`, `early_stopping_rounds=50`, MSE loss on respective transformed target).

- **Model A (baseline):** target = `net_5day_straddle_return_after_costs`
- **Model B (condition-aligned):** target = `return / max(expected_event_risk, risk_floor)`, where `expected_event_risk = pre_event_ATM_IV_5d × √(5/252)` and `risk_floor = 20th percentile of expected_event_risk` in training data.

| Criterion | Threshold | Computed by |
|---|---|---|
| Net excess return (B − A) | ≥ +50 bps annualised | `signals/twin_model_trainer.py::compare_twin_models` |
| Sharpe delta (B − A) | ≥ +0.2 | (same) |
| Paired t-statistic on per-event diff | ≥ 2.0 | (same) |

Model B is adopted only if **all three** thresholds are met. Otherwise the condition-aligned hypothesis is falsified for cycle 1 — the methodology (the twin-model framework itself) is not falsified, only the cycle 1 hypothesis.

## 9. Audit Data Model

The schema (`sql/schema.sql`) records every gate decision, trade, leg, and LLM activity. All rows are append-only; foreign keys are enforced.

- `runs` — cycle metadata (run_id, prereg_version, methodology_commit, status, initial_nav_usd)
- `gate_decisions` — every candidate's deterministic gate evaluation (signal pass + 6 risk gates + critic verdict + final decision)
- `trades` and `trade_legs` — placed positions with entry/exit Greeks, cost breakdown, realized P&L
- `agents_log` — LLM activity, prompt version, JSON-schema validity, cost (advisory only; never blocks the gate)

The `trades.decision_id` → `gate_decisions.decision_id` link is the canonical audit chain from "why we considered this trade" to "what filled".

## 10. LLM Roles (Advisory Only)

LLMs may be used for code generation, postmortem summaries, and analysis but **never decide trades, never see broker keys, never override the kill switch, never modify the running paper system mid-cycle**. All LLM output is JSON-schema validated; invalid output is discarded. The Critic LLM can flag issues but cannot block trades — only the deterministic risk gate blocks.

Per `strategic_lock.md` § LLM Role Constraints. Activity logged in `agents_log`.

## 11. Decision Framework Post-Cycle

| Outcome | Action |
|---|---|
| All 7 gates pass at base AND +50% costs | Promote to consideration for live capital deployment (≤1% of treasury), subject to CEO + investor written approval per `capital_policy.md` § Live Capital Policy |
| Any gate fails at +50% costs but passes at base | **Pivot** — re-examine cost assumptions, possibly tighten size/strike selection, then run paper_v2 |
| Any gate fails at base | **Stop** — the hypothesis is falsified for cycle 1; do not run paper_v2 of the same hypothesis |

Per `strategic_lock.md` § Authorization, modifications require explicit CEO override AND reset the cycle clock to day 0.

## 12. Out of Scope for Cycle 1

Verbatim from `strategic_lock.md` § What's Out of Scope for Cycle 1:

- Heston / Dupire / SABR calibration in critical path (research-only, may inform Phase 2 if time permits)
- CPCV-options validation (deferred to cycle 2)
- Custom LightGBM Sharpe objective (target engineering substitutes)
- Live broker integration (cycle 2+ only)
- Multi-asset expansion (focus is SPY core + Russell 1000 satellite only)
- Crypto, futures, FX (out of scope entirely for Year 1)

## 13. Sign-Off

**Effective:** This document becomes locked when tagged `paper_v1.0-frozen` in git.
**Reviewed by:** CEO (project owner)
**Locked under:** turbo-carnival project authority, 2026-05-06 (draft)

---

**No further architecture debate. The next decisions are operational (start the 90-day cycle).**
