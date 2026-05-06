# Strategic Lock — The Plan

**Version:** v1.0
**Locked:** 2026-05-06
**Authority:** CEO commitment, no further architecture debate during build
**Modification rule:** Changes require explicit CEO override + reset of any in-progress cycle

---

## The Decision

**Path:** Hybrid options strategy
**Engine A (Core, 70% capital):** Volatility risk premium harvesting via SPY iron condors
**Engine B (Satellite, 30% capital):** Earnings volatility selection on Russell 1000 (top-50 by options ADV) using twin-model A/B design with condition-aligned target

This was selected over four alternatives:
- Path A (S&P 100 swing trading) → rejected: alpha-dead at retail scale per FINSABER (Li 2025)
- Path B-US (small/mid event-driven equities) → rejected: wastes PDE/numerical-methods background
- Path B-global (global event-driven) → rejected: operational complexity unjustified at this scale
- Path B-options pure (earnings only, no VRP core) → rejected: single-hypothesis bet, no income floor

Hybrid strictly dominates Path B-options pure as an investor proposition because the VRP core provides a structurally-supported income floor that the satellite alpha hunt does not require for the project to demonstrate value.

## Capital Sizing Rationale

**$10,000 paper notional** — see `capital_policy.md`. This is the smallest viable amount because:

| Constraint | Value | Why this is the floor |
|---|---|---|
| Min positions for diversification (core) | 4-6 simultaneous | Below this, single-position blowup risk dominates |
| Min positions for satellite | 2-4 simultaneous | Below this, sample size <50 events per cycle (gate floor) |
| Cost drag tolerance | <50% of expected returns | Below $10k, costs become the dominant return factor |
| SPX iron condor margin | ~$5,000/contract | Forces SPY-only at $10k (SPX is out of reach) |

Below $10k, one sleeve must be dropped or the gate-floor sample size cannot be reached. We do not test the hybrid hypothesis at sub-$10k capital.

## Universe (Phase 0 → Cycle 1)

### Core Sleeve
- **Underlying:** SPY only (SPX is out of capital reach at $10k)
- **Expiry:** 30-45 DTE on entry, weekly cadence
- **Structure:** Iron condor with $5-wide wings, ~10-15 delta short strikes

### Satellite Sleeve
- **Universe:** Russell 1000 names with liquid weekly options
- **Liquidity filter (eligibility, point-in-time):**
  - Trailing 30-day options ADV ≥ 5,000 contracts
  - Trailing 30-day OI ≥ 10,000 contracts
  - Trailing 30-day median ATM bid-ask spread ≤ 8% of mid
  - ≥ 8 historical earnings events with chain data
- **Cap:** Top 50 names by options ADV per quarter (down from 75 due to $10k notional → satellite size constraint)

## Phased Rollout

| Phase | Weeks | Deliverable |
|---|---|---|
| Phase 0 | 1-2 | Skeleton + dummy SPY-momentum strategy + LIVE DASHBOARD |
| Phase 1 | 3-5 | Polygon Free pipeline + yfinance + vol surface engine + Greeks aggregation |
| Phase 2 | 6-8 | VRP core implementation + risk envelope + scenario stress + baselines |
| Phase 3 | 9-10 | Earnings satellite (twin-model) + Friday spike validates targets |
| Freeze | 11 | `paper_v1_prereg.md` written, locked, git-tagged |
| Frozen Cycle | 12-24 | 90-day paper run, no changes, dashboard live |
| Eval | 25 | Compute gates, postmortem, decide promote/pivot/stop |

## Twin-Model Design (Satellite)

**Both models:** LightGBM gradient-boosted trees, identical hyperparameters
- max_depth=4
- n_estimators=500
- learning_rate=0.05
- early_stopping_rounds=50
- Loss: MSE on respective transformed target

**Model A (baseline):** Target = `net_5day_straddle_return_after_costs`
**Model B (condition-aligned):** Target = `return / max(expected_event_risk, risk_floor)`

Where:
- `expected_event_risk = pre_event_ATM_IV_5d × √(5/252)`
- `risk_floor = 20th percentile of expected_event_risk in training data`

Both compared via paired t-test on per-event differences. Condition-aligned hypothesis is supported only if Model B beats Model A on:
1. Net excess return: B − A ≥ +50 bps annualised
2. Sharpe delta: Sharpe(B) − Sharpe(A) ≥ +0.2
3. Paired t-statistic: t ≥ 2.0

If Model B does not beat Model A by all three margins, the condition-aligned thesis is falsified for cycle 1 (not the methodology — methodology is the test framework itself).

## Pre-Registered Gates (Cycle 1)

All seven must hold simultaneously at Base AND +50% costs:

1. DSR ≥ 0 with t ≥ 3.84
2. PBO < 0.5
3. Net excess vs `max(B0, B1, B2, B3)` ≥ 200 bps annualised
4. Maximum drawdown < 15% NAV
5. Adversarial stress drop < 25% relative
6. Regime stability: no HMM partition underperforming best benchmark by >5%
7. Sample size ≥ 50 events (floor; aim for ≥150 if feasibility allows)

Baselines:
- B0 = no-trade (cash at risk-free rate)
- B1 = equal-weight long straddle (every event)
- B2 = defined-risk short vol (iron butterfly)
- B3 = dumb logistic baseline (3 features only)

## What's Out of Scope for Cycle 1

- Heston/Dupire/SABR calibration in critical path (research-only, may inform Phase 2 if time permits)
- CPCV-options validation (deferred to cycle 2)
- Custom LightGBM Sharpe objective (target engineering substitutes)
- Live broker integration (cycle 2+ only)
- Multi-asset expansion (focus is SPY core + Russell 1000 satellite only)
- Crypto, futures, FX (out of scope entirely for Year 1)

## LLM Role Constraints

- LLMs **never** see broker keys
- LLMs **never** decide trades
- LLMs **never** modify the running paper system mid-cycle
- LLM output is **always** strict JSON validated to schema; invalid output discarded
- The Critic LLM can flag issues but cannot block trades; only deterministic risk gate blocks
- LLM verdicts logged in `eval/`, never in the trade gate path

## Authorization

**Signed by:** CEO commitment, 2026-05-06
**Reviewed by:** Three independent AI critiques (Codex, Perplexity, Claude) converged on hybrid structure
**Effective:** Phase 0 begins immediately
**Modification:** Requires explicit CEO override and resets any in-progress cycle to day 0

---

**No further architecture debate.** The next decisions are implementation details (schema fields, SQL design, code structure) and external dependencies (Polygon free tier signup, IBKR Pro paper account, optional Greenwich WRDS access).
