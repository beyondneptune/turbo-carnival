# Phase Status

**Phase:** All code phases complete + prereg locked at v1.0 — awaiting CEO `git tag paper_v1.0-frozen`
**Machine:** Mac mini (production target; Phase 0 development allowed)
**Frozen cycle:** Not started — code changes are permitted
**Last updated:** 2026-05-06

---

## Completed

- Project skeleton (directory layout per `README.md`)
- `metadata/strategic_lock.md` — locked hybrid VRP core + earnings satellite strategy
- `metadata/capital_policy.md` — $10k paper notional, sleeve allocation, kill switch limits
- `sql/schema.sql` — `runs`, `gate_decisions`, `agents_log`, `trades`, `trade_legs`
- `dashboard/app.py` — read-only Streamlit dashboard skeleton (overview, runs, gate decisions, trades, agent activity)
- `scripts/run_phase0_demo.py` — synthetic SPY-momentum demo seed; **append-only by default** (auto-generated unique `run_id`; `--run-id <id>` and `--reset-demo` flags); populates all 5 tables for one demo run
- `tests/` + root `conftest.py` — pytest coverage for schema constraints (16 tests) and dashboard helpers (14 tests)

### Phase 1 (ingest + features)

- `ingest/yfinance_adapter.py` — yfinance daily price adapter (8 tests)
- `ingest/polygon_free.py` — Polygon REST scaffold w/ rate limiting (22 tests, mocked)
- `features/greeks.py` — Black-Scholes Greeks + leg aggregation (30 tests, QuantLib-validated to 1e-14)
- `features/vol_surface.py` — non-parametric implied vol surface (32 tests)

### Phase 2 (signals + risk + paper exec)

- `signals/vrp_core_spy.py` — SPY iron condor structurer + entry/exit rules (38 tests)
- `risk/envelope.py` — pre-trade risk gates (vega caps, concentration, scenario stress) (21 tests)
- `risk/kill_switch.py` — deterministic kill switch (intraday loss, cycle DD, cash buffer) (20 tests)
- `paper_exec/simulator.py` — fill simulator with slippage + commission (42 tests)

### Phase 3 (earnings satellite)

- `signals/earnings_universe.py` — Russell 1000 liquidity filter, top-50 by options ADV (25 tests)
- `signals/earnings_satellite.py` — twin-model targets + ATM straddle structurer (41 tests)
- `signals/twin_model_trainer.py` — LightGBM A/B trainer with paired-t evaluation (19 tests)

### Eval framework

- `eval/baselines.py` — B0 (cash@rf), B1 (long straddle), B2 (iron butterfly), B3 (dumb logistic) (33 tests)
- `eval/metrics.py` — DSR, PBO via CSCV, Sharpe, max DD, HMM regime stability (38 tests)
- `eval/gates.py` — the 7 pre-registered gate evaluator at base + 50% costs (32 tests)

### Freeze prep

- `metadata/paper_v1_prereg.md` — **pre-registration v1.0** (15 sections: now includes Annualisation Conventions, Adversarial Stress Scenarios, Gate 6 HMM specs, B0 risk-free rate snapshot, and Historical Data / Simulation Use Policy)

**Total tests: 431, all passing in ~2s** under `.venv` (Python 3.11.15, libomp installed for lightgbm).

## Locked at v1.0 (CEO has the prereg)

| # | Decision | Locked value |
|---|---|---|
| 1 | B0 risk-free rate | FRED `DGS3MO` snapshot 2026-05-05 = **3.69%**, hardcoded as `eval/baselines.py::RISK_FREE_RATE_PAPER_V1`, no refetch in cycle |
| 2 | Gate 6 HMM specs | 2-state Gaussian HMM, daily SPY returns, cycle-window fit, `random_state=42` |
| 3 | Adversarial stress scenarios | 4 pre-registered: −5%/+10vol, **−10%/+30vol (adversarial)**, +5%/+5vol, flat/−5vol |
| 4 | Annualisation conventions | 252 trading days for strategy/Sharpe; 252/5 events/year for twin-model; 365 calendar geometric for B0 cash |
| 5 | Gate 7 sample-size scope | Live cycle satellite trades only — no walk-forward replay padding |

## Next

**Wall-clock and decision-bound** — coding is done.

- CEO `git tag paper_v1.0-frozen && git push origin paper_v1.0-frozen` (your hand on the tag)
- Operational prep before day 0 actually runs: Polygon Free key in `.env`, daily orchestrator script (`scripts/run_daily_paper.py`), cron / launchd schedule
- Pre-freeze dry run on live market data Mon–Fri this week (allowed per § 13 of prereg) — plumbing verification only, NOT cycle evidence
- Day 0 = first day daily orchestrator runs successfully; Day 90 = cycle eval via `eval/gates.py::evaluate_full`

## Active Constraint

Once the `paper_v1` frozen cycle begins, **no code changes are permitted mid-cycle**. Any change resets the cycle clock to day 0 per `metadata/strategic_lock.md` § Modification rule. During Phase 0 (now), normal development applies.

---

**Update rule:** Edit this file as Phase 0 milestones land. When `paper_v1` is tagged and the cycle starts, switch the heading to `Cycle: paper_v1 frozen, day N of 90`.
