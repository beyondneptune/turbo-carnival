# Phase Status

**Phase:** Phase 2 complete — Phase 3 (earnings satellite) + eval framework + freeze prep next
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
- `scripts/run_phase0_demo.py` — synthetic SPY-momentum demo seed (idempotent; populates all 5 tables under `run_id = paper_v0_demo`)
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

**Total tests: 243, all passing in <2s** under `.venv` (Python 3.11.15, libomp installed for lightgbm).

## Next

**Phase 3 + eval + freeze** (weeks 9–11 per `metadata/strategic_lock.md`):

- `signals/earnings_universe.py` — Russell 1000 universe filter by options ADV
- `signals/earnings_satellite.py` — twin-model A/B framework (return target vs condition-aligned target)
- `signals/twin_model_trainer.py` — LightGBM training + paired-t evaluation harness
- `eval/baselines.py` — B0 (no-trade), B1 (long straddle), B2 (iron butterfly), B3 (dumb logistic)
- `eval/metrics.py` — DSR, PBO, Sharpe, max DD, regime stability
- `eval/gates.py` — the 7 pre-registered gate evaluator
- `metadata/paper_v1_prereg.md` — pre-registration document for the frozen cycle
- (Wall-clock dependent: 90-day frozen cycle and final eval cannot be fast-forwarded)

## Active Constraint

Once the `paper_v1` frozen cycle begins, **no code changes are permitted mid-cycle**. Any change resets the cycle clock to day 0 per `metadata/strategic_lock.md` § Modification rule. During Phase 0 (now), normal development applies.

---

**Update rule:** Edit this file as Phase 0 milestones land. When `paper_v1` is tagged and the cycle starts, switch the heading to `Cycle: paper_v1 frozen, day N of 90`.
