# Phase Status

**Phase:** Phase 0 complete — awaiting Phase 1 kickoff
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
- `tests/` + root `conftest.py` — pytest coverage for schema constraints (16 tests) and dashboard helpers (14 tests); 30 tests, all passing in <1s

## Next

**Phase 0 deliverables complete.** Phase 1 begins on next CEO greenlight (weeks 3–5 per `metadata/strategic_lock.md`):

- Polygon Free pipeline (`ingest/`)
- yfinance adapter for daily SPY + Russell 1000 prices
- Vol surface engine + Greeks aggregation (`features/`)

## Active Constraint

Once the `paper_v1` frozen cycle begins, **no code changes are permitted mid-cycle**. Any change resets the cycle clock to day 0 per `metadata/strategic_lock.md` § Modification rule. During Phase 0 (now), normal development applies.

---

**Update rule:** Edit this file as Phase 0 milestones land. When `paper_v1` is tagged and the cycle starts, switch the heading to `Cycle: paper_v1 frozen, day N of 90`.
