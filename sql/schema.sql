-- trading_system SQLite schema
-- Version: v0.1 (Phase 0 Day 1)
-- Locked: 2026-05-06
--
-- Design principles:
--   1. Append-only — rows are never updated, only superseded
--   2. Audit trail — every gate decision and trade is logged with full context
--   3. Reproducible — given the same inputs, the same trades emerge
--   4. Strategy-agnostic — same schema serves Phase 0 demo, VRP core, and earnings satellite
--   5. Versioned — every row carries the run_id and prereg version that produced it
--
-- Storage strategy:
--   - SQLite for: trade decisions, gate audit, agent activity, run metadata (small, queryable)
--   - Parquet for: options chains, daily prices, feature snapshots (large, columnar)
--
-- Foreign keys are enforced. Run with:
--   PRAGMA foreign_keys = ON;
-- ============================================================================

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

-- ============================================================================
-- runs: cycle and version metadata
-- One row per run/cycle. The trades, gate decisions, and agent activity all
-- reference the run_id so we can replay or audit any cycle in isolation.
-- ============================================================================
CREATE TABLE IF NOT EXISTS runs (
    run_id              TEXT PRIMARY KEY,                  -- e.g., 'paper_v0_demo', 'paper_v1', 'research_2026-05-06_target-spike'
    run_type            TEXT NOT NULL CHECK(run_type IN (
                            'paper_demo',                  -- Phase 0 dummy strategy
                            'paper_frozen',                -- frozen 90-day cycle
                            'research_experiment',         -- offline research, not in paper lane
                            'eval_replay'                  -- replaying trades for evaluation
                        )),
    prereg_version      TEXT,                              -- git tag of the prereg doc, e.g. 'paper_v1.0-frozen'
    methodology_commit  TEXT,                              -- git SHA of the methodology code at start
    started_at_utc      TEXT NOT NULL,                     -- ISO 8601 UTC
    ended_at_utc        TEXT,                              -- NULL while running
    status              TEXT NOT NULL CHECK(status IN (
                            'planned', 'running', 'completed', 'aborted', 'kill_switched'
                        )),
    initial_nav_usd     REAL NOT NULL,                     -- e.g., 10000.00
    notes               TEXT,                              -- free-form CEO notes
    created_at_utc      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_type ON runs(run_type);

-- ============================================================================
-- gate_decisions: the deterministic gate audit log
-- One row per evaluation of a candidate trade. Records WHY each candidate was
-- approved or blocked. Critical for audit, prereg compliance, and postmortem.
-- ============================================================================
CREATE TABLE IF NOT EXISTS gate_decisions (
    decision_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT NOT NULL REFERENCES runs(run_id),
    decision_at_utc     TEXT NOT NULL,                     -- when the gate evaluated this candidate
    underlying_ticker   TEXT NOT NULL,
    sleeve              TEXT NOT NULL CHECK(sleeve IN ('core', 'satellite', 'demo')),
    candidate_summary   TEXT NOT NULL,                     -- e.g., 'SPY 600C/610C/560P/550P iron condor 30DTE'

    -- Signal layer
    signal_score        REAL,                              -- numeric model output, NULL if no model
    signal_threshold    REAL,                              -- threshold required for approval
    signal_pass         INTEGER NOT NULL CHECK(signal_pass IN (0, 1)),

    -- Risk gate components (each must individually pass)
    risk_spread_pass    INTEGER NOT NULL CHECK(risk_spread_pass IN (0, 1)),
    risk_vega_pass      INTEGER NOT NULL CHECK(risk_vega_pass IN (0, 1)),
    risk_scenario_pass  INTEGER NOT NULL CHECK(risk_scenario_pass IN (0, 1)),
    risk_liquidity_pass INTEGER NOT NULL CHECK(risk_liquidity_pass IN (0, 1)),
    risk_pin_pass       INTEGER NOT NULL CHECK(risk_pin_pass IN (0, 1)),
    risk_concentration_pass INTEGER NOT NULL CHECK(risk_concentration_pass IN (0, 1)),

    -- Critic (advisory only — never blocks alone, but logged)
    critic_verdict      TEXT CHECK(critic_verdict IN ('pass', 'warn', 'fail', NULL)),
    critic_reason       TEXT,

    -- Final decision
    final_decision      TEXT NOT NULL CHECK(final_decision IN ('approved', 'blocked')),
    block_reason        TEXT,                              -- which gate blocked, human-readable

    -- Linked trade (NULL if blocked)
    trade_id            INTEGER,                           -- FK to trades.trade_id, set after trade insertion

    -- Reproducibility
    feature_snapshot_path TEXT,                            -- relative path to Parquet snapshot of features used
    model_version       TEXT,                              -- e.g., 'lightgbm_modelA_v1.2_2026-05-06'

    created_at_utc      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_gate_run ON gate_decisions(run_id);
CREATE INDEX IF NOT EXISTS idx_gate_ticker ON gate_decisions(underlying_ticker);
CREATE INDEX IF NOT EXISTS idx_gate_sleeve ON gate_decisions(sleeve);
CREATE INDEX IF NOT EXISTS idx_gate_decision ON gate_decisions(final_decision);
CREATE INDEX IF NOT EXISTS idx_gate_time ON gate_decisions(decision_at_utc);

-- ============================================================================
-- agents_log: LLM agent activity audit trail
-- Every LLM call is logged here. Provides transparency for the dashboard's
-- "agent activity" panel and enables postmortem of who-did-what.
-- ============================================================================
CREATE TABLE IF NOT EXISTS agents_log (
    log_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT REFERENCES runs(run_id),
    invoked_at_utc      TEXT NOT NULL,
    role                TEXT NOT NULL CHECK(role IN (
                            'coder', 'critic', 'summariser', 'analyst', 'comms', 'verifier'
                        )),
    model               TEXT NOT NULL,                     -- e.g., 'claude-opus-4-7', 'gpt-5.5'
    prompt_version      TEXT NOT NULL,                     -- e.g., 'critic_v1.0' from research_llm/prompts/
    input_summary       TEXT,                              -- short summary; full input stored in artifact_path
    output_summary      TEXT,                              -- short summary; full output stored in artifact_path
    tokens_in           INTEGER,
    tokens_out          INTEGER,
    cost_usd            REAL,
    json_schema_valid   INTEGER NOT NULL DEFAULT 1 CHECK(json_schema_valid IN (0, 1)),
    artifact_path       TEXT,                              -- relative path to full prompt+response JSONL
    duration_ms         INTEGER,
    error               TEXT                               -- non-NULL if call failed
);

CREATE INDEX IF NOT EXISTS idx_agents_run ON agents_log(run_id);
CREATE INDEX IF NOT EXISTS idx_agents_role ON agents_log(role);
CREATE INDEX IF NOT EXISTS idx_agents_time ON agents_log(invoked_at_utc);
CREATE INDEX IF NOT EXISTS idx_agents_invalid ON agents_log(json_schema_valid) WHERE json_schema_valid = 0;

-- ============================================================================
-- TODO(human): trades table
-- ============================================================================
-- This is the most load-bearing table in the schema — it captures the actual
-- trades the paper system places. Many valid designs exist; the right one
-- depends on what we want to query and audit later.
--
-- REQUIREMENTS (must be captured somewhere):
--   - Unique trade ID
--   - Run ID (FK to runs)
--   - Decision ID (FK to gate_decisions, since every trade comes from an
--     approved gate decision)
--   - Sleeve ('core' | 'satellite' | 'demo')
--   - Underlying ticker
--   - Strategy structure: iron_condor | long_straddle | short_straddle |
--     iron_butterfly | demo_long_equity (extensible)
--   - Position legs (multi-leg for iron condors, two for straddles, one for
--     demo equity) — each with: type (call/put/stock), strike, expiry,
--     direction (long/short), quantity, entry_fill_price, exit_fill_price
--   - Entry timestamp + intended fill + actual simulated fill
--   - Exit timestamp + intended fill + actual simulated fill (NULL if open)
--   - Greeks at entry (delta, gamma, theta, vega, rho)
--   - Greeks at exit (NULL if open)
--   - Realized P&L (NULL if open)
--   - Cost breakdown: commission + spread + slippage estimate
--   - Status: 'open' | 'closed' | 'expired' | 'force_closed'
--
-- DESIGN DECISIONS YOU MUST MAKE:
--
-- 1. LEGS: in this table (denormalized JSON column) OR separate trade_legs
--    table (normalized)? Iron condors have 4 legs; straddles have 2; demo
--    equity has 1.
--    Tradeoffs:
--      - Denormalized JSON: simpler queries for "show me this trade"; harder
--        to query across legs (e.g., "all positions short the 600 call")
--      - Normalized trade_legs table: cleaner queries; more complex inserts
--
-- 2. PRIMARY KEY: auto-incrementing INTEGER, UUID, ULID (sortable), or
--    composite (run_id, sequence_number)?
--    Tradeoffs:
--      - INTEGER: simplest, fast joins, but not globally unique across runs
--      - UUID: globally unique but unsortable, not human-friendly
--      - ULID: globally unique AND sortable by time
--      - Composite: encodes run order but more verbose in joins
--
-- 3. GREEKS REPRESENTATION: separate columns (delta REAL, gamma REAL, ...)
--    or JSON column with all Greeks?
--    Tradeoffs:
--      - Columns: indexable, queryable directly
--      - JSON: extensible, future-proof if we add second-order Greeks
--
-- 4. P&L LIFECYCLE: store realized_pnl on close OR compute on read from
--    entry/exit fills? Compute-on-read is more reproducible (no risk of
--    storing wrong value) but slower for repeated queries.
--
-- TASK: design and write the CREATE TABLE statement(s) for trades, including
-- any auxiliary tables (e.g., trade_legs) you decide to add. Add the relevant
-- indexes. Write a 1-2 sentence comment block above explaining your design
-- choices for the four decisions above.
-- ============================================================================

-- (your CREATE TABLE statements go here)


-- ============================================================================
-- Future tables (NOT for Phase 0 — placeholder comment only)
-- ============================================================================
-- TODO(later): feature_snapshots — Parquet pointer + checksum, references
--   gate_decisions for replay capability. Not implementing in Phase 0;
--   features stored as Parquet with date-partitioning.
--
-- TODO(later): position_state — current open positions snapshot. Derivable
--   from trades.status='open' for now; promote to dedicated table only if
--   query performance demands it.
--
-- TODO(later): cycle_metrics — per-cycle aggregated DSR, PBO, max DD, etc.
--   Computed from trades + gate_decisions. Stored as a materialized view
--   or batch-computed Parquet, not a maintained table.
-- ============================================================================
