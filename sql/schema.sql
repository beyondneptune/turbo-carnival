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
-- nav_history: one row per paper run day for kill-switch and return bootstrap
-- ============================================================================
CREATE TABLE IF NOT EXISTS nav_history (
    run_id              TEXT NOT NULL REFERENCES runs(run_id),
    date                TEXT NOT NULL,                     -- ISO 'YYYY-MM-DD'
    nav                 REAL NOT NULL,
    intraday_low        REAL NOT NULL,
    created_at_utc      TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (run_id, date)
);

CREATE INDEX IF NOT EXISTS idx_nav_history_run_date ON nav_history(run_id, date);

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
    trade_id            INTEGER,                           -- denormalized cache; canonical audit link is trades.decision_id (one-way, no circular FK)

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
-- trades + trade_legs: the trade ledger
-- ============================================================================
-- Design (Phase 0 Day 1): legs normalized into trade_legs (clean cross-leg
-- queries), INTEGER AUTOINCREMENT PKs (simple, fast joins), Greeks as
-- explicit REAL columns (entry+exit aggregate on trades, entry per-leg on
-- trade_legs), and realized_pnl_usd cached on close with per-leg fills +
-- trade-level cost breakdown retained for audit recompute. Audit link to
-- gate_decisions is one-way via trades.decision_id → gate_decisions.decision_id;
-- gate_decisions.trade_id stays as an unenforced convenience field to avoid
-- a circular FK (lookup the inverse with `SELECT trade_id FROM trades WHERE
-- decision_id = ?`).
-- ============================================================================
CREATE TABLE IF NOT EXISTS trades (
    trade_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT NOT NULL REFERENCES runs(run_id),
    decision_id         INTEGER NOT NULL UNIQUE REFERENCES gate_decisions(decision_id),
    sleeve              TEXT NOT NULL CHECK(sleeve IN ('core', 'satellite', 'demo')),
    strategy_type       TEXT NOT NULL CHECK(strategy_type IN (
                            'iron_condor',
                            'long_straddle',
                            'short_straddle',
                            'iron_butterfly',
                            'demo_long_equity'
                        )),
    underlying_ticker   TEXT NOT NULL,

    status              TEXT NOT NULL CHECK(status IN (
                            'open', 'closed', 'expired', 'force_closed'
                        )),
    entry_at_utc        TEXT NOT NULL,                     -- ISO 8601 UTC
    exit_at_utc         TEXT,                              -- NULL while open

    -- Aggregate Greeks at entry (signed, contract-level)
    entry_delta         REAL,
    entry_gamma         REAL,
    entry_theta         REAL,
    entry_vega          REAL,
    entry_rho           REAL,

    -- Aggregate Greeks at exit (NULL while open)
    exit_delta          REAL,
    exit_gamma          REAL,
    exit_theta          REAL,
    exit_vega           REAL,
    exit_rho            REAL,

    -- Cost & P&L (USD). realized_pnl_usd is a cache populated on close;
    -- per-leg fills in trade_legs plus this breakdown allow recomputation.
    commission_usd      REAL CHECK(commission_usd >= 0),
    spread_cost_usd     REAL CHECK(spread_cost_usd >= 0),  -- bid-ask cost estimate
    slippage_usd        REAL,                              -- intended − actual, summed over legs
    realized_pnl_usd    REAL,                              -- NULL while open

    notes               TEXT,
    created_at_utc      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_trades_run ON trades(run_id);
CREATE INDEX IF NOT EXISTS idx_trades_decision ON trades(decision_id);
CREATE INDEX IF NOT EXISTS idx_trades_underlying ON trades(underlying_ticker);
CREATE INDEX IF NOT EXISTS idx_trades_sleeve ON trades(sleeve);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_entry ON trades(entry_at_utc);

-- ----------------------------------------------------------------------------
-- trade_legs: one row per leg. Iron condor = 4 legs, straddle = 2, demo
-- equity = 1. Direction is encoded as a string; quantity is positive.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS trade_legs (
    leg_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id            INTEGER NOT NULL REFERENCES trades(trade_id),
    leg_number          INTEGER NOT NULL,                  -- 1..N within the trade
    instrument_type     TEXT NOT NULL CHECK(instrument_type IN ('call', 'put', 'stock')),
    direction           TEXT NOT NULL CHECK(direction IN ('long', 'short')),
    quantity            INTEGER NOT NULL CHECK(quantity > 0),
    multiplier          INTEGER NOT NULL DEFAULT 100 CHECK(multiplier > 0),  -- 100 for options, 1 for stock
    strike              REAL,                              -- NULL for stock
    expiry              TEXT,                              -- ISO 'YYYY-MM-DD', NULL for stock

    -- Fills: intended (decision-time mid) vs actual (simulated)
    entry_intended_price REAL,
    entry_fill_price    REAL NOT NULL,
    exit_intended_price REAL,
    exit_fill_price     REAL,                              -- NULL while leg open

    -- Per-leg Greeks at entry (risk decomposition / audit)
    entry_delta         REAL,
    entry_gamma         REAL,
    entry_theta         REAL,
    entry_vega          REAL,
    entry_rho           REAL,

    created_at_utc      TEXT NOT NULL DEFAULT (datetime('now')),

    UNIQUE(trade_id, leg_number)
);

CREATE INDEX IF NOT EXISTS idx_legs_trade ON trade_legs(trade_id);
CREATE INDEX IF NOT EXISTS idx_legs_expiry ON trade_legs(expiry);


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
