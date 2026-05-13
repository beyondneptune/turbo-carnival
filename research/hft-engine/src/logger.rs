//! SQLite logger.
//!
//! One DB at `config.sqlite_path`. Five tables:
//!   * decisions       — one row per decision cycle (fired or not, with reasoning JSON)
//!   * orders          — one row per Alpaca order side (entry + exit)
//!   * latency_samples — periodic round-trip measurements
//!   * nav_snapshots   — NAV/peak/drawdown at each closed trade
//!   * regime_events   — halts, kill-file detection, latency-breach events, etc.
//!
//! We write through a small wrapper so the rest of the codebase can call
//! `log_decision(...)` without juggling Connection lifetimes. Connection is
//! wrapped in a Mutex (sqlite is single-writer anyway, contention is trivial
//! at 30s cadence).

use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Clone)]
pub struct SqliteLogger {
    conn: Arc<Mutex<Connection>>,
}

impl SqliteLogger {
    pub async fn open(path: &str) -> Result<Self> {
        if let Some(parent) = Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).ok();
            }
        }
        let conn = Connection::open(path).with_context(|| format!("open sqlite {path}"))?;
        conn.execute_batch(
            r#"
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;

            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                fired INTEGER NOT NULL,
                predicted_bps REAL,
                confidence REAL,
                spread_bps REAL,
                latency_p99_ms REAL,
                spy_mid REAL,
                es_close REAL,
                reasoning TEXT
            );

            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                side TEXT NOT NULL,
                symbol TEXT NOT NULL,
                qty REAL NOT NULL,
                limit_price REAL,
                alpaca_order_id TEXT,
                http_status INTEGER,
                response_excerpt TEXT,
                dry_run INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS latency_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                rtt_micros INTEGER NOT NULL,
                http_status INTEGER NOT NULL,
                ok INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS nav_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                nav_usd REAL NOT NULL,
                peak_nav_usd REAL NOT NULL,
                drawdown_pct REAL NOT NULL,
                consecutive_losses INTEGER NOT NULL,
                halted INTEGER NOT NULL,
                note TEXT
            );

            CREATE TABLE IF NOT EXISTS regime_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                kind TEXT NOT NULL,
                detail TEXT
            );

            CREATE TABLE IF NOT EXISTS shadow_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                spy_mid REAL NOT NULL,
                spy_bid REAL,
                spy_ask REAL,
                spread_bps REAL,
                predicted_bps REAL,
                confidence REAL,
                latency_p99_ms REAL,
                samples INTEGER,
                would_fire INTEGER NOT NULL,
                actual_spy_30s REAL,
                actual_spy_60s REAL,
                actual_spy_90s REAL,
                actual_bps_30s REAL,
                actual_bps_60s REAL,
                actual_bps_90s REAL,
                evaluated INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_shadow_ts
              ON shadow_predictions(ts);
            CREATE INDEX IF NOT EXISTS idx_shadow_evaluated
              ON shadow_predictions(evaluated);
            "#,
        )
        .context("creating tables")?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    pub fn raw(&self) -> Arc<Mutex<Connection>> {
        self.conn.clone()
    }

    pub async fn log_decision(
        &self,
        fired: bool,
        predicted_bps: Option<f64>,
        confidence: Option<f64>,
        spread_bps: Option<f64>,
        latency_p99_ms: Option<f64>,
        spy_mid: Option<f64>,
        es_close: Option<f64>,
        reasoning: &str,
    ) -> Result<()> {
        let ts = chrono::Utc::now().to_rfc3339();
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO decisions
             (ts, fired, predicted_bps, confidence, spread_bps, latency_p99_ms, spy_mid, es_close, reasoning)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![ts, fired as i64, predicted_bps, confidence, spread_bps, latency_p99_ms, spy_mid, es_close, reasoning],
        )
        .context("insert decisions")?;
        Ok(())
    }

    pub async fn log_order(
        &self,
        side: &str,
        symbol: &str,
        qty: f64,
        limit_price: Option<f64>,
        alpaca_order_id: Option<&str>,
        http_status: Option<u16>,
        response_excerpt: Option<&str>,
        dry_run: bool,
    ) -> Result<()> {
        let ts = chrono::Utc::now().to_rfc3339();
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO orders (ts, side, symbol, qty, limit_price, alpaca_order_id, http_status, response_excerpt, dry_run)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![ts, side, symbol, qty, limit_price, alpaca_order_id, http_status.map(|v| v as i64), response_excerpt, dry_run as i64],
        )
        .context("insert orders")?;
        Ok(())
    }

    pub async fn log_latency(
        &self,
        rtt_micros: u64,
        http_status: u16,
        ok: bool,
        at: chrono::DateTime<chrono::Utc>,
    ) -> Result<()> {
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO latency_samples (ts, rtt_micros, http_status, ok) VALUES (?, ?, ?, ?)",
            params![at.to_rfc3339(), rtt_micros as i64, http_status as i64, ok as i64],
        )
        .context("insert latency_samples")?;
        Ok(())
    }

    pub async fn log_nav_snapshot(
        &self,
        nav_usd: f64,
        peak_nav_usd: f64,
        drawdown_pct: f64,
        consecutive_losses: u32,
        halted: bool,
        note: &str,
    ) -> Result<()> {
        let ts = chrono::Utc::now().to_rfc3339();
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO nav_snapshots (ts, nav_usd, peak_nav_usd, drawdown_pct, consecutive_losses, halted, note)
             VALUES (?, ?, ?, ?, ?, ?, ?)",
            params![ts, nav_usd, peak_nav_usd, drawdown_pct, consecutive_losses, halted as i64, note],
        )
        .context("insert nav_snapshots")?;
        Ok(())
    }

    pub async fn log_regime(&self, kind: &str, detail: &str) -> Result<()> {
        let ts = chrono::Utc::now().to_rfc3339();
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO regime_events (ts, kind, detail) VALUES (?, ?, ?)",
            params![ts, kind, detail],
        )
        .context("insert regime_events")?;
        Ok(())
    }

    /// Log a shadow prediction. Called EVERY decision cycle when shadow_only is
    /// active — independent of whether the gate would have fired. Outcomes are
    /// filled in later by `update_shadow_outcome` once enough time has passed.
    pub async fn log_shadow_prediction(
        &self,
        spy_mid: f64,
        spy_bid: Option<f64>,
        spy_ask: Option<f64>,
        spread_bps: Option<f64>,
        predicted_bps: Option<f64>,
        confidence: Option<f64>,
        latency_p99_ms: Option<f64>,
        samples: Option<u32>,
        would_fire: bool,
    ) -> Result<i64> {
        let ts = chrono::Utc::now().to_rfc3339();
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO shadow_predictions
             (ts, spy_mid, spy_bid, spy_ask, spread_bps, predicted_bps, confidence,
              latency_p99_ms, samples, would_fire)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                ts, spy_mid, spy_bid, spy_ask, spread_bps, predicted_bps, confidence,
                latency_p99_ms, samples.map(|v| v as i64), would_fire as i64,
            ],
        ).context("insert shadow_predictions")?;
        Ok(conn.last_insert_rowid())
    }

    /// Pending rows that still need outcome backfill (evaluated=0).
    /// Returns (id, ts_rfc3339, spy_mid_at_log).
    pub async fn pending_shadow_rows(&self, limit: usize) -> Result<Vec<(i64, String, f64)>> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(
            "SELECT id, ts, spy_mid FROM shadow_predictions
             WHERE evaluated = 0 ORDER BY id ASC LIMIT ?",
        ).context("prepare pending shadow")?;
        let rows = stmt
            .query_map([limit as i64], |r| {
                Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?, r.get::<_, f64>(2)?))
            })?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    /// Fill in actual SPY values + bps moves at one or more horizons. Sets
    /// `evaluated=1` when all three horizons are populated.
    pub async fn update_shadow_outcome(
        &self,
        id: i64,
        spy_at_30s: Option<f64>,
        spy_at_60s: Option<f64>,
        spy_at_90s: Option<f64>,
        bps_at_30s: Option<f64>,
        bps_at_60s: Option<f64>,
        bps_at_90s: Option<f64>,
        mark_evaluated: bool,
    ) -> Result<()> {
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE shadow_predictions SET
                actual_spy_30s = COALESCE(?, actual_spy_30s),
                actual_spy_60s = COALESCE(?, actual_spy_60s),
                actual_spy_90s = COALESCE(?, actual_spy_90s),
                actual_bps_30s = COALESCE(?, actual_bps_30s),
                actual_bps_60s = COALESCE(?, actual_bps_60s),
                actual_bps_90s = COALESCE(?, actual_bps_90s),
                evaluated = MAX(evaluated, ?)
             WHERE id = ?",
            params![
                spy_at_30s, spy_at_60s, spy_at_90s,
                bps_at_30s, bps_at_60s, bps_at_90s,
                mark_evaluated as i64,
                id,
            ],
        ).context("update shadow_predictions")?;
        Ok(())
    }
}
