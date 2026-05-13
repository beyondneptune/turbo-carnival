//! NAV tracker.
//!
//! Single-row SQLite-backed state. Tracks current NAV, peak NAV (for drawdown),
//! and consecutive losing trades. Cap at `initial_nav_usd` from config — we
//! refuse to let book size grow above the configured cap even if PnL is +ve
//! (this is a paper-research engine; we don't want NAV inflation distorting
//! the size of trades).

use anyhow::{Context, Result};
use rusqlite::{params, Connection};

#[derive(Debug, Clone)]
pub struct NavState {
    pub nav_usd: f64,
    pub peak_nav_usd: f64,
    pub consecutive_losses: u32,
    pub halted: bool,
}

impl NavState {
    pub fn drawdown_pct(&self) -> f64 {
        if self.peak_nav_usd <= 0.0 {
            return 0.0;
        }
        (self.peak_nav_usd - self.nav_usd) / self.peak_nav_usd
    }
}

/// Load NAV state. Initialises a fresh row at `initial_nav` if none exists.
pub fn load_or_init(conn: &Connection, initial_nav: f64) -> Result<NavState> {
    conn.execute(
        "CREATE TABLE IF NOT EXISTS nav_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            nav_usd REAL NOT NULL,
            peak_nav_usd REAL NOT NULL,
            consecutive_losses INTEGER NOT NULL DEFAULT 0,
            halted INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        )",
        [],
    )
    .context("create nav_state")?;

    let existing: Option<(f64, f64, u32, i64)> = conn
        .query_row(
            "SELECT nav_usd, peak_nav_usd, consecutive_losses, halted FROM nav_state WHERE id = 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .ok();

    if let Some((nav, peak, losses, halted)) = existing {
        return Ok(NavState {
            nav_usd: nav,
            peak_nav_usd: peak,
            consecutive_losses: losses,
            halted: halted != 0,
        });
    }

    let now = chrono::Utc::now().to_rfc3339();
    conn.execute(
        "INSERT INTO nav_state (id, nav_usd, peak_nav_usd, consecutive_losses, halted, updated_at)
         VALUES (1, ?, ?, 0, 0, ?)",
        params![initial_nav, initial_nav, now],
    )
    .context("init nav_state")?;
    Ok(NavState {
        nav_usd: initial_nav,
        peak_nav_usd: initial_nav,
        consecutive_losses: 0,
        halted: false,
    })
}

/// Apply a realized PnL (USD) to NAV, update peak, persist, return new state.
pub fn apply_pnl(conn: &Connection, st: &NavState, pnl_usd: f64) -> Result<NavState> {
    let mut next = st.clone();
    next.nav_usd += pnl_usd;
    if next.nav_usd > next.peak_nav_usd {
        next.peak_nav_usd = next.nav_usd;
    }
    if pnl_usd < 0.0 {
        next.consecutive_losses = next.consecutive_losses.saturating_add(1);
    } else if pnl_usd > 0.0 {
        next.consecutive_losses = 0;
    }
    persist(conn, &next)?;
    Ok(next)
}

/// Mark the state as halted (e.g. drawdown breach) and persist.
pub fn mark_halted(conn: &Connection, st: &NavState) -> Result<NavState> {
    let mut next = st.clone();
    next.halted = true;
    persist(conn, &next)?;
    Ok(next)
}

fn persist(conn: &Connection, st: &NavState) -> Result<()> {
    let now = chrono::Utc::now().to_rfc3339();
    conn.execute(
        "UPDATE nav_state
         SET nav_usd = ?, peak_nav_usd = ?, consecutive_losses = ?, halted = ?, updated_at = ?
         WHERE id = 1",
        params![st.nav_usd, st.peak_nav_usd, st.consecutive_losses, st.halted as i64, now],
    )
    .context("persist nav_state")?;
    Ok(())
}
