//! Configuration loader.
//!
//! Loads YAML config + credentials from env (dotenv'd). Single immutable struct
//! shared (cloned) across tasks. No hot-reload — restart to change settings.

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct GateCfg {
    pub predicted_spread_multiplier: f64,
    pub confidence_floor: f64,
    pub latency_p99_cap_ms: f64,
    pub min_predicted_bps_abs: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SizingCfg {
    pub notional_usd: f64,
    pub hold_seconds: u64,
    // Reserved for v0.2 — we currently enforce single-position discipline
    // implicitly via the synchronous hold/exit pattern in the decision loop.
    #[allow(dead_code)]
    pub max_open_positions: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RiskCfg {
    pub initial_nav_usd: f64,
    pub max_drawdown_pct: f64,
    pub max_consecutive_losses: u32,
    pub latency_breach_ms: f64,
    pub latency_breach_minutes: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct IntervalsCfg {
    pub latency_seconds: u64,
    pub data_feed_seconds: u64,
    pub decision_seconds: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PredictorCfg {
    pub rls_forgetting_factor: f64,
    pub warmup_samples: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EndpointsCfg {
    pub alpaca_paper_base: String,
    pub alpaca_data_base: String,
    pub yahoo_es_chart: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SafetyCfg {
    pub confirm_first_n_trades: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FileConfig {
    pub gate: GateCfg,
    pub sizing: SizingCfg,
    pub risk: RiskCfg,
    pub intervals: IntervalsCfg,
    pub predictor: PredictorCfg,
    pub endpoints: EndpointsCfg,
    pub safety: SafetyCfg,
}

/// Credentials sourced from environment only — never written to disk via this code.
#[derive(Clone)]
pub struct Credentials {
    pub alpaca_key: String,
    pub alpaca_secret: String,
}

impl std::fmt::Debug for Credentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Credentials")
            .field("alpaca_key", &"***REDACTED***")
            .field("alpaca_secret", &"***REDACTED***")
            .finish()
    }
}

/// Full resolved configuration.
#[derive(Debug, Clone)]
pub struct Config {
    pub file: FileConfig,
    pub creds: Credentials,
    pub dry_run: bool,
    pub require_confirm: bool,
    pub sqlite_path: String,
    pub kill_file: String,
}

impl Config {
    /// Load YAML at `yaml_path`, merge env credentials, apply CLI overrides.
    pub fn load(yaml_path: &Path, dry_run: bool) -> Result<Self> {
        let raw = std::fs::read_to_string(yaml_path)
            .with_context(|| format!("reading config {}", yaml_path.display()))?;
        let file: FileConfig = serde_yaml::from_str(&raw).context("parsing YAML config")?;

        // Credentials must come from env (dotenv-loaded earlier in main).
        let alpaca_key = std::env::var("ALPACA_FALCON_KEY")
            .map_err(|_| anyhow!("ALPACA_FALCON_KEY not set in env (see council/.env.local)"))?;
        let alpaca_secret = std::env::var("ALPACA_FALCON_SECRET")
            .map_err(|_| anyhow!("ALPACA_FALCON_SECRET not set in env"))?;

        let require_confirm = std::env::var("HFT_REQUIRE_CONFIRM")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        let sqlite_path = std::env::var("HFT_SQLITE_PATH").unwrap_or_else(|_| {
            "/Users/weeoto/turbo-carnival/research/live/hft_engine_decisions.sqlite3".into()
        });
        let kill_file = std::env::var("HFT_KILL_FILE").unwrap_or_else(|_| {
            "/Users/weeoto/turbo-carnival/research/live/STOP_HFT_ENGINE".into()
        });

        Ok(Self {
            file,
            creds: Credentials {
                alpaca_key,
                alpaca_secret,
            },
            dry_run,
            require_confirm,
            sqlite_path,
            kill_file,
        })
    }
}
