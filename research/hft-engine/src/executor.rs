//! Selective executor.
//!
//! The executor evaluates the "spaceship gravity assist" alignment of:
//!   * predicted SPY return >> current spread
//!   * rolling hit-rate above threshold
//!   * latency p99 below cap
//!   * no regime pause / NAV halt
//!
//! Only when ALL of those line up do we place a marketable-limit IOC order.
//! After `hold_seconds` we close at marketable-limit IOC. Realized PnL is
//! computed from filled prices on success, or from mid-prices if the order
//! is rejected / not filled (dry-run case).

use crate::config::Config;
use crate::data_feed::SpyQuote;
use crate::logger::SqliteLogger;
use crate::signal::Prediction;
use anyhow::{anyhow, Result};
use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{info, warn};

#[derive(Debug, Clone, Serialize)]
pub struct GateDecision {
    pub fired: bool,
    pub reasoning: Vec<String>,
    pub predicted_bps: f64,
    pub spread_bps: f64,
    pub confidence: f64,
    pub latency_p99_ms: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

impl Side {
    pub fn as_str(self) -> &'static str {
        match self {
            Side::Buy => "buy",
            Side::Sell => "sell",
        }
    }
    pub fn opposite(self) -> Self {
        match self {
            Side::Buy => Side::Sell,
            Side::Sell => Side::Buy,
        }
    }
}

#[derive(Debug, Deserialize)]
struct AlpacaOrderResp {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    #[allow(dead_code)] // surfaced into SQLite via the raw response_excerpt
    status: Option<String>,
    #[serde(default)]
    filled_qty: Option<String>,
    #[serde(default)]
    filled_avg_price: Option<String>,
}

#[derive(Clone)]
pub struct Executor {
    cfg: Config,
    client: Client,
    headers: HeaderMap,
    logger: SqliteLogger,
    confirm_count: Arc<Mutex<u32>>,
}

impl Executor {
    pub fn new(cfg: Config, logger: SqliteLogger) -> Result<Self> {
        let mut headers = HeaderMap::new();
        headers.insert(
            "APCA-API-KEY-ID",
            HeaderValue::from_str(&cfg.creds.alpaca_key)?,
        );
        headers.insert(
            "APCA-API-SECRET-KEY",
            HeaderValue::from_str(&cfg.creds.alpaca_secret)?,
        );
        headers.insert("Content-Type", HeaderValue::from_static("application/json"));
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()?;
        Ok(Self {
            cfg,
            client,
            headers,
            logger,
            confirm_count: Arc::new(Mutex::new(0)),
        })
    }

    /// Evaluate the selective gate. Pure function over inputs.
    pub fn gate(
        &self,
        pred: &Prediction,
        quote: &SpyQuote,
        latency_p99_ms: Option<f64>,
        nav_halted: bool,
        regime_pause: bool,
    ) -> GateDecision {
        let mut reasoning: Vec<String> = Vec::new();
        let g = &self.cfg.file.gate;
        let spread_bps = quote.spread_bps();
        let p99 = latency_p99_ms.unwrap_or(f64::INFINITY);

        let mut fired = true;

        if nav_halted {
            fired = false;
            reasoning.push("nav_halted".into());
        }
        if regime_pause {
            fired = false;
            reasoning.push("regime_pause".into());
        }
        if !spread_bps.is_finite() || spread_bps <= 0.0 {
            fired = false;
            reasoning.push(format!("spread_invalid:{spread_bps}"));
        }
        if !p99.is_finite() || p99 > g.latency_p99_cap_ms {
            fired = false;
            reasoning.push(format!(
                "latency_p99_over_cap:{p99:.2}>={:.2}",
                g.latency_p99_cap_ms
            ));
        }
        if pred.confidence < g.confidence_floor {
            fired = false;
            reasoning.push(format!(
                "confidence_below_floor:{:.3}<{:.3}",
                pred.confidence, g.confidence_floor
            ));
        }
        if pred.predicted_bps.abs() < g.min_predicted_bps_abs {
            fired = false;
            reasoning.push(format!(
                "predicted_abs_below_floor:{:.3}<{:.3}",
                pred.predicted_bps.abs(),
                g.min_predicted_bps_abs
            ));
        }
        if spread_bps.is_finite() && spread_bps > 0.0 {
            let threshold = g.predicted_spread_multiplier * spread_bps;
            if pred.predicted_bps.abs() < threshold {
                fired = false;
                reasoning.push(format!(
                    "predicted_below_spread_multiple:{:.3}<{:.3}",
                    pred.predicted_bps.abs(),
                    threshold
                ));
            }
        }

        if fired {
            reasoning.push("all_gates_aligned".into());
        }

        GateDecision {
            fired,
            reasoning,
            predicted_bps: pred.predicted_bps,
            spread_bps,
            confidence: pred.confidence,
            latency_p99_ms: p99,
        }
    }

    /// Place a marketable-limit IOC order for SPY. Returns (filled_price, qty).
    /// In dry_run mode, "fills" at the marketable side of the book and logs.
    pub async fn place_order(&self, side: Side, qty: u64, quote: &SpyQuote) -> Result<(f64, u64)> {
        let raw_limit = match side {
            // Buying: pay the offer plus a small cushion (1 bp) to be marketable.
            Side::Buy => quote.ask * (1.0 + 0.0001),
            // Selling: hit the bid minus a small cushion.
            Side::Sell => quote.bid * (1.0 - 0.0001),
        };
        // Alpaca rejects sub-penny prices on equities (HTTP 422 code 42210000).
        // Round to the nearest cent in the direction that keeps the order marketable.
        let limit_price = match side {
            Side::Buy  => (raw_limit * 100.0).ceil()  / 100.0,
            Side::Sell => (raw_limit * 100.0).floor() / 100.0,
        };

        if self.cfg.dry_run {
            info!(
                "[DRY-RUN] {} {} SPY @ limit {:.4} (mid {:.4})",
                side.as_str(),
                qty,
                limit_price,
                quote.mid()
            );
            self.logger
                .log_order(
                    side.as_str(),
                    "SPY",
                    qty as f64,
                    Some(limit_price),
                    None,
                    None,
                    Some("dry_run"),
                    true,
                )
                .await
                .ok();
            let fill_px = match side {
                Side::Buy => quote.ask,
                Side::Sell => quote.bid,
            };
            return Ok((fill_px, qty));
        }

        if self.cfg.require_confirm {
            let mut n = self.confirm_count.lock().await;
            if *n < self.cfg.file.safety.confirm_first_n_trades {
                let prompt = format!(
                    "CONFIRM trade #{}/{} {} {} SPY @ limit {:.4} [y/N]: ",
                    *n + 1,
                    self.cfg.file.safety.confirm_first_n_trades,
                    side.as_str(),
                    qty,
                    limit_price
                );
                let confirmed = prompt_yes_no(&prompt).await?;
                if !confirmed {
                    return Err(anyhow!("user declined confirm"));
                }
                *n += 1;
            }
        }

        let body = serde_json::json!({
            "symbol": "SPY",
            "qty": qty.to_string(),
            "side": side.as_str(),
            "type": "limit",
            "time_in_force": "ioc",
            "limit_price": format!("{:.2}", limit_price),
        });
        let url = format!("{}/v2/orders", self.cfg.file.endpoints.alpaca_paper_base);
        let resp = self
            .client
            .post(&url)
            .headers(self.headers.clone())
            .json(&body)
            .send()
            .await?;
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            warn!(
                "Alpaca order rejected: status={} body={}",
                status,
                excerpt(&text, 256)
            );
            self.logger
                .log_order(
                    side.as_str(),
                    "SPY",
                    qty as f64,
                    Some(limit_price),
                    None,
                    Some(status.as_u16()),
                    Some(&excerpt(&text, 256)),
                    false,
                )
                .await
                .ok();
            return Err(anyhow!("alpaca order HTTP {}", status));
        }
        let parsed: AlpacaOrderResp = serde_json::from_str(&text).unwrap_or(AlpacaOrderResp {
            id: None,
            status: None,
            filled_qty: None,
            filled_avg_price: None,
        });
        let order_id = parsed.id.clone();
        self.logger
            .log_order(
                side.as_str(),
                "SPY",
                qty as f64,
                Some(limit_price),
                order_id.as_deref(),
                Some(status.as_u16()),
                Some(&excerpt(&text, 256)),
                false,
            )
            .await
            .ok();

        let filled_qty: u64 = parsed
            .filled_qty
            .as_deref()
            .and_then(|s| s.parse::<f64>().ok())
            .map(|f| f as u64)
            .unwrap_or(0);
        let fill_px: f64 = parsed
            .filled_avg_price
            .as_deref()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or_else(|| match side {
                Side::Buy => quote.ask,
                Side::Sell => quote.bid,
            });
        Ok((fill_px, filled_qty))
    }
}

fn excerpt(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        format!("{}...", &s[..n])
    }
}

async fn prompt_yes_no(prompt: &str) -> Result<bool> {
    // Block on stdin in a spawned thread so we don't stall the tokio runtime.
    let prompt = prompt.to_string();
    let answer = tokio::task::spawn_blocking(move || {
        use std::io::{self, BufRead, Write};
        let stdout = io::stdout();
        let mut handle = stdout.lock();
        let _ = handle.write_all(prompt.as_bytes());
        let _ = handle.flush();
        let stdin = io::stdin();
        let mut line = String::new();
        stdin
            .lock()
            .read_line(&mut line)
            .map(|_| line.trim().to_ascii_lowercase())
            .unwrap_or_default()
    })
    .await
    .unwrap_or_default();
    Ok(answer == "y" || answer == "yes")
}
