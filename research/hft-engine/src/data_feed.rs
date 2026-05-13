//! Market data feed.
//!
//! Pulls:
//!   * SPY top-of-book from Alpaca `/v2/stocks/SPY/quotes/latest` (data API).
//!   * ES futures 1-min bars from Yahoo Finance `chart` endpoint.
//!
//! We hold two rolling windows (last 30 SPY quotes, last 20 ES bars) for the
//! signal layer. No streaming/websockets — REST polling is fine at 30s cadence
//! and dramatically simpler.

use anyhow::{anyhow, Context, Result};
use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::Client;
use serde::Deserialize;
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, warn};

const SPY_WINDOW: usize = 30;
const ES_WINDOW: usize = 20;

#[derive(Debug, Clone)]
pub struct SpyQuote {
    // Kept for SQLite forensic dumps and v0.2 staleness checks.
    #[allow(dead_code)]
    pub at: chrono::DateTime<chrono::Utc>,
    pub bid: f64,
    pub ask: f64,
    // Sizes retained — v0.2 will gate on min-size at top of book.
    #[allow(dead_code)]
    pub bid_size: u64,
    #[allow(dead_code)]
    pub ask_size: u64,
}

impl SpyQuote {
    pub fn mid(&self) -> f64 {
        (self.bid + self.ask) / 2.0
    }
    pub fn spread_bps(&self) -> f64 {
        let m = self.mid();
        if m <= 0.0 {
            return f64::NAN;
        }
        (self.ask - self.bid) / m * 10_000.0
    }
}

#[derive(Debug, Clone)]
pub struct EsBar {
    pub ts_secs: i64,
    pub close: f64,
}

#[derive(Debug, Default)]
pub struct FeedInner {
    pub spy: VecDeque<SpyQuote>,
    pub es: VecDeque<EsBar>,
}

#[derive(Clone)]
pub struct DataFeed {
    inner: Arc<RwLock<FeedInner>>,
    client: Client,
    alpaca_data_base: String,
    yahoo_es_url: String,
    alpaca_headers: HeaderMap,
}

#[derive(Debug, Clone)]
pub struct FeedSnapshot {
    pub spy: Vec<SpyQuote>,
    pub es: Vec<EsBar>,
}

// ---- Alpaca quote payload --------------------------------------------------

#[derive(Debug, Deserialize)]
struct AlpacaQuoteResp {
    quote: AlpacaQuote,
}

#[derive(Debug, Deserialize)]
struct AlpacaQuote {
    #[serde(rename = "bp")]
    bid_price: f64,
    #[serde(rename = "ap")]
    ask_price: f64,
    #[serde(rename = "bs", default)]
    bid_size: u64,
    #[serde(rename = "as", default)]
    ask_size: u64,
    #[serde(rename = "t")]
    timestamp: String,
}

// ---- Yahoo chart payload ---------------------------------------------------

#[derive(Debug, Deserialize)]
struct YahooChartResp {
    chart: YahooChart,
}

#[derive(Debug, Deserialize)]
struct YahooChart {
    result: Option<Vec<YahooResult>>,
    error: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct YahooResult {
    timestamp: Option<Vec<i64>>,
    indicators: YahooIndicators,
}

#[derive(Debug, Deserialize)]
struct YahooIndicators {
    quote: Vec<YahooQuoteArr>,
}

#[derive(Debug, Deserialize)]
struct YahooQuoteArr {
    // Yahoo's chart endpoint returns `quote: [{}]` (empty object) outside
    // exchange hours — none of close/open/high/low are present. Allow that
    // case to deserialize cleanly so we can return a clean "no data" error.
    #[serde(default)]
    close: Vec<Option<f64>>,
}

impl DataFeed {
    pub fn new(
        alpaca_data_base: String,
        yahoo_es_url: String,
        key: &str,
        secret: &str,
    ) -> Result<Self> {
        let mut headers = HeaderMap::new();
        headers.insert("APCA-API-KEY-ID", HeaderValue::from_str(key)?);
        headers.insert("APCA-API-SECRET-KEY", HeaderValue::from_str(secret)?);
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .user_agent("hft-engine/0.1 (research)")
            .build()?;
        Ok(Self {
            inner: Arc::new(RwLock::new(FeedInner::default())),
            client,
            alpaca_data_base,
            yahoo_es_url,
            alpaca_headers: headers,
        })
    }

    /// Refresh both SPY quote and ES bars. Errors in one do not block the other.
    pub async fn refresh(&self) -> (Option<SpyQuote>, Option<EsBar>) {
        let spy_res = self.fetch_spy_quote().await;
        let es_res = self.fetch_es_latest_bar().await;

        let mut guard = self.inner.write().await;
        let spy_out = match spy_res {
            Ok(q) => {
                if guard.spy.len() == SPY_WINDOW {
                    guard.spy.pop_front();
                }
                guard.spy.push_back(q.clone());
                Some(q)
            }
            Err(e) => {
                warn!("SPY quote fetch failed: {e:#}");
                None
            }
        };
        let es_out = match es_res {
            Ok(bar) => {
                // Only push when timestamp advances (yahoo returns 1m bars).
                let is_new = guard.es.back().map(|b| b.ts_secs < bar.ts_secs).unwrap_or(true);
                if is_new {
                    if guard.es.len() == ES_WINDOW {
                        guard.es.pop_front();
                    }
                    guard.es.push_back(bar.clone());
                }
                Some(bar)
            }
            Err(e) => {
                warn!("ES bar fetch failed: {e:#}");
                None
            }
        };
        (spy_out, es_out)
    }

    pub async fn snapshot(&self) -> FeedSnapshot {
        let g = self.inner.read().await;
        FeedSnapshot {
            spy: g.spy.iter().cloned().collect(),
            es: g.es.iter().cloned().collect(),
        }
    }

    /// Most recent SPY quote.
    pub async fn latest_spy(&self) -> Option<SpyQuote> {
        self.inner.read().await.spy.back().cloned()
    }

    async fn fetch_spy_quote(&self) -> Result<SpyQuote> {
        let url = format!("{}/v2/stocks/SPY/quotes/latest", self.alpaca_data_base);
        let resp = self
            .client
            .get(&url)
            .headers(self.alpaca_headers.clone())
            .send()
            .await
            .context("alpaca quote GET")?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!("alpaca quote HTTP {} body={}", status, body));
        }
        let parsed: AlpacaQuoteResp = resp.json().await.context("alpaca quote JSON")?;
        let q = parsed.quote;
        let at = chrono::DateTime::parse_from_rfc3339(&q.timestamp)
            .map(|d| d.with_timezone(&chrono::Utc))
            .unwrap_or_else(|_| chrono::Utc::now());
        debug!(
            "SPY quote bid={} ask={} spread_bps={:.2}",
            q.bid_price,
            q.ask_price,
            if (q.bid_price + q.ask_price) > 0.0 {
                (q.ask_price - q.bid_price) / ((q.bid_price + q.ask_price) / 2.0) * 10_000.0
            } else {
                f64::NAN
            }
        );
        Ok(SpyQuote {
            at,
            bid: q.bid_price,
            ask: q.ask_price,
            bid_size: q.bid_size,
            ask_size: q.ask_size,
        })
    }

    async fn fetch_es_latest_bar(&self) -> Result<EsBar> {
        let resp = self
            .client
            .get(&self.yahoo_es_url)
            .send()
            .await
            .context("yahoo ES GET")?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!("yahoo ES HTTP {} body_len={}", status, body.len()));
        }
        let parsed: YahooChartResp = resp.json().await.context("yahoo ES JSON")?;
        if let Some(err) = parsed.chart.error {
            return Err(anyhow!("yahoo ES error: {err}"));
        }
        let result = parsed
            .chart
            .result
            .and_then(|mut v| v.pop())
            .ok_or_else(|| anyhow!("yahoo ES: no result"))?;
        let timestamps = match result.timestamp {
            Some(t) if !t.is_empty() => t,
            _ => return Err(anyhow!("yahoo ES: no timestamps (outside trading window?)")),
        };
        let closes = result
            .indicators
            .quote
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("no quote indicators"))?
            .close;

        if closes.is_empty() {
            return Err(anyhow!(
                "yahoo ES: empty close array (outside trading window?)"
            ));
        }

        // Walk from the back to find the most recent non-null close.
        for i in (0..timestamps.len().min(closes.len())).rev() {
            if let Some(c) = closes[i] {
                if c.is_finite() && c > 0.0 {
                    return Ok(EsBar {
                        ts_secs: timestamps[i],
                        close: c,
                    });
                }
            }
        }
        Err(anyhow!("no valid ES bar in response"))
    }
}

/// Spawn the data feed refresh loop. Returns the join handle.
pub fn spawn_feed(feed: DataFeed, interval_secs: u64) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(std::time::Duration::from_secs(interval_secs));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            tick.tick().await;
            let _ = feed.refresh().await;
        }
    })
}
