//! Continuous Alpaca round-trip latency profiler.
//!
//! Pings `/v2/clock` on a fixed cadence, stores last 100 samples in a ring,
//! and exposes percentile getters. We need this because the selective gate
//! refuses to trade if p99 is too high — a trade only works if the round-trip
//! is well below the prediction horizon (30–60s), and rising tail latency is
//! the first sign the link to Alpaca has degraded.

use anyhow::Result;
use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::Client;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::{debug, warn};

// Hard upper bound on samples retained — prevents runaway memory if probe rate spikes.
const RING_CAPACITY: usize = 600;

// Samples older than this are dropped before each percentile call. A bounded
// time window (instead of count-only) lets the cold-start outlier expire
// naturally and prevents `p99 = max(samples)` anchoring whenever n < ring size.
const SAMPLE_RETENTION_SECS: i64 = 300;

// Discard the first N samples on startup — the first probe usually pays a
// TLS handshake + connection-pool fill cost that does not represent steady-state.
const WARMUP_DISCARD: usize = 3;

#[derive(Debug, Clone)]
pub struct LatencySample {
    pub at: chrono::DateTime<chrono::Utc>,
    pub rtt_micros: u64,
    pub http_status: u16,
    pub ok: bool,
}

#[derive(Debug, Default)]
pub struct LatencyInner {
    samples: VecDeque<LatencySample>,
    total_probes: usize,
}

#[derive(Clone)]
pub struct LatencyProfiler {
    inner: Arc<RwLock<LatencyInner>>,
    client: Client,
    base_url: String,
    headers: HeaderMap,
}

impl LatencyProfiler {
    pub fn new(base_url: String, key: &str, secret: &str) -> Result<Self> {
        let mut headers = HeaderMap::new();
        headers.insert("APCA-API-KEY-ID", HeaderValue::from_str(key)?);
        headers.insert("APCA-API-SECRET-KEY", HeaderValue::from_str(secret)?);
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .pool_idle_timeout(std::time::Duration::from_secs(30))
            .build()?;
        Ok(Self {
            inner: Arc::new(RwLock::new(LatencyInner::default())),
            client,
            base_url,
            headers,
        })
    }

    /// Single probe — returns the recorded sample.
    pub async fn probe(&self) -> LatencySample {
        let url = format!("{}/v2/clock", self.base_url);
        let start = Instant::now();
        let result = self
            .client
            .get(&url)
            .headers(self.headers.clone())
            .send()
            .await;
        let elapsed = start.elapsed();
        let rtt_micros = elapsed.as_micros() as u64;
        let (http_status, ok) = match &result {
            Ok(resp) => (resp.status().as_u16(), resp.status().is_success()),
            Err(_) => (0u16, false),
        };
        let sample = LatencySample {
            at: chrono::Utc::now(),
            rtt_micros,
            http_status,
            ok,
        };
        {
            let mut guard = self.inner.write().await;
            guard.total_probes = guard.total_probes.saturating_add(1);
            // Skip warmup samples (TLS handshake + pool fill) so they don't anchor p99.
            if guard.total_probes > WARMUP_DISCARD {
                // Expire samples older than retention window — bounded time view.
                let cutoff = chrono::Utc::now()
                    - chrono::Duration::seconds(SAMPLE_RETENTION_SECS);
                while let Some(front) = guard.samples.front() {
                    if front.at < cutoff {
                        guard.samples.pop_front();
                    } else {
                        break;
                    }
                }
                // Hard cap on count to defend against pathological probe storms.
                while guard.samples.len() >= RING_CAPACITY {
                    guard.samples.pop_front();
                }
                guard.samples.push_back(sample.clone());
            }
        }
        if !ok {
            warn!(
                "latency probe failed: status={} rtt_ms={:.2}",
                http_status,
                rtt_micros as f64 / 1000.0
            );
        } else {
            debug!("latency probe rtt_ms={:.2}", rtt_micros as f64 / 1000.0);
        }
        sample
    }

    /// Returns the most recent sample, if any.
    pub async fn latest(&self) -> Option<LatencySample> {
        self.inner.read().await.samples.back().cloned()
    }

    /// Returns all samples currently buffered (cloned).
    #[allow(dead_code)] // part of the documented profiler API; used by analysis tools
    pub async fn snapshot(&self) -> Vec<LatencySample> {
        self.inner.read().await.samples.iter().cloned().collect()
    }

    #[allow(dead_code)] // documented API; surfaced by analysis & v0.2 dashboards
    pub async fn latency_p50_ms(&self) -> Option<f64> {
        self.percentile_ms(0.50).await
    }
    #[allow(dead_code)] // documented API; surfaced by analysis & v0.2 dashboards
    pub async fn latency_p95_ms(&self) -> Option<f64> {
        self.percentile_ms(0.95).await
    }
    pub async fn latency_p99_ms(&self) -> Option<f64> {
        self.percentile_ms(0.99).await
    }

    #[allow(dead_code)] // documented API; surfaced by analysis & v0.2 dashboards
    pub async fn latency_stddev_ms(&self) -> Option<f64> {
        let guard = self.inner.read().await;
        let n = guard.samples.len();
        if n < 2 {
            return None;
        }
        let mean_us: f64 = guard
            .samples
            .iter()
            .map(|s| s.rtt_micros as f64)
            .sum::<f64>()
            / n as f64;
        let var_us: f64 = guard
            .samples
            .iter()
            .map(|s| {
                let d = s.rtt_micros as f64 - mean_us;
                d * d
            })
            .sum::<f64>()
            / (n as f64 - 1.0);
        Some(var_us.sqrt() / 1000.0)
    }

    async fn percentile_ms(&self, q: f64) -> Option<f64> {
        let guard = self.inner.read().await;
        if guard.samples.is_empty() {
            return None;
        }
        let mut v: Vec<u64> = guard.samples.iter().map(|s| s.rtt_micros).collect();
        v.sort_unstable();
        // Nearest-rank percentile — simple and unambiguous.
        let n = v.len();
        let mut rank = (q * n as f64).ceil() as usize;
        if rank == 0 {
            rank = 1;
        }
        if rank > n {
            rank = n;
        }
        Some(v[rank - 1] as f64 / 1000.0)
    }
}

/// Spawn the background profiler task. Returns the profiler handle; caller
/// keeps a clone and may query it from any task.
pub fn spawn_profiler(profiler: LatencyProfiler, interval_secs: u64) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(std::time::Duration::from_secs(interval_secs));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            tick.tick().await;
            let _ = profiler.probe().await;
        }
    })
}
