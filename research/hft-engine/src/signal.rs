//! Online Recursive Least Squares predictor: ES 1-min return -> SPY 60s return.
//!
//! We learn a 1-D RLS regression with forgetting factor lambda, plus a small
//! ring of recent (predicted_sign, realized_sign) pairs to estimate a rolling
//! hit rate that we surface as "confidence". This is intentionally simple: a
//! single feature, no intercept, no L2 — because the prior on edge is low and
//! we'd rather a clearly-attributable signal than an over-fit model.

use crate::data_feed::FeedSnapshot;
use std::collections::VecDeque;
use tracing::debug;

const HITRATE_WINDOW: usize = 50;

#[derive(Debug, Clone)]
pub struct Prediction {
    /// Predicted SPY 60s return (basis points).
    pub predicted_bps: f64,
    /// Rolling hit-rate over the last `HITRATE_WINDOW` realized labels.
    pub confidence: f64,
    /// Most-recent ES 1-min return (bps) used as the feature.
    pub es_ret_bps: f64,
    /// Most-recent SPY 30s mid-return (bps), used for trade-result attribution.
    pub spy_ret_bps: f64,
    /// Total observations the predictor has been updated with.
    pub samples: usize,
}

impl Prediction {
    #[allow(dead_code)] // useful constructor for tests / dashboards
    pub fn zero() -> Self {
        Self {
            predicted_bps: 0.0,
            confidence: 0.0,
            es_ret_bps: 0.0,
            spy_ret_bps: 0.0,
            samples: 0,
        }
    }
}

/// Online single-feature RLS estimator. Maintains weight w and inverse-cov P.
pub struct RlsPredictor {
    weight: f64,
    p: f64,
    lambda: f64,
    warmup: usize,
    samples: usize,
    hits: VecDeque<bool>,
}

impl RlsPredictor {
    pub fn new(lambda: f64, warmup: usize) -> Self {
        Self {
            weight: 0.0,
            p: 1000.0, // large initial inverse-cov => weight moves freely early on
            lambda,
            warmup,
            samples: 0,
            hits: VecDeque::with_capacity(HITRATE_WINDOW),
        }
    }

    #[allow(dead_code)] // inspector — used for offline diagnostics
    pub fn samples(&self) -> usize {
        self.samples
    }

    #[allow(dead_code)] // inspector — used for offline diagnostics
    pub fn weight(&self) -> f64 {
        self.weight
    }

    /// Hit-rate over the most recent realized labels (0.0 before any).
    pub fn hit_rate(&self) -> f64 {
        if self.hits.is_empty() {
            return 0.0;
        }
        let hits = self.hits.iter().filter(|h| **h).count();
        hits as f64 / self.hits.len() as f64
    }

    /// Predict the next SPY 60s return (bps) from the current ES 1m return (bps).
    pub fn predict(&self, es_ret_bps: f64) -> f64 {
        self.weight * es_ret_bps
    }

    /// Update with one realized (feature, label) pair.
    pub fn update(&mut self, es_ret_bps: f64, spy_realized_bps: f64) {
        // Standard scalar RLS update.
        let x = es_ret_bps;
        let denom = self.lambda + x * self.p * x;
        if denom.abs() < 1e-12 {
            return;
        }
        let k = (self.p * x) / denom;
        let err = spy_realized_bps - self.weight * x;
        self.weight += k * err;
        self.p = (self.p - k * x * self.p) / self.lambda;

        self.samples += 1;
        // Hit-rate tracking — only count signed agreement when both have signal.
        let predicted_sign = (self.weight * x).signum();
        let realized_sign = spy_realized_bps.signum();
        let hit = predicted_sign != 0.0
            && realized_sign != 0.0
            && (predicted_sign - realized_sign).abs() < 1e-9;
        if self.hits.len() == HITRATE_WINDOW {
            self.hits.pop_front();
        }
        self.hits.push_back(hit);

        debug!(
            "RLS update n={} w={:.6} p={:.4} hitrate={:.3} err={:.4}",
            self.samples,
            self.weight,
            self.p,
            self.hit_rate(),
            err
        );
    }

    /// Build a Prediction from a feed snapshot. Uses the most-recent two ES bars
    /// and the most-recent SPY mid vs the mid ~60s ago for context.
    pub fn build(&self, feed: &FeedSnapshot) -> Prediction {
        let es_ret_bps = es_return_bps(feed);
        let spy_ret_bps = spy_return_bps(feed, 2); // 2 samples back ~= 60s at 30s cadence

        let predicted_bps = self.predict(es_ret_bps);
        let confidence = if self.samples < self.warmup {
            0.0
        } else {
            self.hit_rate()
        };

        Prediction {
            predicted_bps,
            confidence,
            es_ret_bps,
            spy_ret_bps,
            samples: self.samples,
        }
    }
}

/// ES 1-min return in basis points from the last two bars in the window.
pub fn es_return_bps(feed: &FeedSnapshot) -> f64 {
    if feed.es.len() < 2 {
        return 0.0;
    }
    let last = feed.es[feed.es.len() - 1].close;
    let prev = feed.es[feed.es.len() - 2].close;
    if prev <= 0.0 {
        return 0.0;
    }
    (last - prev) / prev * 10_000.0
}

/// SPY return in bps comparing the most recent mid to the mid `lag` samples ago.
/// Returns 0.0 if not enough history.
pub fn spy_return_bps(feed: &FeedSnapshot, lag: usize) -> f64 {
    if feed.spy.len() <= lag {
        return 0.0;
    }
    let last = feed.spy[feed.spy.len() - 1].mid();
    let prev = feed.spy[feed.spy.len() - 1 - lag].mid();
    if prev <= 0.0 {
        return 0.0;
    }
    (last - prev) / prev * 10_000.0
}
