//! Predictive Selective Trading engine — daemon entrypoint.
//!
//! NOT real HFT. We're on a Mac in London hitting Alpaca's US paper API across
//! ~100-200ms of round-trip latency. The thesis being prototyped: combining a
//! lead-lag predictor (ES futures -> SPY) with a strict selectivity gate can
//! extract a small medium-frequency edge on a 30-60s horizon, while staying
//! well clear of the microsecond regime we cannot compete in.
//!
//! Loop topology:
//!   * latency profiler   — ticks every 5s, GETs /v2/clock, records RTT.
//!   * data feed          — every 30s, pulls SPY top-of-book + ES 1m bars.
//!   * decision loop      — every 30s, builds prediction, runs gate, may trade.
//!
//! Kill paths: SIGINT, SIGTERM, kill-file presence, NAV halt, latency breach.

mod config;
mod data_feed;
mod executor;
mod latency;
mod logger;
mod nav;
mod signal;

use crate::config::Config;
use crate::data_feed::{spawn_feed, DataFeed};
use crate::executor::{Executor, Side};
use crate::latency::{spawn_profiler, LatencyProfiler};
use crate::logger::SqliteLogger;
use crate::signal::RlsPredictor;
use anyhow::{Context, Result};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::signal::unix::{signal, SignalKind};
use tokio::sync::Mutex;
use tracing::{error, info, warn};

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    // Parse minimal CLI: --dry-run, --config <path>, --shadow-only, --invert-signal
    let args: Vec<String> = std::env::args().collect();
    let dry_run       = args.iter().any(|a| a == "--dry-run");
    let shadow_only   = args.iter().any(|a| a == "--shadow-only");
    let invert_signal = args.iter().any(|a| a == "--invert-signal");
    let mut config_path = PathBuf::from("config/default.yaml");
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--config" {
            if let Some(p) = args.get(i + 1) {
                config_path = PathBuf::from(p);
            }
        }
        i += 1;
    }

    // Logging: default RUST_LOG=info. Always pretty-print, ISO timestamps.
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_level(true)
        .init();

    // Load env from council/.env.local (preferred) or whatever dotenv finds first.
    let _ = dotenv::from_path("/Users/weeoto/turbo-carnival/council/.env.local");
    let _ = dotenv::dotenv();

    let cfg = Config::load(&config_path, dry_run)
        .with_context(|| format!("loading config {}", config_path.display()))?;
    if shadow_only {
        info!("STARTING IN SHADOW-ONLY MODE: every prediction logged, NO orders ever");
    } else if cfg.dry_run {
        info!("STARTING IN DRY-RUN: no orders will hit Alpaca");
    } else {
        info!("STARTING IN LIVE PAPER MODE (Alpaca paper account)");
    }
    if invert_signal {
        info!("SIGNAL INVERTED: predicted_bps will be negated before gate evaluation");
    }
    info!("sqlite_path={}", cfg.sqlite_path);
    info!("kill_file={}", cfg.kill_file);
    info!("config_path={}", config_path.display());

    // SQLite logger.
    let sql_logger = SqliteLogger::open(&cfg.sqlite_path).await?;
    sql_logger.log_regime("engine_start", &format!("dry_run={dry_run}")).await.ok();

    // Latency profiler (live thread).
    let profiler = LatencyProfiler::new(
        cfg.file.endpoints.alpaca_paper_base.clone(),
        &cfg.creds.alpaca_key,
        &cfg.creds.alpaca_secret,
    )?;
    let lat_handle = spawn_profiler(profiler.clone(), cfg.file.intervals.latency_seconds);

    // A small forwarder task: every probe interval, snapshot latest sample
    // into SQLite so we have an auditable history.
    let lat_for_log = profiler.clone();
    let log_for_lat = sql_logger.clone();
    let lat_log_interval = cfg.file.intervals.latency_seconds;
    let lat_log_handle = tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_secs(lat_log_interval));
        tick.tick().await; // initial — gives the profiler a moment to record one
        loop {
            tick.tick().await;
            if let Some(s) = lat_for_log.latest().await {
                let _ = log_for_lat
                    .log_latency(s.rtt_micros, s.http_status, s.ok, s.at)
                    .await;
            }
        }
    });

    // Data feed.
    let feed = DataFeed::new(
        cfg.file.endpoints.alpaca_data_base.clone(),
        cfg.file.endpoints.yahoo_es_chart.clone(),
        &cfg.creds.alpaca_key,
        &cfg.creds.alpaca_secret,
    )?;
    let feed_handle = spawn_feed(feed.clone(), cfg.file.intervals.data_feed_seconds);

    // RLS predictor — shared state across cycles.
    let predictor = Arc::new(Mutex::new(RlsPredictor::new(
        cfg.file.predictor.rls_forgetting_factor,
        cfg.file.predictor.warmup_samples,
    )));

    // NAV state.
    let nav_arc = Arc::new(Mutex::new({
        let conn = sql_logger.raw();
        let guard = conn.lock().await;
        nav::load_or_init(&guard, cfg.file.risk.initial_nav_usd)?
    }));
    {
        let n = nav_arc.lock().await;
        sql_logger
            .log_nav_snapshot(
                n.nav_usd,
                n.peak_nav_usd,
                n.drawdown_pct(),
                n.consecutive_losses,
                n.halted,
                "startup",
            )
            .await
            .ok();
    }

    let executor = Executor::new(cfg.clone(), sql_logger.clone())?;

    // Latency-breach watchdog state — track time of first breach.
    let breach_start: Arc<Mutex<Option<chrono::DateTime<chrono::Utc>>>> =
        Arc::new(Mutex::new(None));

    // Main decision loop.
    let decision_cfg = cfg.clone();
    let decision_feed = feed.clone();
    let decision_profiler = profiler.clone();
    let decision_predictor = predictor.clone();
    let decision_nav = nav_arc.clone();
    let decision_logger = sql_logger.clone();
    let decision_executor = executor.clone();
    let decision_breach = breach_start.clone();
    let decision_handle = tokio::spawn(async move {
        let mut tick =
            tokio::time::interval(Duration::from_secs(decision_cfg.file.intervals.decision_seconds));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        // Give feed/profiler a short head-start so the first cycle has data.
        tokio::time::sleep(Duration::from_secs(2)).await;
        loop {
            tick.tick().await;
            if let Err(e) = decide_once(
                &decision_cfg,
                &decision_feed,
                &decision_profiler,
                &decision_predictor,
                &decision_nav,
                &decision_logger,
                &decision_executor,
                &decision_breach,
                shadow_only,
                invert_signal,
            )
            .await
            {
                error!("decision cycle error: {e:#}");
            }
        }
    });

    // Shadow evaluator: every 15s, scan unevaluated shadow predictions and
    // backfill actual SPY moves at 30/60/90s lookaheads. Runs in shadow mode
    // OR live mode (HERMES shadow data is useful too).
    let shadow_logger = sql_logger.clone();
    let shadow_feed = feed.clone();
    let shadow_handle = tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_secs(15));
        loop {
            tick.tick().await;
            if let Err(e) = evaluate_shadow_once(&shadow_logger, &shadow_feed).await {
                warn!("shadow evaluator: {e:#}");
            }
        }
    });

    // Kill-file watcher.
    let kill_logger = sql_logger.clone();
    let kill_file_path = cfg.kill_file.clone();
    let kill_handle = tokio::spawn(async move {
        let p = std::path::Path::new(&kill_file_path);
        loop {
            tokio::time::sleep(Duration::from_secs(2)).await;
            if p.exists() {
                kill_logger
                    .log_regime("kill_file_detected", &kill_file_path)
                    .await
                    .ok();
                warn!("kill file present at {kill_file_path} -- requesting shutdown");
                break;
            }
        }
    });

    // Wait for any termination signal.
    let mut sigint = signal(SignalKind::interrupt())?;
    let mut sigterm = signal(SignalKind::terminate())?;
    tokio::select! {
        _ = sigint.recv() => info!("SIGINT received"),
        _ = sigterm.recv() => info!("SIGTERM received"),
        _ = kill_handle => info!("kill-file shutdown"),
    }

    // Graceful shutdown: cancel background tasks.
    lat_handle.abort();
    lat_log_handle.abort();
    feed_handle.abort();
    decision_handle.abort();
    shadow_handle.abort();
    sql_logger.log_regime("engine_stop", "shutdown_signal").await.ok();

    info!("hft-engine stopped cleanly");
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn decide_once(
    cfg: &Config,
    feed: &DataFeed,
    profiler: &LatencyProfiler,
    predictor_arc: &Arc<Mutex<RlsPredictor>>,
    nav_arc: &Arc<Mutex<nav::NavState>>,
    sql_logger: &SqliteLogger,
    executor: &Executor,
    breach_start: &Arc<Mutex<Option<chrono::DateTime<chrono::Utc>>>>,
    shadow_only: bool,
    invert_signal: bool,
) -> Result<()> {
    // Latency breach watchdog: if p99 exceeds breach_ms continuously for
    // breach_minutes, mark NAV halted (regime event).
    let p99 = profiler.latency_p99_ms().await;
    {
        let mut b = breach_start.lock().await;
        match p99 {
            Some(v) if v > cfg.file.risk.latency_breach_ms => {
                let now = chrono::Utc::now();
                let start = b.get_or_insert(now);
                let mins = (now - *start).num_seconds() / 60;
                if mins >= cfg.file.risk.latency_breach_minutes as i64 {
                    let conn = sql_logger.raw();
                    let guard = conn.lock().await;
                    let st = nav_arc.lock().await.clone();
                    if !st.halted {
                        let halted = nav::mark_halted(&guard, &st)?;
                        *nav_arc.lock().await = halted.clone();
                        sql_logger
                            .log_regime("nav_halted_latency_breach", &format!("p99={v:.2}"))
                            .await
                            .ok();
                        warn!("HALTED: latency p99 {v:.2}ms exceeded for {mins} min");
                    }
                }
            }
            _ => {
                *b = None;
            }
        }
    }

    let snap = feed.snapshot().await;
    let mut prediction = {
        let p = predictor_arc.lock().await;
        p.build(&snap)
    };
    // Apply sign inversion AFTER the predictor builds. The RLS still learns
    // the real ES→SPY relationship; we just negate the trading signal so the
    // gate decides as if the predictor said the opposite. Tests whether the
    // predictor has an accidentally-inverted sign, or whether the inverse
    // direction (divergence-continuation vs mean-reversion) has edge.
    if invert_signal {
        prediction.predicted_bps = -prediction.predicted_bps;
    }

    let latest_spy = feed.latest_spy().await;
    let nav_state = nav_arc.lock().await.clone();
    let regime_pause = nav_state.halted
        || nav_state.consecutive_losses >= cfg.file.risk.max_consecutive_losses
        || nav_state.drawdown_pct() >= cfg.file.risk.max_drawdown_pct;

    // No SPY quote yet? Log a no-fire decision and return.
    let Some(quote) = latest_spy else {
        sql_logger
            .log_decision(
                false,
                Some(prediction.predicted_bps),
                Some(prediction.confidence),
                None,
                p99,
                None,
                snap.es.last().map(|b| b.close),
                "no_spy_quote_yet",
            )
            .await
            .ok();
        info!(
            "cycle: no SPY quote yet (es_n={} spy_n={})",
            snap.es.len(),
            snap.spy.len()
        );
        return Ok(());
    };

    let gate = executor.gate(&prediction, &quote, p99, nav_state.halted, regime_pause);
    let reasoning_json = serde_json::to_string(&gate).unwrap_or_default();
    sql_logger
        .log_decision(
            gate.fired,
            Some(prediction.predicted_bps),
            Some(prediction.confidence),
            Some(quote.spread_bps()),
            p99,
            Some(quote.mid()),
            snap.es.last().map(|b| b.close),
            &reasoning_json,
        )
        .await
        .ok();

    // Always log a shadow prediction — this is what makes calibration possible
    // without ever needing to trade. Outcomes are filled in by the evaluator.
    sql_logger
        .log_shadow_prediction(
            quote.mid(),
            Some(quote.bid),
            Some(quote.ask),
            Some(quote.spread_bps()),
            Some(prediction.predicted_bps),
            Some(prediction.confidence),
            p99,
            Some(prediction.samples as u32),
            gate.fired,
        )
        .await
        .ok();

    info!(
        "cycle pred={:.3}bps spread={:.3}bps conf={:.3} p99={} samples={} fired={} reasons={:?}",
        prediction.predicted_bps,
        quote.spread_bps(),
        prediction.confidence,
        p99.map(|v| format!("{v:.1}ms")).unwrap_or_else(|| "n/a".into()),
        prediction.samples,
        gate.fired,
        gate.reasoning
    );

    // Even when we DON'T fire, we still use the realized SPY 60s move to
    // refine the RLS predictor — this is how confidence eventually rises
    // out of the cold-start regime without ever needing to trade.
    if snap.spy.len() >= 3 && snap.es.len() >= 2 {
        let mut p = predictor_arc.lock().await;
        p.update(prediction.es_ret_bps, prediction.spy_ret_bps);
    }

    if !gate.fired {
        return Ok(());
    }

    // In shadow-only mode we log the prediction (above) and stop — never trade.
    if shadow_only {
        info!("[shadow-only] gate fired but skipping order placement");
        return Ok(());
    }

    // Sizing.
    let spy_price = quote.mid();
    if !(spy_price > 0.0 && spy_price.is_finite()) {
        warn!("SPY price invalid, skipping trade");
        return Ok(());
    }
    let qty = (cfg.file.sizing.notional_usd / spy_price).floor() as u64;
    if qty == 0 {
        warn!("computed qty=0, skipping");
        return Ok(());
    }

    let side = if prediction.predicted_bps > 0.0 {
        Side::Buy
    } else {
        Side::Sell
    };

    info!(
        "FIRING {} {} SPY (notional≈${:.0}, pred {:+.2}bps, conf {:.2})",
        side.as_str(),
        qty,
        qty as f64 * spy_price,
        prediction.predicted_bps,
        prediction.confidence
    );

    let entry = executor.place_order(side, qty, &quote).await;
    let (entry_px, filled_qty) = match entry {
        Ok((px, q)) => (px, q),
        Err(e) => {
            warn!("entry order failed: {e:#}");
            return Ok(());
        }
    };

    if filled_qty == 0 && !cfg.dry_run {
        warn!("entry did not fill (IOC); abandoning cycle");
        return Ok(());
    }

    // Hold and exit.
    tokio::time::sleep(Duration::from_secs(cfg.file.sizing.hold_seconds)).await;
    feed.refresh().await;
    let exit_quote = feed
        .latest_spy()
        .await
        .unwrap_or(quote.clone());

    let exit_side = side.opposite();
    let exit_qty = if cfg.dry_run { qty } else { filled_qty };
    let (exit_px, _) = match executor.place_order(exit_side, exit_qty, &exit_quote).await {
        Ok(v) => v,
        Err(e) => {
            warn!("exit order failed: {e:#}; using mid for PnL");
            (exit_quote.mid(), exit_qty)
        }
    };

    let pnl_per_share = match side {
        Side::Buy => exit_px - entry_px,
        Side::Sell => entry_px - exit_px,
    };
    let pnl_usd = pnl_per_share * exit_qty as f64;
    info!(
        "CLOSED {} SPY entry={:.4} exit={:.4} pnl=${:.2}",
        exit_qty, entry_px, exit_px, pnl_usd
    );

    // Apply PnL & persist.
    let conn = sql_logger.raw();
    let guard = conn.lock().await;
    let st = nav_arc.lock().await.clone();
    let next = nav::apply_pnl(&guard, &st, pnl_usd)?;
    drop(guard);
    *nav_arc.lock().await = next.clone();
    sql_logger
        .log_nav_snapshot(
            next.nav_usd,
            next.peak_nav_usd,
            next.drawdown_pct(),
            next.consecutive_losses,
            next.halted,
            &format!("trade_closed pnl={pnl_usd:.2}"),
        )
        .await
        .ok();

    if next.drawdown_pct() >= cfg.file.risk.max_drawdown_pct
        || next.consecutive_losses >= cfg.file.risk.max_consecutive_losses
    {
        let conn2 = sql_logger.raw();
        let g2 = conn2.lock().await;
        let halted = nav::mark_halted(&g2, &next)?;
        *nav_arc.lock().await = halted;
        sql_logger
            .log_regime(
                "nav_halted_drawdown_or_losses",
                &format!(
                    "dd={:.4} losses={}",
                    next.drawdown_pct(),
                    next.consecutive_losses
                ),
            )
            .await
            .ok();
    }

    Ok(())
}


/// Scans unevaluated shadow predictions and backfills actual SPY moves at
/// 30/60/90s lookaheads using the data feed's most recent SPY mid. A row is
/// marked `evaluated=1` only when the 90s horizon is filled.
async fn evaluate_shadow_once(
    sql_logger: &SqliteLogger,
    feed: &DataFeed,
) -> Result<()> {
    let pending = sql_logger.pending_shadow_rows(200).await?;
    if pending.is_empty() {
        return Ok(());
    }
    let latest_spy = match feed.latest_spy().await {
        Some(q) => q,
        None => return Ok(()),
    };
    let now = chrono::Utc::now();
    let spy_now = latest_spy.mid();
    for (id, ts_str, mid_at_log) in pending {
        let t0 = match chrono::DateTime::parse_from_rfc3339(&ts_str) {
            Ok(t) => t.with_timezone(&chrono::Utc),
            Err(_) => continue,
        };
        let age_s = (now - t0).num_seconds();
        if age_s < 30 || mid_at_log <= 0.0 {
            continue;
        }
        let bps = ((spy_now - mid_at_log) / mid_at_log) * 1e4;
        let (mut spy_30, mut spy_60, mut spy_90) = (None, None, None);
        let (mut bps_30, mut bps_60, mut bps_90) = (None, None, None);
        let mut mark = false;
        if age_s >= 30 { spy_30 = Some(spy_now); bps_30 = Some(bps); }
        if age_s >= 60 { spy_60 = Some(spy_now); bps_60 = Some(bps); }
        if age_s >= 90 { spy_90 = Some(spy_now); bps_90 = Some(bps); mark = true; }
        sql_logger.update_shadow_outcome(
            id, spy_30, spy_60, spy_90, bps_30, bps_60, bps_90, mark,
        ).await.ok();
    }
    Ok(())
}
