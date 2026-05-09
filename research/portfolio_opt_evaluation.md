# Portfolio_Opt Evaluation

Repository reviewed:

```text
https://github.com/AlasteirHo/Portfolio_Opt.git
```

Local inspection path:

```text
/private/tmp/Portfolio_Opt
```

## Verdict

Partially useful later. Do not merge into paper_v1.

## Useful Ideas

- Hierarchical Risk Parity portfolio construction.
- Risk parity / equal-risk-contribution optimizer.
- Max-Sharpe and min-variance optimizer comparisons.
- Ledoit-Wolf shrinkage covariance estimation.
- Volatility targeting overlay.
- Drift-based rebalance trigger.
- Chandelier-style trailing stop overlay.
- Backtest metrics and walk-forward reporting patterns.

## Not Directly Useful For paper_v1

- No options chain pricing pipeline.
- No earnings-event straddle logic.
- No Polygon options data integration.
- No Year 1 / Year 2 frozen evidence orchestration.
- Alpaca stock/ETF daily data path does not replace OPRA/Polygon options data.

## Future Use

Consider adapting concepts into paper_v2 after paper_v1 verdict:

- portfolio-level capital allocator across strategy sleeves;
- volatility target for total portfolio exposure;
- HRP or risk-parity allocator across multiple frozen champion strategies;
- risk-overlay ablation tests.

Any reuse should be reimplemented or vendored deliberately with tests rather
than copied directly into the current evidence path.
