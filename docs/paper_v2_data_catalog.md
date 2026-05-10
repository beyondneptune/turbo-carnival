# paper_v2 Data Catalog

This catalog is for paper_v2 feature engineering and postmortems. Do not wire
these fields into paper_v1 after the Year 2 holdout window has been assembled;
doing so changes the strategy specification and resets the cycle.

## Current Local Cache

Default location: `data/offline_context/`

- `yfinance/daily/{symbol}.parquet`: daily OHLCV/context bars.
- `yfinance/actions/{symbol}.parquet`: dividends and splits.
- `fred/{series_id}.parquet`: normalized FRED series with
  `date`, `value`, `series_id`, `description`.
- `manifest.json`: file count, byte count, and fetch summaries.

## Already Cached

- Underlyings: paper_v1 satellite tickers plus `SPY`.
- Market ETFs: broad equity, rates, credit, commodities, semis, banks.
- Sector ETFs: `XLB`, `XLC`, `XLE`, `XLF`, `XLI`, `XLK`, `XLP`, `XLRE`,
  `XLU`, `XLV`, `XLY`.
- Volatility: `^VIX`, `^VVIX`, `^VIX3M` and FRED `VIXCLS`.
- Macro/rates/credit: Treasury yields/curves, SOFR/Fed funds, credit OAS,
  WTI, dollar index, CPI, unemployment, payrolls, industrial production.

## Added Low-Disruption Sources

These are added to `scripts/fetch_offline_context_data.py` because they fit the
existing parquet/status/manifest pattern.

- Yahoo context tickers: `UUP`, `FXE`, `FXY`, `VIXY`, `SVXY`, `^VXN`,
  `^RVX`, `^MOVE`.
- FRED volatility series: `VXNCLS`, `RVXCLS`, `OVXCLS`, `GVZCLS`.
- FRED liquidity/rates: `EFFR`, `IORB`, `WALCL`, `RRPONTSYD`.
- FRED credit yields: `BAMLH0A0HYM2EY`, `BAMLC0A0CMEY`.
- FRED financial conditions: `NFCI`, `ANFCI`, `STLFSI4`.
- FRED macro/FX/uncertainty: `DEXUSEU`, `DEXJPUS`, `USEPUINDXD`,
  `ICSA`, `CCSA`, `UMCSENT`.

## Next Free Sources Worth Adding

These should be separate source families, not squeezed into the FRED/Yahoo
fetch paths.

- SEC EDGAR company facts/submissions: filings, as-of-safe fundamentals, 8-K
  event markers. Needs CIK mapping and SEC-compliant User-Agent handling.
- FINRA short interest: twice-monthly sentiment/crowding features. Needs symbol
  normalization and publication-date lag handling.
- OCC volume/open-interest reports: options market participation context. Needs
  a dedicated parser and careful contract/class mapping.
- Cboe public market statistics: aggregate option volume and put/call ratios.
  Useful for regime context, but not a replacement for historical NBBO chains.
- CFTC Commitment of Traders: weekly futures positioning for index/rates/oil
  regime context.
- Ken French / AQR factor libraries: equity factor regime controls.

## Leakage Rules

- Use publication/release date, not period end date, for macro and filings.
- Never let Year 2 target outcomes choose paper_v2 features or thresholds.
- Fit transforms on Year 1 only; apply frozen transforms to Year 2.
- Treat missing data as a feature-quality problem, not a reason to silently
  forward-fill across event boundaries without tests.

