# Step 1: Initial Framework — Options Data Sources Research

**Topic:** Free and low-cost options market data sources for systematic trading research, with focus on local storage strategies, UK-accessible options, alternative routes (academic, scraping, self-collection), and rate-limit workarounds.

**Date:** 2026-05-06
**Project context:** $10k paper notional hybrid options strategy (VRP core on SPX/SPY + earnings satellite on Russell 1000 top-50 by ADV). UK-based engineer. Need 2+ years history minimum, prefer 5+. Budget preference: free first, then $30/mo, only paid >$200/mo if justified.

---

## Items List (Data Sources)

### A. Free / Freemium APIs (Stocks + Limited Options)
1. **Polygon.io Free** — 5 calls/min, 2yr history, EOD options chains, free tier
2. **yfinance (Yahoo Finance unofficial Python lib)** — free, deep stock history, current options chain, no historical IV
3. **Alpha Vantage Free** — 25 calls/day, stock focus, very limited options
4. **Tiingo Free** — stock prices, fundamentals, no options
5. **Finnhub Free** — stocks, basic options, 60 calls/min
6. **Twelve Data Free** — 800 calls/day, stocks+forex, limited options
7. **EOD Historical Data** — paid but cheap, broad coverage
8. **Marketstack Free** — 1000 calls/month, stocks only
9. **Stooq** — free CSV downloads, broad historical stocks
10. **dxFeed CPaaS sample** — limited demo data

### B. Brokerage APIs (Account Required)
11. **Interactive Brokers TWS API / IB Gateway / Client Portal API** — full options, paper account free
12. **Alpaca** — free, stocks-focused, no real options API
13. **TD Ameritrade / Schwab Trader API** — TD legacy migrated to Schwab, has options
14. **Tradier (US only)** — free with brokerage, full options API
15. **E*TRADE API** — US, free with account
16. **tastytrade API (US only)** — options-focused, excellent
17. **Saxo Bank OpenAPI** — UK-accessible, has options
18. **Webull API** — limited
19. **Robinhood (unofficial)** — fragile, US-only

### C. Paid Sources (Cheap Tier, $30–$200/mo)
20. **ThetaData** (~$30+/mo) — historical options, decent quality
21. **Polygon Options Starter** ($199/mo) — 5yr history, intraday
22. **Tiingo Power** ($30+/mo) — better limits but no options
23. **Intrinio Bronze** — moderate cost, options data
24. **Finnhub paid tiers**

### D. Paid Sources (Expensive Tier, $250+/mo)
25. **ORATS Data** (~$250+/mo) — vol surface focus, high quality
26. **ORATS Backtester** — even more expensive
27. **CBOE LiveVol / DataShop** — institutional-grade
28. **Refinitiv (Reuters Eikon/Workspace)**
29. **Bloomberg Terminal/API** — gold standard, $24k+/yr
30. **Algoseek** — tick data, expensive
31. **TickData**

### E. Academic / Institutional
32. **WRDS (Wharton Research Data Services)** — university subscription gateway
33. **OptionMetrics IvyDB** (via WRDS) — gold standard historical options, 1996+
34. **CRSP** (via WRDS) — stock prices, returns, going back decades
35. **CRSP/Compustat Merged** — fundamentals + prices
36. **TAQ (via WRDS)** — trade and quote data
37. **Datastream / Refinitiv via university library**
38. **Bloomberg Terminal via university library**
39. **NYU Stern Volatility Lab** — free academic vol research
40. **JKU AURA** — academic options research dataset

### F. Public / Scraped / Free Bulk
41. **SEC EDGAR** — filings (10-K, 10-Q, 8-K, earnings), free
42. **FRED (Federal Reserve Economic Data)** — macroeconomic, free, deep history
43. **Yahoo Finance HTML scraping** — gray area legally
44. **Nasdaq.com scraping**
45. **Cboe.com public historical downloads** — limited but real
46. **OPRA tape** — rarely accessible to retail
47. **CFTC Commitment of Traders** — free, weekly futures positioning
48. **Federal Reserve H.15** — interest rates

### G. Free Historical Dataset Dumps
49. **Kaggle options datasets** — community-uploaded, quality varies
50. **OpenBB Terminal data** — open-source aggregator
51. **GitHub options historical data repos** — community
52. **Academic paper replication datasets** — JFE/JFQA dataverse, Harvard Dataverse
53. **AQR factor data** — free factor returns
54. **Ken French data library** — free factor and portfolio returns
55. **Data Bento sample**

### H. Self-Collection / Build-Forward
56. **Daily snapshot pipeline** — collect what you can today, accumulate over months/years
57. **Web scraping current chains daily** — Yahoo, Nasdaq, brokers
58. **Discord / Reddit data dumps** — unreliable but free
59. **Polygon Free + 5+ years of accumulation** — start now, deeper history later

### I. UK-Specific Considerations
60. **Saxo Bank UK** — options accessible, OpenAPI
61. **IBKR UK** — gold standard for UK retail options API
62. **London Stock Exchange data** — limited historical
63. **ICE Futures UK** — futures data
64. **EuroNext** — European options
65. **Greenwich University WRDS** (if subscribed) — could change everything

### J. Specialized / Alternative
66. **VIX historical** — Cboe direct download, free
67. **Open Interest data** — various sources
68. **dividends and corporate actions** — yfinance, SEC
69. **Earnings calendars** — Yahoo, Earnings Whisper, Zacks
70. **Analyst estimates dispersion** — Refinitiv I/B/E/S, Estimize free

---

## Field Framework (per source)

### Basic Info
- `name`: source name
- `type`: API / CSV download / scraping / brokerage-tied / academic-gateway
- `provider`: company/organization
- `homepage`: URL
- `signup_url`: where to register

### Coverage
- `assets`: stocks / options / futures / fx / crypto / fundamentals / news
- `markets`: US / UK / EU / global
- `option_types`: equity / index / weekly / monthly / leaps

### Historical Depth
- `history_years`: how far back data is available
- `start_date`: earliest available date
- `gap_handling`: any known gaps in history

### Granularity
- `frequency`: tick / second / minute / hour / EOD / weekly
- `eod_timestamp`: when EOD snapshot is taken
- `intraday_available`: yes/no

### Options-Specific Fields
- `chains`: full chain available?
- `greeks_provided`: yes/no (or computed by us)
- `iv_provided`: yes/no
- `vol_surface_provided`: raw chain only or fitted surface?
- `bid_ask_provided`: real-time, delayed, or last-trade only
- `oi_provided`: open interest available?

### Access
- `rate_limit_per_minute`
- `rate_limit_per_day`
- `rate_limit_per_month`
- `concurrent_connections`
- `access_methods`: REST / WebSocket / FTP / SFTP / SDK / file download
- `python_sdk`: official / community / none
- `auth_type`: API key / OAuth / brokerage login / institutional

### Cost
- `free_tier`: yes/no
- `free_tier_limits`: specific constraints
- `cheapest_paid_tier`: $/month
- `cheapest_full_tier`: $/month for serious use

### Legal / ToS
- `redistribution_allowed`: yes/no
- `scraping_risk`: low/medium/high
- `commercial_use_allowed`: yes/no
- `account_kyc_required`: yes/no
- `geographic_restrictions`: any?

### UK Specifics
- `uk_accessible`: yes/no/partial
- `uk_payment_methods`: GBP cards/bank, USD only?
- `uk_data_protection_compliance`: GDPR considerations

### Storage / Integration
- `storage_format_recommended`: parquet / sqlite / csv / hdf5
- `partitioning_scheme`: by date / by ticker / by month
- `incremental_update_supported`: yes/no
- `local_cache_friendly`: yes/no

### Quality
- `known_issues`: gaps, errors, late updates
- `data_validation`: how to verify quality
- `community_reputation`: trader/dev community feedback

---

## Recommended Local Storage Architectures (to be researched in depth)

1. **Parquet + Hive partitioning** — read with Pandas/Polars/DuckDB
2. **DuckDB** — embedded analytics DB, reads Parquet directly, SQL on cached files
3. **SQLite** — simple, single-file, great for small project
4. **HDF5 / PyTables** — older standard, time-series friendly
5. **Arrow IPC files** — fast, columnar
6. **TimescaleDB** — Postgres extension for time-series (overkill for $10k project)
7. **InfluxDB** — time-series database
8. **Plain CSV with date partitioning** — simple, slow, large
9. **Pickle dataframes** — Python-only, fast, fragile
10. **Feather files** — Arrow IPC, fast

For our scale (75 names × 504 days × ~100 contracts/day = ~3.8M rows), **DuckDB on Parquet** is likely optimal.

---

## Topic Slug
`options-data-sources`

## Output Directory
`C:\NIYAZ JIJU\trading_system\research\options-data-sources\`
