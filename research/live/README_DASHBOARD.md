# Portfolio Dashboard — 4 Verified Strategies

Single-page Streamlit dashboard for live monitoring of all 4 paper-shadow strategies.

- File: `research/live/portfolio_dashboard.py`
- Default port: **8503** (system_dashboard uses 8502, default Streamlit is 8501)
- Refresh: **30 seconds**, fragment-based (no full page reload)
- Read-only against each strategy's local sqlite ledger

## Strategies tracked

| Strategy | DB file | Kill-switch file |
| --- | --- | --- |
| SVXY Sweet Spot           | `svxy_decisions.sqlite3`        | `STOP_SVXY`         |
| iter_18 ADR Arbitrage     | `adr_decisions.sqlite3`         | `KILL_SWITCH`       |
| iter_3 BTC Funding Carry  | `btc_funding_decisions.sqlite3` | `STOP_BTC_FUNDING`  |
| iter_6 Nikkei Gap         | `nikkei_decisions.sqlite3`      | `STOP_NIKKEI`       |

All four panels render even when the underlying host has not been launched yet —
missing DBs show as `NOT_DEPLOYED` rather than crashing the page. Status can
also be supplied via the A31 strategy registry at `meta_framework.sqlite3`.

## How to launch (LAN-accessible)

```bash
cd /Users/weeoto/turbo-carnival
.venv/bin/streamlit run research/live/portfolio_dashboard.py \
    --server.port 8503 --server.address 0.0.0.0
```

Run it inside `tmux` (and ideally `caffeinate -i`) on the Mac mini so it
survives terminal disconnects:

```bash
tmux new -ds portfolio_dash \
    "caffeinate -i /Users/weeoto/turbo-carnival/.venv/bin/streamlit run \
        /Users/weeoto/turbo-carnival/research/live/portfolio_dashboard.py \
        --server.port 8503 --server.address 0.0.0.0"
```

Re-attach with `tmux attach -t portfolio_dash`. Stop with `tmux kill-session -t portfolio_dash`.

## How to access from your phone

1. Make sure your phone is on the **same WiFi** as the Mac mini.
2. Find the Mac mini's LAN IP (one of these):
   ```bash
   ipconfig getifaddr en0     # WiFi
   ipconfig getifaddr en1     # Ethernet
   ```
   Or open the dashboard locally first — the header shows the URL it detected,
   e.g. `http://192.168.1.42:8503/`.
3. On your phone's browser navigate to `http://<MAC_IP>:8503` — e.g.
   `http://192.168.1.42:8503`. Bookmark it for one-tap access.

If you cannot reach it from the phone:

- macOS firewall: System Settings → Network → Firewall → allow incoming for
  `python3` / Streamlit.
- Confirm port 8503 is actually listening: `lsof -i :8503` on the Mac mini.

## What each panel shows

### Header bar
- **Total NAV** across all 4 strategies.
- **Daily P&L** (combined unrealized).
- **Open positions** count (live broker rows).
- **Last update** timestamp from the most recent host ledger write.

### Strategy panels (one per row)
Each strategy renders a row with:

- **Status pill**: `SHADOW` / `TINY_LIVE` / `LIVE` / `KILLED` / `NOT_DEPLOYED`.
  Status comes from `meta_framework.sqlite3` if present; otherwise from the
  stop-file convention (kill-switch file present → `KILLED`).
- **Today's signal**: human-readable sentence pulled from the strategy's
  latest `decisions` row (e.g. for SVXY: `VIX/VIX3M = 0.87 → LONG SVXY`).
- **Current positions**: most recent positions snapshot from the broker, as
  logged by the host script.
- **Last 5 trades**: timestamp, side, qty, price (and P&L where the schema
  supplies it).
- **Rolling 30-day stats**: Sharpe, max drawdown %, win-rate %.
- **Kill switch button** (see below).

### Portfolio combined
- **Combined daily P&L %** chart, last 90 days, equal-weight across legs.
- **Combined drawdown %** area chart.
- **Sharpe bar chart** comparing the combined Sharpe to each individual leg.
- **Pairwise correlation heatmap** of leg daily returns over the last 30 days.

### Risk
- **Gross exposure / NAV** aggregate.
- **Per-strategy capital allocation** table (NAV $, gross $, NAV %).
- **Distance to nearest kill criterion** with a horizontal headroom bar — e.g.
  `ADR pair NTES: -3.1% / kill at -5% → 38% headroom`.
- **Promotion checkpoints**: days each strategy has been running in
  paper-shadow and days until the next 90-day review.

## Kill switch usage

Each strategy panel has a red **KILL** button. Pressing it writes the
strategy's stop file (`STOP_SVXY`, `KILL_SWITCH`, `STOP_BTC_FUNDING`, or
`STOP_NIKKEI`) into `research/live/`. The host scripts check that file each
iteration and **halt without flattening positions** — investigate before
removing the stop file.

When a strategy is already killed, the same panel shows a **Resume** button
that deletes the stop file. The host will pick up trading again on its next
cycle (or you need to relaunch the daemon depending on the host's loop
structure — check the host's README for confirmation flow).

## Architecture notes

- Each panel is a `st.fragment(run_every=30)` so refreshes are independent
  and the whole page does not reload.
- All sqlite reads are opened **read-only** via the `file:?mode=ro` URI to
  avoid contending with the host writers.
- Missing tables / missing DBs / missing columns are handled defensively —
  the dashboard never raises on schema drift. It just shows `—` where data
  is unavailable.
- Mobile responsive CSS shrinks padding and KPI widths below 768px so the
  layout reads well on a phone in portrait mode.
