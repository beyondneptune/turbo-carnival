# IBKR Paper-Trading Setup (Mac mini)

## Why this is a manual step

The IBKR API is not username/password-based. Your Python script connects to a
**running IB Gateway** (or TWS) on localhost. The credentials are entered into
the Gateway GUI at startup — never directly in code.

So you need to:

1. Download IB Gateway (or TWS) once.
2. Log into it with your paper credentials.
3. Enable the API in its settings.
4. Leave it running while your script trades.

After that, `ingest/ibkr_client.py` connects automatically.

## One-time download + setup

### Step 1 — Download IB Gateway (recommended over TWS for headless trading)

1. Go to https://www.interactivebrokers.com/en/trading/ibgateway-stable.php
2. Download the macOS installer (.dmg).
3. Install. The app will be `IB Gateway` in /Applications.

(TWS works too but uses more memory. Gateway is lighter and is what production
algo trading typically uses.)

### Step 2 — Log in once with paper credentials

Paper credentials are stored locally in `council/.env.local` (gitignored), not
in this file. When you launch IB Gateway, **select "Paper Trading" mode** (NOT
Live) and use the credentials from that file.

Note: IBKR username/password are entered ONLY in the Gateway GUI. They are not
used by Python code. The API connects to the running Gateway via socket.

### Step 3 — Enable the API

In IB Gateway, go to:

`Configure → Settings → API → Settings`

Then:
- ✅ Enable ActiveX and Socket Clients
- ✅ Read-Only API: **uncheck** (so we can place orders)
- Socket port: `4002` (default for paper Gateway; TWS paper uses 7497)
- Master API client ID: leave blank
- Trusted IPs: add `127.0.0.1`
- ⬜ Allow connections from localhost only (recommended)

Click OK, then re-launch the Gateway to pick up the changes.

### Step 4 — Verify the connection

From the repo root:

```bash
.venv/bin/python ingest/ibkr_client.py
```

You should see something like:

```json
{
  "connected": true,
  "summary_keys": ["AvailableFunds__USD", "BuyingPower__USD", ...],
  "summary_sample": {"AvailableFunds__USD": "1000000.00", ...}
}
```

If you see `ConnectionRefusedError`, the Gateway isn't running OR the API port
isn't enabled.

## What about credentials in code

We deliberately do NOT store the username/password in `.env.local` or anywhere
else in the repo. They aren't used by the API. They're only used to log into
the Gateway GUI once — after that, the Gateway maintains the session.

For automated headless servers there's a separate IBC (IB Controller) tool that
auto-launches Gateway and re-logs in on schedule. Out of scope for tonight.

## Currency note (USD vs GBP)

Your IBKR paper account is denominated in USD by default. You can hold
multi-currency balances, but:

- SPY options are USD-denominated. They cannot be priced or settled in GBP.
- If you fund the live account in GBP, IBKR auto-converts to USD on the first
  trade with a small FX fee. Cheaper to fund USD directly.
- For showing returns to a UK-based investor in GBP, just multiply by the
  current GBP/USD rate (~0.79 as of May 2026). Cosmetic conversion only.

There is no version of this strategy that trades GBP-denominated instruments —
SPY options exist only in USD. If you want a GBP-native version, the closest
analogue would be FTSE 100 options on the LSE (different contract structure,
different underlying, would need its own backtest).

## After setup

Once Gateway is running, the Python adapter can:
- Fetch SPY option chains in real time
- Pull account NAV / buying power / positions
- Place 4-leg iron condor combo orders (after the scaffold's `place_iron_condor_paper`
  is wired — currently raises `NotImplementedError`)

The scaffold at `ingest/ibkr_client.py` is the bridge.
