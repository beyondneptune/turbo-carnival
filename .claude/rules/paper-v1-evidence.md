---
paths:
  - "scripts/run_paper_v1_evidence.py"
  - "scripts/fetch_paper_v1_data.py"
  - "scripts/run_daily_paper.py"
  - "signals/**/*.py"
  - "features/**/*.py"
  - "risk/**/*.py"
  - "paper_exec/**/*.py"
  - "eval/**/*.py"
---

# Paper v1 Evidence Rules

- Official evidence must use real cached option pricing. Do not silently fall back to synthetic prices for final `paper_v1` results.
- If a cache has contract metadata but lacks `bid`, `ask`, `mid`, `underlying_close`, or `atm_iv`, build a resumable pricing augmentation pass before replay.
- Prefer narrow pricing first: satellite ATM call/put entry+exit and SPY core condor legs entry+exit. Wide vol-surface pricing is a later enhancement.
- Write augmented pricing beside raw metadata, e.g. `chains_priced/`, so raw cache remains auditable.
- Free-tier Polygon runs must be idempotent, resumable, and status-writing. Avoid refetching files that already exist and validate cleanly.
- The final evidence run should fail closed if required pricing is absent unless an explicit non-evidence dry-run flag is used.
- Synthetic modes are allowed only for tests, demos, and plumbing validation, never for the official Year 2 verdict.
