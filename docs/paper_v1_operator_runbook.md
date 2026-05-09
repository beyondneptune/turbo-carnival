# paper_v1 Operator Runbook

This runbook is for the transition from the Polygon chain-pricing augment to
the frozen Year 1 train / Year 2 held-out evidence run.

## Current Rule

Do not change paper_v1 strategy, gates, model targets, or evaluation windows
while an evidence cycle is active. Infrastructure/readiness/dashboard changes
are allowed when they do not alter execution decisions.

## Monitor Augment

Dashboard:

```text
http://100.117.50.80:8501
```

Open:

```text
Live -> Paper v1 Run -> Behind the scenes
```

Terminal status:

```bash
.venv/bin/python - <<'PY'
import json
s = json.load(open("data/paper_v1_status.json"))
print("phase:", s.get("phase"))
for name, stage in s.get("stages", {}).items():
    if isinstance(stage, dict) and stage.get("status") == "running":
        print(name, stage.get("done"), "/", stage.get("total"), stage.get("files_written"))
PY
```

Expected phases:

```text
augmenting -> augment_complete
```

Expected priced cache roots:

```text
data/cache/year1/chains_priced/
data/cache/year2/chains_priced/
```

## After `augment_complete`

Run the readiness check first if available:

```bash
.venv/bin/python scripts/check_paper_v1_readiness.py
```

Machine-readable form:

```bash
.venv/bin/python scripts/check_paper_v1_readiness.py --json
```

If readiness passes, run the frozen evidence orchestrator:

```bash
caffeinate -imsu .venv/bin/python scripts/run_paper_v1_evidence.py
```

The orchestrator must read cached data only. It should not call Polygon.

## Expected Evidence Sequence

1. Load Year 1/Year 2 cache.
2. Prefer `chains_priced/` over raw `chains/`.
3. Fail closed if official verdict would require synthetic option prices.
4. Train models on Year 1.
5. Freeze model selection.
6. Replay Year 2 through gates and execution simulator.
7. Persist verdict to `data/paper_v1_status.json`.
8. Show final gate verdict in dashboard.

## Do Not

- Do not restart the augment job unless it is clearly dead.
- Do not run synthetic verdict mode for the official result.
- Do not tune on Year 2 and call it OOS.
- Do not merge Portfolio_Opt into paper_v1.
- Do not enable live trading from this cycle.

## Useful Side Data

Offline context data lives here:

```text
data/offline_context/
```

Refresh command:

```bash
.venv/bin/python scripts/fetch_offline_context_data.py
```

This data is for postmortem/regime context and future paper_v2 features. It is
not part of the frozen paper_v1 decision path unless explicitly promoted in a
new prereg/version.
