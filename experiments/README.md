# Experiments

Offline research lives here. These runs are intentionally isolated from the live
paper DB and status files.

Use this area to test rolling train/test windows, feature ideas, cost stresses,
and risk overlays. Outputs go under `experiments/runs/` and are gitignored.

Quick start:

```bash
.venv/bin/python experiments/run_window_experiments.py \
  --train-days 180,252,365 \
  --test-days 60,90 \
  --step-days 30 \
  --max-windows 12
```

The runner writes:

- `summary.csv` - one row per window
- `summary.json` - full result payload
- `windows.json` - generated window definitions
- `exp*/experiment.db` - scratch audit DB for that window
- `exp*/status.json` - scratch status file for that window

Rules:

- Do not use `data/trading_system.db` for experiments.
- Do not write `data/paper_v1_status.json` from experiments.
- Do not name experiment runs `paper_v1_evidence_*`; the watchdog treats that
  as official evidence.
- A good experiment is allowed to inspire a future cycle, but it does not alter
  the current paper month rules.

