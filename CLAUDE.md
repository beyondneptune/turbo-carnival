# turbo-carnival Claude Code Instructions

## Project State
- `paper_v1` is a Year 1 train / Year 2 held-out out-of-sample evidence run for a hybrid options strategy.
- The current hard blocker is real option-chain pricing: do not run a final verdict on metadata-only chains or synthetic fallback prices.
- Polygon Free is allowed. Optimize for resumable, cache-first jobs under the 5 rpm limit rather than paid-only assumptions.

## Hard Rules
- Never tag `paper_v1.0-frozen`; the tag is a CEO action.
- Never treat synthetic/demo/generated option prices as official `paper_v1` evidence.
- If chain pricing columns (`bid`, `ask`, `mid`, `underlying_close`, `atm_iv`) are missing, fix/augment the cache before running the final evidence verdict.
- LLMs are read-only during evidence generation: no model, threshold, feature, risk-gate, metadata, or code changes after evidence results are inspected.
- Do not add `Co-Authored-By` or AI attribution trailers to commits.
- Do not rewrite/amend pushed commits or force-push unless explicitly asked.

## Workflow
- Explore first when the task touches evidence, metadata, or API/cache behavior. Then patch narrowly and verify.
- For `/loop` style requests, distinguish terminal idle states from real bugs. `cache_complete` is terminal; stale status is only a bug if the dashboard mislabels it as hung.
- Prefer root-cause fixes over user decisions when the issue is an implementation gap.
- Keep generated data out of git. Runtime files under `data/` are local cache/output.

## Verification
- Full tests: `.venv/bin/python -m pytest tests/ -q`
- Lint: `.venv/bin/ruff check ingest/ features/ signals/ risk/ paper_exec/ eval/ scripts/ dashboard/ tests/ conftest.py`
- Dashboard LAN run: `.venv/bin/streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 8501`
- Long data jobs should be run with `caffeinate` and preferably `tmux`; write progress to `data/paper_v1_status.json`.

## Current Evidence Policy
- Year 1: training/calibration only.
- Year 2: one-shot held-out OOS evidence after freeze/tag.
- Optional 90-day forward paper run is confirmatory and reported separately.
- Gate 7 counts only held-out Year 2 satellite trades produced by the frozen strategy.
