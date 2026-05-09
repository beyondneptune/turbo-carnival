---
name: paper-v1-loop
description: Use when the user asks to keep /loop running, fix paper_v1 bugs until none remain, triage dashboard stale states, or continue the evidence run.
---

# Paper v1 Loop Workflow

1. Check repo and process state:
   - `git status -sb`
   - inspect `data/paper_v1_status.json`
   - check whether Streamlit responds on LAN if relevant.
2. Classify the state:
   - Terminal idle (`cache_complete`, `complete`) is not a hang.
   - A stale running phase, missing required cache fields, failed tests, or stale crash stages are bugs.
3. If pricing is missing from chains, treat it as an implementation bug:
   - build/resume narrow chain pricing augmentation first;
   - write to `chains_priced/` or equivalent audit-preserving output;
   - do not run official evidence on synthetic prices.
4. Patch root causes narrowly and add tests when feasible.
5. Verify:
   - ruff check for touched Python surface;
   - targeted tests, then full tests if shared behavior changed;
   - status JSON/dashboard behavior for status-related fixes.
6. Commit and push only when the user asks for GitHub update or the current task explicitly includes it.

Default decision: if asked `narrow go` vs `wide go`, choose `narrow go` for `paper_v1` unless the user explicitly requests vol-surface research coverage.
