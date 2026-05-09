---
paths:
  - "dashboard/app.py"
  - "scripts/fetch_paper_v1_data.py"
  - "scripts/run_paper_v1_evidence.py"
---

# Dashboard and Status Rules

- `data/paper_v1_status.json` is the dashboard contract. Keep it atomic, readable, and backward compatible.
- Terminal phases such as `cache_complete`, `complete`, `error`, and `interrupted` should not be shown as "hung" merely because they are old.
- Clear stale crash stages when a later successful run supersedes them, or make the dashboard classify them as historical.
- Mobile viewing matters: keep dashboard panels responsive and useful on a phone over LAN.
- System metrics should prioritize CPU, per-core load, memory, disk, process RSS, API call counts, cache progress, and elapsed time. Apple GPU metrics are not required for `paper_v1`.
