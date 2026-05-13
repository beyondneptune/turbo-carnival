# Archive

Files moved here are **inactive** — kept for study reference, post-mortem, or historical record. None of these are imported, executed, or read by the live trading system.

```
archive/
├── deprecated_dashboards/       superseded UI layers
│   ├── simple_live.py           Streamlit version (replaced by FastAPI pro_live.py)
│   └── live_monitor.py          FastAPI v1 (basic, replaced by pro_live.py v2)
│
├── deprecated_hosts/             older Python strategy hosts (current versions live in research/live/)
│   ├── atlas_shadow_hft.py      Python shadow lab — superseded by Rust --shadow-only
│   ├── iter_18_adr_host.py      pre-Alpaca ADR host (now iter_18_adr_alpaca_host.py)
│   ├── iter_6_nikkei_host.py    early Nikkei host (now iter_6_nikkei_etf_host.py)
│   ├── iter_3_btc_funding_host.py  early BTC funding (now iter_3_kraken_btc_funding_host.py)
│   ├── iter_26_uk_ftse_host.py  UK FTSE — built but failed validation, never deployed
│   └── exp_ofi_bandit_host.py   OFI bandit — built per single-account framework correction, never launched
│
├── old_reports/                  one-off markdown reports from earlier phases
│   ├── HANDOFF_TO_CODEX.md
│   ├── HARD_RECOMMENDATION_FOR_TOMORROW.md
│   ├── FINAL_VERDICT_14PCT_TARGET.md
│   ├── RECOMMENDATIONS.md
│   └── today_postmortem.py      one-off NIM panel runner for 2026-05-13 ADR losses
│
└── yesterday_experiments/        previous-day experiments
    └── fullonriskmodelterminator/   3-lane risk model (decommissioned 2026-05-12)
```

## Why these are kept (not deleted)

- **Study cases** — the deprecated_hosts show the evolution from yfinance → broker-native; useful for new contributors
- **Post-mortem evidence** — if a strategy reverts to an old behavior, we can diff against the archived version
- **Council artifacts** (under `council/*_runs/` — gitignored) — historical record of LLM-driven design reviews

## Why these are NOT in `research/live/`

Anything in `research/live/` is assumed to be a candidate for active deployment. Archive separation prevents accidental tmux launches against superseded code paths.
