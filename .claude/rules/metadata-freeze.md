---
paths:
  - "metadata/**/*.md"
---

# Metadata and Freeze Rules

- Metadata is part of the preregistered methodology. Be precise and avoid relative wording where exact dates or constants are needed.
- `paper_v1.0-frozen` is a CEO tag. Claude may prepare commits but must not create the tag unless explicitly instructed.
- Any post-evidence tweak to models, thresholds, risk gates, features, costs, or metadata creates a new cycle (`paper_v2`) rather than repairing `paper_v1`.
- Historical Year 2 evidence is admissible only as a one-shot held-out test after the freeze point. It must not inform training, feature selection, threshold tuning, or reruns.
- Weekly LLM review may summarize and diagnose, but cannot modify the running/evidence methodology inside `paper_v1`.
