# Council

This folder runs an NVIDIA NIM LLM council over the local trading research
outputs.

The council does not let models trade. It asks models to critique our evidence,
then saves every prompt and response so the result is auditable.

## Setup

Option A, paste the key into the local ignored file:

```bash
council/.env.local
```

Use this format:

```bash
NVIDIA_API_KEY="your_key_here"
```

Option B, export it in your shell:

```bash
export NVIDIA_API_KEY="your_key_here"
```

The key is read from the environment and is never written to disk.

## Check Available Models

```bash
.venv/bin/python council/run_council.py list-models
.venv/bin/python council/run_council.py validate-models --mode core
.venv/bin/python council/run_council.py validate-models --mode expanded
```

If NVIDIA renames or removes a model, edit `council/config.default.json`.

## Run Core Council

```bash
.venv/bin/python council/run_council.py run --mode core --skip-unavailable
```

This runs the main six-member council:

- Chair / synthesizer
- Quant research judge
- Risk manager
- Execution/options specialist
- Portfolio allocator
- Devil's advocate

Outputs go to:

```text
council/runs/<timestamp>/
```

Important files:

- `context.md` - the evidence packet sent to the models
- `prompts/` - exact prompts
- `member_responses/` - individual model answers
- `council_report.md` - final chair synthesis

## Run Expanded Council

```bash
.venv/bin/python council/run_council.py run --mode expanded --skip-unavailable
```

Limit member count if you want a smaller broad run:

```bash
.venv/bin/python council/run_council.py run --mode expanded --max-members 12 --skip-unavailable
```

## Dry Run

Use this before spending API calls:

```bash
.venv/bin/python council/run_council.py run --mode core --dry-run
```

This writes the evidence packet and prompts without calling NVIDIA.

## Ask A Follow-Up

After a council run:

```bash
.venv/bin/python council/run_council.py ask --run latest --question "Which SPY core variant should we paper next and why?"
```

The answer is saved under:

```text
council/runs/<timestamp>/followups/
```

## Recover A Report From A Partial Run

If some model endpoints are slow but enough responses have already been saved:

```bash
.venv/bin/python council/run_council.py synthesize --run latest
```

Or point to a specific run:

```bash
.venv/bin/python council/run_council.py synthesize --run council/runs/20260510T150329Z
```

## Research Lab Mode

The council can also propose new experiments, not just review old ones.

Dry run:

```bash
.venv/bin/python council/research_lab.py propose --mode expanded --max-calls 30 --dry-run
```

Live broad proposal run:

```bash
.venv/bin/python council/research_lab.py propose --mode expanded --max-calls 30 --workers 6 --skip-unavailable
```

This writes:

```text
council/lab_runs/<timestamp>/validated_spy_core_specs.json
council/lab_runs/<timestamp>/lab_summary.md
council/lab_runs/<timestamp>/raw_responses/
```

If a broad run is partially complete and one endpoint is hanging, finalize from
the responses already saved:

```bash
.venv/bin/python council/research_lab.py finalize --run latest
```

Then run the validated model-proposed SPY-core tests:

```bash
.venv/bin/python experiments/run_spy_core_strategy_grid.py \
  --spec-json council/lab_runs/<timestamp>/validated_spy_core_specs.json \
  --output-dir experiments/runs/council_spy_grid_<timestamp>
```

Safety boundary: NIM models propose tests; local deterministic code validates
the parameters and runs the backtest. Models do not directly trade or edit the
live runner.

## How Codex / Claude Communicate With The Council

Codex and Claude do not directly become part of NVIDIA NIM. The workflow is:

1. Local scripts generate evidence files and charts.
2. `council/run_council.py` packages that evidence into `context.md`.
3. NIM models independently write critiques.
4. The chair model synthesizes the critiques into `council_report.md`.
5. Codex/Claude can read that report and continue implementation.

That gives us a clean audit trail instead of an untraceable chat.
