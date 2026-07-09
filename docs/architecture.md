# VERA-MH Architecture

Validation of Ethical and Responsible AI in Mental Health: simulate mental-health conversations, evaluate them against a clinical rubric, and aggregate scores for comparison across provider agents.

This document describes the **target architecture**. Implementation may lag; see [Migration from current layout](#migration-from-current-layout) for known gaps. [README.md](../README.md) covers setup and CLI usage; this doc defines structure, data flow, and what **must** hold.

## System overview

Two independent pipelines share infrastructure only:

- **Generation** — persona LLM ↔ provider-agent LLM → transcript files
- **Judging** — judge LLM walks rubric questions → per-dimension severity → scores

They never import each other. A full workflow runs generation, judging, scoring, and optional pooling in sequence.

All user-facing operations go through **`vera.py`** subcommands. Domain packages are libraries; they are not invoked directly as scripts.

```text
data/personas.tsv ──► generate ──► p_*/conversations/*.txt
                                        │
                                        ▼
                                   judge ──► j_*/results.csv
                                        │
                                        ▼
                                   score ──► j_*/scores/
                                        │
                                        ▼
                                   pool ──► j_pooled__*/
```

## Domain model

| Concept | Location | Notes |
|---------|----------|-------|
| Persona | `data/personas.tsv` | Simulated patient; drives the user-side LLM |
| Transcript | `p_*/conversations/*.txt` | Turn-by-turn chat log |
| Rubric | `data/rubric.tsv` | Question flow, dimensions, severity |
| Evaluation run | `j_*/` | TSV results, logs, metadata |
| Dimension score | `judge/score.py` | Aggregated from rubric answers |
| Pooled scores | `judge/pool` (target) | Merges multiple judge runs for headline numbers |

Deep dives: [judge.md](./judge.md) (question flow and rubric navigation), [structured-output.md](./structured-output.md) (judge response schema), [README.md](../README.md) (output folder layout and resume semantics).

## Layer model

```text
CLI orchestrator (vera.py)
    ↓ subcommands delegate to
Domain packages (generate_conversations/, judge/)
    ↓ use
Infrastructure (llm_clients/)
    ↓ use
Shared utilities (utils/)
```

**Import rules:**

- Domain packages do not import each other.
- Infrastructure does not import domain packages.
- `utils/` is a leaf layer — it does not import domain or infrastructure packages.

**Supporting paths** (not in the import graph):

| Path | Role |
|------|------|
| `data/` | Committed evaluation inputs (personas, rubrics, prompts) |
| `output/` | Runtime artifacts (gitignored) |
| `scripts/` | Pipeline helpers until absorbed into `vera pool` |
| `tests/` | Permanent tests |
| `tmp_tests/` | Scratch experiments (not committed) |

## CLI surface

Exactly **one** root-level orchestrator: **`vera.py`**. Subcommands parse arguments and delegate to domain runners; they contain no business logic.

| Subcommand | Delegates to | Purpose |
|------------|--------------|---------|
| `vera generate` | `generate_conversations.runner` | Simulate conversations → `p_*` run folder |
| `vera judge` | `judge.runner` | Evaluate transcripts → `j_*` evaluation folder |
| `vera score` | `judge.score` | Aggregate `results.csv` → scores and visualizations |
| `vera pool` | `judge.pool` (target) | Merge multiple judge runs |
| `vera pipeline` | orchestration layer | Full workflow; passes paths between steps |

Recommended invocations:

```bash
uv run python vera.py pipeline --user-agent ... --provider-agent ...
uv run python vera.py generate -u ... -p ... -t 6 -r 1
uv run python vera.py judge -f output/p_... -j ...
uv run python vera.py score -r output/.../results.csv
uv run python vera.py pool path/to/j_* path/to/j_*
```

Resume semantics live on the relevant subcommand (`generate --resume`, `judge --resume`, `pipeline --resume-generate`, etc.). Flag names and folder layout are documented in [README.md](../README.md).

## Package responsibilities

| Package / path | Owns | Key modules |
|----------------|------|-------------|
| `generate_conversations/` | Simulation, turns, batch runner | `conversation_simulator.py`, `runner.py` |
| `judge/` | Rubric navigation, LLM judge, scoring | `question_navigator.py`, `llm_judge.py`, `score.py` |
| `llm_clients/` | Provider abstraction and registration | `llm_interface.py`, `llm_factory.py` |
| `utils/` | Cross-cutting types and I/O | `role.py`, `naming.py`, `conversation_layout.py` |

**Extension points:**

- New LLM provider → [evaluating.md](./evaluating.md)
- Structured judge responses → [structured-output.md](./structured-output.md)

Scoring and pooling are **libraries** invoked by `vera score` and `vera pool`, not separate root-level entry points.

## Data flow and artifacts

By default, generation writes under `output/` (or a user-specified parent):

```text
output/
└── p_<user>__a_<agent>__t<turns>__r<runs>__<timestamp>/
    ├── conversations/
    │   ├── *.txt
    │   └── logs/
    └── evaluations/
        └── j_<judge>__.../
            ├── results.csv
            ├── logs/
            └── scores/          ← created by vera score
```

Pooled runs (`j_pooled__.../`) sit alongside `p_*` folders when merging multiple judge outputs. Full naming rules, legacy flat folders, and `output/adhoc` behavior: [README.md](../README.md).

## Invariants

Agents and contributors must comply. Import boundaries are documented in the [Layer model](#layer-model) section above.

### MUST

- **Single CLI:** orchestration lives in `vera.py` only.
- **Subcommands:** `generate`, `judge`, `score`, `pool`, `pipeline` (add or remove only via [ESCALATE](#escalate-stop-and-ask)).
- **Generation:** conversation simulation logic stays in `generate_conversations/`.
- **Judging:** rubric scoring, navigation, and evaluation output stay in `judge/`.
- **LLM providers:** new providers implement [llm_clients/llm_interface.py](../llm_clients/llm_interface.py) and register in [llm_clients/llm_factory.py](../llm_clients/llm_factory.py).
- **Shared types:** cross-layer enums (e.g. `Role`) live in `utils/` — not duplicated in domain packages.
- **Data:** committed evaluation inputs in `data/`; runtime artifacts in `output/` (gitignored).
- **Tests:** permanent tests in `tests/`; one-off experiments in `tmp_tests/` (not committed).
- **Dependencies:** add packages via `uv add` / `uv add --dev`; update lockfile in the same change.

### MUST NOT

- Import `judge/` from `generate_conversations/` or vice versa.
- Import `judge/` or `generate_conversations/` from `llm_clients/`.
- Import `judge/`, `generate_conversations/`, or `llm_clients/` from `utils/`.
- Add root-level Python scripts (including keeping `generate.py`, `judge.py`, or `run_pipeline.py` as entry points after migration).
- Put domain logic in `vera.py` — keep the CLI layer thin.
- Commit generated output under `output/` or secrets in `.env`.
- Bypass architecture checks (pyright, required CI) to merge structural changes.

### ESCALATE (stop and ask)

Stop work and request maintainer approval before proceeding when a task would:

- Add a new top-level package or move code between `judge/` and `generate_conversations/`.
- Change import boundaries documented in this file.
- Add a new runtime dependency or raise minimum Python version.
- Change judge rubric/scoring contracts, pipeline output layout, or CLI flags affecting run folders.
- Add or remove a `vera.py` subcommand.
- Refactor across multiple domain packages in one change without maintainer review.

For large multi-file features (new judge dimensions, pipeline CLI changes), consider an [OpenSpec](https://github.com/Fission-AI/OpenSpec) change under `openspec/changes/` if the team adopts that workflow — see [AGENTS.md](../AGENTS.md).

## Enforcement

Target state for automated checks:

| Mechanism | What it checks |
|-----------|----------------|
| `uv run pyright` | Type checking (basic mode) |
| Pre-commit | Ruff format/lint |
| CI | Ruff, pyright, `pytest -m "not live"` |
| `.github/CODEOWNERS` | Human review on `vera.py`, import boundaries, domain packages |

Run before pushing structural changes:

```bash
uv run pyright
uv run pytest -m "not live"
```

## Migration from current layout

Implementation may not match the target yet. Known gaps:

| Target | Current | Migration |
|--------|---------|-----------|
| `vera.py` | `generate.py`, `judge.py`, `run_pipeline.py` | Extract argparse/`main()` into subcommand modules; wire `vera.py`; deprecate old scripts; delete after one release |
| `vera score` | `python -m judge.score` | Remove standalone `__main__` from `judge/score.py` once `vera score` exists |
| `vera pool` | `scripts/pool_vera_scores.py` | Move logic into `judge/`; remove or thin the script |
| `.github/CODEOWNERS` | Not present | Add file or drop reference from enforcement table |

Legacy scripts should print a deprecation message pointing at the equivalent `vera.py` subcommand until removed.

## Changing this architecture

To add an exception or new boundary:

1. Update this document with rationale.
2. Update [AGENTS.md](../AGENTS.md) if agent stop/escalate rules change.
3. Update [README.md](../README.md) if CLI flags or output layout change.
