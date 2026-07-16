# VERA-MH Architecture

Validation of Ethical and Responsible AI in Mental Health: simulate mental-health conversations, evaluate them against one or more clinical rubrics, and aggregate scores for each rubric for comparison across chatbots under test.

This document describes the **target architecture**. Implementation may lag; see [Migration from current layout](#migration-from-current-layout) for known gaps. [README.md](../README.md) covers setup and CLI usage; [vera-cli-use-cases.md](./vera-cli-use-cases.md) covers the CLI/config surface in detail; this doc defines structure, data flow, and what **must** hold.

## Entity vocabulary

Three entities, each with a single-letter prefix used throughout the CLI, config, and file naming (see [vera-cli-use-cases.md](./vera-cli-use-cases.md)):

- **`u` — user**: the persona-side LLM simulating the user.
- **`c` — chatbot**: the provider/agent LLM under test.
- **`j` — judge**: the LLM evaluating a transcript against a rubric.

## System overview

Two independent pipelines share infrastructure only:

- **Generation** — user LLM ↔ chatbot LLM → transcript files
- **Judging** — judge LLM walks rubric questions → per-dimension severity → scores

They never import each other. A full workflow runs generation, judging, scoring, and optional pooling in sequence.

All user-facing operations go through **`vera.py`** subcommands. Domain packages are libraries; they are not invoked directly as scripts.

```text
data/personas.tsv ──► generate ──► c_<chatbot>/<run>/conversations/*.json
                                        │
                                        ▼
                                   judge ──► c_<chatbot>/<run>/evaluations/<rubric>/j_*/results.csv
                                        │
                                        ▼
                                   score ──► .../j_*/scores/
                                        │
                                        ▼
                                   pool ──► pooled result across evaluation folders
```

## Domain model

| Concept | Location | Notes |
|---------|----------|-------|
| Persona | one or more persona files | Simulated user; drives the user-side (`u`) LLM. Duplicate persona names across files are possible — disambiguated by file + name. |
| Transcript | `c_<chatbot>/<run>/conversations/*.json` | Turn-by-turn chat log; filename encodes persona file + name + chatbot model |
| Rubric | rubric files | Question flow, dimensions, severity. Rubric-derived data files (dimension names, rating values) live outside `src/`, accessible to all packages. |
| Evaluation run | `j_<judge>_<timestamp>_<sha>/` | TSV results, logs, metadata; flat per-run, not nested under a persistent per-judge-model parent |
| Dimension score | `scoring/score.py` | Aggregated from rubric answers |
| Pooled scores | `scoring/pool.py` | Concatenates multiple evaluation folders (`vera pool`) |

Deep dives: [judge.md](./judge.md) (question flow and rubric navigation), [structured-output.md](./structured-output.md) (judge response schema), [vera-cli-use-cases.md](./vera-cli-use-cases.md) (CLI/config surface, naming scheme in full).

## Layer model

```text
CLI orchestrator (vera.py) — thin, no business logic
    ↓ delegates to
Domain packages (generate_conversations/, judge/, scoring/)
    ↓ register handlers with
Workers (workers/) — queue protocol, worker pool, job dispatch
    ↓ used by domain handlers
Infrastructure (llm_clients/)
    ↓ used by all above
Shared utilities (utils/) — leaf layer
```

**Import rules:**

- Domain packages (`generate_conversations/`, `judge/`, `scoring/`) do not import each other.
- `workers/` does not import domain packages — domain registers handlers upward (inversion of control), never the reverse.
- `llm_clients/` does not import domain packages or `workers/`.
- `utils/` is a leaf layer — it does not import domain, `workers/`, or `llm_clients/` packages.
- `Role` is defined once, in `utils/role.py` — no package defines its own copy.
- All folder/file naming and layout logic (the `c_`/`u_`/`j_` scheme, timestamp/sha composition, persistent-vs-flat nesting) lives in a single `utils/` module (extending `utils/conversation_layout.py`), never duplicated across handlers — this scheme changes over time, and centralizing it means a revision touches one file, not every caller.

**Supporting paths** (not in the import graph):

| Path | Role |
|------|------|
| `data/` | Committed evaluation inputs (personas, rubrics, prompts) |
| `output/` | Runtime artifacts (gitignored) |
| `scripts/` | Pipeline helpers, including `distribute_files.py` |
| `tests/` | Permanent tests |
| `tmp_tests/` | Scratch experiments (not committed) |

## CLI surface

Exactly **one** root-level orchestrator: **`vera.py`**. Subcommands parse arguments and delegate to domain runners; they contain no business logic. Full flag/config reference: [vera-cli-use-cases.md](./vera-cli-use-cases.md).

| Subcommand | Delegates to | Purpose |
|------------|--------------|---------|
| `vera generate` | `generate_conversations.runner` | Simulate conversations → `c_<chatbot>/<run>/conversations/` |
| `vera judge` | `judge.runner` | Evaluate transcripts → `evaluations/<rubric>/j_*` |
| `vera score` | `scoring.score` | Aggregate `results.csv` → scores and visualizations |
| `vera pool` | `scoring.pool` | Concatenate multiple evaluation folders into one pooled result |
| `vera pipeline` | orchestration layer | Full workflow for one chatbot; passes paths between steps |
| `vera resume` | orchestration layer | Reads `config.json` (sha-verified) + `state.json`, continues an incomplete run |

Config and CLI flags are strictly either/or, never combined for the same run — see [vera-cli-use-cases.md](./vera-cli-use-cases.md#config-mechanism). `-u`/`-j` shorthand selects models/repeats for the user/judge side respectively; bespoke sampling knobs are config-only.

```bash
uv run python vera.py pipeline --config run.json
uv run python vera.py generate -u gpt:1 sonnet:2
uv run python vera.py judge -j claude:1 --conversations output/c_sonnet/<run>/conversations/
uv run python vera.py score -r output/.../results.csv
uv run python vera.py pool --evaluations path/to/evaluations/... path/to/evaluations/...
uv run python vera.py resume --config output/c_sonnet/<run>/config.json
```

## Package responsibilities

| Package / path | Owns | Key modules |
|----------------|------|-------------|
| `generate_conversations/` | Simulation, turns, batch runner (pure core; handler owns I/O) | `conversation_simulator.py`, `runner.py` |
| `judge/` | Rubric navigation, LLM judge (pure core; handler owns I/O) | `question_navigator.py`, `llm_judge.py` |
| `scoring/` | Aggregation, visualization, pooling — split out of `judge/` | `score.py`, `score_viz.py`, `pool.py` |
| `workers/` | Shared queue protocol, worker pool, job dispatch | `queue.py`, `job_context.py` |
| `llm_clients/` | Provider plugin registry; providers self-register, factory resolves by prefix | `llm_interface.py`, `llm_factory.py` |
| `utils/` | Cross-cutting types, naming/layout, I/O helpers | `role.py`, `naming.py`, `conversation_layout.py` |

**Extension points:**

- New LLM provider → [evaluating.md](./evaluating.md)
- Structured judge responses → [structured-output.md](./structured-output.md)

## Data flow and artifacts

By default, generation writes under `output/` (or a user-specified parent). Full naming rationale: [vera-cli-use-cases.md](./vera-cli-use-cases.md#naming).

```text
output/
└── c_sonnet/                                       ← persistent per-chatbot directory
    └── <timestamp>_<sha>/
        ├── conversations/
        │   └── u_<persona-file>_<persona-name>_c_sonnet.json
        └── evaluations/
            └── <rubric_name>/
                └── j_claude_<timestamp>_<sha>/      ← judge stays flat, no persistent parent
                    ├── config.json
                    ├── config.json.sha256
                    ├── state.json
                    ├── results.csv
                    └── scores/                       ← created by vera score

output/evaluations/<config-sha256>/<rubric_name>/    ← standalone judging (vera judge run on its own)
```

Every run-id is `<model>_<timestamp>_<sha256-of-config.json>`. `config.json` is an immutable copy of the resolved config, hash-verified via its sidecar; `state.json` is the separate, mutable file that tracks resume progress.

## Invariants

Agents and contributors must comply. Import boundaries are documented in the [Layer model](#layer-model) section above.

### MUST

- **Single CLI:** orchestration lives in `vera.py` only.
- **Subcommands:** `generate`, `judge`, `score`, `pool`, `pipeline`, `resume` (add or remove only via [ESCALATE](#escalate-stop-and-ask)).
- **Generation:** conversation simulation logic stays in `generate_conversations/`; the simulator core is pure (no filesystem, no logging) — the handler owns all I/O.
- **Judging:** rubric navigation and LLM-judge logic stay in `judge/`, also pure-core-plus-handler. Judge never auto-scores — `vera score`/`vera pool` are separate subcommands.
- **Scoring:** aggregation, visualization, and pooling stay in `scoring/`, never re-absorbed into `judge/`.
- **Config vs CLI:** `--config` and CLI flags are strictly either/or for a given run — never combined. `generation` and `judging` blocks in `config.json` are completely orthogonal; model selection for one must never influence the other.
- **Naming/layout:** all folder/file naming logic (the `c_`/`u_`/`j_` scheme) lives in one `utils/` module — never duplicated across handlers.
- **Traceability:** every run writes an immutable `config.json` (+ `.sha256` sidecar) and a separate, mutable `state.json`. `state.json` records both the requested and actual-resolved model identifier.
- **LLM providers:** new providers implement [llm_clients/llm_interface.py](../llm_clients/llm_interface.py) and register in [llm_clients/llm_factory.py](../llm_clients/llm_factory.py). `models[].name` in config is always a specific model identifier, never a bare provider name.
- **Shared types:** cross-layer enums (e.g. `Role`) live in `utils/` — not duplicated in domain packages.
- **Data:** committed evaluation inputs in `data/`; runtime artifacts in `output/` (gitignored).
- **Tests:** permanent tests in `tests/`; one-off experiments in `tmp_tests/` (not committed).
- **Dependencies:** add packages via `uv add` / `uv add --dev`; update lockfile in the same change.

### MUST NOT

- Import between `generate_conversations/`, `judge/`, and `scoring/`.
- Import domain packages from `workers/` — domain registers handlers upward, never the reverse.
- Import domain packages or `workers/` from `llm_clients/`.
- Import domain packages, `workers/`, or `llm_clients/` from `utils/`.
- Add root-level Python scripts (including keeping `generate.py`, `judge.py`, or `run_pipeline.py` as entry points after migration).
- Put domain logic in `vera.py` — keep the CLI layer thin.
- Commit generated output under `output/` or secrets in `.env`.
- Overwrite an existing run's output folder silently — collision on an already-existing folder errors out (no overwrite, no auto-suffix).
- Bypass architecture checks (pyright, required CI) to merge structural changes.

### ESCALATE (stop and ask)

Stop work and request maintainer approval before proceeding when a task would:

- Add a new top-level package or move code between `generate_conversations/`, `judge/`, or `scoring/`.
- Change import boundaries documented in this file.
- Add a new runtime dependency or raise minimum Python version.
- Change judge rubric/scoring contracts, pipeline output layout, naming scheme, or CLI flags affecting run folders.
- Add or remove a `vera.py` subcommand.
- Refactor across multiple domain packages in one change without maintainer review.

For large multi-file features (new judge dimensions, pipeline CLI changes), consider an [OpenSpec](https://github.com/Fission-AI/OpenSpec) change under `openspec/changes/` if the team adopts that workflow — see [AGENTS.md](../AGENTS.md).

## Enforcement

Target state for automated checks:

| Mechanism | What it checks |
|-----------|----------------|
| `uv run pyright` | Type checking — blocking (not continue-on-error) as of migration Phase 3 |
| Pre-commit | Ruff format/lint |
| CI | Ruff, pyright, `pytest -m "not live"` — coverage gate raised 30% → 60% as of migration Phase 3 |
| import-linter + grimp | Declarative layer contracts (`pyproject.toml`) + custom import-graph assertions — added in migration Phase 3 |
| `.github/CODEOWNERS` | Human review on `vera.py`, import boundaries, domain packages |

Run before pushing structural changes:

```bash
uv run pyright
uv run pytest -m "not live"
```

## Migration from current layout

4 phases, scoped by risk rather than by package. Multi-rubric support gets its own phase, separate from the scoring split — moving code (Phase 2) is mechanical and behavior-preserving, while multi-rubric support (Phase 3) is genuinely new behavior:

| Phase | Goal | Key changes |
|-------|------|--------------|
| **1 — Cosmetic** | `vera.py` exists, nothing else changes | Thin wrapper calling the existing `generate.py`/`judge.py`/`run_pipeline.py` as-is; no internal refactor, no behavior change |
| **2 — Scoring split** | Extract `scoring/` | `score.py`/`score_viz.py`/`pool.py` move out of `judge/` into `scoring/`; pure move, no new behavior |
| **3 — Multi-rubric support** | Support multiple rubrics per run | `config.json`'s `judging.rubrics[]` schema implemented; per-rubric judge-model overrides; per-rubric `evaluations/<rubric>/` folder separation built |
| **4 — Substantial refactoring** | Everything else | `workers/` unification (both `generate_conversations/runner.py` and `judge/runner.py` already hand-roll the same asyncio-queue worker-pool pattern independently — concrete evidence for this step); `llm_clients/` plugin-registry formalization; import-linter/grimp enforcement; quality-gate tightening (pyright blocking, coverage 30%→60%); root-clutter cleanup (see below); `u`/`c`/`j` naming scheme and `config.json`/`state.json` traceability model implemented |

**Root-clutter disposition** (Phase 4 cleanup): `logging/`, `logs/`, `conversations/` (root), `human_validation/` — delete (legacy/duplicate generated dumps, already gitignored, superseded by the target `output/c_*` layout). `score_comparisons/` — keep the directory but stop committing generated CSV/PNG output; moves under `scoring/`. `spring_scripts/`, `openspec/`, `alba.md`, `Untitled.ipynb` — untracked scratch, outside architecture scope. `distribute_files.py` — move into `scripts/`. `docker-compose.yml` volume mounts — fix stale references (`./evaluations` doesn't exist, `./logs` is omitted) to match the target layout.

**Open item, not yet designed:** different personas may need different system prompts, so persona definitions should be linked to their own system prompt rather than assuming one global prompt for all personas. Exact mechanism (a `system_prompt` field on the persona object, a `system_prompt_file` pointer, or something else) is undecided — revisit when the persona schema itself gets formalized (likely Phase 3 or 4).

Legacy scripts should print a deprecation message pointing at the equivalent `vera.py` subcommand until removed.

## Changing this architecture

To add an exception or new boundary:

1. Update this document with rationale.
2. Update [AGENTS.md](../AGENTS.md) if agent stop/escalate rules change.
3. Update [README.md](../README.md) if CLI flags or output layout change.
