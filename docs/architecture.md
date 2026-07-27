# VERA-MH Architecture

Validation of Ethical and Responsible AI in Mental Health: simulate mental-health conversations, evaluate them against one or more clinical rubrics, and aggregate scores for each rubric for comparison across chatbots under test.

This document describes the **target architecture**. Implementation may lag; see [Migration from current layout](#migration-from-current-layout) for known gaps. [README.md](../README.md) covers setup and CLI usage; [vera-cli-use-cases.md](./vera-cli-use-cases.md) covers the CLI/config surface in detail; this doc defines structure, data flow, and what **must** hold. [ARCHITECTURE-SPINE.md](./ARCHITECTURE-SPINE.md) is the terse, numbered invariants-only contract this doc is derived from — cite an `AD-n` from there when a change needs to reference a specific rule.

## Entity vocabulary

Three entities (`u`/`c`/`j`) run through the CLI, config, and file naming — full definitions in [vera-cli-use-cases.md#entity-vocabulary](./vera-cli-use-cases.md#entity-vocabulary) (canonical).

## System overview

Two independent pipelines share infrastructure only:

- **Generation** — user LLM ↔ chatbot LLM → transcript files
- **Judging** — judge LLM walks rubric questions → per-dimension severity → scores

They never import each other. A full workflow runs generation, judging, score, and optional pooling in sequence.

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
| Chatbot | `-c`/`generation.chatbot` | Provider/agent LLM under test (the `c` in `c_<chatbot>/`); selected the same way as the user-side (`u`) and judge-side (`j`) models, never inferred from context |
| Transcript | `c_<chatbot>/<run>/conversations/*.json` | Turn-by-turn chat log; filename encodes persona file + name + chatbot model |
| Rubric | rubric files | Question flow, dimensions, severity. Rubric-derived data files (dimension names, rating values) live outside `src/`, accessible to all packages. |
| Evaluation run | `j_<judge>_<timestamp>_<sha>/` | TSV results, logs, metadata; flat per-run, not nested under a persistent per-judge-model parent |
| Dimension score | `score/score.py` | Aggregated from rubric answers |
| Pooled scores | `score/pool.py` | Concatenates multiple evaluation folders (`vera pool`) |

Deep dives: [judge.md](./judge.md) (question flow and rubric navigation), [structured-output.md](./structured-output.md) (judge response schema), [vera-cli-use-cases.md](./vera-cli-use-cases.md) (CLI/config surface, naming scheme in full).

## Layer model

```text
CLI orchestrator (vera.py) — thin, no business logic
    ↓ delegates to
Domain packages (generate/, judge/, score/)
    ↓ register handlers with
Workers (workers/) — queue protocol, worker pool, job dispatch
    ↓ used by domain handlers
Infrastructure (llm_clients/, storage/)
    ↓ used by all above
Shared utilities (utils/) — leaf layer
```

**Import rules:**

- Domain packages (`generate/`, `judge/`, `score/`) do not import each other.
- `workers/` does not import domain packages — domain registers handlers upward (inversion of control), never the reverse.
- `llm_clients/` and `storage/` do not import domain packages or `workers/`.
- `utils/` is a leaf layer — it does not import domain, `workers/`, `llm_clients/`, or `storage/` packages.
- `Role` is defined once, in `utils/role.py` — no package defines its own copy.
- All folder/file naming and layout logic (the `c_`/`u_`/`j_` scheme, timestamp/sha composition, persistent-vs-flat nesting) lives in `utils/naming.py`, with `utils/conversation_layout.py` building on it (never the reverse) — never duplicated across handlers. This scheme changes over time, and centralizing it means a revision touches one file, not every caller. `utils/naming.py` builds keys/paths only; it never persists bytes — that's `storage/`'s job.
- Interfaces live with their implementations, not in a separate cross-cutting package: `llm_clients/` holds `LLMInterface` plus every provider, `workers/` holds `QueueProtocol` plus `LocalQueue`/`SQSQueue`, `storage/` holds `StorageBackend` plus `LocalFilesystemStorage`/`S3Storage` — grouped by concern (Common Closure Principle), not by "abstract vs. concrete."

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
| `vera generate` | `generate.runner` | Simulate conversations → `c_<chatbot>/<run>/conversations/` |
| `vera judge` | `judge.runner` | Evaluate transcripts → `evaluations/<rubric>/j_*` |
| `vera score` | `score.score` | Aggregate `results.csv` → scores and visualizations |
| `vera pool` | `score.pool` | Concatenate multiple evaluation folders into one pooled result |
| `vera pipeline` | orchestration layer | Full workflow for one chatbot; passes paths between steps |
| `vera resume` | orchestration layer | Reads `config.json` (sha-verified) + `state.json`, continues an incomplete run |

Config and CLI flags are strictly either/or, never combined for the same run — see [vera-cli-use-cases.md](./vera-cli-use-cases.md#config-mechanism). `-c` selects the chatbot under test; `-u`/`-j` shorthand selects models/repeats for the user/judge side respectively; bespoke sampling knobs are config-only. `--rubric` selects the rubric-bundle manifest (see [Rubric bundle manifest](#rubric-bundle-manifest) below) — always a list, though only a length-1 list is supported until Phase 4. `-c` is required for `generate`/`pipeline` whenever `--config` isn't used — there is no default chatbot.

```bash
uv run python vera.py pipeline --config run.json
uv run python vera.py generate -c sonnet -u gpt:1
uv run python vera.py judge -j claude:1 --rubric data/si_rubric.json --conversations output/c_sonnet/<run>/conversations/
uv run python vera.py score -r output/.../results.csv
uv run python vera.py pool --evaluations path/to/evaluations/... path/to/evaluations/...
uv run python vera.py resume --config output/c_sonnet/<run>/config.json
```

`vera judge` never takes `-c`: judging is decoupled from chatbot selection by design (the `generation`/`judging` orthogonality invariant, below) — the chatbot is already implicit in whichever `--conversations` folder is passed in.

### Rubric bundle manifest

A rubric is a self-describing bundle, not a bare `.tsv` path with assumed sibling filenames. `--rubric`/`judging.rubrics[]` entries point at a manifest file:

```json
{
  "rubric_file": "rubric.tsv",
  "rubric_prompt_beginning_file": "rubric_prompt_beginning.txt",
  "question_prompt_file": "question_prompt.txt",
  "personas": ["data/personas.tsv"]
}
```

Paths are relative to the manifest's own folder — distinct from `config.json`, whose paths resolve relative to `$ROOT` (the directory containing `vera.py`), never to the manifest, the config file's own location, or the CLI's working directory (see [Config mechanism](./vera-cli-use-cases.md#config-mechanism)). `personas` is **informational only** — it documents which personas this rubric is intended/validated for, for humans and tooling to discover; it does not make generation consume it automatically. Generation still chooses personas independently (the `generation`/`judging` orthogonality invariant holds). This manifest shape is exactly what a `judging.rubrics[]` config entry looks like once Phase 3 formalizes the schema — the format isn't thrown away when the CLI is replaced, it's the design.

**Example — two path fields, two different anchors:**

```text
project-root/                    ← $ROOT (contains vera.py)
├── vera.py
├── configs/
│   └── run.json                 ← config.json
└── data/
    ├── personas_a.json
    └── si_rubric_bundle.json    ← manifest
        (personas: ["personas/si_personas.tsv"])
```

- `configs/run.json`'s `generation.personas: ["data/personas_a.json"]` always means `project-root/data/personas_a.json` — resolved against `$ROOT`, no matter where you run `vera.py` from or where `run.json` itself lives.
- `data/si_rubric_bundle.json`'s `personas: ["personas/si_personas.tsv"]` always means `data/personas/si_personas.tsv` — resolved against the manifest's own folder (`data/`), so the whole `data/` folder stays portable if copied into a different checkout, independent of `$ROOT`.

Same shape of field, same-looking relative string, two different rules — hence calling both out explicitly here rather than leaving it implicit.

**Separation of concerns, since these two now overlap in subject matter:** the manifest describes what a rubric **is** — its content, files, and intended personas — and changes rarely. `config.json`'s `judging.rubrics[].models` describes how to **run** it for a given invocation — which judge models, repeats, per-rubric overrides — and changes every run. Judge-model defaults belong in `config.json`, never in the manifest; the manifest never carries execution knobs.

**`--target <name>` shorthand:** the manifest's `personas` field stays informational-only for every invocation shape *except one*. `vera pipeline --target <name>` resolves `<name>` to one rubric bundle manifest and expands to setting **both** `generation.personas` and `judging.rubrics` from it in a single shot — the manifest's `personas` field becomes the actual, authoritative generation input only for this shorthand. Every other invocation (`--rubric` plus independently-specified generation personas, or `--config` with both blocks set explicitly) keeps `generation`/`judging` fully orthogonal — `--target` is one deliberate, named exception, not a general weakening of that invariant.

## Package responsibilities

| Package / path | Owns | Key modules |
|----------------|------|-------------|
| `generate/` | Simulation, turns, batch runner (pure core; handler owns I/O) | `conversation_simulator.py`, `runner.py` |
| `judge/` | Rubric navigation, LLM judge (pure core; handler owns I/O) | `question_navigator.py`, `llm_judge.py` |
| `score/` | Aggregation, visualization, pooling — split out of `judge/` | `score.py`, `score_viz.py`, `pool.py` |
| `workers/` | Shared queue protocol, worker pool, job dispatch | `queue.py`, `job_context.py` |
| `llm_clients/` | Provider plugin registry; providers self-register, factory resolves by prefix | `llm_interface.py`, `llm_factory.py` |
| `storage/` | Storage backend abstraction; raw bytes+keys, knows nothing about run semantics | `storage_backend.py`, `local_filesystem_storage.py` |
| `utils/` | Cross-cutting types, naming/layout, I/O helpers | `role.py`, `naming.py`, `conversation_layout.py` |

**Extension points:**

- New LLM provider → [evaluating.md](./evaluating.md)
- Structured judge responses → [structured-output.md](./structured-output.md)
- New storage backend (e.g. S3) → implement `storage/storage_backend.py`'s `StorageBackend` (`write(key, bytes)` / `read(key)` / `exists(key)`)

## Data flow and artifacts

By default, generation writes under `output/` (or a user-specified parent). Full naming rationale: [vera-cli-use-cases.md](./vera-cli-use-cases.md#naming).

```text
output/
└── c_sonnet/                                              ← persistent per-chatbot directory
    └── <nickname>_<timestamp>_<sha>/                      ← e.g. prophetic-bullfrog_20260713-1530_a1b2c3
        ├── conversations/
        │   └── u_<persona-file>_<persona-name>_c_sonnet.json
        └── evaluations/
            └── <rubric_name>/
                └── j_claude_<nickname>_<timestamp>_<sha>/ ← judge stays flat, no persistent parent
                    ├── config.json
                    ├── config.json.sha256
                    ├── state.json
                    ├── results.csv
                    └── scores/                            ← created by vera score

output/evaluations/<config-sha256>/<rubric_name>/          ← standalone judging (vera judge run on its own)
```

Every run-id is `<nickname>_<timestamp>_<sha256-of-config.json>` when nested under a per-model parent (`c_sonnet/`, where the model is already given by the parent folder), or `<model>_<nickname>_<timestamp>_<sha>` when flat (`j_claude_.../`, where nothing else identifies the model). `<nickname>` is a generated human-memorable tag (e.g. via a word-pair generator), purely for a person to recognize a run at a glance — it carries no identifying information itself and is never a substitute for the sha; it never needs to encode which model was used, since the surrounding path already does that job. `config.json` is an immutable copy of the resolved config, hash-verified via its sidecar; `state.json` is the separate, mutable file that tracks resume progress.

**Single canonical hash, not two independent computations:** the sha256 is computed exactly once, by exactly one function, and that single value is what both the run-id folder name's `<sha>` component and the `config.json.sha256` sidecar's content contain — never two independently-computed values that could drift apart. Rejected: embedding the hash in `config.json`'s own filename (e.g. `config.<sha>.json`) — doesn't remove the need for a verification step (a corrupted file wouldn't automatically stop matching its own filename; verification still requires hashing the actual bytes, which is the sidecar's job), and would put the same value in a third place with no added integrity benefit.

## Invariants

Agents and contributors must comply. Import boundaries are documented in the [Layer model](#layer-model) section above.

### MUST

- **Single CLI:** orchestration lives in `vera.py` only.
- **Subcommands:** `generate`, `judge`, `score`, `pool`, `pipeline`, `resume` (add or remove only via [ESCALATE](#escalate-stop-and-ask)).
- **Generation:** conversation simulation logic stays in `generate/`; the simulator core is pure (no filesystem, no logging) — the handler owns all I/O.
- **Judging:** rubric navigation and LLM-judge logic stay in `judge/`, also pure-core-plus-handler. Judge never auto-scores — `vera score`/`vera pool` are separate subcommands. **Rubric navigation logic lives in code, never in the prompt:** which question is asked next given an answer is determined entirely by `QuestionNavigator` walking `question_flow_data` parsed from the rubric TSV — the judge LLM answers/judges the current question only, and is never asked to decide or influence what comes next.
- **Scoring:** aggregation, visualization, and pooling stay in `score/`, never re-absorbed into `judge/`.
- **Config vs CLI:** `--config` and CLI flags are strictly either/or for a given run — never combined. `generation` and `judging` blocks in `config.json` are completely orthogonal; model selection for one must never influence the other.
- **Naming/layout:** all folder/file naming logic (the `c_`/`u_`/`j_` scheme) lives in one `utils/` module — never duplicated across handlers.
- **Traceability:** every run writes an immutable `config.json` (+ `.sha256` sidecar) and a separate, mutable `state.json`. `state.json` records both the requested and actual-resolved model identifier.
- **LLM providers:** new providers implement [llm_clients/llm_interface.py](../llm_clients/llm_interface.py) and register in [llm_clients/llm_factory.py](../llm_clients/llm_factory.py). `models[].name` in config is always a specific model identifier, never a bare provider name.
- **Shared types:** cross-layer enums (e.g. `Role`) live in `utils/` — not duplicated in domain packages.
- **Data:** committed evaluation inputs in `data/`; runtime artifacts in `output/` (gitignored). Rubric and persona content (dimensions, question flows, prompt text, persona definitions) MUST live in `data/`, never embedded in code — VERA-MH must be usable by non-developers who add or edit rubrics and personas without touching Python. No domain package may hardcode rubric/persona content as an alternative to reading it from `data/`.
- **Tests:** permanent tests in `tests/`; one-off experiments in `tmp_tests/` (not committed).
- **Dependencies:** add packages via `uv add` / `uv add --dev`; update lockfile in the same change.

### MUST NOT

- Import between `generate/`, `judge/`, and `score/`.
- Import domain packages from `workers/` — domain registers handlers upward, never the reverse.
- Import domain packages or `workers/` from `llm_clients/` or `storage/`.
- Import domain packages, `workers/`, `llm_clients/`, or `storage/` from `utils/`.
- Add root-level Python scripts (including keeping `generate.py`, `judge.py`, or `run_pipeline.py` as entry points after migration).
- Put domain logic in `vera.py` — keep the CLI layer thin.
- Commit generated output under `output/` or secrets in `.env`.
- Overwrite an existing run's output folder silently — collision on an already-existing folder errors out (no overwrite, no auto-suffix).
- Bypass architecture checks (pyright, required CI) to merge structural changes.

### Stable interfaces (agent-coding optimization)

This codebase is optimized for agent coding: most files are safe for an agent to change freely within a package's own boundaries, but a small set of **stable interfaces** are rarely meant to change and require a design doc before modification, not just a PR. These are called out individually in [`.github/CODEOWNERS`](../.github/CODEOWNERS) (not just covered by their package's blanket rule) so their significance is visible at a glance:

| File | What it stabilizes |
|------|---------------------|
| `llm_clients/llm_interface.py` | `LLMInterface` ABC (Python's `abc.ABC` — Abstract Base Class) — every provider implements this |
| `workers/queue.py` | `QueueProtocol` ABC — `LocalQueue`/future `SQSQueue` implement this |
| `utils/role.py` | `Role` — the single shared definition across all packages |
| `utils/naming.py` | The naming/layout module — single source of truth for run-id and folder-naming logic |
| `utils/config_schema.py` | `config.json` schema — the contract every subcommand's `--config` resolves against |
| `storage/storage_backend.py` | `StorageBackend` ABC — `LocalFilesystemStorage`/future `S3Storage` implement this |

A change to any of these is an [ESCALATE](#escalate-stop-and-ask) case: write a short design doc (what's changing, why, what it breaks) before opening the PR.

**What "enforced" means concretely, not just documented convention:** two mechanisms, one required-review and one required-evidence, not one or the other —

- **CODEOWNERS requires review** from a maintainers team on every file in the table above (and on `docs/architecture.md`/`docs/vera-cli-use-cases.md` themselves), so a PR touching one can't merge without a maintainer's approval, full stop.
- **A CI check requires the design doc itself, not just a reviewer's say-so:** on `pull_request`, if the diff touches any stable-interface file, the check greps the PR description for a link matching the design-doc convention and fails (as a required, merge-blocking status check) if none is found. This exists specifically so "write a design doc first" can't be quietly skipped when a PR is small or the reviewer is moving fast — the requirement is checked by CI, not left to a human to remember to ask for.

Both need the underlying maintainers team and CI workflow to actually exist and be wired into branch protection before either is a real gate rather than a documented intention.

`utils/role.py`'s *members* are expected to be renamed to track the `u`/`c`/`j` vocabulary (e.g. `PROVIDER` → `CHATBOT`, per [vera-cli-use-cases.md#entity-vocabulary](./vera-cli-use-cases.md#entity-vocabulary)) once the file is committed to this repo — that rename is a normal PR, not an ESCALATE case. Only removing or repurposing the `Role` type itself needs a design doc.

### ESCALATE (stop and ask)

Stop work and request maintainer approval before proceeding when a task would:

- Add a new top-level package or move code between `generate/`, `judge/`, or `score/`.
- Change import boundaries documented in this file.
- Change any [stable interface](#stable-interfaces-agent-coding-optimization) — requires a design doc first.
- Add a new runtime dependency or raise minimum Python version.
- Change judge rubric/score contracts, pipeline output layout, naming scheme, or CLI flags affecting run folders.
- Add or remove a `vera.py` subcommand.
- Refactor across multiple domain packages in one change without maintainer review.

**Documenting a phase in this architecture doc's migration plan does not pre-clear its ESCALATE requirements.** Every phase — even one that matches this plan exactly — still needs fresh maintainer sign-off before starting, not just at the planning stage. This is deliberate: it ensures a human is actually looking at the moment of highest-risk change (e.g. Phase 5's `workers/` rewrite), not just at whenever this doc was written.

For large multi-file features (new judge dimensions, pipeline CLI changes), an [OpenSpec](https://github.com/Fission-AI/OpenSpec) change under `openspec/changes/` is required once Phase O adopts that workflow — see [AGENTS.md](../AGENTS.md) and Phase O below. Until Phase O lands, `openspec/` stays empty scaffolding, not an active practice.

## Enforcement

Target state for automated checks:

| Mechanism | What it checks |
|-----------|----------------|
| `uv run pyright` | Type checking — blocking (not continue-on-error) as of migration Phase 5 |
| Pre-commit | Ruff format/lint |
| CI | Ruff, pyright, `pytest -m "not live"` — coverage gate raised 30% → 60% as of migration Phase 5 |
| import-linter | Declarative layer contracts (`pyproject.toml`) — added incrementally as each new boundary is created (Phase 2: `judge/` ⊥ `score/`; Phase 3: `utils/` leaf), completed with the full contract (all [Layer model](#layer-model) boundaries) in Phase 5 |
| grimp | Custom import-graph assertions — added in migration Phase 5 |
| `.github/CODEOWNERS` | Human review on `vera.py`, import boundaries, domain packages |
| CI (design-doc gate) | Required status check on `pull_request`: fails if the diff touches a [stable interface](#stable-interfaces-agent-coding-optimization) and the PR description has no design-doc link — not yet built |

Run before pushing structural changes:

```bash
uv run pyright
uv run pytest -m "not live"
```

## Migration from current layout

6 phases, scoped by risk and by dependency order — each phase only builds on artifacts that already exist by the time it runs. Every phase carries an explicit **Done when** bar; a phase isn't complete because its bullet list of changes landed, it's complete when that bar is met. **Every phase's Done-when implicitly includes updating `README.md`, `AGENTS.md`, and any other doc that phase's changes make stale** — stated once here rather than repeated in every row. `judging.rubrics` is a **list from day one, at every phase** starting with Phase 0 — only a length-1 list is supported/validated until Phase 4; this is the first step toward multi-rubric, not a hidden permanent limitation. **Phase S (storage abstraction) and Phase O (OpenSpec adoption) are both orthogonal to 0-5** — neither depends on, or is depended on by, the numbered phases or each other — so unlike the rest of the table, their position is a suggested default slot, not a strict dependency order; pull either earlier or run them in parallel with any other phase if a concrete need shows up (an actual S3 requirement for Phase S; a large multi-file feature that would benefit from an OpenSpec change for Phase O).

| Phase | Goal | Key changes | Done when |
|-------|------|--------------|-----------|
| **0 — De-risk multi-rubric** | Prove out the rubric-bundle-manifest design cheaply, with code that survives into Phase 1 rather than being thrown away | Add a small, reusable helper (e.g. `judge/rubric_config.py::RubricConfig.load_bundle()` or similar — a library function, not inline in `judge.py`'s `main()`) that reads a **rubric bundle manifest** (see [Rubric bundle manifest](#rubric-bundle-manifest)) and loads the `RubricConfig` it describes. Wire `judge.py`'s existing `--rubrics` flag (`nargs="+"`, already list-shaped) to call this helper with `args.rubrics[0]`; passing more than one manifest prints a warning and uses only the first (list-from-day-one, length-1-for-now — the first step, not the final limitation). Because the helper lives in the library layer `judge.runner` already delegates to, Phase 1's `vera judge` reuses it directly instead of reimplementing rubric loading from scratch. Contained to `judge/` — no score split, no naming scheme, no `config.json`, no stable interfaces touched | `pytest -m "not live"` green; `--rubrics <manifest path>` actually loads that rubric bundle instead of the hardcoded `data/` one; a fixture for a second rubric bundle exists in `tests/fixtures/` to make this testable |
| **S — Storage abstraction** *(orthogonal — see note above)* | Decouple "what path/key to use" from "how to persist bytes," so a future non-local backend (S3, etc.) is a new implementation, not a rewrite | New `storage/` package: `StorageBackend` (ABC) with `write(key, bytes)` / `read(key)` / `exists(key)`, plus `LocalFilesystemStorage` as the default implementation — mirrors the `LLMInterface`/`QueueProtocol` idiom (interface and implementations live together in one concern-scoped package). The backend knows nothing about run semantics; `utils/naming.py` still builds keys/paths, `storage/` just persists what it's given. All domain/`workers/` code that currently touches the filesystem directly switches to calling through `StorageBackend` instead | `pytest -m "not live"` green; no domain or `workers/` code calls `open()`/`pathlib` file-write directly for run artifacts — everything routes through `StorageBackend`; `LocalFilesystemStorage` is behaviorally identical to today's direct-filesystem writes |
| **O — Adopt OpenSpec** *(orthogonal — see note above)* | Turn "consider an OpenSpec change if the team adopts that workflow" from a maybe into an actual, exercised requirement | Populate `openspec/changes/` with a real OpenSpec change document the next time a large multi-file feature lands (the existing ESCALATE trigger: new judge dimensions, pipeline CLI changes). Currently `openspec/` is empty scaffolding — this phase is "done" only once a real change has actually gone through it, not just once the config exists | A qualifying multi-file feature has shipped with a real OpenSpec change document under `openspec/changes/`, and the ESCALATE section's language is updated from "if the team adopts" to a firm MUST for future qualifying changes |
| **1 — New CLI + config** | `vera.py` fully replaces the top-level scripts | Not cosmetic: `vera.py` subcommands (`generate`, `judge`, `score`, `pool`, `pipeline`, `resume`) become the only entry points. `vera judge` exposes rubric selection via `--rubric` (reusing Phase 0's bundle-manifest helper directly) — rubric selection is not lost during the CLI replacement. `-u`/`-j`/`--sample` shorthand and `--config` ship now, using an **informal** `config.json` shape (mirrors the flags, `generation`/`judging` orthogonal blocks, mutual-exclusivity-with-CLI rule, `judging.rubrics` already a list) — not yet locked as a stable interface. The resolved-form-printed-at-start traceability behavior ships now too (stdout only). Internals underneath still call the existing `generate`/`judge` code as-is — no score split, old `p_*`/`j_*` output layout retained until Phase 2/3. `generate.py`/`judge.py`/`run_pipeline.py` are **deleted entirely at the end of this phase** — no deprecation-stub period. **Known risk, accepted explicitly:** configs written against this informal shape may need updating once Phase 3 formalizes `utils/config_schema.py` as a stable interface | `pytest -m "not live"` green; `vera.py` is the only documented entry point, and `generate.py`/`judge.py`/`run_pipeline.py` no longer exist in the repo; `-u`/`-j`/`--sample`/`--config`/`--rubric` all functional against the existing internals; a **structural** parity/regression test suite (same call sequence, arguments, and output structure — see the generation-testing note below) proves the new CLI wires into the same underlying engine calls the deleted scripts made. Comparatively low risk: Phase 1 only replaces the front end, the generation/judging engine itself (`generate/`, `judge/` internals) is untouched here — the higher-risk moment is Phase 5, where the engine itself changes |
| **2 — Scoring split** | Extract `score/` | `score.py`/`score_viz.py`/`pool.py` move out of `judge/` into `score/`; pure move, no new behavior. `vera.py` (the only entry point since Phase 1) gets its imports updated directly — no shim needed, since the legacy root scripts no longer exist. A minimal import-linter contract is added covering only the `judge/` ⊥ `score/` boundary this phase creates | `pytest -m "not live"` green; the `judge/` ⊥ `score/` import-linter contract passes; no code imports `judge.score`/`judge.pool` |
| **3 — Traceability & naming** | Harden Phase 1's config shape; add persistence; swap in the new naming scheme | `utils/config_schema.py` formalizes Phase 1's informal `config.json` shape into a stable interface (design doc required for future changes) — `judging.rubrics` stays a list (continuing the list-from-day-one approach already used since Phase 0/1), still length-1-only in practice until Phase 4. Adds the persisted artifacts: `config.json` written to disk + `state.json` + `.sha256` sidecar; `utils/naming.py` (the naming/layout module, already tracked today implementing the legacy `p_`/`a_` scheme) is **rewritten** for the `c_`/`u_`/`j_` scheme, retiring the `p_*`/`j_*` layout Phase 1 kept. Existing `output/` run folders under the old layout are left alone — no migration script, only new runs use the new layout. **Acknowledged compatibility break:** anything outside `vera.py` that parses the old `p_*`/`j_*` pattern directly (`spring_scripts/`, `distribute_files.py`, `score_comparison.py`, notebooks, human-review tooling) breaks the moment new runs use `c_*`/`u_*`/`j_*` instead — accepted, since the vast majority of real usage goes through the CLI, not direct path-parsing. **This break is about auto-discovery, not about reading old data at all:** `vera judge --conversations <old p_* folder>` and `vera score -r <old results.csv>` keep working against existing old-layout output, since both take an explicit path and read files by their own format, never by re-deriving meaning from the parent folder's naming pattern. What genuinely doesn't carry over is `vera resume` on an old run — `config.json`/`.sha256`/`state.json` didn't exist under the old layout, so there's nothing for `resume` to read regardless of naming scheme. Import-linter contract extended to cover `utils/` as a leaf | `pytest -m "not live"` green; a run's `config.json` round-trips through `vera resume`; `utils/` leaf-layer import-linter contract passes |
| **4 — Multi-rubric support** | Support multiple rubrics per run | Built on Phase 3's `config.json`/naming: `judging.rubrics[]` now supports length > 1 (the list shape has existed since Phase 0 — this phase lifts the length-1 restriction, it doesn't introduce the list); per-rubric judge-model overrides; per-rubric `evaluations/<rubric>/` folder separation | `pytest -m "not live"` green; a config with 2+ rubrics produces separated `evaluations/<rubric>/` output for each; **and** a pre-existing length-1 `judging.rubrics` config from Phase 1-3 still produces identical behavior — backward compatibility with the single-rubric case is verified, not just the new multi-rubric case |

**Adding a new rubric only requires Phase 0-4, not Phase 5.** Everything a new rubric needs — the bundle-manifest format, `--rubric`/`--target` selection, and (once Phase 4 lands) running it alongside other rubrics in the same config — is available as soon as Phase 4 is done. Phase 5 replaces the generation/judging *engine* underneath (`workers/` unification, concurrency handling); it doesn't change what a rubric is or how one gets added. A new rubric can ship on Phase 4 alone and doesn't need to wait for Phase 5 to land first.
| **5 — Substantial refactoring** | Everything else | `workers/` unification (both `generate/runner.py` and `judge/runner.py` already hand-roll the same asyncio-queue worker-pool pattern independently — concrete evidence for this step); `llm_clients/` plugin-registry formalization; import-linter contract completed (all remaining boundaries) + grimp; quality-gate tightening (pyright blocking, coverage 30%→60%); root-clutter cleanup (see below). **This is the phase that actually rewrites the generation/judging engine** (both runners move off their own hand-rolled asyncio queues onto the shared `workers/` pool) — see the engine-testing note below, this needs more care than any structural CLI change. **Check whether `scripts/pool_vera_scores.py` is still needed** at this point — if so, fold its logic into `score/pool.py` and remove or thin the script; if `vera pool` has already fully superseded it by now, just delete it. **Provider concurrency limits, not yet addressed anywhere in this doc:** the shared `workers/` pool can fan out many parallel jobs (e.g. `-u gpt:5 sonnet:5`); `llm_clients/` already has per-call retry/backoff but nothing caps per-provider *concurrency*, so real parallelism at this phase could blow through a provider's rate limit for the first time. Needs a concurrency cap per provider before this phase is considered done, not just noted as a known gap | `pytest -m "not live"` green; pyright blocking in CI; coverage ≥ 60%; full import-linter contract (all layer boundaries from the [Layer model](#layer-model) section) passes; no root-level Python entry points except `vera.py`; all three testing tiers described below (structural parity, live-LLM smoke test, manual spot-check), specifically for the `workers/` migration; a per-provider concurrency cap exists and is exercised by at least one test that fans out more jobs than the cap allows |

**Rollback:** a phase that ships broken is reverted via its own PR(s) — there is no dual-path/feature-flag expectation carrying old and new behavior side by side during a phase's rollout.

**Phase 5's `workers/` unification is the highest-risk moment for regressions, not Phase 1 — because it's the first phase that actually rewrites the generation/judging engine rather than just the CLI in front of it.** Phase 1 leaves `generate/`/`judge/` internals untouched, so its risk is just "does the new CLI call the same functions with the same arguments" — mechanical and fully covered by structural tests. Phase 5 replaces how those runners actually execute (moving both off independent hand-rolled `asyncio.Queue` worker pools onto the shared `workers/` queue/dispatch), which can change concurrency behavior, timing, and error handling even when the LLM-calling logic itself is unchanged. On top of that, `vera generate`'s output is non-deterministic by nature (sampling, and model responses can drift over time even at fixed settings) and that non-determinism compounds turn over turn across a multi-turn conversation — "behaviorally unchanged" can't mean exact-output diffing the way it might for more structural code. `vera judge` shares the same non-determinism but compounds it far less (one Q&A per rubric dimension, not N open-ended turns). Recommended approach for Phase 5, not yet built:

- **Structural parity (automated, blocking):** using the existing mock harness (`tests/mocks/mock_llm.py`), assert the `workers/`-based runners make the same sequence of calls with the same arguments, respect the same turn-termination logic, and write the same file/output structure as the pre-`workers/` runners did — deterministic, CI-gateable, catches concurrency/wiring regressions specifically introduced by the queue/dispatch rewrite.
- **Live-LLM smoke test (automated, blocking, tolerance-based):** none of this is about whether a conversation's *content* is good — that's inherently non-deterministic and not what Phase 5 risks breaking. What Phase 5 can break is whether the new `workers/`-based engine can still **generate a conversation of the same shape** the old engine did, under real API conditions. A `@pytest.mark.live` test against a small `--sample` batch asserts exactly that, with pass/fail criteria, not a read-through:
  - every sampled run completes without an unhandled exception
  - conversation count matches exactly what was requested (persona × repeat — no silently dropped or duplicated runs)
  - each conversation reaches its expected turn count, or terminates via a documented condition rather than a crash
  - wall-clock time per conversation stays within a tolerance band (e.g. ≤2× a recorded baseline) — catches a concurrency/timing regression from the `workers/` rewrite without requiring exact-timing match
  - a deliberately-injected failure (mock one provider call to error) still triggers the existing retry policy and resolves the same way it does today
- **Manual spot-check (qualitative, genuinely not automatable):** a human reads a handful of the same sampled transcripts for "is this coherent, on-topic, clinically sensible" — content quality, which is the one thing no tolerance band can certify. This is the only piece of Phase 5's testing that stays a judgment call; everything else above is pass/fail.

None of this exists yet. Phase 5's Done-when bar should not be considered met on `pytest -m "not live"` (the default, non-live suite) passing alone — the live smoke test and the manual spot-check both need to run, each for the different thing it actually catches.

**Root-clutter disposition** (Phase 5 cleanup): `logging/`, `logs/`, `conversations/` (root), `human_validation/` — delete (legacy/duplicate generated dumps, already gitignored, superseded by the target `output/c_*` layout). `score_comparisons/` — keep the directory but stop committing generated CSV/PNG output; moves under `score/`. `spring_scripts/`, `openspec/`, `alba.md`, `Untitled.ipynb` — untracked scratch, outside architecture scope. `distribute_files.py` — move into `scripts/`. `docker-compose.yml` volume mounts — fix stale references (`./evaluations` doesn't exist, `./logs` is omitted) to match the target layout.

**Open item, not yet designed:** different personas may need different system prompts, so persona definitions should be linked to their own system prompt rather than assuming one global prompt for all personas. Exact mechanism (a `system_prompt` field on the persona object, a `system_prompt_file` pointer, or something else) is undecided — revisit when the persona schema itself gets formalized (likely Phase 3 or Phase 5).

## Changing this architecture

To add an exception or new boundary:

1. Update this document with rationale.
2. Update [AGENTS.md](../AGENTS.md) if agent stop/escalate rules change.
3. Update [README.md](../README.md) if CLI flags or output layout change.
