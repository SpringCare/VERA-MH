# VERA-MH Architecture

Validation of Ethical and Responsible AI in Mental Health: simulate mental-health conversations, evaluate them against one or more clinical rubrics, and aggregate scores for each rubric for comparison across chatbots under test.

This document describes the **target architecture**. Implementation may lag; see [Migration from current layout](#migration-from-current-layout) for known gaps. [README.md](../README.md) covers setup and CLI usage; [vera-cli-use-cases.md](./vera-cli-use-cases.md) covers the CLI/config surface in detail; this doc defines structure, data flow, and what **must** hold. [ARCHITECTURE-SPINE.md](./ARCHITECTURE-SPINE.md) is the terse, numbered invariants-only contract this doc is derived from — cite an `AD-n` from there when a change needs to reference a specific rule.

These are our best guesses as of now, not settled forever — expect some of this to change once actual implementation surfaces things design discussion alone couldn't.

## Entity vocabulary

Three entities (`u`/`c`/`j`) run through the CLI, config, and file naming — full definitions in [vera-cli-use-cases.md#entity-vocabulary](./vera-cli-use-cases.md#entity-vocabulary) (canonical).

## System overview

Two independent pipelines share infrastructure only:

- **Generation** — user LLM ↔ chatbot LLM → transcript files
- **Judging** — judge LLM walks rubric questions → per-dimension severity → scores

They never import each other. A full workflow runs generation, judging, score, and optional pooling in sequence.

All user-facing operations go through **`vera.py`** subcommands. Domain packages are libraries; they are not invoked directly as scripts.

```text
data/<target>/manifest.json ──► generate ──► c_<chatbot>/<run>/conversations/*.json
                                        │
                                        ▼
                                   judge ──► c_<chatbot>/<run>/evaluations/<target>/j_*/results.csv
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
| Target | `data/<target>/manifest.json` | Complete evaluation bundle: rubric, personas, and the prompts needed by generation and judging. |
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
CLI layer
├── vera.py — sole executable; builds the root parser and dispatches
└── vera_cli/
    ├── <command>.py — flags, defaults, resolution, and thin adapter
    ├── config.py — shared config input helpers
    └── targets.py — shared target-manifest resolution
    ↓ calls
Domain packages (generate/, judge/, score/)
    ↓ register handlers with
Workers (workers/) — queue protocol, worker pool, job dispatch
    ↓ used by domain handlers
Infrastructure (llm_clients/, storage/)
    ↓ used by all above
Shared utilities (utils/) — leaf layer
```

**Import rules:**

- `vera.py` owns only the root parser, explicit subcommand registration, and
  dispatch. It contains no command-specific flags, defaults, or business logic.
- `vera_cli/` may import domain packages and `utils/`. Domain packages never
  import `vera_cli/`.
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

Exactly **one** root-level executable: **`vera.py`**. It loads the CLI arguments,
requests a fully resolved invocation from `vera_cli/`, dispatches the selected
command, and renders CLI errors. Full flag/config reference:
[vera-cli-use-cases.md](./vera-cli-use-cases.md).

### CLI runtime boundary

The CLI layer has two levels of responsibility:

- `vera.py` builds the root parser, explicitly registers each supported
  subcommand, parses once, and dispatches. It contains no command-specific
  flags, defaults, resolution, or business logic.
- One `vera_cli/<command>.py` adapter per subcommand keeps that command's flags,
  CLI defaults, canonical resolution, and thin call to the domain function
  together. Shared config input and target-manifest mechanics stay in
  `vera_cli/config.py` and `vera_cli/targets.py`. Command adapters never invoke
  another CLI parser or subprocess. The contract a command must satisfy, and
  what registering one means, are in
  [../vera_cli/README.md](../vera_cli/README.md).

`utils/config_schema.py` owns schema validation and canonical serialization. It
does not parse CLI arguments, read config or manifest files, resolve paths, or
define CLI behavior defaults.

Domain entry points accept resolved domain values rather than `argparse`
namespaces, input config files, or target manifests. They define no CLI behavior
defaults. Pooling likewise delegates to the function owned by the scoring domain
rather than to a script entry point.

Legacy root scripts may remain temporarily while their replacement feature is
migrated. During that transition, `vera_cli` may import the reusable function
from a root script, but never its argument parser. For generation, the temporary
flow is `vera_cli.generate` → `generate.run_for_user_models` → `generate.main`.
Removing `generate.py` and moving those functions plus the existing
`generate_conversations/` code into the permanent `generate/` package is one
atomic later change, so a root `generate.py` module and a top-level `generate/`
package never coexist.

`run_for_user_models` and its `_legacy_model_config` helper are explicit
stopgaps. They put the expansion of a run's user models, and the flattening of
`ModelSpec` into the legacy dict signature, on the domain side of the boundary
rather than in the CLI. Both are deleted when the generation domain accepts
`ModelSpec` directly, which is also what lets `generate` and `judge` describe
models identically.

| Subcommand | Delegates to | Purpose |
|------------|--------------|---------|
| `vera generate` | generation application function (temporarily `generate.run_for_user_models`) | Simulate conversations → `c_<chatbot>/<run>/conversations/` |
| `vera judge` | `judge.runner` | Evaluate transcripts → `evaluations/<target>/j_*` |
| `vera score` | `score.score` | Aggregate `results.csv` → scores and visualizations |
| `vera pool` | `score.pool` | Concatenate multiple evaluation folders into one pooled result |
| `vera pipeline` | orchestration layer | Full workflow for one chatbot; passes paths between steps |
| `vera resume` | orchestration layer | Reads `config.json` (sha-verified) + `state.json`, continues an incomplete run |

**Deferred resume contract:** `vera resume` is part of the target CLI. The constraints already adopted in `ARCHITECTURE-SPINE.md` — `state.json` as the single mutable artifact `vera resume` writes with a single-writer rule (AD-18), resume's exemption from the run-collision check (AD-24), and its path-first stage contracts (AD-23) — remain stable and are not reopened by this note. What's still unspecified is the surrounding execution machinery: complete run hierarchy, state ownership beyond the single-writer rule, task identity, retry/idempotency semantics, and partial-write recovery behavior. Those must be specified in a dedicated design document and adopted as a stable contract in a later migration phase before resume implementation is considered complete.

### Input resolution

Config and run-defining CLI flags are strictly either/or, never combined for the
same run — see [vera-cli-use-cases.md](./vera-cli-use-cases.md#config-mechanism).
Debug and presentation controls such as `--sample`, `--debug`, and `--print`
may accompany either input form. Executed runs record `sample` and `debug` as
invocation metadata in their immutable `config.json`, so it describes how the
run actually executed; `--print` creates no run and is not persisted. `-c`
selects the chatbot under test; `-u`/`-j` shorthand selects models/repeats for
the user/judge side respectively; bespoke sampling knobs are config-only.
`--target` selects a complete target, while `--personas` and `--rubric`
explicitly select only the persona or rubric component of a named target. Each
component includes its associated prompt from that target's manifest.
`--rubric` remains list-shaped, though only a length-1 list is supported until
Phase 4. `-c` is required for `generate`/`pipeline` whenever `--config` isn't
used — there is no default chatbot.

Generation behavior defaults are defined only at the CLI flag boundary. A
config-driven run provides the corresponding generation fields explicitly; it
does not inherit or merge CLI defaults. Both input forms resolve to a complete
`RunConfig`, and the generation runner receives every parameter explicitly and
defines no behavioral defaults of its own.

Consequently, config-driven generation explicitly provides `turns`, `output`,
`max_concurrent`, `max_total_words`, `persona_speaks_first`, `sessions`, and
`persona_context_template`, using `null` where the schema allows no limit or no
session list.

Standalone judging follows the same rule: `JudgingConfig.conversations` mirrors
`--conversations`. A standalone `judge` config must provide it; a `pipeline`
config may omit it because the generation stage supplies the resolved
conversation paths directly.

```bash
uv run python vera.py pipeline --config run.json
uv run python vera.py generate -c sonnet -u gpt:1 --target SI
uv run python vera.py generate -c sonnet -u gpt:1 --personas SI
uv run python vera.py judge -j claude:1 --rubric SI --conversations output/c_sonnet/<run>/conversations/
uv run python vera.py score -r output/.../results.csv
uv run python vera.py pool --evaluations path/to/evaluations/... path/to/evaluations/...
uv run python vera.py resume --config output/c_sonnet/<run>/config.json
```

`vera judge` never takes `-c`: judging is decoupled from chatbot selection by design (the `generation`/`judging` orthogonality invariant, below) — the chatbot is already implicit in whichever `--conversations` folder is passed in.

### Target manifest

A **target** is the complete, reusable combination of a rubric, personas, and
the prompts required to generate and judge conversations. Every target is
defined by a file named `manifest.json`; “manifest” is the representation and
“target” is the domain concept.

The manifest is complete rather than tailored to one command:

```json
{
  "rubric_file": "rubric.tsv",
  "rubric_prompt_beginning_file": "rubric_prompt_beginning.txt",
  "question_prompt_file": "question_prompt.txt",
  "personas": ["personas.tsv"],
  "persona_context_template_file": "persona_context_template.txt"
}
```

All five fields are required. A command may consume only part of the target, but
that does not make the remaining fields optional: `generate` uses the personas
and persona prompt, while `judge` uses the rubric and judging prompts.

Paths are relative to the manifest's own folder — distinct from `config.json`,
whose paths resolve relative to `$ROOT` (the directory containing `vera.py`),
never to the manifest, the config file's own location, or the CLI's working
directory (see [Config mechanism](./vera-cli-use-cases.md#config-mechanism)).

**Example — two path fields, two different anchors:**

```text
project-root/                         ← $ROOT (contains vera.py)
├── vera.py
├── configs/
│   └── run.json                      ← config.json
└── data/
    └── SI/
        ├── manifest.json             ← target manifest
        ├── personas.tsv
        ├── persona_context_template.txt
        ├── rubric.tsv
        ├── rubric_prompt_beginning.txt
        └── question_prompt.txt
```

- `configs/run.json`'s `generation.personas: ["data/SI/personas.tsv"]` always means `project-root/data/SI/personas.tsv` — resolved against `$ROOT`, no matter where you run `vera.py` from or where `run.json` itself lives.
- `data/SI/manifest.json`'s `personas: ["personas.tsv"]` always means `data/SI/personas.tsv` — resolved against the manifest's own folder, so the whole target directory stays portable if copied into a different checkout, independent of `$ROOT`.

Same shape of field, same-looking relative string, two different rules — hence calling both out explicitly here rather than leaving it implicit.

**Separation of concerns:** the target manifest describes the static evaluation
bundle and changes rarely. `config.json` describes how to run it — models,
repeats, and per-run overrides — and changes every run. Model defaults and other
execution knobs belong in CLI flag definitions or `config.json`, never in the
manifest.

**Whole-target and explicit-component selection:** `--target <name-or-path>`
loads the whole manifest. `vera generate --target ...` consumes its persona and
persona-prompt fields; `vera judge --target ...` consumes its rubric and
judging-prompt fields; `vera pipeline --target ...` consumes both.

For standalone judging, `--target SI` and `--rubric SI` therefore resolve to
the same rubric and judging prompts. The difference is wording, not runtime
behavior: `--target` says “select SI's complete target” even though judge uses
only its judging component, while `--rubric` explicitly names that component.
Both forms remain available so a caller can express either intent consistently
with `generate` and `pipeline`.

Advanced callers may keep generation and judging independent. `--personas`
`<name-or-manifest-path>` selects only the personas and persona prompt from that
target, while `--rubric <name-or-manifest-path>` selects only its rubric and
judging prompts. For example, `--personas HFO --rubric SI` deliberately combines
the persona side of HFO with the rubric side of SI. A pipeline may use both
explicit flags instead of `--target`; neither flag pulls in the target's other
component.

A top-level `target` in an input config mirrors whole-target selection. Setting
it alongside explicit `generation.personas` or `judging.rubrics` is an error,
never a merge or override.

`--target all` enumerates targets; it does not merge their personas, prompts, or
rubrics. The resolver produces one complete canonical invocation per target so
each target retains its own persona and judging prompts.

Target expansion is complete before `--print`, persistence, or command dispatch.
The canonical `RunConfig` contains concrete persona, context-template, rubric,
and judging-prompt paths rather than `target`, so a persisted run never needs to
re-resolve a manifest that may later change. An incomplete manifest is not a
valid target, and there is no fallback to SI data. Every invocation that
explicitly specifies personas and rubrics keeps generation and judging
orthogonal.

## Package responsibilities

| Package / path | Owns | Key modules |
|----------------|------|-------------|
| `vera_cli/` | One cohesive adapter per command plus shared config/target helpers | `generate.py`, `judge.py`, `config.py`, `targets.py` |
| `generate/` | Simulation, turns, batch runner (pure core; handler owns I/O) | `conversation_simulator.py`, `runner.py` |
| `judge/` | Rubric navigation, LLM judge, improvement reporting (pure core; handler owns I/O) | `question_navigator.py`, `llm_judge.py`, `scripts/summarize_results.py` |
| `score/` | Aggregation, visualization, pooling — split out of `judge/` | `score.py`, `score_viz.py`, `pool.py` |
| `workers/` | Shared queue protocol, worker pool, job dispatch | `queue.py`, `job_context.py` |
| `llm_clients/` | Provider plugin registry; providers self-register, factory resolves by prefix | `llm_interface.py`, `llm_factory.py` |
| `storage/` | Storage backend abstraction; raw bytes+keys, knows nothing about run semantics | `storage_backend.py`, `local_filesystem_storage.py` |
| `utils/` | Cross-cutting types, naming/layout, I/O helpers | `role.py`, `naming.py`, `conversation_layout.py` |

**Extension points:**

- New LLM provider → [evaluating.md](./evaluating.md)
- Structured judge responses → [structured-output.md](./structured-output.md)
- New storage backend (e.g. S3) → implement `storage/storage_backend.py`'s `StorageBackend` (`write(key, bytes)` / `read(key)` / `exists(key)`)
- Improvement reporting today (`scripts/summarize_results.py`) is deterministic aggregation over `judge/`'s own output, so it belongs under `judge/`. If reporting grows to call an LLM for creative improvement suggestions rather than just stats, that's a different concern (it calls `llm_clients/`, like `generate/`/`judge/` do) and should split into its own `report/` package rather than being absorbed into `judge/` or `score/` — not yet designed, revisit if that feature is actually built.

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

output/evaluations/<config-sha256>/                        ← persistent per-config directory, same role as c_sonnet/
    └── <rubric_name>/                                     ← standalone judging (vera judge run on its own)
        └── j_claude_<nickname>_<timestamp>_<sha>/         ← same run-root shape as the nested case -- re-runs don't collide
```

Every run-id is `<nickname>_<timestamp>_<sha256-of-config.json>` when nested under a per-model parent (`c_sonnet/`, where the model is already given by the parent folder), or `<model>_<nickname>_<timestamp>_<sha>` when flat (`j_claude_.../`, where nothing else identifies the model). `<nickname>` is a generated human-memorable tag (e.g. via a word-pair generator), purely for a person to recognize a run at a glance — it carries no identifying information itself and is never a substitute for the sha; it never needs to encode which model was used, since the surrounding path already does that job. `config.json` is an immutable copy of the resolved config, hash-verified via its sidecar; `state.json` is the separate, mutable file that tracks resume progress.

`<config-sha256>/` in the standalone-judging path is a **persistent, per-config grouping directory**, the same role `c_sonnet/` plays for generation — it is never itself a run-root and is never checked for collisions. The actual run-root inside it is still a freshly-generated `j_claude_<nickname>_<timestamp>_<sha>/` folder, so re-invoking `vera judge` standalone with an identical config produces a new, distinct run grouped alongside prior ones under the same `<config-sha256>/`, not an error and not a silent overwrite of the prior run.

**Single canonical hash, not two independent computations:** the sha256 is computed exactly once, by exactly one function, and that single value is what both the run-id folder name's `<sha>` component and the `config.json.sha256` sidecar's content contain — never two independently-computed values that could drift apart. Rejected: embedding the hash in `config.json`'s own filename (e.g. `config.<sha>.json`) — doesn't remove the need for a verification step (a corrupted file wouldn't automatically stop matching its own filename; verification still requires hashing the actual bytes, which is the sidecar's job), and would put the same value in a third place with no added integrity benefit.

## Invariants

Agents and contributors must comply. Import boundaries are documented in the [Layer model](#layer-model) section above.

### MUST

- **Single CLI:** `vera.py` is the sole executable; CLI support code lives in
  `vera_cli/`, and domain behavior remains in domain packages.
- **Subcommands:** `generate`, `judge`, `score`, `pool`, `pipeline`, `resume` (add or remove only via [ESCALATE](#escalate-stop-and-ask)).
- **Resolved boundary:** flags, config, paths, and targets resolve to canonical
  values before print, persistence, or dispatch. Domain functions never parse
  CLI/config inputs or define CLI behavior defaults.
- **Targets:** every `manifest.json` defines one complete target (rubric,
  personas, and prompts). `--target` selects the complete bundle;
  `--personas <target>` and `--rubric <target>` remain explicit component-level
  alternatives and include the selected component's prompts.
- **Generation:** conversation simulation logic stays in `generate/`; the simulator core is pure (no filesystem, no logging) — the handler owns all I/O.
- **Judging:** rubric navigation and LLM-judge logic stay in `judge/`, also pure-core-plus-handler. Judge never auto-scores — `vera score`/`vera pool` are separate subcommands. **Rubric navigation logic lives in code, never in the prompt:** which question is asked next given an answer is determined entirely by `QuestionNavigator` walking `question_flow_data` parsed from the rubric TSV — the judge LLM answers/judges the current question only, and is never asked to decide or influence what comes next.
- **Scoring:** aggregation, visualization, and pooling stay in `score/`, never re-absorbed into `judge/`.
- **Config vs CLI:** `--config` and run-defining CLI flags are strictly either/or for a given run. `--sample <N>`, `--debug`, and `--print` may accompany either form; executed runs record `sample` and `debug` as invocation metadata in their immutable `config.json`, while `--print` creates no run. `generation` and `judging` blocks in `config.json` are completely orthogonal; model selection for one must never influence the other.
- **Naming/layout:** all folder/file naming logic (the `c_`/`u_`/`j_` scheme) lives in one `utils/` module — never duplicated across handlers.
- **Traceability:** every run writes an immutable `config.json` (+ `.sha256` sidecar) and a separate, mutable `state.json`. `state.json` records both the requested and actual-resolved model identifier.
- **LLM providers:** new providers implement [llm_clients/llm_interface.py](../llm_clients/llm_interface.py) and register in [llm_clients/llm_factory.py](../llm_clients/llm_factory.py). Every model-list entry's `name` in config is always a specific model identifier, never a bare provider name.
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
| `utils/conversation_layout.py` | Builds directly on `utils/naming.py` and is inseparable from it in practice — protected alongside it, not covered by a separate rationale |
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
| **0 — De-risk multi-rubric** | Prove the target-manifest format on both generation and judging before the unified CLI replaces legacy scripts | Treat every manifest as one complete target containing a rubric, personas, and both prompt sets. Legacy judge code may consume only its rubric fields and legacy generation code only its persona fields, but both read the same complete manifest. Preserve the existing explicit persona/rubric paths so Phase 1 can offer both whole-target and component-level selection | `pytest -m "not live"` green; one complete fixture target drives both legacy generation and judging; incomplete manifests fail validation; explicit persona/rubric selection remains covered |
| **S — Storage abstraction** *(orthogonal — see note above)* | Decouple "what path/key to use" from "how to persist bytes," so a future non-local backend (S3, etc.) is a new implementation, not a rewrite | New `storage/` package: `StorageBackend` (ABC) with `write(key, bytes)` / `read(key)` / `exists(key)`, plus `LocalFilesystemStorage` as the default implementation — mirrors the `LLMInterface`/`QueueProtocol` idiom (interface and implementations live together in one concern-scoped package). The backend knows nothing about run semantics; `utils/naming.py` still builds keys/paths, `storage/` just persists what it's given. All domain/`workers/` code that currently touches the filesystem directly switches to calling through `StorageBackend` instead | `pytest -m "not live"` green; no domain or `workers/` code calls `open()`/`pathlib` file-write directly for run artifacts — everything routes through `StorageBackend`; `LocalFilesystemStorage` is behaviorally identical to today's direct-filesystem writes |
| **O — Adopt OpenSpec** *(orthogonal — see note above)* | Turn "consider an OpenSpec change if the team adopts that workflow" from a maybe into an actual, exercised requirement | Populate `openspec/changes/` with a real OpenSpec change document the next time a large multi-file feature lands (the existing ESCALATE trigger: new judge dimensions, pipeline CLI changes). Currently `openspec/` is empty scaffolding — this phase is "done" only once a real change has actually gone through it, not just once the config exists | A qualifying multi-file feature has shipped with a real OpenSpec change document under `openspec/changes/`, and the ESCALATE section's language is updated from "if the team adopts" to a firm MUST for future qualifying changes |
| **1 — New CLI + config** | `vera.py` fully replaces the top-level scripts | Add whole-target selection through `--target` and top-level config `target`. Preserve explicit component selection through `--personas <target>` and `--rubric <target>`, so callers can combine the persona side of one target with the rubric side of another. Resolve target manifests before print or dispatch, then call existing domain behavior directly. Ship `-u`/`-j`/`--sample` and the informal config shape; delete `generate.py`/`run_pipeline.py` at the end of the phase. **`judge.py` is the one exception and outlives this phase:** `vera judge` ships without `--resume`, because the resume contract is deferred (see the Deferred resume contract note above), so `judge.py` is retained *solely* as the resume entry point until `vera resume` exists. It is not a general escape hatch — no new work targets it, and it is deleted the moment `vera resume` lands. Judging output also keeps the existing `<gen_run>/evaluations/j_*` layout in this phase; Phase 3 renames it and Phase 4 adds the `<target>/` segment. **Acknowledged compatibility break:** `vera judge` drops legacy `judge.py`'s fallback of writing evaluations to `evaluations/` relative to the working directory when the input is a flat transcript folder rather than a generation run — it errors and requires `-o/--output` instead. Accepted because that fallback detached evaluations from the conversations that produced them, and because a bare relative path resolves against the working directory on the CLI but against the repository root in a config. Reading old flat-layout conversations still works with an explicit `-o`, satisfying the read-old-data guarantee above | `pytest -m "not live"` green; `vera.py` is the only documented entry point for everything except resume; target and explicit-component paths have structural parity tests; `--config`, `--target`, `--personas`, `--rubric`, `-u`, `-j`, and `--sample` are functional; `generate.py` and `run_pipeline.py` are gone |
| **2 — Scoring split** | Extract `score/` | `score.py`/`score_viz.py`/`pool.py` move out of `judge/` into `score/`; pure move, no new behavior. `vera.py` (the only entry point since Phase 1) gets its imports updated directly — no shim needed, since the legacy root scripts no longer exist. A minimal import-linter contract is added covering only the `judge/` ⊥ `score/` boundary this phase creates | `pytest -m "not live"` green; the `judge/` ⊥ `score/` import-linter contract passes; no code imports `judge.score`/`judge.pool` |
| **3 — Traceability & naming** | Harden Phase 1's config shape; add persistence; swap in the new naming scheme | `utils/config_schema.py` formalizes Phase 1's informal `config.json` shape into a stable interface (design doc required for future changes) — `judging.rubrics` stays a list (continuing the list-from-day-one approach already used since Phase 0/1), still length-1-only in practice until Phase 4. Adds the persisted artifacts: `config.json` written to disk + `state.json` + `.sha256` sidecar; `utils/naming.py` (the naming/layout module, already tracked today implementing the legacy `p_`/`a_` scheme) is **rewritten** for the `c_`/`u_`/`j_` scheme, retiring the `p_*`/`j_*` layout Phase 1 kept. Existing `output/` run folders under the old layout are left alone — no migration script, only new runs use the new layout. **Acknowledged compatibility break:** anything outside `vera.py` that parses the old `p_*`/`j_*` pattern directly (`spring_scripts/`, `distribute_files.py`, `score_comparison.py`, notebooks, human-review tooling) breaks the moment new runs use `c_*`/`u_*`/`j_*` instead — accepted, since the vast majority of real usage goes through the CLI, not direct path-parsing. **This break is about auto-discovery, not about reading old data at all:** `vera judge --conversations <old p_* folder>` and `vera score -r <old results.csv>` keep working against existing old-layout output, since both take an explicit path and read files by their own format, never by re-deriving meaning from the parent folder's naming pattern. What genuinely doesn't carry over is `vera resume` on an old run — `config.json`/`.sha256`/`state.json` didn't exist under the old layout, so there's nothing for `resume` to read regardless of naming scheme. Import-linter contract extended to cover `utils/` as a leaf | `pytest -m "not live"` green; a run's `config.json` round-trips through `vera resume`; `utils/` leaf-layer import-linter contract passes |
| **4 — Multi-rubric support** | Support multiple rubrics per run | Built on Phase 3's `config.json`/naming: `judging.rubrics[]` now supports length > 1 (the list shape has existed since Phase 0 — this phase lifts the length-1 restriction, it doesn't introduce the list); per-rubric judge-model overrides; per-rubric `evaluations/<target>/` folder separation, one folder per rubric-providing target (the segment is named for the target the rubric came from, not for the rubric file, so it matches `c_<chatbot>`; with `--rubric` that may differ from the run's own target). Also enables `vera judge --target all`, deferred from Phase 1 because judging output could not be attributed to a rubric before this segment existed | `pytest -m "not live"` green; a config with 2+ rubrics produces separated `evaluations/<target>/` output for each; **and** a pre-existing length-1 `judging.rubrics` config from Phase 1-3 still produces identical behavior — backward compatibility with the single-rubric case is verified, not just the new multi-rubric case; `vera judge --target all` no longer errors and its output is attributable per target |

**Adding a new target or rubric only requires Phase 0-4, not Phase 5.** A complete target is added as `data/<target>/manifest.json`; an expert may still select its rubric component explicitly with `--rubric`. Phase 5 replaces the generation/judging *engine* underneath and does not change target or rubric definition.
| **5 — Substantial refactoring** | Everything else | `workers/` unification (both `generate/runner.py` and `judge/runner.py` already hand-roll the same asyncio-queue worker-pool pattern independently — concrete evidence for this step); `llm_clients/` plugin-registry formalization; import-linter contract completed (all remaining boundaries) + grimp; quality-gate tightening (pyright blocking, coverage 30%→60%); root-clutter cleanup (see below). **This is the phase that actually rewrites the generation/judging engine** (both runners move off their own hand-rolled asyncio queues onto the shared `workers/` pool) — see the engine-testing note below, this needs more care than any structural CLI change. **Check whether `scripts/pool_vera_scores.py` is still needed** at this point — if so, fold its logic into `score/pool.py` and remove or thin the script; if `vera pool` has already fully superseded it by now, just delete it. **Provider concurrency limits, not yet addressed anywhere in this doc:** the shared `workers/` pool can fan out many parallel jobs (e.g. `-u gpt:5 sonnet:5`); `llm_clients/` already has per-call retry/backoff but nothing caps per-provider *concurrency*, so real parallelism at this phase could blow through a provider's rate limit for the first time. Needs a concurrency cap per provider before this phase is considered done, not just noted as a known gap | `pytest -m "not live"` green; pyright blocking in CI; coverage ≥ 60%; full import-linter contract (all layer boundaries from the [Layer model](#layer-model) section) passes; no root-level Python entry points except `vera.py`; all four testing tiers described below (structural parity, live-LLM smoke test, semantic similarity check, manual spot-check), specifically for the `workers/` migration; a per-provider concurrency cap exists and is exercised by at least one test that fans out more jobs than the cap allows |

**Rollback:** a phase that ships broken is reverted via its own PR(s) — there is no dual-path/feature-flag expectation carrying old and new behavior side by side during a phase's rollout.

**Phase 5's `workers/` unification is the highest-risk moment for regressions, not Phase 1 — because it's the first phase that actually rewrites the generation/judging engine rather than just the CLI in front of it.** Phase 1 leaves `generate/`/`judge/` internals untouched, so its risk is just "does the new CLI call the same functions with the same arguments" — mechanical and fully covered by structural tests. Phase 5 replaces how those runners actually execute (moving both off independent hand-rolled `asyncio.Queue` worker pools onto the shared `workers/` queue/dispatch), which can change concurrency behavior, timing, and error handling even when the LLM-calling logic itself is unchanged. On top of that, `vera generate`'s output is non-deterministic by nature (sampling, and model responses can drift over time even at fixed settings) and that non-determinism compounds turn over turn across a multi-turn conversation — "behaviorally unchanged" can't mean exact-output diffing the way it might for more structural code. `vera judge` shares the same non-determinism but compounds it far less (one Q&A per rubric dimension, not N open-ended turns). Recommended approach for Phase 5, not yet built:

- **Structural parity (automated, blocking):** using the existing mock harness (`tests/mocks/mock_llm.py`), assert the `workers/`-based runners make the same sequence of calls with the same arguments, respect the same turn-termination logic, and write the same file/output structure as the pre-`workers/` runners did — deterministic, CI-gateable, catches concurrency/wiring regressions specifically introduced by the queue/dispatch rewrite.
- **Live-LLM smoke test (automated, blocking, tolerance-based):** none of this is about whether a conversation's *content* is good — that's inherently non-deterministic and not what Phase 5 risks breaking. What Phase 5 can break is whether the new `workers/`-based engine can still **generate a conversation of the same shape** the old engine did, under real API conditions. A `@pytest.mark.live` test against a small `--sample` batch asserts exactly that, with pass/fail criteria, not a read-through:
  - every sampled run completes without an unhandled exception
  - conversation count matches exactly what was requested (persona × repeat — no silently dropped or duplicated runs)
  - each conversation reaches its expected turn count, or terminates via a documented condition rather than a crash
  - wall-clock time per conversation stays within a tolerance band (e.g. ≤2× a recorded baseline) — catches a concurrency/timing regression from the `workers/` rewrite without requiring exact-timing match
  - a deliberately-injected failure (mock one provider call to error) still triggers the existing retry policy and resolves the same way it does today
- **Semantic similarity check (automated, statistical, blocking):** the middle ground between "did it run" (smoke test) and "does it read well" (manual) — whether the new engine's conversations are still saying the *same kind of thing* as the old engine's, without requiring identical wording, which non-determinism rules out. For each sampled `(persona, config)` pair, generate one conversation on the old engine and one on the new engine, embed both (via the existing LLM provider's embedding endpoint), and compute cosine similarity. Compare that old-vs-new similarity distribution against a **calibrated baseline**: the same old-vs-old similarity distribution from two independent old-engine runs of the same `(persona, config)` pairs — the expected variance from LLM sampling alone, with nothing about the engine changed. A statistically significant drop in old-vs-new similarity relative to that baseline (e.g. a one-sided test at a pre-agreed significance level) flags semantic drift the structural and smoke tiers can't see, without over-flagging the ordinary variance LLM sampling already produces run to run.
- **Manual spot-check (qualitative, genuinely not automatable):** a human reads a handful of the same sampled transcripts for "is this coherent, on-topic, clinically sensible" — content quality, which is the one thing no tolerance band or similarity score can certify. This is the only piece of Phase 5's testing that stays a judgment call; everything else above is pass/fail or statistical.

None of this exists yet. Phase 5's Done-when bar should not be considered met on `pytest -m "not live"` (the default, non-live suite) passing alone — the live smoke test, the semantic similarity check, and the manual spot-check all need to run, each for the different thing it actually catches.

**Root-clutter disposition** (Phase 5 cleanup): `logging/`, `logs/`, `conversations/` (root), `human_validation/` — delete (legacy/duplicate generated dumps, already gitignored, superseded by the target `output/c_*` layout). `score_comparisons/` — keep the directory but stop committing generated CSV/PNG output; moves under `score/`. `spring_scripts/`, `openspec/`, `alba.md`, `Untitled.ipynb` — untracked scratch, outside architecture scope. `distribute_files.py` — move into `scripts/`. `docker-compose.yml` volume mounts — fix stale references (`./evaluations` doesn't exist, `./logs` is omitted) to match the target layout.

**Open item, not yet designed:** different personas may need different system prompts, so persona definitions should be linked to their own system prompt rather than assuming one global prompt for all personas. Exact mechanism (a `system_prompt` field on the persona object, a `system_prompt_file` pointer, or something else) is undecided — revisit when the persona schema itself gets formalized (likely Phase 3 or Phase 5).

## Changing this architecture

To add an exception or new boundary:

1. Update this document with rationale.
2. Update [AGENTS.md](../AGENTS.md) if agent stop/escalate rules change.
3. Update [README.md](../README.md) if CLI flags or output layout change.
