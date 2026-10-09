# VERA-MH Architecture

Validation of Ethical and Responsible AI in Mental Health: simulate mental-health conversations, evaluate them against one or more clinical rubrics, and aggregate scores for each rubric for comparison across chatbots under test.

This document describes the **target architecture**. Implementation may lag; [roadmap.md](./roadmap.md) lists the remaining work in order. [README.md](../README.md) covers setup and CLI usage; [vera-cli-use-cases.md](./vera-cli-use-cases.md) covers the CLI/config surface in detail; this doc defines structure, data flow, and what **must** hold. [ARCHITECTURE-SPINE.md](./ARCHITECTURE-SPINE.md) is the terse, numbered invariants-only contract this doc is derived from — cite an `AD-n` from there when a change needs to reference a specific rule.

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
data/<target>/manifest.json ──► generate ──► <target>/c_<chatbot>/u_*/conversations/*.json
                                        │
                                        ▼
                                   judge ──► <target>/c_<chatbot>/u_*/evaluations/j_*/results.csv
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
| Transcript | `<target>/c_<chatbot>/u_<user-model>_<sha>_<ts>/conversations/*.json` | Turn-by-turn chat log; filename encodes persona file + name + chatbot model |
| Rubric | rubric files | Question flow, dimensions, severity. Rubric-derived data files (dimension names, rating values) live outside `src/`, accessible to all packages. |
| Evaluation run | `j_<judge>_<timestamp>_<sha>/` | TSV results, logs, metadata; flat per-run, not nested under a persistent per-judge-model parent |
| Dimension score | `score/aggregate.py` | Aggregated from rubric answers |
| Pooled scores | `score/pool.py` | Concatenates multiple evaluation folders into `<target>/c_<chatbot>/pooled/` (`vera pool`); writes only the merged `results.csv`, `pool_metadata.json`, and `scores/` — never copies transcripts or per-question TSVs |

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

Legacy scripts live in `legacy/` until they are removed, and nothing outside
`legacy/` and its tests imports them. For generation, the flow is
`vera_cli.generate` → `generate.run_for_user_models` →
`generate.run_generation`.

`run_for_user_models` and its `_legacy_model_config` helper are explicit
stopgaps. They put the expansion of a run's user models, and the flattening of
`ModelSpec` into the legacy dict signature, on the domain side of the boundary
rather than in the CLI. Both are deleted when the generation domain accepts
`ModelSpec` directly, which is also what lets `generate` and `judge` describe
models identically.

| Subcommand | Delegates to | Purpose |
|------------|--------------|---------|
| `vera generate` | generation application function (temporarily `generate.run_for_user_models`) | Simulate conversations → `<target>/c_<chatbot>/u_*/conversations/` |
| `vera judge` | `judge.runner` | Evaluate transcripts → `<target>/c_<chatbot>/u_*/evaluations/j_*` |
| `vera score` | `score.run_scoring` | Aggregate `results.csv` → scores and visualizations |
| `vera pool` | `score.pool` | Concatenate multiple evaluation folders into one pooled result → `<target>/c_<chatbot>/pooled/u_<a>+<b>_*` |
| `vera pipeline` | orchestration layer | Full workflow for one chatbot; passes paths between steps |
| `vera resume` | orchestration layer | Reads `config.json` (sha-verified) + `state.json`, continues an incomplete run |

**Deferred resume contract:** `vera resume` is part of the target CLI. The constraints already adopted in `ARCHITECTURE-SPINE.md` — `state.json` as the single mutable artifact `vera resume` writes with a single-writer rule (AD-18), resume's exemption from the run-collision check (AD-24), and its path-first stage contracts (AD-23) — remain stable and are not reopened by this note. What's still unspecified is the surrounding execution machinery: complete run hierarchy, state ownership beyond the single-writer rule, task identity, retry/idempotency semantics, and partial-write recovery behavior. **The invocation shape is also still open**, and is not settled by the `vera resume --config <path>` sketch above: that form still requires retrieving the run's path, whereas review feedback on `--into` asked for the original command plus a flag, with the run folder discovered rather than supplied. Whether resume is a subcommand or a flag, and whether it self-discovers the run, belongs in that design document alongside the machinery questions. Those must be specified in a dedicated design document and adopted as a stable contract in a later migration phase before resume implementation is considered complete.

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
`--rubric` remains list-shaped, though only a length-1 list is supported (multiple
rubrics are evaluated as one target per rubric, by design; see
[roadmap.md](./roadmap.md#completed)). `-c` is required for `generate`/`pipeline` whenever `--config` isn't
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
| `score/` | Aggregation, visualization, pooling — split out of `judge/` | `run.py`, `aggregate.py`, `viz.py`, `pool.py` |
| `workers/` | Shared queue protocol, worker pool, job dispatch | `queue.py`, `job_context.py` |
| `llm_clients/` | Provider plugin registry; providers self-register, factory resolves by prefix | `llm_interface.py`, `llm_factory.py` |
| `storage/` | Storage backend abstraction; raw bytes+keys, knows nothing about run semantics | `storage_backend.py`, `local_filesystem_storage.py` |
| `utils/` | Cross-cutting types, naming/layout, I/O helpers | `role.py`, `naming.py`, `conversation_layout.py` |

**Module naming.** Domain packages (`generate/`, `judge/`, `score/`) use the same file roles: `run.py` holds the `run_<verb>` application entry point and the package `__init__` re-exports it (`from score import run_scoring`); the engine lives in `runner.py` or a topic module (`aggregate.py`, `pool.py`); charts go in `viz.py`, package helpers in `utils.py`. No module repeats its package name. `generate/main.py` is the one remaining exception until it becomes `generate/run.py`.

**Extension points:**

- New LLM provider → [evaluating.md](./evaluating.md)
- Structured judge responses → [structured-output.md](./structured-output.md)
- New storage backend (e.g. S3) → implement `storage/storage_backend.py`'s `StorageBackend` (`write(key, bytes)` / `read(key)` / `exists(key)`)
- Improvement reporting today (`scripts/summarize_results.py`) is deterministic aggregation over `judge/`'s own output, so it belongs under `judge/`. If reporting grows to call an LLM for creative improvement suggestions rather than just stats, that's a different concern (it calls `llm_clients/`, like `generate/`/`judge/` do) and should split into its own `report/` package rather than being absorbed into `judge/` or `score/` — not yet designed, revisit if that feature is actually built.

## Data flow and artifacts

By default, generation writes under `output/` (or a user-specified parent). Full naming rationale: [vera-cli-use-cases.md](./vera-cli-use-cases.md#naming).

```text
output/
└── SI/                                                    ← target root: scores never cross this boundary
    └── c_sonnet/                                          ← persistent per-chatbot directory
        ├── u_gpt-5.2_<sha>_<timestamp>/                   ← generation run: the user model the parent does not name
        │   ├── conversations/
        │   │   └── u_<persona-file>_<persona-name>_c_sonnet.json
        │   └── evaluations/
        │       └── j_claude_<sha>_<timestamp>/            ← judge stays flat, no persistent parent
        │           ├── config.json
        │           ├── config.json.sha256
        │           ├── state.json
        │           ├── results.csv
        │           └── scores/                            ← created by vera score
        └── pooled/                                        ← persistent container, same role as c_sonnet/
            └── u_gpt-5.2+claude-opus-4-5_<sha>_<timestamp>/
                ├── results.csv
                ├── pool_metadata.json
                └── scores/

output/SI/evaluations/<config-sha256>/                      ← persistent per-config directory, same role as c_sonnet/
    └── j_claude_<sha>_<timestamp>/                         ← standalone judging; re-runs don't collide
```

**`<target>` is the outermost segment** because scores produced under different targets are never comparable — the layout makes mixing them structurally impossible rather than merely discouraged, and `ls output/SI/` answers "which chatbots have been evaluated against SI". This does not contradict generation having no knowledge of *rubrics*: a target is personas, prompts, *and* a rubric, and generation already consumes the persona files and context template from it, so conversations genuinely belong to the target that produced them. What remains true is that a run's conversations live together under one target root regardless of which rubrics later judge them. **Because the path already names the target, the `evaluations/<rubric_name>/` segment earlier drafts carried is gone** — and with it the stated blocker for `vera judge --target all`.

Every run-id is `<model>_<sha256-of-config.json>_<timestamp>`, where `<model>` is whichever model the surrounding path does not already name: the *user* model for a generation run (`c_sonnet/` gives the chatbot), the *judge* model for a judging run, and the `+`-joined set of user models for a pooled result. The timestamp goes last, uniformly — everything sharing a prefix still sorts chronologically, and one ordering rule is easier to hold than one per artifact kind. **There is no *generated* nickname:** an earlier draft minted a human-memorable tag (`prophetic-bullfrog`) so a person could refer to a run without quoting a sha, but the model name does that job better and is also what a reader wants to know, so the arbitrary tag is retired. An **optional human label** may be supplied per run and is *added* rather than substituted — `u_<user-model>[_<label>]_<sha>_<timestamp>` — defaulting to none, so the plain form is the model name alone. Additive because a label says why a run exists and should not cost the reader the ability to see what it ran against. It is run-defining, lives in `config.json`, and participates in the config sha like every other field; it is not spelled `--run-id`, being a decorative handle rather than an identifier. `config.json` is an immutable copy of the resolved config, hash-verified via its sidecar; `state.json` is the separate, mutable file that tracks resume progress.

`pooled/` and `<config-sha256>/` are **persistent grouping directories**, the same role `c_sonnet/` plays for generation — neither is itself a run-root and neither is checked for collisions. The run-root inside them is a freshly-generated `u_<a>+<b>_<sha>_<timestamp>/` or `j_claude_<sha>_<timestamp>/` folder, so re-pooling the same combination, or re-invoking `vera judge` standalone with an identical config, produces a new distinct run alongside prior ones — not an error and not a silent overwrite. The pooled combination is named in the path deliberately, so listing one directory says which pools exist without opening a metadata file; it renders at most three model names before appending `+Nmore`, and `pool_metadata.json` stays the authoritative record of exact sources. Sources spanning more than one chatbot have no `c_<chatbot>/` to nest under and group under a top-level `pooled/<config-sha256>/` instead — the same nested-versus-standalone split judging makes.

**Known risk — component-level selection.** `--personas A --rubric B` generates from A and judges with B, so the run lives under `output/A/` while nothing in the path records the rubric's origin. Supported, and deliberately left without a conditional path segment, since mixing components is an expert path. The risk is that the target root's "scores never cross this boundary" guarantee does not hold for such a run, so anything globbing a target root — `vera pool` above all — could merge two rubrics' scores silently. The fix is verification, not naming: aggregation must read rubric identity from each source's recorded config and refuse to merge disagreeing sources, which is what the persisted rubric fingerprint in [Rubric-agnostic scoring](./roadmap.md#rubric-agnostic-scoring) provides.

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

**Documenting a phase in the [roadmap](./roadmap.md) does not pre-clear its ESCALATE requirements.** Every phase — even one that matches this plan exactly — still needs fresh maintainer sign-off before starting, not just at the planning stage. This is deliberate: it ensures a human is actually looking at the moment of highest-risk change (e.g. the [Shared engine](./roadmap.md#shared-engine) phase's `workers/` rewrite), not just at whenever this doc was written.

For large multi-file features (new judge dimensions, pipeline CLI changes), an [OpenSpec](https://github.com/Fission-AI/OpenSpec) change under `openspec/changes/` is required once the [OpenSpec](./roadmap.md#openspec) phase adopts that workflow — see [AGENTS.md](../AGENTS.md). Until then, `openspec/` stays empty scaffolding, not an active practice.

## Enforcement

Target state for automated checks:

| Mechanism | What it checks |
|-----------|----------------|
| `uv run pyright` | Type checking — blocking per package as each reaches zero errors ([Quality gates](./roadmap.md#quality-gates)) |
| Pre-commit | Ruff format/lint |
| CI | Ruff, pyright, `pytest -m "not live"` — coverage gate set in `pyproject.toml` |
| import-linter | Declarative layer contracts (`pyproject.toml`) — added incrementally as each new boundary is created (`judge/` ⊥ `score/` in [Scoring split](./roadmap.md#scoring-split), `utils/` leaf in [Traceability](./roadmap.md#traceability)), completed with the full contract (all [Layer model](#layer-model) boundaries) in [Quality gates](./roadmap.md#quality-gates) |
| grimp | Custom import-graph assertions — added in [Quality gates](./roadmap.md#quality-gates) |
| `.github/CODEOWNERS` | Human review on `vera.py`, import boundaries, domain packages |
| CI (design-doc gate) | Required status check on `pull_request`: fails if the diff touches a [stable interface](#stable-interfaces-agent-coding-optimization) and the PR description has no design-doc link — not yet built |

Run before pushing structural changes:

```bash
uv run pyright
uv run pytest -m "not live"
```

## Changing this architecture

To add an exception or new boundary:

1. Update this document with rationale.
2. Update [AGENTS.md](../AGENTS.md) if agent stop/escalate rules change.
3. Update [README.md](../README.md) if CLI flags or output layout change.
