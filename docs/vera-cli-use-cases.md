---
status: resolved
updated: 2026-07-16
---

# `vera.py` — Use Cases and CLI/Config Design

This document lists the intended use cases for the `vera.py` entry point, the CLI/config surface that supports them, and the naming scheme for run artifacts. Originally circulated as a draft for review (PR #170); all open questions from that review have been resolved and are marked below.

## Entity vocabulary

Three entities, each with a single-letter prefix used throughout the CLI, config, and file naming:

- **`u` — user**: the persona-side LLM simulating the user.
- **`c` — chatbot**: the provider/agent LLM under test (previously called "provider" or "agent" inconsistently — `chatbot` is now the standard term).
- **`j` — judge**: the LLM evaluating a transcript against a rubric.

**These letters are `vera.py`-only and are not the same flags as today's scripts.** `generate.py`/`judge.py` already use `-c`/`-r` for unrelated things (`-c` is `--max-concurrent` in `generate.py` and `--conversation` in `judge.py`; `-r` is `--runs` in `generate.py` and `--rubrics` in `judge.py`). `vera.py` intentionally repurposes them for the `u`/`c`/`j` vocabulary above. There is no coexistence window: Phase 1 of the migration (see [architecture.md#migration-from-current-layout](./architecture.md#migration-from-current-layout)) deletes `generate.py`/`judge.py`/`run_pipeline.py` entirely in the same change that ships `vera.py`, so the old and new meanings of `-c`/`-r` never need to be told apart at runtime.

## Minimum required arguments

CLI shorthand and the input `config.json` are deliberately mirrored, flag-for-field — the same information is required either way, just spelled differently:

| Subcommand | CLI shorthand — minimum required | Input `config.json` — minimum required fields |
|---|---|---|
| `generate` | `-c <chatbot>` + `-u <model:repeats...>` (≥1) + (`--personas <file...>` **or** `--target <name>`) | `generation.chatbot`, `generation.user` (≥1), and (`generation.personas` (≥1) **or** top-level `target`) |
| `judge` | `-j <model:repeats...>` (≥1) + `--conversations <folder...>` + `--rubric <manifest>` | `judging.models` (≥1), `judging.rubrics` (≥1), and `judging.conversations` (≥1) |
| `pipeline` | everything `generate` needs **and** everything `judge` needs — or just `-c`, `-u`, `-j`, `--target <name>` (`--target` covers persona+rubric in one shot) | both `generation` and `judging` blocks fully populated, or `-c`/`-u`/`-j`-equivalent fields plus a top-level `target`; `judging.conversations` is omitted because generation supplies it |
| `score` | the results path (`-r <results.csv>`) — nothing else | n/a — `score` reads an existing `results.csv`, not a run config |
| `pool` | `--evaluations <folder...>` (≥1) | n/a |
| `resume` | `--config <run's own config.json>` — that's the entire invocation | n/a (it *is* the config being resumed) |

**`target` mirrors `--target` exactly, including its mutual-exclusivity rule:** a top-level `target` field in the input `config.json` expands to `generation.personas` + `judging.rubrics` from the resolved rubric bundle manifest, the same expansion `--target <name>` performs on the CLI. **Setting `target` alongside explicit `generation.personas` or `judging.rubrics` in the same input config is an error** — rejected outright, not silently merged or overridden — mirroring the existing either/or rule between CLI flags and `--config`. The run's own immutable `config.json` artifact (written to `output/.../config.json`, the one `vera resume` reads) always stores the fully-expanded form — `target` is resolved away before that artifact is written, the same way `-u`/`-j` shorthand is already resolved into concrete model entries today, so `vera resume` never has to re-resolve a manifest that might have changed on disk since the run started.

## Use case 1 — End-to-end test of one LLM

Generate -> judge -> score chained, for exactly one chatbot under test, in a single invocation.

```
vera pipeline --config run.json
```

**Resolved:** stays single-chatbot-per-invocation. Comparing chatbots is always an external loop over single-chatbot pipeline runs, consistent with use case 2. Native multi-chatbot support (one combined comparison report) is not built now — flagged as a possible future addition if a real need emerges, not ruled out permanently.

**`--target <name>` shorthand:** `vera pipeline --target SI` resolves `SI` to one rubric bundle manifest (see [Rubric bundle manifest](./architecture.md#rubric-bundle-manifest)) and sets *both* the generation personas and the judging rubric from it in one shot — for the common case of "run the canonical test for X." This is the one deliberate exception to personas/rubrics being chosen independently; every other invocation (`--rubric` plus separately-specified personas, or explicit `--config` blocks) keeps generation and judging fully orthogonal. `--target` never selects the chatbot — `-c`/`generation.chatbot` is required regardless of whether `--target` is used.

**No implicit "run everything":** if neither `--target` nor `--rubric`/`judging.rubrics` is given, the CLI errors rather than defaulting to some or all rubrics. To deliberately run every known evaluator, use `--target all`, which resolves every rubric bundle manifest — an explicit opt-in, not a default.

## Use case 2 — Batch generate across personas

One chatbot under test, generated against multiple personas, each carrying its own user-side LLM.

Two independent, equally valid ways to run this — not a two-step sequence:

**Option A — config file:**
```
vera generate --config run.json
```

**Option B — CLI shorthand:**
```
vera generate -c sonnet -u gpt:1 sonnet:2 --personas data/personas.tsv
```
Bare-minimum required flags for Option B: `-c <chatbot>` (the chatbot under test — no default), `-u <model:repeats...>` (at least one user-side model), and either `--personas <file...>` or `--target <name>` (personas have no silently-assumed default file either). `--rubric`/`--target` are additionally required if this invocation will also be judged.

Generation behavior is also controlled at this input boundary. CLI invocations
default to `--turns 3`, `--output output`, unlimited concurrency, no total-word
cap, persona-first ordering, one unnamed session, and
`--persona-context-template data/SI/persona_context_template.txt` when explicit
persona files are selected. `--target` obtains the context template from its
manifest instead. The generation runner itself has no parameter defaults.

Or, using `--target` to pull personas from a rubric bundle manifest instead of naming them explicitly:
```
vera generate -c sonnet -u gpt:1 --target SI
```

`-u <model>:<repeats> ...` selects the user-side LLM(s) and how many full passes over the configured persona set to run with each. With 10 personas configured, `-u gpt:1 sonnet:2` means 10 conversations with gpt, 20 with sonnet (2 full passes). Each `(model, repeat)` pass gets its own run under that model's persistent output directory (see Naming below).

Comparing across *chatbots* is done by looping this invocation externally, once per chatbot — not a built-in cross-product flag.

Personas come from one or more persona files, each containing multiple personas; duplicate persona names across files are possible (disambiguated by file + name, see Naming).

## Use case 3 — Judge existing conversations

Judge one **or more** existing transcript folders against one or more rubrics. Each rubric has a default judge-LLM set, overridable per rubric. Multiple judges per rubric are supported (for judge-agreement analysis). Judging is decoupled from generation — point it at whatever conversation folders you need, with no enforced coupling to originating personas.

```
vera judge --config run.json
vera judge -j claude:1 gpt:2 --conversations output/c_sonnet/<run>/conversations/ --rubric data/SI/rubric_manifest.json
```

No `-c` here: judging is decoupled from chatbot selection by design (see the orthogonality invariant above) — the chatbot is already implicit in whichever `--conversations` folder is passed in.

`-j <model>:<repeats> ...` mirrors `-u`'s syntax for the judge side. `repeats` here means re-running the same transcript through the same judge model N times, to measure judge consistency/variance.

`--rubric`/`judging.rubrics[]` entries point at a [rubric bundle manifest](./architecture.md#rubric-bundle-manifest) (canonical definition), not a bare `.tsv` path.

**Resolved (multi-folder judge output):** judge keeps results independent per folder; the `score/` layer aggregates across folders when needed, not judge itself. There are also in-between options — e.g. kept separate, but the score layer aggregates them.

**Resolved (multi-rubric output layout):** generated conversations for a run all live together in one unified folder regardless of which rubrics will later judge them — generation has no knowledge of rubrics. Judging output IS separated per rubric, since different rubrics produce different, non-comparable scores. (The unified-conversations choice is soft, not a hard invariant — may change if a concrete need for per-rubric conversation grouping emerges.)

## Use case 4 — Smoke test

Run the full pipeline (or generate/judge alone) against a small sample of personas/rubrics/judges, to sanity-check that a config works end-to-end before spending the full LLM-call budget.

```
vera pipeline --config run.json --sample 2
```

`--sample N` overrides the config's full persona (and rubric/judge, where relevant) list at run time. This avoids hand-maintaining a separate small-scale config just for smoke testing.

**`--sample` is the sole, named exception to the CLI/`--config` either-or rule** (AD-17 in [ARCHITECTURE-SPINE.md](./ARCHITECTURE-SPINE.md)): it's a debug-only cap on how much of the config's already-resolved lists get used, never a way to set information `config.json` itself carries, and it's never written into the run's own `config.json` artifact -- `vera resume` on a sampled run still resolves against the full original config. No other flag gets this treatment.

## Use case 5 — Pool

Concatenate multiple existing evaluation output folders into one pooled result.

```
vera pool --evaluations <folder> <folder> ...
```

Owned by `score/`, consistent with score owning aggregation-across-runs (as opposed to judging's own within-run aggregation). This formalizes and generalizes the `vera pool` subcommand as a first-class, general-purpose capability.

## Use case 6 — Resume

```
vera resume --config output/c_sonnet/<run>/config.json
```

Reads the immutable `config.json` (verifying its `.sha256` sidecar first) plus the adjacent `state.json`, determines what remains incomplete, and continues. Built in from the start, not retrofitted.

`vera resume` works identically regardless of which stage was interrupted — `generate`, `judge`, or a full `pipeline` — since it reads whichever `state.json` the target run directory has, not stage-specific logic.

## Config mechanism

- `--config <path>` — JSON file, for local use.
- `--config -` — read JSON from stdin.
- `VERA_RUN_CONFIG` env var — inline JSON content, for remote/CI dispatch where uploading or mounting a file isn't convenient.
- **CLI flags and `--config` are strictly either/or, never combined for the same run.** A given piece of information (model selection/repeats, sampling knobs, persona/rubric lists) is supplied via one or the other, never both — the implementation rejects the combination rather than silently merging. **`--sample <N>` (see [Use case 4](#use-case-4--smoke-test)) is the one deliberate exception:** it MAY be passed alongside `--config`, since it never carries information `config.json` defines -- it only caps how much of the already-resolved lists get used for that invocation, purely for debugging, and it's never written into the run's own `config.json` artifact.
- Internally, `--config` always resolves to the same canonical flag-set the CLI would produce, so there is exactly one resolved form regardless of input path. The tool prints this resolved form at run start for terminal/CI-log visibility (it does not write to the shell's own history — an opt-in `--print` flag emits the resolved flag-string with no execution, for a caller who wants to `eval` it into their own shell explicitly).
- JSON, not YAML — robust when passed as a one-line env var or stdin payload with no escaping ambiguity.
- **Path fields inside `config.json` (`generation.personas`, etc.) resolve relative to `$ROOT`** — the directory containing `vera.py` — never relative to the current working directory the CLI was invoked from, and never relative to `config.json`'s own location. This is a single rule regardless of how the config arrives (`--config <file>`, `--config -`, or `VERA_RUN_CONFIG`), so a config's meaning never depends on where your shell happens to be or where you saved the file. This is distinct from the [rubric bundle manifest](./architecture.md#rubric-bundle-manifest), which deliberately resolves relative to *itself* instead, so a manifest folder stays portable across checkouts — `config.json` doesn't need that property, since it's checkout-specific by nature.

### `config.json` shape

Top-level `generation` and `judging` blocks are **completely orthogonal** — model selection for one must never influence or be influenced by the other. Model-list fields (a list, not an object keyed by name, so the same model can appear twice with different knobs) are named per entity rather than a bare `models`, since `generation` has two LLM roles (`chatbot`, `user`) competing for that name; `judging` keeps `models` since only one LLM role exists there:

```json
{
  "generation": {
    "chatbot": {"name": "claude-sonnet-2026xxxx", "repeats": 1},
    "user": [
      {"name": "claude-sonnet-2026xxxx", "repeats": 1, "temperature": 0.7},
      {"name": "gpt-5", "repeats": 2}
    ],
    "personas": ["data/personas_a.json", "data/personas_b.json"],
    "turns": 3,
    "output": "output",
    "max_concurrent": null,
    "max_total_words": null,
    "persona_speaks_first": true,
    "sessions": null,
    "persona_context_template": "data/SI/persona_context_template.txt"
  },
  "judging": {
    "conversations": ["output/c_sonnet/example/conversations"],
    "models": [
      {"name": "claude-sonnet-2026xxxx", "repeats": 1}
    ],
    "rubrics": [
      {"name": "SI"},
      {"name": "PHQ9", "models": [{"name": "gpt-5", "repeats": 2}]}
    ]
  }
}
```

`judging.conversations` is required for standalone `vera judge` configs. A
`pipeline` config omits it because the generation stage supplies the resolved
conversation paths without combining config input with a CLI flag.

These generation behavior fields are required in config mode, including
explicit `null` where no limit or session list is intended. Config mode never
inherits the CLI defaults. When top-level `target` supplies the persona bundle,
`persona_context_template` is `null` and the selected manifest must define
`persona_context_template_file`.

`generation.chatbot` is the chatbot under test — same shape as one entry in `generation.user`, but a single object, not a list (only one chatbot per run; see use case 1). It is distinct from `generation.user`, which is the user-side (`u`) LLM list — the two share a field shape but are never conflated: naming one `chatbot` and the other `user` (rather than both `models`) makes which is which unambiguous at the field-name level, not just from prose.

Each list entry's `name` is always a **specific model identifier** (e.g. `claude-sonnet-2026xxxx`), using the provider's own naming — never a bare provider name like `"openai"`. Bespoke sampling knobs (temperature, top_p, max_tokens) are config-only, never expressible via `-u`/`-j` shorthand; a model named only via the shorthand gets the provider's environment-sourced defaults. Provider connection details (endpoint, API version, region) stay env-sourced only, never overridable here.

## Per-run artifacts

- **`config.json`** — immutable copy of the resolved config as actually used, written once at run start, never modified. Records both the *requested* model identifier and the *actual-resolved* one returned by the provider (relevant when aliases like "latest" resolve to a dated model).
- **`config.json.sha256`** — sidecar checksum (hash lives outside the file it hashes, avoiding self-reference). Computed exactly once, by exactly one function; that single value is what both this sidecar's content and the run-id folder name's `<sha>` component contain — never two independent computations that could drift apart. The hash never gets embedded in `config.json`'s own filename — that would put the same value in a third place without adding any integrity benefit, since a corrupted file wouldn't automatically stop matching its own filename.
- **`state.json`** — separate, mutable file tracking run progress (completed items, errors, output paths so far). The only file `vera resume` writes to.
- **Folder-already-exists behavior:** error out (no overwrite, no auto-suffix), as the default for now.

## Naming

`p_` is retired — too generic. The u/c/j vocabulary applies at both the run-folder and individual-file level:

```
output/
  c_sonnet/                                                 <- persistent per-chatbot directory
    <nickname>_<timestamp>_<sha>/                            <- e.g. prophetic-bullfrog_20260713-1530_a1b2c3
      conversations/
        u_<persona-file>_<persona-name>_c_sonnet.json
      evaluations/
        <rubric_name>/
          j_claude_<nickname>_<timestamp>_<sha>/            <- judge stays flat, no persistent per-judge-model parent
            results.csv

  evaluations/<config-sha256>/                                <- persistent per-config directory, same role as c_sonnet/
    <rubric_name>/                                             <- standalone judging (vera judge run on its own)
      j_claude_<nickname>_<timestamp>_<sha>/                   <- same run-root shape as the nested case -- re-runs don't collide
        results.csv
```

- **Generation** groups persistently by chatbot model (`c_sonnet/` accumulates every run against that model).
- **Judging** stays flat per-run — intentionally asymmetric, not an inconsistency.
- Every run-id is `<nickname>_<timestamp>_<sha>` when nested under a per-model parent (model already given by `c_sonnet/`), or `<model>_<nickname>_<timestamp>_<sha>` when flat (`j_claude_...`, nothing else names the model): readable (which model, where applicable), recognizable (`<nickname>` — a generated human-memorable tag, purely so a person can refer to a run without quoting a sha; carries no identity of its own and is never a substitute for it), ordered (when), integrity-checked (sha256 of `config.json`). The nickname never needs to encode which model was used — the surrounding path already does that.
- Standalone judging (against one or many existing folders, not chained from a pipeline run) has no single parent run to nest under, so it groups under its own top-level directory — `evaluations/<config-sha256>/` — sibling to the `c_*` directories. **This container is not a run-root**, exactly like `c_sonnet/` isn't one for generation: it exists purely so every run of one exact config is discoverable in one place, and it accumulates runs rather than being collision-checked itself. The run-root actually created and checked for collisions is still the `j_claude_<nickname>_<timestamp>_<sha>/` folder inside it — so re-invoking `vera judge` standalone with an identical config produces a new run alongside prior ones, never an error and never a silent overwrite, for the same reason re-running generation against the same chatbot never collides.

All naming/layout construction logic MUST live in a single `utils/` module (extending `utils/conversation_layout.py`), never duplicated across `generate/`, `judge/`, or `score/` handlers — this scheme has already changed multiple times during design and is expected to keep evolving.
