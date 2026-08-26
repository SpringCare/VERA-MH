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

**These letters are `vera.py`-only and are not the same flags as the legacy
scripts.** `generate.py`, `judge.py`, and `run_pipeline.py` are legacy entry
points in the process of being removed: each is deleted once its replacement
`vera.py` command ships, and they exist only to keep unmigrated workflows
running in the meantime. Do not build on them.

`judge.py` is the one that lingers. `vera judge` ships without `--resume` because
the resume contract is deferred, so `judge.py` stays as the *only* way to resume
an evaluation until `vera resume` exists — and for nothing else. It is deleted
when `vera resume` lands.

They already use `-c`/`-r` for unrelated things (`-c` is `--max-concurrent` in
`generate.py` and `--conversation` in `judge.py`; `-r` is `--runs` in
`generate.py` and `--rubrics` in `judge.py`). `vera.py` intentionally repurposes
them for the `u`/`c`/`j` vocabulary above. The two meanings never need to be
told apart at runtime, because you invoke either `vera.py` or a legacy script
explicitly.

A **target** is separate from the u/c/j entities: it is a reusable evaluation
bundle containing a rubric, personas, and all generation/judging prompts. A
target is represented by `data/<target>/manifest.json`.

## Minimum required arguments

CLI shorthand and input `config.json` resolve to the same canonical form. CLI
component selectors use target names for convenience; explicit config fields
may state the resulting concrete paths directly.

| Subcommand | CLI shorthand — minimum required | Input `config.json` — minimum required fields |
|---|---|---|
| `generate` | `-c <chatbot>` + `-u <model:repeats...>` (≥1) + (`--target <name-or-path>` **or** explicit `--personas <name-or-manifest-path>`) | `generation.chatbot`, `generation.user` (≥1), and (top-level `target` **or** explicit `generation.personas` + `generation.persona_context_template`) |
| `judge` | `-j <model:repeats...>` (≥1) + `--conversations <folder>` (exactly 1 — see below) + (`--target <name-or-path>` **or** explicit `--rubric <name-or-manifest-path>`) | `judging.models` (≥1), `judging.conversations` (list, length 1), and (top-level `target` **or** explicit `judging.rubrics` (≥1)) |
| `pipeline` | everything `generate` needs **and** everything `judge` needs — or just `-c`, `-u`, `-j`, `--target <name>` (`--target` covers persona+rubric in one shot) | both `generation` and `judging` blocks fully populated, or `-c`/`-u`/`-j`-equivalent fields plus a top-level `target`; `judging.conversations` is omitted because generation supplies it |
| `score` | the results path (`-r <results.csv>`) — nothing else | n/a — `score` reads an existing `results.csv`, not a run config |
| `pool` | `--evaluations <folder...>` (≥1) | n/a |
| `resume` | `--config <run's own config.json>` — that's the entire invocation | n/a (it *is* the config being resumed) |

**`target` mirrors `--target` exactly, including its mutual-exclusivity rule:**
a top-level `target` field in the input `config.json` resolves one complete target
manifest and expands its personas, persona prompt, rubric, and judging prompts.
Setting `target` alongside explicit `generation.personas` or
`judging.rubrics` is an error — rejected outright, not silently merged or
overridden. The run's own immutable `config.json` always stores the fully
expanded form, so `vera resume` never re-resolves a manifest that might have
changed.

Whole-target selection is optional. Callers who want independent control may
use `--personas HFO` for the persona side and `--rubric SI` for the rubric side;
a pipeline can use both explicit forms instead of `--target`.

## Use case 1 — End-to-end test of one LLM

Generate -> judge -> score chained, for exactly one chatbot under test, in a single invocation.

```
vera pipeline --config run.json
```

**Resolved:** stays single-chatbot-per-invocation. Comparing chatbots is always an external loop over single-chatbot pipeline runs, consistent with use case 2. Native multi-chatbot support (one combined comparison report) is not built now — flagged as a possible future addition if a real need emerges, not ruled out permanently.

**`--target <name-or-path>` shorthand:** `vera pipeline --target SI` resolves
`data/SI/manifest.json` and selects its personas, persona prompt, rubric, and
judging prompts in one shot — the common case of “run the canonical test for
SI.” It never selects the chatbot. Explicit `--personas` and `--rubric` remain
available when the caller deliberately wants different components.

**No implicit "run everything":** if neither `--target` nor explicit
`--rubric`/`judging.rubrics` is given where judging is requested, the CLI errors.
`--target all` deliberately resolves every discovered target manifest as a
separate invocation. It never merges personas or prompts from different targets.

**`--conversations` takes exactly one folder.** This preserves today's
`judge.py --folder` behavior, which accepts a single conversation run folder
(nested `p_*__/conversations/`, or a legacy flat folder). The flag name and the
`judging.conversations` config field are nonetheless **list-shaped from day one**,
following the same reasoning as [AD-20](./ARCHITECTURE-SPINE.md#ad-20--judgingrubrics-is-a-list-from-day-one)
for `judging.rubrics`: only a length-1 list is accepted or validated, but locking
a scalar shape now would force a breaking config change if multi-folder judging
ever ships. More than one folder is a clear error, not a silent truncation. Note
that judging several folders is already expressible today — judge each, then
combine with `vera pool`.

**`vera judge --target all` is deferred, not disallowed.** It errors for now and
is scheduled for Phase 4; `--target all` keeps its full meaning for `generate`
throughout.

The blocker is output attribution, not semantics. Judging every target means
evaluating the same conversations under N rubrics, which resolves cleanly to N
separate length-1 runs — but until Phase 4 adds the `evaluations/<target>/`
segment, all N land in `<gen_run>/evaluations/j_*` distinguishable only by
timestamp, because the judge run folder encodes judge model and time, not the
rubric that produced it. Erroring until the path can attribute the result is
preferable to writing output nobody can later tell apart. Phase 4 lifts this
alongside multi-rubric support, and the error message should say so.

## Use case 2 — Batch generate across personas

One chatbot under test, generated against multiple personas, each carrying its own user-side LLM.

Two independent, equally valid ways to run this — not a two-step sequence:

**Option A — config file:**
```
vera generate --config run.json
```

**Option B — CLI shorthand:**
```
vera generate -c sonnet -u gpt:1 sonnet:2 --personas SI
```
Bare-minimum required flags for Option B: `-c <chatbot>` (the chatbot under
test — no default), `-u <model:repeats...>` (at least one user-side model), and
either a complete `--target` or explicit `--personas <target>`. `--personas` is
a supported first-class path, not only a compatibility fallback; it selects the
persona files and persona prompt from that target's manifest.

Generation behavior is also controlled at this input boundary. CLI invocations
default to `--turns 30`, `--output output`, unlimited concurrency, no total-word
cap, persona-first ordering, and one unnamed session. The explicit persona
component includes the files and context template resolved from the target named
by `--personas`; `--target` resolves the same fields while also selecting the
rubric side. The generation runner itself has no parameter defaults.

Or, using `--target` to select the complete target manifest instead of naming
the generation inputs explicitly:
```
vera generate -c sonnet -u gpt:1 --target SI
```

`-u <model>:<repeats> ...` selects the user-side LLM(s) and how many full passes over the configured persona set to run with each. With 10 personas configured, `-u gpt:1 sonnet:2` means 10 conversations with gpt, 20 with sonnet (2 full passes). Each `(model, repeat)` pass gets its own run under that model's persistent output directory (see Naming below).

Comparing across *chatbots* is done by looping this invocation externally, once per chatbot — not a built-in cross-product flag.

Personas come from one or more persona files, each containing multiple personas; duplicate persona names across files are possible (disambiguated by file + name, see Naming).

## Use case 3 — Judge existing conversations

Judge one **or more** existing transcript folders using either a complete target
or an explicitly selected rubric. Each rubric has a default judge-LLM set,
overridable per rubric. Multiple judges per rubric are supported. Judging remains
decoupled from generation.

```
vera judge --config run.json
vera judge -j claude:1 --conversations output/c_sonnet/<run>/conversations/ --target SI
vera judge -j claude:1 gpt:2 --conversations output/c_sonnet/<run>/conversations/ --rubric SI
```

No `-c` here: judging is decoupled from chatbot selection by design (see the orthogonality invariant above) — the chatbot is already implicit in whichever `--conversations` folder is passed in.

`-j <model>:<repeats> ...` mirrors `-u`'s syntax for the judge side. `repeats` here means re-running the same transcript through the same judge model N times, to measure judge consistency/variance.

`--target` consumes the rubric and judging prompts from the selected complete
[target manifest](./architecture.md#target-manifest). Explicit `--rubric SI`
consumes only SI's rubric and judging prompts and does not select SI's personas
or persona prompt. Both flags accept a target name or manifest path, never a bare
TSV with implicitly assumed sibling prompts.

For standalone `judge`, these are two ways to word the same effective request:

```text
--target SI  → select SI, from which judge consumes the rubric and judge prompts
--rubric SI  → explicitly select SI's rubric and judge prompts
```

They resolve to identical judging inputs. The distinction becomes meaningful
for `pipeline`, where `--target` also supplies the generation personas and
persona prompt, while `--rubric` supplies only the judging component.

**Resolved (multi-folder judge output):** judge keeps results independent per folder; the `score/` layer aggregates across folders when needed, not judge itself. There are also in-between options — e.g. kept separate, but the score layer aggregates them.

**Resolved (multi-rubric output layout):** generated conversations for a run all live together in one unified folder regardless of which rubrics will later judge them — generation has no knowledge of rubrics. Judging output IS separated per rubric, since different rubrics produce different, non-comparable scores. (The unified-conversations choice is soft, not a hard invariant — may change if a concrete need for per-rubric conversation grouping emerges.)

## Use case 4 — Smoke test

Run the full pipeline (or generate/judge alone) against a small sample of personas/rubrics/judges, to sanity-check that a config works end-to-end before spending the full LLM-call budget.

```
vera pipeline --config run.json --sample 2
```

`--sample N` caps the config's full persona (and rubric/judge, where relevant)
list at run time. This avoids hand-maintaining a separate small-scale config
just for smoke testing.

**`--sample` is the sole behavior-altering exception to the CLI/`--config`
either-or rule** (AD-17 in
[ARCHITECTURE-SPINE.md](./ARCHITECTURE-SPINE.md)): it caps how much of the
already-resolved lists get used rather than replacing those lists. The executed
value is recorded in the run's persisted `config.json` invocation metadata, so
`vera resume` retains the same sampled scope. `--debug` is recorded alongside
it; `--print` creates no run and therefore has nothing to persist.

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
- **Run-defining CLI flags and `--config` are strictly either/or, never combined for the same run.** A given piece of information (model selection/repeats, sampling knobs, persona/rubric lists) is supplied via one or the other, never both — the implementation rejects the combination rather than silently merging. Invocation controls `--sample`, `--debug`, and `--print` MAY accompany config input. Executed runs persist `sample` and `debug` under `invocation` in their immutable `config.json`; `--print` exits without creating a run.
- Internally, `--config` always resolves to the same canonical flag-set the CLI would produce, so there is exactly one resolved form regardless of input path. The tool prints this resolved form at run start for terminal/CI-log visibility (it does not write to the shell's own history — an opt-in `--print` flag emits the resolved flag-string with no execution, for a caller who wants to `eval` it into their own shell explicitly).
- JSON, not YAML — robust when passed as a one-line env var or stdin payload with no escaping ambiguity.
- **Path fields inside `config.json` (`generation.personas`, etc.) resolve relative to `$ROOT`** — the directory containing `vera.py` — never relative to the current working directory or the config file. A [target manifest](./architecture.md#target-manifest) deliberately resolves its fields relative to its own directory so the complete target remains portable.

For example, given `configs/run.json` and `data/SI/manifest.json`:

```text
configs/run.json:       "personas": ["data/SI/personas.tsv"]
                                    └─ resolves from $ROOT

data/SI/manifest.json:  "personas": ["personas.tsv"]
                                    └─ resolves from data/SI/

Both resolve to:         $ROOT/data/SI/personas.tsv
```

The different anchors let a run config remain checkout-relative while a target
directory remains portable as a unit.

### `config.json` shape

The persisted run artifact includes generated `invocation` metadata in addition
to the resolved generation and judging inputs. A fresh input config does not
need to author this block: the CLI records the effective `--debug` and
`--sample` values when it creates the run. `vera resume` reads those persisted
values and retains the same invocation scope.

Top-level `generation` and `judging` blocks are **completely orthogonal** — model selection for one must never influence or be influenced by the other. Model-list fields (a list, not an object keyed by name, so the same model can appear twice with different knobs) are named per entity rather than a bare `models`, since `generation` has two LLM roles (`chatbot`, `user`) competing for that name; `judging` keeps `models` since only one LLM role exists there:

```json
{
  "invocation": {
    "debug": false,
    "sample": null
  },
  "generation": {
    "chatbot": {"name": "claude-sonnet-2026xxxx", "repeats": 1},
    "user": [
      {"name": "claude-sonnet-2026xxxx", "repeats": 1, "temperature": 0.7},
      {"name": "gpt-5", "repeats": 2}
    ],
    "personas": ["data/SI/personas.tsv"],
    "turns": 30,
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

In an input config, `judging.rubrics` is the explicit component-selection path:
each entry identifies a target name or manifest and consumes only its rubric and
judging prompts. In the persisted canonical config, those entries contain the
resolved concrete paths rather than a manifest reference.

These generation behavior fields are required in config mode, including
explicit `null` where no limit or session list is intended. Config mode never
inherits the CLI defaults. Top-level `target` expands a complete manifest into
the concrete generation and judging fields before the canonical config is
printed or persisted. Explicit `generation.personas` and `judging.rubrics`
remain supported when no top-level target is set.

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
