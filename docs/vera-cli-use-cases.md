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

## Use case 1 — End-to-end test of one LLM

Generate -> judge -> score chained, for exactly one chatbot under test, in a single invocation.

```
vera pipeline --config run.json
```

**Resolved:** stays single-chatbot-per-invocation. Comparing chatbots is always an external loop over single-chatbot pipeline runs, consistent with use case 2. Native multi-chatbot support (one combined comparison report) is not built now — flagged as a possible future addition if a real need emerges, not ruled out permanently.

**`--target <name>` shorthand:** `vera pipeline --target SI` resolves `SI` to one rubric bundle manifest (see [Rubric bundle manifest](../architecture.md#rubric-bundle-manifest)) and sets *both* the generation personas and the judging rubric from it in one shot — for the common case of "run the canonical test for X." This is the one deliberate exception to personas/rubrics being chosen independently; every other invocation (`--rubric` plus separately-specified personas, or explicit `--config` blocks) keeps generation and judging fully orthogonal.

## Use case 2 — Batch generate across personas

One chatbot under test, generated against multiple personas, each carrying its own user-side LLM.

```
vera generate --config run.json
vera generate -u gpt:1 sonnet:2
```

`-u <model>:<repeats> ...` selects the user-side LLM(s) and how many full passes over the configured persona set to run with each. With 10 personas configured, `-u gpt:1 sonnet:2` means 10 conversations with gpt, 20 with sonnet (2 full passes). Each `(model, repeat)` pass gets its own run under that model's persistent output directory (see Naming below).

Comparing across *chatbots* is done by looping this invocation externally, once per chatbot — not a built-in cross-product flag.

Personas come from one or more persona files, each containing multiple personas; duplicate persona names across files are possible (disambiguated by file + name, see Naming).

## Use case 3 — Judge existing conversations

Judge one **or more** existing transcript folders against one or more rubrics. Each rubric has a default judge-LLM set, overridable per rubric. Multiple judges per rubric are supported (for judge-agreement analysis). Judging is decoupled from generation — point it at whatever conversation folders you need, with no enforced coupling to originating personas.

```
vera judge --conversations output/c_sonnet/<run>/conversations/ --config run.json
vera judge -j claude:1 gpt:2
```

`-j <model>:<repeats> ...` mirrors `-u`'s syntax for the judge side. `repeats` here means re-running the same transcript through the same judge model N times, to measure judge consistency/variance.

`--rubric`/`judging.rubrics[]` entries point at a [rubric bundle manifest](../architecture.md#rubric-bundle-manifest) (canonical definition), not a bare `.tsv` path.

**Resolved (multi-folder judge output):** judge keeps results independent per folder; the `score/` layer aggregates across folders when needed, not judge itself. There are also in-between options — e.g. kept separate, but the score layer aggregates them.

**Resolved (multi-rubric output layout):** generated conversations for a run all live together in one unified folder regardless of which rubrics will later judge them — generation has no knowledge of rubrics. Judging output IS separated per rubric, since different rubrics produce different, non-comparable scores. (The unified-conversations choice is soft, not a hard invariant — may change if a concrete need for per-rubric conversation grouping emerges.)

## Use case 4 — Smoke test

Run the full pipeline (or generate/judge alone) against a small sample of personas/rubrics/judges, to sanity-check that a config works end-to-end before spending the full LLM-call budget.

```
vera pipeline --config run.json --sample 2
```

`--sample N` overrides the config's full persona (and rubric/judge, where relevant) list at run time. This avoids hand-maintaining a separate small-scale config just for smoke testing.

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

## Config mechanism

- `--config <path>` — JSON file, for local use.
- `--config -` — read JSON from stdin.
- `VERA_RUN_CONFIG` env var — inline JSON content, for remote/CI dispatch where uploading or mounting a file isn't convenient.
- **CLI flags and `--config` are strictly either/or, never combined for the same run.** A given piece of information (model selection/repeats, sampling knobs, persona/rubric lists) is supplied via one or the other, never both — the implementation rejects the combination rather than silently merging.
- Internally, `--config` always resolves to the same canonical flag-set the CLI would produce, so there is exactly one resolved form regardless of input path. The tool prints this resolved form at run start for terminal/CI-log visibility (it does not write to the shell's own history — an opt-in `--print` flag emits the resolved flag-string with no execution, for a caller who wants to `eval` it into their own shell explicitly).
- JSON, not YAML — robust when passed as a one-line env var or stdin payload with no escaping ambiguity.

### `config.json` shape

Top-level `generation` and `judging` blocks are **completely orthogonal** — model selection for one must never influence or be influenced by the other. Each follows the same models-list pattern (a list, not an object keyed by name, so the same model can appear twice with different knobs):

```json
{
  "generation": {
    "models": [
      {"name": "claude-sonnet-2026xxxx", "repeats": 1, "temperature": 0.7},
      {"name": "gpt-5", "repeats": 2}
    ],
    "personas": ["data/personas_a.json", "data/personas_b.json"]
  },
  "judging": {
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

`models[].name` is always a **specific model identifier** (e.g. `claude-sonnet-2026xxxx`), using the provider's own naming — never a bare provider name like `"openai"`. Bespoke sampling knobs (temperature, top_p, max_tokens) are config-only, never expressible via `-u`/`-j` shorthand; a model named only via the shorthand gets the provider's environment-sourced defaults. Provider connection details (endpoint, API version, region) stay env-sourced only, never overridable here.

## Per-run artifacts

- **`config.json`** — immutable copy of the resolved config as actually used, written once at run start, never modified. Records both the *requested* model identifier and the *actual-resolved* one returned by the provider (relevant when aliases like "latest" resolve to a dated model).
- **`config.json.sha256`** — sidecar checksum (hash lives outside the file it hashes, avoiding self-reference). Computed exactly once, by exactly one function; that single value is what both this sidecar's content and the run-id folder name's `<sha>` component contain — never two independent computations that could drift apart. The hash never gets embedded in `config.json`'s own filename — that would put the same value in a third place without adding any integrity benefit, since a corrupted file wouldn't automatically stop matching its own filename.
- **`state.json`** — separate, mutable file tracking run progress (completed items, errors, output paths so far). The only file `vera resume` writes to.
- **Folder-already-exists behavior:** error out (no overwrite, no auto-suffix), as the default for now.

## Naming

`p_` is retired — too generic. The u/c/j vocabulary applies at both the run-folder and individual-file level:

```
output/
  c_sonnet/                                          <- persistent per-chatbot directory
    <timestamp>_<sha>/
      conversations/
        u_<persona-file>_<persona-name>_c_sonnet.json
      evaluations/
        <rubric_name>/
          j_claude_<timestamp>_<sha>/                <- judge stays flat, no persistent per-judge-model parent
            results.csv

  evaluations/<config-sha256>/<rubric_name>/          <- standalone judging (vera judge run on its own), unchanged
```

- **Generation** groups persistently by chatbot model (`c_sonnet/` accumulates every run against that model).
- **Judging** stays flat per-run — intentionally asymmetric, not an inconsistency.
- Every run-id is `<model>_<timestamp>_<sha>`: readable (which model), ordered (when), integrity-checked (sha256 of `config.json`).
- Standalone judging (against one or many existing folders, not chained from a pipeline run) has no single parent run to nest under, so it uses its own top-level identity — the `config.json` sha256 — sibling to the `c_*` directories.

All naming/layout construction logic MUST live in a single `utils/` module (extending `utils/conversation_layout.py`), never duplicated across `generate/`, `judge/`, or `score/` handlers — this scheme has already changed multiple times during design and is expected to keep evolving.
