# The three JSON files: manifest, config, run record

**Status:** draft for discussion. Connected to [run-lifecycle.md](./run-lifecycle.md) (`config.json` and its hash), [output-layout.md](./output-layout.md) (where the run record lives), and [create-an-evaluation.md](../create-an-evaluation.md) (what someone writing a target edits).

## Three files, three jobs

| File | Who writes it | What it says | Example |
|---|---|---|---|
| **Target manifest** | Whoever builds an evaluation | What the evaluation is: its personas, rubric, and prompts | `data/SI/manifest.json` |
| **Run config** | Whoever runs an evaluation | How to run it: which models, how many turns, which target | `configs/recommended-SI.json`, `--config` |
| **Run record** | `vera`, at the start of every run | Exactly what ran: the config with every default filled in, plus provenance | `config.json` in the run folder (not written yet) |

The first two exist today and have grown separately; the third is planned in run-lifecycle.md. This doc settles what belongs in each, so that each fact has one home.

## Problems today

**Manifest**

- **Judging-only targets aren't possible.** All five fields are required, and every file must exist, so a target meant only for judging real conversations still has to ship a personas file and a context template it never uses.
- **Scoring knobs live in code, not the manifest.** The risk-level column (`Short Current Suicide Risk Level`) and its order (`None`, `Low`, `High`, `Immediate`) are hardcoded in scoring, so any other target's risk breakdown comes back empty. `persona_annotation_columns` is the one scoring-related field that *is* in the manifest.
- **No identity.** There is no name, description, or version, so nothing records which edition of a rubric a run used; the folder name stands in.
- **A legacy copy.** `rubric_manifest.json` is a symlink kept for the old scripts.

**Run config**

- **Every behavior field is required.** That makes each config file a complete description, but adding any option breaks every existing config until it's edited: adding `generation.user_language` (#269) meant touching all four files in `configs/` and every doc example.
- **Execution settings mixed with run-defining ones.** `max_concurrent` sits in `generation` and `judging` next to `turns` and the models, though it changes how fast a run goes, not what it produces. Only `debug`, `sample`, and `into` are kept apart, in `InvocationConfig`.
- **Placeholders that only fail at run time.** `recommended-SI.json` ships with `"REPLACE-WITH-THE-MODEL-UNDER-TEST"` as the chatbot; it loads fine and every conversation then fails ("Unsupported model").
- **Two ways to pass the same thing.** A config file, or `VERA_RUN_CONFIG` (what `--print` emits), with different path rules from CLI flags (config paths resolve against the repo root, CLI paths against the working directory).

**Run record**

- **Doesn't exist yet.** Nothing persists what a run actually used, so reproducing one means reconstructing it from folder names and logs.

## Proposal

### Target manifest: what the evaluation is

```json
{
  "name": "SI",
  "description": "Suicidal ideation",
  "version": "1.2",
  "rubric_file": "rubric.tsv",
  "rubric_prompt_beginning_file": "rubric_prompt_beginning.txt",
  "question_prompt_file": "question_prompt.txt",
  "personas": ["personas.tsv"],
  "persona_context_template_file": "persona_context_template.txt",
  "persona_annotation_columns": ["Short Current Suicide Risk Level"],
  "risk_level": {"column": "Short Current Suicide Risk Level", "order": ["None", "Low", "High", "Immediate"]}
}
```

- **Rubric fields required; persona fields required only to generate.** `vera judge` accepts a manifest without personas; `vera generate` and `vera pipeline` say which field is missing.
- **Scoring knobs move here from code:** the risk-level column and its order. (The roadmap's [Rubric-agnostic scoring](../roadmap.md#rubric-agnostic-scoring) later replaces the persona column with a rubric question; the manifest is where that pointer lives either way.)
- **`name`, `description`, `version`** identify the evaluation. The run record copies them, so a score always says which edition of a rubric produced it.
- `rubric_manifest.json` is removed with the legacy scripts.

### Run config: how to run it

```json
{
  "target": "SI",
  "generation": {
    "chatbot": {"name": "gpt-5.4"},
    "user": [{"name": "gpt-5.2"}, {"name": "claude-opus-4-5"}],
    "turns": 30
  },
  "judging": {"models": [{"name": "gpt-5.4", "reasoning_effort": "low"}]},
  "execution": {"max_concurrent": 10}
}
```

- **Defaults allowed.** A config states what differs from the defaults. Reproducibility comes from the run record, which writes every value out, not from forcing every config file to.
- **Execution settings in their own section:** `max_concurrent`, `debug`, `sample`, `into`. They are excluded from the run's hash, because they don't change the result.
- **Placeholders rejected at load.** A model name that no provider recognizes fails before anything runs, naming the field.
- **One path rule:** paths in a config file resolve against the config file's own folder, like the manifest's.

### Run record: what ran

The run record is the run config **resolved**: every default filled in, every path absolute, every model ID as sent, plus a `provenance` section.

```json
{
  "format_version": 1,
  "target": {"name": "SI", "version": "1.2", "files": {"rubric.tsv": "sha256:…", "personas.tsv": "sha256:…"}},
  "generation": {"…": "every field, with defaults written out"},
  "judging": {"…": "…"},
  "provenance": {"vera_version": "2.1.0", "git_commit": "…", "started_at": "…", "models_resolved": {"gpt-5.4": "gpt-5.4-2026-08-01"}}
}
```

- **Hashed:** everything except `provenance` and the execution section. Same hash, same experiment (run-lifecycle.md).
- **Input files by checksum.** Editing a rubric or personas file changes the hash, so two runs are never mistaken for the same experiment when their inputs differed.
- **Code version recorded, not hashed,** so a docs-only commit doesn't make every run look like a different experiment.
- **Requested vs. resolved model:** the requested ID is part of the settings (hashed); what the provider reports back goes in `provenance` (not hashed). This settles the disagreement run-lifecycle.md found between architecture.md and the use-cases doc.
- **`--config <run>/config.json` re-runs it.** A run record is a valid run config, so reproducing a run is one command.

## Open questions

1. **Defaults in config files.** The current rule (every field required) was chosen so a config file alone describes a run. With a run record that does that job, are defaults acceptable, or should configs stay complete?
2. **Where execution settings go.** A separate `execution` section, or keep `max_concurrent` per stage and only exclude it from the hash?
3. **Manifest `version`.** Free text set by the rubric's authors, or derived from the files' checksums?
4. **Validating model names at load.** A provider lookup needs network and keys at resolve time; acceptable, or check only for obvious placeholders?
