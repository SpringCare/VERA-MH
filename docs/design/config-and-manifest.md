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
- **Scoring knobs move here from code:** the risk-level column and its order, and the label charts and CSV headers put on scores (today the constant `SCORE_NAME = "VERA-MH v1.2"` in `score/utils.py`), which becomes the manifest's `name` and `version`. Scoring reads them from the run record, which names the target that produced the results. (The roadmap's [Rubric-agnostic scoring](../roadmap.md#rubric-agnostic-scoring) later replaces the persona column with a rubric question; the manifest is where that pointer lives either way.)
- **`name`, `description`, `version`** identify the evaluation for people. `version` is free text set by the rubric's authors ("1.2"); it is helpful but not trusted. The run record adds what is: the checksum of every file and the git commit the target folder was last changed in (see below), so a score always says exactly which edition of a rubric produced it.
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

- **Defaults allowed.** A config states what differs from the defaults, and the run record writes every value out explicitly, defaults included. Reproducibility comes from the run record, not from forcing every config file to be complete.
- **Execution settings in their own section:** `max_concurrent`, `debug`, `sample`, `into`, `allow_uncommitted`. The run record writes them out like everything else, in case they're useful later, but they are excluded from the run's hash, because they don't change the result.
- **Model names checked at load.** Every model name is looked up with its provider before anything runs, so a placeholder or a typo fails in seconds, naming the field, instead of in every conversation. This needs network and keys when a run starts, which every real run needs anyway; `--print` skips the check.
- **One path rule:** paths in a config file resolve against the config file's own folder, like the manifest's.

### Run record: what ran

The run record is the run config **resolved**: every default filled in, every path absolute, every model ID as sent, plus a `provenance` section.

```json
{
  "format_version": 1,
  "target": {
    "name": "SI", "version": "1.2",
    "git_commit": "65566744…", "uncommitted_changes": false,
    "files": {"rubric.tsv": "sha256:…", "personas.tsv": "sha256:…"}
  },
  "generation": {"…": "every field, with defaults written out"},
  "judging": {"…": "…"},
  "provenance": {"vera_version": "2.1.0", "git_commit": "…", "started_at": "…", "models_resolved": {"gpt-5.4": "gpt-5.4-2026-08-01"}}
}
```

- **Hashed:** everything except `provenance` and the execution section. Same hash, same experiment (run-lifecycle.md).
- **Input files by checksum, plus the commit.** Each file's checksum is the source of truth: editing a rubric or personas file changes the hash, so two runs are never mistaken for the same experiment when their inputs differed. Next to it, `git_commit` is the last commit that changed the target folder, so a reader can open that exact rubric on GitHub, and `uncommitted_changes` says whether the files on disk differed from that commit; a run only starts with it `true` when `--allow-uncommitted` was given (decision 5). The checksums are hashed; the commit is provenance and isn't.
- **Code version recorded, not hashed,** so a docs-only commit doesn't make every run look like a different experiment.
- **Requested vs. resolved model:** the requested ID is part of the settings (hashed); what the provider reports back goes in `provenance` (not hashed). This settles the disagreement run-lifecycle.md found between architecture.md and the use-cases doc.
- **`--config <run>/config.json` re-runs it.** A run record is a valid run config, so reproducing a run is one command.

## Decisions from review

1. **Defaults are allowed in config files,** and the run record writes every value out explicitly.
2. **Execution settings are recorded but not hashed,** in their own section.
3. **The manifest `version` is human text, and checksums are the source of truth.** The run record holds both, plus the git commit of the target folder.
4. **Model names are validated with the provider at load,** to catch problems before a run spends anything.
5. **Uncommitted rubric edits stop a run.** If the target's files differ from their last commit, `vera` refuses to start and names the changed files, unless `--allow-uncommitted` is given. That flag is an execution setting: recorded, not hashed. The run record then has `uncommitted_changes: true`, so a run made that way is always identifiable. This protects published numbers from coming from a rubric nobody can look up.

## Implementation plan

Each step is its own PR into `release/2.1`. None changes scores, so the [safety net](../safe-changes.md) stays green throughout.

| | PR | What changes | Depends on |
|---|---|---|---|
| A | Manifest: judge-only targets and identity | Rubric fields stay required; persona fields are required only to generate. Optional `name`, `description`, `version`, validated. | — |
| B | Config defaults allowed | Config files may leave out behavior fields, which take the CLI defaults. | — |
| C | The run record | Each run folder gets `config.json`: the resolved config with every default written out, input-file checksums, and `provenance`; plus its `.sha256`. Written into today's run folders; [run-lifecycle.md](./run-lifecycle.md) later moves where it lives. | B |
| D | Uncommitted rubric edits stop a run | Refuse to start unless `--allow-uncommitted`; the record then says `uncommitted_changes: true`. | C |
| E | Execution section | `max_concurrent`, `debug`, `sample`, `into`, `allow_uncommitted` move to `execution`, outside the hash. The old `generation.max_concurrent` keeps working with a deprecation warning. | C |
| F | Scoring knobs from the manifest | The risk-level column and order, and the chart label (`SCORE_NAME`), come from the manifest, found through the run record. | C |
| G | Model names checked at load | Each model name is looked up with its provider before anything runs; `--print` skips it. | — |

A and B first (independent), then C, then D, E, F. G can land any time.

