# How `vera pipeline` resolves its input

`vera pipeline` runs generation, then judging, then scoring, feeding each
stage's output into the next. That one property—the stages are
chained—makes its input different from every other command's, and this
document explains the difference, because it looks like a schema
inconsistency when you meet it in the code or in a config file.

For what the command is and how to run it, see
[`vera_cli/README.md`](../vera_cli/README.md) and
[`vera-cli-use-cases.md`](./vera-cli-use-cases.md). For the layering rule this
command follows (three stages, pooling is its own subcommand), see
[`architecture.md`](./architecture.md).

## The stages are chained, so some values cannot exist yet

A single-stage command is given everything it needs:

| Command | Needs | Where it comes from |
|---|---|---|
| `vera generate` | personas, models, turns | the caller |
| `vera judge` | **a conversations folder**, rubric, models | the caller |
| `vera score` | **a `results.csv`** | the caller |

For a pipeline, the two bold values are produced *by the run itself*:

```
generate ──▶ <run>/conversations ──▶ judge ──▶ <evaluation>/results.csv ──▶ score
```

Nobody can state `judging.conversations` before the pipeline starts, because
the folder does not exist and its name—which encodes the models, the target
and a timestamp—is minted by generation. The same is true of
`scoring.results`.

So a pipeline config states **fewer** fields than a judge config, not more, and
`judging.conversations` is rejected rather than accepted-and-overridden:
stating it is a claim about the run that cannot be true.

```jsonc
{
  "target": "SI",
  "generation": { ... },          // complete: fully knowable up front
  "judging": {                     // complete except `conversations`
    "models": [ ... ],
    "max_concurrent": 10,
    "per_judge": false
  },
  "scoring": { "personas": "data/SI/personas.tsv" }   // optional; no `results`
}
```

`--print` round-trips this form: it emits a valid `vera pipeline --config`
document, which is deliberately *not* a valid `vera judge` one.

## How the code represents it: specs and configs

`vera_cli/pipeline.py` resolves every field it can before any model is
called, so a bad rubric, an unknown target or a malformed `max_concurrent`
fails before any budget is spent. What it cannot resolve, it holds as a
*spec*:

```
JudgingSpec     models, rubrics, max_concurrent, per_judge
  └─ JudgingConfig  + conversations, output

ScoringSpec     personas, skip_risk_analysis
  └─ ScoringConfig  + results, output
```

Each config is a subclass of its spec, adding the fields that name concrete
folders. That direction is deliberate:

- **A config is always complete.** `JudgingConfig` still requires
  `conversations`; nothing was made optional, so `vera judge` cannot be
  handed an incomplete config and never has to re-check.
- **A config can stand in for a spec, never the reverse.** A spec lacks
  fields, so a spec cannot be passed off as something runnable.
- **`complete` turns a spec into a config.** `PipelineRun` holds a
  generation `RunConfig`, a `JudgingSpec` and a `ScoringSpec`; once
  generation has produced a run folder, `_judge_and_score` calls
  `judging.complete(conversations=..., output=...)`, and once judging has
  written `results.csv`, `scoring.complete(results=..., output=None)`.

`PipelineRun.to_dict` therefore delegates to all three objects, and its
output is the pipeline config shown above.

Two alternatives were considered and rejected. Making `conversations` and
`results` optional on the configs themselves would delete the most code, but
it leaves a config that does not declare a field it needs, with the
requirement restated later in each command. A flag on the config (say,
`is_pipeline`) that relaxes which fields are required has the same problem
with an extra branch: the type no longer tells you whether `conversations`
is there, so every consumer has to check the flag first.

## Deferred: `--target all`

`vera pipeline --target all` errors, as `vera judge --target all` does. A
pipeline has no attribution problem—each target would generate its own
conversations, so each evaluation would land under its own run folder—but
supporting it means a second rubric-resolution path that walks back from a
run's persona file to the target that produced it, and no use case has asked
for it. It can be widened later without breaking any existing invocation.
