# How `vera pipeline` resolves its input

`vera pipeline` runs generation, then judging, then scoring, feeding each
stage's output into the next. That one property — the stages are chained —
makes its input different from every other command's, and this document
explains the difference, because it looks like a schema inconsistency when you
meet it in the code or in a config file.

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
the folder does not exist and its name — which encodes the models, the target
and a timestamp — is minted by generation. The same is true of
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

## What that costs in the code today

`vera_cli/pipeline.py` resolves in two halves. Everything knowable before a
model is called still is — a bad rubric or an unknown target fails before any
budget is spent — and the `JudgingConfig` is constructed per generated run
folder inside `_execute`, once the folder exists.

The visible cost is that `PipelineRun` cannot simply hold three config objects.
It holds a complete generation `RunConfig` plus the judging and scoring fields
as loose values, because `JudgingConfig` requires `conversations` and
`ScoringConfig` requires `results`. That is why `PipelineRun.to_dict`
hand-builds the judging and scoring sections instead of delegating, and why
`_judge_and_score` assembles a `JudgingConfig` from parts.

**Known follow-up.** Making `JudgingConfig.conversations` and
`ScoringConfig.results` optional would collapse `PipelineRun` to three real
config objects and delete all three of those seams. The open question is what
that does to `vera judge`: an optional field in the shared dataclass must not
become a way to hand `judge` an incomplete config. The answer is to move the
requirement to the command that has it — `vera judge` validates that
`conversations` is present when it resolves input, the same place it already
rejects `--target all` and derives its output folder — so the dataclass
describes what the type can hold and the command describes what that command
demands. Until that is done, the loose fields stay.

## Deferred: `--target all`

`vera pipeline --target all` errors, as `vera judge --target all` does. A
pipeline has no attribution problem — each target would generate its own
conversations, so each evaluation would land under its own run folder — but
supporting it means a second rubric-resolution path that walks back from a
run's persona file to the target that produced it, and no use case has asked
for it. It can be widened later without breaking any existing invocation.
