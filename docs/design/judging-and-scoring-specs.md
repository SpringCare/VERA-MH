# Judging and scoring specs

Status: Accepted
Date: 2026-09-23

## Context

`vera pipeline` must validate judging and scoring input before generation
spends any budget, but two required fields do not exist yet at that point:
`JudgingConfig.conversations` and `ScoringConfig.results` name folders that the
earlier stages create (see [pipeline.md](../pipeline.md)). `JudgingConfig` also
requires `output`, which the pipeline derives from the same folder.

## Decision

`utils/config_schema.py` splits each of the two sections into a spec and a
config:

| Type | Holds | Complete? |
|---|---|---|
| `JudgingSpec` | `models`, `rubrics`, `max_concurrent`, `per_judge` | no: says *how* to judge |
| `JudgingConfig(JudgingSpec)` | spec + `conversations`, `output` | yes: says *what* to judge |
| `ScoringSpec` | `personas`, `skip_risk_analysis` | no: says *how* to score |
| `ScoringConfig(ScoringSpec)` | spec + `results`, `output` | yes: says *what* to score |

A spec is what can be stated before the run starts. A config is a spec plus
the fields that name concrete folders, and every one of them is required.
`spec.complete(...)` takes exactly the missing fields and returns the config.

Only `vera pipeline` holds specs. `vera judge` and `vera score` are given their
folders up front, so they build configs directly, as before.

The config subclasses the spec, not the reverse: a complete config can stand in
wherever a spec is expected, but a spec can never be passed where a runnable
config is required.

Rejected alternatives:

- **Make `conversations` and `results` optional** and have each command require
  them. The config would stop declaring a field it needs, and the requirement
  would be restated in every command.
- **An `is_pipeline` flag** that relaxes which fields are required. The type
  would no longer say whether `conversations` is present, so every consumer
  would have to check the flag.

## Consequences

- No config-file format changes. Existing `judge`, `score` and `pipeline`
  configs resolve unchanged.
- Pipeline configs now have `judging.max_concurrent` and `judging.per_judge`
  type-checked at resolve time. Before, a bad value failed only after
  generation had finished.
- `JudgingConfig`'s field order changes: inherited fields come first.
  Construction is keyword-only everywhere in the repo, so nothing breaks, but
  positional construction would.
