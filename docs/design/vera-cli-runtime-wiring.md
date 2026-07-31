# VERA CLI runtime wiring

## Context

`vera.py` already defined the Phase 1 command and config shape from the architecture
contract on `feat/VERA_2.0`, but its handlers stopped after parsing. Wiring those
handlers exposed one ambiguity: standalone judging needs conversation paths, while
AD-17 requires an invocation to use either JSON config or run-defining command-line
flags, never both.

## Decision

- `JudgingConfig` owns `conversations`, mirroring `--conversations`.
- Config-sourced paths resolve from the repository root, as required by AD-28.
- `--sample` remains an invocation-only debug cap and is removed from serialized
  `RunConfig`. Per AD-17, it is the sole flag allowed alongside `--config`;
  `--debug` and `--print` remain CLI-only.
- `vera.py` delegates to parser-independent application functions:
  `generate_conversations.run_generation`, `judge.run_judging`, and
  `judge.score.score_results_file`. It never imports or dynamically loads the
  legacy CLI entry points.
- The generation application function accepts resolved persona files, never a
  rubric manifest. `--target` is an orchestration convenience: `vera.py` reads
  its manifest, validates that it defines personas, and passes those paths to
  generation while passing the manifest itself only to judging.
- A manifest's `personas` field remains optional for judge-only use. It is
  required contextually when that manifest is selected through `--target` for
  generation; a missing list fails explicitly and never falls back to SI data.
- The legacy scripts are compatibility adapters while callers migrate. Their
  parsers are not architectural dependencies and can be deleted separately.
  The legacy `generate.py --rubric-manifest` adapter performs its own manifest
  translation before calling the generation application function.
- Pooling already exposes `pool_evaluation_directories`; `vera.py` calls that
  function rather than its script entry point.
- `vera resume` fails explicitly until the architecture's checksum, state ownership,
  and partial-write recovery contract is implemented.

## Compatibility

Existing config files without `judging.conversations` remain valid for `pipeline`,
where conversations come from the generation stage. A standalone `judge` config must
now include that field instead of combining `--config` with `--conversations`.
