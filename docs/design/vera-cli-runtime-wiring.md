# VERA CLI runtime wiring decision record

Status: Accepted
Date: 2026-08-06

## Context

Wiring the unified CLI requires changes to `utils/config_schema.py`, a stable
interface. Standalone judging also exposed an ambiguity: it needs conversation
paths, while AD-17 forbids combining JSON config with run-defining CLI flags.

## Decision

The unified CLI runtime and configuration contracts are canonical in:

- [Architecture: CLI runtime boundary](../architecture.md#cli-runtime-boundary)
- [Architecture: input resolution](../architecture.md#input-resolution)
- [Architecture: target manifest](../architecture.md#target-manifest)
- [CLI/config use cases](../vera-cli-use-cases.md)

This record captures the rationale for the corresponding stable-interface
changes without duplicating those contracts here. The implementation resolves
every input into canonical values before dispatch and calls parser-independent
domain functions.

## Consequences

- Existing pipeline configs may omit `judging.conversations`, because generation
  supplies those paths. Standalone judge configs must now include it and may no
  longer combine `--config` with `--conversations`.
- Generation configs that relied on Python or CLI defaults must add the required
  generation behavior fields explicitly. CLI invocations retain defaults at the
  flag-definition boundary.
- Target manifests are complete bundles of rubric, personas, and prompts.
  `--target` selects the bundle, while explicit `--personas` and `--rubric`
  select the named target's persona or rubric component, including that
  component's associated prompts.
- Legacy root parsers remain usable only during their feature-by-feature
  replacement. A unified command may temporarily call an importable function
  from the corresponding root module; it must not invoke that module's parser or
  run it as a subprocess. Generation uses `generate.main` until the root module
  is atomically replaced by the permanent `generate/` package.
