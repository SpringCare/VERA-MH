---
status: draft
updated: 2026-07-10
---

# `vera.py` — Use Cases for Review

This document lists the intended use cases for the future `vera.py` entry point, so we can confirm nothing is missing before designing the CLI surface and config schema in detail.

**Status: draft, open for review.** Please flag anything missing or wrong.

## Use case 1 — End-to-end test of one LLM

Generate -> judge -> score chained, for exactly one provider/agent LLM under test, in a single invocation.

```
vera pipeline --config run.json
```

**Open question:** should this ever support multiple providers natively (one combined comparison report), or is comparing providers always done by invoking this once per provider (external loop)?

## Use case 2 — Batch generate across personas

One provider/agent LLM under test, generated against multiple personas, each carrying its own persona-side LLM.

```
vera generate --config run.json
```

Comparing across *providers* (not personas) is done by looping this invocation externally, once per provider — not a built-in cross-product flag.

## Use case 3 — Judge existing conversations

Judge one **or more** existing transcript folders against one or more rubrics. Each rubric has a default judge-LLM set, overridable per rubric. Multiple judges per rubric are supported (for judge-agreement analysis).

```
vera judge --conversations p_run_a/ p_run_b/ --config run.json
```

**Open question:** when judging 2+ folders together on the same rubrics, is the output one combined comparison result (side-by-side per-folder scores), or independent per-folder `results.csv` (batching as convenience only, no cross-folder aggregation)? This is a separate open question from use case 1's — resolving one does not resolve the other.

## Use case 4 — Smoke test

Run the full pipeline (or generate/judge alone) against a small sample of personas/rubrics/judges, to sanity-check that a config works end-to-end before spending the full LLM-call budget.

```
vera pipeline --config run.json --sample 2
```

`--sample N` overrides the config's full persona (and rubric/judge, where relevant) list at run time. This avoids hand-maintaining a separate small-scale config just for smoke testing.

## Config mechanism (applies to all use cases)

- `--config <path>` — JSON file, for local use.
- `--config -` — read JSON from stdin.
- `VERA_RUN_CONFIG` env var — inline JSON content, for remote/CI dispatch where uploading or mounting a file isn't convenient.
- Flags passed alongside `--config` override the file's values.
- JSON (not YAML): robust when passed as a one-line env var or stdin payload with no escaping ambiguity, and consistent with the existing `manifest.json` output convention.
- The same JSON schema is intended to be reused for the per-run output `manifest.json` — the input spec and the output record share one shape.

## Open questions summary

1. Multi-provider pipeline: native support in `vera pipeline`, or always an external loop? (Use case 1)
2. Multi-folder judge output: combined comparison, or independent per-folder results? (Use case 3)

## Not yet covered — flag if needed

- Dry-run / validate-only (check config and inputs resolve without spending any LLM calls)
- Resuming a partially-completed run
- Listing available personas/rubrics/judges
