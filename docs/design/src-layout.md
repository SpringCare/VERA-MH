# Move the source code under `src/vera_mh/`

Status: Proposed
Date: 2026-10-26

## Context

The repo root holds six Python packages (`generate/`, `judge/`, `score/`,
`utils/`, `llm_clients/`, `vera_cli/`) and `vera.py`, next to data, configs,
docs, tests, legacy scripts, and publication data. Three problems follow:

- **Hard to find the code.** A newcomer opening the repo sees more than 30
  entries at the top level and cannot tell which ones are the product.
- **Generic top-level names.** The wheel built by `pyproject.toml` installs
  `utils`, `judge`, `generate`, and `score` as separate top-level packages. Any
  other installed library with a `utils` package clashes with ours.
- **Tests may not test what users get.** Because the packages sit at the root,
  Python can import them straight from the checkout instead of from the
  installed package, so a packaging mistake (a missing file in
  `only-include`, say) can pass every test.

The Python packaging guide recommends the *src layout*: an importable package
inside a `src/` folder, which is not itself a package. `uv init --lib`,
Hatch, and Poetry all generate it.

## Decision

Move the six packages into one package, `src/vera_mh/`:

```
src/
  vera_mh/
    __init__.py
    cli.py            # was vera.py
    generate/
    judge/
    score/
    utils/
    llm_clients/
    vera_cli/
vera.py               # two-line shim: from vera_mh.cli import main
```

Imports gain the `vera_mh.` prefix: `from judge.runner import X` becomes
`from vera_mh.judge.runner import X`.

What stays at the root: `tests/`, `docs/`, `data/`, `configs/`, `scripts/`,
`publication_data/`, `legacy/`, and `score_comparisons/`. Nothing in the main
code imports `legacy/` or `score_comparisons/`, and the latter is mostly CSV
and PNG output; moving them would add churn for no gain.

### Alternatives considered

- **Flat `src/` (`src/judge/`, `src/utils/`, ...).** Keeps every import
  unchanged, but keeps the generic top-level names. Cleans up the root without
  fixing the clash.
- **Leave it as is.** The clutter grows with every new package, such as
  `workers/` and `storage/` from the architecture plan.

## Consequences

### What changes

| Where | Change |
|---|---|
| Python imports | 348 lines in 95 files get the `vera_mh.` prefix |
| Dotted names in strings | About 400, mostly `mock.patch("judge.x.y")` targets in tests |
| `pyproject.toml` | `[project.scripts] vera = "vera_mh.cli:main"`; hatch builds `src/vera_mh`; `--cov=src/vera_mh`; import-linter `root_packages = ["vera_mh"]`, contracts renamed to `vera_mh.judge` etc. |
| CI | The pyright gate lists `src/vera_mh/{generate,score,utils,vera_cli}` |
| Docs and agent files | About 25 files name package paths: `AGENTS.md`, `README.md`, `docs/*.md`, `.claude/commands/*.md`, `.github/CODEOWNERS` |
| Dockerfile | Still `COPY . .`; check `vera --help` runs in the image |

### Risks

- **Missed `mock.patch` strings.** An import mistake fails at collection time,
  but a wrong patch target fails only when that test runs, or patches nothing
  and passes. Rewrite them with the same script as the imports, then grep for
  any old dotted name left in a string.
- **Open branches conflict.** Every file moves, so any open branch that
  touches these packages has to be rebased. Merge or park those first, and
  start new work (such as the job loop) after the move.
- **Results on disk.** Run folders hold CSV, JSON, and transcripts, with no
  pickled objects or module paths, so existing results should stay readable.
  Confirm by scoring an old run after the move.
- **`python -m` commands.** `python -m score.compare` becomes
  `python -m vera_mh.score.compare`. Update the module docstrings and docs
  that show the old form.

### Plan

One PR, three commits so each is easy to review:

1. `git mv` only, plus the `vera.py` shim. Nothing else changes, so Git keeps
   each file's history.
2. Rewrite imports and dotted strings with a script, kept in the PR
   description so reviewers can rerun it.
3. Config, CI, docs, and agent files.

### How we check it

1. `uv run pytest -m "not live"`: catches broken imports and patch targets.
2. A clean `uv sync`, then `vera --help` and `python vera.py --help` from
   outside the repo: the package installs and both entry points work.
3. A small `vera pipeline` run (two personas, one run each) on the branch and
   on `main`: same files written, scores within normal LLM noise. The
   score-guard tests from #270 cover the arithmetic.
4. `docker build` then `vera --help` inside the image.
5. `vera score` on a run folder made before the move.
