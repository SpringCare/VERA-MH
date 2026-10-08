# VERA-MH Roadmap

Future structural work, in order. [architecture.md](./architecture.md) describes the target state and the rules that must hold; this file describes how the code gets there. Update it when a phase lands: move the phase to [Completed](#completed) and record it in [CHANGELOG.md](../CHANGELOG.md).

Smaller items that belong to no phase live in the root [`TODO`](../TODO).

## How phases work

- **Phases have names, not numbers.** Code comments and docs cite them by name ("Traceability"), so reordering or inserting a phase never changes what an old reference means.
- **Order is dependency order.** Each phase builds only on what earlier phases leave behind. Storage and OpenSpec are independent of the others and can run whenever a concrete need appears.
- **"Done when" is the bar.** A phase isn't done because its changes landed; it's done when its Done-when holds. Every Done-when also includes updating `README.md`, `AGENTS.md`, and any other doc the phase makes stale.
- **Every phase keeps [`legacy/`](../legacy/) working.** Removing the legacy scripts is [deferred](#deferred). Until then, legacy adapts and the domain doesn't bend: `legacy/` may import domain packages and `utils/`, never the reverse, and when a phase changes a signature or helper a legacy script uses, the old-shape adapter moves into `legacy/` instead of staying in the domain. The parity tests (`tests/integration/test_cli_parity.py`, `test_score_parity.py`) must stay green.
- **Planning a phase doesn't pre-clear it.** Each phase still needs maintainer sign-off before it starts (see [ESCALATE](./architecture.md#escalate-stop-and-ask)).
- **Rollback:** a phase that ships broken is reverted through its own PRs. There is no feature-flag period carrying old and new behavior side by side.

## Phases

| Phase | Goal | Done when |
|-------|------|-----------|
| [Scoring split](#scoring-split) | `score/` owns pooling and stops importing `judge/` | `vera pool` matches `scripts/pool_vera_scores.py` on the same inputs; `vera pipeline` ends with the pooled score; import-linter `judge/` ⊥ `score/` passes in CI |
| [Rubric-agnostic scoring](#rubric-agnostic-scoring) | Scoring works for any rubric and never discards completed evaluations | A non-SI target scores and groups by risk with no SI assumptions; a rubric that ends on `END` keeps its collected answers |
| [Traceability](#traceability) | Every run records what produced it, under the target-rooted layout | New runs write `config.json`, `state.json`, and checksums under `output/<target>/`; `vera judge --target all` works; `utils/` leaf contract passes |
| [Model spec boundary](#model-spec-boundary) | Domains take `ModelSpec` directly | `run_for_user_models` and `_legacy_model_config` are gone; per-model judge parameters work |
| [Shared engine](#shared-engine) | One worker pool for generation and judging, with provider rate limits | Both runners use `workers/`; a per-provider concurrency cap is enforced; all four [engine testing](#engine-testing) tiers pass |
| [Quality gates](#quality-gates) | Static checks block the build | Pyright blocks on every package; full import-linter and grimp contracts pass |
| [Storage](#storage) *(independent)* | Persistence goes through a backend interface | No domain or `workers/` code writes run artifacts directly |
| [OpenSpec](#openspec) *(independent)* | OpenSpec is a real practice, not scaffolding | A qualifying feature has shipped with an OpenSpec change |

### Scoring split

- **`vera pool`**, implemented in `score/pool.py`. `scripts/pool_vera_scores.py` becomes a thin legacy wrapper over it (moved to `legacy/`), so `legacy/run_recommended_vera_pipeline.sh` keeps working. Two user-visible strings already promise this command: `vera judge --conversations` help (`vera_cli/judge.py`) and a `JudgingConfig` error (`utils/config_schema.py`).
- **`vera pipeline` ends with the final score.** Today a multi-user-model pipeline leaves N per-user-model scores and no headline number, and the caller finishes the job by hand with the pool script. The pooled score is the answer the pipeline was run to get, so the pipeline should produce it. An earlier attempt was reverted for three reasons, each now addressed:
  1. The pooled artifact had nowhere correct to live. #215 documents its home at `<target>/c_<chatbot>/pooled/`; until [Traceability](#traceability) builds that layout, pooled output stays where the pool script writes it today.
  2. It inverted the layering (`vera_cli/` imported from `scripts/`). Fixed by `score/pool.py` above.
  3. It seemed to contradict "judge never auto-scores". That rule is about `vera judge` staying non-scoring, which still holds; `pipeline` exists to run every stage.

  A standalone `vera pool` stays useful for pooling evaluations judged separately, across targets, or re-pooled later.
- **Shared helpers leave `judge/`.** `score/` still imports `judge.score_utils`, `judge.constants`, and `judge.utils`, and `judge/runner.py` and `judge/answers.py` also use `score_utils`. Split it: what only scoring needs moves to `score/utils.py`, what both need moves to `utils/`.
- **Import-linter contract** for `judge/` ⊥ `score/`, the first contract in `pyproject.toml`.
- **Decide where `judge/score_comparison*.py` live** (likely `score/`), and where `score_comparisons/` output goes.

### Rubric-agnostic scoring

Scoring today assumes the SI rubric in several places, and one assumption silently drops data. Start with the bug.

- **Stop discarding completed evaluations that end on `END`.** `GOTO=END` means two incompatible things: "a screening gate failed, so every dimension is Not Relevant" and "the questionnaire finished, so score what was collected". `_ask_all_questions` returns the terminating question on any `END` (`judge/llm_judge.py`), and `_calculate_results` then takes the discard-all branch unconditionally, so finishing a rubric is indistinguishable from failing its first gate. All dimensions become Not Relevant and the collected answers are thrown away at scoring time. The answers survive in `answers/answers.tsv`, so affected runs can be re-scored without re-judging, but aggregates computed before the fix can rest on a fraction of the conversations judged. `docs/rubric.md` teaches the mistake ("use `END` to stop without assigning severity"), and `tests/fixtures/rubric_assign_end.tsv` contains it unobserved. `ASSIGN_END` is unaffected. Fix in two PRs, kept separate so a migration defect and a scoring defect can't be confused in the numbers:
  1. Gate the discard-all branch on whether any dimension is still unvisited (the discriminator `ASSIGN_END` already uses), with a regression test.
  2. Migrate the vocabulary to `STOP_SCORE` / `STOP_NOT_RELEVANT` / `STOP_ASSIGN` / `SKIP_DIMENSION>>{ID}` and add validator rules that make the bug unrepresentable, chiefly an explicit stop token on every terminating row. A blank `GOTO` on a final row is a second overload ("next row" vs "finish and score") and is the only reason `data/SI/rubric.tsv` scores correctly today.

  Design record: [rubric-terminal-goto-vocabulary.md](./design/rubric-terminal-goto-vocabulary.md).
- **Explicit `dimension_verdicts`.** Replace the `ASSIGN_END` / `NOT_RELEVANT>>` marker strings in `dimension_answers` with a verdict map: handlers record a verdict instead of overwriting a dimension's answers with a synthetic record, and `_determine_dimension_scores` reads it instead of substring-matching question text. `dimension_answers` becomes append-only and truthful, `yes_question_id` can be computed rather than regexed out of prose, and `answers.tsv` can source from `dimension_answers` directly. Rewrites about 10 tests that assert on marker strings, plus the fixture under `tests/fixtures/eval_tsv_rebuild__20260415_140037/`. Land it before or alongside the vocabulary migration above, since the verdict map replaces the marker matching that migration would otherwise carry forward.
- **`risk_level` from the judge, not persona metadata.** Today `add_risk_levels_to_dataframe` / `load_personas_risk_levels` (`judge/score_utils.py`), called from `score_results_by_risk`, join a personas TSV column literally named "Short Current Suicide Risk Level" on a persona name parsed from the filename. Any rubric whose personas file lacks that column silently yields "Unknown" for every row, and `scores_by_risk.json` comes back empty. Instead: declare `risk_level_question_id` in the rubric manifest, read that column from `answers/answers.tsv`, and take the grouping order from the question's answer options rather than the hardcoded `RISK_LEVEL_ORDER`. Retire the persona lookup and its `--personas-tsv` wiring, and relabel the "Persona Risk Level" chart axis. Existing `results.csv` files predate the column, so consumers must tolerate it being blank.
- **No SI-derived global dimensions.** Remove them from scoring and visualization; persist the rubric's dimensions and a rubric fingerprint with evaluation outputs, so later scoring and comparison use the rubric the run was judged with.

### Traceability

- **Persisted run records.** `utils/config_schema.py` formalizes the config shape into a stable interface (design doc required for later changes). Each run writes `config.json` once at start, plus `state.json` and a `.sha256` sidecar. The config records the checksum of every input file it uses (rubrics, personas, prompts), so a changed input fails loudly instead of silently.
- **Naming.** Rewrite `utils/naming.py` for the `c_`/`u_`/`j_` scheme. The `p_`/`j_` helpers the legacy scripts use (`legacy/judge.py`, `legacy/run_pipeline.py`) move into `legacy/`, which keeps writing the old layout.
- **Target-rooted output.** New runs write under `output/<target>/`, with pooled results under `c_<chatbot>/pooled/` (see [Data flow and artifacts](./architecture.md#data-flow-and-artifacts)). The target root attributes judging output on its own, which unblocks `vera judge --target all`: its only blocker was output attribution.
- **Resume.** `--into` already resumes a run folder. Decide whether a `state.json`-backed `vera resume` still adds enough to build.
- **Import-linter:** `utils/` is a leaf.
- **Acknowledged compatibility break.** Anything outside `vera.py` that parses `p_*`/`j_*` names directly (`spring_scripts/`, `distribute_files.py`, `score_comparison.py`, notebooks, human-review tooling) stops auto-discovering new runs. Reading old data keeps working: `vera judge --conversations <old p_* folder>` and `vera score -r <old results.csv>` take explicit paths and never derive meaning from folder names. Existing `output/` folders are left alone; there is no migration script.

### Model spec boundary

- Generation and judging take `ModelSpec` directly. Delete the `run_for_user_models` / `_legacy_model_config` stopgaps in `generate/main.py`; `legacy/generate.py` builds its own `ModelSpec` instead.
- Per-model judge parameters: lift the "all judge models must share provider parameters" check in `utils/config_schema.py`.
- Afterwards `generate` and `judge` describe models identically.

### Shared engine

- `workers/` pool used by both `generate/runner.py` and `judge/runner.py`, which today each hand-roll the same asyncio-queue pattern.
- **Per-provider concurrency cap** in `llm_clients/`. The shared pool can fan out many parallel jobs (e.g. `-u gpt:5 sonnet:5`); `llm_clients/` has per-call retry and backoff but nothing caps concurrency per provider. Done when a test fans out more jobs than the cap allows and the cap holds.
- `llm_clients/` plugin-registry formalization.
- This is the riskiest phase: the first one that rewrites the engine instead of the CLI in front of it. It needs all four tiers below.

#### Engine testing

The engine rewrite can change concurrency, timing, and error handling even when the LLM-calling logic is unchanged. Generation output is non-deterministic, and that compounds turn over turn, so "unchanged" can't mean exact-output diffing. None of these tiers exists yet, and passing `pytest -m "not live"` alone doesn't meet the bar:

- **Structural parity (automated, blocking).** With the mock harness (`tests/mocks/mock_llm.py`), assert the `workers/`-based runners make the same calls with the same arguments, respect the same turn-termination logic, and write the same output structure as before.
- **Live smoke test (automated, blocking, tolerance-based).** A `@pytest.mark.live` test on a small `--sample` batch, checking the engine still produces conversations of the same shape under real API conditions:
  - every sampled run completes without an unhandled exception
  - conversation count matches what was requested (persona × repeat; no dropped or duplicated runs)
  - each conversation reaches its expected turn count or ends through a documented condition
  - wall-clock time per conversation stays within a tolerance band (e.g. ≤2× a recorded baseline)
  - an injected provider failure still triggers the existing retry policy and resolves the same way
- **Semantic similarity (automated, statistical, blocking).** For each sampled `(persona, config)` pair, generate one conversation on the old engine and one on the new, embed both, and compute cosine similarity. Compare that distribution against an old-vs-old baseline from two independent old-engine runs. A significant drop relative to the baseline (one-sided test at a pre-agreed level) flags semantic drift without over-flagging ordinary sampling variance.
- **Manual spot-check (qualitative).** A human reads a handful of the sampled transcripts for coherence, topicality, and clinical sense, the one thing no automated check certifies.

### Quality gates

- **Pyright, package by package.** Add a blocking pyright step for the packages with zero errors (today `generate/`, `score/`, `utils/`, `vera_cli/`, `vera.py`), and add each remaining package as its errors are fixed: `judge/`, then `llm_clients/`, then the rest. Done when no package is left on the non-blocking run.
- **Import contracts.** Full import-linter contract (all [Layer model](./architecture.md#layer-model) boundaries) plus grimp import-graph assertions.
- **Root-clutter cleanup:**
  - delete `logging/`, `logs/`, root `conversations/`, `human_validation/` (old generated dumps, already gitignored)
  - `score_comparisons/`: keep the directory, stop committing generated CSV/PNG output
  - move `distribute_files.py` into `scripts/`
  - fix stale `docker-compose.yml` volume mounts (`./evaluations` doesn't exist, `./logs` is missing)
  - `spring_scripts/`, `openspec/`, `alba.md`, `Untitled.ipynb`: untracked scratch, out of scope

### Storage

Decouple "which path or key" from "how bytes are persisted", so a non-local backend (S3, etc.) is a new implementation rather than a rewrite. New `storage/` package: a `StorageBackend` interface with `write(key, bytes)`, `read(key)`, `exists(key)`, and `LocalFilesystemStorage` as the default. The backend knows nothing about run semantics; `utils/naming.py` still builds keys. Domain and `workers/` code that writes run artifacts calls through the backend.

**Done when:** no domain or `workers/` code writes run artifacts with `open()` or `pathlib` directly, and `LocalFilesystemStorage` behaves identically to today's direct writes. Start it when a concrete need (an actual S3 requirement) shows up.

### OpenSpec

`openspec/` is empty scaffolding. The next large multi-file feature (the ESCALATE triggers: new judge dimensions, pipeline CLI changes) ships with a real OpenSpec change under `openspec/changes/`.

**Done when:** that change has shipped and the ESCALATE section's "if the team adopts" becomes a firm MUST for later qualifying changes.

## Deferred

- **Removing `legacy/`.** Triggered by a tagged release whose CHANGELOG announces the removal. Then delete `legacy/`, the legacy side of the parity tests, and [legacy-scripts.md](./legacy-scripts.md).
- **Multiple rubrics in one run.** Superseded by targets: each target bundles its own rubric, so judging the same conversations under another rubric is another `vera judge --target <name>` run. `judging.rubrics` stays list-shaped with a length-1 check, so lifting it later wouldn't be a schema break. Per-rubric judge-model overrides go with it.
- **Per-persona system prompts.** Personas may need their own system prompts rather than one global prompt. The mechanism (a `system_prompt` field, a `system_prompt_file` pointer, or something else) is undecided; revisit when the persona schema is formalized, likely in [Traceability](#traceability).

## Completed

- **Target manifests.** One complete fixture target drives both generation and judging; explicit `--personas` / `--rubric` selection is preserved.
- **`vera` CLI and config.** `vera.py` (`generate`, `judge`, `score`, `pipeline`) with `--config`, `--target`, `--personas`, `--rubric`, `-u`, `-j`, `--sample`, and `--into` resume. The old root scripts moved to `legacy/` (#243) instead of being deleted; see [Deferred](#deferred).
- **Package moves.** `generate_conversations/` → `generate/` (#245); scoring out of `judge/` into `score/` with `run.py`, `aggregate.py`, `viz.py` (#246).
