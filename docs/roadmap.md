# VERA-MH Roadmap

Future structural work, in order. [architecture.md](./architecture.md) describes the target state and the rules that must hold; this file describes how the code gets there. Update it when a phase lands: move the phase to [Completed](#completed) and record it in [CHANGELOG.md](../CHANGELOG.md).

Smaller items that belong to no phase are in the [Backlog](#backlog).

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
| [Rubric-agnostic scoring](#rubric-agnostic-scoring) | Scoring works for any rubric, with unambiguous stop tokens | A non-SI target scores and groups by risk with no SI assumptions; every terminating rubric row uses an explicit stop token |
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

Scoring today assumes the SI rubric in several places.

- **Split the overloaded `END` token.** `GOTO=END` means two incompatible things: "a screening gate failed, so every dimension is Not Relevant" and "the questionnaire finished, so score what was collected". The scoring bug this caused, where finishing a rubric was treated like failing its first gate and the collected answers were discarded, is fixed since v2.0.0: `_calculate_results` (`judge/llm_judge.py`) only discards when a dimension is still unvisited. What remains is making the ambiguity unrepresentable: migrate the vocabulary to `STOP_SCORE` / `STOP_NOT_RELEVANT` / `STOP_ASSIGN` / `SKIP_DIMENSION>>{ID}` and add validator rules, chiefly an explicit stop token on every terminating row. A blank `GOTO` on a final row is a second overload ("next row" vs "finish and score").

  **Live case: HFO Q11** (a known issue in v2.0.3). "No Danger" routes to `END` before Supportive Conversation and Follows AI Boundaries are asked, so the gate rule marks all five dimensions Not Relevant and discards the Detects Potential Risk and Confirms Risk ratings already collected. The row's note asks for something no current token expresses: keep earlier ratings, mark this dimension and the later ones Not Relevant. `NOT_RELEVANT>>24` (what SI's Q9 does) keeps earlier ratings but goes on to rate the last two dimensions. Needs a decision from the HFO rubric authors, then a regression test. Ship it ahead of the vocabulary migration if they choose an existing token.

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

- Generation and judging take `ModelSpec` directly. Delete the `run_for_user_models` / `_legacy_model_config` stopgaps in `generate/run.py`; `legacy/generate.py` builds its own `ModelSpec` instead.
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

## Backlog

Smaller items that belong to no phase. Each can ship as its own PR whenever it's useful.

### CLI and run experience

- Replace the command-line experience during a run: today it is impossible to tell a working run from a hung one. `vera generate` prints per-conversation worker lines through plain `print`, so Python block-buffers stdout whenever it is not a tty -- piping to `tail`, redirecting to a file, or running under a wrapper shows nothing at all until the process exits, and the `Completed N / M` summary lands only at the end. The run folder is created up front and per-conversation logs are opened before the first model call, so an empty-looking folder is indistinguishable from a failed one. Nothing reports per-conversation progress (turns done of `--turns`), and there is no connect or overall timeout, so an unreachable provider endpoint leaves every worker parked in TCP SYN_SENT at 0% CPU with no output for as long as the caller waits -- the failure mode looks exactly like slow-but-working. Replace with a progress surface that is truthful under redirection: flush or use logging with an explicit stream, print a line per conversation as it starts and finishes rather than only in aggregate, surface turns-completed while a conversation is in flight, and fail fast with a named error on connect timeout instead of hanging. Judging has the same problem -- `🔍 Judging N conversations` then silence until the batch ends.
- Fail a pipeline immediately when a generation stage produces no conversations, instead of one stage later in judging. A stage whose every conversation was skipped -- an invalid model name is the easy way in, since the provider 400s on every call and each conversation is dropped after its retries -- still returns normally, so `_judge_and_score` runs next and dies in `load_conversations` with `FileNotFoundError: No .txt files found in: <run>/conversations` (judge/rubric_config.py:603). The traceback then names judging and the transcripts folder, so it reads as a path or layout problem when the real failure was upstream in generation and is only recoverable from the per-conversation logs. Under `--target all` or several `-u` models it is worse: the first model can succeed and the crash surfaces only when a later one fails, after that model's generation budget is already spent. Guard between the stages in `vera_cli/pipeline.py::_execute` and raise something that names the stage, the run folder, and the skip count -- "generation produced 0 of N conversations; all were skipped, see <run>/conversations/logs/" -- and decide whether a partial stage (some conversations skipped, some written) should warn and continue or stop, which is the same question `--sample` raises and is worth stating explicitly rather than leaving to whatever the glob happens to find. `generate` on its own has the milder form of this: it prints "Generated 0 conversations" and exits 0, so a caller scripting it cannot tell success from total failure by exit code.
- Decide whether `vera pipeline` needs per-stage debug. Today `--debug` covers the whole run, because `set_debug` flips a process global and `debug` lives on `InvocationConfig`. A `--debug-stage generate|judge|score` flag toggling that global around each stage in `vera_cli/pipeline.py:_execute` would be ~10 lines and no schema change. Revisit once someone has had to debug one stage of a long pipeline; drop this item if that never happens.
- Move each `vera` subcommand's argparse `register()` out of its command module (`vera_cli/generate.py`, `judge.py`, `score.py`, `pipeline.py`) into a parser-only module. Do all four in one change, since `vera generate` is the reference the other three copy.
- Print the resolved settings at the start of a run; LLM classes should report the model actually used.

### Config and providers

- Validate that `GenerationConfig`'s path fields (`personas`, `output`, `persona_context_template`) are resolved, as `RubricFiles` now does: a relative value is accepted today and means a different file depending on the reader. See #209.
- Move `load_dotenv()` out of import time in llm_clients/config.py, and read API keys lazily instead of as class attributes. Today importing llm_clients.config has the side effect of loading .env, and Config.ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY etc. are evaluated once at class-definition time, so the values freeze on first import. That is why tests/unit/llm_clients/test_config.py has to call importlib.reload(config) to observe a patched environment. Fix: call load_dotenv() explicitly from the CLI entry points (vera.py and the scripts in legacy/) rather than at import, and turn the Config key attributes into properties or a classmethod so each read consults os.environ. Scope: only 14 Config.<ATTR> uses in source (8 of them in llm_clients/azure_llm.py), but ~92 in tests, plus unwinding the reload pattern. Worth its own PR off main; it does not depend on the CircleCI work. Two partial mitigations already landed, so this is a cleanup and not urgent: the .env path is pinned to the repo so the search cannot escape into ~/.env, and an autouse fixture in tests/conftest.py replaces real provider credentials with dummies outside live tests. Not done yet: as of 2026-10-08, llm_clients/config.py still calls load_dotenv() at import and reads the keys into class attributes.
- Standardize how persona, agent, and judge information is represented.

### Tests

- Consolidate the live tests: one small live end-to-end run, mocked tests per stage, no live test per stage. The `vera` CLI already has that shape -- `tests/unit/test_vera_cli.py` / `test_vera_judge.py` / `test_vera_score.py` / `test_vera_pipeline.py` mock the stages, and `tests/integration/test_vera_pipeline_e2e.py` runs all three for real on cheap models, 4 turns (next to a mocked run of the same pipeline). The expensive live suite is `tests/integration/test_scoring.py`, which drives the deprecated legacy scripts (`generate.py`, `judge.py`, `run_pipeline.py`; see docs/legacy-scripts.md) with 7 tests on reasoning models (`Config.DEFAULT_*`: `gpt-5.4`, `claude-sonnet-5`) at 6 turns, each generating its own conversations. Generation dominates: one run measured 58.6 s generating vs 5.3 s judging, about 10 s per turn, and `temperature=0.0` is dropped as unsupported by both models so it buys no determinism. Now: delete `test_run_pipeline_vs_individual_calls`, which runs the full pipeline twice (separate commands, then `run_pipeline.main()`) and only asserts each result's percentages are in range and `worst_band` is set -- it passed with 0 relevant conversations, duplicates `test_complete_pipeline_single_persona` plus `test_run_pipeline_integration`, and its `"Omar"` persona is ignored because `run_generate_cli` never passes it; and cut `TEST_CONFIG["TURNS"]` from 6 to 4 to match the `vera` live test. When the legacy scripts are removed: delete `test_scoring.py` with them, first moving `test_pipeline_error_handling`'s intent (a failed CLI run reports a readable cause, #218) to a mocked `vera` test, since the `vera` live test covers only the success path.
- Replace the class-name -> test-file mapping in `tests/unit/llm_clients/test_coverage.py` with a declared one. Each `TestLLMBase` subclass sets `llm_class = OpenAILLM` (required by the base), and the coverage checks compare the set of declared classes against `get_all_llm_classes()`, then run the per-class checks on the test class that declares it. That deletes `implementation_to_test_file_stem` (#240), the AST parsing in `get_test_class_inheritance`, and the `if test_file_path.exists()` guards that let a wrong mapping skip a class silently. It also makes `test_all_test_classes_inherit_from_appropriate_base` a hard requirement for free (a test class that doesn't inherit the base can't declare coverage), so that test's "informational for now" warning goes away. Upstream cost: about 1-2 hours, one line in each of the 6 provider test classes plus the coverage rewrite and `tests/unit/llm_clients/README.md`. Downstream cost: any fork client whose tests don't subclass `TestLLMBase` fails CI after sync until it implements `create_llm` / `get_provider_name` / `get_mock_patches` and passes the shared contract tests, so give forks notice before landing it.
- Once #223 lands (`Config.DEFAULT_*` model constants plus `tests/integration/test_model_availability.py`), add cheap-tier constants to `Config` and have `tests/integration/test_vera_pipeline_e2e.py` read its model lists from them, so the availability check also covers the models that test uses.

### Other

- A better logging system.
- Find out why async judges don't run concurrently.

## Deferred

- **Removing `legacy/`.** Triggered by a tagged release whose CHANGELOG announces the removal. Then delete `legacy/`, the legacy side of the parity tests, and [legacy-scripts.md](./legacy-scripts.md).
- **Per-persona system prompts.** Personas may need their own system prompts rather than one global prompt. The mechanism (a `system_prompt` field, a `system_prompt_file` pointer, or something else) is undecided; revisit when the persona schema is formalized, likely in [Traceability](#traceability).

## Completed

- **Target manifests.** One complete fixture target drives both generation and judging; explicit `--personas` / `--rubric` selection is preserved.
- **`vera` CLI and config.** `vera.py` (`generate`, `judge`, `score`, `pipeline`) with `--config`, `--target`, `--personas`, `--rubric`, `-u`, `-j`, `--sample`, and `--into` resume. The old root scripts moved to `legacy/` (#243) instead of being deleted; see [Deferred](#deferred).
- **Multi-rubric evaluation, through targets.** This is the intended end state, not a stopgap. Evaluating the same conversations with several evaluators (rubrics) means one target per rubric: each target bundles its rubric with its personas and prompts, and each is judged separately with `vera judge --target <name>`, writing its own results. Keeping rubrics separate is deliberate, because scores from different rubrics aren't comparable and must never be merged. The earlier plan for several rubrics inside one run (`judging.rubrics` with length > 1, per-rubric judge-model overrides) is dropped. `judging.rubrics` keeps its list shape with a length-1 check, so revisiting it later wouldn't break the schema. What remains is convenience, not capability: `vera judge --target all` comes with [Traceability](#traceability), once the output path attributes each target.
- **Package moves.** `generate_conversations/` → `generate/` (#245); scoring out of `judge/` into `score/` with `run.py`, `aggregate.py`, `viz.py` (#246).
