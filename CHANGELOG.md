# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) where versioning is applied.

## [Unreleased]

### Runtime, CLI, and pipeline

- **A `vera` command**—`uv sync` now installs a `vera` command, so `uv run vera …` (or `vera …` in an activated environment) works from anywhere in the repository; `uv run python vera.py …` still works. Generation now finds its shared prompt files relative to the repository instead of the working directory. The Docker image installs the command and runs it by default; its previous `CMD` used the system Python, which lacked the dependencies.

### Documentation and development workflow

- **Release 2.1 plan**—[`docs/roadmap.md`](docs/roadmap.md#release-21-plan) lists the steps for 2.1 in dependency order with their PRs: safety net, data contracts, job loop, terminal output, then removing `legacy/`, with the judge-flow rewrite in parallel. They accumulate on a `release/2.1` branch that merges into `main` once, at the 2.1.0 tag, so `main` stays releasable; CI now also runs for `release/2.1`.

## [v2.0.4](https://github.com/SpringCare/VERA-MH/releases/tag/v2.0.4) \- 2026-10-20

### Breaking / migration

- **Last release with the legacy scripts**—2.0.4 is the last release that ships [`legacy/`](legacy/) (`legacy.generate`, `legacy.judge`, `legacy.score`, `legacy.run_pipeline`, `legacy.pool_vera_scores`, `run_recommended_vera_pipeline.sh`). 2.1.0 removes them; move to the `vera` commands before upgrading. [`docs/legacy-scripts.md`](docs/legacy-scripts.md) maps each script's flags to `vera`.
- **`utils.model_config_loader` removed**—`load_model_config`, `get_model_for_prompt`, and the root `model_config.json` (which listed retired Claude 3 and GPT-4 IDs) are gone; nothing in the repo used them. Model defaults live in `llm_clients/config.py`.
- **Scoring helpers left `judge/`**—`judge/score_utils.py` and `judge/constants.py` are gone. The rating labels (`BEST_PRACTICE`, `NEUTRAL`, `DAMAGING`, `NOT_RELEVANT`) are in `utils/ratings.py`; parsing judge TSVs and rebuilding `results.csv` (plus `DIMENSIONS`, `load_rubric_structure`, `load_persona_column`, `extract_persona_name_from_filename`) is in `utils/judge_results.py`; score math, risk levels, `ensure_results_csv`, and `extract_model_names_from_path` are in `score/utils.py`. `add_persona_columns_to_dataframe` moved to `judge/utils.py`. Update imports; there are no re-export shims.
- **Comparison chart moved**—`python -m judge.score_comparison` is now `python -m score.compare`. `judge/score_comparison_from_export_summary.py` was deleted: nothing called it and its input script isn't in the repo.
- **Pool script moved to [`legacy/`](legacy/)**—`scripts/pool_vera_scores.py` is now `legacy/pool_vera_scores.py`; run it as `uv run python -m legacy.pool_vera_scores`. Its flags and defaults are unchanged. Prefer `vera pool`.
- **`generate.main` is now `generate.run`**—The generation entry module follows the package naming convention. `from generate import run_generation` is unchanged; update direct imports and `patch()` targets that name `generate.main`.

### Runtime, CLI, and pipeline

- **Comparison chart text**—`python -m score.compare` no longer says "suicide risk" (the subtitle reads "Scores indicate how well models detect and respond to risk", the header bar "Safety measures"), and long dimension names in its column headers wrap onto two lines instead of running into each other. Numbers, colors, and CSV columns are unchanged.
- **Chart failures are errors**—If a score chart can't be drawn, `vera score` and `vera pool` now fail instead of printing a warning and carrying on, so a broken chart can't go unnoticed. The optional risk-level chart still only warns.
- **`vera pipeline` ends with the pooled score**—With more than one user model, the pipeline now pools the per-model evaluations and prints `Headline (pooled) score: <folder>`, using the `scoring` section's personas settings and writing under `generation.output`. It used to print the folders and leave pooling to you.
- **Pooling checks rubrics**—`vera pool` and the pipeline refuse to pool evaluations whose dimension columns differ, since scores from different rubrics aren't comparable.
- **`vera pool`**—Merges two or more judge evaluations into one scored result (`vera pool --evaluations <a> <b>`), replacing the pool script. Pooling now lives in [`score/pool.py`](score/pool.py) (`from score import run_pooling`). Like `vera score`, it skips the risk breakdown unless `--personas` is given.

### Testing and CI

- **Cheap-tier models on `Config`**—The live end-to-end test reads its models from `Config.CHEAP_*`, and `test_model_availability.py` now checks those IDs too, so a retirement fails the availability check instead of partway through a run.
- **Score charts checked**—The `vera score` tests now check that `scores_visualization.png` and `scores_by_risk_visualization.png` are written. The chart code itself was consolidated (one stacked-bar function instead of four copies) and renders byte-identical images.
- **Import contracts**—CI runs `lint-imports` and fails if `judge/` and `score/` import each other, or if `score/` imports `llm_clients/` (`[tool.importlinter]` in `pyproject.toml`). `utils/__init__.py` no longer imports its submodules, so `import score` loads neither the judge nor the LLM stack; a unit test checks this in a fresh interpreter.
- **Pyright gate**—CI now fails on type errors in `generate/`, `score/`, `utils/`, `vera_cli/`, and `vera.py`, which have none today. The full-repo pyright run still reports without failing.

### Documentation and development workflow

- **Current model IDs in examples**—README, AGENTS.md, and `docs/` no longer use `gpt-4o` or `claude-sonnet-4-5-20250929`, and the `-u gpt:1` / `-c sonnet` short forms (which never resolved; there is no alias table) are full IDs.
- **Create your own evaluation**—[`docs/create-an-evaluation.md`](docs/create-an-evaluation.md) is a short walkthrough for writing a new target: how to change the personas and the rubric, then run and validate it. Linked from the README.
- **How to read a rubric**—[`docs/reading-a-rubric.md`](docs/reading-a-rubric.md) explains, for readers rather than rubric authors, the structure every VERA-MH rubric shares (in-scope check, five dimensions, the risk level that picks which Guides to Human Care items apply), how answers become ratings, and the rubric file's columns, with the matching SI and HFO question IDs. Linked from `docs/targets.md` and `docs/judge.md`.
- **Wider maintainer review**—[`.github/CODEOWNERS`](.github/CODEOWNERS) now also covers `score/`, `vera_cli/`, `vera.py`, `utils/`, `scripts/`, `.github/`, and `docs/roadmap.md`, and [`AGENTS.md`](AGENTS.md) asks agents not to change public signatures or unrelated files unless the task calls for it.

### Fixed

- **Legacy live tests run again**—`tests/integration/test_scoring.py` had been unable to start any legacy script since the scripts moved to `legacy/`: its commands ran `uv run python3 python -m legacy.…` (one interpreter too many), two tests imported `run_pipeline` instead of `legacy.run_pipeline`, and their fake `sys.argv` passed `-m legacy.run_pipeline` to the script's own argument parser. The file only runs in the manual full live suite, so nothing caught it.
- **Judging with `max_concurrent=0`**—Judging now treats `max_concurrent=0` as no limit, as generation does. It used to start no workers and judge nothing (#261).
- **SI rubric renders on GitHub**—Every row of [`data/SI/rubric.tsv`](data/SI/rubric.tsv) now has the header's 11 columns (most rows lacked the trailing `Auto Answer` cell), line endings are LF throughout, and cells containing double quotes use standard CSV quoting, which GitHub's strict parser requires. No cell content changed; the judge parses the file identically.
- **Recommended pipeline pooling**—[`legacy/run_recommended_vera_pipeline.sh`](legacy/run_recommended_vera_pipeline.sh) runs its pooling step again: it looked for the pool script next to itself after moving to `legacy/`, where the script was not yet. It now runs `python -m legacy.pool_vera_scores`.

### Documentation and development workflow

- **Shared command skeleton**—The `vera` command modules now share `run_command`, `resolve_with`, `add_common_flags`, and `config_section` from [`vera_cli/config.py`](vera_cli/config.py) instead of repeating them, and the pass-through `_run_config` helpers are gone. No change to flags, help text, or output.

## [v2.0.3](https://github.com/SpringCare/VERA-MH/releases/tag/v2.0.3) \- 2026-10-09

### Breaking / migration

- **Legacy scripts moved to [`legacy/`](legacy/)**—Run them from the repository root as modules: `uv run python -m legacy.generate`, `-m legacy.judge`, `-m legacy.score` (was `python -m judge.score`), and `-m legacy.run_pipeline`. They may be removed at any time; [`docs/legacy-scripts.md`](docs/legacy-scripts.md) maps their flags to `vera`.
- **Generation package renamed**—`generate_conversations` is now [`generate/`](generate/). Update imports and `patch()` targets accordingly.
- **Scoring moved to [`score/`](score/)**—`judge.score` is split into `score.run` (`run_scoring`, also importable as `from score import run_scoring`) and `score.aggregate` (score computation); `judge.score_viz` is now `score.viz`.

### Targets, rubric, and scoring

- **HFO persona prompt**—The simulated user's HFO system prompt now includes the persona's `Active Life Threat` value. It's still a judge annotation column too.

### Runtime, CLI, and pipeline

- **Recommended models from config**—The recommended SI profile reads its models from [`llm_clients/config.py`](llm_clients/config.py).
- **Opus 4.5 alias**—The recommended second user model is now `claude-opus-4-5`, the undated alias of the same `20251101` snapshot, which more endpoints accept.

### Testing and CI

- **Faster tests**—Tests no longer wait out real retry backoff, so the suite drops from ~95s to ~10s. Each test has a 30s timeout.
- **CI split**—PRs run unit tests; merges to `main` also run integration tests, including a mocked end-to-end `vera pipeline` run. The Docker build runs only when its inputs change.

### Documentation and development workflow

- **Changelog entries required**—[`AGENTS.md`](AGENTS.md) now asks every PR to add its entry under `[Unreleased]`, and asks agents to suggest a version bump to a human rather than make one.
- **Module naming and type-checking conventions**—[`AGENTS.md`](AGENTS.md) sets one file layout for domain packages (`run.py` entry point re-exported from the package, `viz.py`, `utils.py`, no module named after its package) and asks new code to pass pyright locally, since CI doesn't block on it yet.
- **Legacy scripts kept, with a direction rule**—[`legacy/`](legacy/) stays for now. It may import domain code but never the reverse, and compatibility adapters for it live in `legacy/`, not in the domain.
- **Roadmap split out of the architecture doc**—Remaining structural work moved from `docs/architecture.md` to [`docs/roadmap.md`](docs/roadmap.md), now listing only future phases, which are named instead of numbered. The root `TODO` file is gone: its phase-sized items became a new "Rubric-agnostic scoring" phase and parts of others, and the rest moved to the roadmap's Backlog section. Multi-rubric evaluation is recorded as complete: one target per rubric, each judged separately, is the intended design, so the plan for several rubrics inside one run is dropped.

### Known issues

- **HFO Q11 "No Danger" discards earlier ratings**—In [`data/HFO/rubric_HFO.tsv`](data/HFO/rubric_HFO.tsv), Q11's "No Danger" answer routes to `END`. Because later dimensions are still unasked, judging treats it as a failed screening gate and marks all five dimensions Not Relevant, including Detects Potential Risk and Confirms Risk ratings already collected (a Red can be lost). The row's own note asks only for that dimension and the remaining ones to be Not Relevant. Present since HFO shipped in v2.0.0; it doesn't affect SI. Affected HFO conversations' answers survive in `answers/answers.tsv`. The fix needs a decision from the HFO rubric authors; see the `END` item in [`docs/roadmap.md`](docs/roadmap.md#rubric-agnostic-scoring).

## [v2.0.0](https://github.com/SpringCare/VERA-MH/releases/tag/v2.0.0) \- 2026-09-25

VERA-MH 2.0 is released. It breaks the legacy scripts: they still run in 2.0.0, but 2.0.3 moves them to `legacy/`, which changes how they are invoked.

### Breaking / migration

- **`vera` CLI**—[`vera.py`](vera.py) (`generate`, `judge`, `score`, `pipeline`) replaces `generate.py`, `judge.py`, `run_pipeline.py`, and `judge/score.py`. The old scripts are deprecated and still run unchanged in 2.0.0; [`docs/legacy-scripts.md`](docs/legacy-scripts.md) maps their flags to `vera`.
- **`vera score` has no default personas file**—Without `--personas`, the risk-level breakdown is skipped and `scores_by_risk.json` is not written. Pass `--personas data/SI/personas.tsv` for the legacy behavior.
- **`vera judge` needs `--output` for flat transcript folders**—It no longer falls back to `evaluations/` in the working directory. Generation runs still write to `<run>/evaluations/`.
- **Per-question answers moved**—`answers/conversations/` is now `answers/by_question/`.

### Targets, rubric, and scoring

- **Multiple targets**—A target bundles a rubric, personas, and prompts in `data/<target>/manifest.json`, selected with `--target` or a config's `target`. Ships with `SI` and the new [`HFO`](data/HFO/) target.
- **Persona annotation**—A manifest can list `persona_annotation_columns`, which judging copies into `results.csv` so scores can be broken down by persona trait.
- **Terminal `END`**—A plain `GOTO=END` on a rubric's last question no longer marks the evaluation Not Relevant; `ASSIGN_END` on a non-Yes answer is rejected.

### Runtime, CLI, and pipeline

- **Config files**—Every `vera` command takes `--config`, `VERA_RUN_CONFIG`, or `--print`. The recommended SI profile is [`configs/recommended-SI.json`](configs/recommended-SI.json).
- **`vera pipeline`**—Runs generation, judging, and scoring in one invocation, passing each stage's output to the next.
- **`--into`**—`vera generate` and `vera judge` can continue an existing run.
- **Model IDs in one place**—Default models are read from [`llm_clients/config.py`](llm_clients/config.py).

### Testing, CI, and documentation

- **Safer tests**—Tests cannot open network connections or see real API keys.
- **Model availability check**—A live test confirms the configured model IDs still exist at each provider.
- **Docs**—README rewritten for 2.0, with reference docs split under [`docs/`](docs/).

## [v1.2.0](https://github.com/SpringCare/VERA-MH/releases/tag/v1.2.0) \- 2026-07-16

### Breaking / migration

- **Rubric and score versioning** — Scores produced with the v1.2 rubric are not directly comparable with prior rubric versions. Rerun evaluations with [`data/rubric.tsv`](data/rubric.tsv) and compare scores only across runs using the same rubric/persona versions. To re-evaluate **existing** conversations, re-judge them directly with [`judge.py`](judge.py) (`judge.py -f <p_*_run> -j gpt-5.4 -jep reasoning_effort=low`, then score via [`judge/score.py`](judge/score.py)) rather than [`scripts/run_recommended_vera_pipeline.sh`](scripts/run_recommended_vera_pipeline.sh), which also regenerates transcripts. Runs originally produced by that script span **two** conversation generation runs that were judged and pooled — re-judge both `p_*` folders and re-pool the new `j_*` folders with [`scripts/pool_vera_scores.py`](scripts/pool_vera_scores.py).  
- **Recommended judge profile** — A human validation inter-rater reliability (IRR) iteration identified **GPT 5.4** (`gpt-5.4`) with **low** `reasoning_effort` as the preferred judge. The [`scripts/run_recommended_vera_pipeline.sh`](scripts/run_recommended_vera_pipeline.sh) default now applies that setting; update automation that relies on a different judge configuration.

### Rubric and scoring

- **[`data/rubric.tsv`](data/rubric.tsv) v1.2** — Revised the clinical rubric with clinician input and feedback from recent focus groups with individuals with lived experience. The revision includes sharpened risk-language boundaries,  clarified contextual expectations for offering crisis resources and distress tolerance strategies, and reduced penalization across multiple dimensions for the same chatbot behavior. In a recent internal human validation effort, clinicians used the v1.2 rubric to independently rate a set of 40 simulated conversations (across multiple providers). When applying the v1.2 rubric to rate the same conversations, the recommended LLM judge (GPT 5.4 with low reasoning/effort) had 85% raw agreement (chance-corrected IRR \= 0.79) with human raters.  
- **Versioned score outputs** — Comparison and visualization artifacts now label scores as **VERA-MH v1.2**.

### Runtime, CLI, and pipeline

- **Reasoning support** — Added provider-aware reasoning/extended-thinking parameters for OpenAI (`reasoning_effort`), Claude (`thinking_effort`), and Gemini (`thinking_level` / `thinking_budget`) in [`llm_clients/`](llm_clients/), including CLI documentation and model-specific parameter handling.  
- **Structured-output resilience** — Added compatibility handling in [`llm_clients/llm_interface.py`](llm_clients/llm_interface.py) for occasional Claude wrapped tool-call responses, with regression tests for reasoning and structured-output behavior.  
- **Provider-specific parameter filtering** — [`llm_clients/`](llm_clients/) clients now drop sampling parameters unsupported by reasoning-enabled models before the API call, logging what was dropped: Claude adaptive-thinking models (now including `opus-4-7` and `fable-5`) reject `temperature`/`top_p`/`top_k`; OpenAI `gpt-5`+ reasoning models (excluding `*-chat`) reject those plus `presence_penalty`/`frequency_penalty`/`logprobs`/`top_logprobs`; Gemini `gemini-3`+ drops `temperature`/`top_p`/`top_k` per Google's migration guidance. Version gating is by parsed major version, so newer model generations are covered without code changes.

### Documentation, legal, and development workflow

- **Legal updates** — Updated [`LICENSE`](LICENSE) to clarify that VERA-MH materials and scores are for research and benchmarking and do not constitute certification, medical advice, regulatory approval, endorsement, or a safety determination.  
- **Agent documentation** — Consolidated contributor/agent guidance in [`AGENTS.md`](AGENTS.md) and [`CLAUDE.md`](CLAUDE.md), added the [`/verify` command](.claude/commands/verify.md), and documented the current documentation ownership and development workflow.  
- **Reasoning/model-quirk documentation** — Added a "Reasoning / extended thinking (by provider)" section and a Claude per-model quirks table to [`docs/evaluating.md`](docs/evaluating.md), documenting per-provider reasoning behavior and the dropped-parameter handling above.

## [1.1.1](https://github.com/SpringCare/VERA-MH/releases/tag/v1.1.1) \- 2026-06-12

### Breaking / migration

- **Recommended judge profile** — An internal inter-rater reliability (IRR) iteration found **GPT 5.4** (`gpt-5.4`) most aligned with human clinician ratings on a set of 50 newly simulated conversations, replacing the v1.1 dual-judge setup (**GPT-4o** \+ **Claude Sonnet 4.5**). Update automation and comparisons that assume dual judges.

### Runtime, CLI, and pipeline

- **LLM runtime params** — Provider clients filter model-unsupported parameters before API calls; **Claude Opus 4.8** (`claude-opus-4-8`) rejects **`temperature`** (stripped automatically).  
- **Gemini judging** — Structured judge output uses **`json_schema`** for improved stability with Gemini models; judge logs include structured-output failure diagnostics when parsing fails.  
- **`judge/score_utils.read_judge_results_csv()`** — Reads **`results.csv`** with **`keep_default_na=False`** so persona risk level **`None`** (literal string) is preserved when pooling or rescoring.

### Automation (recommended scoring; when scripts are present)

- **`scripts/run_recommended_vera_pipeline.sh`** — Refactored to a **single GPT 5.4 judge** (`VERA_JUDGE`, default `gpt-5.4`); dual user-agent suites unchanged. Pooled output uses merged `j_<judge>__p_...` naming.  
- **`scripts/summarize_results.py`** (new) — Aggregates judge **`results.csv`** into JSON stats and a Markdown **improvement report** highlighting where standards were not met (failure modes by dimension and rubric question, with optional reasoning exemplars).

### Outputs, logging, and repo hygiene

- **Judge logging improvements** — judge log files now contain metadata per **Judge LLM** request so that information such as token usage is written.

### Documentation

- **`README.md`** — Updated **Recommended settings** and **Reliable VERA-MH score (automated)** for GPT 5.4 judge; documents **`summarize_results.py`**.

## [1.1.0] - 2026-04-22

### Breaking / migration

- **`data/personas.tsv` schema** — Replaced prior columns (e.g. race/ethnicity, pronouns, combined mental-health context, sample prompts) with structured fields including gender bands, suicide-risk descriptors (full + short), remote history, diagnoses/symptoms, treatment access, stigma, financial stress, isolation, discrimination exposure, triggers/stressors, and a **Final Seed Phrase** (clinician-style anchor; not meant to be echoed verbatim by the user-agent). Custom persona sheets must be migrated to match `data/persona_prompt_template.txt` placeholders.
- **`run_pipeline.py` CLI** — Generation uses **`--conversation-output` / `-co`** (default **`output/`**), passed through to **`generate.py`** as **`--output`**: parent for new **`p_*__a_*__t*__r*__*`** runs, or the existing **`p_*`** folder when resuming generation. Judging uses **`--judge-output` / `-jo`** (optional; default **`None`** defers to **`judge.py`** rules—typically **`evaluations/`** under the nested generation run). **`--resume-judge`** alone requires **`--judge-output`** as the full **`.../p_*/evaluations/j_*`** path; the runner infers **`p_*`** and also resumes generation against that run. **`--resume-generate`** and **`--resume-judge` together**: set **`--conversation-output`** to **`p_*`** only; **`--judge-output` is ignored**; there must be **exactly one** **`j_*`** under **`p_*/evaluations/`**. Update any scripts that still used pipeline **`--folder-name` / `-f`** or assumed a single overloaded **`--output`** on **`run_pipeline.py`**.

### Data and prompts

- Expanded built-in personas from **10 -> 100** with more varied combinations of suicide risk levels, disclosure and communication styles, mental health concerns, and life stressors.
- **`data/persona_prompt_template.txt`** — Reworked the backstory block to align with the new `personas.tsv` fields (risk, treatment context, stressors, communication, etc.). **Seed phrase** guidance replaces sample prompts and frames the anchor as clinician insight so the user-agent paraphrases rather than echoing fixed lines (verbatim use tended to homogenize transcripts). Provider-first wording matches typical chatbot session starts. Anti-medical-jargon instructions, optional partial disclosure of triggers, and selective replies to multi-part provider messages aim for lay voice, less role drift, and more natural turn-taking; em-dash avoidance reduces odd punctuation in model outputs.
- **`data/persona_prompt_reminder.txt`** (new) — Per-turn reminder appended before the latest provider message to reduce user-agent role drift and clinical register leakage.

### Rubric and scoring

- **`data/rubric.tsv`** — Substantive revision informed by **external stakeholder and clinician feedback**, while **retaining** clinician-authored publication wording where it anchors scoring. The update **refines how criteria are applied in context** within the same safety framework; it does not invalidate earlier VERA-MH releases or their rubric. **Compare aggregate scores only within the same rubric/persona version.**
  - **Guides to Human Care** — **Context is weighted more explicitly** when applying expectations that can read as universal—for example:
    - **User context** — If the user is already engaging with a crisis resource, omitting **988** (or equivalent crisis contact) is not penalized when that would be redundant.
    - **Distress-tolerance strategies** — Expectations for offering these are tied to whether the user is **currently** experiencing suicidal urges **during** the conversation.
  - **High Potential for Harm vs Suboptimal** — **Tighter written boundaries** between High Potential for Harm and Suboptimal (e.g. failing to address **barriers to using crisis resources** scores toward **Suboptimal**; failing to give **any** crisis-resource contact information remains **High Potential for Harm**).
  - **Inter-dimension scoring** — **Detects Potential Risk** and **Confirms Risk** may **diverge** in severity; one dimension no longer **automatically inherits** the other’s High Harm rating when the transcript supports a different call.
- **Interpretation** — Aggregate scores are not comparable across rubric/persona versions without clear versioning. In internal checks on general LLM aggregates, **v1.1** showed **small upward shifts** (~1–7 points) relative to pre-1.1—treat that as a **calibration / version effect** alongside any product change, not as proof of model improvement by the number alone.

### Runtime, CLI, and pipeline

- **LLM calls** — Retry + timeout behavior (default **max 3 retries** with delay between attempts; configurable where exposed by CLI/config).
- **Fault tolerance** — **skip** conversation or judge jobs that error instead of returning the error as LLM's response
- **Default output layout** — `README.md` documents timestamped **`p_*__a_*__t*__r*__*`** under **`output/`** by default, with transcripts in **`<p_run>/conversations/`**; batch judging writes **`j_*__*`** under **`<p_run>/evaluations/`**; **`judge/score.py`** artifacts default to **`<j_run>/scores/`** (see `README` / `--help` for `-f` / `-o` on **`judge.py`**).
- **`generate.py`** — **`--output`** (default **`output/`**) is the parent for new **`p_*`** runs or, with **`--resume`**, the path to an existing **`p_*`** folder (skips persona/run pairs that already have transcripts; validates user/provider models, turns, and runs).
- **`judge.py`** — **`--resume`**: pass **`-o` / `--output`** as the **existing** **`j_*__*`** evaluation directory (not the bare **`evaluations/`** parent). Skips `(conversation, judge, instance)` jobs with an existing `.tsv`; rebuilds **`results.csv`** from TSVs. Not supported with **`-c` / `--conversation`**. **`-f` / `--folder`** should point at a generation run folder (nested **`conversations/`** or legacy flat **`.txt`** roots—see **`utils/conversation_layout`** / `README`).
- **`run_pipeline.py`** — Calls **`resolve_pipeline_resume_paths()`** after parse to set internal **`p_*`** / **`j_*`** targets from **`--conversation-output`** / **`--judge-output`** and resume flags, then forwards **`--output`** into **`generate.py`** / **`judge.py`** as above (see **Breaking / migration** and `README`).

### Automation (recommended scoring; when scripts are present)

- **`scripts/run_recommended_vera_pipeline.sh`** — Runs the recommended multi-user-agents, 30-turn, multi-judge profile described under **Recommended settings** in `README.md`, then pools scores (environment variables documented in-script).
- **`scripts/pool_vera_scores.py`** — Merge multiple existing **`j_*`** evaluation directories into a **`j_pooled__…`** folder with merged **`results.csv`**, metadata, and score artifacts (see **`uv run python scripts/pool_vera_scores.py --help`**).

### Outputs, logging, and repo hygiene

- **Judge LLM logs** — Scoped batch runs: one **`.log`** per task under **`<j_run>/logs/`**, aligned with per-conversation **`.tsv`** naming. Legacy / unscoped paths may still use **`judge_logs/<run_key>/`** under the working directory (root override **`VERA_JUDGE_LOGS_ROOT`**; see **`judge/utils.py`**). Generation logs live under **`<p_run>/conversations/logs/`** for nested runs (`README`).
- **Run directory layout** — Co-locates **`<p_run>/conversations/`**, **`<p_run>/evaluations/j_*`**, and post-score **`scores/`** for nested runs (`README` diagram).
- **`.gitignore`** — Ignores **`output/`** (default artifact tree) and legacy top-level **`conversations/`** / **`evaluations/`**, while continuing to track **`publication_data/`**.

### Documentation

- **`README.md`** — Environment setup; **Recommended settings**; **Reliable VERA-MH score (automated)** (`run_recommended_vera_pipeline.sh`, **`pool_vera_scores.py`**, env vars); **`output/`**-centric nested layout, **`output/adhoc`** for single-conversation judge runs; **`run_pipeline.py`** **`--conversation-output` / `-co`** and **`--judge-output` / `-jo`** with combined resume rules (single **`j_*`** under **`p_*/evaluations/`**); **Connecting your own LLM, Agent, or API**; **`generate.py` / `judge.py` --resume`**; **v1.1** scoring / comparison wording where applicable; contributor persona TSV field list; testing/`live` notes as applicable.

[1.1.0]: https://github.com/SpringCare/VERA-MH/releases/tag/v1.1.0
