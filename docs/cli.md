# The `vera` CLI

`vera.py` is the single entry point for VERA-MH 2.0. It has four commands:

| Command | What it does |
|---------|--------------|
| `vera generate` | Simulate conversations between persona-playing user models and the chatbot under test |
| `vera judge` | Rate a folder of conversations against a rubric |
| `vera score` | Turn judge ratings into VERA-MH scores and charts |
| `vera pipeline` | Generate, judge, and score in one invocation, passing each stage's output to the next |

Invoke it as `uv run python vera.py <command> ...`. Every command accepts `--help`.

For the reasoning behind these choices, see [vera-cli-use-cases.md](vera-cli-use-cases.md) and [architecture.md](architecture.md).

## Contents

- [Concepts](#concepts)
- [Two ways to define a run: flags or config](#two-ways-to-define-a-run-flags-or-config)
- [`vera generate`](#vera-generate)
- [`vera judge`](#vera-judge)
- [`vera score`](#vera-score)
- [`vera pipeline`](#vera-pipeline)
- [Continuing an interrupted run: `--into`](#continuing-an-interrupted-run---into)
- [Model parameters and reasoning effort](#model-parameters-and-reasoning-effort)
- [Output layout](#output-layout)
- [Not built yet](#not-built-yet)

## Concepts

Three model roles, each with a one-letter flag used consistently across commands:

- **`-u` user**—the model that role-plays a persona, simulating a person talking to the chatbot.
- **`-c` chatbot**—the model or system under test.
- **`-j` judge**—the model that rates a transcript against the rubric.

Model lists take a count suffix: `-u gpt-5.2:2` runs the full persona set twice with GPT 5.2, and `-j gpt-5.4:3` runs three judge instances. Several models can be given at once: `-u gpt-5.2:1 claude-opus-4-5-20251101:1`. Use dated model IDs (for example `claude-sonnet-4-5-20250929`); shorthand aliases may not resolve.

A **target** is a reusable evaluation bundle—personas, persona prompt, rubric, and judge prompts—stored in `data/<target>/manifest.json`. `--target SI` selects all of it at once. To mix components, use `--personas <target>` on `generate` and `--rubric <target>` on `judge`. Either flag also accepts a manifest path. See [targets.md](targets.md) for what ships.

## Two ways to define a run: flags or config

A run is defined **either** by command-line flags **or** by a JSON config—never both. Combining them is an error, not a merge. The config can come from:

- `--config run.json`—a file
- `--config -`—stdin
- the `VERA_RUN_CONFIG` environment variable—inline JSON, handy for CI

A few **invocation controls** describe how to run, not what the run is, so they may accompany either form:

| Flag | Meaning |
|------|---------|
| `--sample N` | Debug cap: load only N personas per file (`generate`, `pipeline`) or judge only N conversations (`judge`) |
| `-d`, `--debug` | Debug logging |
| `--print` | Print the fully resolved invocation (as a `VERA_RUN_CONFIG=...` command) and exit without running anything |
| `--into <folder>` | Continue an existing run (`generate`, `judge`); see [below](#continuing-an-interrupted-run---into) |

Flag defaults (30 turns, `output/`, unlimited concurrency, persona speaks first) are a CLI convenience only. **A config must state every behavior field explicitly**, including `null` where no limit is intended, so a stored config fully describes a run. Unknown fields are rejected.

Every command prints its resolved config when it starts, so the log of a run shows exactly what it did.

Paths inside a config resolve against the repository root, not the working directory, so checked-in configs work from anywhere. Paths given as flags resolve against the working directory.

`--print` is the easiest way to get a starting config: build the run with flags, add `--print`, and copy the JSON. Checked-in examples live in [`configs/`](../configs/):

```json
{
  "target": "SI",
  "generation": {
    "chatbot": {"name": "<model-under-test>", "repeats": 1},
    "user": [
      {"name": "gpt-5.2", "repeats": 1},
      {"name": "claude-opus-4-5-20251101", "repeats": 1}
    ],
    "turns": 30,
    "output": "output",
    "max_concurrent": 10,
    "max_total_words": null,
    "persona_speaks_first": true,
    "sessions": null
  },
  "judging": {
    "models": [{"name": "gpt-5.4", "repeats": 1, "reasoning_effort": "low"}],
    "max_concurrent": 10,
    "per_judge": false
  },
  "scoring": {
    "personas": "data/SI/personas.tsv",
    "skip_risk_analysis": false
  }
}
```

A top-level `target` expands into the concrete persona and rubric paths. Instead of `target`, you can state `generation.personas` + `generation.persona_context_template` and `judging.rubrics` explicitly; setting both is an error. The full config schema is in [vera-cli-use-cases.md](vera-cli-use-cases.md#configjson-shape).

## `vera generate`

Simulates conversations. Requires `-c`, at least one `-u`, and `--target` or `--personas`.

```bash
uv run python vera.py generate \
  -c gpt-4o \
  -u claude-sonnet-4-5-20250929:1 \
  --target SI
```

| Flag | Description | Default |
|------|-------------|---------|
| `-c`, `--chatbot` | Chatbot model under test | required |
| `-u`, `--user` | User model(s), `model[:repeats]`; each repeat runs the full persona set | required |
| `--target` | Target name or manifest path; `all` runs every target (one run each) | — |
| `--personas` | Use only this target's personas and persona prompt (mutually exclusive with `--target`) | — |
| `-t`, `--turns` | Maximum conversation turns | `30` |
| `-o`, `--output` | Parent directory for new run folders | `output` |
| `--max-concurrent` | Maximum concurrent conversations; set this if the chatbot times out or rate-limits | unlimited |
| `--max-total-words` | Cap on total response words per conversation | unlimited |
| `--provider-speaks-first` | Chatbot speaks first | persona speaks first |
| `--sessions` | Comma-separated session types run in order per persona (e.g. `intake,coaching`) | one session |
| `--user-params` | `k=v,...` provider parameters for every `-u` model | none |
| `--chatbot-params` | `k=v,...` provider parameters for the `-c` model | none |
| `--into` | Continue an existing run folder instead of creating one | — |

**Conversation ending.** A conversation stops at `--turns`, or earlier when the persona emits the end-of-conversation signal defined in the persona prompt. Custom chatbot clients can define their own termination signals.

**Multi-session conversations.** With `--sessions`, each persona runs the listed session types in sequence, each with its own transcript (`..._run1_s1_intake.txt`, `..._run1_s2_coaching.txt`). A chatbot client can implement `prepare_sessions()`, `enter_session()`, and `first_speaker` on [`LLMInterface`](../llm_clients/llm_interface.py) to set up sessions or control who speaks first; see [evaluating.md](evaluating.md).

## `vera judge`

Rates one generation run's conversations. Requires at least one `-j`, `--conversations`, and `--target` or `--rubric`.

```bash
uv run python vera.py judge \
  -j gpt-5.4:1 \
  --conversations output/<generation-run> \
  --target SI
```

| Flag | Description | Default |
|------|-------------|---------|
| `-j`, `--judge` | Judge model(s), `model[:instances]` | required |
| `--conversations` | Generation run folder to judge (exactly one; judge folders separately and pool the results) | required |
| `--target` | Target name or manifest path supplying the rubric | — |
| `--rubric` | Use only this target's rubric and judge prompts (mutually exclusive with `--target`) | — |
| `-o`, `--output` | Parent directory for the new evaluation folder | `<generation run>/evaluations/` |
| `--max-concurrent` | Maximum concurrent judge workers | unlimited |
| `--per-judge` | Apply `--max-concurrent` per judge model instead of in total | total |
| `--judge-params` | `k=v,...` provider parameters for every `-j` model | none |
| `--into` | Continue an existing evaluation folder instead of creating one | — |

**GPT 5.4** (`gpt-5.4`) is the recommended judge: an inter-rater reliability study found it closer to human clinician ratings than the earlier GPT-4o + Claude Sonnet 4.5 pair.

Each evaluation folder contains one `.tsv` per conversation (dimension, rating, and the judge's reasoning), a `results.csv` with every conversation's ratings, and per-conversation judge logs in `logs/`. Use `results.csv` to find conversations with a rating you care about, then open that conversation's `.tsv` to see which rubric question produced it.

Differences from the legacy `judge.py`: there is no single-conversation mode, and a flat folder of `.txt` transcripts (rather than a generation run) requires an explicit `-o`.

## `vera score`

Aggregates a judge run's `results.csv` into VERA-MH scores and charts. The results file is the only required input.

```bash
uv run python vera.py score \
  -r output/<generation-run>/evaluations/<evaluation-run>/results.csv \
  --personas data/SI/personas.tsv
```

| Flag | Description | Default |
|------|-------------|---------|
| `-r`, `--results` | The judge run's `results.csv` | required |
| `-o`, `--output` | Path for the scores JSON | `scores/scores.json` beside `results.csv` |
| `--personas` | Personas file for the risk-level breakdown | none (breakdown skipped) |
| `--skip-risk-analysis` | Skip the risk-level breakdown even when `--personas` is given | off |

The risk-level breakdown currently works only with SI personas. What each output file contains is described in [scoring.md](scoring.md).

## `vera pipeline`

Runs generate, judge, and score in sequence. Its flag form accepts only the three model roles and the target:

```bash
uv run python vera.py pipeline \
  -c gpt-4o \
  -u gpt-5.2:1 claude-opus-4-5-20251101:1 \
  -j gpt-5.4:1 \
  --target SI
```

| Flag | Description |
|------|-------------|
| `-c`, `--chatbot` | Chatbot model under test |
| `-u`, `--user` | User model(s), `model[:repeats]` |
| `-j`, `--judge` | Judge model(s), `model[:instances]` |
| `--target` | Target supplying personas and rubric (`all` is not supported) |
| `--user-params`, `--chatbot-params`, `--judge-params` | Provider parameters per role |

Anything stage-specific—output directory, concurrency, turns, the scoring personas file—comes from `--config`, because a flag like `--max-concurrent` would mean different things to generation and judging. A pipeline config has `generation`, `judging`, and `scoring` sections, and omits `judging.conversations` because generation supplies it. The flag form skips the risk-level breakdown; set `scoring.personas` in a config to get it.

Each user model produces its own generation and evaluation run, and the command prints the evaluation folders at the end. To combine them into one score, see [pooling](scoring.md#pooling-several-evaluations). How the command resolves its input is explained in [pipeline.md](pipeline.md).

## Continuing an interrupted run: `--into`

Generation and judging both spend real LLM calls, so a run that dies partway shouldn't start over. `--into <run folder>` points a new invocation at an existing run folder and skips whatever is already written there:

```bash
# The original run. Creates a new folder under output/ and prints its path.
uv run python vera.py generate -c gpt-4o -u claude-sonnet-4-5-20250929:1 --target SI

# It died partway. Same command, plus --into naming the folder it created:
uv run python vera.py generate -c gpt-4o -u claude-sonnet-4-5-20250929:1 --target SI \
  --into output/p_claude_sonnet_4_5_20250929__a_gpt_4o__t30__r1__20260713_153000
```

`vera judge` takes the same flag, pointed at the evaluation folder it was writing:

```bash
uv run python vera.py judge -j gpt-5.4:1 --conversations output/<generation-run> --target SI \
  --into output/<generation-run>/evaluations/<evaluation-run>
```

- **`--into` replaces `--output`.** They are mutually exclusive: `--output` names a parent to create a new run under; `--into` names the existing run to continue. Keep everything else in the original command the same.
- **You supply the path.** Nothing auto-discovers your last run yet. `--into` fails if the folder doesn't exist, so a typo can't silently start a fresh run.
- **It works with `--config` too**, and is not recorded in the run's config: a run finished across two invocations is the same run.

This is a stateless, per-stage skip that works out what's left from the files on disk. It is deliberately not called `--resume`; that name is reserved for the planned `vera resume` (see [Not built yet](#not-built-yet)). `vera pipeline` does not take `--into`; continue each stage with its own command.

## Model parameters and reasoning effort

Pass provider parameters per role with `--user-params`, `--chatbot-params`, and `--judge-params`, as comma-separated `key=value` pairs:

```bash
--chatbot-params temperature=0.5,max_tokens=1500
--judge-params reasoning_effort=high
```

In a config, put the same keys on the model entry: `{"name": "gpt-5.4", "repeats": 1, "reasoning_effort": "low"}`. Judges default to `temperature=0` unless overridden.

Reasoning-capable models each take a different parameter:

| Provider | Parameter | Values |
|----------|-----------|--------|
| OpenAI (`gpt-5.x`, o-series) | `reasoning_effort` | `low`, `medium`, `high` |
| Claude | `thinking_effort` | `low`, `medium`, `high`, `max` |
| Gemini 3 | `thinking_level` | `low`, `high` |
| Gemini 2.5 | `thinking_budget` | integer token budget |

`thinking_effort` is VERA-MH shorthand, not a native Anthropic field: it is translated into the right `thinking`/`effort`/`budget_tokens` combination for each model in [`llm_clients/claude_llm.py`](../llm_clients/claude_llm.py). Claude forces `temperature=1` whenever thinking is on. The other three pass straight through. Per-provider details are in [evaluating.md](evaluating.md#reasoning--extended-thinking-by-provider).

To route Claude, OpenAI, or Gemini calls through a gateway such as LiteLLM, set `API_BASE_URL` (or the per-provider `ANTHROPIC_BASE_URL` / `OPENAI_BASE_URL` / `GOOGLE_BASE_URL`) in `.env`. The matching `*_API_KEY` must then be a key the gateway accepts.

## Output layout

Each generation run gets its own folder under `--output`. Its evaluations nest inside it:

```
output/
└── p_<user>__a_<chatbot>__t<turns>__r<repeats>__<timestamp>/
    ├── conversations/
    │   ├── <id>_<persona>_<model>_run1.txt
    │   └── logs/                   # one generation log per conversation
    └── evaluations/
        └── j_<judge>__.../
            ├── <conversation>.tsv  # per-conversation ratings and reasoning
            ├── results.csv         # all ratings
            ├── logs/               # per-conversation judge logs
            └── scores/             # written by vera score
```

Folder names still use the legacy `p_` (user) and `a_` (chatbot) prefixes. A layout organized by target and chatbot (`output/<target>/c_<chatbot>/...`) is specified in [vera-cli-use-cases.md](vera-cli-use-cases.md#naming) and not yet implemented.

## Not built yet

These are specified in [architecture.md](architecture.md) but not implemented:

- **`vera pool`**—combine several evaluations into one score. Until then, use [`scripts/pool_vera_scores.py`](scoring.md#pooling-several-evaluations).
- **`vera resume`**—continue a run from its own `config.json` and `state.json`, across stages and without being given a path. Until then, use [`--into`](#continuing-an-interrupted-run---into).
