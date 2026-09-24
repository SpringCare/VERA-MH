# Legacy scripts (deprecated)

> **These scripts will be removed soon.** Use the [`vera` CLI](cli.md) for new work. This page exists so workflows that still depend on a script keep running while they migrate.

| Legacy script | Replacement |
|---------------|-------------|
| `generate.py` | `vera generate` |
| `judge.py` | `vera judge` |
| `python -m judge.score` | `vera score` |
| `run_pipeline.py` | `vera pipeline` |
| `scripts/run_recommended_vera_pipeline.sh` | `vera pipeline --config` with [`configs/recommended-SI.json`](../configs/recommended-SI.json), then [pooling](scoring.md#pooling-several-evaluations) |

**The flags are not the same.** The `vera` CLI uses `-u`/`-c`/`-j` for user/chatbot/judge. The scripts use `-p` for the chatbot, and reuse `-c` and `-r` for unrelated things (`-c` is `--max-concurrent` in `generate.py` and `--conversation` in `judge.py`; `-r` is `--runs` in `generate.py` and `--rubrics` in `judge.py`). Don't copy flags between the two.

The scripts also take rubric bundles only as full manifest paths (`data/SI/rubric_manifest.json`); there is no `--target SI` shorthand. Omitting the flag defaults to SI.

## `generate.py`

```bash
uv run python generate.py -u gpt-4o -p gpt-4o -t 6 -r 1
```

| Flag | `vera generate` equivalent | Description |
|------|----------------------------|-------------|
| `-u`, `--user-agent` | `-u` | User (persona) model |
| `-uep`, `--user-agent-extra-params` | `--user-params` | Extra params for the user model, e.g. `temperature=0.7,max_tokens=1000` |
| `-p`, `--provider-agent` | `-c` | Chatbot under test |
| `-pep`, `--provider-agent-extra-params` | `--chatbot-params` | Extra params for the chatbot |
| `-t`, `--turns` | `-t` | Turns per conversation (required here; defaults to 30 in `vera`) |
| `-r`, `--runs` | `-u model:N` | Runs per persona (required) |
| `-o`, `--output` | `-o` | Parent directory for the new `p_*` run folder (default `output`) |
| `-c`, `--max-concurrent` | `--max-concurrent` | Max concurrent conversations |
| `-w`, `--max-total-words` | `--max-total-words` | Cap on total words per conversation |
| `-mp`, `--max-personas` | `--sample` | Limit the number of personas loaded |
| `--rubric-manifest` | `--target` / `--personas` | Manifest to load personas from |
| `-psf`, `--provider-speaks-first` | `--provider-speaks-first` | Chatbot speaks first |
| `--sessions` | `--sessions` | Comma-separated session types per persona |
| `--resume` | `--into <folder>` | Continue a run; set `--output` to the existing `p_*` folder. Models, turns, and runs must match |
| `-i`, `--run-id` | none | Custom run ID |
| `-pfm`, `--provider-first-message` | none | Static first chatbot message (no LLM call) when the chatbot speaks first |
| `-psp`, `--provider-start-prompt` | none | Prompt sent to the chatbot for its first turn |
| `-usm`, `--user-first-message` | none | Static first persona message (no LLM call) when the persona speaks first |
| `-usp`, `--user-start-prompt` | none | Prompt sent to the user model for its first turn |
| `-d`, `--debug` | `-d` | Debug logging |

## `judge.py`

```bash
uv run python judge.py -f output/<generation-run>/ -j gpt-5.4
```

| Flag | `vera judge` equivalent | Description |
|------|-------------------------|-------------|
| `-f`, `--folder` | `--conversations` | Generation run folder, or a flat folder of `.txt` transcripts |
| `-c`, `--conversation` | none | Judge a single transcript; output defaults to `output/adhoc/single_<timestamp>__<stem>/` |
| `-j`, `--judge-model` | `-j` | Judge model(s), `model` or `model:count` |
| `-jep`, `--judge-model-extra-params` | `--judge-params` | Extra params for the judge (default `temperature=0`) |
| `-r`, `--rubrics` | `--target` / `--rubric` | Rubric manifest (default `data/SI/rubric_manifest.json`) |
| `-l`, `--limit` | `--sample` | Limit conversations judged |
| `-o`, `--output` | `-o` | Parent for the new `j_*` folder. Defaults to `<generation run>/evaluations/`, or `evaluations/` in the working directory for flat folders (`vera judge` requires `-o` there instead) |
| `--resume` | `--into <folder>` | Continue judging; set `-o` to the existing `j_*` folder |
| `-m`, `--max-concurrent` | `--max-concurrent` | Max concurrent workers |
| `-pj`, `--per-judge` | `--per-judge` | Apply `--max-concurrent` per judge model |
| `-vw`, `--verbose-workers` | none | Verbose worker logging |

## `judge/score.py`

```bash
uv run python -m judge.score -r output/<generation-run>/evaluations/<evaluation-run>/results.csv
```

Same flags as `vera score` (`-r`, `-o`, `--skip-risk-analysis`), except `--personas-tsv` / `-p` defaults to `data/SI/personas.tsv`. `vera score --personas` has no default and skips the risk breakdown when omitted.

## `run_pipeline.py`

Runs `generate.py`, `judge.py`, and `judge/score.py` in sequence:

```bash
uv run python run_pipeline.py \
  --user-agent claude-sonnet-4-5-20250929 \
  --provider-agent gpt-4o \
  --runs 2 \
  --turns 10 \
  --judge-model gpt-5.4 \
  --max-personas 5
```

It accepts the generation flags above, plus `--judge-model-extra-params`, `--judge-max-concurrent`, `--judge-per-judge`, `--judge-limit`, `--judge-verbose-workers`, `--rubrics`/`--rubric-manifest`, `--personas-tsv`, and `--skip-risk-analysis`. Output locations are `--conversation-output` / `-co` and `--judge-output` / `-jo`.

To resume:

- **`--resume-generate`**—set `-co` to the existing `p_*` folder.
- **`--resume-judge`**—set `-jo` to the existing `j_*` folder.
- **Both**—set `-co` to the `p_*` folder; it must contain exactly one `j_*` under `evaluations/`.

See `uv run python run_pipeline.py --help` for everything.

## `scripts/run_recommended_vera_pipeline.sh`

Runs the recommended profile through `run_pipeline.py`, once per user model, then pools the two evaluations:

```bash
./scripts/run_recommended_vera_pipeline.sh <provider-agent-model> [extra run_pipeline.py args]
```

Its defaults, and the `VERA_*` environment variables that override them, are documented in the script. Their `vera pipeline` equivalents:

| Script variable | `vera pipeline` equivalent |
|-----------------|----------------------------|
| `VERA_OUTPUT_PARENT` | `generation.output` |
| `VERA_MAX_CONCURRENT` | `generation.max_concurrent` and `judging.max_concurrent` |
| `VERA_MAX_PERSONAS` | `--sample N` |
| `VERA_USER_A`, `VERA_USER_B` | entries in `generation.user` |
| `VERA_JUDGE`, `VERA_JUDGE_EXTRA_PARAMS` | `judging.models` |
| `VERA_POOL_OUTPUT` | `-o` on `scripts/pool_vera_scores.py` |
| `VERA_POOL_SKIP_RISK` | `--skip-risk-analysis` on `scripts/pool_vera_scores.py` |
