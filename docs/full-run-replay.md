# Full-run replay

The full-run replay checks that a code change leaves a real VERA-MH run unchanged. It reruns a saved run of the recommended SI profile (2 user models × 100 personas, 30 turns, judged and pooled) through `vera pipeline` without calling any model, and requires every call and every output to match. CI checks a 10-conversation slice on every PR; the full 200 run locally in about 5 seconds.

This page explains how it works and what to do when it fails. For where it sits among the other guards, see [safe-changes.md](safe-changes.md). The code is `tests/safe_changes/full_run.py`.

## The idea

A model never answers the same way twice, so two fresh runs can't tell a code change from randomness. The replay takes the randomness out: every model reply comes from the saved run. The code then runs for real (building prompts, alternating turns, walking the rubric, scoring, writing files) with only the model calls faked.

The fake sits behind `LLMFactory.create_llm`, the boundary between the pipeline and the LLM clients. That's deliberate: the replay checks the pipeline, not how each client turns a call into a provider request. That belongs in each client's own unit tests.

## What's in a saved run

```
tests/fixtures/full_run/
├── saved/                                  the saved run
│   ├── p_gpt_5_2__…/
│   │   ├── conversations/*.txt             transcripts: what each model said
│   │   └── evaluations/j_…/answers/by_question/*.tsv
│   │                                       the judge's answer to each question, in order
│   ├── p_claude_opus_4_5_20251101__…/      the same for the other user model
│   └── inputs/                             what each client was handed (frozen once)
│       ├── settings.json                   each client's settings, the --sample, the source commit,
│       │                                   and a checksum of each saved transcript and answers file
│       ├── chatbot_system_prompt.txt
│       ├── persona_system_prompts/<persona>.txt
│       ├── judge_system_prompt.txt         with <CONVERSATION> where the transcript goes
│       └── judge_questions/<question id>.txt
└── expected/                               every other output: scores, results.csv,
                                            judge TSVs, answers.tsv, run records
```

The transcripts and judge answers are the run's own files. A saved run doesn't record the system prompts or client settings, though, so `inputs/` holds them, written once by `freeze-inputs`. The committed slice's inputs were frozen from `main` (the commit is in `settings.json`), and match what `main`'s own pipeline hands its clients for these conversations.

The saved transcripts play two parts: they are the replies the fake client gives, and the transcripts the run must reproduce. An edit to one moves both sides at once, so a check would still pass, and the judge would still give its saved answers, whatever the new text says. So `settings.json` keeps a checksum of each saved transcript and judge answers file, and a check refuses to run once one has changed.

`expected/` holds the outputs the saved run can't supply: its scores were computed by the code of its day, and file formats have changed since.

## What happens during a check

Take one conversation, three turns long, with the persona Luca:

```
user: hi
chatbot: no
user: bye
```

`check` runs the pipeline. Each time it creates a client or calls one, the fake client checks what it was handed, then answers from the saved run:

| Call | The pipeline hands the client | Checked against | The fake returns |
|---|---|---|---|
| create user model | model `gpt-5.2`, settings, Luca's persona prompt | `settings.json`, `persona_system_prompts/Luca.txt` | |
| create chatbot | model `gpt-5.4-mini`, settings, system prompt | `settings.json`, `chatbot_system_prompt.txt` | |
| 1. user model starts | nothing | | `hi` |
| 2. chatbot | history: `user: hi` | the transcript so far | `no` |
| 3. user model | history: `user: hi`, `chatbot: no` | the transcript so far | `bye` |
| create judge | model `gpt-5.4`, settings, system prompt with the transcript | `settings.json`, `judge_system_prompt.txt` with the transcript filled in | |
| 4. judge, question 1 | the question prompt | `judge_questions/1.txt`, and that the saved answers ask question 1 next | the saved answer |
| 5. … | | | |

Every turn of the history is checked: its number, its speaker, its text, and that its `input` is the turn before. Speakers are compared as `Role`s, so renaming a `Role` value is no difference. The history is checked as the pipeline hands it over, not as a client turns it into provider messages.

When the run finishes, its outputs are compared:

- transcripts and judge answers, with the saved run's files;
- everything else, with `expected/`.

Absolute paths, timestamps, and the random filename tags are ignored. Logs and charts aren't compared.

## What a failure means

| Message | What happened |
|---|---|
| `saved file changed since the inputs were frozen: …` | Someone edited the saved run. The check stops before running anything |
| `input differs: persona system prompt Luca`, with a diff | The code built a different prompt, or passed different settings (`client user gpt-5.2`) |
| `input not in the frozen inputs: client …` | The code created a client the saved run didn't have, such as one with another model name |
| `frozen input never handed to a client: …` | The code stopped using something it used before |
| `('gpt-5.2', 'Luca'): turn 2 of the history is by USER, not CHATBOT` | The conversation loop changed: order, speakers, or text |
| `('gpt-5.2', 'Luca'): asked for turn 31 of 30` | The conversation ran longer than the saved one |
| `('gpt-5.2', 'Luca'): judge question 4 should be question 7` | The judge walked the rubric differently |
| `…: 12 of the saved turns used` | The conversation ended early |
| `output differs: …/scores.json`, with a diff | Scoring or file writing changed |

A difference in an input doesn't stop the run, so every differing input is listed, then the outputs are compared. A call the saved run can't answer does stop it: that conversation can't continue, so only the first such call is reported.

## Commands

```bash
# Check the committed slice (CI runs this as tests/unit/test_full_run_replay.py)
uv run python -m tests.safe_changes.full_run check tests/fixtures/full_run/saved tests/fixtures/full_run/expected

# Check the full 200, locally
uv run python -m tests.safe_changes.full_run check output output/full_run_expected
```

Run the full check before merging a change to how runs work. The full saved run (the `p_*` folders and the pooled `j_*`) and its `inputs/` and `full_run_expected/` live in the gitignored `output/`.

## When a check fails

First decide whether the change was meant to alter what a run does.

**Not meant to** (a refactor, a cleanup): the failure is a bug in the change. Fix the code; don't touch `inputs/` or `expected/`.

**Meant to**, and an input changed (a new prompt, a new client setting): freeze the inputs again, and review the diff of `inputs/`. It shows exactly what each client is now told.

```bash
uv run python -m tests.safe_changes.full_run freeze-inputs tests/fixtures/full_run/saved --sample 5 --replace
```

**Meant to** change the saved run itself (for example, fixing a transcript): the replay can't tell you how the models would have answered the new text, so prefer a new saved run. If you do edit it, freeze the inputs again with `--replace`, which records the new checksums.

**Meant to**, and an output changed (a scoring fix, a new column): write the expected outputs again, and review the diff of `expected/`.

```bash
uv run python -m tests.safe_changes.full_run update-expected tests/fixtures/full_run/saved tests/fixtures/full_run/expected
```

`update-expected` refuses while the saved run has changed, any input differs from the frozen ones, or the saved transcripts and judge answers aren't reproduced. Inputs come first. Then commit the new files, and say why in the PR and the CHANGELOG, as [safe-changes.md](safe-changes.md#when-a-guard-fails) describes. Never regenerate to make a failure you can't explain go away.

The replies stay what the models said to the old inputs. After a prompt change, the replay still checks that the code works, but the saved conversations no longer show how the models would answer the new prompt. Judge a fresh run for that.

## Adding a saved run

1. Run the pipeline for real with the settings in `pipeline_config` (`full_run.py`), and keep its `p_*` folders, each with its `evaluations/j_*`.
2. If the judge run predates `answers/by_question/`, write those files from its logs: `full_run answers-from-logs <saved run>`.
3. Freeze its inputs: `full_run freeze-inputs <saved run> [--sample N]`. `--sample N` keeps the first N personas of each user model, as `vera pipeline --sample` does, and the saved run must hold exactly those conversations.
4. Write its expected outputs: `full_run update-expected <saved run> <expected>`.
5. Check: `full_run check <saved run> <expected>`.

Freeze from the code the run was made with, or code you've confirmed builds the same prompts, so `inputs/` holds what the models were actually told.

## What it doesn't cover

- **The LLM clients.** The fake replaces them, so how `OpenAILLM` or `ClaudeLLM` builds a provider request, including how the history becomes human and AI messages, is left to their unit tests.
- **Paths the saved run didn't take**: other targets, the chatbot speaking first, `--sessions`, `--into`, a word limit. The other guards and unit tests cover those.
- **The models.** A model can change behind the same ID. See [safe-changes.md](safe-changes.md#what-this-doesnt-cover).
