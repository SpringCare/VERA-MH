# Changing the code without changing the results

VERA-MH scores are published and compared over time, so a change to `generate/`, `judge/`, or `score/` must not move them unless that is the point of the change. Tests for each stage, and one for the whole pipeline, check this on every PR. Read this before touching any of those packages.

## Why fixed inputs, not fresh runs

Generated conversations differ on every run, and no provider setting makes them repeat: Anthropic has no seed, other providers' seeds are best-effort, and the reasoning models ignore temperature. Comparing two fresh runs can't tell a code change from ordinary randomness.

So the tests never ask a model anything. They feed the code fixed inputs (persona files, recorded model replies, scripted judge answers, a results file) and require exactly the outputs those inputs produced before. If they pass, the code behaves as it did; if a model behaves differently tomorrow, that is a separate question (see [What this doesn't cover](#what-this-doesnt-cover)).

## The guards

| Stage | Guard | What it pins | Fixed input | Test | Data |
|---|---|---|---|---|---|
| Generate | **Golden prompts** | What each persona's user model is told: the context template filled with every column, inside the shared prompt template | The personas files | `tests/unit/generate/test_golden_prompts.py` | `tests/fixtures/golden_prompts/` |
| Generate | **Generation replay** | The conversations: turn order, who speaks first, the history each model is sent, when a conversation ends | Recorded user and chatbot replies | `tests/unit/generate/test_replay_generation.py` | `tests/fixtures/replay/` |
| Judge | **Scripted judge** | How answers become ratings: the rubric walk, gates, `ASSIGN_END`, `NOT_RELEVANT>>`, severities | The judge's answer to each question | `tests/unit/judge/test_scripted_rubric_paths.py` | Scenarios in the test file |
| Score | **Golden scoring** | The score math: percentages, the VERA formula, rounding, worst band, risk breakdown | `results.csv` files | `tests/unit/score/test_scoring_golden.py` | `tests/fixtures/golden_scoring/` |
| All | **Pipeline replay** | The stages working together, including the exact prompts the judge is sent | Recorded model replies | `tests/unit/test_replay_pipeline.py` | `tests/fixtures/replay/` |

All are in the unit layer: no API keys, no network, and they run on every PR. Their helper code is in `tests/safe_changes/`; their data in `tests/fixtures/`. A change to one stage fails that stage's guard, so the failure says where to look; the pipeline replay catches what only shows when the stages run together.

- **Golden prompts** builds every SI and HFO persona's prompt with generation's own code, checks it is exactly the shared template (`data/persona_prompt_template.txt`) around the persona's part, and requires that part to match the committed text. Only the persona part is committed, so a change to the shared template shows as one diff, not two hundred.
- **Generation replay** runs `vera generate` on 2 personas and 4 turns with every user and chatbot answer taken from the recording, and requires the same transcripts.
- **Scripted judge** walks the real SI rubric with answers from a script and checks the questions asked, in order, and every dimension's rating. The expected ratings were worked out from `data/SI/rubric.tsv` by hand, so they hold the judge to the rubric, not to whatever the code does today.
- **Golden scoring** rescores each committed `results.csv` and requires the committed `scores.json` and `scores_by_risk.json`, value for value.
- **Pipeline replay** runs `vera pipeline` on the same run, with every model answer (user, chatbot, judge) taken from the recording. A call is matched on everything sent (model, system prompt, history or judge question), so the run fails with `NotInRecording` the moment the code sends a model anything different, and names the call. It then requires the same transcripts, `results.csv`, and `scores.json`, and that every recorded reply was used.

## When a guard fails

First decide whether the change was meant to alter results.

**Not meant to** (a refactor, a cleanup, a new feature that should leave existing runs alone): the failure is a bug in the change. Fix the code; don't touch the expected files.

**Meant to** (a new prompt, a rubric change, a scoring fix): regenerate the guard's expected data, then let the diff speak for the change.

| Guard | Regenerate with | Cost |
|---|---|---|
| Golden prompts | `uv run python -m tests.safe_changes.golden_prompts` | Free |
| Golden scoring | `uv run python -m tests.safe_changes.golden_scoring` | Free |
| Scripted judge | Edit the scenario's expected ratings in the test file | Free |
| Generation and pipeline replay (one recording) | `uv run python -m tests.safe_changes.replay` | A few cents of API calls |

In the PR:

1. Commit the regenerated files. Their diff shows exactly which numbers, prompts, or transcripts changed, and is what reviewers check.
2. Say why in the PR description, and add a CHANGELOG entry: a change to published numbers is a methodology change.
3. Don't regenerate to make an unexplained failure go away. If you can't say why the output changed, the change isn't understood yet.

## What this doesn't cover

The guards prove your code does what it did. They can't prove the models do: a provider can update a model behind the same ID, and an undated alias can start pointing to a new snapshot. That is watched separately: `tests/integration/test_model_availability.py` checks the configured model IDs still resolve, and `.github/workflows/live.yml` runs a live pipeline smoke test on every merge to `main` and weekly. For published or compared numbers, judge saved conversations rather than regenerating them.

## Adding to the guards

- **New persona column or target**: run `python -m tests.safe_changes.golden_prompts` (and add the target to `TARGETS` there).
- **New scoring behavior**: add a variation to `rebuild_cases` in `tests/safe_changes/golden_scoring.py` (or a case folder with its own `results.csv` in `tests/fixtures/golden_scoring/`) and run `python -m tests.safe_changes.golden_scoring --rebuild-cases`.
- **New rubric route or rubric**: add a `Scenario` to the scripted judge test, with expected ratings worked out from the rubric, not from running the code.
- **New call to a model** (a new stage, a new prompt): re-record the replay so the call is covered.
