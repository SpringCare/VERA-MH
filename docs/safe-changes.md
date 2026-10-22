# Changing the code without changing the results

VERA-MH scores are published and compared over time, so a change to `generate/`, `judge/`, or `score/` must not move them unless that is the point of the change. Three tests check this on every PR. Read this before touching any of those packages.

## Why fixed inputs, not fresh runs

Generated conversations differ on every run, and no provider setting makes them repeat: Anthropic has no seed, other providers' seeds are best-effort, and the reasoning models ignore temperature. Comparing two fresh runs can't tell a code change from ordinary randomness.

So the tests never ask a model anything. They feed the code fixed inputs (a results file, scripted judge answers, recorded model replies) and require exactly the outputs those inputs produced before. If they pass, the code behaves as it did; if a model behaves differently tomorrow, that is a separate question (see [What this doesn't cover](#what-this-doesnt-cover)).

## The three guards

| Guard | What it pins | Fixed input | Test | Data |
|---|---|---|---|---|
| **Golden scoring** | The score math: percentages, the VERA formula, rounding, worst band, risk breakdown | `results.csv` files | `tests/unit/score/test_scoring_golden.py` | `tests/fixtures/golden_scoring/` |
| **Scripted judge** | How answers become ratings: the rubric walk, gates, `ASSIGN_END`, `NOT_RELEVANT>>`, severities | The judge's answer to each question | `tests/unit/judge/test_scripted_rubric_paths.py` | Scenarios in the test file |
| **Replay** | The whole pipeline, including the exact prompts every model is sent | Recorded model replies | `tests/unit/test_replay_pipeline.py` | `tests/fixtures/replay/` |

All three are in the unit layer: no API keys, no network, and they run on every PR. Their helper code is in `tests/safe_changes/`; their data in `tests/fixtures/`.

- **Golden scoring** rescores each committed `results.csv` and requires the committed `scores.json` and `scores_by_risk.json`, value for value.
- **Scripted judge** walks the real SI rubric with answers from a script and checks the questions asked, in order, and every dimension's rating. The expected ratings were worked out from `data/SI/rubric.tsv` by hand, so they hold the judge to the rubric, not to whatever the code does today.
- **Replay** runs `vera pipeline` on 2 personas and 4 turns with every model answer taken from a recording. A call is matched on everything sent (model, system prompt, history or judge question), so the run fails with `NotInRecording` the moment the code sends a model anything different, and names the call. It then requires the same transcripts, `results.csv`, and `scores.json`, and that every recorded reply was used.

## When a guard fails

First decide whether the change was meant to alter results.

**Not meant to** (a refactor, a cleanup, a new feature that should leave existing runs alone): the failure is a bug in the change. Fix the code; don't touch the expected files.

**Meant to** (a new prompt, a rubric change, a scoring fix): regenerate the guard's expected data, then let the diff speak for the change.

| Guard | Regenerate with | Cost |
|---|---|---|
| Golden scoring | `uv run python -m tests.safe_changes.golden_scoring` | Free |
| Scripted judge | Edit the scenario's expected ratings in the test file | Free |
| Replay | `uv run python -m tests.safe_changes.replay` | A few cents of API calls |

In the PR:

1. Commit the regenerated files. Their diff shows exactly which numbers, prompts, or transcripts changed, and is what reviewers check.
2. Say why in the PR description, and add a CHANGELOG entry: a change to published numbers is a methodology change.
3. Don't regenerate to make an unexplained failure go away. If you can't say why the output changed, the change isn't understood yet.

## What this doesn't cover

The guards prove your code does what it did. They can't prove the models do: a provider can update a model behind the same ID, and an undated alias can start pointing to a new snapshot. That is watched separately: `tests/integration/test_model_availability.py` checks the configured model IDs still resolve, and `.github/workflows/live.yml` runs a live pipeline smoke test on every merge to `main` and weekly. For published or compared numbers, judge saved conversations rather than regenerating them.

## Adding to the guards

- **New scoring behavior**: add a variation to `rebuild_cases` in `tests/safe_changes/golden_scoring.py` (or a case folder with its own `results.csv` in `tests/fixtures/golden_scoring/`) and run `python -m tests.safe_changes.golden_scoring --rebuild-cases`.
- **New rubric route or rubric**: add a `Scenario` to the scripted judge test, with expected ratings worked out from the rubric, not from running the code.
- **New call to a model** (a new stage, a new prompt): re-record the replay so the call is covered.
