# A judge flow that decides each outcome once

**Status:** draft for discussion. Covers roadmap items "Explicit `dimension_verdicts`" and "Split the overloaded `END` token" ([Rubric-agnostic scoring](../roadmap.md#rubric-agnostic-scoring)); builds on [rubric-terminal-goto-vocabulary.md](./rubric-terminal-goto-vocabulary.md).

## Problem

The judge's job is small: walk the rubric's questions, collect answers, score each dimension. What makes `judge/llm_judge.py` hard to follow is that **every way a walk can end is decided in three places**:

1. **Navigator** (`judge/question_navigator.py`, `get_next_question`). Returns the next question *and* the raw GOTO string, and treats `END`, `ASSIGN_END`, and `NOT_RELEVANT>>` as special.
2. **Walk loop** (`_ask_all_questions`, 158 lines). Parses the GOTO string again, writes synthetic answer records into `dimension_answers`, and on END returns a question ID that means one of three things.
3. **Result step** (`_calculate_results`). Works out after the fact what happened: re-reads the rubric (`_answer_triggered_assign_end`), checks whether every dimension was visited to tell a gate END from a final END, and `_determine_dimension_scores` searches answer text for the marker strings:

   ```python
   if "ASSIGN_END" in answers[0].get("question", "") or "NOT_RELEVANT>>" in answers[0].get("reasoning", ""):
   ```

Knock-on effects:

- `dimension_answers` isn't a true record. `NOT_RELEVANT>>` replaces a dimension's real answers with a synthetic one, so a second list, `question_log`, exists only to keep what was actually asked.
- Scores depend on English sentences: reword a marker message, or have a reasoning text that mentions "ASSIGN_END", and the score changes.
- Downstream, `yes_question_id` is regexed out of the reasoning prose (`utils/judge_results.py`), including a special case for the ` (ASSIGN_END)` suffix.
- Verbose output is printed twice per event (in the handler and in the caller).

## Proposal

Interpret GOTO **once**, when the rubric loads, into a typed route. The walk acts on routes; scoring reads a plain record.

```python
# Parsed from the rubric's GOTO column at load time (judge/rubric_config.py)
GoTo(next_id)             # a question ID, or blank = next row
NotRelevantThen(next_id)  # today's NOT_RELEVANT>>{ID}
StopAndScore()            # today's ASSIGN_END, and END after the last question
StopNotRelevant()         # today's END at a screening gate
```

```python
async def walk(self) -> Walk:
    answers, verdicts = [], {}
    question = self.rubric.first_question
    while question:
        answer = await self.ask(question)
        answers.append(answer)                       # one record, append-only
        match question.route_for(answer):
            case GoTo(next_id):
                question = self.rubric[next_id]
            case NotRelevantThen(next_id):
                verdicts[question.dimension] = NOT_RELEVANT
                question = self.rubric[next_id]
            case StopAndScore():
                verdicts |= {d: NOT_RELEVANT for d in self.unvisited(answers)}
                break
            case StopNotRelevant():
                verdicts = {d: NOT_RELEVANT for d in self.rubric.dimensions}
                break
    return Walk(answers, verdicts)


def score(walk: Walk, dimension: str) -> Result:      # pure: no LLM, no rubric re-reads
    if dimension in walk.verdicts:
        return Result(walk.verdicts[dimension])
    return score_from_severity(walk.answers_for(dimension))
```

What this gives:

- **Each outcome is decided once**, in the `case` that handles it. No re-reading the rubric, no visited-dimensions heuristic, no substring search.
- **One answer list**, append-only. `question_log` goes away; `answers.tsv` and `yes_question_id` come straight from it.
- **Scoring is a pure function**, which is what [architecture.md](../architecture.md) already asks of `judge/` ("pure core; handler owns I/O") and isn't true today.
- **The stop-token rename becomes small.** Each explicit token (`STOP_SCORE`, `STOP_NOT_RELEVANT`, …) is one route type. Today's guess between a gate END and a final END becomes something the rubric states.

## Plan

Three PRs, each behavior-preserving until the last:

1. **Typed routes parsed at load.** `RubricConfig` turns the GOTO column into routes; the navigator shrinks to a lookup. Still reads today's `END` / `ASSIGN_END` / `NOT_RELEVANT>>`, mapping `END` to `StopAndScore` or `StopNotRelevant` by today's rule. No behavior change.
2. **The walk loop and verdicts.** Replace `_ask_all_questions`, `_calculate_results`, the two handlers, and the marker matching with the walk above and a pure `score`. No behavior change; about 10 tests that assert on marker strings, and the fixture under `tests/fixtures/eval_tsv_rebuild__20260415_140037/`, are rewritten to assert on verdicts.
3. **Explicit stop tokens and validator rules.** Migrate `data/SI/rubric.tsv`, `data/HFO/rubric_HFO.tsv`, and the test rubrics; every terminating row must name its stop. This is the only PR that changes rubric files.

The guard for 1 and 2 is that scores don't move: the SI and HFO rubric integration tests and `test_judge_against_clinician_ratings.py`.

## Open questions

1. **HFO Q11.** "No Danger" routes to `END` before the last two dimensions are asked, so today all five become Not Relevant and earlier ratings are discarded. The row's note asks to keep earlier ratings and mark the rest Not Relevant, which is `StopAndScore` here. Needs the HFO rubric authors' confirmation; it can ship ahead of step 3 if they pick an existing token.
2. **Blank GOTO on a final row.** Today it means "next row", which on the last row means "finish and score". Make it an error once explicit stop tokens exist, or keep it as shorthand for `StopAndScore`?
3. **Where routes live.** On the question (`question.route_for(answer)`), or in a separate table the walk consults? The first reads better; the second keeps rubric data free of behavior.
