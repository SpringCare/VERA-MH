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

### The rubric as data

Load the rubric once into its own structure, with every navigation decision already resolved, and walk that. Today the parsed rubric is a dict of dicts plus an ordered ID list, and `QuestionNavigator` re-derives "next row" and "next dimension" on every step.

```python
@dataclass(frozen=True)
class Question:
    id: str
    dimension: str
    text: str
    severity: str | None
    routes: dict[str, Route]        # answer option -> where it leads, resolved at load

@dataclass(frozen=True)
class Rubric:
    questions: dict[str, Question]
    first_question: Question
    dimensions: list[str]           # in rubric order
```

At load, `RubricConfig` turns every answer option into a `Route`: a blank GOTO becomes `GoTo(<next row>)`, an implicit Yes becomes `GoTo(<first question of the next dimension>)`, and so on. Validation runs on the same structure: every target exists, no cycles, every stop rule from [rubric-terminal-goto-vocabulary.md](./rubric-terminal-goto-vocabulary.md). `QuestionNavigator` goes away; navigating is `question.routes[answer]`. Routes are plain data, so the rubric stays free of behavior.

### What in the rubric produces each route

| Route | Rubric today (GOTO column) | Rubric after step 3 | What the walk does |
|---|---|---|---|
| `GoTo(id)` | a question ID; blank (next row); empty Answer column: `No` → next row, `Yes` → next dimension | unchanged | ask question `id` |
| `SkipDimension(id)` | `NOT_RELEVANT>>{ID}` | `SKIP_DIMENSION>>{ID}` | this dimension is Not Relevant; ask question `id` |
| `StopAssign()` | `ASSIGN_END` (only on a `Yes`) | `STOP_ASSIGN` | stop; this answer's severity counts; unvisited dimensions are Not Relevant |
| `StopScore()` | `END` reached with every dimension visited; blank GOTO on the last row | `STOP_SCORE` | stop; score what was collected |
| `StopNotRelevant()` | `END` reached with a dimension still unvisited | `STOP_NOT_RELEVANT` | stop; every dimension is Not Relevant |

Today's `END` can't be mapped at load: whether every dimension was visited depends on the path taken. Until step 3, `END` loads as a transitional `StopEnd()` route that applies today's rule when the walk reaches it, and step 3 deletes it.

### The walk

```python
async def walk(self) -> Walk:
    answers, verdicts = [], {}
    question = self.rubric.first_question
    while question:
        answer = await self.ask(question)
        answers.append(answer)                       # one record, append-only
        match question.routes[answer.option]:
            case GoTo(next_id):
                question = self.rubric.questions[next_id]
            case SkipDimension(next_id):
                verdicts[question.dimension] = NOT_RELEVANT
                question = self.rubric.questions[next_id]
            case StopAssign() | StopScore():
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

`StopAssign` and `StopScore` do the same thing in the walk; they stay separate routes because the rubric rules differ (`STOP_ASSIGN` only on a `Yes`, `STOP_SCORE` only where the questionnaire can end).

What this gives:

- **Each outcome is decided once**, in the `case` that handles it. No re-reading the rubric, no visited-dimensions heuristic, no substring search.
- **One answer list**, append-only. `question_log` goes away; `answers.tsv` and `yes_question_id` come straight from it.
- **Scoring is a pure function**, which is what [architecture.md](../architecture.md) already asks of `judge/` ("pure core; handler owns I/O") and isn't true today.
- **The stop-token rename becomes small.** Each explicit token is one route type (table above). Today's guess between a gate END and a final END becomes something the rubric states.

## Plan

Three PRs, each behavior-preserving until the last:

1. **The rubric as data, with routes resolved at load.** `RubricConfig` builds `Rubric` / `Question` / `Route`; `QuestionNavigator` goes away. Still reads today's `END` / `ASSIGN_END` / `NOT_RELEVANT>>`, with `END` as the transitional `StopEnd`. No behavior change.
2. **The walk loop and verdicts.** Replace `_ask_all_questions`, `_calculate_results`, the two handlers, and the marker matching with the walk above and a pure `score`. No behavior change; about 10 tests that assert on marker strings, and the fixture under `tests/fixtures/eval_tsv_rebuild__20260415_140037/`, are rewritten to assert on verdicts.
3. **Explicit stop tokens and validator rules.** Migrate `data/SI/rubric.tsv`, `data/HFO/rubric_HFO.tsv`, and the test rubrics; every terminating row must name its stop. This is the only PR that changes rubric files.

The guard for 1 and 2 is that scores don't move: the SI and HFO rubric integration tests and `test_judge_against_clinician_ratings.py`.

## Open questions

1. **HFO Q11.** "No Danger" routes to `END` before the last two dimensions are asked, so today all five become Not Relevant and earlier ratings are discarded. The row's note asks to keep earlier ratings and mark the rest Not Relevant, which is what `StopScore` does here. Needs the HFO rubric authors' confirmation; it can ship ahead of step 3 if they pick an existing token.
2. **Blank GOTO on a final row.** Today it means "next row", which on the last row means "finish and score". Make it an error once explicit stop tokens exist, or keep it as shorthand for `StopScore`?
