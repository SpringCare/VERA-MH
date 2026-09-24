# Rubric terminal GOTO vocabulary

Status: Proposed
Date: 2026-09-16

## Context

The rubric `GOTO` column overloads `END` with two incompatible meanings, and the
judge cannot tell them apart:

1. **A screening gate failed** — the conversation is out of scope, so every
   dimension is legitimately `Not Relevant`. `data/SI/rubric.tsv` uses `END`
   this way at Q1 and Q3.
2. **The questionnaire finished** — the last question was answered, so the
   collected answers should be scored normally.

`QuestionNavigator.get_next_question` returns the `GOTO` verbatim for both, and
`LLMJudge._ask_all_questions` returns the terminating question ID on any `END`
(`judge/llm_judge.py:544`). `_calculate_results` then takes the discard-all
branch unconditionally (`judge/llm_judge.py:325-331`), so reaching the end of a
rubric is indistinguishable from failing its first gate. Every dimension is
overwritten with `Not Relevant` and a full set of collected answers is thrown
away at scoring time.

`docs/rubric.md:123` actively teaches the mistake. It tells rubric authors to
"use `END` to stop without assigning severity", which is what a terminal `No`
answer wants to express and is exactly what silently discards the evaluation.

`tests/fixtures/rubric_assign_end.tsv` already contains the defect: its terminal
`Q2` routes both `Yes` and `No` to `END`, so a conversation that reaches `Q2`
loses its answers — including, on `Yes`, a Yellow severity finding the judge had
just recorded. The fixture is used only for navigation assertions, so nothing
observes the discarded scores.

A second, quieter overload sits in the same column. An empty `GOTO` means "fall
through to the next row", but on the final row `_get_next_row_question` returns
`None`, the navigation loop exits, and `_ask_all_questions` returns `None` —
which *is* the score-normally path. So "questionnaire complete" is currently
expressible only by omission, and only from the last row. `data/SI/rubric.tsv`
scores correctly because its terminal question happens to have a blank `GOTO`;
it relies on an unnamed behavior rather than on a rule.

The `ASSIGN_END` path is not affected. `_handle_assign_end` only overwrites
dimensions not yet visited (`judge/llm_judge.py:781`), and
`_calculate_results:299-310` re-derives the `ASSIGN_END` route from the rubric
precisely so the last-question case still works when no unvisited dimension
remains.

## Decision

Replace `END` with two named tokens and rename the remaining special values so
every terminating cell shares a `STOP_` prefix. Each token names the intended
*interpretation* rather than the navigation side effect:

| Token | Meaning | Replaces |
|-------|---------|----------|
| `STOP_NOT_RELEVANT` | Screening gate failed: conversation out of scope, all dimensions `Not Relevant` | `END` on gate questions |
| `STOP_SCORE` | Questionnaire complete: score the collected answers | `END` on terminal questions, and blank `GOTO` on any terminal row |
| `STOP_ASSIGN` | Stop, assign this question's severity to the current dimension, mark remaining dimensions `Not Relevant` | `ASSIGN_END` |
| `SKIP_DIMENSION>>{ID}` | Mark the current dimension `Not Relevant`, continue at `{ID}` | `NOT_RELEVANT>>{ID}` |
| `{ID}` or blank | Jump to a question, or fall through to the next row | unchanged |

`END` is removed rather than redefined. Keeping it as an alias for either
meaning preserves the ambiguity this record exists to eliminate.

`RubricConfig._validate_navigation` gains three rules that make the defect
unrepresentable at load time:

1. **Every terminating row carries an explicit stop token.** A row that ends the
   flow — the last question, or any row whose blank `GOTO` falls off the end —
   must name `STOP_SCORE`, `STOP_NOT_RELEVANT`, or `STOP_ASSIGN`. This retires
   "termination by omission" and is what turns `data/SI/rubric.tsv` from
   accidentally correct into explicitly correct.
2. **`STOP_ASSIGN` is valid only on a `Yes`.** Already enforced
   (`judge/rubric_config.py:416-431`); carried over unchanged.
3. **`STOP_NOT_RELEVANT` must be reachable with at least one dimension
   unvisited.** It must genuinely be a gate. A `STOP_NOT_RELEVANT` positioned
   where every dimension has already been answered is a rubric error, because it
   can only discard completed work — the failure this record addresses becomes a
   load-time rejection instead of silent data loss.

### Rejected alternative: a compound token

A slot-based token such as `ASSIGN>>END>>NOT_RELEVANT`, encoding the three
independent effects (where to go next, what happens to the current dimension,
what happens to the others), was considered and rejected. It is explicit, but it
turns one cell into a micro-DSL with roughly eighteen slot combinations, most of
them meaningless, each needing a validator rule and a documentation sentence.
Rubric cells are authored by clinical reviewers in a spreadsheet, so the cost
lands on the least technical participants, and the failure mode only moves from
"which `END` did I mean" to "which combination did I mean". The closed set of
named outcomes above covers every case the rubrics actually use.

### Rejected alternative: clear the terminal `END` cell

Blanking the terminal `GOTO` restores correct scoring for one rubric with no
code change. It is rejected as the permanent fix because it leaves the trap
armed for the next rubric author, and it relies on the same
termination-by-omission behavior that rule 1 exists to remove. It remains
acceptable as an immediate unblock for an in-flight run.

### Migration sequencing

Two pull requests, deliberately not combined:

1. **Correctness fix.** Gate the discard-all branch at
   `judge/llm_judge.py:325-331` on whether any dimension is still unvisited: all
   dimensions answered means the `END` was a completion, so fall through to
   `_determine_dimension_scores`; otherwise keep the existing gate behavior.
   This mirrors the discriminator already used for `ASSIGN_END` a few lines
   above. Add a regression test over a terminal-`END` fixture, and correct
   `tests/fixtures/rubric_assign_end.tsv`. Affected runs can then be re-scored
   from their persisted `answers/answers.tsv` without re-judging, since the
   answers were always collected — only the scoring step discarded them.
2. **Vocabulary migration.** The renames and the three validator rules, as a
   behavior-preserving change landing on top of PR 1's passing test.

Splitting them keeps a data-correctness fix independent of a schema migration.
Combined, a migration defect and a scoring defect would be indistinguishable in
the resulting numbers.

## Consequences

- **Breaking change to the rubric TSV format.** Every existing rubric must be
  migrated; `END` and `ASSIGN_END` no longer load. This is intentional — a
  rubric carrying the old vocabulary is a rubric whose terminal behavior is
  ambiguous. Rubrics are small, version-controlled, and few, so the migration is
  a mechanical edit rather than a compatibility surface worth maintaining.
- **Blast radius of the rename.** `judge/question_navigator.py` (token set),
  `judge/rubric_config.py` (validation), `judge/llm_judge.py` (branching and
  synthetic markers), `judge/score_utils.py:515` (a regex parsing the
  `(ASSIGN_END)` marker out of reasoning prose), `judge/answers.py`,
  `data/SI/rubric.tsv`, `tests/fixtures/rubric_assign_end.tsv`, `docs/rubric.md`,
  `docs/judge.md`, `docs/ARCHITECTURE-SPINE.md`, and roughly seventy test
  assertions that match on the marker strings.
- **Interacts with the planned `dimension_verdicts` refactor** recorded in
  `TODO`, which replaces the synthetic `ASSIGN_END` / `NOT_RELEVANT>>` marker
  records in `dimension_answers` with an explicit verdict map. Both touch the
  same code region. The verdict map subsumes the marker-string matching that
  this rename would otherwise have to carry forward, so it should land first or
  alongside step 2; the correctness fix in step 1 does not depend on either.
- **Historical scores computed before the correctness fix are not comparable**
  to scores computed after it. Any rubric whose terminal question routed to
  `END` under-counts `total_relevant_conversations` and computes its aggregate
  over the surviving subset, so a published aggregate can rest on a small
  fraction of the conversations that were actually judged. Re-score before
  comparing across the fix.
- **`Not Relevant` remains a navigation outcome.** A cleaner model treats
  screening as a declared rubric phase and out-of-scope as a property of the
  conversation, which is what the `total_relevant_conversations` field in
  `scores.json` already approximates. That is a larger refactor; the token split
  captures most of the benefit and does not foreclose it.
