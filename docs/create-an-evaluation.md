# Create your own evaluation

An evaluation in VERA-MH is a **target**: a folder in `data/` holding the personas the user model plays and the rubric the judge applies. To evaluate a different kind of risk, you write a new target. No code changes are needed. The formats are covered in [targets.md](targets.md) and [rubric.md](rubric.md); this page is the short version.

## 1. Start from a copy

```bash
cp -r data/HFO data/MY_TARGET
```

HFO is the more complete example (it uses `persona_annotation_columns`). Rename the files if you like, and update `data/MY_TARGET/manifest.json` to match.

## 2. Write the personas

`personas.tsv` has one row per person the user model will play.

- **The columns are yours.** Choose the details that matter for your risk: background, how the risk shows up, how openly they disclose it, communication style. Keep a `Name` column.
- **Cover the whole range.** Include personas with no risk, low risk, and the most urgent risk. The rubric's in-scope check and false-positive items only get exercised by people who aren't at risk.
- **Show the user model what it needs.** `persona_context_template.txt` decides which columns go into the user model's prompt. Every `{Column}` it uses must exist in the TSV.
- **Record a risk level** in its own column, and list it under `persona_annotation_columns` in the manifest. The judge then copies it into `results.csv`, so you can break scores down by it.

## 3. Write the rubric

`rubric.tsv` is the question flow the judge walks. Keep the shape and change the content:

- **Keep the five dimensions and their names** (Detects Potential Risk, Confirms Risk, Guides to Human Care, Supportive Conversation, Follows AI Boundaries). Scoring requires these names for now; see [rubric.md](rubric.md#current-compatibility-assumptions).
- **Rewrite the questions for your risk.** Phrase each as "did the chatbot do something wrong?", so "No" is the good answer, and put what a judge needs to answer reliably in the `Examples` column.
- **Mark each question's severity.** `Red` is high potential for harm, `Yellow` is suboptimal. Within a dimension, put every Red question before any Yellow one; the loader checks this.
- **Keep the flow:** an in-scope check first, then the dimensions in order, with a risk-level question choosing which Guides to Human Care items apply. Route between questions with the `GOTO` column (`END`, `ASSIGN_END`, `NOT_RELEVANT>>ID`; see [rubric.md](rubric.md#rubric-tsv-contract)).
- **Check the judge prompts.** `rubric_prompt_beginning.txt` and `question_prompt.txt` are generic in the shipped targets. Edit them only if your judge needs domain context.

The rubric is validated when it loads: a `GOTO` to a missing question, a cycle, or Red after Yellow fails with the question named.

## 4. Run it

```bash
# Smoke test: 2 personas, then read the transcripts and the judge's answers
uv run python vera.py pipeline -c <chatbot> -u <user-model>:1 -j <judge>:1 --target MY_TARGET --sample 2

# Full run
uv run python vera.py pipeline -c <chatbot> -u <user-model>:1 -j <judge>:1 --target MY_TARGET
```

Before trusting the scores, check that the judge agrees with clinicians: have clinicians rate a sample of conversations with the same rubric and compare their answers with the judge's, as was done for SI. Scores from different targets measure different things and are never comparable.
