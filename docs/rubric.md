# Adding a Compatible Rubric

The current rubric loader and judge support multiple rubric bundles. However,
scoring and visualization are not yet fully rubric-driven. A new rubric works
end to end only under the compatibility assumptions below.

## Current compatibility assumptions

A new rubric must use these exact dimension names:

1. `Detects Potential Risk`
2. `Confirms Risk`
3. `Guides to Human Care`
4. `Supportive Conversation`
5. `Follows AI Boundaries`

These names are currently loaded from `data/SI/rubric.tsv` by the scoring,
visualization, and comparison code. The judge can evaluate other dimensions,
but rebuilding `results.csv` would omit them.

The rubric must also retain the existing VERA scoring semantics:

- `Red` severity maps to `High Potential for Harm`.
- `Yellow` severity maps to `Suboptimal but Low Potential for Harm`.
- A dimension with no triggered severity maps to `Best Practice`.
- A skipped dimension may map to `Not Relevant`.

Risk-level scoring additionally assumes personas use the levels `None`, `Low`,
`High`, and `Immediate`. Supply a compatible persona file or pass
`--skip-risk-analysis`.

## Bundle layout

A complete bundle used for both generation and judging has this shape:

```text
data/NEW_RUBRIC/
├── rubric_manifest.json
├── rubric.tsv
├── rubric_prompt_beginning.txt
├── question_prompt.txt
├── personas.tsv
└── persona_context_template.txt
```

`personas.tsv` and `persona_context_template.txt` are needed for conversation
generation. Judging existing conversations only requires the rubric and judge
prompt files.

Manifest paths are relative to the directory containing the manifest:

```json
{
  "rubric_file": "rubric.tsv",
  "rubric_prompt_beginning_file": "rubric_prompt_beginning.txt",
  "question_prompt_file": "question_prompt.txt",
  "personas": ["personas.tsv"],
  "persona_context_template_file": "persona_context_template.txt"
}
```

The rubric beginning prompt must accept `{conversation}`. The question prompt
must accept `{question}`, `{examples_section}`, and `{options}`.

## Rubric TSV contract

Use the same tab-separated columns as `data/SI/rubric.tsv`:

- `Question ID`
- `Dimension`
- `Severity`
- `Risk Type`
- `Question`
- `Examples`
- `Answer`
- `GOTO`
- `Auto Answer` (optional)

The loader enforces these navigation rules:

- Every primary question row has a unique, non-empty question ID.
- Every primary question row explicitly declares its dimension.
- Rows containing additional answers leave `Question ID` blank.
- Every GOTO question target exists.
- The navigation graph contains no cycles.
- Question IDs are opaque strings; they do not need to be numeric or sequential.
- `Auto Answer=true` is valid only when the question has exactly one explicit
  answer. The judge selects that answer without an LLM call.

Supported terminal and special GOTO values are:

- `END`: stop and mark all dimensions Not Relevant.
- `ASSIGN_END`: assign the current question's severity and mark later dimensions
  Not Relevant.
- `NOT_RELEVANT>>{ID}`: mark the current dimension Not Relevant and continue at
  the specified question ID.

For more detail about ordinary Yes/No navigation, see
[How the Rubric Works](judge.md).

## Running a compatible rubric

Judge existing conversations with the new bundle:

```bash
uv run python judge.py \
  --folder output/my-run \
  --judge-model <model> \
  --rubrics data/NEW_RUBRIC/rubric_manifest.json
```

For a complete generation, judging, and scoring run, select the bundle for both
generation and judging and provide its personas for risk analysis:

```bash
uv run python run_pipeline.py \
  --user-agent <model> \
  --provider-agent <model> \
  --judge-model <model> \
  --rubrics data/NEW_RUBRIC/rubric_manifest.json \
  --rubric-manifest data/NEW_RUBRIC/rubric_manifest.json \
  --personas-tsv data/NEW_RUBRIC/personas.tsv
```

`--rubrics` and `--rubric-manifest` are independent: the former selects the
evaluation rubric, while the latter selects generation personas and their context
template. There is currently no symbolic rubric-name shorthand, so pass the full
manifest path.

When generating an improvement report, pass the new TSV explicitly so question
IDs are joined to the correct question text:

```bash
uv run python scripts/summarize_results.py \
  --results output/my-run/evaluations/my-evaluation/results.csv \
  --rubric data/NEW_RUBRIC/rubric.tsv
```

## Current limitation

Rubrics with different dimension names are not yet supported end to end. The
remaining work is to persist the selected rubric's dimensions and identity with
evaluation output, then pass those dimensions into result aggregation, scoring,
visualization, and comparison instead of loading them globally from the SI rubric.
