# Targets, personas, and prompts

A **target** is a self-contained evaluation bundle: the personas the user model plays, the prompts that turn a persona into instructions, the clinical rubric, and the judge prompts. Scores from different targets measure different things and are never comparable.

## Targets that ship

| Target | Focus | Folder |
|--------|-------|--------|
| `SI` | Suicidal ideation: detecting and confirming risk, guiding to human care, supportive conversation, AI boundaries | [`data/SI/`](../data/SI/) |
| `HFO` | Harm from others: risk of violence or abuse toward the user | [`data/HFO/`](../data/HFO/) ([rubric summary](../data/HFO/rubric_summary.md)) |

Select one with `--target SI`. `vera generate --target all` runs every target; `vera judge` and `vera pipeline` take one target at a time. Checked-in run configs for both targets are in [`configs/`](../configs/).

## The manifest

Each target folder has a `manifest.json` naming its files, relative to the folder itself so the bundle stays portable:

```json
{
  "rubric_file": "rubric.tsv",
  "rubric_prompt_beginning_file": "rubric_prompt_beginning.txt",
  "question_prompt_file": "question_prompt.txt",
  "personas": ["personas.tsv"],
  "persona_context_template_file": "persona_context_template.txt"
}
```

| Field | Used by | Purpose |
|-------|---------|---------|
| `personas` | generate | One or more persona TSVs |
| `persona_context_template_file` | generate | Which persona columns go into the user model's prompt, and how |
| `rubric_file` | judge | The rubric's question flow |
| `rubric_prompt_beginning_file` | judge | The judge's system prompt |
| `question_prompt_file` | judge | Template for asking each rubric question |
| `persona_annotation_columns` | judge | Optional persona columns copied into `results.csv` (HFO uses `Active Life Threat`) |

`--target` uses the whole bundle. `--personas <target>` (generate) and `--rubric <target>` (judge) use only one side. Every flag also accepts a manifest path.

## Personas

Each TSV row is one persona. SI personas ([`data/SI/personas.tsv`](../data/SI/personas.tsv), 100 rows) include demographics, suicide risk level, current and past suicidal thoughts and behaviors, how openly they disclose risk, communication style, reaction to chatbot responses, diagnoses, treatment access, stigma, financial stress, social isolation, discrimination exposure, background, recent stressors, and a final seed phrase. HFO personas describe the harm type and imminence, the person of concern, risk indicators, the user's fear, and barriers to safety.

The full persona set gives a more thorough and less variable score than a slice; use `--sample N` only for smoke tests.

## How a persona becomes a prompt

1. The target's persona context template (for example [`data/SI/persona_context_template.txt`](../data/SI/persona_context_template.txt)) is filled in with the persona's TSV values.
2. The result is inserted at `{persona_context}` in the shared [`data/persona_prompt_template.txt`](../data/persona_prompt_template.txt), which holds behavior common to every target—staying in character as the person seeking help, and when to end the conversation.

To change which persona details the user model sees, edit the target's context template. Edit the shared template only for behavior that should apply to every target.

## Adding personas or a target

- **More personas:** add rows to the target's TSV with the same columns.
- **A new target:** create `data/<name>/` with a `manifest.json` and the files it names; `--target <name>` then finds it. Scoring still assumes the SI dimension names, so read [rubric.md](rubric.md) for the compatibility requirements first. [judge.md](judge.md) explains how the rubric's question flow works.
