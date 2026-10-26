# Create your own evaluation

VERA-MH ships with two evaluations: suicide risk (SI) and harm from others (HFO). To evaluate a different kind of risk, you make your own. You don't need to write any code: an evaluation is a folder of files, a table of personas and a table of rubric questions, that you edit directly. The file formats are described in full in [targets.md](targets.md) and [rubric.md](rubric.md); this page is the short version.

**Who should write it.** Personas and a rubric encode expert judgment about what a risk looks like and what a good response is. Get input from clinicians, subject-matter experts, and people with lived experience of the risk, and ideally have one of those groups lead the work.

## 1. Copy an existing evaluation

Each evaluation lives in its own folder inside `data/`. Make a copy of `data/SI/` next to it and give it your evaluation's name, for example `data/MY_TARGET/`. From a terminal opened in the project folder:

```bash
cp -r data/SI data/MY_TARGET     # copy the SI folder to a new folder named MY_TARGET
```

The name of the folder is how you'll refer to your evaluation later (`--target MY_TARGET`). The copy contains:

| File | What it is |
|---|---|
| `personas.tsv` | The people the simulated user plays, one per row |
| `persona_context_template.txt` | Which details of each persona the simulated user is told |
| `rubric.tsv` | The questions the judge answers about each conversation |
| `rubric_prompt_beginning.txt`, `question_prompt.txt` | The judge's instructions. These are generic and usually don't need changes |
| `manifest.json` | The list of the files above. Leave it as is unless you rename a file |

`.tsv` files are tables; keep them tab-separated when you save.

## 2. Write the personas

In `personas.tsv`, each row is one person the simulated user will play in a conversation.

- **The columns are up to you.** Pick the details that matter for your kind of risk: background, how the risk shows up, how openly they talk about it, how they communicate. Keep the `Name` column.
- **Cover the whole range.** Include people with no risk, some risk, and the most urgent risk. The rubric's first questions check whether the chatbot overreacts to people who aren't at risk, which only works if some personas aren't.
- **Record each persona's risk level** in its own column. To have it copied into the results, add the column name to the manifest, for example `"persona_annotation_columns": ["Risk Level"]`.

`persona_context_template.txt` is the text the simulated user receives about its persona. Each `{Column Name}` in it is replaced by that persona's value from the table, so every column name in braces must exactly match a column in `personas.tsv`. Add or remove lines to control what the simulated user knows.

## 3. Write the rubric

In `rubric.tsv`, each row is a question the judge answers about the conversation, or one of a question's possible answers; [reading-a-rubric.md](reading-a-rubric.md) explains how a rubric is read. Keep the structure and change the wording.

- **Keep the five dimensions and their exact names:** Detects Potential Risk, Confirms Risk, Guides to Human Care, Supportive Conversation, Follows AI Boundaries. Scoring depends on these names for now.
- **Rewrite the questions for your kind of risk.** Phrase each one as "did the chatbot do something wrong?", so "No" is the good answer. Use the `Examples` column to tell the judge how to decide.
- **Set each question's severity** in the `Severity` column: `Red` for responses that could increase the risk of harm to the user (for example, increase the risk of suicide), `Yellow` for an unpleasant conversational experience that is unlikely to cause lasting harm beyond the conversation. Within a dimension, put all the Red questions before the Yellow ones.
- **Keep the flow.** The first questions decide whether the conversation involves your kind of risk at all; one later question classifies how urgent the risk is, which picks the matching Guides to Human Care questions. The `GOTO` column says which question comes after each answer; the meanings of its special values are in [rubric.md](rubric.md#rubric-tsv-contract).

When you run the evaluation, VERA-MH checks the rubric first and stops with a message naming the question if something is wrong, for example a `GOTO` pointing to a question that doesn't exist.

## 4. Run it

These commands run the whole evaluation: the simulated users talk to the chatbot you're testing, the judge rates each conversation with your rubric, and the ratings are turned into scores. Replace the names in `< >` with the models you want to use.

```bash
# A quick test with 2 personas
uv run vera pipeline -c <chatbot> -u <user-model>:1 -j <judge>:1 --target MY_TARGET --sample 2

# The full evaluation, with every persona
uv run vera pipeline -c <chatbot> -u <user-model>:1 -j <judge>:1 --target MY_TARGET
```

- `-c` is the chatbot being tested, `-u` the model playing the personas, `-j` the model acting as judge.
- `--target MY_TARGET` is your folder's name.
- `--sample 2` uses only the first 2 personas. Run this first, and read a couple of conversations and the judge's answers in the `output/` folder before the full run.

Then do a gut check on a first batch of results before any formal validation: read the conversations and the judge's answers, and check that the rubric catches the range of harms you expect and doesn't flag good responses. Adjust the rubric and run again until it does.

Before trusting the scores, check that the judge agrees with experts: have experts in your kind of risk rate a sample of conversations with the same rubric and compare their answers with the judge's, as was done for SI. Scores from different evaluations measure different things, so never compare them with each other.

## Judging real conversations instead

To apply your rubric to existing conversations (for example, real user conversations) rather than simulated ones, you don't need personas: skip step 2 and leave the copied persona files in place, since the manifest still has to name them. Put each conversation in its own `.txt` file in one folder, written like the generated ones (`user:` and `chatbot:` turns, separated by a blank line), then judge and score them:

```bash
# Judge every .txt file in the folder with your rubric; results go under the output folder
uv run vera judge -j <judge>:1 --rubric MY_TARGET --conversations <folder> -o <output folder>

# Score the results.csv the judge wrote
uv run vera score -r <output folder>/<judging folder>/results.csv
```
