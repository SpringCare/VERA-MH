# One folder per run, organized by stage

**Status:** draft for discussion. Changes the target layout in [architecture.md, Data flow and artifacts](../architecture.md#data-flow-and-artifacts); connected to [run-lifecycle.md](./run-lifecycle.md) (naming, `config.json`, conversation files), [job-loop.md](./job-loop.md) (the pipeline as one job list), and [terminal-interface.md](./terminal-interface.md) (`events.jsonl`).

## Problem

A pipeline run with two user models today leaves its output in four places:

```
output/
├── p_gpt_5_2__a_gpt_5_4__t30__r1__20261015_194125/          ← user model 1: conversations, then its evaluation
│   ├── conversations/
│   └── evaluations/j_gpt-5.4x1_..._20261015_203122_567__p_gpt_5_2__a_gpt_5_4__t30__r1__20261015_194125/
├── p_claude_opus_4_5__a_gpt_5_4__t30__r1__20261015_200935/  ← user model 2, same again
│   ├── conversations/
│   └── evaluations/j_gpt-5.4x1_..._20261015_204154_506__p_claude_opus_4_5__...
└── j_gpt-5.4x1__p_claude_opus_4_5+gpt_5_2__a_gpt_5_4__t30__r1__20261015_205205/   ← the headline score, beside them
```

To find the result of one run you need four timestamps, and nothing groups them. The target layout in architecture.md tidies the names (`c_<chatbot>/u_<user>_<sha>_<ts>/`, `pooled/`) but keeps the same split: one run folder per user model, plus a separate pooled folder for the score that matters.

That split mirrors how the code used to run (one generation pool and one judging pool per user model), not what a run is. With the pipeline as one job list across all user models ([job-loop.md](./job-loop.md)), a run is one thing: one model under test, one set of settings, every user model's conversations, judged and scored together.

## Proposal

One folder per run, under the target and the model under test, with one subfolder per stage, the same stages as the code and the commands (`generate`, `judge`, `score`):

```
output/
└── SI/                                      ← target: scores never cross this boundary
    └── gpt-5.4/                             ← model under test: every run against it
        └── 20261015-194125_3f9a1c/          ← one run: timestamp, then short config hash
            ├── config.json                  ← the resolved settings, written once (run-lifecycle.md)
            ├── config.json.sha256
            ├── state.json                   ← resume progress
            ├── events.jsonl                 ← what happened, for scripts (terminal-interface.md)
            ├── generate/
            │   ├── gpt-5.2/                 ← one folder per user model
            │   │   ├── Brian_r1.jsonl       ← persona, repeat; header carries every detail
            │   │   └── logs/
            │   └── claude-opus-4-5/
            ├── judge/
            │   └── gpt-5.4_8d2e10/          ← one folder per judging: judge model, settings hash
            │       ├── evaluations/         ← one file per conversation
            │       ├── results.csv          ← every conversation, with a user_model column
            │       └── logs/
            └── score/
                └── gpt-5.4_8d2e10/          ← one folder per judging, same name
                    ├── scores.json          ← all user models together, and each one separately
                    ├── scores_by_risk.json
                    └── *.png
```

- **The run folder is the result.** Everything a run produced is under one path, and the headline score is always `score/<judging>/scores.json`. The separate `pooled/` folder goes away: scoring every user model together *is* the run's score, and `scores.json` also carries each user model's own numbers.
- **Stages mirror the code.** `generate/`, `judge/`, and `score/` match the packages and the `vera` commands, so where a file lives says which step wrote it. `vera judge --into <run>` adds a folder under `judge/`; `vera score --into <run>` adds one under `score/`.
- **Judging can be repeated.** A run's conversations can be judged again with another judge model or settings without touching the first judging: each gets its own `judge/<judge>_<hash>/` and matching `score/` folder. This replaces the standalone `evaluations/<config-sha256>/` grouping.
- **No prefixes.** `c_`, `u_`, `j_`, `p_`, `a_` existed so a flat folder name could say what each part was. Here a name's position in the tree says it, so folders are just model names.
- **Shorter conversation names.** A conversation is unique by persona and repeat within its user model's folder, so the random six-character tag goes, and with it the resume bug where a persona name with a space or dot is never found (run-lifecycle.md).

## What changes elsewhere

| Today or in the current target | With this layout |
|---|---|
| One run root per user model | One per run; `create_run_root` (run-lifecycle.md) is called once per invocation |
| `pooled/` folder and `vera pool` for the headline score | The run's own `score/`; `vera pool` remains only for combining *different* runs |
| User model read from the folder name | A `user_model` column in `results.csv` and in each conversation's header |
| Standalone judging in `evaluations/<config-sha256>/` | A new `judge/` folder inside the run that produced the conversations |
| `--into <run>` resumes one user model's generation | `--into <run>` resumes the whole run: missing conversations, then missing evaluations and scores |

## Open questions

1. **Stage folders or artifact folders?** `generate/`, `judge/`, `score/` mirror the code and commands. The alternative names what's inside (`conversations/`, `evaluations/`, `scores/`). Proposal: stages, so the layout and the code read the same way.
2. **Judging names.** `gpt-5.4_8d2e10` (judge model plus a hash of its settings) keeps repeat judgings apart. Two judge models in one judging (today's `-j a:1 b:1`) would need a joined name; `+`-joined like today, or one judging per judge model?
3. **Conversations from elsewhere.** Judging transcripts that `vera generate` didn't produce has no run to nest under. A run folder with no `generate/` (its `config.json` naming the transcript folder), or a separate `output/<target>/external/`?
4. **Combining runs.** Pooling two different runs (for example, two days of the same configuration) still needs a home. Under the model under test (`gpt-5.4/pooled/…`), or only as an explicit output path?
5. **Old folders.** Read by explicit path during the deprecation window from run-lifecycle.md, or converted once by a small script?
