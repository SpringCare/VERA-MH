# What `vera` shows while it runs

**Status:** draft for discussion. Builds on the event stream in the job-loop design ([#266](https://github.com/SpringCare/VERA-MH/pull/266), `docs/design/job-loop.md`, decision 4) and the run record in the run-lifecycle design ([#268](https://github.com/SpringCare/VERA-MH/pull/268), `docs/design/run-lifecycle.md`); both are still in review.

## Problem

A two-persona smoke run of `vera pipeline` prints about 150 lines, and the ones that matter are hard to find. From a real run where the chatbot model name was a leftover placeholder:

```
{ ...40 lines of resolved config JSON... }
🔄 Generating conversations with the following parameters:
  - Persona model: {'model': 'gpt-5.2'}
  ...14 more parameter lines...
Running 2 conversations with max concurrency: 10 (2 workers)
[Worker 0] (1/2) Abigail (run 1)
Creating OpenAI LLM with parameters:
  Model: gpt-5.2
  ...4 more lines, repeated for every client...
Skipped conversation (Abigail, run 1): Unsupported model: <chatbot model>
...the same block again for every persona and user model...
✅ Generated 0 conversations → output/p_gpt_5_2__a_<chatbot model>__... (2 skipped)
🎯 LLM Judge | Models: gpt-5.4x1
Traceback (most recent call last):
  ...
FileNotFoundError: No .txt files found in: output/p_gpt_5_2__a_<chatbot model>__.../conversations
```

- **The cause is buried.** "Unsupported model" appears four times between setup noise, generation then reports success with a ✅, and the run dies one stage later with a traceback about a missing folder.
- **No sense of progress.** There's no count of what's done, no elapsed time, no idea which conversation is slow. A full run is hours of `[Worker 3] (57/198) Brian (run 1)` lines, and a hung provider looks exactly like a working one.
- **Unreadable when redirected.** Output is block-buffered when stdout isn't a terminal, so `| tee run.log` shows nothing until the run ends (roadmap backlog).
- **Not parseable.** There's no machine-readable record of what happened. Everything is free-form `print()`: about 290 calls across `generate/`, `judge/`, `score/`, `llm_clients/`, and `vera_cli/`, and only three modules use `logging`.
- **The end result isn't summarized.** The headline score is one line among hundreds, and nothing says which rubric items drove it.

## Three audiences, three channels

| Who | Channel | What they get |
|---|---|---|
| A person watching | stdout | A live display in a terminal; short, flushed, timestamped lines otherwise |
| A person debugging | log files in the run folder | Everything: client settings, raw calls, retries, tracebacks (per-job logs already exist) |
| A script or notebook | `events.jsonl` in the run folder | One JSON line per event, the same events the display shows |

**Domain code stops printing.** Generation, judging, and scoring emit events to a reporter (job-loop design, [#266](https://github.com/SpringCare/VERA-MH/pull/266), decision 4); the reporter decides what to show. Client setup details (`Creating OpenAI LLM with parameters…`) and worker chatter go to the log files, not stdout.

## What it looks like

### Start of run

The 40-line config dump becomes a header. The full resolved config is written to the run folder (`config.json`, run-lifecycle design, [#268](https://github.com/SpringCare/VERA-MH/pull/268)) and still shown by `--print`.

```
vera pipeline | SI | Mexican Spanish
  chatbot  gpt-5.4
  users    gpt-5.2, claude-opus-4-5     99 personas x 1 repeat, 30 turns
  judge    gpt-5.4
  output   output/SI/c_gpt-5.4/u_gpt-5.2_3f9a1c_20261015_194125
```

### During the run, in a terminal

```
vera pipeline | SI | gpt-5.4                                         elapsed 0:42:10

  Generate  [##############......]  148/198   2 failed   3.1M tokens  est. $9.40
  Judge     [#########...........]   91/198              5.6M tokens  est. $14.10
  Score     waiting for all conversations

  gpt-5.2          Brian     turn 14/30   chatbot responding...      1m12s
  gpt-5.2          Sarah     turn  6/30   user responding...         0m31s
  claude-opus-4-5  Emma      judging      Q23 Supportive Conversation
  claude-opus-4-5  Levi      judging      Q9  Guides to Human Care

  FAIL claude-opus-4-5 | Abigail   rate limited after 4 attempts   logs/...Abigail....log
```

One progress bar per stage, one line per job in flight saying what it's doing, and failures listed as they happen with a pointer to their log.

### During the run, redirected (`| tee run.log`, CI)

```
19:41:25 generate  start  gpt-5.2          Brian     1/198
19:44:06 generate  done   gpt-5.2          Brian     30 turns   2m41s
19:44:07 judge     start  gpt-5.2          Brian
19:44:52 judge     done   gpt-5.2          Brian     45s
19:45:10 generate  FAIL   claude-opus-4-5  Abigail   rate limited after 4 attempts   logs/...
```

One flushed line per event: grep-able, and never silent while work is happening.

### When a stage produces nothing

```
FAIL Generation produced 0 of 4 conversations. Every one failed with:
    Unsupported model: <chatbot model>
  Stopping before judging. Logs: output/.../conversations/logs/
```

The run stops at the stage that broke, names the cause once, and exits non-zero.

### End of run

```
Done in 1:52:40 | 198 conversations | 1 failed (rerun with: vera pipeline --into output/SI/...)
Tokens: generate 6.2M, judge 11.4M | est. cost $51.30

                   conversations   VERA   Best practice   Harm
  gpt-5.2                     99   66.1           43.0%   3.1%
  claude-opus-4-5             98   69.4           45.6%   2.9%
  pooled                     197   67.9           44.3%   3.0%

  Most-flagged items
    Q35  doesn't remind the user it's an AI       170 / 197
    Q28  not collaborative, back-and-forth        123 / 197
    Q13  resources in a long, dense list           78 / 197

  Results  output/SI/c_gpt-5.4/pooled/u_gpt-5.2+claude-opus-4-5_.../
```

## Verbosity

| Flag | Shows |
|---|---|
| (default) | Header, live display or event lines, end summary |
| `--quiet` | End summary and errors only, for scripts |
| `--verbose` | Adds per-turn events (user/chatbot responding) as lines |
| `--debug` | Today's level of detail, to the log files and stdout |

`events.jsonl` is written at every level. A notebook or dashboard reads it instead of parsing stdout.

## Implementation

1. **Events and a reporter interface, built into the job loop** (#266): job started, job finished, job failed, turn started (user/chatbot), judge question asked, stage finished, with token usage on each finished call. Each stage gets them as it moves onto the loop, so no hooks are added to the runners the loop deletes.
2. **Plain-line reporter, `events.jsonl`, fail-fast, and the end summary.** This alone fixes buffering, the buried cause, and the missing summary, and works without a terminal.
3. **Move the ~290 `print()` calls to logging or events.** Mechanical, package by package; client setup and worker chatter go to the log files.
4. **The live terminal display**, using [rich](https://github.com/Textualize/rich) (`Progress`, `Live`, `Table`), in a terminal only. When stdout isn't a terminal, the plain-line reporter from step 2 stays in charge: rich's `Live` doesn't turn updates into lines there, it shows only its final frame, which would bring back the silent-until-finished output under `| tee`.

Steps 1 and 2 land with the loop: its first PR ships the plain-line reporter and `events.jsonl`, so each stage is readable under `| tee` and in CI as soon as it runs on the loop. Steps 3 and 4 follow, each its own PR.

## Decisions

Settled in review on #271.

1. **`rich` is added** for the live display. Its own progress bar draws Unicode, so the display uses a small custom column for the ASCII bar (see 4).
2. **Tokens and cost are shown** per stage, in the live display and the end summary. Responses already carry token usage in their metadata, so token counts are exact. Cost is an estimate from a per-model price table next to the model defaults in `llm_clients/config.py`, labeled `est.`; a model with no price shows tokens only. Both also go into `events.jsonl`.
3. **Ctrl-C follows the usual two-step convention** (as `docker compose`, `cargo`, and most job runners do):
   - **First Ctrl-C** stops starting new jobs and cancels the running ones, keeps everything already finished, prints the `--into` command that resumes the run, and exits with 130 (the shell's code for an interrupt). It doesn't wait for running jobs to finish: a 30-turn conversation can take minutes, and `--into` reruns an unfinished conversation from the start anyway, so waiting buys nothing.
   - **A second Ctrl-C** exits immediately.
4. **ASCII only, for now.** No emoji or other Unicode symbols on stdout: `#`/`.` progress bars, `FAIL` instead of a cross, `|` as a separator, `...` instead of an ellipsis. Output reads the same in every terminal, CI log, and file.
5. **Resuming a pipeline needs `vera pipeline --into`.** Today only `generate` and `judge` take `--into`. The end summary and the Ctrl-C message print the command that resumes the run, so for `vera pipeline` that command has to exist first. It comes with the pipeline's single job list (#266): the run's `config.json` fixes every job, so resuming the pipeline is the same skip-what's-done as for `generate` and `judge`, followed by scoring and pooling. Until then, a pipeline's summary prints the `generate --into` and `judge --into` commands for each unfinished run folder.
