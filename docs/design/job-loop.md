# One job loop for generation and judging

**Status:** accepted; the loop is built in `workers/` ([#308](https://github.com/SpringCare/VERA-MH/pull/308)). Judging and generation move onto it next (roadmap, "Steps 4 and 5, PR by PR"). The sketches below show the shape; the code is the reference.

## Problem

Generation and judging each hand-roll the same asyncio queue-and-workers pool, and the two copies have drifted:

| | Generate (`generate/runner.py`) | Judge (`judge/runner.py`) |
|---|---|---|
| Job | 4-tuple `(dict, max_turns, idx, run)` | 7-tuple, with output folder, rubric, and params repeated in every job |
| Resume | Parses transcript filenames back into persona/run/session, checked twice (per job and per session) | Checks the TSV basename in the output folder |
| `max_concurrent=0` | No limit | Was zero workers (fixed in #261) |
| Failures | Three layers of `except`; recorded as skipped results | One `except` in the worker; failure not recorded, worked out afterwards by subtraction |
| Summary | One place | Printed in three places |

On top of that, each stage has 4–5 layers that mostly forward arguments: `vera_cli/<cmd>._execute` → `run_*` → runner `__init__` / `judge_conversations` → `batch_evaluate…` → `_run_workers_with_queue` → `_worker`. Many of those layers exist for `legacy/`.

## Who does what

```
vera_cli/generate.py     resolve and validate settings → RunConfig        (no jobs, no I/O beyond reading inputs)
        │
        ▼
generate/run.py          run_generation(config): submit the first jobs, run the loop
generate/conversation.py the Session job: what generating one session means
        │
        ▼
workers/loop.py          JobLoop: one cap per job kind; skip done jobs, run the rest, collect outcomes
```

Judging is the same shape: `vera_cli/judge.py` → `judge/run.py` (`run_judging`) → `judge/evaluation.py` (the `Evaluation` job) → `workers/loop.py`. The CLI never calls the loop directly; it resolves everything up front and calls one domain entry point, which keeps the layer rule in [architecture.md](../architecture.md) (`vera_cli` → domain → `workers/`).

## Proposal

### The loop

All the concurrency code in the repo, in one class. It's a class rather than a function because the jobs the stage submits and the follow-ups that finished jobs hand on share the same caps, the same running tasks, and the same outcomes; it also gives an external queue one thing to replace, anything with `submit` and `run` (decision 7). Each job kind (`JobKind.SESSION`, `JobKind.EVALUATION`) has its own cap, an `asyncio.Semaphore`. `submit` only adds a job to a list; `run` starts every job in its own task inside an `asyncio.TaskGroup`, and each task waits for a free slot of its kind. The tasks waiting on a kind's semaphore are that kind's queue. The loop never retries a job: retries happen per call, inside it (decision 2).

```python
# workers/loop.py (simplified: events and timing left out)
class JobLoop:
    def __init__(self, max_concurrent: Mapping[JobKind, int | None], *, reporters=(), stage="jobs"):
        self.limits = {kind: asyncio.Semaphore(cap) if cap else contextlib.nullcontext()
                       for kind, cap in max_concurrent.items()}  # None or 0: no limit
        self.jobs: list[Job] = []
        self.outcomes: list[Outcome] = []

    def submit(self, job: Job) -> None:
        """Add a job for `run()` to run."""
        self.jobs.append(job)

    async def run(self) -> list[Outcome]:
        """Run every submitted job and the follow-ups they hand on; return every outcome."""
        async with asyncio.TaskGroup() as self.tasks:        # also waits for tasks added while it waits
            for job in self.jobs:
                self.tasks.create_task(self._run_one(job))
        self.emit(Event(EventType.STAGE_FINISHED, data=counts_by_status))
        return self.outcomes

    async def _run_one(self, job: Job) -> None:
        try:                                                  # any error is this job's outcome
            if job.done():                                    # resume: its output is already on disk,
                status = Status.SKIPPED
            else:
                async with self.limits[job.kind]:
                    set_current_job(job, self.emit)           # where report() sends this job's events
                    await job.run()
                status = Status.FINISHED
            follow_ups = job.follow_ups()                     # but the jobs after it may not be
        except ReporterError:
            raise                                             # a reporter bug stops the run (below)
        except Exception as error:
            self._record(Outcome(job, Status.FAILED, error=error))   # rerun later with --into
            return
        self._record(Outcome(job, status))
        for next_job in follow_ups:
            self.tasks.create_task(self._run_one(next_job))
```

The loop emits an event when each job starts, finishes, fails, or is skipped, and one `STAGE_FINISHED` event with the counts when `run` ends, named by `stage` (decision 4). A failed job's outcome also records how long it ran. A job's own errors are its outcome, so an error that escapes a task is a bug in the loop or a reporter: a reporter that raises becomes a `ReporterError`, the `TaskGroup` cancels the other jobs, and `run` raises it, rather than recording it as the failure of whichever job happened to be reporting.

A done job is skipped but still hands on its follow-ups, and each of those is checked in turn, so resume reaches every missing job however far down the chain it is, and redoes nothing. Because a job takes a slot when it starts, follow-ups are never throttled by how few jobs the stage loaded.

A job is plain data plus three methods (`workers/job.py`):

```python
class Job(Protocol):
    kind: JobKind                                     # SESSION or EVALUATION: picks the cap
                                                      # (and the code a remote worker would run)
    label: str                                        # how events name it, such as "gpt-5.2 Brian"
    def done(self) -> bool: ...                       # is my output already on disk?
    async def run(self) -> None: ...                  # do the work
    def follow_ups(self) -> list["Job"]: ...          # the jobs waiting on me, from my fields and disk alone,
                                                      # so a skipped job hands them on too
```

### Each stage

A stage only submits its first jobs and runs the loop:

```python
# generate/run.py
loop = JobLoop({JobKind.SESSION: max_concurrent}, reporters=reporters, stage="generate")
for persona in personas:
    for repeat in repeats:
        loop.submit(Session(persona, repeat, session_types=config.sessions, position=0,
                            run_dir=run_dir))
outcomes = await loop.run()          # the reporters have already shown each job and the totals

# judge/run.py
loop = JobLoop({JobKind.EVALUATION: max_concurrent}, reporters=reporters, stage="judge")
for conversation in conversations:
    for judge in judges:
        for instance in judge.instances:
            loop.submit(Evaluation(conversation, judge, instance, run_dir))
outcomes = await loop.run()
write_results_csv(run_dir)
```

Sessions are named, not numbered: `session_types` is the configured list (`--sessions intake,coaching`; one unnamed session when unset), and `position` says which entry this job runs. Generation loads only each conversation's first session; `coaching` arrives as `intake`'s follow-up. `Session.run()` is today's per-session body of `run_single_conversation`; `Evaluation.run()` is today's `_evaluate_single_conversation_with_judge`. One `JobLoop` replaces both queue-and-worker pools: no worker IDs, no `get_nowait` loop, no shared results list.

### The pipeline: one job list

`vera pipeline` can make several runs at once: one per user model (`-u gpt-5.4 -u claude-sonnet-5` makes two runs, whose scores are then pooled), and one per target when there are several. Today it runs a generation pool for one run, then a judging pool for it, then the next run, then pools the results: one pool after another, with idle capacity in between. With the loop it becomes one loop for every run: the first session of every conversation of every run is submitted up front, and evaluations are submitted as conversations finish, each kind under its own cap.

```python
# vera_cli/pipeline.py, after creating every run root up front (run-lifecycle.md, step 5)
loop = JobLoop({JobKind.SESSION: generation_cap, JobKind.EVALUATION: judging_cap},
               reporters=reporters, stage="pipeline")
for user_model in user_models:
    for persona in personas:
        for repeat in repeats:
            loop.submit(Session(persona, repeat, session_types=config.sessions, position=0,
                                run_dir=run_dir_for(user_model), judges=judges))
outcomes = await loop.run()
for run_dir in run_dirs:
    write_results_csv(evaluation_dir_for(run_dir))
    score(evaluation_dir_for(run_dir))
pool(run_dirs)
```

A conversation's last session hands on that conversation's `Evaluation` jobs as follow-ups (the `judges` field says which; `vera generate` leaves it empty). They wait for an evaluation slot, so the conversation is judged as soon as it exists, while other conversations are still being generated. Resume works through the same chain: if a run stopped after a conversation's last transcript but before its judging finished, every session of it is skipped as done, and its unfinished evaluations are reached and run. Every user model shares the same caps (decision 6), and the slowest persona no longer holds up the next model's start. Run-level steps (`results.csv`, scores, pooling) still wait for every job, because they need all the rows.

What this needs: the judge's run folder and rubric must exist before generation starts, which the run-lifecycle order already gives (all run roots are created before any job runs).

It also gives the pipeline the resume that `generate` and `judge` have. `vera pipeline` takes no `--into` today. With one job list fixed by the run's `config.json`, `vera pipeline --into <run>` reopens the run roots, skips every job that's done (decision 5), and then runs scoring and pooling as usual. The end summary and the Ctrl-C message of the terminal interface ([#271](https://github.com/SpringCare/VERA-MH/pull/271)) print this command.

## Decisions from review

1. **A job is one session.** Sessions must run in order, so session *k* is a follow-up of session *k − 1*. Generation lists every conversation's first session; a done session is skipped but still hands on its successor, so resume restarts exactly at the first missing session and nothing after it is redone. `done()` stays a single file check.
   - **Provider state crosses sessions as data.** Some chatbots keep the history server-side, under the client's `conversation_id` (`EndpointLLM`). Today that survives between sessions only because the runner reuses one client for all of a conversation's sessions, and it's lost on resume. Instead, a session saves the chatbot's `conversation_id` with its transcript, and the next session reads it from there and creates its client with it. Continuity then holds across job boundaries and resume (a rerun session starts again from the previous session's state).
2. **Retries happen in one place: per call, in `llm_clients/`.** Every LLM call already retries with backoff (`_run_with_retry`, up to `max_llm_retries`), and that covers the whole call, parsing included, so a structured answer that breaks the schema is re-asked too. The loop adds no second mechanism. A failed job is recorded as failed and leaves the queue for good. Its output was never written, so rerunning with `--into` picks it up like any missing job.
   - **Why not retry the job as well.** A job fails only after one of its calls has used up its retries, so the cause is persistent (quota, auth, a model that keeps failing) and an immediate rerun rarely helps. Retrying per call also retries the smallest unit: a rate limit on turn 15 re-sends that one call, where rerunning the job would redo the whole session from turn 1.
   - **The cost.** While a call backs off, its job keeps its slot, so one fewer job of that kind runs meanwhile. The waits are short and capped (`retry_max_delay_seconds`).
   - **The clients still tell the loop.** Each per-call retry emits a `call_retrying` event through the same `report()` jobs use (decision 4), so the display can show "rate limited, retrying 2/4" against the right conversation. That's the only link: no retry logic is shared.
3. **Concurrency per judge model, later.** Today's `per_judge` keeps one cap per judge model. That stays the intended behavior but isn't pressing; the loop has one cap per job kind (decision 6), and a per-model (or per-provider) cap comes with the Shared engine phase. Until then `vera judge --per-judge` keeps working by running one loop per judge model, side by side, each with its own cap. The pipeline's single job list (one loop for every run) needs the real per-model cap before it can honor `per_judge`.
4. **Events and the plain-line reporter come with the loop; the live display is a later PR** ([terminal-interface.md](./terminal-interface.md), #271). The loop emits events to its reporters (`workers/events.py`): job started, finished, failed, or skipped, and the stage finished with its counts. Code inside a job emits the rest through `report()` (`workers/job.py`): a turn started, a judge question asked, a call retrying, a call's token usage. `report()` finds the running job through a context variable the loop sets in each job's task, so nothing is threaded through the simulator, the judge, or the clients; `workers/job.py` has a worked example. Two reporters shipped with the loop (#308): `PlainLineReporter` prints one flushed, timestamped, ASCII line per event, which fixes the "looks hung under redirection" backlog item, and `EventsFileReporter` writes `events.jsonl`. A later PR adds the live display: a progress bar per stage and a box per model saying what it's doing.
5. **Missing work is re-done, not fatal.** The run's `config.json` (see [run-lifecycle.md](./run-lifecycle.md)) fixes the job list, so `done()` over that list is the manifest: rerunning with `--into` does exactly the missing jobs. A stage still stops the pipeline when it produced nothing (nothing to judge), with a message naming the run folder and the failure count; partial failures warn and continue.
   - **Unfinished judging works the same way.** An `Evaluation` is done when its TSV exists, and the TSV is written only after the judge has answered every question it needed. A judging stopped partway through a conversation leaves no TSV, so rerunning with `--into` judges that conversation again from its first question; conversations already judged are skipped. `results.csv` and the scores are rebuilt from the TSVs after the loop, so they always reflect exactly the finished evaluations. Saving progress question by question isn't worth it: one conversation is a few dozen short judge calls.
6. **One cap per job kind.** The config already sets two: `generation.max_concurrent` caps `Session` jobs and `judging.max_concurrent` caps `Evaluation` jobs, so in the pipeline a burst of short judge calls can't crowd out the long generation calls. The loop keeps one cap per job kind instead of one in total. The per-judge-model cap (`per_judge`) still comes later (decision 3).
7. **Jobs stay serializable, so an external queue stays possible.** Jobs are plain data (paths, model names, indices), not closures, so they can be serialized as JSON. A closure would carry live state with it (an LLM client's HTTP session, its API key), which can't or shouldn't travel; instead each job names its `kind`, and the worker looks up the code to run and builds its own clients. That keeps an external queue (a cloud task queue, a cluster scheduler) possible: `submit` is the seam. An external version keeps `submit` and `run`, but `submit` sends the job's JSON to the external queue, workers elsewhere pull from it, and the cap becomes how many workers pull. Stages don't change, since they only call `submit` and `run`. The interface isn't built until there's a concrete need, same rule as the roadmap's Storage phase. What keeps it possible is enforced, not assumed: each job type is a dataclass of JSON-safe fields plus `kind`, and a unit test round-trips every job type through JSON and runs the copy.

## What this removes

`legacy/` is already gone on `release/2.1` (#287), with its parity tests, `test_pipeline.py`, and the leftovers that had no callers (`judge_single_conversation`, `_create_conversation_jobs`; #303). The forwarding layers go in the two PRs before the loop (roadmap, Release 2.1 plan, step 4): `run_for_user_models`, `_legacy_model_config`, and the always-constant parameters with generation taking `GenerationConfig` (#309), and the `verbose_workers` branches with judging taking `JudgingConfig`. What the loop itself removes:

- `_run_workers_with_queue` and both `_worker`s
- The three judge summaries, replaced by the loop's one `STAGE_FINISHED` event
- Failures worked out by subtraction afterwards: every job ends as exactly one outcome

## Testing

The roadmap's four engine-testing tiers apply (structural parity with the mock harness, live smoke, semantic similarity, manual spot-check). `JobLoop` itself gets plain unit tests (`tests/unit/workers/`): skip-on-done, each kind's concurrency cap holds, failures recorded and not raised and never resubmitted, a failed job's outcome records how long it ran, follow-ups run after their parent and not when it fails, a skipped job still hands on its follow-ups and each is checked for done in turn, follow-ups aren't throttled when their kind has no cap, `max_concurrent` of `None` and `0` both unlimited, a reporter error stops the run instead of vanishing, events from inside a job reach every reporter, and every job type survives a JSON round trip (decision 7).
