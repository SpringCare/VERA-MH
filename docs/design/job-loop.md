# One job loop for generation and judging

**Status:** draft for discussion. Lands with the first PR that builds it.

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
generate/run.py          run_generation(config): load the queue with the first jobs, run the loop, report
generate/conversation.py the Session job: what generating one session means
        │
        ▼
workers/loop.py          JobLoop: one queue and one cap per job kind; skip done jobs, run the rest, collect outcomes
```

Judging is the same shape: `vera_cli/judge.py` → `judge/run.py` (`run_judging`) → `judge/evaluation.py` (the `Evaluation` job) → `workers/loop.py`. The CLI never calls the loop directly; it resolves everything up front and calls one domain entry point, which keeps the layer rule in [architecture.md](../architecture.md) (`vera_cli` → domain → `workers/`).

## Proposal

### The loop

All the concurrency code in the repo, in one class. It's a class rather than a function because `submit` is called from three places (the stage, a finished job's follow-ups, a retry) and they all need the same queues, so the queues live on an object instead of in a closure. Each job kind (`session`, `evaluation`) has its own queue and its own cap. Everything that adds work goes through `submit`, which never waits: the stage loading its first jobs, a finished job handing on its follow-ups, and a failed job going back for another try.

```python
# workers/loop.py
class JobLoop:
    def __init__(self, max_concurrent: dict[str, int | None]):
        self.caps = max_concurrent                            # per kind; None or 0 means no limit
        self.queues = {kind: deque() for kind in max_concurrent}
        self.running: dict[asyncio.Task, Job] = {}
        self.outcomes: list[Outcome] = []

    def submit(self, job: Job) -> None:
        """Put a job on its kind's queue. Never waits."""
        if job.done():                                        # resume: its output is already on disk,
            self.outcomes.append(Outcome.skipped(job))
            for next_job in job.follow_ups():                 # but the jobs after it may not be
                self.submit(next_job)
        else:
            self.queues[job.kind].append(job)

    async def run(self) -> list[Outcome]:
        """Run until every queue is empty and nothing is running."""
        while True:
            self.start_jobs_that_fit()
            if not self.running:
                return self.outcomes
            finished, _ = await asyncio.wait(self.running, return_when=asyncio.FIRST_COMPLETED)
            for task in finished:
                self.record(self.running.pop(task), task.exception())

    def start_jobs_that_fit(self) -> None:
        for kind, queue in self.queues.items():
            while queue and self.has_room(kind):
                job = queue.popleft()
                self.running[asyncio.create_task(job.run())] = job

    def has_room(self, kind: str) -> bool:
        cap = self.caps[kind]
        running = sum(1 for job in self.running.values() if job.kind == kind)
        return not cap or running < cap

    def record(self, job: Job, error: BaseException | None) -> None:
        if error is None:
            self.outcomes.append(Outcome.ok(job))
            for next_job in job.follow_ups():
                self.submit(next_job)
        elif job.attempt < job.max_attempts and job.retryable(error):
            self.submit(dataclasses.replace(job, attempt=job.attempt + 1))   # back of its queue
        else:
            self.outcomes.append(Outcome.failed(job, error))
```

A done job is skipped in `submit` but still hands on its follow-ups, and each of those is checked in turn, so resume reaches every missing job however far down the chain it is, and redoes nothing. Because the cap is checked when a job starts, not sized from the first jobs, follow-ups are never throttled by how few jobs the stage loaded.

A job is plain data plus four methods:

```python
class Job(Protocol):
    kind: str                                         # "session" or "evaluation": picks the queue and the cap
                                                      # (and the code a remote worker would run)
    label: str
    attempt: int                                      # 1 on the first try; the loop adds 1 on each retry
    max_attempts: int                                 # set by the stage from the config
    def done(self) -> bool: ...                       # is my output already on disk?
    async def run(self) -> None: ...                  # do the work
    def follow_ups(self) -> list["Job"]: ...          # the jobs waiting on me, from my fields and disk alone,
                                                      # so a skipped job hands them on too
    def retryable(self, error: Exception) -> bool: ...
```

### Each stage

A stage only loads the queue and runs the loop:

```python
# generate/run.py
loop = JobLoop(max_concurrent)
for persona in personas:
    for repeat in repeats:
        loop.submit(Session(persona, repeat, session_types=config.sessions, position=0,
                            run_dir=run_dir, max_attempts=config.attempts))
report(await loop.run())

# judge/run.py
loop = JobLoop(max_concurrent)
for conversation in conversations:
    for judge in judges:
        for instance in judge.instances:
            loop.submit(Evaluation(conversation, judge, instance, run_dir, max_attempts=config.attempts))
report(await loop.run())
write_results_csv(run_dir)
```

Sessions are named, not numbered: `session_types` is the configured list (`--sessions intake,coaching`; one unnamed session when unset), and `position` says which entry this job runs. Generation loads only each conversation's first session; `coaching` arrives as `intake`'s follow-up. That's a different thing from `attempt`, which counts tries at the same session. `Session.run()` is today's per-session body of `run_single_conversation`; `Evaluation.run()` is today's `_evaluate_single_conversation_with_judge`. One `JobLoop` replaces both queue-and-worker pools: no worker IDs, no `get_nowait` loop, no shared results list.

### The pipeline: one job list

`vera pipeline` can make several runs at once: one per user model (`-u gpt-5.4 -u claude-sonnet-5` makes two runs, whose scores are then pooled), and one per target when there are several. Today it runs a generation pool for one run, then a judging pool for it, then the next run, then pools the results: one pool after another, with idle capacity in between. With the loop it becomes one loop for every run: the first session of every conversation of every run goes on the session queue up front, and evaluations join the evaluation queue as conversations finish.

```python
# vera_cli/pipeline.py, after creating every run root up front (run-lifecycle.md, step 5)
loop = JobLoop(max_concurrent)
for user_model in user_models:
    for persona in personas:
        for repeat in repeats:
            loop.submit(Session(persona, repeat, session_types=config.sessions, position=0,
                                run_dir=run_dir_for(user_model), judges=judges,
                                max_attempts=config.attempts))
report(await loop.run())
for run_dir in run_dirs:
    write_results_csv(evaluation_dir_for(run_dir))
    score(evaluation_dir_for(run_dir))
pool(run_dirs)
```

A conversation's last session hands on that conversation's `Evaluation` jobs as follow-ups (the `judges` field says which; `vera generate` leaves it empty). They go on the evaluation queue, so the conversation is judged as soon as it exists, while other conversations are still being generated. Resume works through the same chain: if a run stopped after a conversation's last transcript but before its judging finished, every session of it is skipped as done, and its unfinished evaluations are reached and run. Every user model shares the same caps (decision 6), and the slowest persona no longer holds up the next model's start. Run-level steps (`results.csv`, scores, pooling) still wait for every job, because they need all the rows.

What this needs: the judge's run folder and rubric must exist before generation starts, which the run-lifecycle order already gives (all run roots are created before any job runs).

It also gives the pipeline the resume that `generate` and `judge` have. `vera pipeline` takes no `--into` today. With one job list fixed by the run's `config.json`, `vera pipeline --into <run>` reopens the run roots, skips every job that's done (decision 5), and then runs scoring and pooling as usual. The end summary and the Ctrl-C message of the terminal interface ([#271](https://github.com/SpringCare/VERA-MH/pull/271)) print this command.

## Decisions from review

1. **A job is one session.** Sessions must run in order, so session *k* is a follow-up of session *k − 1*. Generation lists every conversation's first session; a done session is skipped but still hands on its successor, so resume restarts exactly at the first missing session and nothing after it is redone. `done()` stays a single file check.
   - **Provider state crosses sessions as data.** Some chatbots keep the history server-side, under the client's `conversation_id` (`EndpointLLM`). Today that survives between sessions only because the runner reuses one client for all of a conversation's sessions, and it's lost on resume. Instead, a session saves the chatbot's `conversation_id` with its transcript, and the next session reads it from there and creates its client with it. Continuity then holds across job boundaries, retries (a retried session starts again from the previous session's state), and resume.
2. **A failed job goes back on the queue.** LLM calls already retry per call in `llm_clients/` (3 attempts with backoff). The loop adds job-level attempts for what that doesn't cover, such as a judge answer that fails the structured-output schema. Each job carries its attempt count and its `max_attempts`; when it fails with an error its type calls worth retrying (`retryable`), the loop puts a copy with `attempt + 1` at the back of its queue, so other jobs run in the meantime and a retry doesn't hold a slot while it waits. That's how hosted queues do it too (a message carries its delivery count), so it carries over to an external queue unchanged. The stage sets `max_attempts` from the config.
3. **Concurrency per judge model, later.** Today's `per_judge` keeps one cap per judge model. That stays the intended behavior but isn't pressing; the first version has one cap per job kind (decision 6), and a per-model (or per-provider) cap comes with the Shared engine phase.
4. **Progress display is a separate PR** ([terminal-interface.md](./terminal-interface.md), #271). The loop emits events (job started, job finished, and from inside a job: user responding, chatbot responding, judge on question *n*) to a reporter. The first reporter prints one flushed line per event, which fixes the "looks hung under redirection" backlog item. A later PR adds a live display: a progress bar per stage and a box per model saying what it's doing.
5. **Missing work is re-done, not fatal.** The run's `config.json` (see [run-lifecycle.md](./run-lifecycle.md)) fixes the job list, so `done()` over that list is the manifest: rerunning with `--into` does exactly the missing jobs. A stage still stops the pipeline when it produced nothing (nothing to judge), with a message naming the run folder and the failure count; partial failures warn and continue.
   - **Unfinished judging works the same way.** An `Evaluation` is done when its TSV exists, and the TSV is written only after the judge has answered every question it needed. A judging stopped partway through a conversation leaves no TSV, so rerunning with `--into` judges that conversation again from its first question; conversations already judged are skipped. `results.csv` and the scores are rebuilt from the TSVs after the loop, so they always reflect exactly the finished evaluations. Saving progress question by question isn't worth it: one conversation is a few dozen short judge calls.
6. **One cap per job kind.** The config already sets two: `generation.max_concurrent` caps `Session` jobs and `judging.max_concurrent` caps `Evaluation` jobs, so in the pipeline a burst of short judge calls can't crowd out the long generation calls. The loop keeps one queue and one cap per job kind instead of one in total. The per-judge-model cap (`per_judge`) still comes later (decision 3).
7. **Jobs stay serializable, so an external queue stays possible.** Jobs are plain data (paths, model names, indices), not closures, so they can be serialized as JSON. A closure would carry live state with it (an LLM client's HTTP session, its API key), which can't or shouldn't travel; instead each job names its `kind`, and the worker looks up the code to run and builds its own clients. That keeps an external queue (a cloud task queue, a cluster scheduler) possible: the in-memory queues are the seam. An external version keeps `submit` and `run`, but `submit` sends the job's JSON to the external queue, workers elsewhere pull from it, and the cap becomes how many workers pull. Stages don't change, since they only call `submit` and `run`. The interface isn't built until there's a concrete need, same rule as the roadmap's Storage phase. What keeps it possible is enforced, not assumed: each job type is a dataclass of JSON-safe fields plus `kind`, and a unit test round-trips every job type through JSON and runs the copy.

## What this removes

`legacy/` is already gone on `release/2.1` (#287), with its parity tests, `test_pipeline.py`, and the leftovers that had no callers (`judge_single_conversation`, `_create_conversation_jobs`; #303). The forwarding layers go in the two PRs before the loop (roadmap, Release 2.1 plan, step 4): `run_for_user_models`, `_legacy_model_config`, and the always-constant parameters with generation taking `GenerationConfig` (#309), and the `verbose_workers` branches with judging taking `JudgingConfig`. What the loop itself removes:

- `_run_workers_with_queue` and both `_worker`s
- The three judge summaries, replaced by one `report(outcomes)`
- Failures worked out by subtraction afterwards: every job ends as exactly one outcome

## Testing

The roadmap's four engine-testing tiers apply (structural parity with the mock harness, live smoke, semantic similarity, manual spot-check). `JobLoop` itself gets plain unit tests: skip-on-done, each kind's concurrency cap holds, failures recorded and not raised, a retryable failure goes back on the queue with `attempt + 1` and stops at `max_attempts`, non-retryable errors aren't retried, follow-ups run after their parent and not when it fails, a skipped job still hands on its follow-ups and each is checked for done in turn, follow-ups aren't throttled when their kind has no cap, `max_concurrent` of `None` and `0` both unlimited, and every job type survives a JSON round trip (decision 7).
