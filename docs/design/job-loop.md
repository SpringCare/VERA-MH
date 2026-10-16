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
generate/run.py          run_generation(config): list the jobs, hand them to the loop, report
generate/conversation.py the Session job: what generating one session means
        │
        ▼
workers/loop.py          run_jobs(...): skip done jobs, run the rest with a concurrency cap, collect outcomes
```

Judging is the same shape: `vera_cli/judge.py` → `judge/run.py` (`run_judging`) → `judge/evaluation.py` (the `Evaluation` job) → `workers/loop.py`. The CLI never calls the loop directly; it resolves everything up front and calls one domain entry point, which keeps the layer rule in [architecture.md](../architecture.md) (`vera_cli` → domain → `workers/`).

## Proposal

### The loop

All the concurrency code in the repo, in one function:

```python
# workers/loop.py
async def run_jobs(jobs: list[Job], max_concurrent: int | None, attempts: int = 1) -> list[Outcome]:
    todo = [job for job in jobs if not job.done()]          # resume = skip what's on disk
    limit = asyncio.Semaphore(max_concurrent or len(todo) or 1)

    async def run_one(job: Job) -> list[Outcome]:
        async with limit:
            outcome = await run_with_retries(job, attempts)
        # A finished job may hand back the work that depended on it.
        follow_ups = outcome.follow_ups if outcome.ok else []
        return [outcome, *await gather_flat(run_one(next_job) for next_job in follow_ups)]

    finished = await gather_flat(run_one(job) for job in todo)
    return [Outcome.skipped(job) for job in jobs if job not in todo] + finished


async def run_with_retries(job: Job, attempts: int) -> Outcome:
    for attempt in range(1, attempts + 1):
        try:
            return Outcome.ok(job, follow_ups=await job.run())
        except Exception as error:
            if attempt == attempts or not job.retryable(error):
                return Outcome.failed(job, error)
```

A job is plain data plus two methods:

```python
class Job(Protocol):
    label: str
    def done(self) -> bool: ...                       # is my output already on disk?
    async def run(self) -> list["Job"]: ...           # do the work; return jobs that were waiting on me
    def retryable(self, error: Exception) -> bool: ...
```

### Each stage

```python
# generate/run.py
jobs = [first_unfinished_session(persona, repeat, run_dir)
        for persona in personas for repeat in repeats]
report(await run_jobs(jobs, max_concurrent, attempts=config.attempts))

# judge/run.py
jobs = [Evaluation(conversation, judge, instance, run_dir)
        for conversation in conversations
        for judge in judges
        for instance in judge.instances]
report(await run_jobs(jobs, max_concurrent, attempts=config.attempts))
write_results_csv(run_dir)
```

`Session.run()` is today's per-session body of `run_single_conversation`; `Evaluation.run()` is today's `_evaluate_single_conversation_with_judge`. A semaphore plus `gather` replaces both queue-and-worker pools: no worker IDs, no `get_nowait` loop, no shared results list.

## Decisions from review

1. **A job is one session.** Sessions must run in order, so session *k* is a follow-up of session *k − 1*: its `run()` returns the next session's job. Generation lists only the first unfinished session of each conversation. `done()` stays a single file check, and resume restarts exactly at the first missing session.
2. **Retries live in the loop, the policy in the stage.** LLM calls already retry per call in `llm_clients/` (3 attempts with backoff). The loop adds job-level attempts for what that doesn't cover, such as a judge answer that fails the structured-output schema. Each job type says which errors are worth retrying (`retryable`); the stage picks the number of attempts.
3. **Concurrency per judge model, later.** Today's `per_judge` keeps one cap per judge model. That stays the intended behavior but isn't pressing; the first version has one global cap, and a per-model (or per-provider) cap comes with the Shared engine phase.
4. **Progress display is a separate PR.** The loop emits events (job started, job finished, and from inside a job: user responding, chatbot responding, judge on question *n*) to a reporter. The first reporter prints one flushed line per event, which fixes the "looks hung under redirection" backlog item. A later PR adds a live display: a progress bar per stage and a box per model saying what it's doing.
5. **Missing work is re-done, not fatal.** The run's `config.json` (see [run-lifecycle.md](./run-lifecycle.md)) fixes the job list, so `done()` over that list is the manifest: rerunning with `--into` does exactly the missing jobs. A stage still stops the pipeline when it produced nothing (nothing to judge), with a message naming the run folder and the failure count; partial failures warn and continue.

## Open questions

1. **Stream generation into judging?** Today a pipeline judges only after every conversation is generated. With follow-ups, a finished session could return its `Evaluation` jobs, so judging starts while generation continues. Gains: wall-clock time, and a broken judge setup fails in minutes instead of after generation. Costs: generation and judging compete for the concurrency cap, and the judge needs its run folder and rubric ready before generation starts. Proposal: keep the stage barrier in the first version; the follow-up mechanism makes streaming a later, local change.
2. **External queue.** Jobs are plain data (paths, model names, indices), not closures, so they can be serialized as JSON. A closure would carry live state with it (an LLM client's HTTP session, its API key), which can't or shouldn't travel; instead each job names its `kind`, and the worker looks up the code to run and builds its own clients. That keeps an external queue (a cloud task queue, a cluster scheduler) possible: `run_jobs` becomes one implementation of an executor interface, and a remote one would submit serialized jobs instead. Proposal: don't build the interface until there's a concrete need, same rule as the roadmap's Storage phase; just keep jobs serializable.

## What this removes

Once `legacy/` is gone (it needs a tagged release that announces the removal; roadmap, Deferred):

- `run_for_user_models` and `_legacy_model_config`; `run_generation`'s always-constant parameters (`persona_names`, `run_id`, `verbose`, the list-of-one `persona_files`)
- The `verbose_workers` branches, `judge_single_conversation`, and the legacy helpers in `judge/utils.py`
- `_run_workers_with_queue`, both `_worker`s, the dead `_create_conversation_jobs`, and the three judge summaries
- The CLI parity tests and most of `test_pipeline.py`

## Testing

The roadmap's four engine-testing tiers apply (structural parity with the mock harness, live smoke, semantic similarity, manual spot-check). `run_jobs` itself gets plain unit tests: skip-on-done, concurrency cap holds, failures recorded and not raised, retries stop at `attempts` and skip non-retryable errors, follow-ups run after their parent and not when it fails, `max_concurrent` of `None` and `0` both unlimited.
