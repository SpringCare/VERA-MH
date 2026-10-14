# One job loop for generation and judging

**Status:** draft for discussion, not committed. Lands with the first PR that builds it.

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

## Proposal

One function holds all the concurrency code in the repo:

```python
# workers/loop.py
async def run_jobs(jobs: list[Job], max_concurrent: int | None) -> list[Outcome]:
    todo = [job for job in jobs if not job.done()]          # resume = skip what's on disk
    limit = asyncio.Semaphore(max_concurrent or len(todo) or 1)

    async def run_one(job: Job) -> Outcome:
        async with limit:
            try:
                await job.run()
                return Outcome.ok(job)
            except Exception as error:
                return Outcome.failed(job, error)

    finished = await asyncio.gather(*(run_one(job) for job in todo))
    return [Outcome.skipped(job) for job in jobs if job not in todo] + finished
```

A job is anything with a label, a way to tell whether its output already exists, and a way to produce it:

```python
class Job(Protocol):
    label: str
    def done(self) -> bool: ...
    async def run(self) -> None: ...
```

Each stage then reads top to bottom:

```python
# generate
jobs = [Conversation(persona, run, run_dir) for persona in personas for run in runs]
report(await run_jobs(jobs, max_concurrent))

# judge
jobs = [Evaluation(conversation, judge, instance, run_dir)
        for conversation in conversations
        for judge in judges
        for instance in judge.instances]
report(await run_jobs(jobs, max_concurrent))
write_results_csv(run_dir)
```

`Conversation.run()` is today's `run_single_conversation`; `Evaluation.run()` is today's `_evaluate_single_conversation_with_judge`. A semaphore plus `gather` replaces both queue-and-worker pools: no worker IDs, no `get_nowait` loop, no shared results list.

## What this removes

Once `legacy/` is gone (it needs a tagged release that announces the removal; roadmap, Deferred):

- `run_for_user_models` and `_legacy_model_config`; `run_generation`'s always-constant parameters (`persona_names`, `run_id`, `verbose`, the list-of-one `persona_files`)
- The `verbose_workers` branches, `judge_single_conversation`, and the legacy helpers in `judge/utils.py`
- `_run_workers_with_queue`, both `_worker`s, the dead `_create_conversation_jobs`, and the three judge summaries
- The CLI parity tests and most of `test_pipeline.py`

## Open questions

1. **Resume granularity.** Multi-session generation skips per session, inside the job. Should a job be one session instead of one persona × run, so `done()` stays a single file check?
2. **`per_judge` pools.** Keep one limit per judge model, or replace it with a per-provider concurrency cap in `llm_clients/` (roadmap, Shared engine)?
3. **Progress.** `report` prints one flushed line as each job starts and finishes. That also fixes the backlog item about runs that look hung under redirection. Enough, or do we want turns-completed while a conversation runs?
4. **Fail fast.** Should `run_jobs` (or the caller) raise when every job failed? This is the backlog item where a generation stage with zero conversations only fails one stage later, in judging.
5. **Retries.** Keep retries per LLM call only (`llm_clients`), or also retry a failed job once?

## Testing

The roadmap's four engine-testing tiers apply (structural parity with the mock harness, live smoke, semantic similarity, manual spot-check). `run_jobs` itself gets plain unit tests: skip-on-done, concurrency cap holds, failures recorded and not raised, `max_concurrent` of `None` and `0` both unlimited.
