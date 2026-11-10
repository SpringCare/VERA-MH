"""What happens during a run, as events a reporter shows or writes down.

The loop emits the job events itself: started, finished, failed, skipped, and
the stage finished at the end. Code running inside a job emits the rest through
`workers.job.report()` (a turn started, a judge question asked, an LLM call
retrying, a call's token usage).

Every event's fields are JSON-safe, so the same event can be printed as a line
or written to `events.jsonl`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol


class JobKind(StrEnum):
    """What a job does, which picks its concurrency cap. A `StrEnum`, so each
    one is also its plain string value ("session"), as `events.jsonl` records."""

    SESSION = "session"  # one session of a conversation (generation)
    EVALUATION = "evaluation"  # one conversation judged (judging)


class EventType(StrEnum):
    """Every kind of event. A `StrEnum`, so each one is also its plain string
    value ("job_started"), which is what `events.jsonl` records."""

    # Emitted by the loop, about one job.
    JOB_STARTED = "job_started"
    JOB_FINISHED = "job_finished"  # data: elapsed
    JOB_FAILED = "job_failed"  # data: error
    JOB_SKIPPED = "job_skipped"
    # Emitted by the loop once, when `JobLoop.run()` ends: about the whole run.
    STAGE_FINISHED = "stage_finished"  # data: stage, done, failed, skipped

    # Emitted from inside a job, with `workers.job.report()`.
    TURN_STARTED = "turn_started"  # data: turn, speaker ("user" or "chatbot")
    QUESTION_ASKED = "question_asked"  # data: question_id
    # LLM calls retry themselves (llm_clients/); the loop never retries a job.
    CALL_RETRYING = "call_retrying"  # data: attempt, max_attempts, error
    CALL_FINISHED = "call_finished"  # data: model, input_tokens, output_tokens


# The word each event shows as on the terminal (`PlainLineReporter`). Turns and
# questions show only with `verbose`, as a run has many of them; a call
# retrying always shows, since it's how a slow or failing provider shows up.
# Token usage (`CALL_FINISHED`) has no word: it only goes to `events.jsonl`.
WORDS = {
    EventType.JOB_STARTED: "start",
    EventType.JOB_FINISHED: "done",
    EventType.JOB_FAILED: "FAIL",
    EventType.JOB_SKIPPED: "skip",
    EventType.CALL_RETRYING: "retry",
    EventType.TURN_STARTED: "turn",
    EventType.QUESTION_ASKED: "ask",
}
VERBOSE_ONLY = {EventType.TURN_STARTED, EventType.QUESTION_ASKED}


def _now() -> datetime:
    # A named function, not a lambda, so test_run_root's clock rule can allow
    # this one `datetime.now()` call by name.
    return datetime.now().astimezone()


@dataclass(frozen=True)
class Event:
    """One thing that happened.

    `job` and `kind` name the job it's about. `STAGE_FINISHED` is the one event
    that isn't about a job but about the whole run (its counts are in `data`),
    so it has neither.
    """

    type: EventType
    job: str | None = None
    kind: JobKind | None = None
    data: dict[str, Any] = field(default_factory=dict)
    time: datetime = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        """The event as one flat JSON object, as written to `events.jsonl`."""
        return {
            **self.data,
            "time": self.time.isoformat(timespec="seconds"),
            "type": self.type.value,
            "kind": self.kind,
            "job": self.job,
        }


class Reporter(Protocol):
    """Shows or records events: see `workers.reporters`."""

    def emit(self, event: Event) -> None: ...
