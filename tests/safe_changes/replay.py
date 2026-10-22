"""Replay: record a small real pipeline run once, then rerun it without any model.

Recording runs `vera pipeline` against real models and writes every call (the
model, its system prompt, exactly what was sent, and the reply) to
tests/fixtures/replay/calls.jsonl, plus the run's outputs to `expected/`.
tests/unit/test_replay_pipeline.py reruns the same pipeline answering every
call from that file, and requires the same outputs.

A call is matched on everything that was sent, so the replay fails with
`NotInRecording` the moment the code sends a model anything different.

Record again only when a change is meant to alter what the models are sent,
then review the diff. It makes real API calls (a few cents with these models):

    uv run python -m tests.safe_changes.replay
"""

import csv
import io
import json
import os
import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional, Type, TypeVar
from unittest.mock import patch

import vera
from llm_clients.llm_factory import LLMFactory
from llm_clients.llm_interface import JudgeLLM, Role
from vera_cli import config as cli_config

REPO_ROOT = Path(__file__).resolve().parents[2]
RECORDING = REPO_ROOT / "tests/fixtures/replay/calls.jsonl"
EXPECTED = REPO_ROOT / "tests/fixtures/replay/expected"

# Small and cheap on purpose: enough to exercise every kind of call (user,
# chatbot, judge) and every stage, not to say anything about the models.
CHATBOT = "gpt-5.4-nano"
USER = "gpt-5.4-nano"
JUDGE = "gpt-5.4-nano"
PERSONAS = 2
TURNS = 4

T = TypeVar("T")


# ---------------------------------------------------------------- what a call is


def what_the_model_sees(history: Optional[list[dict[str, Any]]]):
    """The conversation history without each turn's `logging` entry.

    That entry holds timings, token counts, and request IDs, which differ on
    every run and are never sent to the provider.
    """
    if history is None:
        return None
    return [{k: v for k, v in turn.items() if k != "logging"} for turn in history]


def describe_call(model: str, system_prompt: str, kind: str, sent: Any) -> dict:
    """Everything that identifies a call. Recording and replay both use this."""
    return {"model": model, "system_prompt": system_prompt, "kind": kind, "sent": sent}


def call_key(call: dict) -> str:
    return json.dumps(call, sort_keys=True, ensure_ascii=False)


# ------------------------------------------------------------------- recording


class Recorder:
    """Wraps the real LLM clients and appends every call to a file."""

    def __init__(self, path: Path):
        self.path = path
        self.real_create_llm = LLMFactory.create_llm

    def save(self, call: dict, reply: Any) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({**call, "reply": reply}, ensure_ascii=False) + "\n")

    def create_llm(self, **settings: Any):
        """Stands in for `LLMFactory.create_llm`: a real client, recorded."""
        llm = self.real_create_llm(**settings)
        model = settings["model_name"]
        real_generate = llm.generate_response

        async def generate_response(conversation_history=None):
            reply = await real_generate(conversation_history)
            sent = what_the_model_sees(conversation_history)
            self.save(describe_call(model, llm.system_prompt, "text", sent), reply)
            return reply

        llm.generate_response = generate_response

        if isinstance(llm, JudgeLLM):
            real_structured = llm.generate_structured_response

            async def generate_structured_response(message, response_model):
                reply = await real_structured(message, response_model)
                call = describe_call(model, llm.system_prompt, "structured", message)
                self.save(call, reply.model_dump())
                return reply

            llm.generate_structured_response = generate_structured_response
        return llm


# ---------------------------------------------------------------------- replay


class NotInRecording(AssertionError):
    """The code sent a model something it never sent while recording."""


class Replayer:
    """Answers every call from a recording, in the order the calls were made."""

    def __init__(self, path: Path):
        self.replies: dict[str, list[Any]] = defaultdict(list)
        for line in path.read_text(encoding="utf-8").splitlines():
            recorded = json.loads(line)
            reply = recorded.pop("reply")
            self.replies[call_key(recorded)].append(reply)
        # Generation skips a conversation whose call fails, so a miss would
        # otherwise surface later as "no transcripts". Kept to report the cause.
        self.misses: list[str] = []

    def create_llm(self, **settings: Any) -> "ReplayLLM":
        """Stands in for `LLMFactory.create_llm`."""
        return ReplayLLM(self, **settings)

    def answer(self, call: dict) -> Any:
        replies = self.replies.get(call_key(call))
        if not replies:
            sent = call["sent"]
            last = sent[-1] if isinstance(sent, list) and sent else sent
            message = (
                f"{call['model']} was sent something the recording doesn't have.\n"
                f"Last input: {str(last)[:500]}"
            )
            self.misses.append(message)
            raise NotInRecording(message)
        return replies.pop(0)

    def unused_replies(self, kind: Optional[str] = None) -> int:
        """Recorded calls the code never made, optionally only of one kind.

        A leftover reply means the code made fewer calls than when recorded.
        `kind="text"` counts only user and chatbot turns, for a run that
        generates without judging.
        """
        return sum(
            len(replies)
            for key, replies in self.replies.items()
            if kind is None or json.loads(key)["kind"] == kind
        )


class ReplayLLM(JudgeLLM):
    """An LLM client whose every answer comes from a `Replayer`."""

    def __init__(
        self,
        replayer: Replayer,
        *,
        model_name: str,
        name: str,
        role: Role,
        system_prompt: Optional[str] = None,
        first_message: Optional[str] = None,
        start_prompt: Optional[str] = None,
        **provider_settings: Any,
    ):
        super().__init__(
            name,
            role,
            system_prompt,
            first_message=first_message,
            start_prompt=start_prompt,
        )
        self.replayer = replayer
        self.model_name = model_name
        # Generation logs print these; they don't affect what's replayed.
        self.temperature = provider_settings.get("temperature")
        self.max_tokens = provider_settings.get("max_tokens")

    async def start_conversation(self) -> str:
        if self.first_message is not None:
            return self.first_message
        return await self.generate_response(self.get_initial_prompt_turns())

    async def generate_response(self, conversation_history=None) -> str:
        sent = what_the_model_sees(conversation_history)
        self._set_response_metadata("replay")
        return self.replayer.answer(
            describe_call(self.model_name, self.system_prompt, "text", sent)
        )

    async def generate_structured_response(
        self, message: Optional[str], response_model: Type[T]
    ) -> T:
        self._set_response_metadata("replay")
        reply = self.replayer.answer(
            describe_call(self.model_name, self.system_prompt, "structured", message)
        )
        return response_model.model_validate(reply)  # type: ignore[attr-defined]

    def set_system_prompt(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt


# ---------------------------------------------------------------- the recorded run


def pipeline_config(output: Path) -> dict[str, Any]:
    """The run being recorded. One job at a time, so replies replay in order."""
    return {
        "target": "SI",
        "generation": {
            "chatbot": {"name": CHATBOT, "repeats": 1},
            "user": [{"name": USER, "repeats": 1}],
            "turns": TURNS,
            "output": str(output),
            "max_concurrent": 1,
            "max_total_words": None,
            "persona_speaks_first": True,
            "user_language": "English",
            "sessions": None,
        },
        "judging": {
            "models": [{"name": JUDGE, "repeats": 1}],
            "max_concurrent": 1,
            "per_judge": False,
        },
        "scoring": {"personas": None, "skip_risk_analysis": True},
    }


def run_vera(command: str, work: Path, create_llm: Callable[..., Any]) -> Path:
    """Run `vera <command>` on the recorded run's config, with `create_llm` in
    place of the LLM factory. Returns the generation run folder.

    `command` is "pipeline" (generate, judge, score) or "generate" (only the
    conversations, from the same recording).
    """
    config = pipeline_config(work / "output")
    if command == "generate":
        config = {"target": config["target"], "generation": config["generation"]}
    work.mkdir(parents=True, exist_ok=True)
    config_file = work / f"{command}.json"
    config_file.write_text(json.dumps(config), encoding="utf-8")
    os.environ.pop(cli_config.VERA_RUN_CONFIG_ENV, None)
    with patch.object(LLMFactory, "create_llm", side_effect=create_llm):
        exit_code = vera.main(
            [command, "--config", str(config_file), "--sample", str(PERSONAS)]
        )
    if exit_code != 0:
        raise RuntimeError(f"vera {command} exited with {exit_code}")
    (run_folder,) = (work / "output").iterdir()
    return run_folder


def run_pipeline(work: Path, create_llm: Callable[..., Any]) -> Path:
    return run_vera("pipeline", work, create_llm)


def run_generate(work: Path, create_llm: Callable[..., Any]) -> Path:
    return run_vera("generate", work, create_llm)


# Each conversation file starts with a random 6-character tag.
RANDOM_TAG = re.compile(r"^[0-9a-f]{6}_")


def outputs(run_folder: Path) -> dict[str, str]:
    """The run's outputs, keyed by a stable name, with run-specific values removed.

    Random tags, timestamps, and folder names differ on every run; everything
    else must match exactly.
    """
    found = transcripts(run_folder)

    (evaluation,) = (run_folder / "evaluations").iterdir()
    with open(evaluation / "results.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["filename"] = RANDOM_TAG.sub("", row["filename"])
        row.pop("run_id")
    rows.sort(key=lambda row: (row["filename"], row["judge_instance"]))
    table = io.StringIO()
    writer = csv.DictWriter(table, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    found["results.csv"] = table.getvalue()

    scores = evaluation / "scores" / "scores.json"
    found["scores.json"] = scores.read_text(encoding="utf-8")
    return found


def transcripts(run_folder: Path) -> dict[str, str]:
    """The run's conversations, keyed like the recording's expected/ folder."""
    return {
        f"transcripts/{RANDOM_TAG.sub('', path.name)}": path.read_text(encoding="utf-8")
        for path in sorted((run_folder / "conversations").glob("*.txt"))
    }


def expected_outputs() -> dict[str, str]:
    """The outputs committed with the recording."""
    return {
        str(path.relative_to(EXPECTED)): path.read_text(encoding="utf-8")
        for path in sorted(EXPECTED.rglob("*"))
        if path.is_file()
    }


def record() -> None:
    """Run the pipeline against real models and save the calls and outputs."""
    RECORDING.unlink(missing_ok=True)
    RECORDING.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        run_folder = run_pipeline(Path(tmp), Recorder(RECORDING).create_llm)
        produced = outputs(run_folder)

    if EXPECTED.exists():
        shutil.rmtree(EXPECTED)
    for name, text in produced.items():
        path = EXPECTED / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    calls = len(RECORDING.read_text(encoding="utf-8").splitlines())
    print(f"Recorded {calls} calls and {len(produced)} outputs in {RECORDING.parent}")


if __name__ == "__main__":
    record()
