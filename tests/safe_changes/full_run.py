"""Full-run replay: check a code change against a saved full run, with no model calls.

The small replay (`replay.py`) covers a handful of conversations. This one uses a
saved run of the recommended SI profile (two user models x 100 conversations,
judged and pooled), so a change is checked against real conversations at scale.

`check` runs `vera pipeline` with a fake LLM client that unrolls the saved run:
each user and chatbot call gets the next turn of the saved transcript, each judge
call the judge's next saved answer. On every call, the client first checks what
the pipeline handed it (the client's settings, its system prompt, the history or
judge question) against the saved run and its frozen inputs. Then the run's
outputs must equal the saved transcripts and judge answers, and the expected
scores, `results.csv`, judge TSVs, and run records.

    uv run python -m tests.safe_changes.full_run check <saved run> <expected>

A saved run keeps its conversations and judge answers, but not the system prompts
and client settings it was run with. `freeze-inputs` writes those once into
`<saved run>/inputs/`, from the code at hand, with a checksum of each saved
file so an edit to the saved run fails the check; `update-expected` writes the
expected outputs, for a change meant to alter them. See
docs/full-run-replay.md for how it works and when to use each command.

The slice CI checks is committed in `tests/fixtures/full_run/`: `saved/` holds 10
saved conversations, their judge answers and frozen inputs, and `expected/` the
other outputs. The full run stays local (`output/`).
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import csv
import difflib
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Optional, Type, TypeVar
from unittest.mock import patch

from tests.safe_changes.replay import RANDOM_TAG, what_the_model_sees
from vera_mh import cli as vera
from vera_mh.judge.answers import (
    COL_ANSWER,
    COL_EXPLANATION,
    COL_QUESTION_ID,
    conversation_answers_dir,
    write_answers_tsv,
    write_conversation_answers,
)
from vera_mh.judge.rubric_config import RubricConfig
from vera_mh.llm_clients.llm_factory import LLMFactory
from vera_mh.llm_clients.llm_interface import JudgeLLM
from vera_mh.utils.conversation_utils import format_conversation_summary
from vera_mh.utils.role import Role
from vera_mh.vera_cli import config as cli_config

REPO_ROOT = Path(__file__).resolve().parents[2]
# The committed slice CI checks. The full run is kept locally.
COMMITTED_SAVED_RUN = REPO_ROOT / "tests/fixtures/full_run/saved"
COMMITTED_EXPECTED = REPO_ROOT / "tests/fixtures/full_run/expected"

# The settings of the saved run, as it was started. The model names are the
# ones it was run with, because they are part of what every client is handed.
CHATBOT = "gpt-5.4-mini"
USERS = ["gpt-5.2", "claude-opus-4-5-20251101"]
JUDGE = {"name": "gpt-5.4", "repeats": 1, "reasoning_effort": "low"}
TURNS = 30

T = TypeVar("T")

# A saved conversation: its turns in order, each (speaker, text).
Turns = list[tuple[Role, str]]

# Run-specific values that differ between two runs of the same code.
TIMESTAMP = re.compile(r"\d{8}_\d{6}(_\d{3})?")


def pipeline_config(output: Path) -> dict[str, Any]:
    """The saved run's settings, writing to `output`."""
    return {
        "target": "SI",
        "generation": {
            "chatbot": {"name": CHATBOT, "repeats": 1},
            "user": [{"name": user, "repeats": 1} for user in USERS],
            "turns": TURNS,
            "output": str(output),
            "max_concurrent": None,
            "max_total_words": None,
            "first_speaker": "user",
            "user_language": "English",
            "sessions": None,
        },
        "judging": {"models": [JUDGE], "max_concurrent": None, "per_judge": False},
        "scoring": {"personas": "data/SI/personas.tsv", "skip_risk_analysis": False},
    }


# ------------------------------------------------------------ reading the saved run


class SavedRunError(ValueError):
    """The saved run can't be read back exactly."""


def read_transcript(path: Path) -> Turns:
    """A saved transcript's turns, read back exactly.

    Transcripts are `user: <text>` / `chatbot: <text>` blocks separated by a blank
    line. To be sure a reply containing such a line wasn't split in two, the turns
    are written out again with the formatter that wrote the file, and the result
    must match the file byte for byte.
    """
    text = path.read_text(encoding="utf-8")
    ended = re.compile(r"\n\[CONVERSATION ENDED - (\w+) signaled termination\]\n$")
    turns: Turns = []
    early_termination = False
    blocks = re.split(r"\n\n(?=(?:user|chatbot): )", text)
    for i, block in enumerate(blocks):
        label, _, rest = block.partition(": ")
        # Only the last turn keeps the newline the formatter ends each turn
        # with; between turns, the split has already removed it.
        is_last = i == len(blocks) - 1
        reply = rest[:-1] if is_last and rest.endswith("\n") else rest
        marker = ended.search(rest)
        if marker:
            reply = rest[: marker.start()]
            early_termination = True
        speaker = Role.USER if label == "user" else Role.CHATBOT
        turns.append((speaker, reply))

    history: list[dict[str, Any]] = [
        {"speaker": s.value, "response": r} for s, r in turns
    ]
    if early_termination:
        history[-1]["early_termination"] = True
    if format_conversation_summary(history) != text:
        raise SavedRunError(f"can't read {path.name} back exactly")
    for (speaker, _), (next_speaker, _) in zip(turns, turns[1:]):
        if speaker == next_speaker:
            raise SavedRunError(f"{path.name}: two turns in a row by {speaker.name}")
    return turns


def read_judge_log(path: Path) -> list[tuple[str, dict[str, Any]]]:
    """Each question the judge was asked, in order: (question ID, reply).

    Only for runs from before the judge wrote `answers/` (see
    `answers_from_logs`); seeding reads the answers files.
    """
    record_start = re.compile(
        r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3} - \S+ - \w+ - ", re.MULTILINE
    )
    messages = record_start.split(path.read_text(encoding="utf-8"))[1:]
    asked: list[tuple[str, dict[str, Any]]] = []
    question_id: Optional[str] = None
    for message in messages:
        message = message[:-1] if message.endswith("\n") else message
        heading = re.fullmatch(r"--- Question (\S+) ---", message)
        if heading:
            question_id = heading.group(1)
        elif message.startswith("STRUCTURED RESPONSE:\n"):
            if question_id is None:
                raise SavedRunError(f"{path.name}: a reply with no question")
            asked.append((question_id, _structured_reply(message, path)))
            question_id = None
    return asked


def _structured_reply(message: str, path: Path) -> dict[str, Any]:
    """`answer='Yes' reasoning='...'` (how the reply was logged) as a dict."""
    found = re.fullmatch(
        r"STRUCTURED RESPONSE:\n"
        r"answer=('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\") "
        r"reasoning=('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")",
        message,
        re.DOTALL,
    )
    if found is None:
        raise SavedRunError(f"{path.name}: can't read a judge reply")
    return {
        "answer": ast.literal_eval(found.group(1)),
        "reasoning": ast.literal_eval(found.group(2)),
    }


def answers_from_logs(saved_folder: Path) -> int:
    """Write each judge run's `answers/` from its logs, for runs from before the
    judge wrote them. Uses the judge's own writers, so the files are exactly the
    ones a current run writes. Returns how many conversations were written."""
    written = 0
    for evaluation in sorted(saved_folder.glob("p_*/evaluations/j_*")):
        for log in sorted((evaluation / "logs").glob("*.log")):
            entries = [
                {"question_id": question_id, **reply}
                for question_id, reply in read_judge_log(log)
            ]
            output = conversation_answers_dir(evaluation) / f"{log.stem}.tsv"
            write_conversation_answers(output, entries)
            written += 1
        write_answers_tsv(evaluation, si_rubric().question_order)
    return written


def read_saved_answers(path: Path) -> list[tuple[str, dict[str, Any]]]:
    """A conversation's `answers/by_question` file: (question ID, reply) in the
    order the judge asked them."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if lines[0].split("\t") != [COL_QUESTION_ID, COL_ANSWER, COL_EXPLANATION]:
        raise SavedRunError(f"{path.name}: unexpected columns")
    answers = []
    for line in lines[1:]:
        question_id, answer, explanation = line.split("\t")
        answers.append((question_id, {"answer": answer, "reasoning": explanation}))
    return answers


def si_rubric() -> RubricConfig:
    """The SI rubric the saved run was judged with, loaded by the judge's code."""
    folder = REPO_ROOT / "data/SI"
    return asyncio.run(
        RubricConfig.from_paths(
            rubric_file=str(folder / "rubric.tsv"),
            rubric_prompt_beginning_file=str(folder / "rubric_prompt_beginning.txt"),
            question_prompt_file=str(folder / "question_prompt.txt"),
        )
    )


class SavedRun:
    """A saved full run's conversations and judge answers, by (user model, persona)."""

    def __init__(self, folder: Path):
        users = "|".join(re.escape(user) for user in USERS)
        name = re.compile(rf"^[0-9a-f]{{6}}_(?P<persona>.+)_(?P<user>{users})_run1$")
        self.turns: dict[tuple[str, str], Turns] = {}
        self.transcripts: dict[tuple[str, str], str] = {}
        self.judge_answers: dict[tuple[str, str], list[tuple[str, dict]]] = {}
        self.question_text = {
            question_id: data["question"]
            for question_id, data in si_rubric().question_flow_data.items()
        }
        for run in sorted(folder.glob("p_*")):
            (evaluation,) = (run / "evaluations").glob("j_*")
            for path in sorted((run / "conversations").glob("*.txt")):
                found = name.match(path.stem)
                if found is None:
                    raise SavedRunError(f"unexpected transcript name: {path.name}")
                key = (found["user"], found["persona"])
                self.turns[key] = read_transcript(path)
                self.transcripts[key] = path.read_text(encoding="utf-8")
                by_question = conversation_answers_dir(evaluation)
                (answers,) = by_question.glob(f"{path.stem}_*.tsv")
                self.judge_answers[key] = read_saved_answers(answers)
        if not self.turns:
            raise SavedRunError(f"no saved conversations under {folder}")


# ------------------------------------------------------------------ frozen inputs

# What the pipeline hands the clients, other than the conversation itself, is
# frozen in `<saved run>/inputs/`. Each input is a (kind, name) pair: a client's
# settings by "<role> <model>", a persona's system prompt by persona, a judge
# question by its rubric ID. The chatbot's and the judge's system prompts are
# the same for every conversation, so their name is empty.
INPUTS = "inputs"
CLIENT = "client"
PERSONA_PROMPT = "persona system prompt"
CHATBOT_PROMPT = "chatbot system prompt"
JUDGE_PROMPT = "judge system prompt"
JUDGE_QUESTION = "judge question"
InputKey = tuple[str, str]

# Where each kind of text input is kept, under `inputs/`. Client settings are
# in `settings.json`.
INPUT_FILES = {
    CHATBOT_PROMPT: "chatbot_system_prompt.txt",
    JUDGE_PROMPT: "judge_system_prompt.txt",
    PERSONA_PROMPT: "persona_system_prompts/{name}.txt",
    JUDGE_QUESTION: "judge_questions/{name}.txt",
}

# Stands for the transcript in the frozen judge system prompt, which is the same
# for every conversation apart from the transcript in it.
CONVERSATION = "<CONVERSATION>"


def input_label(key: InputKey) -> str:
    kind, name = key
    return f"{kind} {name}" if name else kind


def read_inputs(folder: Path) -> tuple[dict[str, Any], dict[InputKey, Any]]:
    """A saved run's `settings.json` (the `--sample`, the commit, the saved
    files' checksums, the client settings), and its frozen inputs."""
    if not folder.is_dir():
        raise SavedRunError(f"no frozen inputs in {folder}: run freeze-inputs first")
    settings = json.loads((folder / "settings.json").read_text(encoding="utf-8"))
    inputs: dict[InputKey, Any] = {
        (CLIENT, name): client for name, client in settings["clients"].items()
    }
    for kind in (CHATBOT_PROMPT, JUDGE_PROMPT):
        inputs[(kind, "")] = (folder / INPUT_FILES[kind]).read_text(encoding="utf-8")
    for kind in (PERSONA_PROMPT, JUDGE_QUESTION):
        pattern = INPUT_FILES[kind].format(name="*")
        for path in sorted(folder.glob(pattern)):
            inputs[(kind, path.stem)] = path.read_text(encoding="utf-8")
    return settings, inputs


def write_inputs(
    folder: Path,
    inputs: dict[InputKey, Any],
    sample: Optional[int],
    frozen_from: str,
    saved_files: dict[str, str],
) -> None:
    clients = {name: value for (kind, name), value in inputs.items() if kind == CLIENT}
    settings = {
        "frozen_from": frozen_from,
        "sample": sample,
        "saved_files": saved_files,
        "clients": clients,
    }
    folder.mkdir(parents=True)
    (folder / "settings.json").write_text(
        json.dumps(settings, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for (kind, name), text in inputs.items():
        if kind == CLIENT:
            continue
        path = folder / INPUT_FILES[kind].format(name=name)
        path.parent.mkdir(parents=True, exist_ok=True)
        # newline="": written exactly, with no line endings translated.
        path.write_text(text, encoding="utf-8", newline="")


def current_commit() -> str:
    """The commit the code at hand is from, noting uncommitted changes to it."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    changes = subprocess.run(
        ["git", "status", "--porcelain", "--", "src", "data"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return f"{commit} (with uncommitted changes)" if changes else commit


def _as_json(value: Any) -> Any:
    """`value` as it reads back from JSON, so it compares equal to a frozen one."""
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _as_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, indent=2)


class InputsRecorder:
    """Keeps every input the pipeline hands the clients, for `freeze-inputs`."""

    def __init__(self) -> None:
        self.inputs: dict[InputKey, Any] = {}

    def see(self, key: InputKey, value: Any) -> Optional[str]:
        """Keep an input; return why the run can't go on, or None."""
        if key in self.inputs and self.inputs[key] != value:
            return (
                f"{input_label(key)} differs between two calls, so it can't be frozen"
            )
        self.inputs[key] = value
        return None


class InputsChecker:
    """Compares every input the pipeline hands the clients with the frozen ones.

    A difference doesn't stop the run: the saved replies don't depend on it, so
    the run goes on and every differing input is reported, once.
    """

    def __init__(self, frozen: dict[InputKey, Any]):
        self.frozen = frozen
        self.seen: set[InputKey] = set()
        self.problems: dict[InputKey, str] = {}

    def see(self, key: InputKey, value: Any) -> Optional[str]:
        self.seen.add(key)
        if key in self.problems:
            return None
        if key not in self.frozen:
            self.problems[key] = (
                f"input not in the frozen inputs: {input_label(key)}\n"
                f"{_as_text(value)[:DIFF_LINE_WIDTH]}"
            )
        elif value != self.frozen[key]:
            diff = _short_diff(
                _as_text(self.frozen[key]), _as_text(value), "frozen", "this run"
            )
            self.problems[key] = f"input differs: {input_label(key)}\n{diff}"
        return None

    def unused(self) -> list[str]:
        return [
            f"frozen input never handed to a client: {input_label(key)}"
            for key in self.frozen
            if key not in self.seen
        ]


# ------------------------------------------------------------- unrolling the run


class ReplayMismatch(AssertionError):
    """The code made a call the saved run can't answer."""


# The fields of each turn in the history a client is handed, as the conversation
# simulator writes them, without `logging` (see `what_the_model_sees`).
HISTORY_FIELDS = {"turn", "speaker", "input", "response", "early_termination"}


def history_difference(sent: list[dict[str, Any]], turns: Turns) -> Optional[str]:
    """How the history a client was handed differs from the saved turns, or None.

    Each turn's `speaker` is compared as a `Role`, so renaming a `Role` value
    isn't a difference. Each `input` must be the turn before's text, except the
    first turn's: that is the user model's start prompt, which the client
    supplies itself rather than being handed it.
    """
    for i, (turn, (speaker, text)) in enumerate(zip(sent, turns)):
        where = f"turn {i + 1} of the history"
        if set(turn) != HISTORY_FIELDS:
            return f"{where} has the fields {sorted(turn)}"
        if turn["turn"] != i + 1:
            return f"{where} is numbered {turn['turn']}"
        try:
            sent_speaker = Role(turn["speaker"])
        except ValueError:
            return f"{where} has the speaker {turn['speaker']!r}, which is no Role"
        if sent_speaker != speaker:
            return f"{where} is by {sent_speaker.name}, not {speaker.name}"
        if turn["response"] != text:
            diff = _short_diff(text, turn["response"], "saved", "sent")
            return f"{where} differs from the saved turn\n{diff}"
        if i > 0 and turn["input"] != turns[i - 1][1]:
            return f"{where} has an input that isn't the turn before"
        if turn["early_termination"] is not False:
            return f"{where} is marked as ending the conversation"
    return None


class Unroller:
    """Answers every call from the saved run, and hands `inputs` what each client
    was given: an `InputsRecorder` to freeze them, an `InputsChecker` to check."""

    def __init__(self, saved: SavedRun, inputs: InputsRecorder | InputsChecker):
        self.saved = saved
        self.inputs = inputs
        self.used_turns: dict[tuple[str, str], int] = defaultdict(int)
        self.used_judge_calls: dict[tuple[str, str], int] = defaultdict(int)
        # A conversation's persona and chatbot are created one right after the
        # other, with no await between, so the chatbot belongs to the last persona.
        self.last_persona: Optional[tuple[str, str]] = None
        self.failures: list[str] = []

    def create_llm(self, **settings: Any) -> "SavedRunLLM":
        """Stands in for `LLMFactory.create_llm`."""
        role: Optional[Role] = settings["role"]
        model = settings["model_name"]
        system_prompt = settings.get("system_prompt") or ""
        client = {
            k: v for k, v in settings.items() if k not in ("role", "system_prompt")
        }
        if role == Role.USER:
            persona = settings["name"].removeprefix(f"{model} ")
            key = (model, persona)
            if key not in self.saved.turns:
                raise self._fail(f"no saved conversation for {key}")
            self.last_persona = key
            # The client's name holds the persona, so it's frozen without it.
            client["name"] = settings["name"].replace(persona, "<persona>")
            self._see((PERSONA_PROMPT, persona), system_prompt)
        elif role == Role.CHATBOT:
            if self.last_persona is None:
                raise self._fail("a chatbot was created before any persona")
            key = self.last_persona
            self._see((CHATBOT_PROMPT, ""), system_prompt)
        else:
            key = self._conversation_judged(system_prompt)
            transcript = self.saved.transcripts[key]
            self._see(
                (JUDGE_PROMPT, ""), system_prompt.replace(transcript, CONVERSATION)
            )
        who = role.value if role is not None else "judge"
        self._see((CLIENT, f"{who} {model}"), _as_json(client))
        return SavedRunLLM(self, key, **settings)

    def _conversation_judged(self, system_prompt: str) -> tuple[str, str]:
        found = [
            key for key, text in self.saved.transcripts.items() if text in system_prompt
        ]
        if len(found) != 1:
            raise self._fail(
                f"the judge's prompt holds {len(found)} saved conversations, not 1"
            )
        return found[0]

    def text_reply(
        self, key: tuple[str, str], role: Role, history: list[dict[str, Any]]
    ) -> str:
        """The saved turn after `history`, which must be the saved turns so far."""
        turns = self.saved.turns[key]
        n = len(history)
        if n >= len(turns):
            raise self._fail(f"{key}: asked for turn {n + 1} of {len(turns)}")
        difference = history_difference(history, turns)
        if difference:
            raise self._fail(f"{key}: {difference}")
        speaker, reply = turns[n]
        if speaker != role:
            raise self._fail(
                f"{key}: turn {n + 1} is by {speaker.name}, not {role.name}"
            )
        self.used_turns[key] = n + 1
        return reply

    def judge_reply(self, key: tuple[str, str], prompt: Optional[str]) -> dict:
        """The saved answer to the judge's next question in this conversation.

        The k-th question asked gets the k-th saved answer, and the prompt must
        be asking that question. A few rubric questions are asked in two
        dimensions with the same wording, so wording alone can't tell those
        apart; the order does, and the frozen prompt of each question ID is
        compared in full.
        """
        answers = self.saved.judge_answers[key]
        k = self.used_judge_calls[key]
        if k >= len(answers):
            raise self._fail(f"{key}: judge question {k + 1} of {len(answers)}")
        question_id, reply = answers[k]
        expected = f"Question: {self.saved.question_text[question_id]}"
        if not (prompt or "").startswith(expected):
            raise self._fail(
                f"{key}: judge question {k + 1} should be question {question_id}"
            )
        self._see((JUDGE_QUESTION, question_id), prompt or "")
        self.used_judge_calls[key] = k + 1
        return reply

    def _see(self, key: InputKey, value: Any) -> None:
        problem = self.inputs.see(key, value)
        if problem:
            raise self._fail(problem)

    def _fail(self, message: str) -> ReplayMismatch:
        # Generation skips a conversation whose call fails, so the cause is kept
        # to report instead of a missing transcript later.
        self.failures.append(message)
        return ReplayMismatch(message)

    def unused(self) -> list[str]:
        """Saved turns and judge answers the code never asked for."""
        left = []
        for key, turns in self.saved.turns.items():
            if self.used_turns[key] != len(turns):
                left.append(f"{key}: {self.used_turns[key]} of the saved turns used")
            if self.used_judge_calls[key] != len(self.saved.judge_answers[key]):
                left.append(f"{key}: not every saved judge answer was used")
        return left


class SavedRunLLM(JudgeLLM):
    """An LLM client that answers from the saved run, for one conversation."""

    def __init__(
        self,
        unroller: Unroller,
        key: tuple[str, str],
        *,
        model_name: str,
        name: str,
        role: Optional[Role],
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
        self.unroller = unroller
        self.key = key
        self.model_name = model_name
        self.temperature = provider_settings.get("temperature")
        self.max_tokens = provider_settings.get("max_tokens")

    async def start_conversation(self) -> str:
        self._set_response_metadata("replay")
        return self.unroller.text_reply(self.key, self.conversation_role(), [])

    async def generate_response(self, conversation_history=None) -> str:
        self._set_response_metadata("replay")
        sent = what_the_model_sees(conversation_history) or []
        return self.unroller.text_reply(self.key, self.conversation_role(), sent)

    async def generate_structured_response(
        self, message: Optional[str], response_model: Type[T]
    ) -> T:
        self._set_response_metadata("replay")
        reply = self.unroller.judge_reply(self.key, message)
        return response_model.model_validate(reply)  # type: ignore[attr-defined]

    def set_system_prompt(self, system_prompt: str) -> None:
        # The frozen inputs hold each client's system prompt as it was created.
        raise self.unroller._fail("a system prompt was changed after creation")


# ------------------------------------------------------------------------- outputs


def outputs(output: Path, work: Path) -> dict[str, str]:
    """Every compared output under `output`, keyed by a stable name.

    Folder names lose their timestamps and files their random tags; contents lose
    the work folder's and the repository's paths, and timestamps. Logs and charts
    aren't compared.
    """
    found: dict[str, str] = {}
    for path in sorted(output.rglob("*")):
        if not _compared(path):
            continue
        if "logs" in path.parts or path.suffix == ".png":
            continue
        if path.name == "pool_metadata.json":  # paths and a creation time only
            continue
        # The run record's hash covers absolute paths, which here are the
        # temporary work folder's, so it differs on every run.
        if path.name == "config.json.sha256":
            continue
        found[_stable_name(path.relative_to(output))] = _normalized(path, work)
    return found


# The outputs a replay must reproduce from the saved run itself: its transcripts
# and judge answers. The other outputs are compared with `expected/`.
SAVED_OUTPUTS = (
    "p_*/conversations/*.txt",
    "p_*/evaluations/j_*/answers/by_question/*.tsv",
)


def saved_outputs(saved_folder: Path) -> dict[str, str]:
    """The saved run's transcripts and judge answers, keyed like `outputs`."""
    # Absolute, so that a relative folder such as `output` isn't also replaced
    # wherever the word appears in a transcript.
    saved_folder = saved_folder.resolve()
    return {
        _stable_name(path.relative_to(saved_folder)): _normalized(path, saved_folder)
        for pattern in SAVED_OUTPUTS
        for path in sorted(saved_folder.glob(pattern))
    }


def _from_saved_run(name: str) -> bool:
    return any(PurePosixPath(name).match(pattern) for pattern in SAVED_OUTPUTS)


def _stable_name(relative: Path) -> str:
    return "/".join(
        # No angle brackets in file names: Windows doesn't allow them.
        RANDOM_TAG.sub("", TIMESTAMP.sub("TIMESTAMP", part))
        for part in relative.parts
    )


def _normalized(path: Path, work: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if path.name == "results.csv":
        text = _normalized_rows(text, delimiter=",")
    elif path.name == "answers.tsv":
        text = _normalized_rows(text, delimiter="\t")
    elif path.name == "config.json":
        text = _without_provenance(text)
    # Run records hold absolute paths: the temporary work folder's, and the
    # repository's for input files, which differs on every machine.
    # The resolved path first: on macOS the temporary folder is a symlink
    # (/var -> /private/var), and the run writes the resolved path.
    text = text.replace(str(work.resolve()), "<WORK>").replace(str(work), "<WORK>")
    text = text.replace(str(REPO_ROOT), "<REPO>")
    return TIMESTAMP.sub("<TS>", text)


def read_expected(expected_root: Path) -> dict[str, str]:
    return {
        path.relative_to(expected_root).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(expected_root.rglob("*"))
        if _compared(path)
    }


def output_differences(wanted: dict[str, str], produced: dict[str, str]) -> list[str]:
    problems = []
    differing = 0
    for name in sorted(set(wanted) | set(produced)):
        if name not in produced:
            problems.append(f"missing output: {name}")
        elif name not in wanted:
            problems.append(f"unexpected output: {name}")
        elif produced[name] != wanted[name]:
            differing += 1
            if differing <= DIFFS_SHOWN:
                diff = _short_diff(wanted[name], produced[name], "expected", "this run")
                problems.append(f"output differs: {name}\n{diff}")
            else:
                problems.append(f"output differs: {name}")
    return problems


# How much of each differing output or input a failed check shows.
DIFFS_SHOWN = 5
DIFF_LINES = 20
DIFF_LINE_WIDTH = 200


def _short_diff(before: str, after: str, before_name: str, after_name: str) -> str:
    """The changed lines, cut to a readable size."""
    lines = list(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            before_name,
            after_name,
            lineterm="",
            n=0,
        )
    )
    shown = [
        line if len(line) <= DIFF_LINE_WIDTH else line[:DIFF_LINE_WIDTH] + "..."
        for line in lines[:DIFF_LINES]
    ]
    if len(lines) > DIFF_LINES:
        shown.append(f"... {len(lines) - DIFF_LINES} more diff lines")
    return "\n".join(shown)


def _compared(path: Path) -> bool:
    """A file, and not a hidden one: macOS Finder writes a binary `.DS_Store`
    into every folder it opens, which is no output of a run."""
    return path.is_file() and not path.name.startswith(".")


def _normalized_rows(text: str, delimiter: str) -> str:
    """A per-conversation table (results.csv, answers.tsv) with random tags
    removed and rows in a fixed order: they're written in the order the
    conversations finished, which varies."""
    rows = list(csv.DictReader(io.StringIO(text), delimiter=delimiter))
    if not rows:
        return text
    for row in rows:
        row["filename"] = RANDOM_TAG.sub("", row["filename"])
    rows.sort(key=lambda row: (row["filename"], row.get("judge_instance", "")))
    table = io.StringIO()
    writer = csv.DictWriter(
        table, fieldnames=list(rows[0]), delimiter=delimiter, lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return table.getvalue()


def _without_provenance(text: str) -> str:
    """The run record without `provenance`: its start time and git commit
    change with every run and every commit, not with what the run did."""
    record = json.loads(text)
    record.pop("provenance", None)
    return json.dumps(record, indent=2, sort_keys=True) + "\n"


# ---------------------------------------------------------------------- commands


def run(work: Path, create_llm: Any, sample: Optional[int]) -> Path:
    """Run the saved run's pipeline in `work`, with `create_llm` in place of the
    LLM factory, and return its output folder."""
    work.mkdir(parents=True, exist_ok=True)
    config_file = work / "pipeline.json"
    config_file.write_text(
        json.dumps(pipeline_config(work / "output")), encoding="utf-8"
    )
    sample_args = ["--sample", str(sample)] if sample is not None else []
    os.environ.pop(cli_config.VERA_RUN_CONFIG_ENV, None)
    with patch.object(LLMFactory, "create_llm", side_effect=create_llm):
        exit_code = vera.main(["pipeline", "--config", str(config_file), *sample_args])
    if exit_code != 0:
        raise RuntimeError(f"vera pipeline exited with {exit_code}")
    return work / "output"


def replay(
    saved: SavedRun, inputs: InputsRecorder | InputsChecker, sample: Optional[int]
) -> tuple[dict[str, str], list[str]]:
    """Run the pipeline with every call answered from the saved run.

    Returns the run's outputs, and the calls the saved run couldn't answer and
    the saved turns and answers never asked for. A call the saved run can't
    answer stops its conversation, so then only the first is returned, with no
    outputs: the rest follow from it.
    """
    unroller = Unroller(saved, inputs)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        try:
            output = run(work, unroller.create_llm, sample)
        except Exception:
            if unroller.failures:
                return {}, unroller.failures[:1]
            raise
        if unroller.failures:
            return {}, unroller.failures[:1]
        produced = outputs(output, work)
    return produced, unroller.unused()


def check(saved_folder: Path, expected: Path) -> list[str]:
    """Replay a saved run and list every difference; empty means it matched."""
    settings, frozen = read_inputs(saved_folder / INPUTS)
    changes = saved_files_changes(saved_folder, settings["saved_files"])
    if changes:
        return changes
    saved = SavedRun(saved_folder)
    checker = InputsChecker(frozen)
    produced, problems = replay(saved, checker, settings["sample"])
    # Inputs first: a call the saved run can't answer may follow from one.
    problems = [*checker.problems.values(), *problems]
    if not produced:
        return problems
    wanted = {**saved_outputs(saved_folder), **read_expected(expected)}
    return [*problems, *checker.unused(), *output_differences(wanted, produced)]


def saved_files_checksums(saved_folder: Path) -> dict[str, str]:
    """sha256 of each saved transcript and judge answers file, by its path.

    Hashed as text, with line endings normalized, so a checkout that turns
    `\n` into `\r\n` doesn't count as a change.
    """
    return {
        path.relative_to(saved_folder).as_posix(): hashlib.sha256(
            path.read_text(encoding="utf-8").encode("utf-8")
        ).hexdigest()
        for pattern in SAVED_OUTPUTS
        for path in sorted(saved_folder.glob(pattern))
    }


def saved_files_changes(saved_folder: Path, recorded: dict[str, str]) -> list[str]:
    """The saved files that changed since the inputs were frozen.

    The saved transcripts are both the replies the fake client gives and the
    transcripts the run must reproduce, so an edit to one moves both sides and
    a check would still pass. The checksums make such an edit fail instead.
    """
    current = saved_files_checksums(saved_folder)
    changes = []
    for name in sorted(set(recorded) | set(current)):
        if name not in current:
            changes.append(f"saved file missing since the inputs were frozen: {name}")
        elif name not in recorded:
            changes.append(f"saved file added since the inputs were frozen: {name}")
        elif current[name] != recorded[name]:
            changes.append(f"saved file changed since the inputs were frozen: {name}")
    return changes


def update_expected(saved_folder: Path, expected: Path) -> list[str]:
    """Write the outputs the code at hand produces from the saved run into
    `expected`. Refuses, returning why, if an input differs from the frozen ones
    or the saved transcripts and judge answers changed or aren't reproduced."""
    settings, frozen = read_inputs(saved_folder / INPUTS)
    changes = saved_files_changes(saved_folder, settings["saved_files"])
    if changes:
        return changes
    saved = SavedRun(saved_folder)
    checker = InputsChecker(frozen)
    produced, problems = replay(saved, checker, settings["sample"])
    problems = [*checker.problems.values(), *problems]
    if produced:
        from_saved = {
            name: text for name, text in produced.items() if _from_saved_run(name)
        }
        problems += checker.unused()
        problems += output_differences(saved_outputs(saved_folder), from_saved)
    if problems:
        return problems

    if expected.exists():
        shutil.rmtree(expected)
    for name, text in produced.items():
        if _from_saved_run(name):
            continue
        path = expected / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(f"Wrote {len(list(expected.rglob('*.*')))} expected outputs into {expected}")
    return []


def freeze_inputs(
    saved_folder: Path, sample: Optional[int], replace: bool
) -> list[str]:
    """Write what the code at hand hands the clients into `<saved run>/inputs/`,
    with a checksum of each saved transcript and judge answers file.

    Refuses, returning why, if the saved transcripts and judge answers aren't
    reproduced, or an input differs between two calls that should share it.
    """
    folder = saved_folder / INPUTS
    if folder.exists() and not replace:
        return [f"{folder} exists; pass --replace to freeze the inputs again"]
    saved = SavedRun(saved_folder)
    recorder = InputsRecorder()
    produced, problems = replay(saved, recorder, sample)
    if produced:
        from_saved = {
            name: text for name, text in produced.items() if _from_saved_run(name)
        }
        problems += output_differences(saved_outputs(saved_folder), from_saved)
    if problems:
        return problems

    if folder.exists():
        shutil.rmtree(folder)
    write_inputs(
        folder,
        recorder.inputs,
        sample,
        frozen_from=current_commit(),
        saved_files=saved_files_checksums(saved_folder),
    )
    print(f"Froze {len(recorder.inputs)} inputs into {folder}")
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    check_parser = commands.add_parser("check", help="replay a saved run")
    check_parser.add_argument("saved_run", type=Path)
    check_parser.add_argument("expected", type=Path)
    update_parser = commands.add_parser(
        "update-expected",
        help="write the outputs the code produces, for a change meant to alter them",
    )
    update_parser.add_argument("saved_run", type=Path)
    update_parser.add_argument("expected", type=Path)
    freeze_parser = commands.add_parser(
        "freeze-inputs",
        help="write the system prompts and client settings into <saved run>/inputs/",
    )
    freeze_parser.add_argument("saved_run", type=Path)
    freeze_parser.add_argument(
        "--sample", type=int, default=None, help="the run's first N personas only"
    )
    freeze_parser.add_argument(
        "--replace", action="store_true", help="replace existing frozen inputs"
    )
    logs_parser = commands.add_parser(
        "answers-from-logs",
        help="write answers/ from judge logs, for runs from before answers/ existed",
    )
    logs_parser.add_argument("saved_run", type=Path)
    args = parser.parse_args()

    if args.command == "answers-from-logs":
        written = answers_from_logs(args.saved_run)
        print(f"Wrote the answers of {written} judged conversations")
        return 0
    if args.command == "check":
        problems = check(args.saved_run, args.expected)
    elif args.command == "update-expected":
        problems = update_expected(args.saved_run, args.expected)
    else:
        problems = freeze_inputs(args.saved_run, args.sample, args.replace)
    # The pipeline's own progress goes to stdout, which is buffered when
    # redirected; flush it so the result comes last.
    sys.stdout.flush()
    print("\n\n".join(problems), file=sys.stderr)
    if args.command == "check":
        print(
            "matches the saved run" if not problems else f"{len(problems)} differences"
        )
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
