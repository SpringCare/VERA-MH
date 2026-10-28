"""Full-run replay: check a code change against a saved full run, with no model calls.

The small replay (`replay.py`) covers a handful of conversations. This one uses a
saved run of the recommended SI profile (two user models x 100 conversations,
judged and pooled), so a change is checked against real conversations at scale.

It works in two steps.

**Seed** turns the saved run into a recording, without calling any model. It runs
`vera pipeline` with a fake LLM that answers from the saved run: each user and
chatbot turn from the saved transcript, each judge question from the judge's
saved answers (`answers/by_question/`, one file per conversation, in the order
the questions were asked). Seeding is strict. Every history the code sends must
equal the saved transcript up to that turn, every judge question must be the one
the saved answers have next, and every saved turn and answer must be used. It
records every call in full, keyed like `replay.py` (model, system prompt,
everything sent), plus the outputs the current code produces. That is the
baseline.

**Check** replays the baseline with the same matching as `replay.py`: a call must
match a recorded call on everything sent, every recorded reply must be used, and
the outputs (transcripts, judge TSVs, results.csv, scores) must be identical.
Absolute paths, timestamps, and the random filename tags are ignored; log files
and charts aren't compared.

    uv run python -m tests.safe_changes.full_run seed <saved run> <baseline>
    uv run python -m tests.safe_changes.full_run check <baseline>

Or both in one: `check <baseline> --saved <saved run>` seeds first when the
baseline is missing or the saved run changed (never because the code changed),
then checks.

`<saved run>` is the folder holding the saved `p_*` generation runs (each with its
`evaluations/j_*` judge run). Seed again, which costs nothing, after a change
that is meant to alter what the models are sent or what the run writes, and
review the diff. `seed --sample N` keeps the first N personas. The slice CI checks
is committed in `tests/fixtures/full_run/`: `saved/` holds its 10 saved
conversations and judge answers (absolute paths replaced by `<REPO>`), and
`baseline/` what they were seeded into:

    uv run python -m tests.safe_changes.full_run seed \
        tests/fixtures/full_run/saved tests/fixtures/full_run/baseline --sample 5

The full run and its baseline stay local (`output/full_run_baseline/`).

A run from before the judge wrote `answers/` keeps them only in its judge logs;
`answers-from-logs <saved run>` writes them out once, with the judge's own
writers:

    uv run python -m tests.safe_changes.full_run answers-from-logs <saved run>
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
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Type, TypeVar
from unittest.mock import patch

from tests.safe_changes.replay import (
    RANDOM_TAG,
    call_key,
    describe_call,
    what_the_model_sees,
)
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
# The committed slice: the saved conversations it was seeded from, and the
# baseline CI checks. The full run and its baseline are kept locally.
COMMITTED_SAVED_RUN = REPO_ROOT / "tests/fixtures/full_run/saved"
COMMITTED_BASELINE = REPO_ROOT / "tests/fixtures/full_run/baseline"

# The settings of the saved run, as it was started. The model names are the
# ones it was run with, because they are part of every recorded call.
CHATBOT = "gpt-5.4-mini"
USERS = ["gpt-5.2", "claude-opus-4-5-20251101"]
JUDGE = {"name": "gpt-5.4", "repeats": 1, "reasoning_effort": "low"}
TURNS = 30

T = TypeVar("T")

# A saved conversation: its turns in order, each (speaker, text), where the
# speaker is a `Role` value ("persona" or "provider").
Turns = list[tuple[str, str]]

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
        speaker = Role.PERSONA.value if label == "user" else Role.PROVIDER.value
        turns.append((speaker, reply))

    history: list[dict[str, Any]] = [{"speaker": s, "response": r} for s, r in turns]
    if early_termination:
        history[-1]["early_termination"] = True
    if format_conversation_summary(history) != text:
        raise SavedRunError(f"can't read {path.name} back exactly")
    for (speaker, _), (next_speaker, _) in zip(turns, turns[1:]):
        if speaker == next_speaker:
            raise SavedRunError(f"{path.name}: two turns in a row by {speaker}")
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


# ------------------------------------------------------------------------ seeding


class SeedMismatch(AssertionError):
    """The code did something the saved run didn't."""


class Seeder:
    """Answers every call from a saved run and records it, call by call."""

    def __init__(self, saved: SavedRun, sample: Optional[int]):
        self.saved = saved
        self.calls: list[dict[str, Any]] = []
        self.used_turns: dict[tuple[str, str], int] = defaultdict(int)
        self.used_judge_calls: dict[tuple[str, str], int] = defaultdict(int)
        # A conversation's persona and chatbot are created one right after the
        # other, with no await between, so the chatbot belongs to the last persona.
        self.last_persona: Optional[tuple[str, str]] = None
        self.failures: list[str] = []
        self.sample = sample

    def create_llm(self, **settings: Any) -> "SeedLLM":
        """Stands in for `LLMFactory.create_llm`."""
        role = settings["role"]
        if role == Role.PERSONA:
            model = settings["model_name"]
            key = (model, settings["name"].removeprefix(f"{model} "))
            if key not in self.saved.turns:
                raise SeedMismatch(f"no saved conversation for {key}")
            self.last_persona = key
        elif role == Role.PROVIDER:
            if self.last_persona is None:
                raise SeedMismatch("a chatbot was created before any persona")
            key = self.last_persona
        else:
            key = self._conversation_judged(settings.get("system_prompt") or "")
        return SeedLLM(self, key, **settings)

    def _conversation_judged(self, system_prompt: str) -> tuple[str, str]:
        found = [
            key for key, text in self.saved.transcripts.items() if text in system_prompt
        ]
        if len(found) != 1:
            raise SeedMismatch(
                f"the judge's prompt matches {len(found)} saved conversations, not 1"
            )
        return found[0]

    def text_reply(self, key: tuple[str, str], role: Role, history: list) -> str:
        turns = self.saved.turns[key]
        sent = [(t["speaker"], t["response"]) for t in history if "speaker" in t]
        n = len(sent)
        if sent != turns[:n]:
            raise self._fail(f"{key}: the history sent differs from the saved one")
        if n >= len(turns):
            raise self._fail(f"{key}: asked for turn {n + 1} of {len(turns)}")
        speaker, reply = turns[n]
        if speaker != role.value:
            raise self._fail(f"{key}: turn {n + 1} belongs to {speaker}, not {role}")
        self.used_turns[key] = n + 1
        return reply

    def judge_reply(self, key: tuple[str, str], prompt: Optional[str]) -> dict:
        """The saved answer to the judge's next question in this conversation.

        The k-th question asked gets the k-th saved answer, and the prompt must
        be asking that question. A few rubric questions are asked in two
        dimensions with the same wording, so wording alone can't tell those
        apart; the order does, and the baseline records every exact prompt.
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
        self.used_judge_calls[key] = k + 1
        return reply

    def _fail(self, message: str) -> SeedMismatch:
        # Generation skips a conversation whose call fails, so the cause is kept
        # to report instead of a missing transcript later.
        self.failures.append(message)
        return SeedMismatch(message)

    def unused(self, keys: list[tuple[str, str]]) -> list[str]:
        """Saved turns and judge answers the code never asked for."""
        left = []
        for key in keys:
            if self.used_turns[key] != len(self.saved.turns[key]):
                left.append(f"{key}: {self.used_turns[key]} of the saved turns used")
            if self.used_judge_calls[key] != len(self.saved.judge_answers[key]):
                left.append(f"{key}: not every saved judge answer was used")
        return left

    def record(self, call: dict[str, Any], reply: Any) -> None:
        self.calls.append({"key": _hashed(call), "reply": reply})


def _hashed(call: dict[str, Any]) -> str:
    """A call's key as a sha256, so a full run's recording stays a manageable size.

    Matching is unchanged: two calls get the same hash only if everything sent
    is the same.
    """
    return hashlib.sha256(call_key(call).encode("utf-8")).hexdigest()


class SeedLLM(JudgeLLM):
    """An LLM client that answers from the saved run, for one conversation."""

    def __init__(
        self,
        seeder: Seeder,
        key: tuple[str, str],
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
        self.seeder = seeder
        self.key = key
        self.model_name = model_name
        self.temperature = provider_settings.get("temperature")
        self.max_tokens = provider_settings.get("max_tokens")

    async def start_conversation(self) -> str:
        return await self.generate_response(self.get_initial_prompt_turns())

    async def generate_response(self, conversation_history=None) -> str:
        sent = what_the_model_sees(conversation_history)
        self._set_response_metadata("replay")
        reply = self.seeder.text_reply(self.key, self.role, sent or [])
        call = describe_call(self.model_name, self.system_prompt, "text", sent)
        self.seeder.record(call, reply)
        return reply

    async def generate_structured_response(
        self, message: Optional[str], response_model: Type[T]
    ) -> T:
        self._set_response_metadata("replay")
        reply = self.seeder.judge_reply(self.key, message)
        call = describe_call(self.model_name, self.system_prompt, "structured", message)
        self.seeder.record(call, reply)
        return response_model.model_validate(reply)  # type: ignore[attr-defined]

    def set_system_prompt(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt


# ------------------------------------------------------------------------ checking


class HashedReplayer:
    """Answers every call from a seeded baseline, matching on everything sent."""

    def __init__(self, path: Path):
        self.replies: dict[str, list[Any]] = defaultdict(list)
        for line in path.read_text(encoding="utf-8").splitlines():
            recorded = json.loads(line)
            self.replies[recorded["key"]].append(recorded["reply"])
        self.misses: list[str] = []

    def create_llm(self, **settings: Any) -> "HashedReplayLLM":
        return HashedReplayLLM(self, **settings)

    def answer(self, call: dict[str, Any]) -> Any:
        replies = self.replies.get(_hashed(call))
        if not replies:
            sent = call["sent"]
            last = sent[-1] if isinstance(sent, list) and sent else sent
            message = (
                f"{call['model']} ({call['kind']}) was sent something the baseline "
                f"doesn't have.\nLast input: {str(last)[:500]}"
            )
            self.misses.append(message)
            raise AssertionError(message)
        return replies.pop(0)

    def unused_replies(self) -> int:
        return sum(len(replies) for replies in self.replies.values())


class HashedReplayLLM(SeedLLM):
    """Same client shape as `SeedLLM`, answering from a `HashedReplayer`."""

    def __init__(self, replayer: HashedReplayer, **settings: Any):
        super().__init__(None, ("", ""), **settings)  # type: ignore[arg-type]
        self.replayer = replayer

    async def generate_response(self, conversation_history=None) -> str:
        sent = what_the_model_sees(conversation_history)
        self._set_response_metadata("replay")
        call = describe_call(self.model_name, self.system_prompt, "text", sent)
        return self.replayer.answer(call)

    async def generate_structured_response(
        self, message: Optional[str], response_model: Type[T]
    ) -> T:
        self._set_response_metadata("replay")
        call = describe_call(self.model_name, self.system_prompt, "structured", message)
        return response_model.model_validate(self.replayer.answer(call))  # type: ignore[attr-defined]


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
        relative = path.relative_to(output)
        name = "/".join(
            # No angle brackets in file names: Windows doesn't allow them.
            RANDOM_TAG.sub("", TIMESTAMP.sub("TIMESTAMP", part))
            for part in relative.parts
        )
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
        found[name] = TIMESTAMP.sub("<TS>", text)
    return found


# How much of each differing output a failed check shows.
DIFFS_SHOWN = 5
DIFF_LINES = 20
DIFF_LINE_WIDTH = 200


def _short_diff(expected: str, produced: str) -> str:
    """The changed lines, baseline first, cut to a readable size."""
    lines = list(
        difflib.unified_diff(
            expected.splitlines(),
            produced.splitlines(),
            "baseline",
            "this run",
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


# --------------------------------------------------------------- seed and check


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


def saved_run_fingerprint(saved_folder: Path) -> str:
    """sha256 of every saved file seeding reads: transcripts and judge answers.

    A baseline stores the fingerprint of the saved run it was seeded from, so a
    check can tell when the saved run has changed and the baseline is stale.
    """
    digest = hashlib.sha256()
    files = sorted(
        [
            *saved_folder.glob("p_*/conversations/*.txt"),
            *saved_folder.glob("p_*/evaluations/j_*/answers/by_question/*.tsv"),
        ]
    )
    for path in files:
        digest.update(path.relative_to(saved_folder).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def seed(saved_folder: Path, baseline: Path, sample: Optional[int]) -> None:
    """Turn a saved run into a baseline: recorded calls and expected outputs."""
    saved = SavedRun(saved_folder)
    seeder = Seeder(saved, sample)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        try:
            output = run(work, seeder.create_llm, sample)
        finally:
            if seeder.failures:
                raise SeedMismatch(seeder.failures[0])
        produced = outputs(output, work)

    used = [key for key in saved.turns if seeder.used_turns[key]]
    left = seeder.unused(used)
    if left:
        raise SeedMismatch("the code used less of the saved run:\n" + "\n".join(left))

    if baseline.exists():
        shutil.rmtree(baseline)
    (baseline / "expected").mkdir(parents=True)
    settings = {"sample": sample, "saved_run": saved_run_fingerprint(saved_folder)}
    (baseline / "settings.json").write_text(
        json.dumps(settings, indent=2) + "\n", encoding="utf-8"
    )
    with open(baseline / "calls.jsonl", "w", encoding="utf-8") as f:
        # Sorted, so seeding again gives a readable diff rather than the calls
        # in whatever order the concurrent run happened to make them. The sort
        # is stable, so repeated identical calls keep their order.
        for call in sorted(seeder.calls, key=lambda call: call["key"]):
            f.write(json.dumps(call, ensure_ascii=False) + "\n")
    for name, text in produced.items():
        path = baseline / "expected" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(
        f"Seeded {len(seeder.calls)} calls from {len(used)} conversations and "
        f"{len(produced)} outputs into {baseline}"
    )


def seed_if_stale(
    saved_folder: Path, baseline: Path, sample: Optional[int]
) -> Optional[str]:
    """Seed `baseline` only if its inputs changed; say why, or None if it's fresh.

    The inputs are the saved run, never the code: a baseline is never seeded
    because a check failed, or every code change would quietly become the new
    expected output. When a change is meant to alter the outputs, run `seed`
    yourself and review the diff.
    """
    settings_file = baseline / "settings.json"
    if not settings_file.is_file():
        reason = "no baseline yet"
    else:
        settings = json.loads(settings_file.read_text(encoding="utf-8"))
        recorded = settings.get("saved_run")
        if recorded == saved_run_fingerprint(saved_folder):
            return None
        reason = (
            "the baseline doesn't record which saved run it came from"
            if recorded is None
            else "the saved run changed since the baseline was seeded"
        )
        sample = settings["sample"]
    seed(saved_folder, baseline, sample)
    return reason


def check(baseline: Path) -> list[str]:
    """Replay a baseline and list every difference; empty means the run matched."""
    settings = json.loads((baseline / "settings.json").read_text(encoding="utf-8"))
    sample = settings["sample"]
    replayer = HashedReplayer(baseline / "calls.jsonl")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        try:
            output = run(work, replayer.create_llm, sample)
        except Exception:
            if replayer.misses:
                return replayer.misses[:1]
            raise
        if replayer.misses:
            return replayer.misses[:1]
        produced = outputs(output, work)

    expected_root = baseline / "expected"
    expected = {
        path.relative_to(expected_root).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(expected_root.rglob("*"))
        if _compared(path)
    }
    problems = []
    differing = 0
    for name in sorted(set(expected) | set(produced)):
        if name not in produced:
            problems.append(f"missing output: {name}")
        elif name not in expected:
            problems.append(f"unexpected output: {name}")
        elif produced[name] != expected[name]:
            differing += 1
            if differing <= DIFFS_SHOWN:
                diff = _short_diff(expected[name], produced[name])
                problems.append(f"output differs: {name}\n{diff}")
            else:
                problems.append(f"output differs: {name}")
    if replayer.unused_replies():
        problems.append(f"{replayer.unused_replies()} recorded calls were never made")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    seed_parser = commands.add_parser("seed", help="turn a saved run into a baseline")
    seed_parser.add_argument("saved_run", type=Path)
    seed_parser.add_argument("baseline", type=Path)
    seed_parser.add_argument("--sample", type=int, default=None)
    logs_parser = commands.add_parser(
        "answers-from-logs",
        help="write answers/ from judge logs, for runs from before answers/ existed",
    )
    logs_parser.add_argument("saved_run", type=Path)
    check_parser = commands.add_parser("check", help="replay a baseline")
    check_parser.add_argument("baseline", type=Path)
    check_parser.add_argument(
        "--saved",
        type=Path,
        default=None,
        help="the saved run: seed first if the baseline is missing or the saved "
        "run changed (never because the code changed)",
    )
    check_parser.add_argument(
        "--sample", type=int, default=None, help="for a baseline seeded from scratch"
    )
    args = parser.parse_args()

    if args.command == "seed":
        seed(args.saved_run, args.baseline, args.sample)
        return 0
    if args.command == "answers-from-logs":
        written = answers_from_logs(args.saved_run)
        print(f"Wrote the answers of {written} judged conversations")
        return 0
    if args.saved is not None:
        reason = seed_if_stale(args.saved, args.baseline, args.sample)
        if reason:
            print(f"Seeded {args.baseline} first: {reason}")
    problems = check(args.baseline)
    # The pipeline's own progress goes to stdout, which is buffered when
    # redirected; flush it so the result comes last.
    sys.stdout.flush()
    print("\n\n".join(problems), file=sys.stderr)
    print("matches the baseline" if not problems else f"{len(problems)} differences")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
