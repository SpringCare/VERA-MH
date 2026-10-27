"""Tests for `utils/run_root.py`, and the scans that keep run creation there.

The scans read the source with `ast` rather than relying on import-linter,
which sees imports but not calls (docs/design/run-lifecycle.md, Enforcement).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.mocks.run_record import minimal_run_record
from vera_mh.utils.run_root import RunRootExistsError, create_run_root

SRC = Path(__file__).resolve().parents[3] / "src" / "vera_mh"

# Folder creation allowed outside `create_run_root`, by (file, function). Each
# of these writes inside a run that already exists; none creates a run folder.
# A new entry needs the same justification.
FOLDER_CREATION_ALLOWED = {
    ("generate/runner.py", "run_single_conversation"): "conversations/ and logs/",
    ("judge/answers.py", "write_conversation_answers"): "answers/ per conversation",
    ("judge/answers.py", "write_answers_tsv"): "the run's answers.tsv folder",
    ("judge/llm_judge.py", "__init__"): "the judge's log folder",
    ("judge/llm_judge.py", "_save_iterative_evaluation"): "one evaluation TSV",
    ("judge/runner.py", "judge_conversations"): "an existing run under --into",
    ("score/aggregate.py", "scores_output_dir"): "scores/ inside a judge run",
    ("score/aggregate.py", "save_results_json"): "scores.json",
    ("score/aggregate.py", "score_results_by_risk"): "scores_by_risk.json",
    ("score/compare.py", "create_comparison_graphic"): "a chart file",
    ("score/pool.py", "run_pooling"): "scores/ inside the pooled run",
    ("score/viz.py", "_save_figure"): "a chart file",
    ("utils/conversation_utils.py", "save_conversation_to_file"): "a transcript",
    ("utils/logging_utils.py", "setup_conversation_logger"): "a log file",
    ("utils/run_root.py", "create_run_root"): "the one place run folders start",
}

# `datetime.now()` allowed outside `run_started_at`. These measure durations or
# stamp log lines; none names a run.
CLOCK_ALLOWED = {
    ("generate/runner.py", "run_conversations"): "run duration",
    ("judge/runner.py", "_worker"): "log line timestamps",
    ("judge/runner.py", "_run_workers_with_queue"): "log line timestamps",
    ("judge/runner.py", "judge_conversations"): "batch duration",
    ("llm_clients/llm_interface.py", "_set_response_metadata"): "response time",
    ("utils/naming.py", "add_timestamp_to_path"): "comparison chart file names",
    ("utils/run_root.py", "run_started_at"): "the run's one clock read",
}


def _calls(attribute: str, receiver: str | None = None) -> set[tuple[str, str]]:
    """Every (file, enclosing function) in `src/vera_mh` that calls `.attribute`.

    With `receiver`, only calls on that bare name count (`datetime.now`, not
    `something.now`).
    """
    found: set[tuple[str, str]] = set()
    for path in sorted(SRC.rglob("*.py")):
        relative = path.relative_to(SRC).as_posix()

        def visit(node: ast.AST, function: str) -> None:
            for child in ast.iter_child_nodes(node):
                enclosing = function
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    enclosing = child.name
                if (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and child.func.attr == attribute
                    and (
                        receiver is None
                        or (
                            isinstance(child.func.value, ast.Name)
                            and child.func.value.id == receiver
                        )
                    )
                ):
                    found.add((relative, function))
                visit(child, enclosing)

        visit(ast.parse(path.read_text(encoding="utf-8")), "<module>")
    return found


def test_creates_the_run_folder_and_its_parents(tmp_path: Path) -> None:
    parent = tmp_path / "output" / "nested"
    root = create_run_root(parent, "run_1", minimal_run_record())

    assert root == tmp_path / "output" / "nested" / "run_1"
    assert root.is_dir()


def test_refuses_a_run_folder_that_already_exists(tmp_path: Path) -> None:
    create_run_root(tmp_path, "run_1", minimal_run_record())

    with pytest.raises(RunRootExistsError, match="run_1"):
        create_run_root(tmp_path, "run_1", minimal_run_record())


def test_folders_are_created_only_in_allowed_places() -> None:
    created = _calls("mkdir") | _calls("makedirs")

    unexpected = sorted(created - FOLDER_CREATION_ALLOWED.keys())
    assert not unexpected, (
        "New folder creation outside create_run_root. A new run folder must use "
        "utils.run_root.create_run_root; a folder inside an existing run needs "
        f"an entry in FOLDER_CREATION_ALLOWED: {unexpected}"
    )


def test_clock_is_read_only_in_allowed_places() -> None:
    clocks = _calls("now", receiver="datetime")

    unexpected = sorted(clocks - CLOCK_ALLOWED.keys())
    assert not unexpected, (
        "New datetime.now() call. A run's timestamps come from "
        "utils.run_root.run_started_at; a duration or log stamp needs an entry "
        f"in CLOCK_ALLOWED: {unexpected}"
    )


def test_allow_lists_have_no_stale_entries() -> None:
    """An entry whose call was removed should go, so the lists stay honest."""
    created = _calls("mkdir") | _calls("makedirs")
    clocks = _calls("now", receiver="datetime")

    assert sorted(FOLDER_CREATION_ALLOWED.keys() - created) == []
    assert sorted(CLOCK_ALLOWED.keys() - clocks) == []
