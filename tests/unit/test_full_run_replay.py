"""Replay a slice of a saved full run and require identical outputs.

tests/fixtures/full_run/ holds a slice of a saved run of the recommended SI
profile (`tests/safe_changes/full_run.py`): its transcripts and judge answers,
the system prompts and client settings it was run with (`saved/inputs/`), and
the other outputs (`expected/`). Every user, chatbot, and judge answer comes
from the saved run, so the replay is deterministic. A code change that alters
what a client is handed fails on that input; one that alters a transcript,
judge TSV, results.csv, or score fails the comparison.
"""

import shutil
from pathlib import Path

import pytest

from tests.safe_changes import full_run


@pytest.mark.timeout(120)
def test_full_run_slice_matches_the_saved_run() -> None:
    problems = full_run.check(full_run.COMMITTED_SAVED_RUN, full_run.COMMITTED_EXPECTED)
    assert not problems, "\n\n".join(problems)


@pytest.mark.timeout(120)
def test_hidden_files_in_the_expected_outputs_are_ignored(tmp_path: Path) -> None:
    """macOS Finder drops a binary `.DS_Store` into every folder it opens."""
    expected = tmp_path / "expected"
    shutil.copytree(full_run.COMMITTED_EXPECTED, expected)
    (expected / ".DS_Store").write_bytes(b"\x00\x00\x00\x01Bud1\x80")

    problems = full_run.check(full_run.COMMITTED_SAVED_RUN, expected)
    assert not problems, "\n\n".join(problems)


@pytest.mark.timeout(120)
def test_a_changed_output_fails_and_shows_what_changed(tmp_path: Path) -> None:
    expected = tmp_path / "expected"
    shutil.copytree(full_run.COMMITTED_EXPECTED, expected)
    scores = next(expected.glob("p_*/evaluations/j_*/scores/scores.json"))
    original = scores.read_text(encoding="utf-8")
    edited_text = original.replace('"vera_score": ', '"vera_score": 1', 1)
    assert edited_text != original
    scores.write_text(edited_text, encoding="utf-8")

    problems = full_run.check(full_run.COMMITTED_SAVED_RUN, expected)

    assert len(problems) == 1
    assert problems[0].startswith("output differs: ")
    produced = next(line for line in original.splitlines() if '"vera_score": ' in line)
    edited = produced.replace('"vera_score": ', '"vera_score": 1', 1)
    assert f"-{edited}" in problems[0]
    assert f"+{produced}" in problems[0]


@pytest.mark.timeout(120)
def test_a_changed_input_fails_and_shows_what_changed(tmp_path: Path) -> None:
    saved = tmp_path / "saved"
    shutil.copytree(full_run.COMMITTED_SAVED_RUN, saved)
    prompt = saved / full_run.INPUTS / "chatbot_system_prompt.txt"
    produced = prompt.read_text(encoding="utf-8")
    prompt.write_text("You are a careful AI assistant.", encoding="utf-8")

    problems = full_run.check(saved, full_run.COMMITTED_EXPECTED)

    assert len(problems) == 1
    assert problems[0].startswith("input differs: chatbot system prompt\n")
    assert "-You are a careful AI assistant." in problems[0]
    assert f"+{produced}" in problems[0]
