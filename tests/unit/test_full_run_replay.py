"""Replay a slice of a saved full run and require identical outputs.

The baseline in tests/fixtures/full_run/ was seeded from a saved run of the
recommended SI profile (`tests/safe_changes/full_run.py`): every user, chatbot,
and judge answer comes from that run, so the replay is deterministic. A code
change that alters what a model is sent fails on the call; one that alters a
transcript, judge TSV, results.csv, or score fails the comparison.
"""

import shutil
from pathlib import Path

import pytest

from tests.safe_changes import full_run


@pytest.mark.timeout(120)
def test_full_run_slice_matches_the_baseline() -> None:
    baseline: Path = full_run.COMMITTED_BASELINE
    if not (baseline / "calls.jsonl").is_file():
        pytest.skip("No baseline yet: see tests/safe_changes/full_run.py")

    problems = full_run.check(baseline)
    assert not problems, "\n\n".join(problems)


@pytest.mark.timeout(120)
def test_hidden_files_in_a_baseline_are_ignored(tmp_path: Path) -> None:
    """macOS Finder drops a binary `.DS_Store` into every folder it opens."""
    baseline = full_run.COMMITTED_BASELINE
    if not (baseline / "calls.jsonl").is_file():
        pytest.skip("No baseline yet: see tests/safe_changes/full_run.py")
    copy = tmp_path / "baseline"
    shutil.copytree(baseline, copy)
    (copy / "expected" / ".DS_Store").write_bytes(b"\x00\x00\x00\x01Bud1\x80")

    problems = full_run.check(copy)
    assert not problems, "\n\n".join(problems)


@pytest.mark.timeout(120)
def test_a_changed_output_fails_and_shows_what_changed(tmp_path: Path) -> None:
    baseline = full_run.COMMITTED_BASELINE
    if not (baseline / "calls.jsonl").is_file():
        pytest.skip("No baseline yet: see tests/safe_changes/full_run.py")
    copy = tmp_path / "baseline"
    shutil.copytree(baseline, copy)
    transcript = next((copy / "expected").glob("p_*/conversations/*.txt"))
    original = transcript.read_text(encoding="utf-8")
    transcript.write_text(original.replace("user: ", "user: not ", 1), encoding="utf-8")

    problems = full_run.check(copy)

    assert len(problems) == 1
    # Long lines are cut in the diff, so match on how each line starts.
    first_turn = original.splitlines()[0]
    edited = first_turn.replace("user: ", "user: not ", 1)
    assert f"-{edited[:100]}" in problems[0]
    assert f"+{first_turn[:100]}" in problems[0]
