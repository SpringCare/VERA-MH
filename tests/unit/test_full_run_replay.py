"""Replay a slice of a saved full run and require identical outputs.

The baseline in tests/fixtures/full_run/ was seeded from a saved run of the
recommended SI profile (`tests/safe_changes/full_run.py`): every user, chatbot,
and judge answer comes from that run, so the replay is deterministic. A code
change that alters what a model is sent fails on the call; one that alters a
transcript, judge TSV, results.csv, or score fails the comparison.
"""

from pathlib import Path

import pytest

from tests.safe_changes import full_run


@pytest.mark.timeout(120)
def test_full_run_slice_matches_the_baseline() -> None:
    baseline: Path = full_run.COMMITTED_BASELINE
    if not (baseline / "calls.jsonl").is_file():
        pytest.skip("No baseline yet: see tests/safe_changes/full_run.py")

    assert full_run.check(baseline) == []
