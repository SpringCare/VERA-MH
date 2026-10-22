"""Replay a recorded pipeline run and require identical outputs.

The models' answers come from tests/fixtures/replay/calls.jsonl, so the run is
deterministic: the same code must produce the same transcripts, results.csv,
and scores.json. A change to what the models are sent fails with
`NotInRecording`; a change to how answers become ratings or scores fails the
comparison. See docs/safe-changes.md.
"""

from pathlib import Path

import pytest

from tests.safe_changes import replay


@pytest.mark.timeout(120)
def test_replayed_run_matches_the_recording(tmp_path: Path):
    if not replay.RECORDING.is_file():
        pytest.skip("No recording yet: run `python -m tests.safe_changes.replay`")

    replayer = replay.Replayer(replay.RECORDING)
    try:
        run_folder = replay.run_pipeline(tmp_path, replayer.create_llm)
    finally:
        # Report a changed prompt as the cause, not the failure it leads to.
        assert not replayer.misses, replayer.misses[0]

    produced = replay.outputs(run_folder)
    expected = replay.expected_outputs()
    assert sorted(produced) == sorted(expected)
    for name in expected:
        assert produced[name] == expected[name], f"{name} differs from the recording"

    assert replayer.unused_replies() == 0, "the code made fewer calls than recorded"
