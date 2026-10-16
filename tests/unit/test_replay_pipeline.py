"""Replay a recorded pipeline run and require identical outputs.

The models' answers come from tests/fixtures/replay/calls.jsonl, so the run is
deterministic: the same code must produce the same transcripts, results.csv,
and scores.json. A change to what the models are sent fails with
`NotInRecording`; a change to how answers become ratings or scores fails the
comparison. See docs/safe-changes.md.
"""

import importlib.util
from pathlib import Path

import pytest

from tests.mocks.replay_llm import replay_llm_factory

# The run's definition lives beside the recording so the recording script and
# this test can't drift apart. tests/fixtures/ is not a package: load by path.
REPLAY = Path(__file__).resolve().parents[1] / "fixtures/replay"
_spec = importlib.util.spec_from_file_location("record", REPLAY / "record.py")
assert _spec and _spec.loader
record = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(record)


@pytest.mark.timeout(120)
def test_replayed_run_matches_the_recording(tmp_path: Path):
    if not record.RECORDING.is_file():
        pytest.skip("No recording yet: run tests/fixtures/replay/record.py")

    create_llm = replay_llm_factory(record.RECORDING)
    try:
        run_folder = record.run_pipeline(tmp_path, create_llm)
    finally:
        # Report a changed prompt as the cause, not the failure it leads to.
        assert not create_llm.misses, create_llm.misses[0]
    produced = record.outputs(run_folder)

    expected = {
        str(path.relative_to(record.EXPECTED)): path.read_text(encoding="utf-8")
        for path in sorted(record.EXPECTED.rglob("*"))
        if path.is_file()
    }
    assert sorted(produced) == sorted(expected)
    for name in expected:
        assert produced[name] == expected[name], f"{name} differs from the recording"

    leftover = sum(len(replies) for replies in create_llm.unused_replies.values())
    assert leftover == 0, f"{leftover} recorded calls were never made"
