"""Replay only the generation step of the recorded run, and require the same
conversations.

The user and chatbot answers come from tests/fixtures/replay/calls.jsonl, so
generation is deterministic: the same code must write the same transcripts.
This pins turn order, who speaks first, the history each model is sent, and
when a conversation ends, with a failure that names generation rather than
the whole pipeline. See docs/safe-changes.md.
"""

from pathlib import Path

import pytest

from tests.safe_changes import replay


@pytest.mark.timeout(60)
def test_replayed_generation_writes_the_recorded_conversations(tmp_path: Path):
    if not replay.RECORDING.is_file():
        pytest.skip("No recording yet: run `python -m tests.safe_changes.replay`")

    replayer = replay.Replayer(replay.RECORDING)
    try:
        run_folder = replay.run_generate(tmp_path, replayer.create_llm)
    finally:
        # Report a changed prompt as the cause, not the failure it leads to.
        assert not replayer.misses, replayer.misses[0]

    expected = {
        name: text
        for name, text in replay.expected_outputs().items()
        if name.startswith("transcripts/")
    }
    assert replay.transcripts(run_folder) == expected
    assert replayer.unused_replies(kind="text") == 0, (
        "generation made fewer calls than recorded"
    )
