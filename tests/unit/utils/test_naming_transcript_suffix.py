"""Unit tests for transcript run suffix parsing."""

import re

from utils.naming import TRANSCRIPT_RUN_SUFFIX_RE


def _match(suffix: str) -> re.Match[str] | None:
    return TRANSCRIPT_RUN_SUFFIX_RE.search(suffix)


class TestTranscriptRunSuffixRe:
    def test_single_session_suffix(self) -> None:
        m = _match("Persona1_mock-model_run1.txt")
        assert m is not None
        assert m.group("run") == "1"
        assert m.group("session_index") is None
        assert m.group("session_type") is None

    def test_multi_session_suffix(self) -> None:
        m = _match("Persona1_mock-model_run1_s1_intake.txt")
        assert m is not None
        assert m.group("run") == "1"
        assert m.group("session_index") == "1"
        assert m.group("session_type") == "intake"

    def test_multi_session_with_workflow_in_stem(self) -> None:
        m = _match("Persona1_mock-model_workflow_run1_s2_coaching.txt")
        assert m is not None
        assert m.group("run") == "1"
        assert m.group("session_index") == "2"
        assert m.group("session_type") == "coaching"
