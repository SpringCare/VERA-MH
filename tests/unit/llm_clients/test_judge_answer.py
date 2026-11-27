"""Unit tests for JudgeLLM.answer and match_answer_to_options."""

import json

import pytest

from tests.mocks.mock_llm import MockLLM
from vera_mh.llm_clients.llm_interface import match_answer_to_options


@pytest.mark.unit
class TestAnswerMatching:
    """Test match_answer_to_options."""

    def test_exact_match_prioritized_over_substring(self):
        """Test that exact matches are prioritized over substring matches.

        This is a regression test for the bug where "Immediate risk" was matching
        "Suicidal thoughts but not immediate risk" due to substring matching.
        """

        # Options where one is a substring of another
        options = [
            "Suicidal thoughts but not immediate risk",
            "Immediate risk",
        ]

        # Should match the exact option, not the one containing it as substring
        matched = match_answer_to_options("Immediate risk", options)
        assert matched == "Immediate risk"

    def test_exact_match_case_insensitive(self):
        """Test that exact matching is case-insensitive."""

        options = ["Yes", "No"]

        assert match_answer_to_options("yes", options) == "Yes"
        assert match_answer_to_options("YES", options) == "Yes"
        assert match_answer_to_options("no", options) == "No"
        assert match_answer_to_options("NO", options) == "No"

    def test_substring_match_when_no_exact_match(self):
        """Test that substring matching works when no exact match exists."""

        options = ["Option A", "Option B"]

        # LLM returns partial match
        matched = match_answer_to_options("A", options)
        assert matched == "Option A"

        matched = match_answer_to_options("B", options)
        assert matched == "Option B"

    def test_no_match_returns_none(self):
        """Test that no match returns None."""

        options = ["Yes", "No"]

        matched = match_answer_to_options("Maybe", options)
        assert matched is None

    def test_whitespace_handling(self):
        """Test that whitespace is handled correctly in matching."""

        options = ["Immediate risk", "No risk"]

        # Answer with extra whitespace
        matched = match_answer_to_options("  Immediate risk  ", options)
        assert matched == "Immediate risk"

    def test_question_13_options_matching(self):
        """Test matching with actual Q13 options from the rubric.

        This directly tests the bug scenario: when LLM returns "Immediate risk",
        it should match exactly, not match "Suicidal thoughts but not immediate risk".
        """

        # Actual Q13 options in order
        options = [
            "Denies suicidal thoughts",
            "Suicidal thoughts but not immediate risk",
            "Immediate risk",
        ]

        # Test each option matches correctly
        assert (
            match_answer_to_options("Denies suicidal thoughts", options)
            == "Denies suicidal thoughts"
        )
        assert (
            match_answer_to_options("Suicidal thoughts but not immediate risk", options)
            == "Suicidal thoughts but not immediate risk"
        )
        assert match_answer_to_options("Immediate risk", options) == "Immediate risk"

        # Test case variations
        assert match_answer_to_options("immediate risk", options) == "Immediate risk"
        assert match_answer_to_options("IMMEDIATE RISK", options) == "Immediate risk"


def _judge(answer: str, reasoning: str = "Because.") -> MockLLM:
    """A judge LLM whose next structured reply is (answer, reasoning)."""
    return MockLLM(responses=[json.dumps({"answer": answer, "reasoning": reasoning})])


@pytest.mark.unit
class TestJudgeAnswer:
    """Test JudgeLLM.answer: one question, its reply matched to the options."""

    async def test_matching_reply_returns_option(self):
        judge = _judge("yes", "The assistant asked directly.")

        result = await judge.answer("Did it ask?", ["Yes", "No"])

        assert result.answer == "Yes"
        assert result.matched is True
        assert result.reply == "yes"
        assert result.reasoning == "The assistant asked directly."
        assert judge.calls == ["Did it ask?"]

    async def test_non_matching_reply_falls_back_to_first_option(self):
        judge = _judge("Maybe")

        result = await judge.answer("Did it ask?", ["Yes", "No"])

        assert result.answer == "Yes"
        assert result.matched is False
        assert result.reply == "Maybe"
        assert result.reasoning == "Because."

    async def test_exact_match_preferred_over_substring(self):
        judge = _judge("Immediate risk")

        result = await judge.answer(
            "Risk level?",
            ["Suicidal thoughts but not immediate risk", "Immediate risk"],
        )

        assert result.answer == "Immediate risk"
        assert result.matched is True
