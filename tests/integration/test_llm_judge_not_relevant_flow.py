"""Integration tests for LLMJudge NOT_RELEVANT>> flow with real rubric data.

These tests verify the complete integration flow using actual rubric files
and production-like data, rather than mocked fixtures.
"""

import pytest

from judge.constants import NOT_RELEVANT
from judge.llm_judge import LLMJudge


@pytest.mark.integration
class TestNotRelevantGotoIntegrationAdvanced:
    """Test advanced NOT_RELEVANT>> handling scenarios with real rubric data.

    This tests the specific bug where _store_answer runs before
    _handle_not_relevant_goto, causing the dimension to be scored as
    "Best Practice" instead of "Not Relevant".
    """

    async def test_not_relevant_goto_overwrites_stored_answer(
        self, rubric_config_factory
    ):
        """Test that _handle_not_relevant_goto properly overwrites stored answers.

        This directly tests the fix: _handle_not_relevant_goto should overwrite
        the dimension_answers entry even if it already exists.

        Without the fix: dimension gets scored as "Best Practice"
        With the fix: dimension gets scored as "Not Relevant"
        """
        rubric_config = await rubric_config_factory(rubric_file="rubric_simple.tsv")
        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_config=rubric_config,
        )

        # Simulate the problematic state: dimension_answers already has an entry
        # (this is what _store_answer creates before _handle_not_relevant_goto runs)
        dimension_answers = {
            "Detects Risk": [
                {
                    "question_id": "5",
                    "question": 'Select "Rate this dimension Not Relevant".',
                    "answer": "Rate this dimension Not Relevant",
                    "severity": None,
                    "reasoning": "Special case",
                }
            ]
        }

        # This should overwrite the existing entry with NOT_RELEVANT marker
        judge._handle_not_relevant_goto(
            question_id="5",
            answer="Rate this dimension Not Relevant",
            current_dimension="Detects Risk",
            dimension_answers=dimension_answers,
            verbose=False,
        )

        # Verify the entry was overwritten with NOT_RELEVANT marker
        dimension_entry = dimension_answers["Detects Risk"]
        assert len(dimension_entry) == 1
        assert "NOT_RELEVANT>>" in dimension_entry[0]["question"]
        assert "NOT_RELEVANT>>" in dimension_entry[0]["reasoning"]
        assert dimension_entry[0]["answer"] == "Not Relevant"

        # Verify scoring recognizes the marker
        results = judge._determine_dimension_scores(dimension_answers, verbose=False)
        assert results["Detects Risk"]["score"] == NOT_RELEVANT
