"""Integration tests for the judge's NOT_RELEVANT>> flow with real rubric data.

These tests verify the complete integration flow using actual rubric files
and production-like data, rather than mocked fixtures.
"""

import pytest

from vera_mh.judge.rubric_config import mark_not_relevant
from vera_mh.judge.scoring import determine_dimension_scores
from vera_mh.utils.ratings import NOT_RELEVANT


@pytest.mark.integration
class TestNotRelevantGotoIntegrationAdvanced:
    """Test advanced NOT_RELEVANT>> handling scenarios with real rubric data.

    This tests the specific bug where store_answer runs before
    mark_not_relevant, causing the dimension to be scored as
    "Best Practice" instead of "Not Relevant".
    """

    async def test_not_relevant_goto_overwrites_stored_answer(
        self, rubric_config_factory
    ):
        """Test that mark_not_relevant properly overwrites stored answers.

        This directly tests the fix: mark_not_relevant should overwrite
        the dimension_answers entry even if it already exists.

        Without the fix: dimension gets scored as "Best Practice"
        With the fix: dimension gets scored as "Not Relevant"
        """
        rubric_config = await rubric_config_factory(rubric_file="rubric_simple.tsv")

        # Simulate the problematic state: dimension_answers already has an entry
        # (this is what store_answer creates before mark_not_relevant runs)
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
        mark_not_relevant(
            question_id="5",
            answer="Rate this dimension Not Relevant",
            current_dimension="Detects Risk",
            dimension_answers=dimension_answers,
        )

        # Verify the entry was overwritten with NOT_RELEVANT marker
        dimension_entry = dimension_answers["Detects Risk"]
        assert len(dimension_entry) == 1
        assert "NOT_RELEVANT>>" in dimension_entry[0]["question"]
        assert "NOT_RELEVANT>>" in dimension_entry[0]["reasoning"]
        assert dimension_entry[0]["answer"] == "Not Relevant"

        # Verify scoring recognizes the marker
        results = determine_dimension_scores(rubric_config, dimension_answers)
        assert results["Detects Risk"]["score"] == NOT_RELEVANT
