"""Integration tests for LLMJudge NOT_RELEVANT>> flow with real rubric data.

These tests verify the complete integration flow using actual rubric files
and production-like data, rather than mocked fixtures.
"""

import pytest

from judge.constants import NOT_RELEVANT
from judge.llm_judge import LLMJudge
from judge.rubric_config import RubricConfig


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

    async def test_question_13_full_flow_integration(self):
        """Test that Q13 with NOT_RELEVANT>>25 properly marks dimension as Not Relevant.

        This tests the full integration flow:
        1. Q13 is answered with "Denies suicidal thoughts"
        2. Q13's goto is "NOT_RELEVANT>>25" which should trigger
           _handle_not_relevant_goto
        3. Dimension should be scored as "Not Relevant", not "Best Practice"

        This is a regression test for the bug where _store_answer runs before
        _handle_not_relevant_goto, preventing the marker from being set.
        """
        # Load main rubric.tsv directly from data/ directory
        rubric_config = await RubricConfig.load(
            rubric_folder="data",
            rubric_file="rubric.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
            question_prompt_file="question_prompt.txt",
        )
        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_config=rubric_config,
        )

        # Simulate the full _ask_all_questions flow starting from Q13
        dimension_answers = {}

        # Manually simulate what _ask_all_questions does for Q13:
        # 1. Get question data
        question_data = judge.navigator.get_question_data("13")
        current_dimension = "Guides to Human Support"  # Q13's actual dimension
        answer_text = "Denies suicidal thoughts"  # Q13's NOT_RELEVANT answer option
        reasoning = "User explicitly denied suicidal thoughts"

        # 2. Store answer (this happens first in _ask_all_questions)
        judge._store_answer(
            dimension_answers,
            question_data,
            "13",
            answer_text,
            current_dimension,
            reasoning,
        )

        # 3. Get next question and goto value
        next_question_id, goto_value = judge.navigator.get_next_question(
            "13", answer_text
        )

        # 4. Handle the NOT_RELEVANT>> goto (this should overwrite the stored answer)
        if goto_value and goto_value.startswith("NOT_RELEVANT>>"):
            judge._handle_not_relevant_goto(
                "13", answer_text, current_dimension, dimension_answers, verbose=False
            )

        # 5. Verify the entry was overwritten with NOT_RELEVANT marker
        assert current_dimension in dimension_answers
        dimension_entry = dimension_answers[current_dimension]
        assert len(dimension_entry) == 1
        assert "NOT_RELEVANT>>" in dimension_entry[0]["question"]
        assert "NOT_RELEVANT>>" in dimension_entry[0]["reasoning"]
        assert dimension_entry[0]["answer"] == "Not Relevant"

        # 6. Verify scoring recognizes the marker
        results = judge._determine_dimension_scores(dimension_answers, verbose=False)
        assert results[current_dimension]["score"] == NOT_RELEVANT

        # Verify that Q13 indeed has NOT_RELEVANT>>25 goto
        assert (
            goto_value == "NOT_RELEVANT>>25"
        ), f"Expected NOT_RELEVANT>>25, got {goto_value}"
