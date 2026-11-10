"""Unit tests for walking a rubric (RubricConfig.walk) and its answer helpers."""

import pytest

from vera_mh.judge.rubric_config import (
    RubricConfig,
    mark_not_relevant,
    store_answer,
)
from vera_mh.judge.scoring import determine_dimension_scores


@pytest.mark.unit
class TestNotRelevantGotoIntegration:
    """Test NOT_RELEVANT>> handling integration."""

    async def test_auto_answer_skips_llm(self):
        rubric_config = RubricConfig(
            dimensions=["Test Dimension"],
            question_flow_data={
                "auto": {
                    "dimension": "Test Dimension",
                    "risk_type": "",
                    "question": "Choose the only answer.",
                    "examples": "",
                    "severity": None,
                    "auto_answer": True,
                    "answers": [{"option": "Only answer", "goto": "END"}],
                }
            },
            question_order=["auto"],
            rubric_prompt_beginning="{conversation}",
            question_prompt_template="{question}",
        )

        async def fail_if_called(*args, **kwargs):
            pytest.fail("The LLM must not be called for an auto-answer question")

        walk = await rubric_config.walk(fail_if_called, "auto")

        entry = walk.dimension_answers["Test Dimension"][0]
        assert entry["answer"] == "Only answer"
        assert entry["reasoning"] == "Automatically selected by rubric"

    async def test_answer_with_not_relevant_goto_scores_not_relevant(
        self, rubric_config_factory
    ):
        """Test that when answer's goto is NOT_RELEVANT>>, dimension gets Not Relevant.

        This tests the end-to-end flow where:
        1. A question is answered triggering NOT_RELEVANT>> goto
        2. The dimension should be scored as "Not Relevant"

        This is a focused test for the specific scenario where NOT_RELEVANT>> goto
        should result in Not Relevant scoring.
        """
        rubric_config = await rubric_config_factory(rubric_file="rubric_simple.tsv")

        # Start with empty dimension_answers
        dimension_answers = {}
        current_dimension = "Test Dimension"

        # Mock question data
        mock_question_data = {
            "question": "Test question that leads to a not-relevant goto",
            "dimension": current_dimension,
            "severity": "Red",
        }

        # Simulate an answer that would trigger NOT_RELEVANT>> goto
        answer_text = "Rate this dimension Not Relevant"
        question_id = "1"
        reasoning = "User chose to mark dimension as not relevant"

        # Store the answer first (normal flow)
        store_answer(
            dimension_answers,
            mock_question_data,
            question_id,
            answer_text,
            current_dimension,
            reasoning,
        )

        # Verify initial state - normal answer stored
        assert len(dimension_answers[current_dimension]) == 1
        initial_entry = dimension_answers[current_dimension][0]
        assert "NOT_RELEVANT>>" not in initial_entry["question"]
        assert initial_entry["answer"] == answer_text

        # Simulate the NOT_RELEVANT>> goto being processed
        mark_not_relevant(
            question_id,
            answer_text,
            current_dimension,
            dimension_answers,
        )

        # Verify the entry was overwritten with NOT_RELEVANT marker
        assert len(dimension_answers[current_dimension]) == 1
        not_relevant_entry = dimension_answers[current_dimension][0]
        assert "NOT_RELEVANT>>" in not_relevant_entry["question"]
        assert "NOT_RELEVANT>>" in not_relevant_entry["reasoning"]
        assert not_relevant_entry["answer"] == "Not Relevant"

        # Most important: Verify dimension gets scored as "Not Relevant"
        results = determine_dimension_scores(rubric_config, dimension_answers)
        assert results[current_dimension]["score"] == "Not Relevant"
        assert "NOT_RELEVANT>>" in results[current_dimension]["reasoning"]
        assert results[current_dimension]["yes_question_id"] == ""
        assert results[current_dimension]["yes_reasoning"] == ""
