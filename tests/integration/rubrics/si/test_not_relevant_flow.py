"""SI-specific integration coverage for its NOT_RELEVANT route."""

import pytest

from vera_mh.judge.rubric_config import RubricConfig
from vera_mh.judge.scoring import determine_dimension_scores
from vera_mh.utils.ratings import NOT_RELEVANT


@pytest.mark.integration
async def test_q9_not_relevant_goto_is_processed_by_rubric_walk():
    """Protect the orchestration of the SI rubric's Q9 NOT_RELEVANT route.

    Focused tests cover navigation, answer storage, and scoring independently. This
    regression test calls RubricConfig.walk so it fails if that method stops
    invoking mark_not_relevant after storing the Q9 answer. The LLM boundary
    is mocked because answer generation is unrelated to the behavior under test.
    """
    rubric_config = await RubricConfig.load(
        rubric_folder="data/SI",
        rubric_file="rubric.tsv",
        rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        question_prompt_file="question_prompt.txt",
    )
    asked_question_ids = []

    async def mock_ask_single_question(question_id, question_data):
        asked_question_ids.append(question_id)
        if question_id == "9":
            return (
                "No suicidal thoughts present",
                "User explicitly denied suicidal thoughts",
            )
        return question_data["answers"][0]["option"], "Mock reasoning"

    walk = await rubric_config.walk(mock_ask_single_question, "9")
    dimension_answers = walk.dimension_answers

    question_data = rubric_config.question_flow_data.get("9")
    assert question_data is not None
    dimension = question_data["dimension"]
    dimension_entry = dimension_answers[dimension]

    assert asked_question_ids[:2] == ["9", "23"]
    assert len(dimension_entry) == 1
    assert "NOT_RELEVANT>>" in dimension_entry[0]["question"]
    assert "NOT_RELEVANT>>" in dimension_entry[0]["reasoning"]
    assert dimension_entry[0]["answer"] == NOT_RELEVANT

    results = determine_dimension_scores(rubric_config, dimension_answers)
    assert results[dimension]["score"] == NOT_RELEVANT
