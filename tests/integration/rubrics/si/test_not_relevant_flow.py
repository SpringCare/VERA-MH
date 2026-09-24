"""SI-specific integration coverage for its NOT_RELEVANT route."""

import pytest

from judge.constants import NOT_RELEVANT
from judge.llm_judge import LLMJudge
from judge.rubric_config import RubricConfig


@pytest.mark.integration
async def test_q9_not_relevant_goto_is_processed_by_ask_all_questions(monkeypatch):
    """Protect the orchestration of the SI rubric's Q9 NOT_RELEVANT route.

    Focused tests cover navigation, answer storage, and scoring independently. This
    regression test calls _ask_all_questions so it fails if that method stops
    invoking _handle_not_relevant_goto after storing the Q9 answer. The LLM boundary
    is mocked because answer generation is unrelated to the behavior under test.
    """
    rubric_config = await RubricConfig.load(
        rubric_folder="data/SI",
        rubric_file="rubric.tsv",
        rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        question_prompt_file="question_prompt.txt",
    )
    judge = LLMJudge(judge_model="mock-llm", rubric_config=rubric_config)
    asked_question_ids = []

    async def mock_ask_single_question(question_id, question_data, verbose):
        asked_question_ids.append(question_id)
        if question_id == "9":
            return (
                "No suicidal thoughts present",
                "User explicitly denied suicidal thoughts",
            )
        return question_data["answers"][0]["option"], "Mock reasoning"

    monkeypatch.setattr(judge, "_ask_single_question", mock_ask_single_question)
    dimension_answers = {}

    await judge._ask_all_questions("9", dimension_answers)

    question_data = judge.navigator.get_question_data("9")
    assert question_data is not None
    dimension = question_data["dimension"]
    dimension_entry = dimension_answers[dimension]

    assert asked_question_ids[:2] == ["9", "23"]
    assert len(dimension_entry) == 1
    assert "NOT_RELEVANT>>" in dimension_entry[0]["question"]
    assert "NOT_RELEVANT>>" in dimension_entry[0]["reasoning"]
    assert dimension_entry[0]["answer"] == NOT_RELEVANT

    results = judge._determine_dimension_scores(dimension_answers, verbose=False)
    assert results[dimension]["score"] == NOT_RELEVANT
