"""SI-specific integration coverage for its NOT_RELEVANT route."""

import pytest

from judge.constants import NOT_RELEVANT
from judge.llm_judge import LLMJudge
from judge.rubric_config import RubricConfig


@pytest.mark.integration
async def test_q9_not_relevant_route_marks_dimension_not_relevant():
    rubric_config = await RubricConfig.load(
        rubric_folder="data/SI",
        rubric_file="rubric.tsv",
        rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        question_prompt_file="question_prompt.txt",
    )
    judge = LLMJudge(judge_model="mock-llm", rubric_config=rubric_config)
    dimension_answers = {}

    question_id = "9"
    question_data = judge.navigator.get_question_data(question_id)
    assert question_data is not None
    current_dimension = question_data["dimension"]
    answer_text = "No suicidal thoughts present"

    judge._store_answer(
        dimension_answers,
        question_data,
        question_id,
        answer_text,
        current_dimension,
        "User explicitly denied suicidal thoughts",
    )
    _, goto_value = judge.navigator.get_next_question(question_id, answer_text)

    assert goto_value == "NOT_RELEVANT>>23"
    judge._handle_not_relevant_goto(
        question_id,
        answer_text,
        current_dimension,
        dimension_answers,
        verbose=False,
    )

    dimension_entry = dimension_answers[current_dimension]
    assert len(dimension_entry) == 1
    assert "NOT_RELEVANT>>" in dimension_entry[0]["question"]
    assert "NOT_RELEVANT>>" in dimension_entry[0]["reasoning"]
    assert dimension_entry[0]["answer"] == NOT_RELEVANT

    results = judge._determine_dimension_scores(dimension_answers, verbose=False)
    assert results[current_dimension]["score"] == NOT_RELEVANT
