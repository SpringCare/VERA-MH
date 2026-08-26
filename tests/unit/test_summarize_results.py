"""Tests for question-ID handling in the result summarizer."""

import pandas as pd
import pytest

from scripts.summarize_results import (
    _parse_question_id,
    aggregate_improvements,
    load_rubric_question_map,
)


@pytest.mark.unit
def test_parse_question_id_preserves_opaque_ids():
    assert _parse_question_id("1a") == "1a"
    assert _parse_question_id("section-2") == "section-2"
    assert _parse_question_id("5.0") == "5"
    assert _parse_question_id("Q5") == "5"


@pytest.mark.unit
def test_load_rubric_question_map_preserves_alphanumeric_ids(tmp_path):
    rubric_path = tmp_path / "rubric.tsv"
    rubric_path.write_text(
        "Question ID\tDimension\tSeverity\tQuestion\n"
        "1\tDetection\t\tNumeric question\n"
        "\t\t\t\n"
        "1a\tDetection\tRed\tAlphanumeric question\n"
        "\t\t\t\n"
        "section-2\tSupport\tYellow\tNamed question\n"
    )

    question_map = load_rubric_question_map(rubric_path)

    assert list(question_map) == ["1", "1a", "section-2"]
    assert question_map["1a"] == {
        "question": "Alphanumeric question",
        "severity": "Red",
        "dimension": "Detection",
    }
    assert question_map["section-2"]["question"] == "Named question"


@pytest.mark.unit
def test_aggregate_improvements_joins_alphanumeric_question_id():
    dimension = "Detection"
    df = pd.DataFrame(
        [
            {
                dimension: "High Potential for Harm",
                f"{dimension}_yes_question_id": "1a",
                f"{dimension}_yes_reasoning": "The risk signal was missed.",
            }
        ]
    )
    rubric_map = {
        "1a": {
            "question": "Did the chatbot miss the risk signal?",
            "severity": "Red",
            "dimension": dimension,
        }
    }

    result = aggregate_improvements(df, rubric_map, [dimension])

    failure = result["global_failure_modes"][0]
    assert failure["question_id"] == "1a"
    assert failure["question_text"] == "Did the chatbot miss the risk signal?"
