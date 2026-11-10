"""Unit tests for the question-flow rubric data a loaded RubricConfig gives LLMJudge.

Rubric parsing is implemented in RubricConfig (_parse_rubric); LLMJudge walks
the RubricConfig it is given. Scoring is tested in test_scoring.py, the rubric
walk in test_rubric_walk.py, and answer matching in
tests/unit/llm_clients/test_judge_answer.py.
"""

import pytest


@pytest.mark.unit
class TestParseQuestionFlowRubric:
    """Test rubric data as loaded by RubricConfig and used by LLMJudge.

    Verifies question_flow_data and question_order on the RubricConfig match
    expected structure (parsing is done in RubricConfig.load / _parse_rubric).
    """

    async def test_parse_single_row_question(self, rubric_config_factory):
        """Test parsing a simple single-row question with two answers."""
        rubric_config = await rubric_config_factory(rubric_file="rubric_single_row.tsv")

        # Verify the question was parsed
        assert "1" in rubric_config.question_flow_data
        question = rubric_config.question_flow_data["1"]

        # Check basic structure
        assert question["dimension"] == "Dimension A"
        assert question["severity"] == "Red"
        assert question["question"] == "Single row question?"
        assert question["examples"] == "Test example"

        # Check answers
        assert len(question["answers"]) == 2
        assert question["answers"][0]["option"] == "Yes"
        assert question["answers"][0]["goto"] == "2"
        assert question["answers"][1]["option"] == "No"
        assert question["answers"][1]["goto"] == "END"

    async def test_parse_multi_row_question(self, rubric_config_factory):
        """Test parsing a question that spans multiple rows with multiple options."""
        rubric_config = await rubric_config_factory(rubric_file="rubric_multi_row.tsv")

        # Verify multi-row question parsing
        assert "1" in rubric_config.question_flow_data
        question = rubric_config.question_flow_data["1"]

        # Check that all options from multiple rows are captured
        assert len(question["answers"]) == 3
        assert question["answers"][0]["option"] == "Option A"
        assert question["answers"][0]["goto"] == "2"
        assert question["answers"][1]["option"] == "Option B"
        assert question["answers"][1]["goto"] == "3"
        assert question["answers"][2]["option"] == "Option C"
        assert question["answers"][2]["goto"] == "END"

        # Dimension and severity should only be on the first row
        assert question["dimension"] == "Dimension A"
        assert question["severity"] == "Yellow"

    async def test_parse_goto_logic(self, rubric_config_factory):
        """Test parsing GOTO navigation logic including numeric and END values."""
        rubric_config = await rubric_config_factory(rubric_file="rubric_goto_logic.tsv")

        # Check question 1 with numeric GOTO
        q1 = rubric_config.question_flow_data["1"]
        assert q1["answers"][0]["goto"] == "3"
        assert q1["answers"][1]["goto"] == "5"

        # Check question 2 with END GOTO
        q2 = rubric_config.question_flow_data["2"]
        assert q2["answers"][0]["goto"] == "END"

        # Verify question order
        assert rubric_config.question_order == ["1", "2", "3", "5"]

    async def test_parse_goto_numeric_as_string(self, rubric_config_factory):
        """Test that numeric GOTO values are converted to strings correctly."""
        rubric_config = await rubric_config_factory(rubric_file="rubric_goto_logic.tsv")

        # Verify that numeric GOTOs are stored as strings (not floats)
        q1 = rubric_config.question_flow_data["1"]
        assert isinstance(q1["answers"][0]["goto"], str)
        assert q1["answers"][0]["goto"] == "3"
        assert not q1["answers"][0]["goto"].endswith(".0")

    async def test_parse_severity_values(self, rubric_config_factory):
        """Test parsing different severity values (Red, Yellow, Green, None)."""
        rubric_config = await rubric_config_factory(
            rubric_file="rubric_severity_values.tsv"
        )

        # Check Red severity
        assert rubric_config.question_flow_data["1"]["severity"] == "Red"

        # Check Yellow severity
        assert rubric_config.question_flow_data["2"]["severity"] == "Yellow"

        # Check Green severity
        assert rubric_config.question_flow_data["3"]["severity"] == "Green"

        # Check None/empty severity
        assert rubric_config.question_flow_data["4"]["severity"] is None

    async def test_parse_dimension_extraction(self, rubric_config_factory):
        """Test that dimension names are extracted correctly."""
        rubric_config = await rubric_config_factory(
            rubric_file="rubric_severity_values.tsv"
        )

        # Check different dimensions
        assert rubric_config.question_flow_data["1"]["dimension"] == "Dimension A"
        assert rubric_config.question_flow_data["2"]["dimension"] == "Dimension B"
        assert rubric_config.question_flow_data["3"]["dimension"] == "Dimension C"
        assert rubric_config.question_flow_data["4"]["dimension"] == "Dimension D"

    async def test_parse_repeated_dimension_rows(self, rubric_config_factory):
        """Test multiple primary rows that explicitly repeat a dimension."""
        rubric_config = await rubric_config_factory(
            rubric_file="rubric_empty_dimensions.tsv"
        )

        # Question 1 has dimension
        assert rubric_config.question_flow_data["1"]["dimension"] == "Dimension A"

        # Question 2 explicitly repeats the same dimension
        assert rubric_config.question_flow_data["2"]["dimension"] == "Dimension A"

        # Question 3 has different dimension
        assert rubric_config.question_flow_data["3"]["dimension"] == "Dimension B"

    async def test_parse_question_with_examples(self, rubric_config_factory):
        """Test parsing questions with and without examples."""
        rubric_config = await rubric_config_factory(
            rubric_file="rubric_with_examples.tsv"
        )

        # Question with examples
        q1 = rubric_config.question_flow_data["1"]
        assert "Example 1" in q1["examples"]
        assert "Example 2" in q1["examples"]

        # Question without examples
        q2 = rubric_config.question_flow_data["2"]
        assert q2["examples"] == ""

    async def test_parse_question_order(self, rubric_config_factory):
        """Test that question_order list maintains the correct sequence."""
        rubric_config = await rubric_config_factory(rubric_file="rubric_goto_logic.tsv")

        # Verify order matches appearance in file
        assert rubric_config.question_order == ["1", "2", "3", "5"]

        # Verify all questions in order exist in data
        for qid in rubric_config.question_order:
            assert qid in rubric_config.question_flow_data

    async def test_parse_questions_without_explicit_answers(
        self, rubric_config_factory
    ):
        """Test that questions without explicit answers get default Yes/No."""
        rubric_config = await rubric_config_factory(rubric_file="rubric_no_answers.tsv")

        # Both questions should have default Yes/No answers
        q1 = rubric_config.question_flow_data["1"]
        assert len(q1["answers"]) == 2
        assert q1["answers"][0]["option"] == "Yes"
        assert q1["answers"][0]["goto"] is None
        assert q1["answers"][1]["option"] == "No"
        assert q1["answers"][1]["goto"] is None

        q2 = rubric_config.question_flow_data["2"]
        assert len(q2["answers"]) == 2
        assert q2["answers"][0]["option"] == "Yes"
        assert q2["answers"][1]["option"] == "No"

    async def test_parse_risk_type_field(self, rubric_config_factory):
        """Test that risk_type field is parsed correctly."""
        rubric_config = await rubric_config_factory(rubric_file="rubric_single_row.tsv")

        # Risk type should be empty string if not specified
        q1 = rubric_config.question_flow_data["1"]
        assert "risk_type" in q1
        assert q1["risk_type"] == ""

    async def test_parse_question_id_as_string(self, rubric_config_factory):
        """Test that question IDs are stored as strings, not floats."""
        rubric_config = await rubric_config_factory(rubric_file="rubric_goto_logic.tsv")

        # All question IDs should be strings
        for qid in rubric_config.question_order:
            assert isinstance(qid, str)
            assert not qid.endswith(".0")

        # Check specific IDs
        assert "1" in rubric_config.question_flow_data
        assert "3" in rubric_config.question_flow_data
        assert "5" in rubric_config.question_flow_data

    async def test_parse_multiple_questions_same_dimension(self, rubric_config_factory):
        """Test parsing multiple questions within the same dimension."""
        rubric_config = await rubric_config_factory(
            rubric_file="rubric_empty_dimensions.tsv"
        )

        # Questions 1 and 2 explicitly declare the same dimension
        assert rubric_config.question_flow_data["1"]["dimension"] == "Dimension A"
        assert rubric_config.question_flow_data["2"]["dimension"] == "Dimension A"

        # Question 3 starts new dimension
        assert rubric_config.question_flow_data["3"]["dimension"] == "Dimension B"

    async def test_parse_answer_structure(self, rubric_config_factory):
        """Test that answer dictionaries have correct structure."""
        rubric_config = await rubric_config_factory(rubric_file="rubric_multi_row.tsv")

        q1 = rubric_config.question_flow_data["1"]

        # Each answer should have 'option' and 'goto' keys
        for answer in q1["answers"]:
            assert "option" in answer
            assert "goto" in answer
            assert isinstance(answer["option"], str)
            # goto can be str or None

    async def test_parse_complete_question_structure(self, rubric_config_factory):
        """Test that parsed questions have all expected fields."""
        rubric_config = await rubric_config_factory(rubric_file="rubric_single_row.tsv")

        q1 = rubric_config.question_flow_data["1"]

        # Check all expected keys exist
        expected_keys = [
            "dimension",
            "risk_type",
            "question",
            "examples",
            "severity",
            "answers",
        ]
        for key in expected_keys:
            assert key in q1, f"Missing key: {key}"

        # Check types
        assert isinstance(q1["dimension"], str)
        assert isinstance(q1["risk_type"], str)
        assert isinstance(q1["question"], str)
        assert isinstance(q1["examples"], str)
        assert isinstance(q1["answers"], list)
        # severity can be str or None

    async def test_parse_nan_handling(self, rubric_config_factory):
        """Test NaN values handled correctly (empty strings or None)."""
        rubric_config = await rubric_config_factory(rubric_file="rubric_no_answers.tsv")

        # Questions with no Answer/GOTO columns should not have "nan" strings
        q1 = rubric_config.question_flow_data["1"]

        # Check that severity is valid (not "nan" string)
        assert q1["severity"] != "nan"
        # In this fixture, q1 actually has Red severity
        assert q1["severity"] == "Red"

        # Check that default answers don't have "nan" GOTOs
        for answer in q1["answers"]:
            assert answer["goto"] is None
            assert answer["option"] != "nan"
