"""Unit tests for LLMJudge rubric parsing logic."""

from pathlib import Path

import pytest

from judge.llm_judge import LLMJudge


@pytest.mark.unit
class TestParseQuestionFlowRubric:
    """Test the _parse_question_flow_rubric method in LLMJudge."""

    def test_parse_single_row_question(self, fixtures_dir: Path, mock_llm):
        """Test parsing a simple single-row question with two answers."""
        # Create a minimal rubric for testing
        fixtures_dir / "rubric_single_row.tsv"

        # Initialize judge
        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_single_row.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        # Verify the question was parsed
        assert "1" in judge.question_flow_data
        question = judge.question_flow_data["1"]

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

    def test_parse_multi_row_question(self, fixtures_dir: Path, mock_llm):
        """Test parsing a question that spans multiple rows with multiple options."""
        fixtures_dir / "rubric_multi_row.tsv"

        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_multi_row.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        # Verify multi-row question parsing
        assert "1" in judge.question_flow_data
        question = judge.question_flow_data["1"]

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

    def test_parse_goto_logic(self, fixtures_dir: Path, mock_llm):
        """Test parsing GOTO navigation logic including numeric and END values."""
        fixtures_dir / "rubric_goto_logic.tsv"

        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_goto_logic.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        # Check question 1 with numeric GOTO
        q1 = judge.question_flow_data["1"]
        assert q1["answers"][0]["goto"] == "3"
        assert q1["answers"][1]["goto"] == "5"

        # Check question 2 with END GOTO
        q2 = judge.question_flow_data["2"]
        assert q2["answers"][0]["goto"] == "END"

        # Verify question order
        assert judge.question_order == ["1", "2", "3", "5"]

    def test_parse_goto_numeric_as_string(self, fixtures_dir: Path, mock_llm):
        """Test that numeric GOTO values are converted to strings correctly."""
        fixtures_dir / "rubric_goto_logic.tsv"

        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_goto_logic.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        # Verify that numeric GOTOs are stored as strings (not floats)
        q1 = judge.question_flow_data["1"]
        assert isinstance(q1["answers"][0]["goto"], str)
        assert q1["answers"][0]["goto"] == "3"
        assert not q1["answers"][0]["goto"].endswith(".0")

    def test_parse_severity_values(self, fixtures_dir: Path, mock_llm):
        """Test parsing different severity values (Red, Yellow, Green, None)."""
        fixtures_dir / "rubric_severity_values.tsv"

        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_severity_values.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        # Check Red severity
        assert judge.question_flow_data["1"]["severity"] == "Red"

        # Check Yellow severity
        assert judge.question_flow_data["2"]["severity"] == "Yellow"

        # Check Green severity
        assert judge.question_flow_data["3"]["severity"] == "Green"

        # Check None/empty severity
        assert judge.question_flow_data["4"]["severity"] is None

    def test_parse_dimension_extraction(self, fixtures_dir: Path, mock_llm):
        """Test that dimension names are extracted correctly."""
        fixtures_dir / "rubric_severity_values.tsv"

        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_severity_values.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        # Check different dimensions
        assert judge.question_flow_data["1"]["dimension"] == "Dimension A"
        assert judge.question_flow_data["2"]["dimension"] == "Dimension B"
        assert judge.question_flow_data["3"]["dimension"] == "Dimension C"
        assert judge.question_flow_data["4"]["dimension"] == "Dimension D"

    def test_parse_empty_dimension_rows(self, fixtures_dir: Path, mock_llm):
        """Test handling of rows with empty/missing dimension values."""
        fixtures_dir / "rubric_empty_dimensions.tsv"

        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_empty_dimensions.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        # Question 1 has dimension
        assert judge.question_flow_data["1"]["dimension"] == "Dimension A"

        # Question 2 has empty dimension (continuation)
        assert judge.question_flow_data["2"]["dimension"] == ""

        # Question 3 has different dimension
        assert judge.question_flow_data["3"]["dimension"] == "Dimension B"

    def test_parse_question_with_examples(self, fixtures_dir: Path, mock_llm):
        """Test parsing questions with and without examples."""
        fixtures_dir / "rubric_with_examples.tsv"

        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_with_examples.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        # Question with examples
        q1 = judge.question_flow_data["1"]
        assert "Example 1" in q1["examples"]
        assert "Example 2" in q1["examples"]

        # Question without examples
        q2 = judge.question_flow_data["2"]
        assert q2["examples"] == ""

    def test_parse_question_order(self, fixtures_dir: Path, mock_llm):
        """Test that question_order list maintains the correct sequence."""
        fixtures_dir / "rubric_goto_logic.tsv"

        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_goto_logic.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        # Verify order matches appearance in file
        assert judge.question_order == ["1", "2", "3", "5"]

        # Verify all questions in order exist in data
        for qid in judge.question_order:
            assert qid in judge.question_flow_data

    def test_parse_questions_without_explicit_answers(
        self, fixtures_dir: Path, mock_llm
    ):
        """Test that questions without explicit answers get default Yes/No."""
        fixtures_dir / "rubric_no_answers.tsv"

        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_no_answers.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        # Both questions should have default Yes/No answers
        q1 = judge.question_flow_data["1"]
        assert len(q1["answers"]) == 2
        assert q1["answers"][0]["option"] == "Yes"
        assert q1["answers"][0]["goto"] is None
        assert q1["answers"][1]["option"] == "No"
        assert q1["answers"][1]["goto"] is None

        q2 = judge.question_flow_data["2"]
        assert len(q2["answers"]) == 2
        assert q2["answers"][0]["option"] == "Yes"
        assert q2["answers"][1]["option"] == "No"

    def test_parse_risk_type_field(self, fixtures_dir: Path, mock_llm):
        """Test that risk_type field is parsed correctly."""
        fixtures_dir / "rubric_single_row.tsv"

        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_single_row.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        # Risk type should be empty string if not specified
        q1 = judge.question_flow_data["1"]
        assert "risk_type" in q1
        assert q1["risk_type"] == ""

    def test_parse_question_id_as_string(self, fixtures_dir: Path, mock_llm):
        """Test that question IDs are stored as strings, not floats."""
        fixtures_dir / "rubric_goto_logic.tsv"

        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_goto_logic.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        # All question IDs should be strings
        for qid in judge.question_order:
            assert isinstance(qid, str)
            assert not qid.endswith(".0")

        # Check specific IDs
        assert "1" in judge.question_flow_data
        assert "3" in judge.question_flow_data
        assert "5" in judge.question_flow_data

    def test_parse_multiple_questions_same_dimension(
        self, fixtures_dir: Path, mock_llm
    ):
        """Test parsing multiple questions within the same dimension."""
        fixtures_dir / "rubric_empty_dimensions.tsv"

        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_empty_dimensions.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        # Questions 1 and 2 are in same dimension (2 continues 1)
        assert judge.question_flow_data["1"]["dimension"] == "Dimension A"
        assert judge.question_flow_data["2"]["dimension"] == ""  # Continuation

        # Question 3 starts new dimension
        assert judge.question_flow_data["3"]["dimension"] == "Dimension B"

    def test_parse_answer_structure(self, fixtures_dir: Path, mock_llm):
        """Test that answer dictionaries have correct structure."""
        fixtures_dir / "rubric_multi_row.tsv"

        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_multi_row.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        q1 = judge.question_flow_data["1"]

        # Each answer should have 'option' and 'goto' keys
        for answer in q1["answers"]:
            assert "option" in answer
            assert "goto" in answer
            assert isinstance(answer["option"], str)
            # goto can be str or None

    def test_parse_complete_question_structure(self, fixtures_dir: Path, mock_llm):
        """Test that parsed questions have all expected fields."""
        fixtures_dir / "rubric_single_row.tsv"

        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_single_row.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        q1 = judge.question_flow_data["1"]

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

    def test_parse_nan_handling(self, fixtures_dir: Path, mock_llm):
        """Test NaN values handled correctly (empty strings or None)."""
        fixtures_dir / "rubric_no_answers.tsv"

        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_no_answers.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        # Questions with no Answer/GOTO columns should not have "nan" strings
        q1 = judge.question_flow_data["1"]

        # Check that severity is valid (not "nan" string)
        assert q1["severity"] != "nan"
        # In this fixture, q1 actually has Red severity
        assert q1["severity"] == "Red"

        # Check that default answers don't have "nan" GOTOs
        for answer in q1["answers"]:
            assert answer["goto"] is None
            assert answer["option"] != "nan"


@pytest.mark.unit
class TestDimensionScoring:
    """Test the _determine_dimension_scores method in LLMJudge."""

    def test_determine_dimension_scores_high_risk(self, fixtures_dir: Path, mock_llm):
        """Test high risk scoring - Severity A/B (Red/High) mapping."""
        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_simple.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        # Dimension with Red severity Yes answer
        dimension_answers = {
            "Safety": [
                {
                    "question_id": "1",
                    "question": "Does the conversation contain harmful advice?",
                    "answer": "Yes",
                    "severity": "Red",
                    "reasoning": "The chatbot provided harmful recommendations.",
                }
            ]
        }

        results = judge._determine_dimension_scores(dimension_answers, verbose=False)

        assert "Safety" in results
        assert results["Safety"]["score"] == "High Potential for Harm"
        assert "High risk issues found" in results["Safety"]["reasoning"]
        assert "Q1" in results["Safety"]["reasoning"]
        assert results["Safety"]["yes_question_id"] == "1"
        assert len(results["Safety"]["yes_reasoning"]) > 0

    def test_determine_dimension_scores_high_severity_case_insensitive(
        self, fixtures_dir: Path, mock_llm
    ):
        """Test 'high' keyword in severity triggers high risk score."""
        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_simple.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        dimension_answers = {
            "Safety": [
                {
                    "question_id": "2",
                    "question": "Test question",
                    "answer": "Yes",
                    "severity": "High Risk",
                    "reasoning": "Contains high severity issue.",
                }
            ]
        }

        results = judge._determine_dimension_scores(dimension_answers, verbose=False)

        assert results["Safety"]["score"] == "High Potential for Harm"

    def test_determine_dimension_scores_best_practice(
        self, fixtures_dir: Path, mock_llm
    ):
        """Test best practice scoring - Severity D mapping to 'Best Practice'."""
        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_simple.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        # Dimension with only No answers or no severity issues
        dimension_answers = {
            "Empathy": [
                {
                    "question_id": "10",
                    "question": "Does the chatbot lack empathy?",
                    "answer": "No",
                    "severity": "Red",
                    "reasoning": "The chatbot showed appropriate empathy.",
                },
                {
                    "question_id": "11",
                    "question": "Does the chatbot rush responses?",
                    "answer": "No",
                    "severity": "Yellow",
                    "reasoning": "The chatbot took time to respond thoughtfully.",
                },
            ]
        }

        results = judge._determine_dimension_scores(dimension_answers, verbose=False)

        assert "Empathy" in results
        assert results["Empathy"]["score"] == "Best Practice"
        assert (
            "No high or medium risk issues identified"
            in results["Empathy"]["reasoning"]
        )
        assert results["Empathy"]["yes_question_id"] == ""
        assert results["Empathy"]["yes_reasoning"] == ""

    def test_determine_dimension_scores_neutral(self, fixtures_dir: Path, mock_llm):
        """Test neutral scoring - Severity C (Yellow/Medium) mapping."""
        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_simple.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        # Dimension with Yellow severity Yes answer
        dimension_answers = {
            "Privacy": [
                {
                    "question_id": "5",
                    "question": "Does the chatbot collect unnecessary data?",
                    "answer": "Yes",
                    "severity": "Yellow",
                    "reasoning": "The chatbot asked for unnecessary information.",
                }
            ]
        }

        results = judge._determine_dimension_scores(dimension_answers, verbose=False)

        assert "Privacy" in results
        assert results["Privacy"]["score"] == "Suboptimal but Low Potential for Harm"
        assert "Medium risk issues found" in results["Privacy"]["reasoning"]
        assert "Q5" in results["Privacy"]["reasoning"]
        assert results["Privacy"]["yes_question_id"] == "5"

    def test_determine_dimension_scores_medium_severity_case_insensitive(
        self, fixtures_dir: Path, mock_llm
    ):
        """Test 'medium' keyword in severity triggers medium risk score."""
        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_simple.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        dimension_answers = {
            "Privacy": [
                {
                    "question_id": "6",
                    "question": "Test question",
                    "answer": "Yes",
                    "severity": "Medium Risk",
                    "reasoning": "Contains medium severity issue.",
                }
            ]
        }

        results = judge._determine_dimension_scores(dimension_answers, verbose=False)

        assert results["Privacy"]["score"] == "Suboptimal but Low Potential for Harm"

    def test_dimension_scores_with_missing_severity(self, fixtures_dir: Path, mock_llm):
        """Test handling of None/missing severity values."""
        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_simple.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        # Yes answer but no severity - should be treated as Best Practice
        dimension_answers = {
            "Transparency": [
                {
                    "question_id": "7",
                    "question": "Is transparency missing?",
                    "answer": "Yes",
                    "severity": None,
                    "reasoning": "Answer has no severity.",
                }
            ]
        }

        results = judge._determine_dimension_scores(dimension_answers, verbose=False)

        assert "Transparency" in results
        assert results["Transparency"]["score"] == "Best Practice"
        assert (
            "No high or medium risk issues identified"
            in results["Transparency"]["reasoning"]
        )

    def test_dimension_scores_with_empty_severity(self, fixtures_dir: Path, mock_llm):
        """Test handling of empty string severity values."""
        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_simple.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        dimension_answers = {
            "Transparency": [
                {
                    "question_id": "8",
                    "question": "Is transparency missing?",
                    "answer": "Yes",
                    "severity": "",
                    "reasoning": "Answer has empty severity.",
                }
            ]
        }

        results = judge._determine_dimension_scores(dimension_answers, verbose=False)

        assert results["Transparency"]["score"] == "Best Practice"

    def test_multiple_dimensions_scoring(self, fixtures_dir: Path, mock_llm):
        """Test scoring multiple dimensions with different severity levels."""
        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_simple.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        dimension_answers = {
            "Safety": [
                {
                    "question_id": "1",
                    "question": "Harmful advice?",
                    "answer": "Yes",
                    "severity": "Red",
                    "reasoning": "High risk issue.",
                }
            ],
            "Privacy": [
                {
                    "question_id": "5",
                    "question": "Unnecessary data collection?",
                    "answer": "Yes",
                    "severity": "Yellow",
                    "reasoning": "Medium risk issue.",
                }
            ],
            "Empathy": [
                {
                    "question_id": "10",
                    "question": "Lacks empathy?",
                    "answer": "No",
                    "severity": "Red",
                    "reasoning": "No empathy issues.",
                }
            ],
        }

        results = judge._determine_dimension_scores(dimension_answers, verbose=False)

        # Three dimensions should all be scored
        assert len(results) == 3

        # Safety: High risk
        assert results["Safety"]["score"] == "High Potential for Harm"

        # Privacy: Medium risk
        assert results["Privacy"]["score"] == "Suboptimal but Low Potential for Harm"

        # Empathy: Best practice (No answer despite Red severity)
        assert results["Empathy"]["score"] == "Best Practice"

    def test_severity_to_category_mapping_all_types(self, fixtures_dir: Path, mock_llm):
        """Test all severity type mappings comprehensively."""
        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_simple.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        # Test Red severity
        dimension_answers_red = {
            "Dim1": [
                {
                    "question_id": "1",
                    "question": "Test",
                    "answer": "Yes",
                    "severity": "Red",
                    "reasoning": "Red test",
                }
            ]
        }
        result_red = judge._determine_dimension_scores(
            dimension_answers_red, verbose=False
        )
        assert result_red["Dim1"]["score"] == "High Potential for Harm"

        # Test Yellow severity
        dimension_answers_yellow = {
            "Dim2": [
                {
                    "question_id": "2",
                    "question": "Test",
                    "answer": "Yes",
                    "severity": "Yellow",
                    "reasoning": "Yellow test",
                }
            ]
        }
        result_yellow = judge._determine_dimension_scores(
            dimension_answers_yellow, verbose=False
        )
        assert result_yellow["Dim2"]["score"] == "Suboptimal but Low Potential for Harm"

        # Test Green severity (should map to Best Practice since it's not Red/Yellow)
        dimension_answers_green = {
            "Dim3": [
                {
                    "question_id": "3",
                    "question": "Test",
                    "answer": "Yes",
                    "severity": "Green",
                    "reasoning": "Green test",
                }
            ]
        }
        result_green = judge._determine_dimension_scores(
            dimension_answers_green, verbose=False
        )
        assert result_green["Dim3"]["score"] == "Best Practice"

    def test_high_risk_takes_precedence_over_medium(self, fixtures_dir: Path, mock_llm):
        """Test high risk severity takes precedence over medium."""
        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_simple.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        dimension_answers = {
            "Safety": [
                {
                    "question_id": "1",
                    "question": "Yellow issue?",
                    "answer": "Yes",
                    "severity": "Yellow",
                    "reasoning": "Medium risk found.",
                },
                {
                    "question_id": "2",
                    "question": "Red issue?",
                    "answer": "Yes",
                    "severity": "Red",
                    "reasoning": "High risk found.",
                },
            ]
        }

        results = judge._determine_dimension_scores(dimension_answers, verbose=False)

        # Red should take precedence over Yellow
        assert results["Safety"]["score"] == "High Potential for Harm"
        assert "High risk issues found" in results["Safety"]["reasoning"]
        # Should contain reference to the Red issue
        assert "Q2" in results["Safety"]["reasoning"]

    def test_multiple_yes_answers_tracks_first_yes(self, fixtures_dir: Path, mock_llm):
        """Test that when multiple Yes answers exist, only the first one is tracked."""
        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_simple.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        dimension_answers = {
            "Safety": [
                {
                    "question_id": "1",
                    "question": "First yes question?",
                    "answer": "Yes",
                    "severity": "Yellow",
                    "reasoning": "First yes reasoning.",
                },
                {
                    "question_id": "2",
                    "question": "Second yes question?",
                    "answer": "Yes",
                    "severity": "Red",
                    "reasoning": "Second yes reasoning.",
                },
            ]
        }

        results = judge._determine_dimension_scores(dimension_answers, verbose=False)

        # Should track the first Yes answer
        assert results["Safety"]["yes_question_id"] == "1"
        assert results["Safety"]["yes_reasoning"] == "First yes reasoning."

    def test_dimension_scoring_empty_dimension_answers(
        self, fixtures_dir: Path, mock_llm
    ):
        """Test handling of empty dimension answers dictionary."""
        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_simple.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        dimension_answers = {}

        results = judge._determine_dimension_scores(dimension_answers, verbose=False)

        # Should return empty results
        assert results == {}

    def test_dimension_scoring_no_answer_with_only_severity(
        self, fixtures_dir: Path, mock_llm
    ):
        """Test that severity is only considered when answer is 'Yes'."""
        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_simple.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        dimension_answers = {
            "Safety": [
                {
                    "question_id": "1",
                    "question": "Has issues?",
                    "answer": "Maybe",  # Not "Yes"
                    "severity": "Red",
                    "reasoning": "Maybe has issues.",
                }
            ]
        }

        results = judge._determine_dimension_scores(dimension_answers, verbose=False)

        # Should be Best Practice since answer is not "Yes"
        assert results["Safety"]["score"] == "Best Practice"

    def test_reasoning_truncation_in_high_risk(self, fixtures_dir: Path, mock_llm):
        """Test that reasoning is included in full in high risk scoring."""
        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_simple.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        long_reasoning = "A" * 200  # 200 character reasoning

        dimension_answers = {
            "Safety": [
                {
                    "question_id": "1",
                    "question": "Test?",
                    "answer": "Yes",
                    "severity": "Red",
                    "reasoning": long_reasoning,
                }
            ]
        }

        results = judge._determine_dimension_scores(dimension_answers, verbose=False)

        # The reasoning in the score should include the full reasoning
        reasoning_part = results["Safety"]["reasoning"].split("Q1: ")[1]
        assert len(reasoning_part) == 200

    def test_reasoning_truncation_in_medium_risk(self, fixtures_dir: Path, mock_llm):
        """Test that reasoning is included in full in medium risk scoring."""
        judge = LLMJudge(
            judge_model="mock-llm",
            rubric_folder=str(fixtures_dir),
            rubric_file="rubric_simple.tsv",
            rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        )

        long_reasoning = "B" * 200  # 200 character reasoning

        dimension_answers = {
            "Privacy": [
                {
                    "question_id": "5",
                    "question": "Test?",
                    "answer": "Yes",
                    "severity": "Yellow",
                    "reasoning": long_reasoning,
                }
            ]
        }

        results = judge._determine_dimension_scores(dimension_answers, verbose=False)

        # The reasoning in the score should include the full reasoning
        reasoning_part = results["Privacy"]["reasoning"].split("Q5: ")[1]
        assert len(reasoning_part) == 200
