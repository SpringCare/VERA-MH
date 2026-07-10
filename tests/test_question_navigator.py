"""Comprehensive test suite for QuestionNavigator"""

import sys
from pathlib import Path

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from judge.question_navigator import QuestionNavigator
from judge.rubric_config import (
    DETECTS_POTENTIAL_RISK,
    EXPECTED_DIMENSION_NAMES,
    RubricConfig,
)


@pytest.fixture
async def navigator() -> QuestionNavigator:
    """Create a QuestionNavigator instance with the main rubric"""
    # Load production rubric from data/ directory
    rubric_config = await RubricConfig.load(
        rubric_folder="data",
        rubric_file="rubric.tsv",
        rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        question_prompt_file="question_prompt.txt",
    )
    return QuestionNavigator(
        question_flow_data=rubric_config.question_flow_data,
        question_order=rubric_config.question_order,
    )


class TestRubricParsing:
    """Test that the rubric is parsed correctly"""

    async def test_rubric_loads(self, navigator):
        """Test that rubric file loads successfully"""
        assert navigator.question_flow_data is not None
        assert len(navigator.question_flow_data) > 0
        assert len(navigator.question_order) > 0

    async def test_questions_have_required_fields(self, navigator):
        """Test that all questions have required fields"""
        for q_id, q_data in navigator.question_flow_data.items():
            assert "dimension" in q_data
            assert "question" in q_data
            assert "answers" in q_data
            assert len(q_data["answers"]) > 0


class TestBasicNavigation:
    """Test basic navigation between questions"""

    async def test_navigate_with_goto(self, navigator):
        """Test navigation when GOTO has explicit question ID"""
        # Question 4, answer "Yes" should go to question 6
        next_q, goto = navigator.get_next_question("4", "Yes")
        assert next_q == "6"
        assert goto == "6"

    async def test_end_navigation(self, navigator):
        """Test that END GOTO value is returned correctly"""
        # Question 3, answer "No" should have GOTO=END
        next_q, goto = navigator.get_next_question("3", "No")
        assert next_q is None
        assert goto == "END"

    async def test_assign_end_navigation(self, navigator):
        """Test that ASSIGN_END GOTO value is returned correctly"""
        # Question 6, answer "Yes" should have GOTO=ASSIGN_END
        next_q, goto = navigator.get_next_question("3", "Yes")
        assert next_q is None
        assert goto == "ASSIGN_END"


class TestNotRelevantGoto:
    """Test NOT_RELEVANT>>{ID} GOTO behavior"""

    async def test_not_relevant_parsing(self, navigator):
        """Test that NOT_RELEVANT>>ID is parsed correctly"""
        # Question 9 should have "No suicidal thoughts present"
        # answer with NOT_RELEVANT>>22
        q9_data = navigator.get_question_data("9")
        assert q9_data is not None

        # Find the "No suicidal thoughts present" answer
        no_suicidal_thoughts_ans = None
        for ans in q9_data.get("answers", []):
            if "No suicidal thoughts present" in ans["option"]:
                no_suicidal_thoughts_ans = ans
                break

        assert no_suicidal_thoughts_ans is not None, "No suicidal thoughts present"
        assert (
            no_suicidal_thoughts_ans["goto"] == "NOT_RELEVANT>>23"
        ), f"Expected 'NOT_RELEVANT>>23', got '{no_suicidal_thoughts_ans['goto']}'"

    async def test_not_relevant_navigation(self, navigator):
        """Test that NOT_RELEVANT>> navigation extracts question ID correctly"""
        # Answer "No suicidal thoughts present" to question 9
        next_q, goto_value = navigator.get_next_question(
            "9", "No suicidal thoughts present"
        )

        # Should extract question ID 23 from NOT_RELEVANT>>23
        assert next_q == "23", f"Expected next question to be '23', got '{next_q}'"
        assert (
            goto_value == "NOT_RELEVANT>>23"
        ), f"Expected goto_value 'NOT_RELEVANT>>23', got '{goto_value}'"

    async def test_not_relevant_goto_different_from_end(self, navigator):
        """Test that NOT_RELEVANT>> behaves differently from END"""
        # NOT_RELEVANT>> should return a next question ID
        next_q_not_rel, goto_not_rel = navigator.get_next_question(
            "9", "No suicidal thoughts present"
        )

        # END should not return a next question ID
        next_q_end, goto_end = navigator.get_next_question(
            "1", "User-agent refuses the role-play"
        )

        assert next_q_not_rel is not None, "NOT_RELEVANT>> should have next question"
        assert next_q_end is None, "END should have no next question"
        assert goto_not_rel.startswith("NOT_RELEVANT>>")
        assert goto_end == "END"

    async def test_special_case_assigns_and_next_dimension_is_not_relevant(
        self, navigator
    ):
        """
        Test that the special case of assigning severity
        and next dimension is not relevant.
        """
        # Question 2, yes goes to 5
        next_q, goto = navigator.get_next_question("2", "Yes")
        assert next_q == "5"
        assert goto == "5"

        # Question 5, LLM is not called, dimension marked not relevant, move to 9
        next_q, goto = navigator.get_next_question(
            "5", "Rate this dimension Not Relevant"
        )
        assert next_q == "9"
        assert goto == "NOT_RELEVANT>>9"


class TestNavigationPaths:
    """Test specific navigation paths through the rubric"""

    async def test_empty_answer_path(self, navigator):
        """Test navigation path when Answer column is empty"""
        # Q23: Yes (chatbot makes shameful statements) -> Q29 (next dimension)
        next_q, _ = navigator.get_next_question("23", "Yes")
        assert next_q == "30", f"Expected next question 30, got {next_q}"

        # Q23: No (chatbot does not make shameful statements) -> Q24 (next row)
        next_q, _ = navigator.get_next_question("23", "No")
        assert next_q == "24", f"Expected next question 24, got {next_q}"

    async def test_risk_detected_path(self, navigator):
        """Test navigation path when risk is detected"""
        # Q1: Yes (User indicates risk) -> Q2
        next_q, _ = navigator.get_next_question("1", "Yes")
        assert next_q == "2"

        # Q2: Yes (Chatbot misses risk) -> Q5
        next_q, goto = navigator.get_next_question("2", "Yes")
        assert next_q == "5"
        assert goto == "5"

    async def test_no_risk_detected_path(self, navigator):
        """Test navigation path when no risk is detected"""
        # Q1: No (no risk) -> Q3
        next_q, _ = navigator.get_next_question("1", "No")
        assert next_q == "3"

        # Q3: No (no false positives) -> END
        next_q, goto = navigator.get_next_question("3", "No")
        assert next_q is None
        assert goto == "END"

    async def test_false_positive_path(self, navigator):
        """Test navigation path for false positives"""
        # Q1: No (no risk) -> Q3
        next_q, _ = navigator.get_next_question("1", "No")
        assert next_q == "3"

        # Q3: Yes (false positive) -> ASSIGN_END
        next_q, goto = navigator.get_next_question("3", "Yes")
        assert next_q is None
        assert goto == "ASSIGN_END"

    async def test_provides_resources_path(self, navigator):
        """Test navigation through Guides to Human Care dimension"""
        # Q9: Suicidal thoughts but not immediate risk -> Q10
        next_q, _ = navigator.get_next_question(
            "9", "Suicidal thoughts but not immediate risk"
        )
        assert next_q == "10"

        # Q10: Yes (chatbot fails to offer 1+ 24/7 crisis resources) -> Q22
        next_q, goto = navigator.get_next_question("10", "Yes")
        assert next_q == "23"
        assert goto == "23"

        # Q23: No GOTOs, so Yes or No (chatbot makes shameful statements) -> Q24
        for answer, expected_next_q in [("Yes", "30"), ("No", "24")]:
            next_q, goto = navigator.get_next_question("23", answer)
            assert (
                next_q == expected_next_q
            ), f"Expected next question {expected_next_q}, got {next_q}"
            assert goto is None


class TestEdgeCases:
    """Test edge cases and error handling"""

    async def test_invalid_question_id(self, navigator):
        """Test navigation with invalid question ID"""
        next_q, goto = navigator.get_next_question("999", "Yes")
        assert next_q is None
        assert goto is None

    async def test_case_insensitive_answers(self, navigator):
        """Test that answer matching is case-insensitive"""
        next_q1, _ = navigator.get_next_question("4", "Yes")
        next_q2, _ = navigator.get_next_question("4", "yes")
        next_q3, _ = navigator.get_next_question("4", "YES")

        assert next_q1 == next_q2 == next_q3

    async def test_get_question_data(self, navigator):
        """Test retrieving question data"""
        q4_data = navigator.get_question_data("4")
        assert q4_data is not None
        assert q4_data["dimension"] == DETECTS_POTENTIAL_RISK

        # Test invalid question ID
        invalid = navigator.get_question_data("999")
        assert invalid is None

    async def test_all_questions_accessible(self, navigator):
        """Test that all questions in question_order are accessible"""
        for q_id in navigator.question_order:
            q_data = navigator.get_question_data(q_id)
            assert q_data is not None, f"Question {q_id} should be accessible"


class TestQuestionOrderAndDimensions:
    """Test question ordering and dimension structure"""

    async def test_first_question(self, navigator):
        """Test that first question is accessible"""
        assert len(navigator.question_order) > 0
        first_q_id = navigator.question_order[0]
        first_q_data = navigator.get_question_data(first_q_id)
        assert first_q_data is not None

    async def test_dimensions_present(self, navigator):
        """Test that all expected dimensions are present"""
        expected_dimensions = EXPECTED_DIMENSION_NAMES

        found_dimensions = set()
        for q_data in navigator.question_flow_data.values():
            dim = q_data.get("dimension")
            if dim:
                found_dimensions.add(dim)

        assert expected_dimensions.issubset(
            found_dimensions
        ), f"Missing dimensions: {expected_dimensions - found_dimensions}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
