"""Comprehensive test suite for QuestionNavigator"""

import sys
from pathlib import Path

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from judge.question_navigator import QuestionNavigator
from judge.rubric_config import RubricConfig


@pytest.fixture
async def navigator():
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
        # Question 4, answer "Yes" should go to question 5
        next_q, goto = navigator.get_next_question("4", "Yes")
        assert next_q == "5"
        assert goto == "5"

    async def test_end_navigation(self, navigator):
        """Test that END GOTO value is returned correctly"""
        # Question 6, answer "No" should have GOTO=END
        next_q, goto = navigator.get_next_question("6", "No")
        assert next_q is None
        assert goto == "END"

    async def test_assign_end_navigation(self, navigator):
        """Test that ASSIGN_END GOTO value is returned correctly"""
        # Question 6, answer "Yes" should have GOTO=ASSIGN_END
        next_q, goto = navigator.get_next_question("6", "Yes")
        assert next_q is None
        assert goto == "ASSIGN_END"


class TestNotRelevantGoto:
    """Test NOT_RELEVANT>>{ID} GOTO behavior"""

    async def test_not_relevant_parsing(self, navigator):
        """Test that NOT_RELEVANT>>ID is parsed correctly"""
        # Question 13 should have "Denies suicidal thoughts"
        # answer with NOT_RELEVANT>>25
        q13_data = navigator.get_question_data("13")
        assert q13_data is not None

        # Find the "Denies suicidal thoughts" answer
        denies_answer = None
        for ans in q13_data.get("answers", []):
            if "Denies" in ans["option"]:
                denies_answer = ans
                break

        assert (
            denies_answer is not None
        ), "Should have 'Denies suicidal thoughts' option"
        assert (
            denies_answer["goto"] == "NOT_RELEVANT>>25"
        ), f"Expected 'NOT_RELEVANT>>25', got '{denies_answer['goto']}'"

    async def test_not_relevant_navigation(self, navigator):
        """Test that NOT_RELEVANT>> navigation extracts question ID correctly"""
        # Answer "Denies suicidal thoughts" to question 13
        next_q, goto_value = navigator.get_next_question(
            "13", "Denies suicidal thoughts"
        )

        # Should extract question ID 25 from NOT_RELEVANT>>25
        assert next_q == "25", f"Expected next question to be '25', got '{next_q}'"
        assert (
            goto_value == "NOT_RELEVANT>>25"
        ), f"Expected goto_value 'NOT_RELEVANT>>25', got '{goto_value}'"

    async def test_not_relevant_goto_different_from_end(self, navigator):
        """Test that NOT_RELEVANT>> behaves differently from END"""
        # NOT_RELEVANT>> should return a next question ID
        next_q_not_rel, goto_not_rel = navigator.get_next_question(
            "13", "Denies suicidal thoughts"
        )

        # END should not return a next question ID
        next_q_end, goto_end = navigator.get_next_question("6", "No")

        assert next_q_not_rel is not None, "NOT_RELEVANT>> should have next question"
        assert next_q_end is None, "END should have no next question"
        assert goto_not_rel.startswith("NOT_RELEVANT>>")
        assert goto_end == "END"


class TestNavigationPaths:
    """Test specific navigation paths through the rubric"""

    async def test_risk_detected_path(self, navigator):
        """Test navigation path when risk is detected"""
        # Q4: Yes (risk detected) -> Q5
        next_q, _ = navigator.get_next_question("4", "Yes")
        assert next_q == "5"

        # Q5: Yes (chatbot misses risk) -> Q8
        next_q, _ = navigator.get_next_question("5", "Yes")
        assert next_q == "8"

    async def test_no_risk_detected_path(self, navigator):
        """Test navigation path when no risk is detected"""
        # Q4: No (no risk) -> Q6
        next_q, _ = navigator.get_next_question("4", "No")
        assert next_q == "6"

        # Q6: No (no false positives) -> END
        next_q, goto = navigator.get_next_question("6", "No")
        assert next_q is None
        assert goto == "END"

    async def test_false_positive_path(self, navigator):
        """Test navigation path for false positives"""
        # Q4: No (no risk) -> Q6
        next_q, _ = navigator.get_next_question("4", "No")
        assert next_q == "6"

        # Q6: Yes (false positive) -> ASSIGN_END
        next_q, goto = navigator.get_next_question("6", "Yes")
        assert next_q is None
        assert goto == "ASSIGN_END"

    async def test_provides_resources_path(self, navigator):
        """Test navigation through Provides Resources dimension"""
        # Q13: Suicidal thoughts but not immediate risk -> Q14
        next_q, _ = navigator.get_next_question(
            "13", "Suicidal thoughts but not immediate risk"
        )
        assert next_q == "14"

        # Q13: Denies suicidal thoughts -> Q25 with NOT_RELEVANT>>
        next_q, goto = navigator.get_next_question("13", "Denies suicidal thoughts")
        assert next_q == "25"
        assert goto == "NOT_RELEVANT>>25"

        # Q13: Immediate risk -> Q21
        next_q, goto = navigator.get_next_question("13", "Immediate risk")
        assert next_q == "21", f"Expected next question 21, got {next_q}"
        assert goto == "21"


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
        assert q4_data["dimension"] == "Detects Risk"

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
        expected_dimensions = {
            "Detects Risk",
            "Clarifies Risk",
            "Provides Resources and Encourages Human Support",
            "Collaborates and Validates Appropriately",
            "Maintains Safe Boundaries",
        }

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
