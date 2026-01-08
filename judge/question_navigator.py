"""Question navigation logic for rubric-based evaluation."""

from typing import Any, Dict, List, Optional


class QuestionNavigator:
    """Handles navigation through questions based on rubric flow logic."""

    def __init__(
        self,
        question_flow_data: Dict[str, Dict[str, Any]],
        question_order: List[str],
    ):
        """
        Initialize the question navigator with pre-parsed rubric data.

        Args:
            question_flow_data: Dictionary mapping Question ID to question data
            question_order: Ordered list of Question IDs
        """
        self.question_flow_data = question_flow_data
        self.question_order = question_order

    def get_next_question(
        self, current_question_id: str, answer_text: str
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Determine the next question based on current question and answer.

        Navigation Logic:
        1. If the Answer column has non-empty rows (explicit answer options):
           - Match the given answer to an option
           - If GOTO column has a value for that answer, go to that question
           - If GOTO is empty, go to the next row (next question in order)
           - Special GOTO values "END" and "ASSIGN_END" signal to stop evaluation

        2. If the Answer column is empty (implicit Yes/No questions):
           - If answer is "Yes": jump to the first question of the next dimension
           - If answer is "No": go to the next row (next question in order)

        Args:
            current_question_id: Current question ID
            answer_text: The answer given by the evaluator

        Returns:
            Tuple of (next_question_id, goto_value):
            - next_question_id: ID of the next question to ask, or None if end of flow
            - goto_value: The GOTO value from the answer
                ("END", "ASSIGN_END", question ID, or None)
        """
        if current_question_id not in self.question_flow_data:
            return None, None

        question_data = self.question_flow_data[current_question_id]
        answers = question_data.get("answers", [])
        current_dimension = question_data.get("dimension", "")

        goto_value = None
        next_question_id = None

        # Case 1: Explicit answer options exist
        if answers:
            # Find the matching answer option
            for ans in answers:
                if ans["option"].lower() == answer_text.lower():
                    goto_value = ans.get("goto")

                    # If GOTO has a value, use it (could be question ID,
                    # "END", "ASSIGN_END", or "NOT_RELEVANT>>{ID}")
                    if goto_value:
                        # Special values "END", "ASSIGN_END", and
                        # "NOT_RELEVANT>>" will be handled by caller
                        if goto_value not in [
                            "END",
                            "ASSIGN_END",
                        ] and not goto_value.startswith("NOT_RELEVANT>>"):
                            next_question_id = goto_value
                        elif goto_value.startswith("NOT_RELEVANT>>"):
                            # Extract the question ID after NOT_RELEVANT>>
                            next_question_id = goto_value.split(">>")[1]
                    else:
                        # GOTO is empty, go to next row
                        next_question_id = self._get_next_row_question(
                            current_question_id
                        )
                    break

        # Case 2: No explicit answers (implicit Yes/No behavior)
        else:
            if answer_text.lower() == "yes":
                # Go to first question of next dimension
                next_question_id = self._find_next_dimension_question(
                    current_question_id, current_dimension
                )
            elif answer_text.lower() == "no":
                # Go to next row
                next_question_id = self._get_next_row_question(current_question_id)

        return next_question_id, goto_value

    def _get_next_row_question(self, current_question_id: str) -> Optional[str]:
        """
        Get the question in the next row of the rubric.

        Args:
            current_question_id: Current question ID

        Returns:
            Next question ID in order, or None if at end
        """
        try:
            current_index = self.question_order.index(current_question_id)
            if current_index + 1 < len(self.question_order):
                return self.question_order[current_index + 1]
        except (ValueError, AttributeError):
            pass
        return None

    def _find_next_dimension_question(
        self, current_question_id: str, current_dimension: str
    ) -> Optional[str]:
        """
        Find the first question of the next dimension based on row order in the sheet.

        Args:
            current_question_id: Current question ID
            current_dimension: Current dimension name

        Returns:
            Question ID of first question in next dimension, or None
                if no next dimension
        """
        # Find the current question's position in the row order

        try:
            current_index = self.question_order.index(current_question_id)
        except ValueError:
            return None

        # Look for the next question with a different dimension in row order
        for i in range(current_index + 1, len(self.question_order)):
            q_id = self.question_order[i]
            q_data = self.question_flow_data.get(q_id)
            if (
                q_data
                and q_data.get("dimension")
                and q_data["dimension"] != current_dimension
            ):
                return q_id

        return None

    def get_question_data(self, question_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the full data for a specific question.

        Args:
            question_id: Question ID to retrieve

        Returns:
            Dictionary with question data, or None if not found
        """
        return self.question_flow_data.get(question_id)
