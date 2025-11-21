"""Question navigation logic for rubric-based evaluation."""

import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path


class QuestionNavigator:
    """Handles navigation through questions based on rubric flow logic."""

    def __init__(self, rubric_path: str, sep: str = "\t"):
        """
        Initialize the question navigator by loading and parsing the rubric file.

        Args:
            rubric_path: Path to the rubric TSV file
            sep: Separator for the rubric file (default: tab)
        """
        self.rubric_path = Path(rubric_path)
        if not self.rubric_path.exists():
            raise FileNotFoundError(f"Rubric file not found: {rubric_path}")

        # Load and parse the rubric
        self.rubric_df = pd.read_csv(rubric_path, sep=sep)
        self.question_flow_data, self.question_order = self._parse_rubric()

    def _parse_rubric(self) -> tuple[Dict[str, Any], List[str]]:
        """
        Parse the rubric file into a navigable data structure.

        The rubric has questions with potential multi-row answer options.
        Questions have a Question ID, and subsequent rows with blank Question ID
        contain answer options for that question.

        Returns:
            Tuple of (questions_dict, question_order_list):
            - questions_dict: Dictionary mapping Question ID to question data
            - question_order_list: Ordered list of Question IDs as they appear in the file
        """
        questions = {}
        question_order = []
        current_question_id = None
        current_question_data = None

        for idx, row in self.rubric_df.iterrows():
            question_id_raw = row['Question ID'] if pd.notna(row['Question ID']) else None

            # Convert to string and clean up (remove .0 from floats)
            if question_id_raw is not None:
                question_id = str(int(float(question_id_raw))) if isinstance(question_id_raw, (int, float)) else str(question_id_raw).strip()
            else:
                question_id = ""

            # If this row has a Question ID, it's a new question
            if question_id and question_id != 'nan':
                # Save previous question if exists
                if current_question_id and current_question_data:
                    questions[current_question_id] = current_question_data

                # Read severity from the question row (not from answers)
                severity = str(row['Severity']).strip() if pd.notna(row['Severity']) else ""
                severity = severity if severity and severity not in ['nan', ''] else None

                # Start new question
                current_question_id = question_id
                question_order.append(question_id)
                current_question_data = {
                    "dimension": str(row['Dimension']).strip() if pd.notna(row['Dimension']) else "",
                    "risk_type": str(row['Risk Type']).strip() if pd.notna(row['Risk Type']) else "",
                    "question": str(row['Question']).strip() if pd.notna(row['Question']) else "",
                    "examples": str(row['Examples']).strip() if pd.notna(row['Examples']) else "",
                    "severity": severity,  # Severity at question level
                    "answers": []
                }

                # Check if this row also has an answer (single-row question)
                answer = str(row['Answer']).strip() if pd.notna(row['Answer']) else ""
                if answer and answer != 'nan':
                    # If there is an explicit GOTO, use it
                    goto_raw = row['GOTO'] if pd.notna(row['GOTO']) else None
                    goto = str(int(float(goto_raw))) if goto_raw and isinstance(goto_raw, (int, float)) else (str(goto_raw).strip() if goto_raw else None)
                    current_question_data["answers"].append({
                        "option": answer,
                        "goto": goto if goto and goto != 'nan' else None
                    })

            # This is a continuation row with an answer option
            elif current_question_data is not None:
                answer = str(row['Answer']).strip() if pd.notna(row['Answer']) else ""
                if answer and answer != 'nan':
                    goto_raw = row['GOTO'] if pd.notna(row['GOTO']) else None
                    goto = str(int(float(goto_raw))) if goto_raw and isinstance(goto_raw, (int, float)) else (str(goto_raw).strip() if goto_raw else None)
                    current_question_data["answers"].append({
                        "option": answer,
                        "goto": goto if goto and goto != 'nan' else None
                    })

        # Save last question
        if current_question_id and current_question_data:
            questions[current_question_id] = current_question_data

        # Add default Yes/No answers for questions without explicit answers
        for question_id in question_order:
            question_data = questions[question_id]
            if len(question_data["answers"]) == 0:
                # Add default answers: Yes (no GOTO, will use default), No (no GOTO, will use default)
                question_data["answers"] = [
                    {"option": "Yes", "goto": None},
                    {"option": "No", "goto": None}
                ]

        return questions, question_order

    def get_next_question(
        self,
        current_question_id: str,
        answer_text: str
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
            - goto_value: The GOTO value from the answer ("END", "ASSIGN_END", question ID, or None)
        """
        if current_question_id not in self.question_flow_data:
            return None, None

        question_data = self.question_flow_data[current_question_id]
        answers = question_data.get('answers', [])
        current_dimension = question_data.get('dimension', '')

        goto_value = None
        next_question_id = None

        # Case 1: Explicit answer options exist
        if answers:
            # Find the matching answer option
            for ans in answers:
                if ans['option'].lower() == answer_text.lower():
                    goto_value = ans.get('goto')

                    # If GOTO has a value, use it (could be question ID, "END", or "ASSIGN_END")
                    if goto_value:
                        # Special values "END" and "ASSIGN_END" will be handled by caller
                        if goto_value not in ["END", "ASSIGN_END"]:
                            next_question_id = goto_value
                    else:
                        # GOTO is empty, go to next row
                        next_question_id = self._get_next_row_question(current_question_id)
                    break

        # Case 2: No explicit answers (implicit Yes/No behavior)
        else:
            if answer_text.lower() == 'yes':
                # Go to first question of next dimension
                next_question_id = self._find_next_dimension_question(
                    current_question_id, current_dimension
                )
            elif answer_text.lower() == 'no':
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
        self,
        current_question_id: str,
        current_dimension: str
    ) -> Optional[str]:
        """
        Find the first question of the next dimension.

        Args:
            current_question_id: Current question ID
            current_dimension: Current dimension name

        Returns:
            Question ID of first question in next dimension, or None if no next dimension
        """
        current_id_num = int(current_question_id)

        # Find questions with a different dimension that comes after current question
        for q_id in sorted(self.question_flow_data.keys(), key=lambda x: int(x)):
            q_id_num = int(q_id)
            if q_id_num > current_id_num:
                q_data = self.question_flow_data[q_id]
                if q_data['dimension'] and q_data['dimension'] != current_dimension:
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
