"""Pydantic models for structured LLM responses in judge evaluation."""

from pydantic import BaseModel, Field


class QuestionResponse(BaseModel):
    """Structured response for a judge question.

    This model ensures the LLM returns a properly formatted response
    with both an answer (from the available options) and reasoning.
    """

    answer: str = Field(
        description="The selected answer from the provided options (e.g., 'Yes', 'No', 'Denies suicidal thoughts', etc.)"
    )
    reasoning: str = Field(
        description="Brief explanation of why this answer was chosen based on the conversation"
    )

    class Config:
        """Pydantic config."""

        json_schema_extra = {
            "example": {
                "answer": "Yes",
                "reasoning": "The assistant provided empathetic responses that acknowledged the user's feelings.",
            }
        }
