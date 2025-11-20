"""Pydantic schemas for structured LLM outputs."""

from pydantic import BaseModel, Field


class JudgeResponse(BaseModel):
    """Schema for judge evaluation responses with answer and reasoning."""

    answer: str = Field(..., description="The answer to the question (e.g., 'Yes', 'No', 'Not Relevant')")
    reasoning: str = Field(..., description="Brief explanation for the answer")
