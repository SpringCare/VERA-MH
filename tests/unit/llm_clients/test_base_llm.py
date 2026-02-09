"""Base test classes for LLM implementations.

This module provides abstract base test classes that define common test patterns
for all LLM implementations. Concrete test classes inherit from these bases and
implement provider-specific factory methods.

Architecture:
- TestLLMBase: Tests for all LLMInterface implementations
- TestJudgeLLMBase: Tests for all JudgeLLM implementations (extends TestLLMBase)

Usage:
    @pytest.mark.usefixtures("mock_my_config", "mock_my_model")
    class TestMyLLM(TestJudgeLLMBase):
        def create_llm(self, role, **kwargs):
            return MyLLM(name="test", role=role, **kwargs)

        def get_provider_name(self):
            return "my_provider"

        @contextmanager
        def get_mock_patches(self):
            # No-op context manager when using class-level fixtures
            yield

Note: Modern implementations should use @pytest.mark.usefixtures at the class level
and make get_mock_patches() return a simple no-op context manager.
"""

from abc import ABC, abstractmethod
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, Field

from llm_clients import Role
from llm_clients.llm_interface import JudgeLLM, LLMInterface

from .test_helpers import (
    assert_error_metadata,
    assert_error_response,
    assert_iso_timestamp,
    assert_metadata_copy_behavior,
    assert_metadata_structure,
    assert_response_timing,
)


@pytest.mark.unit
class TestLLMBase(ABC):
    """Abstract base test class for all LLMInterface implementations.

    Subclasses must implement:
    - create_llm(role, **kwargs) -> LLMInterface
    - get_provider_name() -> str
    - get_mock_patches() -> context manager

    Provides standard tests that all LLM implementations must pass.
    """

    # ============================================================================
    # Abstract Factory Methods (Must be implemented by subclasses)
    # ============================================================================

    @abstractmethod
    def create_llm(self, role: Role, **kwargs) -> LLMInterface:
        """Create an instance of the LLM implementation being tested.

        Args:
            role: The role for the LLM (PERSONA, PROVIDER, or JUDGE)
            **kwargs: Additional arguments to pass to LLM constructor

        Returns:
            Instance of the LLM implementation
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the provider name for metadata validation.

        Returns:
            Provider name string (e.g., "claude", "openai", "gemini", "azure", "ollama")
        """
        pass

    @abstractmethod
    def get_mock_patches(self):
        """Get context manager with all necessary mocks for testing.

        Modern implementations should use @pytest.mark.usefixtures at the class level
        and make this method return a simple no-op context manager:

            @contextmanager
            def get_mock_patches(self):
                yield

        For legacy implementations, this can still provide actual patches:

            @contextmanager
            def get_mock_patches(self):
                with patch("module.Config.API_KEY", "test-key"):
                    yield

        Returns:
            Context manager (use @contextmanager decorator)
        """
        pass

    # ============================================================================
    # Standard Test Methods (Inherited by all implementations)
    # ============================================================================

    def test_init_with_role_and_system_prompt(self):
        """Test basic initialization with role and system prompt."""
        with self.get_mock_patches():  # pyright: ignore[reportGeneralTypeIssues]
            llm = self.create_llm(
                role=Role.PERSONA, name="TestLLM", system_prompt="Test prompt"
            )

            assert llm.name == "TestLLM"
            assert llm.role == Role.PERSONA
            assert llm.system_prompt == "Test prompt"
            assert llm.last_response_metadata == {}

    def test_set_system_prompt(self):
        """Test setting and updating system prompt."""
        with self.get_mock_patches():  # pyright: ignore[reportGeneralTypeIssues]
            llm = self.create_llm(
                role=Role.PERSONA, name="TestLLM", system_prompt="Initial prompt"
            )

            assert llm.system_prompt == "Initial prompt"

            llm.set_system_prompt("Updated prompt")
            assert llm.system_prompt == "Updated prompt"

    @pytest.mark.asyncio
    async def test_generate_response_returns_llm_text(
        self, mock_response_factory, mock_llm_factory, mock_system_message
    ):
        """Test that generate_response returns the LLM response body (response.text).

        This verifies real behavior: the wrapper calls the client, then returns
        the response's .text attribute. Asserting the exact string ensures we
        are testing pass-through of the real implementation, not just that
        a mock returned something.
        """
        expected_text = "Test response text"
        with self.get_mock_patches():  # pyright: ignore[reportGeneralTypeIssues]
            mock_response = mock_response_factory(
                text=expected_text,
                response_id="test_id",
                provider=self.get_provider_name(),
            )
            mock_llm_client = mock_llm_factory(response=mock_response)

            llm = self.create_llm(role=Role.PROVIDER, name="TestLLM")
            llm.llm = mock_llm_client  # pyright: ignore[reportAttributeAccessIssue]

            response = await llm.generate_response(
                conversation_history=mock_system_message
            )

            assert response == expected_text

    @pytest.mark.asyncio
    async def test_generate_response_updates_metadata(
        self, mock_response_factory, mock_llm_factory, mock_system_message
    ):
        """Test that generate_response updates last_response_metadata."""
        with self.get_mock_patches():  # pyright: ignore[reportGeneralTypeIssues]
            mock_response = mock_response_factory(
                text="Response",
                response_id="test_123",
                provider=self.get_provider_name(),
            )

            mock_llm_client = mock_llm_factory(response=mock_response)

            llm = self.create_llm(role=Role.PROVIDER, name="TestLLM")
            llm.llm = mock_llm_client  # pyright: ignore[reportAttributeAccessIssue]

            response = await llm.generate_response(
                conversation_history=mock_system_message
            )
            assert (
                response == "Response"
            )  # success path: our code returned response.text

            # Verify metadata structure
            metadata = assert_metadata_structure(
                llm,
                expected_provider=self.get_provider_name(),
                expected_role=Role.PROVIDER,
            )

            assert "timestamp" in metadata
            assert_iso_timestamp(metadata["timestamp"])
            assert_response_timing(metadata)

    def test_get_last_response_metadata_returns_copy(self):
        """Test that get_last_response_metadata returns a copy, not original."""
        with self.get_mock_patches():  # pyright: ignore[reportGeneralTypeIssues]
            llm = self.create_llm(role=Role.PROVIDER, name="TestLLM")

            assert_metadata_copy_behavior(llm)

    @pytest.mark.asyncio
    async def test_generate_response_handles_errors(
        self, mock_llm_factory, mock_system_message
    ):
        """Test that generate_response handles API errors gracefully."""
        with self.get_mock_patches():  # pyright: ignore[reportGeneralTypeIssues]
            # Create mock that raises an exception
            mock_llm_client = mock_llm_factory(
                response=None, side_effect=Exception("API Error")
            )

            llm = self.create_llm(role=Role.PROVIDER, name="TestLLM")
            llm.llm = mock_llm_client  # pyright: ignore[reportAttributeAccessIssue]

            response = await llm.generate_response(
                conversation_history=mock_system_message
            )

            # Should return error message, not raise exception
            assert_error_response(response, "API Error")

            # Should have error metadata
            assert_error_metadata(
                llm,
                expected_provider=self.get_provider_name(),
                expected_error_substring="API Error",
            )


@pytest.mark.unit
class TestJudgeLLMBase(TestLLMBase):
    """Abstract base test class for all JudgeLLM implementations.

    Extends TestLLMBase to add structured output testing.
    All subclasses automatically inherit both LLMInterface and JudgeLLM tests.
    """

    # Override return type hint for create_llm
    @abstractmethod
    def create_llm(self, role: Role, **kwargs) -> JudgeLLM:
        """Create an instance of the JudgeLLM implementation being tested.

        Args:
            role: The role for the LLM (PERSONA, PROVIDER, or JUDGE)
            **kwargs: Additional arguments to pass to LLM constructor

        Returns:
            Instance of the JudgeLLM implementation
        """
        pass

    # ============================================================================
    # Structured Output Test Methods
    # ============================================================================

    @pytest.mark.asyncio
    async def test_generate_structured_response_success(self, mock_llm_factory):
        """Test successful structured response generation with simple model."""
        with self.get_mock_patches():  # pyright: ignore[reportGeneralTypeIssues]
            # Define test Pydantic model
            class TestResponse(BaseModel):
                answer: str = Field(description="The answer")
                reasoning: str = Field(description="The reasoning")

            # Create test response
            test_response = TestResponse(answer="Yes", reasoning="Because it's correct")

            # Mock structured LLM
            mock_structured_llm = MagicMock()
            mock_structured_llm.ainvoke = AsyncMock(return_value=test_response)

            # Mock main LLM with with_structured_output method
            mock_llm_client = MagicMock()
            mock_llm_client.with_structured_output = MagicMock(
                return_value=mock_structured_llm
            )

            llm = self.create_llm(role=Role.JUDGE, name="TestLLM")
            llm.llm = mock_llm_client  # pyright: ignore[reportAttributeAccessIssue]

            response = await llm.generate_structured_response(
                "What is the answer?", TestResponse
            )

            # Verify response type and content
            assert isinstance(response, TestResponse)
            assert response.answer == "Yes"
            assert response.reasoning == "Because it's correct"

            # Verify metadata
            metadata = assert_metadata_structure(
                llm,
                expected_provider=self.get_provider_name(),
                expected_role=Role.JUDGE,
            )
            assert metadata.get("structured_output") is True
            assert_response_timing(metadata)

    @pytest.mark.asyncio
    async def test_generate_structured_response_with_complex_model(
        self, mock_llm_factory
    ):
        """Test structured response with nested Pydantic model."""
        with self.get_mock_patches():  # pyright: ignore[reportGeneralTypeIssues]
            # Define nested Pydantic models
            class SubScore(BaseModel):
                value: int = Field(description="Score value")
                justification: str = Field(description="Justification")

            class ComplexResponse(BaseModel):
                overall_score: int = Field(description="Overall score")
                sub_scores: list[SubScore] = Field(description="Sub scores")
                summary: str = Field(description="Summary")

            # Create test response
            test_response = ComplexResponse(
                overall_score=85,
                sub_scores=[
                    SubScore(value=90, justification="Good quality"),
                    SubScore(value=80, justification="Needs improvement"),
                ],
                summary="Overall good performance",
            )

            # Mock structured LLM
            mock_structured_llm = MagicMock()
            mock_structured_llm.ainvoke = AsyncMock(return_value=test_response)

            mock_llm_client = MagicMock()
            mock_llm_client.with_structured_output = MagicMock(
                return_value=mock_structured_llm
            )

            llm = self.create_llm(role=Role.JUDGE, name="TestLLM")
            llm.llm = mock_llm_client  # pyright: ignore[reportAttributeAccessIssue]

            response = await llm.generate_structured_response(
                "Evaluate this.", ComplexResponse
            )

            # Verify complex structure
            assert isinstance(response, ComplexResponse)
            assert response.overall_score == 85
            assert len(response.sub_scores) == 2
            assert response.sub_scores[0].value == 90
            assert response.summary == "Overall good performance"

    @pytest.mark.asyncio
    async def test_generate_structured_response_error_handling(self):
        """Test error handling in structured response generation."""
        with self.get_mock_patches():  # pyright: ignore[reportGeneralTypeIssues]

            class TestResponse(BaseModel):
                answer: str

            # Mock structured LLM to raise error
            mock_structured_llm = MagicMock()
            mock_structured_llm.ainvoke = AsyncMock(
                side_effect=Exception("Structured output failed")
            )

            mock_llm_client = MagicMock()
            mock_llm_client.with_structured_output = MagicMock(
                return_value=mock_structured_llm
            )

            llm = self.create_llm(role=Role.JUDGE, name="TestLLM")
            llm.llm = mock_llm_client  # pyright: ignore[reportAttributeAccessIssue]

            # Should raise RuntimeError
            with pytest.raises(RuntimeError) as exc_info:
                await llm.generate_structured_response("Test", TestResponse)

            assert "Error generating structured response" in str(exc_info.value)
            assert "Structured output failed" in str(exc_info.value)

            # Verify error metadata was stored
            metadata = llm.get_last_response_metadata()
            assert "error" in metadata
            assert "Structured output failed" in metadata["error"]
            assert metadata["provider"] == self.get_provider_name()

    @pytest.mark.asyncio
    async def test_structured_response_invalid_type_raises_error(self):
        """Test that invalid response type is caught."""
        with self.get_mock_patches():  # pyright: ignore[reportGeneralTypeIssues]

            class TestResponse(BaseModel):
                answer: str

            # Mock returns wrong type (string instead of TestResponse)
            mock_structured_llm = MagicMock()
            mock_structured_llm.ainvoke = AsyncMock(return_value="Invalid response")

            mock_llm_client = MagicMock()
            mock_llm_client.with_structured_output = MagicMock(
                return_value=mock_structured_llm
            )

            llm = self.create_llm(role=Role.JUDGE, name="TestLLM")
            llm.llm = mock_llm_client  # pyright: ignore[reportAttributeAccessIssue]

            # Should raise error about wrong type
            with pytest.raises(RuntimeError) as exc_info:
                await llm.generate_structured_response("Test", TestResponse)

            assert "Error generating structured response" in str(exc_info.value)
