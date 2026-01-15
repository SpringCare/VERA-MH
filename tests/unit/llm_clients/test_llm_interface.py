from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_clients.llm_interface import LLMInterface


class ExceptionWithStatusCode(Exception):
    """Exception with status_code attribute for testing."""

    def __init__(self, status_code: int, message: str = ""):
        self.status_code = status_code
        self.response: Any = None  # Can be set for testing Retry-After header
        super().__init__(message or f"HTTP {status_code}")


class ConcreteLLM(LLMInterface):
    """Concrete implementation for testing abstract base class."""

    def __init__(
        self, name: str, system_prompt: Optional[str] = None, max_retries: int = 3
    ):
        super().__init__(name, system_prompt, max_retries=max_retries)
        # Add a mock llm object for __getattr__ testing
        self.llm = MagicMock(spec=["temperature", "max_tokens", "custom_method"])
        self.llm.temperature = 0.7
        self.llm.max_tokens = 1000

    async def generate_response(self, conversation_history=None):
        """Concrete implementation of abstract method."""
        return "test response"

    def set_system_prompt(self, system_prompt: str) -> None:
        """Concrete implementation of abstract method."""
        self.system_prompt = system_prompt


class IncompleteLLM(LLMInterface):
    """Incomplete implementation to test abstract method enforcement."""

    pass


@pytest.mark.unit
class TestLLMInterface:
    """Unit tests for LLMInterface abstract base class."""

    def test_init_with_name_only(self):
        """Test initialization with only name parameter."""
        llm = ConcreteLLM(name="TestLLM")

        assert llm.name == "TestLLM"
        assert llm.system_prompt == ""

    def test_init_with_name_and_system_prompt(self):
        """Test initialization with name and system prompt."""
        prompt = "You are a helpful assistant."
        llm = ConcreteLLM(name="TestLLM", system_prompt=prompt)

        assert llm.name == "TestLLM"
        assert llm.system_prompt == prompt

    def test_get_name(self):
        """Test get_name method (line 30)."""
        llm = ConcreteLLM(name="MyLLM")
        assert llm.get_name() == "MyLLM"

    @pytest.mark.asyncio
    async def test_generate_response_abstract_method(self):
        """Test that generate_response is implemented in concrete class (line 21)."""
        llm = ConcreteLLM(name="TestLLM")
        response = await llm.generate_response(
            conversation_history=[
                {"turn": 0, "speaker": "system", "response": "test message"}
            ]
        )

        assert response == "test response"

    def test_set_system_prompt_abstract_method(self):
        """Test that set_system_prompt is implemented in concrete class (line 26)."""
        llm = ConcreteLLM(name="TestLLM", system_prompt="Initial prompt")
        assert llm.system_prompt == "Initial prompt"

        llm.set_system_prompt("Updated prompt")
        assert llm.system_prompt == "Updated prompt"

    def test_cannot_instantiate_abstract_class(self):
        """Test that LLMInterface cannot be instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            LLMInterface(name="Test")  # type: ignore[abstract]

        assert "Can't instantiate abstract class" in str(exc_info.value)

    def test_incomplete_implementation_raises_error(self):
        """Test that incomplete implementations raise TypeError."""
        with pytest.raises(TypeError) as exc_info:
            IncompleteLLM(name="Incomplete")  # type: ignore[abstract]

        assert "Can't instantiate abstract class" in str(exc_info.value)

    def test_getattr_delegates_to_llm(self):
        """Test that __getattr__ delegates to self.llm (lines 40-41)."""
        llm = ConcreteLLM(name="TestLLM")

        # Access attributes that exist on llm
        assert llm.temperature == 0.7
        assert llm.max_tokens == 1000

    def test_getattr_raises_attribute_error_for_missing_attribute(self):
        """Test that __getattr__ raises AttributeError for missing attributes."""
        llm = ConcreteLLM(name="TestLLM")

        # Try to access attribute that doesn't exist on llm (spec prevents it)
        with pytest.raises(AttributeError) as exc_info:
            _ = llm.nonexistent_attribute

        assert "ConcreteLLM" in str(exc_info.value)
        assert "nonexistent_attribute" in str(exc_info.value)

    def test_getattr_when_llm_not_set(self):
        """Test __getattr__ behavior when self.llm doesn't exist."""

        class MinimalLLM(LLMInterface):
            """Minimal implementation without self.llm."""

            async def generate_response(self, conversation_history=None):
                return "response"

            def set_system_prompt(self, system_prompt: str) -> None:
                self.system_prompt = system_prompt

        llm = MinimalLLM(name="Minimal")

        # Should raise RecursionError when self.llm doesn't exist
        # because hasattr(self, "llm") in __getattr__ calls __getattr__ again
        with pytest.raises(RecursionError):
            _ = llm.some_attribute

    def test_getattr_with_none_llm(self):
        """Test __getattr__ when self.llm is None."""

        class NullLLM(LLMInterface):
            """Implementation with None llm."""

            def __init__(self, name: str, system_prompt: Optional[str] = None):
                super().__init__(name, system_prompt)
                self.llm = None

            async def generate_response(self, conversation_history=None):
                return "response"

            def set_system_prompt(self, system_prompt: str) -> None:
                self.system_prompt = system_prompt

        llm = NullLLM(name="Null")

        # Should raise AttributeError since llm is None
        with pytest.raises(AttributeError):
            _ = llm.temperature

    def test_multiple_instances_have_independent_state(self):
        """Test that multiple LLM instances maintain independent state."""
        llm1 = ConcreteLLM(name="LLM1", system_prompt="Prompt 1")
        llm2 = ConcreteLLM(name="LLM2", system_prompt="Prompt 2")

        assert llm1.name == "LLM1"
        assert llm2.name == "LLM2"
        assert llm1.system_prompt == "Prompt 1"
        assert llm2.system_prompt == "Prompt 2"

        # Modify one shouldn't affect the other
        llm1.set_system_prompt("Modified Prompt 1")
        assert llm1.system_prompt == "Modified Prompt 1"
        assert llm2.system_prompt == "Prompt 2"

    def test_getattr_with_callable_attribute(self):
        """Test __getattr__ works with callable attributes."""
        llm = ConcreteLLM(name="TestLLM")
        llm.llm.custom_method = MagicMock(return_value="method result")

        # Access callable attribute through delegation
        result = llm.custom_method()
        assert result == "method result"
        llm.llm.custom_method.assert_called_once()

    def test_system_prompt_default_empty_string(self):
        """Test that system_prompt defaults to empty string, not None."""
        llm = ConcreteLLM(name="TestLLM")
        assert llm.system_prompt == ""
        assert llm.system_prompt is not None

    def test_getattr_preserves_attribute_type(self):
        """Test that __getattr__ preserves the type of delegated attributes."""

        # Create a fresh mock without spec for this test
        class FlexibleLLM(LLMInterface):
            def __init__(self, name: str, system_prompt: Optional[str] = None):
                super().__init__(name, system_prompt)
                self.llm = MagicMock()
                self.llm.string_attr = "test string"
                self.llm.int_attr = 42
                self.llm.float_attr = 3.14
                self.llm.bool_attr = True
                self.llm.list_attr = [1, 2, 3]

            async def generate_response(self, conversation_history=None):
                return "response"

            def set_system_prompt(self, system_prompt: str) -> None:
                self.system_prompt = system_prompt

        llm = FlexibleLLM(name="TestLLM")

        assert isinstance(llm.string_attr, str)
        assert isinstance(llm.int_attr, int)
        assert isinstance(llm.float_attr, float)
        assert isinstance(llm.bool_attr, bool)
        assert isinstance(llm.list_attr, list)


@pytest.mark.unit
class TestLLMInterfaceRetryLogic:
    """Unit tests for retry logic and error handling in LLMInterface."""

    def test_extract_http_status_code_from_status_code_attribute(self):
        """Test extracting status code from exception.status_code attribute."""
        llm = ConcreteLLM(name="TestLLM")

        class ExceptionWithStatusCode(Exception):
            def __init__(self, status_code):
                self.status_code = status_code
                super().__init__(f"HTTP {status_code}")

        exc = ExceptionWithStatusCode(429)
        assert llm._extract_http_status_code(exc) == 429

    def test_extract_http_status_code_from_response_attribute(self):
        """Test extracting status code from exception.response.status_code."""
        llm = ConcreteLLM(name="TestLLM")

        class MockResponse:
            def __init__(self, status_code):
                self.status_code = status_code

        class ExceptionWithResponse(Exception):
            def __init__(self, status_code):
                self.response = MockResponse(status_code)
                super().__init__(f"HTTP {status_code}")

        exc = ExceptionWithResponse(503)
        assert llm._extract_http_status_code(exc) == 503

    def test_extract_http_status_code_from_error_message(self):
        """Test extracting status code from error message string."""
        llm = ConcreteLLM(name="TestLLM")

        exc = Exception("HTTP status 429: Too Many Requests")
        assert llm._extract_http_status_code(exc) == 429

        exc2 = Exception("Request failed with status_code 503")
        assert llm._extract_http_status_code(exc2) == 503

    def test_extract_http_status_code_error_code_pattern(self):
        """Test extracting status code from 'Error Code' pattern."""
        llm = ConcreteLLM(name="TestLLM")

        exc = Exception("Error Code: 429")
        assert llm._extract_http_status_code(exc) == 429

        exc2 = Exception("Error Code 503 occurred")
        assert llm._extract_http_status_code(exc2) == 503

        exc3 = Exception("Error code: 500")
        assert llm._extract_http_status_code(exc3) == 500

    def test_extract_http_status_code_error_code_with_additional_text(self):
        """Test extracting status code from 'Error code' with additional text after."""
        llm = ConcreteLLM(name="TestLLM")

        # Real-world example from Azure API
        error_msg = (
            "Error code: 400 - {'type': 'error', 'error': "
            "{'type': 'invalid_request_error', 'message': "
            "'messages.2: all messages must have non-empty content except for "
            "the optional final assistant message'}, 'request_id': "
            "'req_011CX84UXrNZGnUz2i9YM7AX'}"
        )
        exc = Exception(error_msg)
        assert llm._extract_http_status_code(exc) == 400

        # Test with different formats
        exc2 = Exception("Error code: 429 - Rate limit exceeded")
        assert llm._extract_http_status_code(exc2) == 429

        exc3 = Exception("Error code: 503 Service unavailable")
        assert llm._extract_http_status_code(exc3) == 503

    def test_extract_http_status_code_no_match(self):
        """Test that None is returned when no status code can be extracted."""
        llm = ConcreteLLM(name="TestLLM")

        exc = Exception("Generic error message")
        assert llm._extract_http_status_code(exc) is None

    @pytest.mark.asyncio
    async def test_retry_with_backoff_empty_response_retries(self):
        """Test that empty response content triggers retry."""
        llm = ConcreteLLM(name="TestLLM", max_retries=3)

        call_count = 0

        async def func_with_empty_then_valid():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call returns empty response
                mock_response = MagicMock()
                mock_response.text = ""
                return mock_response
            else:
                # Subsequent calls return valid response
                mock_response = MagicMock()
                mock_response.text = "Valid response"
                return mock_response

        def validator(response_obj):
            """Validate that response has non-empty content."""
            return bool(response_obj.text and response_obj.text.strip())

        result = await llm._retry_with_backoff(
            func_with_empty_then_valid,
            operation_name="test_operation",
            response_validator=validator,
        )

        assert result.text == "Valid response"
        assert call_count == 2  # Should have retried once

    @pytest.mark.asyncio
    async def test_retry_with_backoff_empty_response_exhausts_retries(self):
        """Test that empty response raises error after max retries."""
        llm = ConcreteLLM(name="TestLLM", max_retries=2)

        async def func_always_empty():
            mock_response = MagicMock()
            mock_response.text = ""
            return mock_response

        def validator(response_obj):
            """Validate that response has non-empty content."""
            return bool(response_obj.text and response_obj.text.strip())

        with pytest.raises(RuntimeError) as exc_info:
            await llm._retry_with_backoff(
                func_always_empty,
                operation_name="test_operation",
                response_validator=validator,
            )

        assert "after 2 retries" in str(exc_info.value)
        assert "response content is empty" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_retry_with_backoff_whitespace_only_response_retries(self):
        """Test that whitespace-only response triggers retry."""
        llm = ConcreteLLM(name="TestLLM", max_retries=3)

        call_count = 0

        async def func_with_whitespace_then_valid():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call returns whitespace-only response
                mock_response = MagicMock()
                mock_response.text = "   \n\t  "
                return mock_response
            else:
                # Subsequent calls return valid response
                mock_response = MagicMock()
                mock_response.text = "Valid response"
                return mock_response

        def validator(response_obj):
            """Validate that response has non-empty content."""
            return bool(response_obj.text and response_obj.text.strip())

        result = await llm._retry_with_backoff(
            func_with_whitespace_then_valid,
            operation_name="test_operation",
            response_validator=validator,
        )

        assert result.text == "Valid response"
        assert call_count == 2  # Should have retried once

    @pytest.mark.asyncio
    async def test_retry_with_backoff_no_validator_passes_through(self):
        """Test that without validator, empty response is returned."""
        llm = ConcreteLLM(name="TestLLM", max_retries=3)

        async def func_with_empty():
            mock_response = MagicMock()
            mock_response.text = ""
            return mock_response

        # No validator provided
        result = await llm._retry_with_backoff(
            func_with_empty,
            operation_name="test_operation",
        )

        assert result.text == ""  # Empty response is returned without validation

    @pytest.mark.asyncio
    async def test_retry_with_backoff_validator_returns_true_immediately(self):
        """Test that valid response passes validator immediately."""
        llm = ConcreteLLM(name="TestLLM", max_retries=3)

        call_count = 0

        async def func_with_valid():
            nonlocal call_count
            call_count += 1
            mock_response = MagicMock()
            mock_response.text = "Valid response"
            return mock_response

        def validator(response_obj):
            """Validate that response has non-empty content."""
            return bool(response_obj.text and response_obj.text.strip())

        result = await llm._retry_with_backoff(
            func_with_valid,
            operation_name="test_operation",
            response_validator=validator,
        )

        assert result.text == "Valid response"
        assert call_count == 1  # Should not retry for valid response

    def test_extract_retry_after_from_headers(self):
        """Test extracting Retry-After header from exception.response.headers."""
        llm = ConcreteLLM(name="TestLLM")

        class MockHeaders:
            def __init__(self, retry_after):
                self._headers = {"Retry-After": str(retry_after)}

            def get(self, key):
                return self._headers.get(key) or self._headers.get(key.lower())

        class MockResponse:
            def __init__(self, retry_after):
                self.headers = MockHeaders(retry_after)

        class ExceptionWithRetryAfter(Exception):
            def __init__(self, retry_after):
                self.response = MockResponse(retry_after)
                super().__init__(f"Rate limited, retry after {retry_after}")

        exc = ExceptionWithRetryAfter(30)
        assert llm._extract_retry_after(exc) == 30

    def test_extract_retry_after_case_insensitive(self):
        """Test that Retry-After header extraction is case-insensitive."""
        llm = ConcreteLLM(name="TestLLM")

        class MockHeaders:
            def __init__(self, retry_after):
                self._headers = {"retry-after": str(retry_after)}

            def get(self, key):
                return self._headers.get(key) or self._headers.get(key.lower())

        class MockResponse:
            def __init__(self, retry_after):
                self.headers = MockHeaders(retry_after)

        class ExceptionWithRetryAfter(Exception):
            def __init__(self, retry_after):
                self.response = MockResponse(retry_after)
                super().__init__("Rate limited")

        exc = ExceptionWithRetryAfter(45)
        assert llm._extract_retry_after(exc) == 45

    def test_extract_retry_after_no_headers(self):
        """Test that None is returned when headers don't exist."""
        llm = ConcreteLLM(name="TestLLM")

        class ExceptionWithoutHeaders(Exception):
            def __init__(self):
                self.response = object()  # No headers attribute
                super().__init__("Error")

        exc = ExceptionWithoutHeaders()
        assert llm._extract_retry_after(exc) is None

    def test_extract_retry_after_no_response(self):
        """Test that None is returned when response doesn't exist."""
        llm = ConcreteLLM(name="TestLLM")

        exc = Exception("Generic error")
        assert llm._extract_retry_after(exc) is None

    def test_extract_retry_after_invalid_value(self):
        """Test that None is returned when Retry-After value is invalid."""
        llm = ConcreteLLM(name="TestLLM")

        class MockHeaders:
            def get(self, key):
                return "invalid"  # Not a number

        class MockResponse:
            def __init__(self):
                self.headers = MockHeaders()

        class ExceptionWithInvalidRetryAfter(Exception):
            def __init__(self):
                self.response = MockResponse()
                super().__init__("Error")

        exc = ExceptionWithInvalidRetryAfter()
        assert llm._extract_retry_after(exc) is None

    @pytest.mark.asyncio
    async def test_retry_with_backoff_429_rate_limit(self):
        """Test retry logic for 429 (Too Many Requests) status code."""
        llm = ConcreteLLM(name="TestLLM", max_retries=3)

        call_count = 0

        async def func_with_429_then_success():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ExceptionWithStatusCode(429, "HTTP status 429: Too Many Requests")
            return "Success after retry"

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await llm._retry_with_backoff(
                func_with_429_then_success, operation_name="test_operation"
            )

        assert result == "Success after retry"
        assert call_count == 2
        # Should sleep once with exponential backoff (2^0 = 1 second)
        assert mock_sleep.call_count == 1
        mock_sleep.assert_called_with(1)

    @pytest.mark.asyncio
    async def test_retry_with_backoff_429_with_retry_after_header(self):
        """Test that 429 respects Retry-After header."""
        from unittest.mock import AsyncMock, patch

        llm = ConcreteLLM(name="TestLLM", max_retries=3)

        call_count = 0

        class MockHeaders:
            def get(self, key):
                return "15" if key.lower() == "retry-after" else None

        class MockResponse:
            def __init__(self):
                self.headers = MockHeaders()

        async def func_with_429_and_retry_after():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                exc = ExceptionWithStatusCode(429, "HTTP status 429: Too Many Requests")
                exc.response = MockResponse()
                raise exc
            return "Success after retry"

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await llm._retry_with_backoff(
                func_with_429_and_retry_after,
                operation_name="test_operation",
            )

        assert result == "Success after retry"
        assert call_count == 2
        # Should use Retry-After header value (15s) instead of backoff
        assert mock_sleep.call_count == 1
        mock_sleep.assert_called_with(15)

    @pytest.mark.asyncio
    async def test_retry_with_backoff_500_internal_server_error(self):
        """Test retry logic for 500 (Internal Server Error) status code."""
        llm = ConcreteLLM(name="TestLLM", max_retries=5)

        call_count = 0

        async def func_with_500_then_success():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ExceptionWithStatusCode(
                    500, "HTTP status 500: Internal Server Error"
                )
            return "Success after retries"

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await llm._retry_with_backoff(
                func_with_500_then_success, operation_name="test_operation"
            )

        assert result == "Success after retries"
        assert call_count == 3
        # Should sleep twice with exponential backoff (2^0=1, 2^1=2)
        assert mock_sleep.call_count == 2
        assert mock_sleep.call_args_list[0][0][0] == 1
        assert mock_sleep.call_args_list[1][0][0] == 2

    @pytest.mark.asyncio
    async def test_retry_with_backoff_500_limited_to_3_retries(self):
        """Test that 500 status code is limited to 3 retries."""
        llm = ConcreteLLM(name="TestLLM", max_retries=10)

        call_count = 0

        async def func_always_500():
            nonlocal call_count
            call_count += 1
            raise ExceptionWithStatusCode(500, "HTTP status 500: Internal Server Error")

        with pytest.raises(RuntimeError) as exc_info:
            await llm._retry_with_backoff(
                func_always_500, operation_name="test_operation"
            )

        assert "after 3 retries" in str(exc_info.value)
        assert call_count == 3  # Limited to 3 retries for 500

    @pytest.mark.asyncio
    async def test_retry_with_backoff_502_bad_gateway(self):
        """Test retry logic for 502 (Bad Gateway) status code."""
        from unittest.mock import AsyncMock, patch

        llm = ConcreteLLM(name="TestLLM", max_retries=5)

        call_count = 0

        async def func_with_502_then_success():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ExceptionWithStatusCode(502, "HTTP status 502: Bad Gateway")
            return "Success after retry"

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await llm._retry_with_backoff(
                func_with_502_then_success, operation_name="test_operation"
            )

        assert result == "Success after retry"
        assert call_count == 2
        assert mock_sleep.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_with_backoff_502_limited_to_3_retries(self):
        """Test that 502 status code is limited to 3 retries."""
        llm = ConcreteLLM(name="TestLLM", max_retries=10)

        call_count = 0

        async def func_always_502():
            nonlocal call_count
            call_count += 1
            raise ExceptionWithStatusCode(502, "HTTP status 502: Bad Gateway")

        with pytest.raises(RuntimeError) as exc_info:
            await llm._retry_with_backoff(
                func_always_502, operation_name="test_operation"
            )

        assert "after 3 retries" in str(exc_info.value)
        assert call_count == 3  # Limited to 3 retries for 502

    @pytest.mark.asyncio
    async def test_retry_with_backoff_503_service_unavailable(self):
        """Test retry logic for 503 (Service Unavailable) status code."""
        from unittest.mock import AsyncMock, patch

        llm = ConcreteLLM(name="TestLLM", max_retries=4)

        call_count = 0

        async def func_with_503_then_success():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ExceptionWithStatusCode(
                    503, "HTTP status 503: Service Unavailable"
                )
            return "Success after retries"

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await llm._retry_with_backoff(
                func_with_503_then_success, operation_name="test_operation"
            )

        assert result == "Success after retries"
        assert call_count == 3
        # Should sleep twice with exponential backoff
        assert mock_sleep.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_with_backoff_504_gateway_timeout(self):
        """Test retry logic for 504 (Gateway Timeout) status code."""
        from unittest.mock import AsyncMock, patch

        llm = ConcreteLLM(name="TestLLM", max_retries=3)

        call_count = 0

        async def func_with_504_then_success():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ExceptionWithStatusCode(504, "HTTP status 504: Gateway Timeout")
            return "Success after retry"

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await llm._retry_with_backoff(
                func_with_504_then_success, operation_name="test_operation"
            )

        assert result == "Success after retry"
        assert call_count == 2
        assert mock_sleep.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_with_backoff_529_overloaded(self):
        """Test retry logic for 529 (Overloaded - Anthropic) status code."""
        from unittest.mock import AsyncMock, patch

        llm = ConcreteLLM(name="TestLLM", max_retries=3)

        call_count = 0

        async def func_with_529_then_success():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ExceptionWithStatusCode(529, "HTTP status 529: Overloaded")
            return "Success after retry"

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await llm._retry_with_backoff(
                func_with_529_then_success, operation_name="test_operation"
            )

        assert result == "Success after retry"
        assert call_count == 2
        assert mock_sleep.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_with_backoff_exponential_backoff_timing(self):
        """Test that exponential backoff timing is correct (2^attempt, max 60s)."""
        from unittest.mock import AsyncMock, patch

        llm = ConcreteLLM(name="TestLLM", max_retries=5)

        call_count = 0

        async def func_with_multiple_503():
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise ExceptionWithStatusCode(
                    503, "HTTP status 503: Service Unavailable"
                )
            return "Success"

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await llm._retry_with_backoff(
                func_with_multiple_503, operation_name="test_operation"
            )

        assert result == "Success"
        assert call_count == 4
        # Should sleep 3 times with exponential backoff: 2^0=1, 2^1=2, 2^2=4
        assert mock_sleep.call_count == 3
        assert mock_sleep.call_args_list[0][0][0] == 1
        assert mock_sleep.call_args_list[1][0][0] == 2
        assert mock_sleep.call_args_list[2][0][0] == 4

    @pytest.mark.asyncio
    async def test_retry_with_backoff_exponential_backoff_capped_at_60(self):
        """Test that exponential backoff is capped at 60 seconds."""
        from unittest.mock import AsyncMock, patch

        llm = ConcreteLLM(name="TestLLM", max_retries=10)

        call_count = 0

        async def func_with_many_503():
            nonlocal call_count
            call_count += 1
            if call_count <= 7:  # Need 7 attempts to reach 2^6 = 64 > 60
                raise ExceptionWithStatusCode(
                    503, "HTTP status 503: Service Unavailable"
                )
            return "Success"

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await llm._retry_with_backoff(
                func_with_many_503, operation_name="test_operation"
            )

        assert result == "Success"
        # Check that wait times are capped at 60
        wait_times = [call[0][0] for call in mock_sleep.call_args_list]
        assert all(wait <= 60 for wait in wait_times)
        # At attempt 6, 2^6 = 64, should be capped to 60
        assert 60 in wait_times

    @pytest.mark.asyncio
    async def test_retry_with_backoff_non_retryable_status_code(self):
        """Test that non-retryable status codes raise immediately."""
        llm = ConcreteLLM(name="TestLLM", max_retries=3)

        async def func_with_400():
            raise ExceptionWithStatusCode(400, "HTTP status 400: Bad Request")

        with pytest.raises(RuntimeError) as exc_info:
            await llm._retry_with_backoff(
                func_with_400, operation_name="test_operation"
            )

        assert "Error in test_operation" in str(exc_info.value)
        assert "400" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_retry_with_backoff_retryable_keyword_in_message(self):
        """Test retryable keywords are retried even without status code."""
        llm = ConcreteLLM(name="TestLLM", max_retries=3)

        call_count = 0

        async def func_with_retryable_message():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # No status code, but has retryable keyword
                raise Exception("Rate limit exceeded")
            return "Success"

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await llm._retry_with_backoff(
                func_with_retryable_message, operation_name="test_operation"
            )

        assert result == "Success"
        assert call_count == 2
        # Should treat as 503 and retry
        assert mock_sleep.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_with_backoff_non_retryable_error_message(self):
        """Test that errors without retryable keywords raise immediately."""
        llm = ConcreteLLM(name="TestLLM", max_retries=3)

        async def func_with_non_retryable_error():
            raise Exception("Invalid API key provided")

        with pytest.raises(RuntimeError) as exc_info:
            await llm._retry_with_backoff(
                func_with_non_retryable_error, operation_name="test_operation"
            )

        assert "Error in test_operation" in str(exc_info.value)
        assert "Invalid API key" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_retry_with_backoff_max_retries_exceeded(self):
        """Test that RuntimeError is raised when max retries are exceeded."""
        llm = ConcreteLLM(name="TestLLM", max_retries=2)

        call_count = 0

        async def func_always_503():
            nonlocal call_count
            call_count += 1
            raise ExceptionWithStatusCode(503, "HTTP status 503: Service Unavailable")

        with pytest.raises(RuntimeError) as exc_info:
            await llm._retry_with_backoff(
                func_always_503, operation_name="test_operation"
            )

        assert "after 2 retries" in str(exc_info.value)
        assert call_count == 2  # max_retries attempts
