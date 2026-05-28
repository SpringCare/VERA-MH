import uuid
from typing import Optional
from unittest.mock import MagicMock

import pytest

from llm_clients import Role
from llm_clients.llm_interface import LLMGenerationFailed, LLMInterface


class ConcreteLLM(LLMInterface):
    """Concrete implementation for testing abstract base class."""

    def __init__(self, name: str, role: Role, system_prompt: Optional[str] = None):
        super().__init__(name, role, system_prompt)
        # Add a mock llm object for __getattr__ testing
        self.llm = MagicMock(spec=["temperature", "max_tokens", "custom_method"])
        self.llm.temperature = 0.7
        self.llm.max_tokens = 1000

    async def start_conversation(self) -> str:
        """Concrete implementation of abstract method."""
        return "test response"

    async def generate_response(self, conversation_history=None):
        """Concrete implementation of abstract method."""
        return "test response"

    def set_system_prompt(self, system_prompt: str) -> None:
        """Concrete implementation of abstract method."""
        self.system_prompt = system_prompt


class IncompleteLLM(LLMInterface):
    """Incomplete implementation to test abstract method enforcement."""

    pass


class RetryHarness(LLMInterface):
    """Concrete LLM for testing _run_with_retry."""

    def __init__(self) -> None:
        super().__init__("harness", Role.PROVIDER)
        self._call_n = 0
        self.fail_count = 0

    async def start_conversation(self) -> str:
        return ""

    async def generate_response(self, conversation_history=None):
        return ""

    def set_system_prompt(self, system_prompt: str) -> None:
        pass

    async def flaky_ok_on_third(self) -> str:
        async def inner() -> str:
            self._call_n += 1
            if self._call_n < 3:
                raise RuntimeError("transient")
            return "done"

        return await self._run_with_retry(inner, provider="harness")

    async def always_fail(self) -> None:
        async def inner() -> str:
            self.fail_count += 1
            raise RuntimeError("always")

        await self._run_with_retry(inner, provider="harness")


@pytest.mark.unit
class TestErrorMatchesNoRetrySubstrings:
    """Tests for _error_matches_no_retry_substrings."""

    def test_true_when_single_substring_present(self):
        err = RuntimeError("Error 429: insufficient_quota")
        assert LLMInterface._error_matches_no_retry_substrings(
            err, ("insufficient_quota",)
        )

    def test_true_when_second_of_multiple_substrings_matches(self):
        err = ValueError("Your credit balance is too low")
        assert LLMInterface._error_matches_no_retry_substrings(
            err, ("insufficient_quota", "credit balance is too low")
        )

    def test_false_when_no_substring_matches(self):
        err = RuntimeError("timeout connecting to host")
        assert not LLMInterface._error_matches_no_retry_substrings(
            err, ("insufficient_quota", "invalid_api_key")
        )

    def test_false_when_substrings_tuple_empty(self):
        err = RuntimeError("insufficient_quota")
        assert not LLMInterface._error_matches_no_retry_substrings(err, ())

    def test_uses_str_of_exception(self):
        class CustomErr(Exception):
            def __str__(self):
                return "billing_hard_limit exceeded"

        assert LLMInterface._error_matches_no_retry_substrings(
            CustomErr(), ("billing_hard_limit",)
        )


@pytest.mark.unit
class TestLLMInterface:
    """Unit tests for LLMInterface abstract base class."""

    def test_init_with_name_only(self):
        """Test initialization with only name parameter."""
        llm = ConcreteLLM(name="TestLLM", role=Role.PROVIDER)

        assert llm.name == "TestLLM"
        assert llm.system_prompt == ""

    def test_init_with_name_and_system_prompt(self):
        """Test initialization with name and system prompt."""
        prompt = "You are a helpful assistant."
        llm = ConcreteLLM(name="TestLLM", role=Role.PROVIDER, system_prompt=prompt)

        assert llm.name == "TestLLM"
        assert llm.system_prompt == prompt

    @pytest.mark.asyncio
    async def test_generate_response_abstract_method(self, mock_system_message):
        """Test that generate_response is implemented in concrete class (line 21)."""
        llm = ConcreteLLM(name="TestLLM", role=Role.PROVIDER)
        response = await llm.generate_response(conversation_history=mock_system_message)

        assert response == "test response"

    def test_set_system_prompt_abstract_method(self):
        """Test that set_system_prompt is implemented in concrete class (line 26)."""
        llm = ConcreteLLM(
            name="TestLLM", role=Role.PROVIDER, system_prompt="Initial prompt"
        )
        assert llm.system_prompt == "Initial prompt"

        llm.set_system_prompt("Updated prompt")
        assert llm.system_prompt == "Updated prompt"

    def test_cannot_instantiate_abstract_class(self):
        """Test that LLMInterface cannot be instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            LLMInterface(name="Test", role=Role.PROVIDER)  # pyright: ignore[reportAbstractUsage]

        assert "Can't instantiate abstract class" in str(exc_info.value)

    def test_incomplete_implementation_raises_error(self):
        """Test that incomplete implementations raise TypeError."""
        with pytest.raises(TypeError) as exc_info:
            IncompleteLLM(name="Incomplete", role=Role.PROVIDER)  # pyright: ignore[reportAbstractUsage]

        assert "Can't instantiate abstract class" in str(exc_info.value)

    def test_getattr_delegates_to_llm(self):
        """Test that __getattr__ delegates to self.llm (lines 40-41)."""
        llm = ConcreteLLM(name="TestLLM", role=Role.PROVIDER)

        # Access attributes that exist on llm
        assert llm.temperature == 0.7
        assert llm.max_tokens == 1000

    def test_getattr_raises_attribute_error_for_missing_attribute(self):
        """Test that __getattr__ raises AttributeError for missing attributes."""
        llm = ConcreteLLM(name="TestLLM", role=Role.PROVIDER)

        # Try to access attribute that doesn't exist on llm (spec prevents it)
        with pytest.raises(AttributeError) as exc_info:
            _ = llm.nonexistent_attribute

        assert "ConcreteLLM" in str(exc_info.value)
        assert "nonexistent_attribute" in str(exc_info.value)

    def test_getattr_when_llm_not_set(self):
        """Test __getattr__ behavior when self.llm doesn't exist."""

        class MinimalLLM(LLMInterface):
            """Minimal implementation without self.llm."""

            async def start_conversation(self) -> str:
                return "response"

            async def generate_response(self, conversation_history=None):
                return "response"

            def set_system_prompt(self, system_prompt: str) -> None:
                self.system_prompt = system_prompt

        llm = MinimalLLM(name="Minimal", role=Role.PROVIDER)

        # Should raise AttributeError
        with pytest.raises(AttributeError):
            _ = llm.some_attribute

    def test_getattr_with_none_llm(self):
        """Test __getattr__ when self.llm is None."""

        class NullLLM(LLMInterface):
            """Implementation with None llm."""

            def __init__(
                self, name: str, role: Role, system_prompt: Optional[str] = None
            ):
                super().__init__(name, role, system_prompt)
                self.llm = None

            async def start_conversation(self) -> str:
                return "response"

            async def generate_response(self, conversation_history=None):
                return "response"

            def set_system_prompt(self, system_prompt: str) -> None:
                self.system_prompt = system_prompt

        llm = NullLLM(name="Null", role=Role.PROVIDER)

        # Should raise AttributeError since llm is None
        with pytest.raises(AttributeError):
            _ = llm.temperature

    def test_multiple_instances_have_independent_state(self):
        """Test that multiple LLM instances maintain independent state."""
        llm1 = ConcreteLLM(name="LLM1", role=Role.PROVIDER, system_prompt="Prompt 1")
        llm2 = ConcreteLLM(name="LLM2", role=Role.PROVIDER, system_prompt="Prompt 2")

        assert llm1.name == "LLM1"
        assert llm2.name == "LLM2"
        assert llm1.system_prompt == "Prompt 1"
        assert llm2.system_prompt == "Prompt 2"
        assert llm1.conversation_id != llm2.conversation_id

        # Modify one shouldn't affect the other
        llm1.set_system_prompt("Modified Prompt 1")
        assert llm1.system_prompt == "Modified Prompt 1"
        assert llm2.system_prompt == "Prompt 2"

    def test_getattr_with_callable_attribute(self):
        """Test __getattr__ works with callable attributes."""
        llm = ConcreteLLM(name="TestLLM", role=Role.PROVIDER)
        llm.llm.custom_method = MagicMock(return_value="method result")

        # Access callable attribute through delegation
        result = llm.custom_method()
        assert result == "method result"
        llm.llm.custom_method.assert_called_once()

    def test_system_prompt_default_empty_string(self):
        """Test that system_prompt defaults to empty string, not None."""
        llm = ConcreteLLM(name="TestLLM", role=Role.PROVIDER)
        assert llm.system_prompt == ""
        assert llm.system_prompt is not None

    def test_getattr_preserves_attribute_type(self):
        """Test that __getattr__ preserves the type of delegated attributes."""

        # Create a fresh mock without spec for this test
        class FlexibleLLM(LLMInterface):
            def __init__(
                self, name: str, role: Role, system_prompt: Optional[str] = None
            ):
                super().__init__(name, role, system_prompt)
                self.llm = MagicMock()
                self.llm.string_attr = "test string"
                self.llm.int_attr = 42
                self.llm.float_attr = 3.14
                self.llm.bool_attr = True
                self.llm.list_attr = [1, 2, 3]

            async def start_conversation(self) -> str:
                return "response"

            async def generate_response(self, conversation_history=None):
                return "response"

            def set_system_prompt(self, system_prompt: str) -> None:
                self.system_prompt = system_prompt

        llm = FlexibleLLM(name="TestLLM", role=Role.PROVIDER)

        assert isinstance(llm.string_attr, str)
        assert isinstance(llm.int_attr, int)
        assert isinstance(llm.float_attr, float)
        assert isinstance(llm.bool_attr, bool)
        assert isinstance(llm.list_attr, list)

    def test_init_sets_conversation_id(self):
        """Test that conversation_id is set at init (e.g. UUID)."""
        llm = ConcreteLLM(name="TestLLM", role=Role.PROVIDER)
        assert llm.conversation_id is not None
        assert isinstance(llm.conversation_id, str)
        assert len(llm.conversation_id) > 0

    def test_create_conversation_id_returns_string(self):
        """Test that create_conversation_id returns a non-empty string (e.g. UUID)."""
        llm = ConcreteLLM(name="TestLLM", role=Role.PROVIDER)
        cid = llm.create_conversation_id()
        assert isinstance(cid, str)
        assert len(cid) > 0

    def test_create_conversation_id_returns_distinct_valid_uuid(self):
        """Test that repeated calls return distinct values and each is a valid UUID."""
        llm = ConcreteLLM(name="TestLLM", role=Role.PROVIDER)
        ids = [llm.create_conversation_id() for _ in range(50)]
        assert len(ids) == len(set(ids)), "ids must be distinct"
        for cid in ids:
            uuid.UUID(cid)  # valid UUID string

    def test_update_conversation_id_from_metadata_leaves_unchanged_when_absent(self):
        """
        Test _update_conversation_id_from_metadata preserves conversation_id
        when key is absent from response metadata.
        """
        llm = ConcreteLLM(name="TestLLM", role=Role.PROVIDER)
        original = llm.conversation_id
        llm._last_response_metadata = {}
        llm._update_conversation_id_from_metadata()
        assert llm.conversation_id == original

    def test_update_conversation_id_from_metadata_overwrites_when_present(self):
        """
        Test _update_conversation_id_from_metadata overwrites self.conversation_id
        with API-returned conversation_id.
        """
        llm = ConcreteLLM(name="TestLLM", role=Role.PROVIDER)
        llm._last_response_metadata = {"conversation_id": "api-provided-id"}
        llm._update_conversation_id_from_metadata()
        assert llm.conversation_id == "api-provided-id"

    @pytest.mark.asyncio
    async def test_conversation_id_available_after_generate_response(self):
        """Test that conversation_id remains set after start_conversation."""
        llm = ConcreteLLM(name="TestLLM", role=Role.PROVIDER)
        await llm.start_conversation()
        assert llm.conversation_id is not None


@pytest.mark.unit
class TestLLMInterfaceHooks:
    """Tests for the extensibility hooks added to LLMInterface."""

    def test_custom_termination_signals_default_empty(self):
        """custom_termination_signals returns [] by default."""
        llm = ConcreteLLM("test", Role.PROVIDER)
        assert llm.custom_termination_signals == []

    def test_post_process_response_noop(self):
        """_post_process_response is a no-op by default."""
        llm = ConcreteLLM("test", Role.PROVIDER)
        assert llm._post_process_response("hello world") == "hello world"
        assert llm._post_process_response("") == ""

    def test_should_discard_response_default_false(self):
        """_should_discard_response returns False regardless of signals."""
        llm = ConcreteLLM("test", Role.PROVIDER)
        assert llm._should_discard_response([]) is False
        assert llm._should_discard_response(["SIGNAL"]) is False

    def test_extract_signals_returns_matching_signals(self):
        """_extract_signals returns only the signals found in the response."""

        class SignalLLM(ConcreteLLM):
            @property
            def custom_termination_signals(self):
                return ["[END]", "[ERROR]"]

        llm = SignalLLM("test", Role.PROVIDER)
        found = llm._extract_signals("Something happened [END] here")
        assert found == ["[END]"]

    def test_extract_signals_returns_empty_when_no_match(self):
        """_extract_signals returns [] when no signals are present."""

        class SignalLLM(ConcreteLLM):
            @property
            def custom_termination_signals(self):
                return ["[END]"]

        llm = SignalLLM("test", Role.PROVIDER)
        assert llm._extract_signals("nothing special here") == []

    def test_extract_signals_returns_multiple_matches(self):
        """_extract_signals returns all signals found in the response."""

        class SignalLLM(ConcreteLLM):
            @property
            def custom_termination_signals(self):
                return ["[END]", "[ERROR]"]

        llm = SignalLLM("test", Role.PROVIDER)
        found = llm._extract_signals("[ERROR] something went wrong [END]")
        assert "[END]" in found
        assert "[ERROR]" in found

    def test_extract_signals_case_insensitive(self):
        """_extract_signals is case-insensitive."""

        class SignalLLM(ConcreteLLM):
            @property
            def custom_termination_signals(self):
                return ["[end]"]

        llm = SignalLLM("test", Role.PROVIDER)
        assert llm._extract_signals("message [END] here") == ["[end]"]

    def test_first_speaker_default_none(self):
        """first_speaker returns None by default."""
        llm = ConcreteLLM("test", Role.PROVIDER)
        assert llm.first_speaker is None

    def test_prepare_sessions_returns_unchanged(self):
        """prepare_sessions returns the same list unmodified."""
        llm = ConcreteLLM("test", Role.PROVIDER)
        sessions = ["intake", "coaching"]
        result = llm.prepare_sessions(sessions)
        assert result == sessions

    @pytest.mark.asyncio
    async def test_setup_is_noop(self):
        """setup() completes without error by default."""
        llm = ConcreteLLM("test", Role.PROVIDER)
        await llm.setup()  # must not raise

    @pytest.mark.asyncio
    async def test_enter_session_is_noop(self):
        """enter_session() completes without error by default."""
        llm = ConcreteLLM("test", Role.PROVIDER)
        await llm.enter_session("intake")  # must not raise

    @pytest.mark.asyncio
    async def test_run_with_retry_succeeds_after_transient_failures(self):
        h = RetryHarness()
        assert await h.flaky_ok_on_third() == "done"
        assert h._call_n == 3

    @pytest.mark.asyncio
    async def test_run_with_retry_raises_after_max_attempts(self):
        h = RetryHarness()
        with pytest.raises(LLMGenerationFailed) as exc:
            await h.always_fail()
        assert "always" in str(exc.value)
        assert exc.value.__cause__ is not None
        assert h.fail_count == 4

    @pytest.mark.asyncio
    async def test_run_with_retry_no_retry_substrings_wraps_without_extra_attempts(
        self,
    ):
        h = RetryHarness()
        attempts = 0

        async def inner() -> str:
            nonlocal attempts
            attempts += 1
            raise RuntimeError("stop: insufficient_quota exceeded")

        with pytest.raises(LLMGenerationFailed) as exc:
            await h._run_with_retry(
                inner,
                no_retry_substrings=("insufficient_quota",),
            )
        assert attempts == 1
        assert "insufficient_quota" in str(exc.value)
        assert exc.value.__cause__ is not None
        assert "insufficient_quota" in str(exc.value.__cause__)

    @pytest.mark.asyncio
    async def test_run_with_retry_llm_generation_failed_propagates(self):
        h = RetryHarness()

        async def inner() -> str:
            raise LLMGenerationFailed("fatal")

        with pytest.raises(LLMGenerationFailed) as exc:
            await h._run_with_retry(inner)
        assert str(exc.value) == "fatal"

    @pytest.mark.asyncio
    async def test_run_with_retry_clears_stale_metadata_on_failure(self):
        h = RetryHarness()
        h.last_response_metadata = {"provider": "stale", "response_id": "old"}

        with pytest.raises(LLMGenerationFailed):
            await h.always_fail()

        metadata = h.last_response_metadata
        assert metadata["provider"] == "harness"
        assert metadata["error"] == "always"
        assert metadata["attempt"] == 4
        assert metadata["max_attempts"] == 4
        assert metadata["retryable"] is True
        assert metadata["will_retry"] is False
        assert metadata.get("response_id") is None

    @pytest.mark.asyncio
    async def test_run_with_retry_on_error_callback_updates_metadata(self):
        h = RetryHarness()

        async def inner() -> str:
            raise RuntimeError("boom")

        def _on_error(
            err: BaseException,
            attempt: int,
            max_attempts: int,
            retryable: bool,
            will_retry: bool,
        ) -> dict[str, object]:
            return {
                "provider_error_type": type(err).__name__,
                "observed_attempt": attempt,
                "attempts_total": max_attempts,
                "observed_retryable": retryable,
                "observed_will_retry": will_retry,
            }

        with pytest.raises(LLMGenerationFailed):
            await h._run_with_retry(inner, provider="harness", on_error=_on_error)

        metadata = h.last_response_metadata
        assert metadata["provider"] == "harness"
        assert metadata["provider_error_type"] == "RuntimeError"
        assert metadata["observed_attempt"] == 4
        assert metadata["attempts_total"] == 4
        assert metadata["observed_retryable"] is True
        assert metadata["observed_will_retry"] is False

    @pytest.mark.asyncio
    async def test_run_with_retry_waits_between_retryable_attempts(self, monkeypatch):
        h = RetryHarness()
        delays = [0.11, 0.22, 0.33]
        observed_sleep_calls: list[float] = []
        observed_metadata_delay_values: list[float] = []

        def fake_compute(self, _attempt_number: int) -> float:
            return delays[len(observed_sleep_calls)]

        async def fake_sleep(delay: float) -> None:
            observed_metadata_delay_values.append(
                h.last_response_metadata["retry_delay_seconds"]
            )
            observed_sleep_calls.append(delay)

        monkeypatch.setattr(LLMInterface, "_compute_retry_delay_seconds", fake_compute)
        monkeypatch.setattr("llm_clients.llm_interface.asyncio.sleep", fake_sleep)

        with pytest.raises(LLMGenerationFailed):
            await h.always_fail()

        assert observed_sleep_calls == delays
        assert observed_metadata_delay_values == [round(d, 3) for d in delays]
        metadata = h.last_response_metadata
        assert metadata["attempt"] == 4
        assert metadata["will_retry"] is False
