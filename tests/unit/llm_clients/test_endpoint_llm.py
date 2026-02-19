"""Unit tests for EndpointLLM class."""

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_clients import Role
from llm_clients.endpoint_llm import EndpointLLM
from llm_clients.llm_interface import DEFAULT_START_PROMPT

from .test_base_llm import TestLLMBase
from .test_helpers import (
    assert_error_metadata,
    assert_error_response,
    assert_iso_timestamp,
    assert_metadata_copy_behavior,
    assert_metadata_structure,
    assert_response_timing,
)


def _make_aiohttp_mock(
    content: str = "Test response text",
    conversation_id: str | None = "server-cid-1",
    status: int = 200,
):
    """Build mock aiohttp ClientSession/post/response for EndpointLLM."""
    resp_mock = MagicMock()
    resp_mock.status = status
    resp_mock.json = AsyncMock(
        return_value={
            "message": {"content": content, "id": "msg-1"},
            "conversation_id": conversation_id,
            "model": "phi4",
        }
    )
    resp_mock.text = AsyncMock(return_value="")

    post_cm = MagicMock()
    post_cm.__aenter__ = AsyncMock(return_value=resp_mock)
    post_cm.__aexit__ = AsyncMock(return_value=None)

    session_mock = MagicMock()
    session_mock.post = MagicMock(return_value=post_cm)

    session_cm = MagicMock()
    session_cm.__aenter__ = AsyncMock(return_value=session_mock)
    session_cm.__aexit__ = AsyncMock(return_value=None)

    client_session_mock = MagicMock(return_value=session_cm)
    return client_session_mock


@pytest.mark.unit
@pytest.mark.usefixtures("mock_endpoint_config")
class TestEndpointLLM(TestLLMBase):
    """Unit tests for EndpointLLM.

    EndpointLLM implements LLMInterface only (no JudgeLLM); it uses aiohttp
    instead of an underlying .llm, so some base tests are overridden.
    """

    def create_llm(self, role: Role, **kwargs):
        if "name" not in kwargs:
            kwargs["name"] = "test-endpoint"
        return EndpointLLM(role=role, **kwargs)

    def get_provider_name(self) -> str:
        return "endpoint"

    @contextmanager
    def get_mock_patches(self):
        with patch(
            "llm_clients.endpoint_llm.aiohttp.ClientSession",
            new_callable=lambda: _make_aiohttp_mock(),
        ):
            yield

    # -------------------------------------------------------------------------
    # Overrides: generate_response uses aiohttp, not llm.llm
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_generate_response_returns_llm_text(
        self, mock_response_factory, mock_llm_factory, mock_system_message
    ):
        expected_text = "Test response text"
        with self.get_mock_patches():
            with patch(
                "llm_clients.endpoint_llm.aiohttp.ClientSession",
                new_callable=lambda: _make_aiohttp_mock(content=expected_text),
            ):
                llm = self.create_llm(role=Role.PROVIDER, name="TestLLM")
                response = await llm.generate_response(
                    conversation_history=mock_system_message
                )
        assert response == expected_text

    @pytest.mark.asyncio
    async def test_generate_response_updates_metadata(
        self, mock_response_factory, mock_llm_factory, mock_system_message
    ):
        with self.get_mock_patches():
            llm = self.create_llm(role=Role.PROVIDER, name="TestLLM")
            await llm.generate_response(conversation_history=mock_system_message)
        metadata = assert_metadata_structure(
            llm,
            expected_provider=self.get_provider_name(),
            expected_role=Role.PROVIDER,
        )
        assert "timestamp" in metadata
        assert_iso_timestamp(metadata["timestamp"])
        assert_response_timing(metadata)

    @pytest.mark.asyncio
    async def test_generate_response_handles_errors(
        self, mock_llm_factory, mock_system_message
    ):
        with self.get_mock_patches():
            with patch(
                "llm_clients.endpoint_llm.aiohttp.ClientSession"
            ) as mock_session_class:
                session_cm = MagicMock()
                session_cm.__aenter__ = AsyncMock(side_effect=Exception("API Error"))
                session_cm.__aexit__ = AsyncMock(return_value=None)
                mock_session_class.return_value = session_cm

                llm = self.create_llm(role=Role.PROVIDER, name="TestLLM")
                response = await llm.generate_response(
                    conversation_history=mock_system_message
                )

        assert_error_response(response, "API Error")
        assert_error_metadata(
            llm,
            expected_provider=self.get_provider_name(),
            expected_error_substring="API Error",
        )

    # -------------------------------------------------------------------------
    # Endpoint-specific tests
    # -------------------------------------------------------------------------

    def test_init_passes_first_message_and_start_prompt_to_super(self):
        with self.get_mock_patches():
            llm = EndpointLLM(
                name="ep",
                role=Role.PROVIDER,
                first_message="Hello",
                start_prompt="Custom start",
            )
        assert llm.first_message == "Hello"
        assert llm.start_prompt == "Custom start"

    def test_init_default_start_prompt(self):
        with self.get_mock_patches():
            llm = EndpointLLM(name="ep", role=Role.PROVIDER)
        assert llm.start_prompt == DEFAULT_START_PROMPT

    @pytest.mark.asyncio
    async def test_start_conversation_returns_first_message_when_set(self):
        with self.get_mock_patches():
            llm = EndpointLLM(
                name="ep",
                role=Role.PROVIDER,
                first_message="Static first reply",
            )
        out = await llm.start_conversation()
        assert out == "Static first reply"
        meta = llm.last_response_metadata
        assert meta.get("static_first_message") is True
        assert meta.get("provider") == "endpoint"

    @pytest.mark.asyncio
    async def test_start_conversation_calls_api_when_no_first_message(self):
        with self.get_mock_patches():
            with patch(
                "llm_clients.endpoint_llm.aiohttp.ClientSession",
                new_callable=lambda: _make_aiohttp_mock(content="First turn from API"),
            ) as mock_session_class:
                llm = EndpointLLM(name="ep", role=Role.PROVIDER)
                out = await llm.start_conversation()
        assert out == "First turn from API"
        mock_session_class.return_value.__aenter__.return_value.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_conversation_id_overwritten_when_endpoint_returns_different(
        self, mock_system_message
    ):
        """Endpoint response conversation_id overwrites client-generated id."""
        client_cid = "client-generated-cid"
        server_cid = "server-returned-cid"
        with self.get_mock_patches():
            with patch(
                "llm_clients.endpoint_llm.aiohttp.ClientSession",
                new_callable=lambda: _make_aiohttp_mock(
                    content="OK", conversation_id=server_cid
                ),
            ):
                llm = EndpointLLM(name="ep", role=Role.PROVIDER)
                llm.conversation_id = client_cid
                await llm.generate_response(conversation_history=mock_system_message)
        assert llm.conversation_id == server_cid

    @pytest.mark.asyncio
    async def test_generate_response_with_empty_conversation_history(self):
        """Verify start_conversation / default start_prompt with empty history."""
        with self.get_mock_patches():
            with patch(
                "llm_clients.endpoint_llm.aiohttp.ClientSession",
                new_callable=lambda: _make_aiohttp_mock(content="Delegated first turn"),
            ):
                llm = EndpointLLM(name="ep", role=Role.PROVIDER)
                out = await llm.generate_response(conversation_history=[])
        assert out == "Delegated first turn"

    @pytest.mark.asyncio
    async def test_generate_response_none_history_delegates_to_start_conversation(
        self,
    ):
        with self.get_mock_patches():
            with patch(
                "llm_clients.endpoint_llm.aiohttp.ClientSession",
                new_callable=lambda: _make_aiohttp_mock(content="Delegated from None"),
            ):
                llm = EndpointLLM(name="ep", role=Role.PROVIDER)
                out = await llm.generate_response(conversation_history=None)
        assert out == "Delegated from None"

    def test_set_system_prompt(self):
        with self.get_mock_patches():
            llm = self.create_llm(
                role=Role.PROVIDER, name="TestLLM", system_prompt="Initial"
            )
        assert llm.system_prompt == "Initial"
        llm.set_system_prompt("Updated")
        assert llm.system_prompt == "Updated"

    def test_getattr_returns_none_for_unknown_attribute(self):
        with self.get_mock_patches():
            llm = EndpointLLM(name="ep", role=Role.PROVIDER)
        assert llm.nonexistent_attr is None

    def test_temperature_and_max_tokens_accessible_from_self(self):
        with self.get_mock_patches():
            llm = EndpointLLM(
                name="ep",
                role=Role.PROVIDER,
                temperature=0.3,
                max_tokens=100,
            )
        assert llm.temperature == 0.3
        assert llm.max_tokens == 100

    def test_last_response_metadata_copy_returns_copy(self):
        with self.get_mock_patches():
            llm = self.create_llm(role=Role.PROVIDER, name="TestLLM")
            assert_metadata_copy_behavior(llm)
