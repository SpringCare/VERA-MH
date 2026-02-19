"""Unit tests for conversation utility functions."""

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from llm_clients.llm_interface import Role
from utils.conversation_utils import (
    add_timestamp_to_path,
    build_langchain_messages,
    ensure_provider_has_last_turn,
    format_conversation_as_string,
    format_conversation_summary,
)


class TestEnsureProviderHasLastTurn:
    """Test ensure_provider_has_last_turn."""

    def test_persona_first_even_unchanged(self) -> None:
        assert ensure_provider_has_last_turn(4, persona_speaks_first=True) == 4
        assert ensure_provider_has_last_turn(6, persona_speaks_first=True) == 6

    def test_persona_first_odd_bumped(self) -> None:
        assert ensure_provider_has_last_turn(3, persona_speaks_first=True) == 4
        assert ensure_provider_has_last_turn(5, persona_speaks_first=True) == 6

    def test_agent_first_odd_unchanged(self) -> None:
        assert ensure_provider_has_last_turn(3, persona_speaks_first=False) == 3
        assert ensure_provider_has_last_turn(5, persona_speaks_first=False) == 5

    def test_agent_first_even_bumped(self) -> None:
        assert ensure_provider_has_last_turn(2, persona_speaks_first=False) == 3
        assert ensure_provider_has_last_turn(4, persona_speaks_first=False) == 5


class TestFormatConversationSummary:
    """Test format_conversation_summary function."""

    def test_invalid_role_value_raises(self) -> None:
        """Invalid speaker role value raises ValueError with turn index."""
        conversation = [
            {"speaker": "persona", "response": "Hello"},
            {"speaker": "invalid_role", "response": "Bad turn"},
        ]
        with pytest.raises(ValueError) as exc_info:
            format_conversation_summary(conversation)
        assert "Invalid role value 'invalid_role' for turn 1" in str(exc_info.value)

    def test_missing_speaker_raises(self) -> None:
        """Missing speaker (None) raises ValueError."""
        conversation = [
            {"speaker": "persona", "response": "Hello"},
            {"response": "No speaker"},
        ]
        with pytest.raises(ValueError) as exc_info:
            format_conversation_summary(conversation)
        assert "Invalid role value 'None' for turn 1" in str(exc_info.value)


class TestBuildLangchainMessages:
    """Test build_langchain_messages function."""

    def test_build_messages_with_no_history(self, mock_system_message):
        """Test with only current message, no history."""
        messages = build_langchain_messages(
            role=Role.PROVIDER,
            conversation_history=mock_system_message,
        )

        assert len(messages) == 1
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].text == "Test"

    def test_build_messages_with_empty_history(self, mock_system_message):
        """Test with empty history list."""
        messages = build_langchain_messages(
            role=Role.PROVIDER,
            conversation_history=mock_system_message,
        )

        assert len(messages) == 1
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].text == "Test"

    def test_build_messages_with_role_enum_values(self):
        """Test that speaker field uses Role enum values correctly."""
        history = [
            {
                "turn": 1,
                "speaker": "persona",
                "input": "Start",
                "response": "Hi, I'm Alice",
            },
            {
                "turn": 2,
                "speaker": "provider",
                "input": "Hi, I'm Alice",
                "response": "Hello Alice, how are you?",
            },
            {
                "turn": 3,
                "speaker": "persona",
                "input": "Hello Alice, how are you?",
                "response": "I'm doing well",
            },
        ]

        # Add turn 4 for the next message
        history.append(
            {
                "turn": 4,
                "speaker": "provider",
                "input": "I'm doing well",
                "response": "How can I help?",
            }
        )

        messages = build_langchain_messages(
            role=Role.PROVIDER, conversation_history=history
        )

        # Should have 4 history messages
        assert len(messages) == 4

        # Turn 1 (odd) should be HumanMessage
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].text == "Hi, I'm Alice"

        # Turn 2 (even) should be AIMessage
        assert isinstance(messages[1], AIMessage)
        assert messages[1].text == "Hello Alice, how are you?"

        # Turn 3 (odd) should be HumanMessage
        assert isinstance(messages[2], HumanMessage)
        assert messages[2].text == "I'm doing well"

        # Turn 4 (even) should be AIMessage
        assert isinstance(messages[3], AIMessage)
        assert messages[3].text == "How can I help?"

    def test_build_messages_with_standard_speaker_names(self):
        """Test that standard speaker names still work correctly."""
        history = [
            {
                "turn": 1,
                "speaker": "persona",
                "input": "Start",
                "response": "Hello",
            },
            {
                "turn": 2,
                "speaker": "provider",
                "input": "Hello",
                "response": "Hi there",
            },
            {
                "turn": 3,
                "speaker": "persona",
                "input": "Hi there",
                "response": "How are you?",
            },
        ]

        messages = build_langchain_messages(
            role=Role.PROVIDER, conversation_history=history
        )

        assert len(messages) == 3
        assert isinstance(messages[0], HumanMessage)
        assert isinstance(messages[1], AIMessage)
        assert isinstance(messages[2], HumanMessage)

    def test_build_messages_long_conversation(self):
        """Test with longer conversation to verify turn alternation."""
        history = [
            {
                "turn": i + 1,
                "speaker": "persona" if i % 2 == 0 else "provider",
                "response": f"Message {i + 1}",
            }
            for i in range(10)
        ]

        messages = build_langchain_messages(
            role=Role.PROVIDER, conversation_history=history
        )

        assert len(messages) == 10

        # Verify alternating pattern
        for i, msg in enumerate(messages):
            turn_number = i + 1
            if turn_number % 2 == 1:
                assert isinstance(msg, HumanMessage)
            else:
                assert isinstance(msg, AIMessage)
            assert msg.text == f"Message {turn_number}"

    def test_build_messages_no_current_message(self):
        """Test with history but no current message."""
        history = [
            {"turn": 1, "speaker": "persona", "response": "Hello"},
            {"turn": 2, "speaker": "provider", "response": "Hi"},
        ]

        messages = build_langchain_messages(
            role=Role.PROVIDER, conversation_history=history
        )

        assert len(messages) == 2
        assert isinstance(messages[0], HumanMessage)
        assert isinstance(messages[1], AIMessage)

    def test_build_messages_empty_inputs(self):
        """Test with all empty inputs."""
        messages = build_langchain_messages(
            role=Role.PERSONA, conversation_history=None
        )

        assert len(messages) == 0

    def test_build_messages_preserves_content(self):
        """Test that message content is preserved exactly."""
        multiline_text = "Line 1\nLine 2\nLine 3"
        unicode_text = "Hello 🌍 世界"

        history = [
            {
                "turn": 1,
                "speaker": "persona",
                "response": multiline_text,
            },
            {
                "turn": 2,
                "speaker": "provider",
                "response": unicode_text,
            },
        ]

        messages = build_langchain_messages(
            role=Role.PERSONA,
            conversation_history=history,
        )

        assert messages[0].text == multiline_text
        assert messages[1].text == unicode_text

    def test_build_messages_skips_none_response(self):
        """Test that turns with None response are skipped."""
        history = [
            {"turn": 1, "speaker": "persona", "response": "Hello"},
            {
                "turn": 2,
                "speaker": "provider",
                "response": None,
            },  # Should be skipped
            {
                "turn": 3,
                "speaker": "persona",
                "response": "Are you there?",
            },
        ]

        messages = build_langchain_messages(
            role=Role.PROVIDER, conversation_history=history
        )

        # Should only have 2 messages (turn 2 skipped)
        assert len(messages) == 2
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].text == "Hello"
        assert isinstance(messages[1], HumanMessage)
        assert messages[1].text == "Are you there?"

    def test_build_messages_with_persona_role_flips_types(self):
        """Test that persona role uses role-based message types correctly."""
        # When LLM is playing persona, persona messages should be AIMessage
        # and provider messages should be HumanMessage
        history = [
            {
                "turn": 1,
                "speaker": "persona",
                "response": "Hello, I need help",
            },
            {
                "turn": 2,
                "speaker": "provider",
                "response": "How can I help you?",
            },
            {
                "turn": 3,
                "speaker": "persona",
                "response": "I'm feeling anxious",
            },
            {
                "turn": 4,
                "speaker": "provider",
                "response": "Tell me more",
            },
        ]

        messages = build_langchain_messages(
            role=Role.PERSONA,
            conversation_history=history,
        )

        assert len(messages) == 4  # 4 history messages
        # Persona's own messages should be AIMessage (what "I" said)
        assert isinstance(messages[0], AIMessage)
        assert messages[0].text == "Hello, I need help"
        # Provider's messages should be HumanMessage (what "they" said)
        assert isinstance(messages[1], HumanMessage)
        assert messages[1].text == "How can I help you?"
        assert isinstance(messages[2], AIMessage)
        assert messages[2].text == "I'm feeling anxious"
        assert isinstance(messages[3], HumanMessage)
        assert messages[3].text == "Tell me more"

    def test_build_messages_without_persona_role_keeps_default(self):
        """Test that provider role uses role-based message types correctly."""
        # When LLM is playing provider, provider messages should be AIMessage
        # and persona messages should be HumanMessage
        history = [
            {"turn": 1, "speaker": "persona", "response": "Hello"},
            {
                "turn": 2,
                "speaker": "provider",
                "response": "Hi there",
            },
            {
                "turn": 3,
                "speaker": "persona",
                "response": "How are you?",
            },
        ]

        messages = build_langchain_messages(
            role=Role.PROVIDER,
            conversation_history=history,
        )

        assert len(messages) == 3
        # Persona's messages should be HumanMessage (what "they" said)
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].text == "Hello"
        # Provider's own messages should be AIMessage (what "I" said)
        assert isinstance(messages[1], AIMessage)
        assert messages[1].text == "Hi there"
        assert isinstance(messages[2], HumanMessage)
        assert messages[2].text == "How are you?"

    def test_build_messages_persona_with_turn_0(self):
        """Test persona role with turn 0 (initial message)."""
        history = [
            {"turn": 0, "response": "Initial message"},
            {"turn": 1, "speaker": "persona", "response": "Hello"},
            {"turn": 2, "speaker": "provider", "response": "Hi"},
        ]

        messages = build_langchain_messages(
            role=Role.PERSONA,
            conversation_history=history,
        )

        assert len(messages) == 3  # turn 0 + 2 history messages
        # Turn 0 should always be HumanMessage regardless of role
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].text == "Initial message"
        # Turn 1 (persona) should be AIMessage when persona role
        assert isinstance(messages[1], AIMessage)
        assert messages[1].text == "Hello"
        # Turn 2 (provider) should be HumanMessage when persona role
        assert isinstance(messages[2], HumanMessage)
        assert messages[2].text == "Hi"

    def test_build_messages_provider_starts_conversation_for_provider_role(self):
        """Test for when PROVIDER starts, messages build correctly for provider role."""
        # Provider starts first (turn 1), then persona responds (turn 2)
        history = [
            {
                "turn": 1,
                "speaker": "provider",
                "response": "Hello, how can I help you today?",
            },
            {
                "turn": 2,
                "speaker": "persona",
                "response": "I'm feeling really anxious",
            },
            {
                "turn": 3,
                "speaker": "provider",
                "response": "I understand. Can you tell me more?",
            },
            {
                "turn": 4,
                "speaker": "persona",
                "response": "It's been happening for weeks",
            },
        ]

        messages = build_langchain_messages(
            role=Role.PROVIDER,
            conversation_history=history,
        )

        assert len(messages) == 4
        # Provider's own messages should be AIMessage (what "I" said)
        assert isinstance(messages[0], AIMessage)
        assert messages[0].text == "Hello, how can I help you today?"
        # Persona's messages should be HumanMessage (what "they" said)
        assert isinstance(messages[1], HumanMessage)
        assert messages[1].text == "I'm feeling really anxious"
        assert isinstance(messages[2], AIMessage)
        assert messages[2].text == "I understand. Can you tell me more?"
        assert isinstance(messages[3], HumanMessage)
        assert messages[3].text == "It's been happening for weeks"

    def test_build_messages_provider_starts_conversation_for_persona_role(self):
        """Test for when PROVIDER starts, messages build correctly for persona role."""
        # Provider starts first (turn 1), then persona responds (turn 2)
        history = [
            {
                "turn": 1,
                "speaker": "provider",
                "response": "Hello, how can I help you today?",
            },
            {
                "turn": 2,
                "speaker": "persona",
                "response": "I'm feeling really anxious",
            },
            {
                "turn": 3,
                "speaker": "provider",
                "response": "I understand. Can you tell me more?",
            },
            {
                "turn": 4,
                "speaker": "persona",
                "response": "It's been happening for weeks",
            },
        ]

        messages = build_langchain_messages(
            role=Role.PERSONA,
            conversation_history=history,
        )

        assert len(messages) == 4
        # Provider's messages should be HumanMessage (what "they" said)
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].text == "Hello, how can I help you today?"
        # Persona's own messages should be AIMessage (what "I" said)
        assert isinstance(messages[1], AIMessage)
        assert messages[1].text == "I'm feeling really anxious"
        assert isinstance(messages[2], HumanMessage)
        assert messages[2].text == "I understand. Can you tell me more?"
        assert isinstance(messages[3], AIMessage)
        assert messages[3].text == "It's been happening for weeks"

    def test_build_messages_provider_starts_with_turn_0(self):
        """Test provider role with turn 0 when provider starts the conversation."""
        history = [
            {"turn": 0, "response": "Initial message"},
            {
                "turn": 1,
                "speaker": "provider",
                "response": "Hello, how can I help?",
            },
            {
                "turn": 2,
                "speaker": "persona",
                "response": "I need support",
            },
        ]

        messages = build_langchain_messages(
            role=Role.PROVIDER,
            conversation_history=history,
        )

        assert len(messages) == 3  # turn 0 + 2 history messages
        # Turn 0 should always be HumanMessage regardless of role
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].text == "Initial message"
        # Turn 1 (provider) should be AIMessage when provider role
        assert isinstance(messages[1], AIMessage)
        assert messages[1].text == "Hello, how can I help?"
        # Turn 2 (persona) should be HumanMessage when provider role
        assert isinstance(messages[2], HumanMessage)
        assert messages[2].text == "I need support"

    def test_build_messages_provider_starts_with_turn_0_for_persona_role(self):
        """Test persona role with turn 0 when provider starts the conversation."""
        history = [
            {"turn": 0, "response": "Initial message"},
            {
                "turn": 1,
                "speaker": "provider",
                "response": "Hello, how can I help?",
            },
            {
                "turn": 2,
                "speaker": "persona",
                "response": "I need support",
            },
        ]

        messages = build_langchain_messages(
            role=Role.PERSONA,
            conversation_history=history,
        )

        assert len(messages) == 3  # turn 0 + 2 history messages
        # Turn 0 should always be HumanMessage regardless of role
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].text == "Initial message"
        # Turn 1 (provider) should be HumanMessage when persona role
        assert isinstance(messages[1], HumanMessage)
        assert messages[1].text == "Hello, how can I help?"
        # Turn 2 (persona) should be AIMessage when persona role
        assert isinstance(messages[2], AIMessage)
        assert messages[2].text == "I need support"


class TestFormatConversationAsString:
    """Test format_conversation_as_string function."""

    def test_format_with_no_history(self, mock_system_message):
        """Test with only current message, no history."""
        result = format_conversation_as_string(
            role=Role.PROVIDER,
            conversation_history=mock_system_message,
        )

        assert result == "Human: Test\n\nAssistant:"

    def test_format_with_system_prompt(self, mock_system_message):
        """Test with system prompt."""
        result = format_conversation_as_string(
            role=Role.PERSONA,
            conversation_history=mock_system_message,
            system_prompt="You are helpful",
        )

        assert result == "System: You are helpful\n\nHuman: Test\n\nAssistant:"

    def test_format_with_conversation_history(self):
        """Test with conversation history."""
        history = [
            {"turn": 1, "speaker": Role.PERSONA, "response": "Hi"},
            {"turn": 2, "speaker": Role.PROVIDER, "response": "Hello"},
            {
                "turn": 3,
                "speaker": Role.PERSONA,
                "response": "How are you?",
            },
            {
                "turn": 4,
                "speaker": Role.PROVIDER,
                "response": "What's your name?",
            },
        ]

        result = format_conversation_as_string(
            role=Role.PROVIDER, conversation_history=history
        )

        expected = (
            "Human: Hi\n\n"
            "Assistant: Hello\n\n"
            "Human: How are you?\n\n"
            "Assistant: What's your name?\n\n"
            "Assistant:"
        )
        assert result == expected

    def test_format_with_system_prompt_and_history(self):
        """Test with both system prompt and history."""
        history = [
            {
                "turn": 1,
                "speaker": Role.PERSONA,
                "response": "Hello",
            },
            {
                "turn": 2,
                "speaker": Role.PROVIDER,
                "response": "Hi there",
            },
            {
                "turn": 3,
                "speaker": Role.PERSONA,
                "response": "Tell me more",
            },
        ]

        result = format_conversation_as_string(
            role=Role.PROVIDER, conversation_history=history, system_prompt="Be concise"
        )

        expected = (
            "System: Be concise\n\n"
            "Human: Hello\n\n"
            "Assistant: Hi there\n\n"
            "Human: Tell me more\n\n"
            "Assistant:"
        )
        assert result == expected

    def test_format_without_current_message(self):
        """Test with history but no current message."""
        history = [
            {"turn": 1, "speaker": Role.PERSONA, "response": "Hello"},
            {"turn": 2, "speaker": Role.PROVIDER, "response": "Hi"},
        ]

        result = format_conversation_as_string(
            role=Role.PROVIDER, conversation_history=history
        )

        expected = "Human: Hello\n\nAssistant: Hi\n\nAssistant:"
        assert result == expected

    def test_format_skips_none_response(self):
        """Test that turns with None response are skipped."""
        history = [
            {"turn": 1, "speaker": Role.PERSONA, "response": "Hello"},
            {"turn": 2, "speaker": Role.PROVIDER, "response": None},
            {
                "turn": 3,
                "speaker": Role.PERSONA,
                "response": "Still there?",
            },
        ]

        result = format_conversation_as_string(
            role=Role.PROVIDER, conversation_history=history
        )

        # Turn 2 should be skipped
        expected = "Human: Hello\n\nHuman: Still there?\n\nAssistant:"
        assert result == expected

    def test_format_empty_inputs(self):
        """Test with all empty inputs."""
        result = format_conversation_as_string(
            role=Role.PERSONA, conversation_history=None, system_prompt=None
        )

        assert result == ""

    def test_format_with_persona_role_flips_types(self):
        """Test that persona role uses role-based message types in string format."""
        persona_system_prompt = "You are roleplaying as a human user"
        history = [
            {"turn": 1, "speaker": Role.PERSONA, "response": "Hello"},
            {
                "turn": 2,
                "speaker": Role.PROVIDER,
                "response": "Hi there",
            },
            {
                "turn": 3,
                "speaker": Role.PERSONA,
                "response": "How are you?",
            },
        ]

        result = format_conversation_as_string(
            role=Role.PERSONA,
            conversation_history=history,
            system_prompt=persona_system_prompt,
        )

        # When persona role, persona messages should be Assistant (what "I" said),
        # provider messages should be Human (what "they" said)
        expected = (
            "System: You are roleplaying as a human user\n\n"
            "Assistant: Hello\n\n"
            "Human: Hi there\n\n"
            "Assistant: How are you?\n\n"
            "Assistant:"
        )
        assert result == expected


class TestAddTimestampToPath:
    """Test add_timestamp_to_path function."""

    def test_timestamp_format(self):
        """Test that timestamp uses correct format (YYYYMMDD_HHMMSS)."""
        path = Path("test.csv")
        result = add_timestamp_to_path(path)

        # Extract timestamp from stem
        # Result format: test_YYYYMMDD_HHMMSS
        stem_parts = result.stem.split("_")
        assert len(stem_parts) == 3
        assert stem_parts[0] == "test"
        date_part = stem_parts[1]
        time_part = stem_parts[2]

        # Verify format: YYYYMMDD_HHMMSS
        assert len(date_part) == 8  # YYYYMMDD
        assert len(time_part) == 6  # HHMMSS
        assert date_part.isdigit()
        assert time_part.isdigit()

    def test_preserves_parent_directory_and_suffix(self):
        """Test that parent directory is preserved."""
        path = Path("folder/subfolder/file.txt")
        result = add_timestamp_to_path(path)

        assert result.parent == path.parent
        assert result.name.startswith("file_")
        assert result.suffix == ".txt"
