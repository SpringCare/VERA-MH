"""Unit tests for conversation utility functions."""

from langchain_core.messages import AIMessage, HumanMessage

from utils.conversation_utils import (
    build_langchain_messages,
    format_conversation_as_string,
)


class TestBuildLangchainMessages:
    """Test build_langchain_messages function."""

    def test_build_messages_with_no_history(self):
        """Test with only current message, no history."""
        messages = build_langchain_messages(
            conversation_history=[{"turn": 0, "speaker": "system", "response": "Hello"}]
        )

        assert len(messages) == 1
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].content == "Hello"

    def test_build_messages_with_empty_history(self):
        """Test with empty history list."""
        messages = build_langchain_messages(
            conversation_history=[{"turn": 0, "speaker": "system", "response": "Hello"}]
        )

        assert len(messages) == 1
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].content == "Hello"

    def test_build_messages_with_custom_speaker_names(self):
        """Test that custom speaker names work correctly using turn indices."""
        # This is the critical test for the bug fix
        history = [
            {
                "turn": 1,
                "speaker": "g gpt4o Alice",  # Custom persona name
                "input": "Start",
                "response": "Hi, I'm Alice",
            },
            {
                "turn": 2,
                "speaker": "therapist-bot",  # Custom agent name
                "input": "Hi, I'm Alice",
                "response": "Hello Alice, how are you?",
            },
            {
                "turn": 3,
                "speaker": "g gpt4o Alice",  # Custom persona name
                "input": "Hello Alice, how are you?",
                "response": "I'm doing well",
            },
        ]

        # Add turn 4 for the next message
        history.append(
            {
                "turn": 4,
                "speaker": "therapist-bot",
                "input": "I'm doing well",
                "response": "How can I help?",
            }
        )

        messages = build_langchain_messages(conversation_history=history)

        # Should have 4 history messages
        assert len(messages) == 4

        # Turn 1 (odd) should be HumanMessage
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].content == "Hi, I'm Alice"

        # Turn 2 (even) should be AIMessage
        assert isinstance(messages[1], AIMessage)
        assert messages[1].content == "Hello Alice, how are you?"

        # Turn 3 (odd) should be HumanMessage
        assert isinstance(messages[2], HumanMessage)
        assert messages[2].content == "I'm doing well"

        # Turn 4 (even) should be AIMessage
        assert isinstance(messages[3], AIMessage)
        assert messages[3].content == "How can I help?"

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
                "speaker": "chatbot",
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

        messages = build_langchain_messages(conversation_history=history)

        assert len(messages) == 3
        assert isinstance(messages[0], HumanMessage)
        assert isinstance(messages[1], AIMessage)
        assert isinstance(messages[2], HumanMessage)

    def test_build_messages_long_conversation(self):
        """Test with longer conversation to verify turn alternation."""
        history = [
            {"turn": i + 1, "speaker": f"speaker_{i}", "response": f"Message {i + 1}"}
            for i in range(10)
        ]

        messages = build_langchain_messages(conversation_history=history)

        assert len(messages) == 10

        # Verify alternating pattern
        for i, msg in enumerate(messages):
            turn_number = i + 1
            if turn_number % 2 == 1:  # Odd turns
                assert isinstance(msg, HumanMessage)
            else:  # Even turns
                assert isinstance(msg, AIMessage)
            assert msg.content == f"Message {turn_number}"

    def test_build_messages_no_current_message(self):
        """Test with history but no current message."""
        history = [
            {"turn": 1, "speaker": "persona", "response": "Hello"},
            {"turn": 2, "speaker": "agent", "response": "Hi"},
        ]

        messages = build_langchain_messages(conversation_history=history)

        assert len(messages) == 2
        assert isinstance(messages[0], HumanMessage)
        assert isinstance(messages[1], AIMessage)

    def test_build_messages_empty_inputs(self):
        """Test with all empty inputs."""
        messages = build_langchain_messages(conversation_history=None)

        assert len(messages) == 0

    def test_build_messages_preserves_content(self):
        """Test that message content is preserved exactly."""
        multiline_text = "Line 1\nLine 2\nLine 3"
        unicode_text = "Hello 🌍 世界"

        history = [
            {"turn": 1, "speaker": "custom_persona", "response": multiline_text},
            {"turn": 2, "speaker": "custom_agent", "response": unicode_text},
        ]

        messages = build_langchain_messages(conversation_history=history)

        assert messages[0].content == multiline_text
        assert messages[1].content == unicode_text

    def test_build_messages_skips_none_response(self):
        """Test that turns with None response are skipped."""
        history = [
            {"turn": 1, "speaker": "persona", "response": "Hello"},
            {"turn": 2, "speaker": "agent", "response": None},  # Should be skipped
            {"turn": 3, "speaker": "persona", "response": "Are you there?"},
        ]

        messages = build_langchain_messages(conversation_history=history)

        # Should only have 2 messages (turn 2 skipped)
        assert len(messages) == 2
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].content == "Hello"
        assert isinstance(messages[1], HumanMessage)
        assert messages[1].content == "Are you there?"

    def test_build_messages_with_persona_role_flips_types(self):
        """Test that persona role flips message types correctly."""
        # When LLM is playing persona, persona turns should be AIMessage
        # and provider turns should be HumanMessage
        persona_system_prompt = "You are roleplaying as a human user"
        history = [
            {"turn": 1, "speaker": "persona", "response": "Hello, I need help"},
            {"turn": 2, "speaker": "provider", "response": "How can I help you?"},
            {"turn": 3, "speaker": "persona", "response": "I'm feeling anxious"},
            {"turn": 4, "speaker": "provider", "response": "Tell me more"},
        ]

        messages = build_langchain_messages(
            conversation_history=history, system_prompt=persona_system_prompt
        )

        assert len(messages) == 4
        # Turn 1 (odd, persona) should be AIMessage when persona role
        assert isinstance(messages[0], AIMessage)
        assert messages[0].content == "Hello, I need help"
        # Turn 2 (even, provider) should be HumanMessage when persona role
        assert isinstance(messages[1], HumanMessage)
        assert messages[1].content == "How can I help you?"
        # Turn 3 (odd, persona) should be AIMessage when persona role
        assert isinstance(messages[2], AIMessage)
        assert messages[2].content == "I'm feeling anxious"
        # Turn 4 (even, provider) should be HumanMessage when persona role
        assert isinstance(messages[3], HumanMessage)
        assert messages[3].content == "Tell me more"

    def test_build_messages_without_persona_role_keeps_default(self):
        """Test that non-persona role keeps default message types."""
        # When LLM is NOT playing persona, default behavior applies
        provider_system_prompt = "You are a helpful therapist"
        history = [
            {"turn": 1, "speaker": "persona", "response": "Hello"},
            {"turn": 2, "speaker": "provider", "response": "Hi there"},
            {"turn": 3, "speaker": "persona", "response": "How are you?"},
        ]

        messages = build_langchain_messages(
            conversation_history=history, system_prompt=provider_system_prompt
        )

        assert len(messages) == 3
        # Turn 1 (odd, persona) should be HumanMessage (default)
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].content == "Hello"
        # Turn 2 (even, provider) should be AIMessage (default)
        assert isinstance(messages[1], AIMessage)
        assert messages[1].content == "Hi there"
        # Turn 3 (odd, persona) should be HumanMessage (default)
        assert isinstance(messages[2], HumanMessage)
        assert messages[2].content == "How are you?"

    def test_build_messages_persona_with_turn_0(self):
        """Test persona role with turn 0 (initial message)."""
        persona_system_prompt = "You are roleplaying as a human user"
        history = [
            {"turn": 0, "speaker": "system", "response": "Initial message"},
            {"turn": 1, "speaker": "persona", "response": "Hello"},
            {"turn": 2, "speaker": "provider", "response": "Hi"},
        ]

        messages = build_langchain_messages(
            conversation_history=history, system_prompt=persona_system_prompt
        )

        assert len(messages) == 3
        # Turn 0 should always be HumanMessage regardless of role
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].content == "Initial message"
        # Turn 1 (persona) should be AIMessage when persona role
        assert isinstance(messages[1], AIMessage)
        assert messages[1].content == "Hello"
        # Turn 2 (provider) should be HumanMessage when persona role
        assert isinstance(messages[2], HumanMessage)
        assert messages[2].content == "Hi"


class TestFormatConversationAsString:
    """Test format_conversation_as_string function."""

    def test_format_with_no_history(self):
        """Test with only current message, no history."""
        result = format_conversation_as_string(
            conversation_history=[{"turn": 0, "speaker": "system", "response": "Hello"}]
        )

        assert result == "Human: Hello\n\nAssistant:"

    def test_format_with_system_prompt(self):
        """Test with system prompt."""
        result = format_conversation_as_string(
            conversation_history=[
                {"turn": 0, "speaker": "system", "response": "Hello"}
            ],
            system_prompt="You are helpful",
        )

        assert result == "System: You are helpful\n\nHuman: Hello\n\nAssistant:"

    def test_format_with_conversation_history(self):
        """Test with conversation history."""
        history = [
            {"turn": 1, "speaker": "persona", "response": "Hi"},
            {"turn": 2, "speaker": "agent", "response": "Hello"},
            {"turn": 3, "speaker": "persona", "response": "How are you?"},
            {"turn": 4, "speaker": "agent", "response": "What's your name?"},
        ]

        result = format_conversation_as_string(conversation_history=history)

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
            {"turn": 1, "speaker": "custom_persona", "response": "Hello"},
            {"turn": 2, "speaker": "custom_agent", "response": "Hi there"},
            {"turn": 3, "speaker": "custom_persona", "response": "Tell me more"},
        ]

        result = format_conversation_as_string(
            conversation_history=history, system_prompt="Be concise"
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
            {"turn": 1, "speaker": "persona", "response": "Hello"},
            {"turn": 2, "speaker": "agent", "response": "Hi"},
        ]

        result = format_conversation_as_string(conversation_history=history)

        expected = "Human: Hello\n\nAssistant: Hi\n\nAssistant:"
        assert result == expected

    def test_format_skips_none_response(self):
        """Test that turns with None response are skipped."""
        history = [
            {"turn": 1, "speaker": "persona", "response": "Hello"},
            {"turn": 2, "speaker": "agent", "response": None},
            {"turn": 3, "speaker": "persona", "response": "Still there?"},
        ]

        result = format_conversation_as_string(conversation_history=history)

        # Turn 2 should be skipped
        expected = "Human: Hello\n\nHuman: Still there?\n\nAssistant:"
        assert result == expected

    def test_format_empty_inputs(self):
        """Test with all empty inputs."""
        result = format_conversation_as_string(
            conversation_history=None, system_prompt=None
        )

        assert result == ""

    def test_format_with_persona_role_flips_types(self):
        """Test that persona role flips message types in string format."""
        persona_system_prompt = "You are roleplaying as a human user"
        history = [
            {"turn": 1, "speaker": "persona", "response": "Hello"},
            {"turn": 2, "speaker": "provider", "response": "Hi there"},
            {"turn": 3, "speaker": "persona", "response": "How are you?"},
        ]

        result = format_conversation_as_string(
            conversation_history=history, system_prompt=persona_system_prompt
        )

        # When persona role, persona turns (odd) should be Assistant,
        # provider turns (even) should be Human
        expected = (
            "System: You are roleplaying as a human user\n\n"
            "Assistant: Hello\n\n"
            "Human: Hi there\n\n"
            "Assistant: How are you?\n\n"
            "Assistant:"
        )
        assert result == expected
