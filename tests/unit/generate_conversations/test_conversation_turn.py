"""Unit tests for ConversationTurn class."""

from langchain_core.messages import AIMessage, HumanMessage

from generate_conversations.conversation_turn import ConversationTurn
from llm_clients import Role


class TestConversationTurnCreation:
    """Test creating ConversationTurn instances."""

    def test_create_with_human_message(self):
        """Test creating a ConversationTurn with HumanMessage."""
        message = HumanMessage(content="Hello world")
        turn = ConversationTurn(
            turn=1,
            speaker=Role.PERSONA,
            input_message="Start conversation",
            response_message=message,
            logging_metadata={"tokens": 100},
        )

        assert turn.turn == 1
        assert turn.speaker == Role.PERSONA
        assert turn.input_message == "Start conversation"
        assert turn.response == "Hello world"
        assert turn.early_termination is False
        assert turn.logging_metadata == {"tokens": 100}
        assert isinstance(turn.response_message, HumanMessage)

    def test_create_with_ai_message(self):
        """Test creating a ConversationTurn with AIMessage."""
        message = AIMessage(content="Hi there!")
        turn = ConversationTurn(
            turn=2,
            speaker=Role.PROVIDER,
            input_message="Hello world",
            response_message=message,
            early_termination=True,
            logging_metadata={"tokens": 50, "model": "gpt-4o"},
        )

        assert turn.turn == 2
        assert turn.speaker == Role.PROVIDER
        assert turn.input_message == "Hello world"
        assert turn.response == "Hi there!"
        assert turn.early_termination is True
        assert turn.logging_metadata == {"tokens": 50, "model": "gpt-4o"}
        assert isinstance(turn.response_message, AIMessage)

    def test_create_with_defaults(self):
        """Test creating with default values."""
        message = HumanMessage(content="Test")
        turn = ConversationTurn(
            turn=1, speaker=Role.PERSONA, input_message="", response_message=message
        )

        assert turn.early_termination is False
        assert turn.logging_metadata is None

    def test_response_property(self):
        """Test that response property returns message content."""
        message = AIMessage(content="Response text")
        turn = ConversationTurn(
            turn=1,
            speaker=Role.PROVIDER,
            input_message="Input",
            response_message=message,
        )

        assert turn.response == "Response text"
        assert turn.response == turn.response_message.text


class TestConversationTurnToDict:
    """Test converting ConversationTurn to dict format."""

    def test_to_dict_basic(self):
        """Test basic to_dict conversion."""
        message = HumanMessage(content="Hello")
        turn = ConversationTurn(
            turn=1,
            speaker=Role.PERSONA,
            input_message="Start",
            response_message=message,
            logging_metadata={"tokens": 100},
        )

        result = turn.to_dict()

        assert result == {
            "turn": 1,
            "speaker": "persona",
            "input": "Start",
            "response": "Hello",
            "early_termination": False,
            "logging": {"tokens": 100},
        }

    def test_to_dict_with_early_termination(self):
        """Test to_dict with early_termination flag."""
        message = AIMessage(content="Goodbye")
        turn = ConversationTurn(
            turn=5,
            speaker=Role.PROVIDER,
            input_message="See you",
            response_message=message,
            early_termination=True,
            logging_metadata={"stop_reason": "end_turn"},
        )

        result = turn.to_dict()

        assert result["early_termination"] is True
        assert result["turn"] == 5

    def test_to_dict_with_empty_metadata(self):
        """Test to_dict when logging_metadata is None."""
        message = HumanMessage(content="Test")
        turn = ConversationTurn(
            turn=1,
            speaker=Role.PERSONA,
            input_message="Input",
            response_message=message,
            logging_metadata=None,
        )

        result = turn.to_dict()

        assert result["logging"] == {}

    def test_to_dict_preserves_complex_metadata(self):
        """Test that to_dict preserves complex metadata structures."""
        complex_metadata = {
            "tokens": 100,
            "usage": {"input": 50, "output": 50},
            "model": "claude-3-opus",
            "timestamp": "2024-01-01T12:00:00",
        }
        message = AIMessage(content="Response")
        turn = ConversationTurn(
            turn=1,
            speaker=Role.PROVIDER,
            input_message="Prompt",
            response_message=message,
            logging_metadata=complex_metadata,
        )

        result = turn.to_dict()

        assert result["logging"] == complex_metadata
        assert result["logging"]["usage"]["input"] == 50


class TestConversationTurnFromDict:
    """Test creating ConversationTurn from dict format."""

    def test_from_dict_persona(self):
        """Test from_dict creates HumanMessage for persona speaker."""
        data = {
            "turn": 1,
            "speaker": "persona",
            "input": "Start",
            "response": "Hello world",
            "early_termination": False,
            "logging": {"tokens": 100},
        }

        # From provider's perspective, persona is "they" (HumanMessage)
        turn = ConversationTurn.from_dict(data, for_role=Role.PROVIDER)

        assert turn.turn == 1
        assert turn.speaker == Role.PERSONA
        assert turn.input_message == "Start"
        assert turn.response == "Hello world"
        assert turn.early_termination is False
        assert turn.logging_metadata == {"tokens": 100}
        assert isinstance(turn.response_message, HumanMessage)
        assert turn.response_message.text == "Hello world"

    def test_from_dict_agent(self):
        """Test from_dict creates AIMessage for agent speaker."""
        data = {
            "turn": 2,
            "speaker": "provider",
            "input": "Hello world",
            "response": "Hi there!",
            "early_termination": False,
            "logging": {"model": "gpt-4o"},
        }

        # From provider's perspective, provider is "I" (AIMessage)
        turn = ConversationTurn.from_dict(data, for_role=Role.PROVIDER)

        assert turn.speaker == Role.PROVIDER
        assert isinstance(turn.response_message, AIMessage)
        assert turn.response_message.text == "Hi there!"

    def test_from_dict_chatbot(self):
        """Test from_dict creates AIMessage for chatbot speaker."""
        data = {
            "turn": 3,
            "speaker": "provider",
            "input": "Question",
            "response": "Answer",
            "early_termination": True,
            "logging": {},
        }

        # From provider's perspective, provider is "I" (AIMessage)
        turn = ConversationTurn.from_dict(data, for_role=Role.PROVIDER)

        assert turn.speaker == Role.PROVIDER
        assert isinstance(turn.response_message, AIMessage)
        assert turn.early_termination is True

    def test_from_dict_with_missing_fields(self):
        """Test from_dict handles missing optional fields."""
        data = {
            "turn": 1,
            "speaker": "persona",
            "input": "",
            "response": "Test",
        }

        # From provider's perspective, persona is "they" (HumanMessage)
        turn = ConversationTurn.from_dict(data, for_role=Role.PROVIDER)

        assert turn.early_termination is False
        assert turn.logging_metadata is None

    def test_from_dict_persona_perspective(self):
        """Test from_dict from persona's perspective creates HumanMessage."""
        data = {
            "turn": 1,
            "speaker": "provider",
            "input": "Question",
            "response": "Answer from provider",
            "early_termination": False,
            "logging": {},
        }

        # From persona's perspective, provider is "they" (HumanMessage)
        turn = ConversationTurn.from_dict(data, for_role=Role.PERSONA)

        assert turn.speaker == Role.PROVIDER
        assert isinstance(turn.response_message, HumanMessage)
        assert turn.response_message.text == "Answer from provider"

    def test_from_dict_roundtrip(self):
        """Test that from_dict(to_dict()) preserves data."""
        original_message = HumanMessage(content="Original content")
        original_turn = ConversationTurn(
            turn=5,
            speaker=Role.PERSONA,
            input_message="Original input",
            response_message=original_message,
            early_termination=True,
            logging_metadata={"key": "value"},
        )

        # Convert to dict and back
        # Use provider's perspective to match original (persona is "they")
        dict_repr = original_turn.to_dict()
        restored_turn = ConversationTurn.from_dict(dict_repr, for_role=Role.PROVIDER)

        # Check all fields match
        assert restored_turn.turn == original_turn.turn
        assert restored_turn.speaker == original_turn.speaker
        assert restored_turn.input_message == original_turn.input_message
        assert restored_turn.response == original_turn.response
        assert restored_turn.early_termination == original_turn.early_termination
        assert restored_turn.logging_metadata == original_turn.logging_metadata
        assert isinstance(restored_turn.response_message, HumanMessage)


class TestConversationTurnEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_response(self):
        """Test handling empty response content."""
        message = AIMessage(content="")
        turn = ConversationTurn(
            turn=1,
            speaker=Role.PROVIDER,
            input_message="Input",
            response_message=message,
        )

        assert turn.response == ""
        assert turn.to_dict()["response"] == ""

    def test_multiline_response(self):
        """Test handling multiline response content."""
        multiline_text = "Line 1\nLine 2\nLine 3"
        message = HumanMessage(content=multiline_text)
        turn = ConversationTurn(
            turn=1,
            speaker=Role.PERSONA,
            input_message="Input",
            response_message=message,
        )

        assert turn.response == multiline_text
        assert turn.to_dict()["response"] == multiline_text

    def test_unicode_content(self):
        """Test handling unicode characters in content."""
        unicode_text = "Hello 🌍 世界 مرحبا"
        message = AIMessage(content=unicode_text)
        turn = ConversationTurn(
            turn=1,
            speaker=Role.PROVIDER,
            input_message="Say hello",
            response_message=message,
        )

        assert turn.response == unicode_text
        dict_repr = turn.to_dict()
        assert dict_repr["response"] == unicode_text

        # Test roundtrip
        # Use provider's perspective to match original (provider is "I")
        restored = ConversationTurn.from_dict(dict_repr, for_role=Role.PROVIDER)
        assert restored.response == unicode_text
