"""Unit tests for ConversationTurn class."""

from langchain_core.messages import AIMessage, HumanMessage

from generate_conversations.conversation_turn import ConversationTurn


class TestConversationTurnCreation:
    """Test creating ConversationTurn instances."""

    def test_create_with_human_message(self):
        """Test creating a ConversationTurn with HumanMessage."""
        message = HumanMessage(content="Hello world")
        turn = ConversationTurn(
            turn=1,
            speaker="persona",
            input_message="Start conversation",
            message=message,
            logging_metadata={"tokens": 100},
        )

        assert turn.turn == 1
        assert turn.speaker == "persona"
        assert turn.input_message == "Start conversation"
        assert turn.response == "Hello world"
        assert turn.early_termination is False
        assert turn.logging_metadata == {"tokens": 100}
        assert isinstance(turn.message, HumanMessage)

    def test_create_with_ai_message(self):
        """Test creating a ConversationTurn with AIMessage."""
        message = AIMessage(content="Hi there!")
        turn = ConversationTurn(
            turn=2,
            speaker="agent",
            input_message="Hello world",
            message=message,
            early_termination=True,
            logging_metadata={"tokens": 50, "model": "gpt-4"},
        )

        assert turn.turn == 2
        assert turn.speaker == "agent"
        assert turn.input_message == "Hello world"
        assert turn.response == "Hi there!"
        assert turn.early_termination is True
        assert turn.logging_metadata == {"tokens": 50, "model": "gpt-4"}
        assert isinstance(turn.message, AIMessage)

    def test_create_with_defaults(self):
        """Test creating with default values."""
        message = HumanMessage(content="Test")
        turn = ConversationTurn(
            turn=1, speaker="persona", input_message="", message=message
        )

        assert turn.early_termination is False
        assert turn.logging_metadata is None

    def test_response_property(self):
        """Test that response property returns message content."""
        message = AIMessage(content="Response text")
        turn = ConversationTurn(
            turn=1, speaker="agent", input_message="Input", message=message
        )

        assert turn.response == "Response text"
        assert turn.response == turn.message.content


class TestConversationTurnToDict:
    """Test converting ConversationTurn to dict format."""

    def test_to_dict_basic(self):
        """Test basic to_dict conversion."""
        message = HumanMessage(content="Hello")
        turn = ConversationTurn(
            turn=1,
            speaker="persona",
            input_message="Start",
            message=message,
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
            speaker="agent",
            input_message="See you",
            message=message,
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
            speaker="persona",
            input_message="Input",
            message=message,
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
            speaker="agent",
            input_message="Prompt",
            message=message,
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

        turn = ConversationTurn.from_dict(data)

        assert turn.turn == 1
        assert turn.speaker == "persona"
        assert turn.input_message == "Start"
        assert turn.response == "Hello world"
        assert turn.early_termination is False
        assert turn.logging_metadata == {"tokens": 100}
        assert isinstance(turn.message, HumanMessage)
        assert turn.message.content == "Hello world"

    def test_from_dict_agent(self):
        """Test from_dict creates AIMessage for agent speaker."""
        data = {
            "turn": 2,
            "speaker": "agent",
            "input": "Hello world",
            "response": "Hi there!",
            "early_termination": False,
            "logging": {"model": "gpt-4"},
        }

        turn = ConversationTurn.from_dict(data)

        assert turn.speaker == "agent"
        assert isinstance(turn.message, AIMessage)
        assert turn.message.content == "Hi there!"

    def test_from_dict_chatbot(self):
        """Test from_dict creates AIMessage for chatbot speaker."""
        data = {
            "turn": 3,
            "speaker": "chatbot",
            "input": "Question",
            "response": "Answer",
            "early_termination": True,
            "logging": {},
        }

        turn = ConversationTurn.from_dict(data)

        assert turn.speaker == "chatbot"
        assert isinstance(turn.message, AIMessage)
        assert turn.early_termination is True

    def test_from_dict_with_missing_fields(self):
        """Test from_dict handles missing optional fields."""
        data = {
            "turn": 1,
            "speaker": "persona",
            "input": "",
            "response": "Test",
        }

        turn = ConversationTurn.from_dict(data)

        assert turn.early_termination is False
        assert turn.logging_metadata is None

    def test_from_dict_roundtrip(self):
        """Test that from_dict(to_dict()) preserves data."""
        original_message = HumanMessage(content="Original content")
        original_turn = ConversationTurn(
            turn=5,
            speaker="persona",
            input_message="Original input",
            message=original_message,
            early_termination=True,
            logging_metadata={"key": "value"},
        )

        # Convert to dict and back
        dict_repr = original_turn.to_dict()
        restored_turn = ConversationTurn.from_dict(dict_repr)

        # Check all fields match
        assert restored_turn.turn == original_turn.turn
        assert restored_turn.speaker == original_turn.speaker
        assert restored_turn.input_message == original_turn.input_message
        assert restored_turn.response == original_turn.response
        assert restored_turn.early_termination == original_turn.early_termination
        assert restored_turn.logging_metadata == original_turn.logging_metadata
        assert isinstance(restored_turn.message, HumanMessage)


class TestConversationTurnEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_response(self):
        """Test handling empty response content."""
        message = AIMessage(content="")
        turn = ConversationTurn(
            turn=1, speaker="agent", input_message="Input", message=message
        )

        assert turn.response == ""
        assert turn.to_dict()["response"] == ""

    def test_multiline_response(self):
        """Test handling multiline response content."""
        multiline_text = "Line 1\nLine 2\nLine 3"
        message = HumanMessage(content=multiline_text)
        turn = ConversationTurn(
            turn=1, speaker="persona", input_message="Input", message=message
        )

        assert turn.response == multiline_text
        assert turn.to_dict()["response"] == multiline_text

    def test_unicode_content(self):
        """Test handling unicode characters in content."""
        unicode_text = "Hello 🌍 世界 مرحبا"
        message = AIMessage(content=unicode_text)
        turn = ConversationTurn(
            turn=1, speaker="agent", input_message="Say hello", message=message
        )

        assert turn.response == unicode_text
        dict_repr = turn.to_dict()
        assert dict_repr["response"] == unicode_text

        # Test roundtrip
        restored = ConversationTurn.from_dict(dict_repr)
        assert restored.response == unicode_text
