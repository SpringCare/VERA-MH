"""Unit tests for ConversationSimulator."""

from unittest.mock import patch

import pytest

from generate_conversations.conversation_simulator import ConversationSimulator
from tests.mocks.mock_llm import MockLLM


@pytest.mark.unit
@pytest.mark.asyncio
class TestConversationSimulator:
    """Test suite for ConversationSimulator class."""

    async def test_start_conversation_basic(self):
        """Test basic conversation flow with mock LLMs."""
        # Arrange
        persona = MockLLM(
            name="test-persona",
            responses=["Hello, I need help", "Thank you for listening"],
        )
        agent = MockLLM(
            name="test-agent", responses=["How can I help you?", "You're welcome"]
        )
        simulator = ConversationSimulator(persona=persona, agent=agent)

        # Act
        history = await simulator.start_conversation(max_turns=4)

        # Assert
        assert len(history) == 4
        assert history[0]["speaker"] == "test-persona"
        assert history[1]["speaker"] == "test-agent"
        assert history[2]["speaker"] == "test-persona"
        assert history[3]["speaker"] == "test-agent"

    async def test_conversation_alternates_speakers(self):
        """Test that conversation properly alternates between User and Chatbot."""
        # Arrange
        persona = MockLLM(
            name="User",
            responses=["First message", "Second message", "Third message"],
        )
        agent = MockLLM(
            name="Chatbot",
            responses=["First reply", "Second reply", "Third reply"],
        )
        simulator = ConversationSimulator(persona=persona, agent=agent)

        # Act
        history = await simulator.start_conversation(max_turns=6)

        # Assert
        assert len(history) == 6
        for i in range(6):
            if i % 2 == 0:
                assert history[i]["speaker"] == "User"
            else:
                assert history[i]["speaker"] == "Chatbot"

    async def test_max_turns_respected(self):
        """Test that conversation stops at max_turns."""
        # Arrange
        persona = MockLLM(name="persona", responses=["msg"] * 10)
        agent = MockLLM(name="agent", responses=["reply"] * 10)
        simulator = ConversationSimulator(persona=persona, agent=agent)

        # Act
        history = await simulator.start_conversation(max_turns=5)

        # Assert
        assert len(history) == 5
        assert history[-1]["turn"] == 5

    async def test_early_termination_detection(self):
        """Test that conversation detects early termination signals."""
        # Arrange
        persona = MockLLM(
            name="persona",
            responses=["Hello", "Goodbye, I have to go now", "Should not appear"],
        )
        agent = MockLLM(name="agent", responses=["Hi there"] * 5)
        simulator = ConversationSimulator(persona=persona, agent=agent)

        # Add termination signals
        simulator.termination_signals = {"goodbye", "bye"}

        # Act
        history = await simulator.start_conversation(max_turns=10)

        # Assert
        # Turn 1: persona says "Hello"
        # Turn 2: agent says "Hi there"
        # Turn 3: persona says "Goodbye..." and terminates
        assert len(history) == 3  # Should stop after persona says goodbye
        assert history[-1]["early_termination"] is True
        assert "goodbye" in history[-1]["response"].lower()

    async def test_conversation_history_structure(self):
        """Test that conversation history has correct structure."""
        # Arrange
        persona = MockLLM(name="persona", responses=["Test message"])
        agent = MockLLM(name="agent", responses=["Test reply"])
        simulator = ConversationSimulator(persona=persona, agent=agent)

        # Act
        history = await simulator.start_conversation(max_turns=2)

        # Assert
        assert len(history) == 2
        for turn in history:
            assert "turn" in turn
            assert "speaker" in turn
            assert "input" in turn
            assert "response" in turn
            assert "early_termination" in turn
            assert "logging" in turn

        # Verify turn numbers are sequential
        assert history[0]["turn"] == 1
        assert history[1]["turn"] == 2

    async def test_empty_initial_input(self):
        """Test handling of None/empty initial input."""
        # Arrange
        persona = MockLLM(name="persona", responses=["Started conversation"])
        agent = MockLLM(name="agent", responses=["Acknowledged"])
        simulator = ConversationSimulator(persona=persona, agent=agent)

        # Act
        history = await simulator.start_conversation(initial_message=None, max_turns=2)

        # Assert
        assert len(history) == 2
        assert (
            history[0]["input"] == "Start the conversation based on the system prompt"
        )
        assert history[0]["response"] == "Started conversation"

    async def test_explicit_initial_message(self):
        """Test conversation with explicit initial message."""
        # Arrange
        persona = MockLLM(name="persona", responses=["Response to custom message"])
        agent = MockLLM(name="agent", responses=["Acknowledged custom"])
        simulator = ConversationSimulator(persona=persona, agent=agent)

        # Act
        history = await simulator.start_conversation(
            initial_message="Custom start", max_turns=2
        )

        # Assert
        assert len(history) == 2
        assert history[0]["input"] == "Custom start"
        assert "Custom start" in persona.calls

    async def test_llm_error_handling(self):
        """Test handling of LLM errors gracefully."""
        # Arrange
        persona = MockLLM(name="persona", responses=["Hello"], simulate_error=False)
        agent = MockLLM(name="agent", responses=[], simulate_error=True)
        simulator = ConversationSimulator(persona=persona, agent=agent)

        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            await simulator.start_conversation(max_turns=2)

        assert "Simulated API error" in str(exc_info.value)

    async def test_metadata_captured(self):
        """Test that metadata is captured in conversation history."""
        # Arrange
        persona = MockLLM(name="persona", responses=["Test"])
        agent = MockLLM(name="agent", responses=["Reply"])
        simulator = ConversationSimulator(persona=persona, agent=agent)

        # Act
        history = await simulator.start_conversation(max_turns=2)

        # Assert
        assert "logging" in history[0]
        assert "logging" in history[1]
        assert history[0]["logging"]["provider"] == "mock"
        assert history[0]["logging"]["model"] == "persona"
        assert "prompt_tokens" in history[0]["logging"]
        assert "completion_tokens" in history[0]["logging"]
        assert "total_tokens" in history[0]["logging"]

    async def test_termination_only_by_persona(self):
        """Test that only persona can trigger early termination, not agent."""
        # Arrange
        persona = MockLLM(name="persona", responses=["Continue talking"] * 5)
        agent = MockLLM(
            name="agent", responses=["Goodbye, bye", "Another reply", "More replies"]
        )
        simulator = ConversationSimulator(persona=persona, agent=agent)
        simulator.termination_signals = {"goodbye", "bye"}

        # Act
        history = await simulator.start_conversation(max_turns=6)

        # Assert - Should complete all turns despite agent saying goodbye
        assert len(history) == 6
        assert all(not turn["early_termination"] for turn in history)

    async def test_multiple_termination_signals(self):
        """Test detection of multiple different termination signals."""
        # Arrange
        persona = MockLLM(
            name="persona",
            responses=["Hello", "Talk to you later, ttyl"],
        )
        agent = MockLLM(name="agent", responses=["Hi"] * 5)
        simulator = ConversationSimulator(persona=persona, agent=agent)
        simulator.termination_signals = {"goodbye", "ttyl", "farewell"}

        # Act
        history = await simulator.start_conversation(max_turns=10)

        # Assert
        # Turn 1: persona says "Hello"
        # Turn 2: agent says "Hi"
        # Turn 3: persona says "Talk to you later, ttyl" and terminates
        assert len(history) == 3
        assert history[-1]["early_termination"] is True

    async def test_response_used_as_next_input(self):
        """Test that each response becomes the next speaker's input."""
        # Arrange
        persona = MockLLM(name="persona", responses=["Message A", "Message C"])
        agent = MockLLM(name="agent", responses=["Message B", "Message D"])
        simulator = ConversationSimulator(persona=persona, agent=agent)

        # Act
        history = await simulator.start_conversation(max_turns=4)

        # Assert
        # Turn 2's input should be turn 1's response
        assert history[1]["input"] == history[0]["response"]
        # Turn 3's input should be turn 2's response
        assert history[2]["input"] == history[1]["response"]
        # Turn 4's input should be turn 3's response
        assert history[3]["input"] == history[2]["response"]

    async def test_early_termination_flag_only_on_last_turn(self):
        """Test early_termination False for all turns except last."""
        # Arrange
        persona = MockLLM(name="persona", responses=["Hello", "Goodbye"])
        agent = MockLLM(name="agent", responses=["Hi"])
        simulator = ConversationSimulator(persona=persona, agent=agent)
        simulator.termination_signals = {"goodbye"}

        # Act
        history = await simulator.start_conversation(max_turns=10)

        # Assert
        # Turn 1: persona says "Hello"
        # Turn 2: agent says "Hi"
        # Turn 3: persona says "Goodbye" and terminates
        assert len(history) == 3
        assert history[0]["early_termination"] is False
        assert history[1]["early_termination"] is False
        assert history[2]["early_termination"] is True

    async def test_no_early_termination_when_no_signals(self):
        """Test conversations run to completion without signals."""
        # Arrange
        persona = MockLLM(
            name="persona",
            responses=["Goodbye", "Bye", "Farewell"],
        )
        agent = MockLLM(name="agent", responses=["OK"] * 5)
        simulator = ConversationSimulator(persona=persona, agent=agent)
        # No termination signals set (empty set by default)

        # Act
        history = await simulator.start_conversation(max_turns=6)

        # Assert - Should run to completion
        assert len(history) == 6
        assert all(not turn["early_termination"] for turn in history)

    async def test_conversation_history_reset_on_new_conversation(self):
        """Test that conversation history is reset when starting a new conversation."""
        # Arrange
        persona = MockLLM(name="persona", responses=["First"] * 10)
        agent = MockLLM(name="agent", responses=["Reply"] * 10)
        simulator = ConversationSimulator(persona=persona, agent=agent)

        # Act
        history1 = await simulator.start_conversation(max_turns=2)
        persona.reset()
        agent.reset()
        history2 = await simulator.start_conversation(max_turns=3)

        # Assert
        assert len(history1) == 2
        assert len(history2) == 3
        assert history2[0]["turn"] == 1  # Should restart from turn 1
        assert simulator.conversation_history == history2
        assert simulator.conversation_history != history1

    async def test_case_insensitive_termination_detection(self):
        """Test that termination signals are detected case-insensitively."""
        # Arrange
        persona = MockLLM(name="persona", responses=["Hello", "GOODBYE and thanks"])
        agent = MockLLM(name="agent", responses=["Hi"] * 5)
        simulator = ConversationSimulator(persona=persona, agent=agent)
        simulator.termination_signals = {"goodbye"}

        # Act
        history = await simulator.start_conversation(max_turns=10)

        # Assert
        # Turn 1: persona says "Hello"
        # Turn 2: agent says "Hi"
        # Turn 3: persona says "GOODBYE and thanks" and terminates
        assert len(history) == 3
        assert history[-1]["early_termination"] is True

    async def test_max_total_words_stopping_condition(self):
        """Test that conversation stops when max_total_words is reached."""
        # Arrange - Use agent named "chatbot" to trigger the max_total_words check
        persona = MockLLM(
            name="User",
            responses=["Hello there", "How are you", "Great thanks", "Goodbye"],
        )
        agent = MockLLM(
            name="chatbot",
            responses=[
                "I am doing well today",  # 5 words
                "Very good indeed",  # 3 words
                "Fantastic",  # 1 word
                "Should not appear",
            ],
        )
        simulator = ConversationSimulator(persona=persona, agent=agent)

        # Act - Set max_total_words to 10, should stop after agent's second response
        history = await simulator.start_conversation(max_turns=10, max_total_words=10)

        # Assert
        # Turn 1: User says "Hello there" (2 words, total: 2)
        # Turn 2: chatbot says "I am doing well today" (5 words, total: 7)
        # Turn 3: User says "How are you" (3 words, total: 10)
        # Turn 4: chatbot says "Very good indeed" (3 words, total: 13)
        # Should stop after turn 4 since chatbot exceeded max_total_words
        assert len(history) == 4
        assert history[-1]["speaker"] == "chatbot"

        # Verify total word count is close to but over the limit
        total_words = sum(len(turn["response"].split()) for turn in history)
        assert total_words >= 10  # Should exceed the limit

    async def test_max_total_words_only_stops_after_chatbot_turn(self):
        """Test that max_total_words only checks after chatbot (agent) speaks."""
        # Arrange
        persona = MockLLM(
            name="User",
            responses=["This is a very long message with many words here"] * 5,
        )
        agent = MockLLM(
            name="chatbot",
            responses=["OK"] * 5,
        )
        simulator = ConversationSimulator(persona=persona, agent=agent)

        # Act - Even though User exceeds limit, should only stop after chatbot
        history = await simulator.start_conversation(max_turns=10, max_total_words=5)

        # Assert - Should complete at least 2 turns (User then chatbot)
        assert len(history) >= 2
        # Last turn should be from chatbot since that's when the check happens
        assert history[-1]["speaker"] == "chatbot"

    async def test_max_total_words_none_runs_to_max_turns(self):
        """Test that when max_total_words is None, conversation runs to max_turns."""
        # Arrange
        persona = MockLLM(
            name="User",
            responses=["Long message with many words"] * 10,
        )
        agent = MockLLM(
            name="chatbot",
            responses=["Even longer response with many many words"] * 10,
        )
        simulator = ConversationSimulator(persona=persona, agent=agent)

        # Act - No max_total_words limit
        history = await simulator.start_conversation(max_turns=6, max_total_words=None)

        # Assert - Should run to max_turns
        assert len(history) == 6

    async def test_save_conversation(self):
        """Test saving conversation to file."""
        # Arrange
        persona = MockLLM(name="test-persona", responses=["Hello"])
        agent = MockLLM(name="test-agent", responses=["Hi there"])
        simulator = ConversationSimulator(persona=persona, agent=agent)

        # Create a conversation
        await simulator.start_conversation(max_turns=2)

        # Act
        with patch(
            "generate_conversations.conversation_simulator.save_conversation_to_file"
        ) as mock_save:
            simulator.save_conversation("test_convo.txt", folder="test_folder")

            # Assert
            mock_save.assert_called_once_with(
                simulator.conversation_history,
                "test_convo.txt",
                "test_folder",
                "test-persona",
            )
