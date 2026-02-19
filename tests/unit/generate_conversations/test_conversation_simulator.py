"""Unit tests for ConversationSimulator."""

from unittest.mock import patch

import pytest

from generate_conversations.conversation_simulator import ConversationSimulator
from llm_clients.llm_interface import DEFAULT_START_PROMPT, Role
from tests.mocks.mock_llm import MockLLM


@pytest.mark.unit
@pytest.mark.asyncio
class TestConversationSimulator:
    """Test suite for ConversationSimulator class."""

    async def test_start_conversation_basic(self):
        """Test basic conversation flow: correct # of turns and speaker alternation."""
        persona = MockLLM(
            name="test-persona",
            role=Role.PERSONA,
            responses=["First", "Second", "Third"],
        )
        agent = MockLLM(
            name="test-agent",
            role=Role.PROVIDER,
            responses=["Reply 1", "Reply 2", "Reply 3"],
        )
        simulator = ConversationSimulator(persona=persona, agent=agent)

        history = await simulator.generate_conversation(max_turns=6)

        assert len(history) == 6
        for i in range(6):
            if i % 2 == 0:
                assert history[i]["speaker"] == "persona"
            else:
                assert history[i]["speaker"] == "provider"

    async def test_max_turns_respected(self):
        """Test that conversation stops at max_turns (provider speaks last)."""
        persona = MockLLM(name="persona", role=Role.PERSONA, responses=["msg"] * 10)
        agent = MockLLM(name="agent", role=Role.PROVIDER, responses=["reply"] * 10)
        simulator = ConversationSimulator(persona=persona, agent=agent)

        # Use even max_turns so no normalization (persona_speaks_first=True)
        history = await simulator.generate_conversation(max_turns=6)

        assert len(history) == 6
        assert history[-1]["turn"] == 6
        assert history[-1]["speaker"] == "provider"

    async def test_provider_speaks_first(self):
        """persona_speaks_first=False: provider speaks first & last (odd max_turns)."""
        persona = MockLLM(
            name="persona",
            role=Role.PERSONA,
            responses=["P1", "P2", "P3"],
        )
        agent = MockLLM(
            name="agent",
            role=Role.PROVIDER,
            responses=["A1", "A2", "A3"],
        )
        simulator = ConversationSimulator(persona=persona, agent=agent)

        history = await simulator.generate_conversation(
            max_turns=5,
            persona_speaks_first=False,
        )

        assert len(history) == 5
        assert history[0]["speaker"] == "provider"
        assert history[-1]["speaker"] == "provider"
        for i in range(5):
            if i % 2 == 0:
                assert history[i]["speaker"] == "provider"
            else:
                assert history[i]["speaker"] == "persona"

    async def test_early_termination_detection(self):
        """Test that conversation detects early termination signals."""
        persona = MockLLM(
            name="persona",
            role=Role.PERSONA,
            responses=["Hello", "Goodbye, I have to go now", "Should not appear"],
        )
        agent = MockLLM(name="agent", role=Role.PROVIDER, responses=["Hi there"] * 5)
        simulator = ConversationSimulator(persona=persona, agent=agent)

        simulator.termination_signal = "Goodbye"

        history = await simulator.generate_conversation(max_turns=10)

        assert len(history) == 3
        assert history[-1]["early_termination"] is True
        assert simulator.termination_signal in history[-1]["response"]

    async def test_default_early_termination_detection(self):
        """Test that conversation detects default early termination signals."""
        persona = MockLLM(
            name="persona",
            role=Role.PERSONA,
            responses=["Hello", "<END OF CONVERSATION>", "Should not appear"],
        )
        agent = MockLLM(name="agent", role=Role.PROVIDER, responses=["Hi there"] * 5)
        simulator = ConversationSimulator(persona=persona, agent=agent)

        history = await simulator.generate_conversation(max_turns=10)

        assert len(history) == 3
        assert history[-1]["early_termination"] is True
        assert simulator.termination_signal in history[-1]["response"]

    async def test_conversation_history_structure(self):
        """Test that conversation history has correct structure."""
        persona = MockLLM(name="persona", role=Role.PERSONA, responses=["Test message"])
        agent = MockLLM(name="agent", role=Role.PROVIDER, responses=["Test reply"])
        simulator = ConversationSimulator(persona=persona, agent=agent)

        history = await simulator.generate_conversation(max_turns=2)

        assert len(history) == 2
        for turn in history:
            assert "turn" in turn
            assert "speaker" in turn
            assert "input" in turn
            assert "response" in turn
            assert "early_termination" in turn
            assert "logging" in turn

        assert history[0]["turn"] == 1
        assert history[1]["turn"] == 2

    async def test_empty_initial_input(self):
        """Test default handling of None/empty initial input."""
        persona = MockLLM(
            name="persona", role=Role.PERSONA, responses=["Started conversation"]
        )
        agent = MockLLM(name="agent", role=Role.PROVIDER, responses=["Acknowledged"])
        simulator = ConversationSimulator(persona=persona, agent=agent)

        history = await simulator.generate_conversation(max_turns=2)

        assert len(history) == 2
        assert (
            history[0]["input"] == "Start the conversation based on the system prompt"
        )
        assert history[0]["response"] == "Started conversation"

    async def test_explicit_first_message(self):
        """Test conversation with static initial message (no LLM call for 1st turn)."""
        persona = MockLLM(
            name="persona",
            role=Role.PERSONA,
            responses=["Response to custom message"],
            first_message="Custom start",
        )
        agent = MockLLM(
            name="agent", role=Role.PROVIDER, responses=["Acknowledged custom"]
        )
        simulator = ConversationSimulator(persona=persona, agent=agent)

        history = await simulator.generate_conversation(max_turns=2)

        assert len(history) == 2
        assert history[0]["response"] == "Custom start"
        assert history[0]["input"] is None  # static first message has no prompt
        assert (
            "Custom start" in agent.calls
        )  # agent was prompted with persona's message

    async def test_llm_error_handling(self):
        """Test handling of LLM errors gracefully."""
        persona = MockLLM(
            name="persona", role=Role.PERSONA, responses=["Hello"], simulate_error=False
        )
        agent = MockLLM(
            name="agent", role=Role.PROVIDER, responses=[], simulate_error=True
        )
        simulator = ConversationSimulator(persona=persona, agent=agent)

        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            await simulator.generate_conversation(max_turns=2)

        assert "Simulated API error" in str(exc_info.value)

    async def test_metadata_captured(self):
        """Test that metadata is captured in conversation history."""
        persona = MockLLM(
            name="persona",
            model_name="persona-model",
            role=Role.PERSONA,
            responses=["Test"],
        )
        agent = MockLLM(
            name="agent",
            model_name="agent-model",
            role=Role.PROVIDER,
            responses=["Reply"],
        )
        simulator = ConversationSimulator(persona=persona, agent=agent)

        history = await simulator.generate_conversation(max_turns=2)

        assert "logging" in history[0]
        assert "logging" in history[1]
        assert history[0]["logging"]["provider"] == "mock"
        assert history[0]["logging"]["model"] == "persona-model"
        assert "prompt_tokens" in history[0]["logging"]
        assert "completion_tokens" in history[0]["logging"]
        assert "total_tokens" in history[0]["logging"]

    async def test_termination_only_by_persona(self):
        """Test that only persona can trigger early termination, not agent."""
        persona = MockLLM(
            name="persona", role=Role.PERSONA, responses=["Continue talking"] * 5
        )
        agent = MockLLM(
            name="agent",
            role=Role.PROVIDER,
            responses=["Goodbye, bye", "Another reply", "More replies"],
        )
        simulator = ConversationSimulator(persona=persona, agent=agent)
        simulator.termination_signal = "Goodbye"

        history = await simulator.generate_conversation(max_turns=6)

        assert len(history) == 6
        assert all(not turn["early_termination"] for turn in history)

    async def test_response_used_as_next_input(self):
        """Test that each response becomes the next speaker's input."""
        persona = MockLLM(
            name="persona", role=Role.PERSONA, responses=["Message A", "Message C"]
        )
        agent = MockLLM(
            name="agent", role=Role.PROVIDER, responses=["Message B", "Message D"]
        )
        simulator = ConversationSimulator(persona=persona, agent=agent)

        history = await simulator.generate_conversation(max_turns=4)

        # Turn 2's input should be turn 1's response
        assert history[1]["input"] == history[0]["response"]
        # Turn 3's input should be turn 2's response
        assert history[2]["input"] == history[1]["response"]
        # Turn 4's input should be turn 3's response
        assert history[3]["input"] == history[2]["response"]

    async def test_early_termination_flag_only_on_last_turn(self):
        """Test early_termination False for all turns except last."""
        persona = MockLLM(
            name="persona", role=Role.PERSONA, responses=["Hello", "Goodbye"]
        )
        agent = MockLLM(name="agent", role=Role.PROVIDER, responses=["Hi"])
        simulator = ConversationSimulator(persona=persona, agent=agent)
        simulator.termination_signal = "Goodbye"  # Must match exact case

        history = await simulator.generate_conversation(max_turns=10)

        assert len(history) == 3
        assert history[0]["early_termination"] is False
        assert history[1]["early_termination"] is False
        assert history[2]["early_termination"] is True

    async def test_conversation_history_reset_on_new_conversation(self):
        """Test that simulator clears history on each new generate_conversation call."""
        persona = MockLLM(name="persona", role=Role.PERSONA, responses=["First"] * 10)
        agent = MockLLM(name="agent", role=Role.PROVIDER, responses=["Reply"] * 10)
        simulator = ConversationSimulator(persona=persona, agent=agent)

        history1 = await simulator.generate_conversation(max_turns=2)
        # Use even max_turns so no normalization (persona_speaks_first=True)
        history2 = await simulator.generate_conversation(max_turns=4)

        assert len(history1) == 2
        assert len(history2) == 4
        assert history2[0]["turn"] == 1
        internal_history_dicts = [t.to_dict() for t in simulator.conversation_history]
        assert internal_history_dicts == history2
        assert internal_history_dicts != history1

    async def test_case_insensitive_termination(self):
        """Test that termination signal is detected even if case doesn't match."""
        persona = MockLLM(
            name="persona", role=Role.PERSONA, responses=["Hello", "GOODBYE and thanks"]
        )
        agent = MockLLM(name="agent", role=Role.PROVIDER, responses=["Hi"] * 5)
        simulator = ConversationSimulator(persona=persona, agent=agent)
        simulator.termination_signal = "goodbye"

        history = await simulator.generate_conversation(max_turns=10)

        assert len(history) == 3
        assert history[-1]["early_termination"] is True

    async def test_max_total_words_stopping_condition(self):
        """Test that conversation stops when max_total_words is reached."""
        persona = MockLLM(
            name="User",
            role=Role.PERSONA,
            responses=["Hello there", "How are you", "Great thanks", "Goodbye"],
        )
        agent = MockLLM(
            name="agent",
            role=Role.PROVIDER,
            responses=[
                "I am doing well today",  # 5 words
                "Very good indeed",  # 3 words
                "Fantastic",  # 1 word
                "Should not appear",
            ],
        )
        simulator = ConversationSimulator(persona=persona, agent=agent)

        history = await simulator.generate_conversation(
            max_turns=10, max_total_words=10
        )

        # Turn 1: User says "Hello there" (2 words, total: 2)
        # Turn 2: agent says "I am doing well today" (5 words, total: 7)
        # Turn 3: User says "How are you" (3 words, total: 10)
        # Turn 4: agent says "Very good indeed" (3 words, total: 13)
        # Should stop after turn 4 since agent exceeded max_total_words
        assert len(history) == 4
        assert history[-1]["speaker"] == "provider"

        total_words = sum(len(turn["response"].split()) for turn in history)
        assert total_words >= 10

    async def test_max_total_words_only_stops_after_chatbot_turn(self):
        """Test that max_total_words only checks after provider speaks."""
        persona = MockLLM(
            name="User",
            role=Role.PERSONA,
            responses=["This is a very long message with many words here"] * 5,
        )
        agent = MockLLM(
            name="agent",
            role=Role.PROVIDER,
            responses=["OK"] * 5,
        )
        simulator = ConversationSimulator(persona=persona, agent=agent)

        # Even though user exceeds limit, should only stop after agent
        history = await simulator.generate_conversation(max_turns=10, max_total_words=5)

        # Should complete at least 2 turns (user then provider)
        assert len(history) >= 2
        # Last turn should be from provider since that's when the check happens
        assert history[-1]["speaker"] == "provider"

    async def test_max_total_words_none_runs_to_max_turns(self):
        """Test that when max_total_words is None, conversation runs to max_turns."""
        persona = MockLLM(
            name="User",
            role=Role.PERSONA,
            responses=["Long message with many words"] * 10,
        )
        agent = MockLLM(
            name="agent",
            role=Role.PROVIDER,
            responses=["Even longer response with many many words"] * 10,
        )
        simulator = ConversationSimulator(persona=persona, agent=agent)

        history = await simulator.generate_conversation(
            max_turns=6, max_total_words=None
        )

        assert len(history) == 6

    async def test_save_conversation(self):
        """Test saving conversation to file."""
        persona = MockLLM(name="test-persona", role=Role.PERSONA, responses=["Hello"])
        agent = MockLLM(name="test-agent", role=Role.PROVIDER, responses=["Hi there"])
        simulator = ConversationSimulator(persona=persona, agent=agent)

        await simulator.generate_conversation(max_turns=2)

        with patch(
            "generate_conversations.conversation_simulator.save_conversation_to_file"
        ) as mock_save:
            simulator.save_conversation("test_convo.txt", folder="test_folder")

            expected_history_dicts = [
                t.to_dict() for t in simulator.conversation_history
            ]
            mock_save.assert_called_once_with(
                expected_history_dicts,
                "test_convo.txt",
                "test_folder",
            )


@pytest.mark.unit
@pytest.mark.asyncio
class TestFirstSpeakerAndFirstMessageCombinations:
    """Tests for persona_speaks_first x start_prompt vs first_message."""

    async def test_persona_first_start_prompt(self):
        """Persona first + start_prompt:
        first turn uses start_prompt, persona responds.
        """
        persona = MockLLM(
            name="persona",
            role=Role.PERSONA,
            responses=["Hello from persona"],
        )
        agent = MockLLM(
            name="agent",
            role=Role.PROVIDER,
            responses=["Hi back"],
        )
        simulator = ConversationSimulator(persona=persona, agent=agent)

        history = await simulator.generate_conversation(max_turns=2)

        assert history[0]["speaker"] == "persona"
        assert history[0]["input"] == DEFAULT_START_PROMPT
        assert history[0]["response"] == "Hello from persona"
        assert history[1]["speaker"] == "provider"

    async def test_persona_first_first_message(self):
        """Persona first + first_message: first turn input None, static response."""
        persona = MockLLM(
            name="persona",
            role=Role.PERSONA,
            responses=["Second message"],
            first_message="Static hello",
        )
        agent = MockLLM(
            name="agent",
            role=Role.PROVIDER,
            responses=["Reply to static"],
        )
        simulator = ConversationSimulator(persona=persona, agent=agent)

        history = await simulator.generate_conversation(max_turns=2)

        assert history[0]["speaker"] == "persona"
        assert history[0]["input"] is None
        assert history[0]["response"] == "Static hello"
        assert "Static hello" in agent.calls
        assert history[1]["speaker"] == "provider"

    async def test_provider_first_start_prompt(self):
        """Provider first, no first_message:
        first turn uses start_prompt, provider responds.
        """
        persona = MockLLM(
            name="persona",
            role=Role.PERSONA,
            responses=["User reply"],
        )
        agent = MockLLM(
            name="agent",
            role=Role.PROVIDER,
            responses=["Provider opening"],
        )
        simulator = ConversationSimulator(persona=persona, agent=agent)

        history = await simulator.generate_conversation(
            max_turns=2,
            persona_speaks_first=False,
        )

        assert len(history) == 3  # assert max_turns + 1 because provider speaks first
        assert history[0]["speaker"] == "provider"
        assert history[0]["input"] == DEFAULT_START_PROMPT
        assert history[0]["response"] == "Provider opening"
        assert history[1]["speaker"] == "persona"
        assert history[2]["speaker"] == "provider"
        assert "Provider opening" in persona.calls

    async def test_provider_first_first_message(self):
        """Provider first + first_message:
        first turn input None, static provider msg.
        """
        persona = MockLLM(
            name="persona",
            role=Role.PERSONA,
            responses=["User reply to provider"],
        )
        agent = MockLLM(
            name="agent",
            role=Role.PROVIDER,
            responses=["Second provider line"],
            first_message="Provider says hello first",
        )
        simulator = ConversationSimulator(persona=persona, agent=agent)

        history = await simulator.generate_conversation(
            max_turns=2,
            persona_speaks_first=False,
        )

        assert len(history) == 3  # assert max_turns + 1 because provider speaks first
        assert history[0]["speaker"] == "provider"
        assert history[0]["input"] is None
        assert history[0]["response"] == "Provider says hello first"
        assert history[1]["speaker"] == "persona"
        assert history[2]["speaker"] == "provider"
        assert "Provider says hello first" in persona.calls
