import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage

from generate_conversations.conversation_turn import ConversationTurn
from llm_clients import LLMInterface
from utils.conversation_utils import (
    ensure_provider_has_last_turn,
    save_conversation_to_file,
)


class ConversationSimulator:
    """Simulates a conversation between two LLM instances."""

    def __init__(self, persona: LLMInterface, agent: LLMInterface):
        self.persona = persona
        self.agent = agent
        self.conversation_history: List[ConversationTurn] = []

        # Define termination signals that indicate persona wants to end the conversation
        self.termination_signal = "<END OF CONVERSATION>"

    def _should_terminate_conversation(
        self, response: str, speaker: LLMInterface
    ) -> bool:
        """
        Check if the response indicates the conversation should end.
        Only terminates if persona (the conversation initiator) signals to end.
        """
        # Only allow persona to terminate the conversation early
        if speaker != self.persona:
            return False

        # Check for exact phrase matches (case insensitive)
        if re.search(re.escape(self.termination_signal), response, re.IGNORECASE):
            return True

        return False

    async def generate_conversation(
        self,
        max_turns: int,
        max_total_words: Optional[int] = None,
        persona_speaks_first: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Start a conversation between the two LLMs with early stopping support.

        Args:
            max_turns: Maximum number of conversation turns
            max_total_words: Optional maximum total words across all responses
            persona_speaks_first: If True, persona speaks first; else agent first.

        Returns:
            List of conversation turns with speaker and message
        """
        self.conversation_history = []
        max_turns = ensure_provider_has_last_turn(max_turns, persona_speaks_first)

        if persona_speaks_first:
            current_speaker = self.persona
            next_speaker = self.agent
        else:
            current_speaker = self.agent
            next_speaker = self.persona

        total_words = 0
        for turn in range(max_turns):
            # start or continue the conversation
            if turn == 0:
                response = await current_speaker.start_conversation()
            else:
                # generate a response based on the conversation history
                history_dicts = [t.to_dict() for t in self.conversation_history]
                response = await current_speaker.generate_response(
                    conversation_history=history_dicts
                )

            total_words += len(response.split())

            # Create LangChain message based on speaker for overall conversation storage
            # Note: each LLM Client will handle rebuilding the message type to
            # always see themselves as AIMessage
            if current_speaker == self.persona:
                lc_message = HumanMessage(content=response)
            else:
                lc_message = AIMessage(content=response)

            # Determine input message for overall conversation metadata tracking.
            # Turn 0: ask the client (first_message vs start_prompt; overridable).
            # Later turns: previous speaker's response.
            if turn == 0:
                input_msg = current_speaker.get_first_turn_input_message()
            else:
                if self.conversation_history:
                    input_msg = self.conversation_history[-1].response
                else:
                    raise ValueError(f"Conversation history is empty on turn {turn}")

            # Record this turn using ConversationTurn
            turn_obj = ConversationTurn(
                turn=turn + 1,
                speaker=current_speaker.role,
                input_message=input_msg,
                response_message=lc_message,
                early_termination=False,
                logging_metadata=current_speaker.last_response_metadata,
            )
            self.conversation_history.append(turn_obj)

            # Check if persona wants to end the conversation
            if self._should_terminate_conversation(response, current_speaker):
                self.conversation_history[-1].early_termination = True
                break

            # Check if we've reached the maximum total words
            # Only check when provider agent is speaking (not persona)
            if (
                current_speaker == self.agent
                and max_total_words is not None
                and total_words >= max_total_words
            ):
                break

            # Switch speakers for next turn
            current_speaker, next_speaker = next_speaker, current_speaker

        # Return dict format for backward compatibility
        return [t.to_dict() for t in self.conversation_history]

    def save_conversation(self, filename: str, folder="conversations") -> None:
        """Save the conversation to a text file."""

        # TODO: why is this two functions
        # Convert to dict format for file saving
        history_dicts = [t.to_dict() for t in self.conversation_history]
        save_conversation_to_file(history_dicts, filename, folder)
