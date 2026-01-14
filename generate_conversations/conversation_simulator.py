from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage

from generate_conversations.conversation_turn import ConversationTurn
from llm_clients import LLMInterface
from utils.conversation_utils import save_conversation_to_file


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

        # Check for exact phrase matches
        if self.termination_signal in response:
            return True

        return False

    async def start_conversation(
        self,
        max_turns: int,
        initial_message: Optional[str] = None,
        max_total_words: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Start a conversation between the two LLMs with early stopping support.

        Args:
            max_turns: Maximum number of conversation turns
            initial_message: Optional initial message (for the first speaker)
                to start the conversation. By default, first speaker is persona.
            max_total_words: Optional maximum total words across all responses


        Returns:
            List of conversation turns with speaker and message
        """
        self.conversation_history = []

        # Set up initial message for the conversation
        if initial_message is None:
            initial_message = "Start the conversation based on the system prompt"

        # IMPORTANT: Persona always starts first (turn 1, 3, 5...)
        # This determines the odd/even pattern in build_langchain_messages()
        # If you change this order, update utils/conversation_utils.py accordingly
        current_speaker = self.persona
        next_speaker = self.agent

        total_words = 0
        for turn in range(max_turns):
            # Record start time for this turn

            # Generate response with conversation history
            # On turn 0, create a "turn 0" entry for the initial message
            # This provides context without being a real conversation turn
            if turn == 0:
                initial_turn = {
                    "turn": 0,
                    "speaker": "system",
                    "response": initial_message,
                }
                history_dicts = [initial_turn]
            else:
                # Convert conversation history to dict format for LLM interface
                history_dicts = [t.to_dict() for t in self.conversation_history]

            response = await current_speaker.generate_response(
                conversation_history=history_dicts
            )

            total_words += len(response.split())

            # Create LangChain message based on speaker
            if current_speaker == self.persona:
                lc_message = HumanMessage(content=response)
            else:
                lc_message = AIMessage(content=response)

            # Determine input message for metadata tracking
            # On turn 0, it's the initial message
            # On subsequent turns, it's the previous speaker's response
            if turn == 0:
                input_msg = initial_message
            else:
                # Get the last turn's response as input for this turn
                input_msg = (
                    self.conversation_history[-1].response
                    if self.conversation_history
                    else ""
                )

            # Record this turn using ConversationTurn
            turn_obj = ConversationTurn(
                turn=turn + 1,
                speaker=current_speaker.get_name(),
                input_message=input_msg,
                message=lc_message,
                early_termination=False,
                logging_metadata=current_speaker.get_last_response_metadata(),
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
        save_conversation_to_file(
            history_dicts, filename, folder, self.persona.get_name()
        )
