from typing import Any, Dict, List, Optional

from llm_clients import LLMInterface
from utils.conversation_utils import save_conversation_to_file


class ConversationSimulator:
    """Simulates a conversation between two LLM instances."""

    def __init__(self, persona: LLMInterface, agent: LLMInterface):
        self.persona = persona
        self.agent = agent
        self.conversation_history: List[Dict[str, Any]] = []

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
        if initial_message is None:
            current_message = "Start the conversation based on the system prompt"
        else:
            current_message = initial_message
        current_speaker = self.persona
        next_speaker = self.agent

        total_words = 0
        for turn in range(max_turns):
            # Record start time for this turn

            # Generate response
            response = await current_speaker.generate_response(current_message)

            total_words += len(response.split())
            # Record this turn
            self.conversation_history.append(
                {
                    "turn": turn + 1,
                    "speaker": current_speaker.get_name(),
                    "input": current_message or "",
                    "response": response,
                    "early_termination": False,
                    "logging": current_speaker.get_last_response_metadata(),
                }
            )

            # Check if we've reached the maximum total words
            # TODO: chatbot should not be hardcoded
            if (
                current_speaker.get_name() == "chatbot"
                and max_total_words is not None
                and total_words >= max_total_words
            ):
                break

            # Switch speakers and use the response as the next input
            current_message = response
            current_speaker, next_speaker = next_speaker, current_speaker

        return self.conversation_history

    def save_conversation(self, filename: str, folder="conversations") -> None:
        """Save the conversation to a text file."""

        # TODO: why is this two functions
        save_conversation_to_file(
            self.conversation_history, filename, folder, self.persona.get_name()
        )
