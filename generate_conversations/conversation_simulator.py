from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel

from llm_clients import LLMInterface
from utils.conversation_utils import save_conversation_to_file


class ScoredResponse(BaseModel):
    """A single response with its probability score."""

    text: str
    probability: float


class ResponseWithScores(BaseModel):
    """Model for multiple responses with confidence scores.

    Note: Uses nested Pydantic model instead of List[Tuple[str, float]]
    because OpenAI's structured output API doesn't support tuple types in
    JSON schema. Tuples must be converted to objects with named fields.
    """

    responses: List[ScoredResponse]


class ConversationSimulator:
    """Simulates a conversation between two LLM instances."""

    def __init__(self, persona: LLMInterface, agent: LLMInterface):
        self.persona = persona
        self.agent = agent
        self.conversation_history: List[Dict[str, Any]] = []

        # Define termination signals that indicate persona wants to end the conversation
        self.termination_signals: Set[str] = set()

        # "goodbye", "bye", "farewell", "talk to you later",
        # "ttyl",
        # "end conversation", "conversation over", "that's all",
        # "nothing more to discuss",
        # "i'm done", "let's end here",
        # "conversation complete", "wrapping up", "final thoughts",
        # "concluding", "to conclude",
        # "in conclusion"
        # }

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

        response_lower = response.lower()

        # Check for exact phrase matches
        for signal in self.termination_signals:
            if signal in response_lower:
                return True

        # Check for common ending patterns
        ending_patterns = [
            # "it was nice",
            # "pleasure talking",
            # "great conversation",
            # "good chat",
            # "until next time"
        ]

        for pattern in ending_patterns:
            if pattern in response_lower:
                return True

        return False

    async def start_conversation(
        self,
        max_turns: int,
        initial_message: Optional[str] = None,
        max_total_words: Optional[int] = None,
        multiple_responses: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Start a conversation between the two LLMs with early stopping support.

        Args:
            max_turns: Maximum number of conversation turns
            initial_message: Optional initial message (for the first speaker)
                to start the conversation. By default, first speaker is persona.
            max_total_words: Optional maximum total words across all responses
            multiple_responses: If True, generate multiple responses with scores
                and select the highest-scored one. Requires JudgeLLM support.

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
            response: str
            score: Optional[float]
            all_responses: Optional[List[Tuple[str, float]]]

            if multiple_responses and hasattr(
                current_speaker, "generate_structured_response"
            ):
                # Generate multiple responses with scores
                # Add instruction to generate multiple responses
                multi_response_message = (
                    f"{current_message}\n\n"
                    "Please provide 5 diverse possible responses as a persona would, "
                    "each with a probability score (0.0-1.0) indicating how likely "
                    "that response is based on the persona's characteristics."
                )
                structured_response = (
                    await current_speaker.generate_structured_response(
                        multi_response_message, ResponseWithScores
                    )
                )
                print(f"Structured response: {structured_response}")
                # Select the response with the highest score
                best_response = max(
                    structured_response.responses, key=lambda x: x.probability
                )
                response = best_response.text
                score = best_response.probability
                # Store all responses in metadata for transparency
                all_responses = [
                    (r.text, r.probability) for r in structured_response.responses
                ]
            else:
                # Generate single response (default behavior)
                # Note: Despite interface definition, implementations return str
                response = await current_speaker.generate_response(current_message)  # type: ignore[assignment]
                score = None
                all_responses = None

            # response is mostly a text string
            total_words += len(response.split())

            # Record this turn
            turn_data = {
                "turn": turn + 1,
                "speaker": current_speaker.get_name(),
                "input": current_message or "",
                "response": response,
                "early_termination": False,
                "logging": current_speaker.get_last_response_metadata(),
            }

            # Add multiple response metadata if available
            if multiple_responses and all_responses is not None:
                turn_data["selected_score"] = score
                turn_data["all_responses"] = all_responses

            self.conversation_history.append(turn_data)

            # Check if persona wants to end the conversation
            if self._should_terminate_conversation(response, current_speaker):
                self.conversation_history[-1]["early_termination"] = True
                break

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
