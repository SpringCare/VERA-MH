"""Represents a single turn in a conversation with LangChain message and metadata."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation with LangChain message and metadata.

    This class wraps a LangChain BaseMessage (HumanMessage or AIMessage) with
    additional metadata needed for conversation tracking, such as turn number,
    speaker identification, input prompts, early termination flags, and logging
    information.

    Attributes:
        turn: Sequential turn number (1-indexed)
        speaker: Identifier for the speaker (e.g., "persona", "chatbot", "agent")
        input_message: The message that prompted this response
        message: The LangChain message object (HumanMessage or AIMessage)
        early_termination: Whether this turn marked the end of conversation
        logging_metadata: Metadata from LLM provider (tokens, timing, etc.)
    """

    turn: int
    speaker: str
    input_message: str
    message: BaseMessage
    early_termination: bool = False
    logging_metadata: Optional[Dict[str, Any]] = None

    @property
    def response(self) -> str:
        """Get the response text from the message.

        Returns:
            The content of the LangChain message
        """
        content = self.message.content
        # LangChain messages can have string or list content; we expect strings
        if isinstance(content, str):
            return content
        # Fallback for unexpected types (shouldn't happen in normal usage)
        return str(content)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to legacy dict format for file export and backward compatibility.

        Returns:
            Dictionary with keys: turn, speaker, input, response,
            early_termination, logging
        """
        return {
            "turn": self.turn,
            "speaker": self.speaker,
            "input": self.input_message,
            "response": self.response,
            "early_termination": self.early_termination,
            "logging": self.logging_metadata or {},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationTurn":
        """Create from legacy dict format.

        Args:
            data: Dictionary with keys: turn, speaker, input, response, etc.

        Returns:
            ConversationTurn instance
        """
        # Determine message type based on speaker
        speaker = data["speaker"]
        if speaker == "persona":
            message = HumanMessage(content=data["response"])
        else:  # chatbot/agent
            message = AIMessage(content=data["response"])

        return cls(
            turn=data["turn"],
            speaker=speaker,
            input_message=data["input"],
            message=message,
            early_termination=data.get("early_termination", False),
            logging_metadata=data.get("logging"),
        )
