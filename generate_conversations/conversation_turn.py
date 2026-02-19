"""Represents a single turn in a conversation with LangChain message and metadata."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from llm_clients.llm_interface import Role


@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation with LangChain message and metadata.

    This class wraps a LangChain BaseMessage (HumanMessage or AIMessage) with
    additional metadata needed for conversation tracking, such as turn number,
    speaker identification, input prompts, early termination flags, and logging
    information.

    Attributes:
        turn (int): Sequential turn number (1-indexed).
        speaker (Role): Role identifier of the speaker (Role.PERSONA or Role.PROVIDER).
        input_message (Optional[str]): The message string that prompted the
            response_message, or None if the response did not come from a prompt.
        response_message (BaseMessage): The LangChain message object
            (HumanMessage or AIMessage).
        early_termination (bool): Whether this turn marked the end of conversation.
        logging_metadata (Optional[Dict[str, Any]]): Metadata from LLM provider
            (tokens, timing, etc.).
    """

    turn: int
    speaker: Role
    input_message: Optional[str]
    response_message: BaseMessage
    early_termination: bool = False
    logging_metadata: Optional[Dict[str, Any]] = None

    @property
    def response(self) -> str:
        """Get the response text from the message.

        Returns:
            The content of the LangChain message
        """
        message_str = self.response_message.text
        # LangChain messages can have string or list content; we expect strings
        if isinstance(message_str, str):
            return message_str

        # Raise an error if the message is not a string
        raise ValueError(f"Unexpected message type: {type(message_str)}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict format for file export.

        Returns:
            Dictionary with keys: turn, speaker, input, response,
            early_termination, logging
        """
        result = {
            "turn": self.turn,
            "speaker": self.speaker.value,
            "input": self.input_message,
            "response": self.response,
            "early_termination": self.early_termination,
            "logging": self.logging_metadata or {},
        }
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any], for_role: Role) -> "ConversationTurn":
        """Create from dict format for the given role.

        If the speaker's role matches the requested role,
        it's an AIMessage (what "I" said) for the requested role.
        Otherwise, it's a HumanMessage (what "they" said) for the requested role.

        For example:
        If the requested for_role is Role.PERSONA, and speaker is Role.PROVIDER,
        then the message is a HumanMessage (what "they" said).
        If the requested for_role is Role.PROVIDER, and speaker is Role.PROVIDER,
        then the message is an AIMessage (what "I" said).

        Args:
            data: Dictionary with keys: turn, speaker, input, response, etc.
                speaker must be a Role enum value (string) or Role enum.
            for_role: Role of the speaker to create the ConversationTurn for.
        Returns:
            ConversationTurn instance
        """
        # Validate required fields
        required_fields = ["turn", "speaker", "input", "response"]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

        turn_speaker = data["speaker"]
        try:
            speaker_role = Role(turn_speaker)
        except ValueError:
            raise ValueError(f"Invalid role value '{turn_speaker}'")

        # Determine message type based on role
        # If the speaker's role matches the requested role,
        # it's an AIMessage (what "I" said).
        if speaker_role == for_role:
            message = AIMessage(content=data["response"])
        else:  # Otherwise, it's a HumanMessage (what "they" said).
            message = HumanMessage(content=data["response"])

        return cls(
            turn=data["turn"],
            speaker=speaker_role,
            input_message=data["input"],
            response_message=message,
            early_termination=data.get("early_termination", False),
            logging_metadata=data.get("logging"),
        )
