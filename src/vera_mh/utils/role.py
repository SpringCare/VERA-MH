"""Who is speaking in a conversation."""

from enum import Enum


class Role(Enum):
    """Role of the LLM in a conversation."""

    USER = "user"  # The persona: an LLM roleplaying a person seeking help
    CHATBOT = "chatbot"  # The chatbot under test
