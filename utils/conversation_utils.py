"""Utilities for conversation management and file operations."""

from datetime import datetime
from functools import cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from llm_clients.llm_interface import Role

from .debug import debug_print


def ensure_provider_has_last_turn(max_turns: int, persona_speaks_first: bool) -> int:
    """
    Return max_turns adjusted so the provider agent always has the last turn.

    - persona_speaks_first=True (persona first): need even number of turns.
    - persona_speaks_first=False (agent first): need odd number of turns.
    """
    if persona_speaks_first and max_turns % 2 != 0:
        debug_print(
            (
                f"Adjusted max_turns {max_turns} -> {max_turns + 1}"
                f"so provider has last turn"
            )
        )
        return max_turns + 1
    if not persona_speaks_first and max_turns % 2 == 0:
        debug_print(
            (
                f"Adjusted max_turns{max_turns} -> {max_turns + 1}  "
                f"so provider has last turn "
                f"(persona_speaks_first={persona_speaks_first}).",
            )
        )
        return max_turns + 1
    return max_turns


def add_timestamp_to_path(path: Path) -> Path:
    """
    Add timestamp to filename before extension.

    Args:
        path: Original path

    Returns:
        Path with timestamp inserted before extension (format: YYYYMMDD_HHMMSS)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = path.stem
    suffix = path.suffix
    return path.parent / f"{stem}_{timestamp}{suffix}"


def generate_conversation_filename(prefix: str = "conversation") -> str:
    """
    Generate a timestamped filename for conversation logs.

    Args:
        prefix: Prefix for the filename

    Returns:
        Formatted filename with timestamp
    """
    temp_path = Path(f"{prefix}.txt")
    timestamped_path = add_timestamp_to_path(temp_path)
    return timestamped_path.name


def save_conversation_to_file(
    conversation_history: List[Dict[str, Any]],
    filename: str,
    folder: str,
) -> None:
    """
    Save conversation history to a text file.

    Args:
        conversation_history: List of conversation turns (speaker is role value)
        filename: Output filename
        folder: Output folder
    """
    # Ensure folder exists
    Path(folder).mkdir(parents=True, exist_ok=True)

    summary = format_conversation_summary(conversation_history)
    with open(f"{folder}/{filename}", "w", encoding="utf-8") as f:
        f.write(summary)


def format_conversation_summary(
    conversation_history: List[Dict[str, Any]],
) -> str:
    """
    Format conversation history into a readable summary.

    Args:
        conversation_history: List of conversation turns (speaker is role value)

    Returns:
        Formatted conversation summary. Persona turns are labeled "user:",
        other roles (e.g. provider) are labeled "chatbot:".
    """
    if not conversation_history:
        return "No conversation recorded."

    summary = ""

    for i, turn in enumerate(conversation_history):
        # Get speaker value (Role enum value string: "persona", "provider", etc.)
        speaker_value = turn.get("speaker")

        # Persona -> "user:", any other role -> "chatbot:"
        try:
            turn_role = Role(speaker_value)
            prefix = "user:" if turn_role == Role.PERSONA else "chatbot:"
        except ValueError, TypeError:
            raise ValueError(f"Invalid role value '{speaker_value}' for turn {i}")

        # Add the response with proper formatting
        summary += f"{prefix} {turn['response']}\n"

        # Check for early termination flag
        if turn.get("early_termination", False):
            summary += f"[CONVERSATION ENDED - {speaker_value} signaled termination]\n"

        # Add empty line between speakers, but not after the last turn
        if i < len(conversation_history) - 1:
            summary += "\n"

    return summary


@cache
def load_persona_role_reminder() -> str:
    """
    Return the reminder that the persona LLM plays the help-seeker,
    not the counselor.
    """
    return Path("data/persona_prompt_reminder.txt").read_text(encoding="utf-8")


def apply_persona_role_reminder(
    messages: List[BaseMessage],
    conversation_history: Optional[List[Dict[str, Any]]],
) -> None:
    """
    If the last complete turn was from the provider, prefix the final HumanMessage
    (chatbot's text) with the persona role reminder.
    """
    if not messages or not conversation_history:
        return
    last_turn: Optional[Dict[str, Any]] = None
    # Get the latest complete turn starting from the end & breaking on first valid turn.
    for turn in reversed(conversation_history):
        if turn.get("turn") is None or turn.get("response") is None:
            continue
        last_turn = turn
        break
    if not last_turn or last_turn.get("turn") == 0:
        return
    speaker = last_turn.get("speaker")
    if speaker is None:
        return
    # History was validated while building messages; Role(speaker) should not raise.
    if Role(speaker) == Role.PROVIDER and isinstance(messages[-1], HumanMessage):
        prefix = load_persona_role_reminder()
        messages[-1] = HumanMessage(content=prefix + messages[-1].text)


def build_langchain_messages(
    role: Role,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> List[BaseMessage]:
    """
    Build a list of LangChain messages from conversation history for the given role.

    Args:
        conversation_history: Optional list of previous conversation turns.
            Each turn is a dict with keys: 'turn', 'speaker', 'response'.
            Turn 0 contains the initial message with speaker="system".
        role: Optional role of the LLM (Role.PERSONA, Role.PROVIDER, or None).
            Messages from the same role are converted to AIMessage (what "I"
            said), messages from the other role are converted to HumanMessage
            (what "they" said).

    Returns:
        List of LangChain message objects (HumanMessage, AIMessage)
    """
    messages: List[BaseMessage] = []

    # Add conversation history if provided
    if conversation_history:
        hist_len = len(conversation_history)
        debug_print(f"[DEBUG] Processing {hist_len} turns from history:")
        for turn in conversation_history:
            turn_number = turn.get("turn")
            text = turn.get("response")

            # Skip turns without turn number or response
            if turn_number is None or text is None:
                continue

            # Special case: turn 0 is for starting the conversation.
            if turn_number == 0:
                debug_print(f"  Turn 0 -> HumanMessage (initial): {text[:50]}...")
                messages.append(HumanMessage(content=text))
                continue

            # Determine message type based on role comparison
            turn_speaker = turn.get("speaker")
            if turn_speaker is not None:
                try:
                    turn_role = Role(turn_speaker)
                    # If the message's role matches the requesting role, it's an
                    # AIMessage (what "I" said). Otherwise, it's a HumanMessage
                    # (what "they" said).
                    if turn_role == role:
                        message = AIMessage(content=text)
                    else:
                        message = HumanMessage(content=text)
                except ValueError, TypeError:
                    raise ValueError(
                        f"Invalid role value '{turn_speaker}' for turn {turn_number}"
                    )
            else:
                raise ValueError(f"Speaker is not provided for turn {turn_number}")

            msg_type = type(message).__name__
            preview = text[:50] + "..." if len(text) > 50 else text
            debug_print(f"  Turn {turn_number} -> {msg_type}: {preview}")
            messages.append(message)

    # Apply persona role reminder to the last provider message if conditions are met.
    if role == Role.PERSONA:
        apply_persona_role_reminder(messages, conversation_history)

    return messages


def format_conversation_as_string(
    role: Role,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Format conversation history as a string for string-based LLMs (e.g., Ollama).

    This function reuses build_langchain_messages() and converts the result to
    a string format with Human/Assistant labels.

    Args:
        role: Role of the LLM (Role.PERSONA or Role.PROVIDER)
        conversation_history: Optional list of previous conversation turns
        system_prompt: Optional system prompt to prepend

    Returns:
        Formatted string with System, Human, and Assistant labels
    """
    full_message = ""

    # Add system prompt if provided
    if system_prompt:
        full_message = f"System: {system_prompt}\n\n"

    # Build LangChain messages using existing utility
    messages = build_langchain_messages(role, conversation_history)

    # Convert messages to string format
    for message in messages:
        if isinstance(message, HumanMessage):
            full_message += f"Human: {message.text}\n\n"
        elif isinstance(message, AIMessage):
            full_message += f"Assistant: {message.text}\n\n"

    # Add "Assistant:" prompt at the end to signal the LLM should respond
    if messages and full_message.endswith("\n\n"):
        full_message += "Assistant:"

    return full_message
