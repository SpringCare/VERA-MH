"""Parse VERA-style conversation transcripts for role-reversal analysis."""

import re
from pathlib import Path
from typing import List, Optional, Tuple

# Run folder from generate.py: p_{user}__a_{provider}__t{N}__r{M}__{timestamp}
RUN_FOLDER_PATTERN = re.compile(r"^p_.+__a_.+__t\d+__r\d+__.+$")


def is_run_folder_name(name: str) -> bool:
    """Return True if dirname matches generated conversation run folder pattern."""
    return bool(RUN_FOLDER_PATTERN.match(name))


def parse_user_and_provider_from_run_folder(dir_name: str) -> Optional[Tuple[str, str]]:
    """
    Extract user-model slug and provider-model slug from a run folder basename.

    Returns:
        (user_slug, provider_slug) or None if the name does not match.
    """
    if not dir_name.startswith("p_") or "__a_" not in dir_name:
        return None
    without_p = dir_name[2:]
    parts = without_p.split("__a_", 1)
    if len(parts) != 2:
        return None
    user_slug, after_a = parts
    if "__t" not in after_a:
        return None
    provider_slug, _ = after_a.split("__t", 1)
    if not user_slug or not provider_slug:
        return None
    return user_slug, provider_slug


def _normalize_speaker(raw: str) -> str:
    key = raw.lower()
    if key in ("user", "persona"):
        return "user"
    if key in ("chatbot", "assistant"):
        return "chatbot"
    return key


def parse_transcript_to_turns(text: str) -> List[Tuple[str, str, int]]:
    """
    Parse transcript text into (speaker, body, dialogue_turn_index).

    Speakers are normalized to 'user' or 'chatbot'. dialogue_turn_index is
    1-based in the full dialogue (each message block is one turn).

    Skips lines that look like metadata, e.g. [CONVERSATION ENDED - ...].
    """
    lines = text.splitlines()
    turns: List[Tuple[str, str, int]] = []
    current_speaker: Optional[str] = None
    current_lines: List[str] = []
    dialogue_turn = 0

    def flush() -> None:
        nonlocal current_speaker, current_lines, dialogue_turn
        if current_speaker is None:
            return
        body = "\n".join(current_lines).strip()
        dialogue_turn += 1
        turns.append((current_speaker, body, dialogue_turn))
        current_speaker = None
        current_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[CONVERSATION ENDED"):
            continue
        m = re.match(
            r"^(user|chatbot|assistant|persona)\s*:\s*(.*)$",
            line,
            re.IGNORECASE,
        )
        if m:
            flush()
            current_speaker = _normalize_speaker(m.group(1))
            first = m.group(2)
            current_lines = [first] if first else []
        elif current_speaker is not None:
            current_lines.append(line)

    flush()
    return turns


def iter_user_messages_with_turns(
    text: str,
) -> List[Tuple[int, int, str]]:
    """
    Return list of (user_message_index, dialogue_turn_index, body) for user turns.

    user_message_index is 1-based among user messages only.
    """
    turns = parse_transcript_to_turns(text)
    out: List[Tuple[int, int, str]] = []
    user_idx = 0
    for speaker, body, dialogue_turn in turns:
        if speaker != "user":
            continue
        user_idx += 1
        out.append((user_idx, dialogue_turn, body))
    return out


def discover_conversation_txt_files(root: Path) -> List[Path]:
    """All .txt files under root whose parent folder name is a valid run folder."""
    root = root.resolve()
    found: List[Path] = []
    for path in sorted(root.rglob("*.txt")):
        if path.is_file() and is_run_folder_name(path.parent.name):
            found.append(path)
    return found


def conversation_relpath(path: Path, root: Path) -> str:
    """Path relative to root using POSIX separators for stable CSV output."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name
