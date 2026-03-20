#!/usr/bin/env python3
"""
Truncate conversation files to a fixed number of turns.

Each turn starts with "user:" or "chatbot:" at the beginning of a line.
Multi-line content after that prefix belongs to the same turn until the next
"user:" or "chatbot:" line.

Usage:
  python3 scripts/truncate_conversations.py -i conversations/HEOR_FIXED -o conversations/HEOR_FIXED_2turn -t 2
"""

import argparse
import re
from pathlib import Path


def split_into_turns(text: str) -> list[tuple[str, str]]:
    """
    Split conversation text into turns. Returns list of (role, content) where
    role is 'user' or 'chatbot' and content includes the rest of that turn
    (including newlines) up to the next turn.
    """
    turns = []
    # Match line that starts with "user:" or "chatbot:" (at start of string or after newline)
    pattern = re.compile(r"^(user|chatbot):\s*", re.IGNORECASE | re.MULTILINE)
    pos = 0
    text = text.rstrip()
    if not text:
        return turns
    while True:
        m = pattern.search(text, pos)
        if m is None:
            break
        role = m.group(1).lower()
        start = m.end()
        next_m = pattern.search(text, start)
        if next_m is None:
            content = text[start:].rstrip()
        else:
            content = text[start : next_m.start()].rstrip()
        turns.append((role, content))
        if next_m is None:
            break
        pos = next_m.start()
    return turns


def turns_to_text(turns: list[tuple[str, str]]) -> str:
    """Format turns back into conversation text."""
    parts = []
    for role, content in turns:
        prefix = "user:" if role == "user" else "chatbot:"
        if content:
            parts.append(f"{prefix} {content}")
        else:
            parts.append(f"{prefix}")
    return "\n\n".join(parts)


def truncate_file(content: str, max_turns: int) -> str:
    """Truncate conversation to first max_turns turns."""
    turns = split_into_turns(content)
    kept = turns[:max_turns]
    return turns_to_text(kept)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Truncate conversation files to a fixed number of turns."
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        required=True,
        help="Input directory containing conversation .txt files (searched recursively)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory; structure under input will be mirrored here",
    )
    parser.add_argument(
        "-t",
        "--turns",
        type=int,
        required=True,
        help="Maximum number of turns to keep (e.g. 2 for user + chatbot)",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Optional suffix for output filenames (e.g. _2turn); default keeps same name",
    )
    args = parser.parse_args()

    if args.turns < 1:
        parser.error("--turns must be at least 1")

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir():
        parser.error(f"Input is not a directory: {input_dir}")

    txt_files = list(input_dir.rglob("*.txt"))
    if not txt_files:
        parser.error(f"No .txt files found under {input_dir}")

    for path in sorted(txt_files):
        rel = path.relative_to(input_dir)
        out_path = output_dir / rel
        if args.suffix:
            out_path = out_path.with_stem(out_path.stem + args.suffix)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        content = path.read_text(encoding="utf-8", errors="replace")
        truncated = truncate_file(content, args.turns)
        out_path.write_text(truncated, encoding="utf-8")

    print(f"Truncated {len(txt_files)} file(s) to {args.turns} turn(s) -> {output_dir}")


if __name__ == "__main__":
    main()
