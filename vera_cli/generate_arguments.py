"""Flags and CLI-only defaults for ``vera generate``."""

from __future__ import annotations

import argparse
from typing import Any

from .config import VERA_RUN_CONFIG_ENV

DEFAULTS: dict[str, Any] = {
    "turns": 3,
    "output": "output",
    "max_concurrent": None,
    "max_total_words": None,
    "persona_speaks_first": True,
    "sessions": None,
}

RUN_FIELDS = (
    "chatbot",
    "user",
    "target",
    "personas",
    "turns",
    "output",
    "max_concurrent",
    "max_total_words",
    "provider_speaks_first",
    "sessions",
)


def add_generate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-c",
        "--chatbot",
        default=argparse.SUPPRESS,
        help="Chatbot model under test (required with CLI-defined runs)",
    )
    parser.add_argument(
        "-u",
        "--user",
        nargs="+",
        metavar="<model>[:<repeats>]",
        default=argparse.SUPPRESS,
        help="User-side model(s) and full persona-set repeats",
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--target",
        default=argparse.SUPPRESS,
        help="Complete target name or manifest path; use 'all' for every target",
    )
    target.add_argument(
        "--personas",
        default=argparse.SUPPRESS,
        help="Target name or manifest path whose personas and prompt should be used",
    )
    parser.add_argument(
        "-t",
        "--turns",
        type=int,
        default=argparse.SUPPRESS,
        help="Maximum conversation turns (default: 3)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=argparse.SUPPRESS,
        help="Parent directory for generation runs (default: output)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=argparse.SUPPRESS,
        help="Maximum concurrent conversations (default: unlimited)",
    )
    parser.add_argument(
        "--max-total-words",
        type=int,
        default=argparse.SUPPRESS,
        help="Maximum response words per conversation (default: unlimited)",
    )
    parser.add_argument(
        "--provider-speaks-first",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Have the chatbot speak first (default: persona speaks first)",
    )
    parser.add_argument(
        "--sessions",
        default=argparse.SUPPRESS,
        help="Comma-separated session types to run in order",
    )
    parser.add_argument(
        "--config",
        help=f"JSON path or '-' for stdin; alternatively set {VERA_RUN_CONFIG_ENV}",
    )
    parser.add_argument(
        "--sample", type=int, help="Debug-only cap on personas loaded per file"
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Enable debug logging"
    )
    parser.add_argument(
        "--print",
        action="store_true",
        dest="print_only",
        help="Print the resolved invocation without executing it",
    )


def explicit_fields(args: argparse.Namespace) -> list[str]:
    return [field for field in RUN_FIELDS if hasattr(args, field)]


def cli_value(args: argparse.Namespace, field: str) -> Any:
    if hasattr(args, field):
        return getattr(args, field)
    if field == "provider_speaks_first":
        return not DEFAULTS["persona_speaks_first"]
    return DEFAULTS[field]


def parse_sessions(value: str | None) -> list[str] | None:
    if value is None:
        return None
    sessions = [session.strip() for session in value.split(",") if session.strip()]
    if not sessions:
        from .config import ConfigError

        raise ConfigError("--sessions must contain at least one non-empty name")
    return sessions
