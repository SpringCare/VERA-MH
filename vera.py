#!/usr/bin/env python3
"""VERA-MH unified command-line entry point."""

from __future__ import annotations

import argparse
from typing import Optional

from vera_cli.config import ConfigError
from vera_cli.generate import register as register_generate
from vera_cli.judge import register as register_judge


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vera", description="VERA-MH unified CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Each command owns its flags and attaches its handler to its subparser.
    register_generate(subparsers)
    register_judge(subparsers)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()

    # Parsing selects a command and gives its handler the resolved CLI namespace.
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except ConfigError as error:
        # Present expected input/configuration failures as standard CLI errors.
        parser.error(str(error))
        return 2  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
