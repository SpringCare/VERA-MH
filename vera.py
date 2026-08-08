#!/usr/bin/env python3
"""VERA-MH unified command-line entry point."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Optional

from vera_cli.arguments import build_parser
from vera_cli.config import ConfigError


def _handlers() -> dict[str, Callable[[argparse.Namespace], int]]:
    from vera_cli import generate_command

    return {"generate": generate_command.run}


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _handlers()[args.command](args)
    except ConfigError as error:
        parser.error(str(error))
        return 2  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
