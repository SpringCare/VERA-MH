"""Top-level parser for the unified CLI."""

from __future__ import annotations

import argparse

from .generate_arguments import add_generate_arguments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vera", description="VERA-MH unified CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate", help="Simulate conversations")
    add_generate_arguments(generate)
    return parser
