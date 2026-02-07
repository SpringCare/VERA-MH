#!/usr/bin/env python3
"""
Main script for judging existing conversations using the LLM Judge system.
This script is separate from conversation generation.
"""

import asyncio

from judge.cli import get_parser, main

if __name__ == "__main__":
    args = get_parser().parse_args()
    print(f"Running judge on: {args.folder or args.conversation}")
    asyncio.run(main(args))
