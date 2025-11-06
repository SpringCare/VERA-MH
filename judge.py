#!/usr/bin/env python3
"""
Main script for judging existing conversations using the LLM Judge system.
This script is separate from conversation generation.
"""

import asyncio
import argparse
from judge import judge_conversations, judge_single_conversation
from judge.llm_judge import LLMJudge


async def main(args):
    """Main async entrypoint for judging conversations."""
    print(f"🎯 LLM Judge | Model: {args.judge_model} | Rubrics: {', '.join(args.rubrics)}")

    #TODO: this judge is used to the single convo case
    # make the API so that it's consisten with one or multi-convo case
    judge = LLMJudge(judge_model=args.judge_model)

    if args.conversation:
        # judge a single conversation file
        await judge_single_conversation(judge, args.conversation, args.rubrics, args.output)
    else:
        # judge all conversations in the folder
        await judge_conversations(
            conversation_folder=args.folder,
            rubrics=args.rubrics,
            judge_model=args.judge_model,
            output_root=args.output,
            limit=args.limit,
            verbose=True,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Judge existing LLM conversations using rubrics")

    # required source
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--conversation", "-c", help="Path to a single conversation file to judge"
    )
    source_group.add_argument(
        "--folder",
        "-f",
        default="conversations",
        help="Folder containing conversation files (default: conversations)"
    )

    # rubrics
    parser.add_argument(
        "--rubrics",
        "-r",
        nargs="+",
        default=["data/rubric.tsv"],
        help="Rubric file(s) to use (default: data/rubric.tsv)"
    )

    # model
    parser.add_argument(
        "--judge-model",
        "-j",
        help="Model to use for judging. Examples: claude-3-5-sonnet-20241022, gemini-1.5-pro, llama3:8b"
    )

    # optional limit
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Limit number of conversations to judge (for debugging)"
    )

    # output folder
    parser.add_argument(
        "--output",
        "-o",
        default="evaluations",
        help="Output folder for evaluation results (default: evaluations)"
    )

    args = parser.parse_args()

    print(f"Running judge on: {args.folder or args.conversation}")
    asyncio.run(main(args))