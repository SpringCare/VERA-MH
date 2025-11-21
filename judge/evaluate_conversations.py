#!/usr/bin/env python3
"""
Script to evaluate conversations using the LLM Judge system.
"""

import argparse
import asyncio
from pathlib import Path
from typing import Optional

from .llm_judge import LLMJudge


async def evaluate_single_conversation(
    conversation_file: str, judge_model: str, output_folder: str = "evaluations"
):
    """Evaluate a single conversation and print results."""

    judge = LLMJudge(judge_model=judge_model)

    try:
        print(f"Evaluating: {conversation_file}")

        _ = await judge.evaluate_conversation(conversation_file, output_folder)

        # Print brief results
        # print_evaluation_results(evaluation)

    except Exception as e:
        print(f"Error: {e}")


async def batch_evaluate_conversations(
    conversation_folder: str,
    rubric_files: list,
    judge_model: str,
    limit: Optional[int] = None,
):
    """Evaluate all conversations in a folder."""

    conversation_path = Path(conversation_folder)
    if not conversation_path.exists():
        print(f"Conversation folder not found: {conversation_folder}")
        return

    # Find all conversation files
    conversation_files = list(conversation_path.glob("*.txt"))
    if not conversation_files:
        print(f"No conversation files found in: {conversation_folder}")
        return

    total_found = len(conversation_files)
    files_to_process = min(limit or total_found, total_found)
    print(f"Found {total_found} files, processing {files_to_process}")

    judge = LLMJudge(judge_model=judge_model)

    # Convert Path objects to strings
    conversation_file_paths = [str(f) for f in conversation_files]

    try:
        results = await judge.batch_evaluate(
            conversation_file_paths,
            rubric_files,
            output_folder="evaluations",
            limit=limit,
        )

        print(f"✅ Completed {len(results)} evaluations → evaluations/")

        # Create summary report
        # create_summary_report(results, "evaluations/summary_report.json")

    except Exception as e:
        print(f"Error in batch evaluation: {e}")


async def main():
    """Main function with CLI argument parsing."""

    parser = argparse.ArgumentParser(
        description="Evaluate LLM conversations using rubrics"
    )
    parser.add_argument(
        "--conversation", "-c", help="Single conversation file to evaluate"
    )
    parser.add_argument("--folder", "-f", help="Folder containing conversation files")
    parser.add_argument(
        "--rubrics",
        "-r",
        nargs="+",
        default=["helpfulness", "safety", "communication"],
        help="Rubric files to use (without .json extension)",
    )
    parser.add_argument(
        "--judge-model",
        "-j",
        default="gpt-4",
        help="Model to use for judging (default: gpt-4)",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        help="Limit number of conversations to evaluate (for debugging)",
    )

    args = parser.parse_args()

    if not args.conversation and not args.folder:
        print("Please specify either --conversation or --folder")
        return

    if args.conversation:
        await evaluate_single_conversation(args.conversation, args.judge_model)
    elif args.folder:
        await batch_evaluate_conversations(
            args.folder, args.rubrics, args.judge_model, args.limit
        )


if __name__ == "__main__":
    asyncio.run(main())
