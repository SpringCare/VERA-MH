#!/usr/bin/env python3
"""
Main script for judging existing conversations using the LLM Judge system.
This script is separate from conversation generation.
"""

import argparse
import asyncio

from judge import judge_conversations, judge_single_conversation
from judge.llm_judge import LLMJudge
from utils.utils import parse_key_value_list


async def main(args):
    """Main async entrypoint for judging conversations."""
    # Parse judge models from args (supports "model" or "model:count" format)
    judge_models = {}
    for model_spec in args.judge_model:
        if ":" in model_spec:
            # Format: "model:count"
            model, count = model_spec.rsplit(":", 1)
            judge_models[model] = int(count)
        else:
            # Format: "model" (defaults to 1 instance)
            judge_models[model_spec] = 1

    models_str = ", ".join(f"{model}x{count}" for model, count in judge_models.items())
    print(f"🎯 LLM Judge | Models: {models_str} | Rubrics: {', '.join(args.rubrics)}")

    if args.conversation:
        # Single conversation with first judge model (single instance)
        first_model = next(iter(judge_models.keys()))
        judge = LLMJudge(
            judge_model=first_model,
            judge_model_extra_params=args.judge_model_extra_params,
        )
        await judge_single_conversation(
            judge, args.conversation, args.rubrics, args.output
        )
    else:
        # Batch evaluation with multiple judges
        await judge_conversations(
            judge_models=judge_models,
            conversation_folder=args.folder,
            max_concurrent=args.max_concurrent,
            rubrics=args.rubrics,
            output_root=args.output,
            limit=args.limit,
            verbose=True,
            judge_model_extra_params=args.judge_model_extra_params,
            per_judge=args.per_judge,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Judge existing LLM conversations using rubrics"
    )

    # required source
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--conversation", "-c", help="Path to a single conversation file to judge"
    )
    source_group.add_argument(
        "--folder",
        "-f",
        default="conversations",
        help="Folder containing conversation files (default: conversations)",
    )

    # rubrics
    parser.add_argument(
        "--rubrics",
        "-r",
        nargs="+",
        default=["data/rubric.tsv"],
        help="Rubric file(s) to use (default: data/rubric.tsv)",
    )

    # model
    parser.add_argument(
        "--judge-model",
        "-j",
        nargs="+",
        required=True,
        help=(
            "Model(s) to use for judging. "
            "Format: 'model' or 'model:count' for multiple instances. "
            "Can specify multiple models: --judge-model model1 model2:3. "
            "Examples: claude-3-5-sonnet-20241022, "
            "claude-3-5-sonnet-20241022:3, "
            "claude-3-5-sonnet-20241022:2 gpt-4o:1"
        ),
    )

    parser.add_argument(
        "--judge-model-extra-params",
        "-jep",
        help=(
            "Extra parameters for the judge model. "
            "Examples: temperature=0.7, max_tokens=1000. "
            "Default: temperature=0 (unless overridden)"
        ),
        type=parse_key_value_list,
        default={},
    )

    # optional limit
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Limit number of conversations to judge (for debugging)",
    )

    # output folder
    parser.add_argument(
        "--output",
        "-o",
        default="evaluations",
        help="Output folder for evaluation results (default: evaluations)",
    )

    # concurrency control
    parser.add_argument(
        "--max-concurrent",
        "-m",
        type=int,
        default=None,
        help=(
            "Maximum number of concurrent workers (default: None). "
            "Set to a high number or omit for unlimited concurrency."
        ),
    )

    parser.add_argument(
        "--per-judge",
        action="store_true",
        help=(
            "If set, --max-concurrent applies per judge model. "
            "Otherwise, it applies to total workers across all judges."
        ),
    )

    args = parser.parse_args()

    print(f"Running judge on: {args.folder or args.conversation}")
    asyncio.run(main(args))
