#!/usr/bin/env python3
"""
Main script for judging existing conversations using the LLM Judge system.
This script is separate from conversation generation.
"""

import argparse
import asyncio
from typing import Optional

from judge import judge_conversations, judge_single_conversation
from judge.llm_judge import LLMJudge
from judge.rubric_config import ConversationData, RubricConfig, load_conversations
from judge.utils import parse_judge_models
from utils.utils import parse_key_value_list


def get_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser (for CLI and testing)."""
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
        help="Path to a conversation run folder "
        "(e.g. conversations/p_model__a_model__t6__r1__timestamp/)",
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
            "Examples: claude-sonnet-4-5-20250929, "
            "claude-sonnet-4-5-20250929:3, "
            "claude-sonnet-4-5-20250929:2 gpt-4o:1"
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
        "-pj",
        action="store_true",
        help=(
            "If set, --max-concurrent applies per judge model. "
            "Otherwise, it applies to total workers across all judges."
        ),
    )

    parser.add_argument(
        "--verbose-workers",
        "-vw",
        action="store_true",
        help="Enable verbose worker logging to show concurrency behavior",
    )

    return parser


async def main(args) -> Optional[str]:
    """Main async entrypoint for judging conversations."""
    # Parse judge models from args (supports "model" or "model:count" format)
    judge_models = parse_judge_models(args.judge_model)

    models_str = ", ".join(f"{model}x{count}" for model, count in judge_models.items())
    print(f"🎯 LLM Judge | Models: {models_str}")

    # Load rubric configuration once at startup
    print("📚 Loading rubric configuration...")
    rubric_config = await RubricConfig.load(rubric_folder="data")

    if args.conversation:
        # Single conversation with first judge model (single instance)
        first_model = next(iter(judge_models.keys()))

        # Load single conversation
        conversation = await ConversationData.load(args.conversation)

        # Create judge with rubric config
        judge = LLMJudge(
            judge_model=first_model,
            rubric_config=rubric_config,
            judge_model_extra_params=args.judge_model_extra_params,
        )
        await judge_single_conversation(judge, conversation, args.output)
        # Single conversation mode doesn't need output folder for pipeline
        print("ℹ️  Single conversation mode: output folder not needed for pipeline")
        return None
    else:
        # Load all conversations at startup
        print(f"📂 Loading conversations from {args.folder}...")
        conversations = await load_conversations(args.folder, limit=args.limit)
        print(f"✅ Loaded {len(conversations)} conversations")

        # Batch evaluation with multiple judges
        from pathlib import Path

        folder_name = Path(args.folder).name

        _, output_folder = await judge_conversations(
            judge_models=judge_models,
            conversations=conversations,
            rubric_config=rubric_config,
            max_concurrent=args.max_concurrent,
            output_root=args.output,
            conversation_folder_name=folder_name,
            verbose=True,
            judge_model_extra_params=args.judge_model_extra_params,
            per_judge=args.per_judge,
            verbose_workers=args.verbose_workers,
        )

        return output_folder


if __name__ == "__main__":
    args = get_parser().parse_args()
    print(f"Running judge on: {args.folder or args.conversation}")
    asyncio.run(main(args))
