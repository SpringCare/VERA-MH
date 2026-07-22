#!/usr/bin/env python3
"""
Main script for judging existing conversations using the LLM Judge system.
This script is separate from conversation generation.
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from judge import judge_conversations, judge_single_conversation
from judge.llm_judge import LLMJudge
from judge.rubric_config import ConversationData, RubricConfig, load_conversations
from judge.utils import (
    build_judge_task_log_path,
    default_adhoc_parent,
    parse_judge_models,
)
from utils.conversation_layout import resolve_conversation_input
from utils.naming import (
    build_single_conversation_run_folder_name,
    is_judge_run_folder_basename,
)
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
        "(nested: p_*__/conversations/, or legacy flat folder of .txt files)",
    )

    # rubrics
    parser.add_argument(
        "--rubrics",
        "-r",
        nargs="+",
        default=["data/rubric_manifest.json"],
        help=(
            "Rubric bundle manifest(s) to use "
            "(default: data/rubric_manifest.json). "
            "Only the first is used; multi-rubric support is not yet implemented."
        ),
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
        default=None,
        help=(
            "Batch: parent directory for a new j_*__* folder (default: "
            "<gen_run>/evaluations/ when -f points at a nested p_* run, else "
            "evaluations/). With --resume, must be the existing j_* run folder. "
            "Single-file (-c): parent for single_<ts>__<stem>/ "
            "(default: output/adhoc; env VERA_ADHOC_PARENT overrides)."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help=(
            "Resume a previous evaluation run from an existing output folder and "
            "skip conversation/judge-instance TSVs that already exist."
        ),
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

    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Enable debug logging (structured LLM responses, message previews)",
    )

    return parser


async def main(args) -> Optional[str]:
    """Main async entrypoint for judging conversations."""
    if args.debug:
        from utils.debug import set_debug

        set_debug(True)

    # Parse judge models from args (supports "model" or "model:count" format)
    judge_models = parse_judge_models(args.judge_model)

    models_str = ", ".join(f"{model}x{count}" for model, count in judge_models.items())
    print(f"🎯 LLM Judge | Models: {models_str}")

    if len(args.rubrics) > 1:
        print(
            f"Warning: multiple rubrics passed ({args.rubrics}); "
            f"multi-rubric support is not yet implemented, "
            f"using only the first: {args.rubrics[0]}",
            file=sys.stderr,
        )

    # Load rubric configuration once at startup
    print("📚 Loading rubric configuration...")
    rubric_config = await RubricConfig.load_bundle(args.rubrics[0])

    if args.conversation:
        # Single conversation with first judge model (single instance)
        first_model = next(iter(judge_models.keys()))

        # Load single conversation
        conversation = await ConversationData.load(args.conversation)

        stem = Path(args.conversation).stem
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        single_name = build_single_conversation_run_folder_name(stem, ts)
        if args.output is None:
            parent = default_adhoc_parent()
        else:
            parent = args.output
        os.makedirs(parent, exist_ok=True)
        out_run = os.path.join(parent, single_name)
        os.makedirs(out_run, exist_ok=True)

        conv_filename = Path(args.conversation).name
        metadata = getattr(conversation, "metadata", None)
        if isinstance(metadata, dict):
            conv_filename = metadata.get("filename", conv_filename)
        log_file = build_judge_task_log_path(
            conv_filename,
            first_model,
            output_folder=out_run,
        )

        # Create judge with rubric config
        judge = LLMJudge(
            judge_model=first_model,
            rubric_config=rubric_config,
            judge_model_extra_params=args.judge_model_extra_params,
            log_file=log_file,
        )
        await judge_single_conversation(judge, conversation, out_run)
        print(f"Evaluation output: {out_run}/")
        return out_run

    transcripts_dir, gen_run, conv_basename = resolve_conversation_input(args.folder)

    print(f"📂 Loading conversations from {transcripts_dir}...")
    conversations = await load_conversations(transcripts_dir, limit=args.limit)
    print(f"✅ Loaded {len(conversations)} conversations")

    judge_kwargs = dict(
        judge_models=judge_models,
        conversations=conversations,
        rubric_config=rubric_config,
        max_concurrent=args.max_concurrent,
        conversation_folder_name=conv_basename,
        verbose=True,
        judge_model_extra_params=args.judge_model_extra_params,
        per_judge=args.per_judge,
        verbose_workers=args.verbose_workers,
        resume=args.resume,
    )
    if args.resume:
        if not args.output:
            raise ValueError(
                "Resume mode requires --output to point to an existing evaluation "
                "run folder (j_*__*)."
            )
        if not os.path.isdir(args.output):
            raise ValueError(
                "Resume mode requires --output to point to an existing "
                "evaluation run folder."
            )
        base = os.path.basename(os.path.normpath(args.output))
        if not is_judge_run_folder_basename(base):
            raise ValueError(
                "Resume mode requires --output to be a judge run folder "
                f"(basename like j_*__*), got {base!r}"
            )
        judge_kwargs["output_folder"] = args.output
    else:
        if args.output is None:
            if gen_run is not None:
                output_root = os.path.join(gen_run, "evaluations")
            else:
                output_root = "evaluations"
                print(
                    "Note: flat conversation folder; writing evaluations under "
                    "evaluations/. New runs use output/p_*__/conversations/.",
                    file=sys.stderr,
                )
        else:
            output_root = args.output
        judge_kwargs["output_root"] = output_root

    _, output_folder = await judge_conversations(**judge_kwargs)

    print(f"Evaluation output: {output_folder}/")
    return output_folder


if __name__ == "__main__":
    args = get_parser().parse_args()
    print(f"Running judge on: {args.folder or args.conversation}")
    asyncio.run(main(args))
