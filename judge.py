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
from typing import Dict, Optional

from judge import judge_single_conversation, run_judging
from judge.llm_judge import LLMJudge
from judge.rubric_config import ConversationData, RubricConfig
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
from utils.rubric_manifest import load_manifest
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
        default=["data/SI/rubric_manifest.json"],
        help=(
            "Rubric bundle manifest(s) to use "
            "(default: data/SI/rubric_manifest.json). "
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


async def _resolve_rubric_paths(manifest_path: str) -> Dict[str, str]:
    """Resolve a rubric bundle manifest to its three concrete file paths.

    Manifest paths are relative to the manifest's own folder. Doing this here
    keeps manifest reading in the CLI layer, so the domain receives paths.
    """
    manifest = await load_manifest(manifest_path)
    folder = Path(manifest_path).parent
    return {
        "rubric_file": str(folder / manifest["rubric_file"]),
        "rubric_prompt_beginning_file": str(
            folder / manifest["rubric_prompt_beginning_file"]
        ),
        "question_prompt_file": str(folder / manifest["question_prompt_file"]),
    }


def _resolve_output_target(
    args, gen_run: Optional[str]
) -> tuple[Optional[str], Optional[str]]:
    """Decide where evaluations go, returning ``(output_root, output_folder)``.

    Exactly one is non-None. ``output_root`` is a parent to mint a new ``j_*``
    run folder under; ``output_folder`` is an exact existing folder to write
    into, which is how resuming lands back in the same place instead of
    starting a new run.

    This is CLI policy, which is why it lives here rather than in the domain.
    """
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
        return None, args.output

    if args.output is not None:
        return args.output, None
    if gen_run is not None:
        return os.path.join(gen_run, "evaluations"), None

    print(
        "Note: flat conversation folder; writing evaluations under "
        "evaluations/. New runs use output/p_*__/conversations/.",
        file=sys.stderr,
    )
    return "evaluations", None


async def main(args) -> Optional[str]:
    """Legacy CLI entry point: resolve ``args``, then call the judging domain.

    This is CLI glue, not a domain entry point. It owns everything specific to
    this script's argument conventions — model shorthand parsing, manifest
    reading, output-location policy, resume validation, and debug setup — and
    hands fully resolved values to `judge.run_judging`.

    `vera judge` does not call this. It calls `run_judging` directly and
    resolves its own inputs, so nothing new belongs here: this script is
    retained only until `vera resume` exists (see docs/architecture.md).
    """
    if args.debug:
        from utils.debug import set_debug

        set_debug(True)

    # Parse judge models from args (supports "model" or "model:count" format)
    judge_models = parse_judge_models(args.judge_model)

    if len(args.rubrics) > 1:
        print(
            f"Warning: multiple rubrics passed ({args.rubrics}); "
            f"multi-rubric support is not yet implemented, "
            f"using only the first: {args.rubrics[0]}",
            file=sys.stderr,
        )

    rubric_paths = await _resolve_rubric_paths(args.rubrics[0])

    if args.conversation:
        # Single-conversation judging is legacy-only: `vera judge` drops this
        # mode, so it is not part of the resolved-value domain entry point.
        models_str = ", ".join(
            f"{model}x{count}" for model, count in judge_models.items()
        )
        print(f"🎯 LLM Judge | Models: {models_str}")
        print("📚 Loading rubric configuration...")
        rubric_config = await RubricConfig.from_paths(**rubric_paths)

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
    output_root, output_folder = _resolve_output_target(args, gen_run)

    _, output_folder = await run_judging(
        judge_models=judge_models,
        **rubric_paths,
        transcripts_dir=transcripts_dir,
        conversation_folder_name=conv_basename,
        limit=args.limit,
        output_root=output_root,
        output_folder=output_folder,
        judge_model_extra_params=args.judge_model_extra_params,
        max_concurrent=args.max_concurrent,
        per_judge=args.per_judge,
        verbose_workers=args.verbose_workers,
        verbose=True,
        resume=args.resume,
    )

    print(f"Evaluation output: {output_folder}/")
    return output_folder


if __name__ == "__main__":
    args = get_parser().parse_args()
    print(f"Running judge on: {args.folder or args.conversation}")
    asyncio.run(main(args))
