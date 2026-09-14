"""Resolved-input entry point for judging conversations.

This is the judging domain's application function: it receives fully resolved
values, loads what those values point at, and runs the evaluation. It is the
counterpart of `generate_conversations.run.run_generation` on the generation
side, and it is what `vera judge` calls.

It deliberately does none of the CLI's work. It does not parse arguments, read
a target manifest, apply defaults, choose an output location, or configure
debug logging — every one of those is resolved by the caller. That boundary is
what lets one function serve both the unified CLI and the legacy `judge.py`
script without either one's conventions leaking into the domain.

Unlike the generation side, this module already sits in the permanent package,
so no temporary root-level boundary function is needed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .rubric_config import RubricConfig, load_conversations
from .runner import judge_conversations


async def run_judging(
    *,
    judge_models: Dict[str, int],
    rubric_file: str,
    rubric_prompt_beginning_file: str,
    question_prompt_file: str,
    transcripts_dir: str,
    conversation_folder_name: Optional[str],
    limit: Optional[int],
    output_dir: str,
    is_existing_run: bool,
    judge_model_extra_params: Dict[str, Any],
    max_concurrent: Optional[int],
    per_judge: bool,
    verbose_workers: bool,
    verbose: bool,
    resume: bool,
) -> tuple[List[Dict[str, Any]], str]:
    """Evaluate a folder of conversations from fully resolved inputs.

    Args:
        judge_models: Judge model name to instance count, already parsed from
            whatever shorthand the caller accepts
        rubric_file: Resolved path to the rubric TSV
        rubric_prompt_beginning_file: Resolved path to the system prompt template
        question_prompt_file: Resolved path to the question prompt template
        transcripts_dir: Resolved directory holding the conversation `.txt`
            files. The caller has already decided whether this is a nested
            `conversations/` directory or a legacy flat folder.
        conversation_folder_name: Basename recorded in output paths, or None
        limit: Cap on conversations loaded, or None for all
        output_dir: Where evaluations go. The caller has already decided this;
            `is_existing_run` says which of the two things it is.
        is_existing_run: False (the normal case, and the only one `vera judge`
            uses) means `output_dir` is a parent to mint a new `j_*` run folder
            under. True means it is an exact existing run folder to write into,
            bypassing run naming -- how `judge.py --resume` lands back in the
            same folder.
        judge_model_extra_params: Provider parameters for the judge model
        max_concurrent: Worker ceiling, or None for unlimited
        per_judge: Whether `max_concurrent` applies per judge model or in total
        verbose_workers: Whether workers log concurrency behavior
        verbose: Whether to print progress
        resume: Whether to skip evaluation TSVs that already exist

    Returns:
        Tuple of (results, output_folder) where output_folder is where the
        evaluations were written.

    Raises:
        FileNotFoundError: If a rubric file or the transcripts directory is
            missing.
    """
    if verbose:
        models_str = ", ".join(
            f"{model}x{count}" for model, count in judge_models.items()
        )
        print(f"🎯 LLM Judge | Models: {models_str}")
        print("📚 Loading rubric configuration...")

    rubric_config = await RubricConfig.from_paths(
        rubric_file=rubric_file,
        rubric_prompt_beginning_file=rubric_prompt_beginning_file,
        question_prompt_file=question_prompt_file,
    )

    if verbose:
        print(f"📂 Loading conversations from {transcripts_dir}...")
    conversations = await load_conversations(transcripts_dir, limit=limit)
    if verbose:
        print(f"✅ Loaded {len(conversations)} conversations")

    # `judge_conversations` still distinguishes the two output modes by which
    # keyword it receives; translating the caller's flag is this layer's job.
    output_target: Dict[str, Any] = (
        {"output_folder": output_dir}
        if is_existing_run
        else {"output_root": output_dir}
    )

    return await judge_conversations(
        judge_models=judge_models,
        conversations=conversations,
        rubric_config=rubric_config,
        conversation_folder_name=conversation_folder_name,
        verbose=verbose,
        judge_model_extra_params=judge_model_extra_params,
        max_concurrent=max_concurrent,
        per_judge=per_judge,
        verbose_workers=verbose_workers,
        resume=resume,
        **output_target,
    )
