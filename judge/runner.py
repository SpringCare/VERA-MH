#!/usr/bin/env python3
"""
Judge Runner - High-level functions for batch conversation evaluation.
Contains the main logic extracted from main_judge.py to reduce code duplication.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .llm_judge import LLMJudge


async def batch_evaluate_with_individual_judges(
    conversation_file_paths: List[str],
    rubrics: List[str],
    judge_model: str,
    output_folder: str,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Evaluate multiple conversations, creating a new LLMJudge instance for each conversation.

    Args:
        conversation_file_paths: List of conversation file paths
        rubrics: List of rubric names to use
        judge_model: Model to use for judging
        output_folder: Folder to save evaluation results
        limit: Optional limit on number of conversations to evaluate

    Returns:
        List of evaluation results
    """
    # Apply limit if specified
    if limit is not None:
        conversation_file_paths = conversation_file_paths[:limit]

    results = []
    total_files = len(conversation_file_paths)

    for i, conversation_file in enumerate(conversation_file_paths, 1):
        print(f"📄 ({i}/{total_files}) {Path(conversation_file).name}")

        # Create a new LLMJudge instance for this conversation
        judge = LLMJudge(judge_model=judge_model)

        # Evaluate conversation with auto-save enabled
        evaluation = await judge.evaluate_conversation(
            conversation_file, output_folder=output_folder, auto_save=True
        )
        results.append(evaluation)

    return results


async def judge_conversations(
    judge_model: str,
    conversation_folder: str,
    rubrics: List[str] = ["rubric.csv"],
    output_root: str = "evaluations",
    limit: Optional[int] = None,
    verbose: bool = True,
    output_folder: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Judge conversations in a folder and return results.

    Args:
        conversation_folder: Folder containing conversation files
        rubrics: List of rubric names to use
        judge_model: Model to use for judging
        output_folder: Output folder for evaluation results
        limit: Optional limit on number of files to process
        verbose: Whether to print status messages

    Returns:
        List of evaluation results

    Raises:
        ValueError: Configuration error
        Exception: Other errors
    """
    if output_folder is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        output_folder = f"{output_root}/j_{judge_model}_{timestamp}__{Path(conversation_folder).name}"

    os.makedirs(output_folder, exist_ok=True)

    if verbose:
        print(f"🎯 LLM Judge | Model: {judge_model} | Rubrics: {', '.join(rubrics)}")

    # Check folder exists
    folder_path = Path(conversation_folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {conversation_folder}")

    # Find conversation files
    conversation_files = list(folder_path.glob("*.txt"))
    if not conversation_files:
        raise FileNotFoundError(f"No .txt files found in: {conversation_folder}")

    total_found = len(conversation_files)

    if limit:
        conversation_files = conversation_files[:limit]
        if verbose:
            print(f"🔍 Found {total_found} files, judging {limit} (debug mode)")
    else:
        if verbose:
            print(f"🔍 Found {total_found} files to judge")

    # Convert to strings
    conversation_file_paths = [str(f) for f in conversation_files]

    # Run batch evaluation with individual judges
    results = await batch_evaluate_with_individual_judges(
        conversation_file_paths, rubrics, judge_model, output_folder, limit=limit
    )

    if verbose:
        print(f"✅ Completed {len(results)} evaluations → {output_folder}/")

    return results


async def judge_single_conversation(
    judge: LLMJudge, conversation_file: str, rubrics: List[str], output_folder: str
) -> Optional[Dict[str, Any]]:
    """
    Judge a single conversation file.

    Args:
        judge: LLMJudge instance
        conversation_file: Path to conversation file
        rubrics: List of rubric names to use
        output_folder: Output folder for results

    Returns:
        Evaluation results or None if failed
    """
    if not Path(conversation_file).exists():
        print(f"❌ File not found: {conversation_file}")
        return None

    print(f"📄 Judging: {Path(conversation_file).name}")

    result = await judge.evaluate_conversation(
        conversation_file, output_folder=output_folder, auto_save=True
    )

    print(f"🟢 Done: {Path(conversation_file).name}")
    return result


async def judge_conversation_folder(
    judge_model: str,
    folder: str,
    rubrics: List[str],
    output_folder: str,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Judge all conversations in a folder, creating individual LLMJudge instances.

    Args:
        judge_model: Model to use for judging
        folder: Folder containing conversation files
        rubrics: List of rubric names to use
        output_folder: Output folder for results
        limit: Optional limit on number of files

    Returns:
        List of evaluation results
    """
    folder_path = Path(folder)
    if not folder_path.exists():
        print(f"❌ Folder not found: {folder}")
        return []

    # Find all conversation files
    conversation_files = list(folder_path.glob("*.txt"))
    if not conversation_files:
        print(f"❌ No .txt files found in: {folder}")
        return []

    total_found = len(conversation_files)

    if limit:
        conversation_files = conversation_files[:limit]
        print(f"🔍 Found {total_found} files, judging {limit} (debug mode)")
    else:
        print(f"🔍 Found {total_found} files to judge")

    # Convert to strings
    conversation_file_paths = [str(f) for f in conversation_files]

    try:
        results = await batch_evaluate_with_individual_judges(
            conversation_file_paths, rubrics, judge_model, output_folder, limit=limit
        )

        print(f"✅ Completed {len(results)} evaluations → {output_folder}/")
        return results

    except Exception as e:
        print(f"❌ Batch evaluation failed: {e}")
        return []
