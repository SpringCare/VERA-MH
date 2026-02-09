"""
Judge Runner - High-level functions for batch conversation evaluation.
Contains the main logic extracted from main_judge.py to reduce code duplication.
"""

import asyncio
import os
from asyncio import Queue
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from .llm_judge import LLMJudge
from .rubric_config import ConversationData, RubricConfig

# In case this needs to be synced in the meta prompt for the judge
EVALUATION_SEPARATOR = ":"


def _parse_evaluation_to_dict(evaluation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse evaluation results into a flat dictionary.

    Args:
        evaluation: Raw evaluation dict with shape:
            {dimension: {"score": str, "reasoning": str,
                        "yes_question_id": str, "yes_reasoning": str}}

    Returns:
        Flattened dict with dimension scores and metadata
    """
    evaluation_dict = {}
    for dimension, values in evaluation.items():
        evaluation_dict[dimension] = values["score"]
        evaluation_dict[f"{dimension}_yes_question_id"] = values.get(
            "yes_question_id", ""
        )
        evaluation_dict[f"{dimension}_yes_reasoning"] = values.get("yes_reasoning", "")
    return evaluation_dict


async def _evaluate_single_conversation_with_judge(
    conversation: "ConversationData",
    judge_model: str,
    judge_instance: int,
    judge_id: int,
    output_folder: str,
    rubric_config: "RubricConfig",
    judge_model_extra_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluate a single conversation with a single judge model instance.

    Args:
        conversation: ConversationData with content and metadata
        judge_model: Model name to use for judging
        judge_instance: Instance number for this judge (1, 2, 3, ...)
        judge_id: Zero-based ID for this judge (0, 1, 2, ...)
        output_folder: Folder to save evaluation results
        rubric_config: Pre-loaded rubric configuration
        judge_model_extra_params: Extra parameters for the judge model

    Returns:
        Dict with filename, run_id, judge_model, judge_instance, judge_id,
        and all dimension scores
    """
    judge = LLMJudge(
        judge_model=judge_model,
        rubric_config=rubric_config,
        judge_model_extra_params=judge_model_extra_params,
    )

    evaluation = await judge.evaluate_conversation_question_flow(
        conversation,
        output_folder=output_folder,
        auto_save=True,
        verbose=False,
        judge_instance=judge_instance,
    )

    try:
        evaluation_dict = _parse_evaluation_to_dict(evaluation)
    except Exception as e:
        print(
            f"Error parsing evaluation for {judge_model} instance {judge_instance}: {e}"
        )
        print("The following dict is malformed:")
        print(evaluation)
        evaluation_dict = {}

    return {
        "filename": conversation.metadata.get("filename", "unknown.txt"),
        "run_id": conversation.metadata.get("run_id", "unknown"),
        "judge_model": judge_model,
        "judge_instance": judge_instance,
        "judge_id": judge_id,
        **evaluation_dict,
    }


def _create_evaluation_jobs(
    conversations: List[ConversationData],
    judge_models: Dict[str, int],
    output_folder: str,
    rubric_config: RubricConfig,
    judge_model_extra_params: Optional[Dict[str, Any]] = None,
) -> List[
    Tuple[ConversationData, str, int, int, str, RubricConfig, Optional[Dict[str, Any]]]
]:
    """
    Create job tuples for all (conversation × judge × instance) combinations.

    Args:
        conversations: List of ConversationData objects
        judge_models: Dict mapping model names to number of instances
        output_folder: Folder to save evaluation results
        rubric_config: Pre-loaded rubric configuration
        judge_model_extra_params: Extra parameters for the judge model

    Returns:
        List of job tuples:
        (conversation, judge_model, instance, judge_id, output_folder,
        rubric_config, extra_params) where judge_id starts from 0 for each model type
    """
    jobs = []
    for conversation in conversations:
        for judge_model, num_instances in judge_models.items():
            for instance in range(1, num_instances + 1):
                judge_id = instance - 1  # Convert 1-based instance to 0-based judge_id
                jobs.append(
                    (
                        conversation,
                        judge_model,
                        instance,
                        judge_id,
                        output_folder,
                        rubric_config,
                        judge_model_extra_params,
                    )
                )
    return jobs


async def _worker(
    worker_id: Union[int, str],
    queue: Queue,
    results: List[Dict[str, Any]],
    total_jobs: int,
    verbose_workers: bool = False,
):
    """
    Worker that processes evaluation jobs from the queue.

    Args:
        worker_id: Unique identifier for this worker (int or str)
        queue: Queue containing job tuples
        results: Shared list to append results to
        total_jobs: Total number of jobs for progress tracking
        verbose_workers: Enable verbose logging for worker lifecycle
    """
    import time
    from datetime import datetime

    job_count = 0

    if verbose_workers:
        start_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[Worker {worker_id}] Started at {start_time}")

    while True:
        try:
            job = queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        job_count += 1

        (
            conversation,
            judge_model,
            instance,
            judge_id,
            output_folder,
            rubric_config,
            extra_params,
        ) = job

        completed = len(results)
        conversation_filename = conversation.metadata.get("filename", "unknown.txt")

        if verbose_workers:
            print(
                f"[Worker {worker_id}] Processing: {conversation_filename} "
                f"for {judge_model} (judge {instance})"
            )
            job_start = time.time()
        else:
            print(
                f"[Worker {worker_id}] ({completed + 1}/{total_jobs}) "
                f"{conversation_filename} | {judge_model} "
                f"(instance {instance}, id {judge_id})"
            )

        try:
            result = await _evaluate_single_conversation_with_judge(
                conversation,
                judge_model,
                instance,
                judge_id,
                output_folder,
                rubric_config,
                extra_params,
            )
            results.append(result)

            if verbose_workers:
                duration = time.time() - job_start
                print(
                    f"[Worker {worker_id}] Completed: {conversation_filename} "
                    f"({duration:.1f}s)"
                )
        except Exception as e:
            print(
                f"[Worker {worker_id}] Failed to evaluate "
                f"{conversation_filename} with {judge_model}: {e}"
            )

        queue.task_done()

    if verbose_workers:
        print(f"[Worker {worker_id}] Finished. Processed {job_count} jobs.")


async def _run_workers_with_queue(
    jobs: List[
        Tuple[
            ConversationData,
            str,
            int,
            int,
            str,
            RubricConfig,
            Optional[Dict[str, Any]],
        ]
    ],
    max_concurrent: Optional[int],
    per_judge: bool = False,
    verbose_workers: bool = False,
) -> List[Dict[str, Any]]:
    """
    Run evaluation jobs using a worker queue with concurrency control.

    Args:
        jobs: List of job tuples
              (conversation, judge_model, instance, judge_id, output_folder,
              rubric_config, extra_params)
        max_concurrent: Maximum number of concurrent workers (None = unlimited)
        per_judge: If True, max_concurrent applies per judge model;
                  if False, total

    Returns:
        List of evaluation results
    """
    results = []
    total_jobs = len(jobs)

    if per_judge:
        # Group jobs by judge model
        jobs_by_model = {}
        for job in jobs:
            judge_model = job[1]
            if judge_model not in jobs_by_model:
                jobs_by_model[judge_model] = []
            jobs_by_model[judge_model].append(job)

        # Create a queue per judge model
        all_workers = []
        for judge_model, model_jobs in jobs_by_model.items():
            queue = Queue()
            for job in model_jobs:
                await queue.put(job)

            # Determine number of workers for this model
            num_workers = len(model_jobs) if max_concurrent is None else max_concurrent
            print(
                f"Starting {num_workers} workers for {judge_model} "
                f"({len(model_jobs)} jobs total)"
                f"(max concurrent: {max_concurrent})"
            )

            # Create workers for this judge model
            workers = [
                asyncio.create_task(
                    _worker(
                        f"{judge_model[:10]}-{i}",
                        queue,
                        results,
                        total_jobs,
                        verbose_workers,
                    )
                )
                for i in range(num_workers)
            ]
            all_workers.extend(workers)

        # Print worker pool summary
        if verbose_workers:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            print("\n[VERBOSE] Worker pool created:")
            print(f"  - Total workers: {len(all_workers)}")
            print("  - Mode: per-judge concurrency")
            print(f"  - Judge models: {', '.join(jobs_by_model.keys())}")
            max_str = max_concurrent if max_concurrent else "unlimited"
            print(f"  - Max concurrent per judge: {max_str}")
            print(f"  - Started at: {timestamp}\n")

        # Wait for all workers to complete
        await asyncio.gather(*all_workers)

    else:
        # Single queue for all jobs
        queue = Queue()
        for job in jobs:
            await queue.put(job)

        # Determine number of workers
        num_workers = total_jobs if max_concurrent is None else max_concurrent
        print(
            f"Starting {num_workers} workers for all jobs ({total_jobs} total)"
            f"(max concurrent: {max_concurrent})"
        )

        # Create workers
        workers = [
            asyncio.create_task(_worker(i, queue, results, total_jobs, verbose_workers))
            for i in range(num_workers)
        ]

        # Print worker pool summary
        if verbose_workers:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            print("\n[VERBOSE] Worker pool created:")
            print(f"  - Total workers: {num_workers}")
            print("  - Mode: global concurrency")
            max_str = max_concurrent if max_concurrent else "unlimited"
            print(f"  - Max concurrent: {max_str}")
            print(f"  - Started at: {timestamp}\n")

        # Wait for all workers to complete
        await asyncio.gather(*workers)

    return results


async def batch_evaluate_with_individual_judges(
    conversations: List[ConversationData],
    judge_models: Dict[str, int],
    output_folder: str,
    rubric_config: RubricConfig,
    max_concurrent: Optional[int],
    per_judge: bool,
    judge_model_extra_params: Optional[Dict[str, Any]] = None,
    verbose_workers: bool = False,
) -> List[Dict[str, Any]]:
    """
    Evaluate conversations with multiple judge models using queue workers.

    Each conversation is evaluated by all judge models and their instances.
    Workers process jobs from a queue with configurable concurrency.

    Args:
        conversations: List of ConversationData objects
        judge_models: Dict mapping model names to number of instances
                     Example: {"claude-3-7-sonnet": 3, "gpt-4o": 2}
        output_folder: Folder to save evaluation results
        rubric_config: Pre-loaded rubric configuration
        max_concurrent: Maximum number of concurrent workers
        per_judge: If True, max_concurrent applies per judge model; if False, total
        judge_model_extra_params: Extra parameters for the judge model

    Returns:
        Flattened list of evaluation results with one row per
        (conversation, judge_model, judge_instance) tuple
    """
    total_files = len(conversations)
    total_judge_instances = sum(judge_models.values())
    total_evaluations = total_files * total_judge_instances

    print(
        f"Evaluating {total_files} conversations with "
        f"{len(judge_models)} judge models "
        f"({total_judge_instances} total instances)..."
    )
    print(f"Total evaluations to run: {total_evaluations}")
    print(
        f"Concurrency: {max_concurrent} workers "
        f"({'per judge model' if per_judge else 'total'})"
    )

    # Create all evaluation jobs
    jobs = _create_evaluation_jobs(
        conversations,
        judge_models,
        output_folder,
        rubric_config,
        judge_model_extra_params,
    )

    # Run workers with queue
    results = await _run_workers_with_queue(
        jobs, max_concurrent, per_judge, verbose_workers
    )

    print(f"Completed {len(results)}/{total_evaluations} evaluations successfully")
    return results


async def judge_conversations(
    judge_models: Dict[str, int],
    conversations: List[ConversationData],
    rubric_config: RubricConfig,
    output_root: str = "evaluations",
    conversation_folder_name: Optional[str] = None,
    verbose: bool = True,
    output_folder: Optional[str] = None,
    save_aggregated_results: bool = True,
    filename: Optional[str] = "results.csv",
    judge_model_extra_params: Optional[Dict[str, Any]] = None,
    max_concurrent: Optional[int] = None,
    per_judge: bool = False,
    verbose_workers: bool = False,
) -> tuple[List[Dict[str, Any]], str]:
    """
    Judge conversations with multiple judge models.

    Args:
        judge_models: Dict mapping model names to number of instances
                     Example: {"claude-3-7-sonnet": 3, "gpt-4o": 2}
        conversations: List of pre-loaded ConversationData objects
        rubric_config: Pre-loaded rubric configuration
        output_root: Root folder for evaluation outputs
        conversation_folder_name: Optional folder name for output path generation
        verbose: Whether to print status messages
        output_folder: Custom output folder (auto-generated if None)
        save_aggregated_results: Whether to save results to CSV
        filename: Name for aggregated results CSV
        judge_model_extra_params: Extra parameters for the judge model
        max_concurrent: Maximum number of concurrent workers
        per_judge: If True, max_concurrent applies per judge model; if False, total

    Returns:
        Tuple of (results, output_folder) where results is a flattened list of
        evaluation results with one row per (conversation, judge_model, judge_instance)
        tuple, and output_folder is the path where evaluations were saved
    """
    if output_folder is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        judges_str = "_".join(
            f"{model}x{count}" for model, count in judge_models.items()
        )

        # Build judge info string with extra parameters
        judge_info = judges_str
        if judge_model_extra_params:
            # Add temperature if present
            if "temperature" in judge_model_extra_params:
                judge_info += f"_temp{judge_model_extra_params['temperature']}"
            # Add max_tokens if present
            if "max_tokens" in judge_model_extra_params:
                judge_info += f"_maxtok{judge_model_extra_params['max_tokens']}"
            # Add other extra params
            for k, v in judge_model_extra_params.items():
                if k not in ["temperature", "max_tokens"]:
                    judge_info += f"_{k}{v}"

        folder_name = conversation_folder_name or "conversations"
        output_folder = f"{output_root}/j_{judge_info}_{timestamp}__{folder_name}"

    os.makedirs(output_folder, exist_ok=True)

    total_found = len(conversations)
    if verbose:
        print(f"🔍 Judging {total_found} conversations")

    # Run batch evaluation with multiple judges
    results = await batch_evaluate_with_individual_judges(
        conversations,
        judge_models,
        output_folder,
        rubric_config,
        max_concurrent,
        per_judge,
        judge_model_extra_params,
        verbose_workers,
    )

    if save_aggregated_results and results:
        # Column order: filename, run_id, judge_model, judge_instance,
        # judge_id, dimensions
        columns = [
            "filename",
            "run_id",
            "judge_model",
            "judge_instance",
            "judge_id",
        ] + [
            k
            for k in results[0].keys()
            if k
            not in ["filename", "run_id", "judge_model", "judge_instance", "judge_id"]
        ]
        pd.DataFrame(results, columns=columns).to_csv(
            f"{output_folder}/{filename}", index=False
        )
    if verbose:
        print(f"✅ Completed {len(results)} evaluations → {output_folder}/")

    return results, output_folder


async def judge_single_conversation(
    judge: LLMJudge,
    conversation: ConversationData,
    output_folder: str,
) -> Optional[Dict[str, Any]]:
    """
    Judge a single conversation.

    Args:
        judge: LLMJudge instance
        conversation: ConversationData with content and metadata
        output_folder: Output folder for results

    Returns:
        Evaluation results or None if failed
    """
    conversation_filename = conversation.metadata.get("filename", "unknown.txt")
    print(f"📄 Judging: {conversation_filename}")

    result = await judge.evaluate_conversation_question_flow(
        conversation, output_folder=output_folder, auto_save=True
    )

    print(f"🟢 Done: {conversation_filename}")
    return result
