"""
Judge Runner - High-level functions for batch conversation evaluation.
Contains the main logic extracted from main_judge.py to reduce code duplication.
"""

import asyncio
import os
from asyncio import Queue
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import pandas as pd

from .llm_judge import LLMJudge
from .rubric_config import ConversationData, RubricConfig
from .score_utils import build_dataframe_from_tsv_files
from .utils import (
    build_evaluation_run_folder_path,
    build_judge_task_log_path,
    judge_evaluation_tsv_filename,
)

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
    conversation_filename = conversation.metadata.get("filename", "unknown.txt")
    log_file = build_judge_task_log_path(
        conversation_filename,
        judge_model,
        judge_instance,
        output_folder=output_folder,
    )
    judge = LLMJudge(
        judge_model=judge_model,
        rubric_config=rubric_config,
        judge_model_extra_params=judge_model_extra_params,
        log_file=log_file,
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
    existing_tsv_basenames: Optional[set[str]] = None,
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
                # With --resume, each completed job leaves a deterministic .tsv in the
                # output folder; skip those so we only enqueue missing work.
                if existing_tsv_basenames is not None:
                    basename = judge_evaluation_tsv_filename(
                        conversation.metadata.get("filename", "unknown.txt"),
                        judge_model,
                        instance,
                    )
                    if basename in existing_tsv_basenames:
                        continue
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


def _list_existing_evaluation_tsv_basenames(output_folder: str) -> set[str]:
    """Return set of existing TSV basenames in output folder."""
    if not os.path.isdir(output_folder):
        return set()
    return {
        name
        for name in os.listdir(output_folder)
        if name.endswith(".tsv") and os.path.isfile(os.path.join(output_folder, name))
    }


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

    if total_jobs == 0:
        print("No evaluation jobs to run (queue is empty).")
        return results

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

            # Do not spawn more worker tasks than jobs (e.g. resume with few pending).
            model_job_n = len(model_jobs)
            if max_concurrent is None:
                num_workers = model_job_n
            else:
                num_workers = min(max_concurrent, model_job_n)
            print(
                f"Starting {num_workers} workers for {judge_model} "
                f"({model_job_n} jobs total) "
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

        # Do not spawn more worker tasks than jobs (e.g. resume with few pending).
        if max_concurrent is None:
            num_workers = total_jobs
        else:
            num_workers = min(max_concurrent, total_jobs)
        print(
            f"Starting {num_workers} workers for all jobs ({total_jobs} total) "
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
    existing_tsv_basenames: Optional[set[str]] = None,
    jobs: Optional[
        List[
            Tuple[
                ConversationData,
                str,
                int,
                int,
                str,
                RubricConfig,
                Optional[Dict[str, Any]],
            ]
        ]
    ] = None,
) -> List[Dict[str, Any]]:
    """
    Evaluate conversations with multiple judge models using queue workers.

    Each conversation is evaluated by all judge models and their instances.
    Workers process jobs from a queue with configurable concurrency.

    Args:
        conversations: List of ConversationData objects
        judge_models: Dict mapping model names to number of instances
                     Example: {"claude-sonnet-4-5": 3, "gpt-4o": 2}
        output_folder: Folder to save evaluation results
        rubric_config: Pre-loaded rubric configuration
        max_concurrent: Maximum number of concurrent workers
        per_judge: If True, max_concurrent applies per judge model; if False, total
        judge_model_extra_params: Extra parameters for the judge model
        existing_tsv_basenames: If set, skip jobs whose evaluation TSV basename
            is in this set (resume mode).
        jobs: If provided, run exactly these jobs instead of calling
            _create_evaluation_jobs (avoids duplicate work when the caller
            already built the queue).

    Returns:
        Flattened list of evaluation results with one row per
        (conversation, judge_model, judge_instance) tuple
    """
    total_files = len(conversations)
    total_judge_instances = sum(judge_models.values())
    total_evaluations = total_files * total_judge_instances

    if jobs is None:
        jobs = _create_evaluation_jobs(
            conversations,
            judge_models,
            output_folder,
            rubric_config,
            judge_model_extra_params,
            existing_tsv_basenames,
        )

    if existing_tsv_basenames is not None:
        pending_n = len(jobs)
        skipped_n = total_evaluations - pending_n
        print(
            f"Queued {pending_n} evaluation job(s); "
            f"{skipped_n} skipped (evaluation TSV already in output folder)."
        )
        print(
            f"Folder scale: {total_files} conversation file(s) × "
            f"{total_judge_instances} judge instance slot(s) = "
            f"{total_evaluations} total slot(s)."
        )
    else:
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

    # Run workers with queue
    results = await _run_workers_with_queue(
        jobs, max_concurrent, per_judge, verbose_workers
    )

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
    resume: bool = False,
) -> tuple[List[Dict[str, Any]], str]:
    """
    Judge conversations with multiple judge models.

    Args:
        judge_models: Dict mapping model names to number of instances
                     Example: {"claude-sonnet-4-5": 3, "gpt-4o": 2}
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
        output_folder = build_evaluation_run_folder_path(
            output_root, judge_info, timestamp, folder_name
        )

    os.makedirs(output_folder, exist_ok=True)

    total_found = len(conversations)
    total_evaluations = len(conversations) * sum(judge_models.values())
    total_judge_instances = sum(judge_models.values())
    existing_tsv_basenames = (
        _list_existing_evaluation_tsv_basenames(output_folder) if resume else None
    )

    resume_jobs: Optional[
        List[
            Tuple[
                ConversationData,
                str,
                int,
                int,
                str,
                RubricConfig,
                Optional[Dict[str, Any]],
            ]
        ]
    ] = None
    if resume:
        resume_jobs = _create_evaluation_jobs(
            conversations,
            judge_models,
            output_folder,
            rubric_config,
            judge_model_extra_params,
            existing_tsv_basenames,
        )
        if verbose:
            pending_jobs = len(resume_jobs)
            skipped_jobs = total_evaluations - pending_jobs
            if total_judge_instances == 1:
                print(
                    f"🔍 Judging {pending_jobs} conversation(s) "
                    f"(skipping {skipped_jobs} already judged; "
                    f"{total_found} loaded from folder)."
                )
            else:
                print(
                    f"🔍 {pending_jobs} evaluation job(s) to run "
                    f"({skipped_jobs} skipped: TSV already on disk; "
                    f"{total_found} conversations loaded)."
                )
    elif verbose:
        print(f"🔍 Judging {total_found} conversations")

    batch_start = datetime.now()
    results = await batch_evaluate_with_individual_judges(
        conversations,
        judge_models,
        output_folder,
        rubric_config,
        max_concurrent,
        per_judge,
        judge_model_extra_params,
        verbose_workers,
        existing_tsv_basenames,
        jobs=resume_jobs,
    )

    ok_n = len(results)
    if resume:
        assert resume_jobs is not None
        skipped_existing_n = total_evaluations - len(resume_jobs)
    else:
        skipped_existing_n = 0
    skipped_error_n = total_evaluations - skipped_existing_n - ok_n

    if save_aggregated_results:
        csv_name = filename or "results.csv"
        out_csv = os.path.join(output_folder, csv_name)
        # On resume: if the initial scan found any evaluation TSV basenames, at
        # least one matching file still exists after the batch (this path does
        # not delete TSVs). If the scan was empty, this run may have written the
        # first TSVs — listdir again.
        if resume and existing_tsv_basenames:
            has_eval_tsvs = True
        else:
            has_eval_tsvs = bool(_list_existing_evaluation_tsv_basenames(output_folder))
        if has_eval_tsvs:
            # Per-job TSVs are source of truth; includes skipped rows on --resume.
            df = build_dataframe_from_tsv_files(Path(output_folder))
            df.to_csv(out_csv, index=False)
        elif results:
            # No TSVs yet (e.g. mocked batch in tests).
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
                not in [
                    "filename",
                    "run_id",
                    "judge_model",
                    "judge_instance",
                    "judge_id",
                ]
            ]
            pd.DataFrame(results, columns=cast(Any, columns)).to_csv(
                out_csv, index=False
            )
    if verbose:
        elapsed_s = (datetime.now() - batch_start).total_seconds()
        print(
            f"\n✅ Completed {ok_n} / {total_evaluations} evaluations "
            f"→ {output_folder}/"
        )
        print(f"  Wall time: {elapsed_s:.2f} seconds")
        if skipped_existing_n:
            print(f"  ({skipped_existing_n} skipped: evaluation already exists)")
        if skipped_error_n:
            print(f"  ({skipped_error_n} skipped due to errors)")

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
