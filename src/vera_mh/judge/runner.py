"""The evaluation jobs of a judging run, and the worker pool that runs them.

`judge.run.run_judging` lists the jobs (one per conversation, judge model, and
instance) and hands them to the pool here. The job loop (docs/design/job-loop.md)
replaces this module with one `Evaluation` job.
"""

import asyncio
import os
from asyncio import Queue
from typing import Any, Dict, List, Optional, Tuple, Union

from .llm_judge import LLMJudge
from .rubric_config import ConversationData, RubricConfig
from .utils import build_judge_task_log_path, judge_evaluation_tsv_filename

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
):
    """
    Worker that processes evaluation jobs from the queue.

    Args:
        worker_id: Unique identifier for this worker (int or str)
        queue: Queue containing job tuples
        results: Shared list to append results to
        total_jobs: Total number of jobs for progress tracking
    """
    while True:
        try:
            job = queue.get_nowait()
        except asyncio.QueueEmpty:
            break

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
        except Exception as e:
            print(
                f"[Worker {worker_id}] Failed to evaluate "
                f"{conversation_filename} with {judge_model}: {e}"
            )

        queue.task_done()


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
) -> List[Dict[str, Any]]:
    """
    Run evaluation jobs using a worker queue with concurrency control.

    Args:
        jobs: List of job tuples
              (conversation, judge_model, instance, judge_id, output_folder,
              rubric_config, extra_params)
        max_concurrent: Maximum number of concurrent workers (None or 0 = unlimited)
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
            # None and 0 both mean no limit, matching generation.
            model_job_n = len(model_jobs)
            if not max_concurrent:
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
                    _worker(f"{judge_model[:10]}-{i}", queue, results, total_jobs)
                )
                for i in range(num_workers)
            ]
            all_workers.extend(workers)

        # Wait for all workers to complete
        await asyncio.gather(*all_workers)

    else:
        # Single queue for all jobs
        queue = Queue()
        for job in jobs:
            await queue.put(job)

        # Do not spawn more worker tasks than jobs (e.g. resume with few pending).
        # None and 0 both mean no limit, matching generation.
        if not max_concurrent:
            num_workers = total_jobs
        else:
            num_workers = min(max_concurrent, total_jobs)
        print(
            f"Starting {num_workers} workers for all jobs ({total_jobs} total) "
            f"(max concurrent: {max_concurrent})"
        )

        # Create workers
        workers = [
            asyncio.create_task(_worker(i, queue, results, total_jobs))
            for i in range(num_workers)
        ]

        # Wait for all workers to complete
        await asyncio.gather(*workers)

    return results
