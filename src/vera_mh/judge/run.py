"""The judging domain's entry point: `run_judging(config)`.

It receives a resolved `JudgingConfig`, loads what it points at, evaluates every
conversation with every judge instance, and writes the run's results. It is the
counterpart of `generate.run_generation` on the generation side, and it is what
`vera judge` and `vera pipeline` call.

It deliberately does none of the CLI's work. It does not parse arguments, read
a target manifest, apply defaults, or choose an output location: the config
already states every one of those. That boundary is what keeps the CLI's
conventions out of the domain.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import pandas as pd

from vera_mh.utils.config_schema import JudgingConfig
from vera_mh.utils.conversation_layout import resolve_conversation_input
from vera_mh.utils.judge_results import build_dataframe_from_tsv_files
from vera_mh.utils.run_record import require_run_record, with_start_time
from vera_mh.utils.run_root import create_run_root, run_started_at

from .answers import conversation_answers_dir, write_answers_tsv
from .rubric_config import RubricConfig, load_conversations
from .runner import (
    _create_evaluation_jobs,
    _list_existing_evaluation_tsv_basenames,
    _run_workers_with_queue,
)
from .utils import add_persona_columns_to_dataframe, build_evaluation_run_folder_path

# The identifying columns that lead each results.csv row.
RESULT_ID_COLUMNS = ["filename", "run_id", "judge_model", "judge_instance", "judge_id"]


async def run_judging(
    config: JudgingConfig,
    *,
    into: Optional[str] = None,
    sample: Optional[int] = None,
    run_record: Optional[Dict[str, Any]] = None,
) -> tuple[List[Dict[str, Any]], str]:
    """Judge the conversations `config` names, and write the run's results.

    Args:
        config: What to judge, with which judge models and rubric, and where
            a new run folder goes (`config.output`)
        into: An existing evaluation run folder to continue (`--into`).
            Evaluations already in it are skipped and new ones land beside
            them. Without it, a new run folder is created under `config.output`.
        sample: Judge only the first N conversations, or None for all
        run_record: Written to a new run folder as `config.json` (see
            `utils/run_record.py`). A continued run keeps the record it
            started with, so this is unused with `into`.

    Returns:
        Tuple of (results, output_folder): one result row per conversation,
        judge model, and instance evaluated in this call, and the folder the
        evaluations were written to.

    Raises:
        FileNotFoundError: If a rubric file or the transcripts directory is
            missing.
    """
    judge_models = {model.name: model.repeats for model in config.models}
    # Every judge model shares one parameter dict (`JudgingSpec.__post_init__`
    # rejects anything else), so the first model's stands for all of them.
    extra_params = dict(config.models[0].extra_params)
    # Exactly one of each, enforced by `JudgingConfig.__post_init__`; the
    # unpacking fails loudly rather than judging only the first if that changes.
    (conversations_folder,) = config.conversations
    (rubric,) = config.rubrics

    models_str = ", ".join(f"{model}x{count}" for model, count in judge_models.items())
    print(f"🎯 LLM Judge | Models: {models_str}")
    print("📚 Loading rubric configuration...")
    rubric_config = await RubricConfig.from_paths(
        rubric_file=rubric.rubric_file,
        rubric_prompt_beginning_file=rubric.rubric_prompt_beginning_file,
        question_prompt_file=rubric.question_prompt_file,
    )

    # The config names the folder the user gave; the transcripts may be in its
    # `conversations/` subfolder. Finding them is idempotent.
    transcripts_dir, _, folder_name = resolve_conversation_input(conversations_folder)
    print(f"📂 Loading conversations from {transcripts_dir}...")
    conversations = await load_conversations(transcripts_dir, limit=sample)
    print(f"✅ Loaded {len(conversations)} conversations")

    if into is None:
        output_folder = _create_run_folder(
            config.output, judge_models, extra_params, folder_name, run_record
        )
        already_judged = None
    else:
        # An existing run being continued, not a new run root.
        output_folder = into
        os.makedirs(output_folder, exist_ok=True)
        already_judged = _list_existing_evaluation_tsv_basenames(output_folder)

    jobs = _create_evaluation_jobs(
        conversations,
        judge_models,
        output_folder,
        rubric_config,
        extra_params,
        already_judged,
    )
    total_instances = sum(judge_models.values())
    total_evaluations = len(conversations) * total_instances
    skipped_existing = total_evaluations - len(jobs)
    _print_plan(
        len(conversations),
        len(judge_models),
        total_instances,
        len(jobs),
        continuing=into is not None,
    )
    print(
        f"Concurrency: {config.max_concurrent} workers "
        f"({'per judge model' if config.per_judge else 'total'})"
    )

    batch_start = datetime.now()
    results = await _run_workers_with_queue(
        jobs, config.max_concurrent, config.per_judge
    )

    _write_results(
        output_folder,
        results,
        rubric_config,
        list(config.personas),
        list(config.persona_annotation_columns),
    )

    elapsed_s = (datetime.now() - batch_start).total_seconds()
    skipped_error = total_evaluations - skipped_existing - len(results)
    print(
        f"\n✅ Completed {len(results)} / {total_evaluations} evaluations "
        f"→ {output_folder}/"
    )
    print(f"  Wall time: {elapsed_s:.2f} seconds")
    if skipped_existing:
        print(f"  ({skipped_existing} skipped: evaluation already exists)")
    if skipped_error:
        print(f"  ({skipped_error} skipped due to errors)")

    return results, output_folder


def _create_run_folder(
    output_root: str,
    judge_models: Dict[str, int],
    extra_params: Dict[str, Any],
    conversation_folder_name: str,
    run_record: Optional[Dict[str, Any]],
) -> str:
    """Create a new evaluation run folder under `output_root`, named after the
    judges, their parameters, the start time, and the conversations judged."""
    started_at = run_started_at()
    timestamp = started_at.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    judge_info = "_".join(f"{model}x{count}" for model, count in judge_models.items())
    if "temperature" in extra_params:
        judge_info += f"_temp{extra_params['temperature']}"
    if "max_tokens" in extra_params:
        judge_info += f"_maxtok{extra_params['max_tokens']}"
    for key, value in extra_params.items():
        if key not in ["temperature", "max_tokens"]:
            judge_info += f"_{key}{value}"

    new_folder = build_evaluation_run_folder_path(
        output_root, judge_info, timestamp, conversation_folder_name or "conversations"
    )
    record = with_start_time(require_run_record(run_record), started_at)
    return str(create_run_root(output_root, Path(new_folder).name, record))


def _print_plan(
    conversations: int,
    judge_models: int,
    total_instances: int,
    pending: int,
    continuing: bool,
) -> None:
    """Say how many evaluations will run, and how many a continued run skips."""
    total = conversations * total_instances
    if not continuing:
        print(f"🔍 Judging {conversations} conversations")
        print(
            f"Evaluating {conversations} conversations with {judge_models} judge "
            f"models ({total_instances} total instances)..."
        )
        print(f"Total evaluations to run: {total}")
        return

    skipped = total - pending
    if total_instances == 1:
        print(
            f"🔍 Judging {pending} conversation(s) "
            f"(skipping {skipped} already judged; "
            f"{conversations} loaded from folder)."
        )
    else:
        print(
            f"🔍 {pending} evaluation job(s) to run "
            f"({skipped} skipped: TSV already on disk; "
            f"{conversations} conversations loaded)."
        )
    print(
        f"Queued {pending} evaluation job(s); "
        f"{skipped} skipped (evaluation TSV already in output folder)."
    )
    print(
        f"Folder scale: {conversations} conversation file(s) × "
        f"{total_instances} judge instance slot(s) = {total} total slot(s)."
    )


def _write_results(
    output_folder: str,
    results: List[Dict[str, Any]],
    rubric_config: RubricConfig,
    personas: List[str],
    persona_annotation_columns: List[str],
) -> None:
    """Write the run's `results.csv` and `answers.tsv`.

    Both are rebuilt from the per-evaluation files in the run folder, not from
    this call's results, so a continued run includes the evaluations it skipped.
    """
    out_csv = os.path.join(output_folder, "results.csv")
    if _list_existing_evaluation_tsv_basenames(output_folder):
        df = build_dataframe_from_tsv_files(Path(output_folder))
        df = _annotate_with_personas(df, personas, persona_annotation_columns)
        df.to_csv(out_csv, index=False)
    elif results:
        # No TSVs yet (e.g. mocked batch in tests).
        columns = RESULT_ID_COLUMNS + [
            key for key in results[0] if key not in RESULT_ID_COLUMNS
        ]
        pd.DataFrame(results, columns=cast(Any, columns)).to_csv(out_csv, index=False)

    if conversation_answers_dir(output_folder).is_dir():
        answers_path = write_answers_tsv(output_folder, rubric_config.question_order)
        print(f"📝 Per-question answers written to: {answers_path}")


def _annotate_with_personas(
    df: "pd.DataFrame",
    personas: Optional[List[str]],
    columns: Optional[List[str]],
) -> "pd.DataFrame":
    """Copy persona annotation columns into the aggregated results frame.

    Annotation is a reporting convenience, not part of the evaluation, so a
    problem here must not lose a judging run that already cost model calls: a
    missing column or unreadable personas file warns and returns the frame
    unannotated rather than raising.

    Only the first personas file is consulted. `judging.personas` is list-shaped
    to mirror the manifest, and every current target names exactly one file; a
    multi-file target would need a documented precedence rule before this reads
    past the first.
    """
    if not columns or not personas:
        return df
    try:
        return add_persona_columns_to_dataframe(df, Path(personas[0]), columns)
    except (KeyError, OSError, ValueError) as error:
        print(f"⚠️  Skipping persona annotation: {error}")
        return df
