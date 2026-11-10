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

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from vera_mh.utils.config_schema import JudgingConfig
from vera_mh.utils.conversation_layout import resolve_conversation_input
from vera_mh.utils.debug import is_debug
from vera_mh.utils.judge_results import build_dataframe_from_tsv_files
from vera_mh.utils.run_record import require_run_record, with_start_time
from vera_mh.utils.run_root import create_run_root, run_started_at
from vera_mh.workers.events import JobKind, Reporter
from vera_mh.workers.job import Outcome, Status
from vera_mh.workers.loop import JobLoop
from vera_mh.workers.reporters import EventsFileReporter, PlainLineReporter

from .answers import conversation_answers_dir, write_answers_tsv
from .evaluation import Evaluation
from .rubric_config import RubricConfig, load_conversations
from .utils import add_persona_columns_to_dataframe, build_evaluation_run_folder_path


async def run_judging(
    config: JudgingConfig,
    *,
    into: Optional[str] = None,
    sample: Optional[int] = None,
    run_record: Optional[Dict[str, Any]] = None,
) -> str:
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
        The folder the evaluations were written to.

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
    else:
        # An existing run being continued, not a new run root. Evaluations
        # already in it are skipped by the loop (`Evaluation.done()`).
        output_folder = into
        os.makedirs(output_folder, exist_ok=True)

    def evaluations(models: Dict[str, int]) -> List[Evaluation]:
        """One per conversation, judge model, and instance, in that order."""
        return [
            Evaluation(
                conversation_path=conversation.metadata["source_path"],
                judge_model=model,
                judge_instance=instance,
                output_folder=output_folder,
                rubric_file=rubric.rubric_file,
                rubric_prompt_beginning_file=rubric.rubric_prompt_beginning_file,
                question_prompt_file=rubric.question_prompt_file,
                extra_params=extra_params,
            )
            for conversation in conversations
            for model, instances in models.items()
            for instance in range(1, instances + 1)
        ]

    reporters: List[Reporter] = [
        PlainLineReporter(verbose=is_debug()),
        EventsFileReporter(Path(output_folder) / "events.jsonl"),
    ]
    cap = {JobKind.EVALUATION: config.max_concurrent}
    if config.per_judge:
        # `max_concurrent` applies to each judge model: one loop per model, side
        # by side, until the loop has a per-model cap (job-loop.md, decision 3).
        loops = []
        for model, instances in judge_models.items():
            loop = JobLoop(cap, reporters=reporters, stage=f"judge {model}")
            for job in evaluations({model: instances}):
                loop.submit(job)
            loops.append(loop)
    else:
        loop = JobLoop(cap, reporters=reporters, stage="judge")
        for job in evaluations(judge_models):
            loop.submit(job)
        loops = [loop]

    async with asyncio.TaskGroup() as tasks:
        runs = [tasks.create_task(loop.run()) for loop in loops]
    outcomes: List[Outcome] = [outcome for run in runs for outcome in run.result()]

    _write_results(
        output_folder,
        rubric_config,
        list(config.personas),
        list(config.persona_annotation_columns),
    )

    print(f"Evaluations in {output_folder}/")
    failed = sum(outcome.status == Status.FAILED for outcome in outcomes)
    if failed:
        print(
            f"{failed} evaluation(s) failed; rerun with `--into {output_folder}` "
            "to judge just those."
        )

    return output_folder


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


def _write_results(
    output_folder: str,
    rubric_config: RubricConfig,
    personas: List[str],
    persona_annotation_columns: List[str],
) -> None:
    """Write the run's `results.csv` and `answers.tsv`.

    Both are rebuilt from the per-evaluation files in the run folder, so a
    continued run includes the evaluations it skipped. With no evaluation in the
    folder (every one failed), neither is written.
    """
    if any(Path(output_folder).glob("*.tsv")):
        df = build_dataframe_from_tsv_files(Path(output_folder))
        df = _annotate_with_personas(df, personas, persona_annotation_columns)
        df.to_csv(os.path.join(output_folder, "results.csv"), index=False)

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
