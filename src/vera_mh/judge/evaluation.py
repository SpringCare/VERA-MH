"""The `Evaluation` job: one conversation judged by one judge model instance.

`judge.run.run_judging` lists one `Evaluation` per conversation, judge model,
and instance, and hands them to the job loop (`workers/loop.py`), which skips
the ones already on disk and runs the rest under the judging cap.

A job is plain data (decision 7 in docs/design/job-loop.md): paths, a model name,
and numbers, so it could be sent to an external queue. It loads what it needs,
the conversation and the rubric, when it runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vera_mh.workers.events import JobKind
from vera_mh.workers.job import Job

from .llm_judge import LLMJudge
from .rubric_config import ConversationData, RubricConfig
from .utils import build_judge_task_log_path, judge_evaluation_tsv_filename


@dataclass
class Evaluation:
    """Judge one conversation with one instance of one judge model."""

    conversation_path: str
    judge_model: str
    judge_instance: int  # 1, 2, ... for each judge model
    output_folder: str
    rubric_file: str
    rubric_prompt_beginning_file: str
    question_prompt_file: str
    extra_params: dict[str, Any] = field(default_factory=dict)
    kind: JobKind = JobKind.EVALUATION  # picks the judging cap

    @property
    def label(self) -> str:
        """How events name it, such as "gpt-5.4 Abigail_run1" (with " i2" for a
        second instance of the same judge)."""
        label = f"{self.judge_model} {Path(self.conversation_path).stem}"
        return label if self.judge_instance == 1 else f"{label} i{self.judge_instance}"

    def _tsv_path(self) -> Path:
        name = judge_evaluation_tsv_filename(
            Path(self.conversation_path).name, self.judge_model, self.judge_instance
        )
        return Path(self.output_folder) / name

    def done(self) -> bool:
        """Is the evaluation's TSV on disk? `LLMJudge` writes it last, once every
        question it needed has an answer, so a judging stopped partway leaves
        none and is redone."""
        return self._tsv_path().is_file()

    async def run(self) -> None:
        """Judge the conversation and write its TSV and per-question answers."""
        conversation = await ConversationData.load(self.conversation_path)
        rubric_config = await RubricConfig.from_paths(
            rubric_file=self.rubric_file,
            rubric_prompt_beginning_file=self.rubric_prompt_beginning_file,
            question_prompt_file=self.question_prompt_file,
        )
        judge = LLMJudge(
            judge_model=self.judge_model,
            rubric_config=rubric_config,
            judge_model_extra_params=self.extra_params,
            log_file=build_judge_task_log_path(
                conversation.metadata["filename"],
                self.judge_model,
                self.judge_instance,
                output_folder=self.output_folder,
            ),
        )
        await judge.evaluate_conversation_question_flow(
            conversation,
            output_folder=self.output_folder,
            auto_save=True,
            judge_instance=self.judge_instance,
        )

    def follow_ups(self) -> list[Job]:
        """Nothing waits on an evaluation: `results.csv` is written once the
        whole loop is done."""
        return []
