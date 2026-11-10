"""An `LLMJudge` ready to judge with an already-loaded rubric, for tests."""

from typing import Any, Dict, Optional

from vera_mh.judge.llm_judge import LLMJudge
from vera_mh.judge.rubric_config import RubricConfig


def make_judge(
    rubric_config: RubricConfig,
    *,
    judge_model: str = "mock-llm",
    extra_params: Optional[Dict[str, Any]] = None,
    log_file: Optional[str] = None,
    output_folder: str = "evaluations",
    judge_instance: int = 1,
) -> LLMJudge:
    """An `LLMJudge` job whose rubric is already loaded, as `run()` would load it.

    Its file fields are placeholders: tests call its judging methods directly
    instead of `run()`, which would load the conversation and rubric from them.
    """
    judge = LLMJudge(
        conversation_path="conversation.txt",
        judge_model=judge_model,
        judge_instance=judge_instance,
        output_folder=output_folder,
        rubric_file="rubric.tsv",
        rubric_prompt_beginning_file="rubric_prompt_beginning.txt",
        question_prompt_file="question_prompt.txt",
        extra_params=extra_params or {},
    )
    judge._prepare(rubric_config, log_file=log_file)
    return judge
