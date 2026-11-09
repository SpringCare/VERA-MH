"""Unit tests for how the judge runner sizes its worker pool."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vera_mh.judge.rubric_config import ConversationData
from vera_mh.judge.runner import _create_evaluation_jobs, _run_workers_with_queue

MOCK_EVALUATION_RESULT = {
    "Safety": {
        "score": "Best Practice",
        "reasoning": "Test",
        "yes_question_id": "",
        "yes_reasoning": "",
    }
}


def _conversations(tmp_path: Path, count: int) -> list[ConversationData]:
    return [
        ConversationData(
            content=f"User: Hello {index}\nAssistant: Hi {index}!",
            metadata={
                "filename": f"conv_{index}.txt",
                "run_id": f"test_{index}",
                "source_path": str(tmp_path / f"conv_{index}.txt"),
            },
        )
        for index in range(count)
    ]


@pytest.fixture
def mock_llm_judge_class():
    with patch("vera_mh.judge.runner.LLMJudge") as mock_class:
        mock_judge = MagicMock()
        mock_judge.evaluate_conversation_question_flow = AsyncMock(
            return_value=MOCK_EVALUATION_RESULT
        )
        mock_class.return_value = mock_judge
        yield mock_class


@pytest.mark.unit
class TestMaxConcurrentZero:
    """`max_concurrent=0` means no limit, as it does for generation."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("per_judge", [False, True])
    async def test_zero_runs_every_job(
        self, tmp_path: Path, rubric_config_factory, mock_llm_judge_class, per_judge
    ):
        rubric_config = await rubric_config_factory(rubric_file="rubric_simple.tsv")

        jobs = _create_evaluation_jobs(
            _conversations(tmp_path, 3), {"gpt-5.4": 1}, str(tmp_path), rubric_config
        )
        results = await _run_workers_with_queue(
            jobs, max_concurrent=0, per_judge=per_judge
        )

        assert len(results) == 3
