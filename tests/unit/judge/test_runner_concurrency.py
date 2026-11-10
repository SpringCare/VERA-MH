"""`max_concurrent=0` means no limit when judging, as it does for generation."""

from pathlib import Path
from unittest.mock import patch

import pytest

from tests.mocks.run_record import minimal_run_record
from vera_mh.judge.llm_judge import LLMJudge
from vera_mh.judge.run import run_judging
from vera_mh.utils.config_schema import JudgingConfig, ModelSpec, RubricFiles


@pytest.mark.unit
class TestMaxConcurrentZero:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("per_judge", [False, True])
    async def test_zero_runs_every_job(
        self, tmp_path: Path, fixtures_dir: Path, per_judge: bool
    ):
        folder = tmp_path / "conversations"
        folder.mkdir()
        for index in range(3):
            (folder / f"conv_{index}.txt").write_text(
                f"user: Hello {index}\nchatbot: Hi {index}!", encoding="utf-8"
            )
        config = JudgingConfig(
            models=[ModelSpec(name="gpt-5.4", repeats=1, extra_params={})],
            rubrics=[
                RubricFiles(
                    rubric_file=str(fixtures_dir / "rubric_simple.tsv"),
                    rubric_prompt_beginning_file=str(
                        fixtures_dir / "rubric_prompt_beginning.txt"
                    ),
                    question_prompt_file=str(fixtures_dir / "question_prompt.txt"),
                )
            ],
            max_concurrent=0,
            per_judge=per_judge,
            conversations=[str(folder)],
            output=str(tmp_path / "output"),
        )

        with patch.object(LLMJudge, "run", autospec=True) as run:
            await run_judging(config, run_record=minimal_run_record())

        assert run.await_count == 3
