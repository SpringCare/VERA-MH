"""Unit tests for skip-already-judged logic in judge runner."""

from pathlib import Path

import pytest

from judge.rubric_config import ConversationData, RubricConfig
from judge.runner import (
    _conversation_already_judged_for_judge_instance,
    _create_evaluation_jobs,
)


@pytest.fixture
def rubric_config_minimal():
    """Minimal RubricConfig for job creation tests."""
    return RubricConfig(
        dimensions=["safety"],
        question_flow_data={"1": {"question": "test"}},
        question_order=["1"],
        rubric_prompt_beginning="test",
        question_prompt_template="test",
    )


@pytest.mark.unit
class TestConversationAlreadyJudgedForJudgeInstance:
    """Tests for _conversation_already_judged_for_judge_instance helper."""

    def test_returns_true_when_matching_tsv_exists(self, tmp_path: Path):
        """Exact conversation + judge model + instance TSV marks that job done."""
        j_foo = tmp_path / "j_foo"
        j_foo.mkdir()
        (j_foo / "conv1_model_x_i1.tsv").write_text("dim\tscore\n")

        conv_judged = ConversationData(
            content="",
            metadata={"filename": "conv1.txt", "run_id": "r1", "source_path": ""},
        )
        conv_not_judged = ConversationData(
            content="",
            metadata={"filename": "conv2.txt", "run_id": "r1", "source_path": ""},
        )

        assert (
            _conversation_already_judged_for_judge_instance(
                conv_judged, tmp_path, "model/x", 1
            )
            is True
        )
        assert (
            _conversation_already_judged_for_judge_instance(
                conv_not_judged, tmp_path, "model/x", 1
            )
            is False
        )

    def test_returns_false_when_only_other_judge_model_tsv_exists(self, tmp_path: Path):
        """Another judge's TSV for the same conversation does not skip this judge."""
        j_foo = tmp_path / "j_foo"
        j_foo.mkdir()
        (j_foo / "conv1_gpt-4o_i1.tsv").write_text("dim\tscore\n")

        conv = ConversationData(
            content="",
            metadata={"filename": "conv1.txt", "run_id": "r1", "source_path": ""},
        )
        assert (
            _conversation_already_judged_for_judge_instance(
                conv, tmp_path, "claude-3-7-sonnet", 1
            )
            is False
        )

    def test_returns_false_when_no_j_dirs(self, tmp_path: Path):
        """When there are no j_* dirs, conversation is not already judged."""
        conv = ConversationData(
            content="",
            metadata={"filename": "conv1.txt", "run_id": "r1", "source_path": ""},
        )
        assert (
            _conversation_already_judged_for_judge_instance(
                conv, tmp_path, "model-a", 1
            )
            is False
        )

    def test_returns_false_when_j_dir_has_no_matching_tsv(self, tmp_path: Path):
        """j_* dir with other .tsv files does not mark this job as done."""
        j_foo = tmp_path / "j_foo"
        j_foo.mkdir()
        (j_foo / "other_conv_model_i1.tsv").write_text("dim\tscore\n")

        conv = ConversationData(
            content="",
            metadata={"filename": "conv1.txt", "run_id": "r1", "source_path": ""},
        )
        assert (
            _conversation_already_judged_for_judge_instance(conv, tmp_path, "model", 1)
            is False
        )


@pytest.mark.unit
class TestCreateEvaluationJobsSkipsJudged:
    """Tests that _create_evaluation_jobs skips already-done evaluation jobs."""

    def test_skips_only_matching_judge_instance_jobs(
        self, tmp_path: Path, rubric_config_minimal: RubricConfig
    ):
        """Only (conversation, judge, instance) with an existing TSV is skipped."""
        set_base = tmp_path / "Set_01"
        set_base.mkdir()
        j_run = set_base / "j_claude-sonnet-4-5x1_20260226_120000__Set_01"
        j_run.mkdir()
        (j_run / "conv1_claude-sonnet-4-5_i1.tsv").write_text("dim\tscore\n")

        output_folder = str(set_base / "j_new_run_20260226_130000__Set_01")

        conversations = [
            ConversationData(
                content="",
                metadata={
                    "filename": "conv1.txt",
                    "run_id": "r1",
                    "source_path": "",
                },
            ),
            ConversationData(
                content="",
                metadata={
                    "filename": "conv2.txt",
                    "run_id": "r1",
                    "source_path": "",
                },
            ),
        ]
        judge_models = {"claude-sonnet-4-5": 2}

        jobs, skipped_count = _create_evaluation_jobs(
            conversations, judge_models, output_folder, rubric_config_minimal
        )

        assert skipped_count == 1
        # conv1: instance 2 only; conv2: both instances → 3 jobs
        assert len(jobs) == 3
        conv1_jobs = [j for j in jobs if j[0].metadata["filename"] == "conv1.txt"]
        assert len(conv1_jobs) == 1
        assert conv1_jobs[0][2] == 2  # instance
