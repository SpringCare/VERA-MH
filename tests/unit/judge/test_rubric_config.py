"""Unit tests for judge rubric configuration."""

import json
from pathlib import Path

import pytest

from judge.rubric_config import RubricConfig


async def load_inline_rubric(tmp_path: Path, rows: str) -> RubricConfig:
    """Load a synthetic rubric through the public file-loading path."""
    rubric_path = tmp_path / "rubric.tsv"
    rubric_path.write_text(
        "Question ID\tDimension\tSeverity\tRisk Type\tQuestion\tExamples\t"
        "Answer\tGOTO\tAuto Answer\n" + rows,
        encoding="utf-8",
    )
    (tmp_path / "rubric_prompt_beginning.txt").write_text(
        "Conversation: {conversation}", encoding="utf-8"
    )
    (tmp_path / "question_prompt.txt").write_text(
        "Question: {question}", encoding="utf-8"
    )
    return await RubricConfig.load(rubric_folder=str(tmp_path))


@pytest.mark.unit
class TestRubricValidation:
    async def test_valid_acyclic_navigation_with_alphanumeric_ids(self, tmp_path):
        config = await load_inline_rubric(
            tmp_path,
            "A1\tFirst\t\t\tStart?\t\tDirect\tC-3\tfalse\n"
            "\t\t\t\t\t\tSequential\t\t\n"
            "B-2\tFirst\tRed\t\tCheck?\t\t\t\t\n"
            "C-3\tSecond\t\t\tFinish?\t\tDone\tEND\t\n",
        )

        assert config.question_order == ["A1", "B-2", "C-3"]

    async def test_missing_goto_target(self, tmp_path):
        with pytest.raises(ValueError, match="targets missing question 'missing'"):
            await load_inline_rubric(
                tmp_path,
                "start\tOnly\t\t\tStart?\t\tContinue\tmissing\t\n",
            )

    async def test_duplicate_question_id(self, tmp_path):
        with pytest.raises(ValueError, match="Duplicate Question ID: 'same'"):
            await load_inline_rubric(
                tmp_path,
                "same\tOne\t\t\tFirst?\t\tDone\tEND\t\n"
                "same\tTwo\t\t\tSecond?\t\tDone\tEND\t\n",
            )

    async def test_self_loop(self, tmp_path):
        with pytest.raises(ValueError, match=r"cycle: self-loop -> self-loop"):
            await load_inline_rubric(
                tmp_path,
                "self-loop\tOnly\t\t\tAgain?\t\tAgain\tself-loop\t\n",
            )

    async def test_multi_question_loop(self, tmp_path):
        with pytest.raises(ValueError, match=r"cycle: A -> B -> C -> A"):
            await load_inline_rubric(
                tmp_path,
                "A\tOne\t\t\tA?\t\tNext\tB\t\n"
                "B\tOne\t\t\tB?\t\tNext\tC\t\n"
                "C\tTwo\t\t\tC?\t\tAgain\tA\t\n",
            )

    async def test_invalid_auto_answer_declaration(self, tmp_path):
        with pytest.raises(ValueError, match="must declare exactly one explicit"):
            await load_inline_rubric(
                tmp_path,
                "auto\tOnly\t\t\tChoose?\t\tFirst\tEND\ttrue\n"
                "\t\t\t\t\t\tSecond\tEND\t\n",
            )


@pytest.mark.unit
class TestLoadBundle:
    """Tests for RubricConfig.load_bundle()."""

    async def test_load_bundle_success(self):
        """Test loading a rubric via a valid bundle manifest."""
        rubric_config = await RubricConfig.load_bundle(
            "tests/fixtures/rubric_manifest_simple.json"
        )
        assert rubric_config.question_flow_data
        assert rubric_config.question_order
        assert rubric_config.rubric_prompt_beginning
        assert rubric_config.question_prompt_template

    async def test_load_bundle_missing_manifest(self):
        """Test that a missing manifest file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            await RubricConfig.load_bundle("tests/fixtures/does_not_exist.json")

    async def test_load_bundle_missing_required_key(self, tmp_path):
        """Test that a manifest missing a required key raises ValueError."""
        manifest_path = tmp_path / "incomplete_manifest.json"
        manifest_path.write_text(
            json.dumps({"rubric_file": "rubric_simple.tsv"}), encoding="utf-8"
        )

        with pytest.raises(ValueError):
            await RubricConfig.load_bundle(str(manifest_path))
