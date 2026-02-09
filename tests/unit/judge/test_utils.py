"""Unit tests for judge utility functions."""

import argparse
from unittest.mock import patch

import pytest

from judge.utils import (
    extract_model_names_from_path,
    extract_persona_name_from_filename,
    load_rubric_structure,
    parse_judge_models,
)


@pytest.mark.unit
class TestLoadRubricStructure:
    """Tests for load_rubric_structure function."""

    def test_load_rubric_structure_basic(self, tmp_path):
        """Test loading a basic rubric with dimensions and options."""
        rubric_content = (
            "Dimension\tBest Practice\tNeutral\tDamaging\n"
            "Empathy\tExample 1\tExample 2\tExample 3\n"
            "Safety\tExample 4\tExample 5\tExample 6\n"
        )
        rubric_file = tmp_path / "test_rubric.tsv"
        rubric_file.write_text(rubric_content)

        dimensions, options = load_rubric_structure(str(rubric_file))

        assert len(dimensions) == 2
        assert "Empathy" in dimensions
        assert "Safety" in dimensions
        assert len(options) == 3
        assert "Best Practice" in options
        assert "Neutral" in options
        assert "Damaging" in options

    def test_load_rubric_structure_question_flow(self, tmp_path):
        """Test loading a question-flow rubric (no option columns)."""
        rubric_content = (
            "Question ID\tDimension\tRisk Type\tQuestion\tExamples\tSeverity\t"
            "Answer\tGOTO\n"
            "1\tDetects Risk\t\tDoes the user show risk?\tExample text\tRed\tYes\t2\n"
            "2\tProbes Risk\t\tDoes chatbot probe?\tExample text\tYellow\tNo\tEND\n"
        )
        rubric_file = tmp_path / "question_flow.tsv"
        rubric_file.write_text(rubric_content)

        dimensions, options = load_rubric_structure(str(rubric_file))

        assert len(dimensions) == 2
        assert "Detects Risk" in dimensions
        assert "Probes Risk" in dimensions
        assert len(options) == 0  # No option columns in question-flow rubric

    def test_load_rubric_structure_with_nan_values(self, tmp_path):
        """Test that NaN values in Dimension column are filtered out."""
        rubric_content = (
            "Dimension\tBest Practice\tNeutral\n"
            "Empathy\tExample 1\tExample 2\n"
            "\tExample 3\tExample 4\n"
            "Safety\tExample 5\tExample 6\n"
        )
        rubric_file = tmp_path / "rubric_with_nans.tsv"
        rubric_file.write_text(rubric_content)

        dimensions, options = load_rubric_structure(str(rubric_file))

        assert len(dimensions) == 2
        assert "Empathy" in dimensions
        assert "Safety" in dimensions
        assert "" not in dimensions

    def test_load_rubric_structure_strips_whitespace(self, tmp_path):
        """Test that whitespace is properly stripped from dimensions and options."""
        rubric_content = (
            "Dimension\t  Best Practice  \t Neutral \n"
            "  Empathy  \tExample 1\tExample 2\n"
            " Safety \tExample 3\tExample 4\n"
        )
        rubric_file = tmp_path / "rubric_whitespace.tsv"
        rubric_file.write_text(rubric_content)

        dimensions, options = load_rubric_structure(str(rubric_file))

        assert "Empathy" in dimensions
        assert "Safety" in dimensions
        assert "Best Practice" in options
        assert "Neutral" in options
        # Verify no whitespace remains
        assert "  Empathy  " not in dimensions
        assert "  Best Practice  " not in options


@pytest.mark.unit
class TestExtractModelNamesFromPath:
    """Tests for extract_model_names_from_path function."""

    def test_extract_from_standard_directory_name(self):
        """Test extracting model names from standard directory format."""
        dir_path = "j_claude_3_opus__p_gpt_4__a_gemini_pro__t10__r5__20231115"
        result = extract_model_names_from_path(dir_path)

        assert result["judge"] == "claude 3 opus"
        assert result["persona"] == "gpt 4"
        assert result["agent"] == "gemini pro"

    def test_extract_from_file_path(self, tmp_path):
        """Test extracting model names when given a file path."""
        dir_name = "j_claude_opus__p_gpt_4__a_gemini__t5__r3__20231115"
        test_dir = tmp_path / dir_name
        test_dir.mkdir()
        results_file = test_dir / "results.csv"
        results_file.write_text("dummy,data\n")

        result = extract_model_names_from_path(str(results_file))

        assert result["judge"] == "claude opus"
        assert result["persona"] == "gpt 4"
        assert result["agent"] == "gemini"

    def test_extract_with_timestamp_in_judge_name(self):
        """Test that timestamps in judge names are properly removed."""
        dir_path = (
            "j_claude_opus_20251112_171754_380__p_gpt_4__a_gemini__t10__r5__20231115"
        )
        result = extract_model_names_from_path(dir_path)

        assert result["judge"] == "claude opus"
        assert result["persona"] == "gpt 4"
        assert result["agent"] == "gemini"

    def test_extract_with_missing_components(self):
        """Test handling of malformed directory names with missing components."""
        dir_path = "j_claude_opus__t10__r5"
        result = extract_model_names_from_path(dir_path)

        # When persona/agent are missing, the judge extraction includes remaining text
        assert "claude opus" in result["judge"]
        assert result["persona"] == "Unknown"
        assert result["agent"] == "Unknown"

    def test_extract_with_no_valid_format(self):
        """Test handling of completely invalid directory names."""
        dir_path = "some_random_directory_name"
        result = extract_model_names_from_path(dir_path)

        assert result["judge"] == "Unknown"
        assert result["persona"] == "Unknown"
        assert result["agent"] == "Unknown"

    def test_extract_with_underscores_in_model_names(self):
        """Test that underscores within model names are converted to spaces."""
        dir_path = (
            "j_claude_3_5_opus__p_gpt_4_turbo__a_gemini_1_5_pro__t10__r5__20231115"
        )
        result = extract_model_names_from_path(dir_path)

        assert result["judge"] == "claude 3 5 opus"
        assert result["persona"] == "gpt 4 turbo"
        assert result["agent"] == "gemini 1 5 pro"


@pytest.mark.unit
class TestExtractPersonaNameFromFilename:
    """Tests for extract_persona_name_from_filename function."""

    def test_extract_persona_basic_format(self):
        """Test extracting persona name from standard filename format."""
        filename = "1a84d1_Brian_g4o_run3_iterative.tsv"
        result = extract_persona_name_from_filename(filename)

        assert result == "Brian"

    def test_extract_persona_without_extension(self):
        """Test extracting persona name from filename without extension."""
        filename = "abc123_Sarah_claude_run1"
        result = extract_persona_name_from_filename(filename)

        assert result == "Sarah"

    def test_extract_persona_with_different_models(self):
        """Test with different model names in filename."""
        filename = "xyz789_Michael_gpt4_run5_iterative.tsv"
        result = extract_persona_name_from_filename(filename)

        assert result == "Michael"

    def test_extract_persona_invalid_format_too_few_parts(self):
        """Test handling of filename with too few underscore-separated parts."""
        filename = "onlyonepart"
        result = extract_persona_name_from_filename(filename)

        assert result is None

    def test_extract_persona_single_underscore(self):
        """Test handling of filename with only one underscore."""
        filename = "hash_name"
        result = extract_persona_name_from_filename(filename)

        assert result == "name"

    def test_extract_persona_empty_string(self):
        """Test handling of empty filename."""
        filename = ""
        result = extract_persona_name_from_filename(filename)

        assert result is None

    def test_extract_persona_with_complex_hash(self):
        """Test with longer hash values."""
        filename = "a1b2c3d4e5_Jessica_openai_run10_iterative.tsv"
        result = extract_persona_name_from_filename(filename)

        assert result == "Jessica"

    def test_extract_persona_preserves_case(self):
        """Test that persona name case is preserved."""
        filename = "123abc_ALLCAPS_model_run1.tsv"
        result = extract_persona_name_from_filename(filename)

        assert result == "ALLCAPS"

    def test_extract_persona_with_numbers_in_name(self):
        """Test persona name containing numbers."""
        filename = "hash_User123_model_run1.tsv"
        result = extract_persona_name_from_filename(filename)

        assert result == "User123"

    def test_extract_persona_with_txt_extension(self):
        """Test that .txt extension works the same as .tsv."""
        filename = "1a84d1_Brian_g4o_run3.txt"
        result = extract_persona_name_from_filename(filename)

        assert result == "Brian"

    def test_extract_persona_handles_exception_gracefully(self):
        """Test that exceptions are handled gracefully with None return."""
        # Pass None to trigger an exception
        with patch("builtins.print"):  # Suppress error message output
            result = extract_persona_name_from_filename(None)

        assert result is None


def _setup_judge_model_arg(argv: list[str]) -> list[str]:
    """Parse argv and return args.judge_model (same type as judge.py CLI)."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--judge-model",
        "-j",
        nargs="+",
        required=True,
        help="Model(s) to use for judging; format 'model' or 'model:count'",
    )
    args = parser.parse_args(argv)
    return args.judge_model


class TestJudgeModelParsing:
    """Test parsing of --judge-model CLI argument (same nargs='+' list as judge.py)."""

    def test_single_model(self):
        """Test parsing a single model without count."""
        judge_model = _setup_judge_model_arg(["-j", "gpt-4o"])
        result = parse_judge_models(judge_model)
        assert result == {"gpt-4o": 1}

    def test_single_model_with_count(self):
        """Test parsing a single model with count."""
        judge_model = _setup_judge_model_arg(["-j", "gpt-4o:3"])
        result = parse_judge_models(judge_model)
        assert result == {"gpt-4o": 3}

    def test_multiple_different_models(self):
        """Test parsing multiple different models."""
        judge_model = _setup_judge_model_arg(
            ["-j", "gpt-4o", "claude-sonnet-4-5-20250929"]
        )
        result = parse_judge_models(judge_model)
        assert result == {"gpt-4o": 1, "claude-sonnet-4-5-20250929": 1}

    def test_multiple_models_with_counts(self):
        """Test parsing multiple models with counts."""
        judge_model = _setup_judge_model_arg(
            ["-j", "gpt-4o:2", "claude-sonnet-4-5-20250929:3"]
        )
        result = parse_judge_models(judge_model)
        assert result == {"gpt-4o": 2, "claude-sonnet-4-5-20250929": 3}

    def test_mixed_models_with_and_without_counts(self):
        """Test parsing mix of models with and without counts."""
        judge_model = _setup_judge_model_arg(
            ["-j", "gpt-4o", "claude-sonnet-4-5-20250929:2"]
        )
        result = parse_judge_models(judge_model)
        assert result == {"gpt-4o": 1, "claude-sonnet-4-5-20250929": 2}

    def test_model_with_multiple_colons(self):
        """Test parsing ollama-style model with colon in name (e.g. llama:7b:3)."""
        judge_model = _setup_judge_model_arg(["-j", "llama:7b:3"])
        result = parse_judge_models(judge_model)
        assert result == {"llama:7b": 3}

    def test_three_models_mixed(self):
        """Test parsing three models with various count specifications."""
        judge_model = _setup_judge_model_arg(
            ["-j", "gpt-4o:2", "claude-sonnet-4-5-20250929", "gpt-3.5-turbo:3"]
        )
        result = parse_judge_models(judge_model)
        assert result == {
            "gpt-4o": 2,
            "claude-sonnet-4-5-20250929": 1,
            "gpt-3.5-turbo": 3,
        }

    def test_large_count(self):
        """Test parsing with large instance count."""
        judge_model = _setup_judge_model_arg(["-j", "gpt-4o:100"])
        result = parse_judge_models(judge_model)
        assert result == {"gpt-4o": 100}

    def test_duplicate_models_last_wins(self):
        """Test that if same model specified twice, last value wins."""
        judge_model = _setup_judge_model_arg(["-j", "gpt-4o:2", "gpt-4o:5"])
        result = parse_judge_models(judge_model)
        assert result == {"gpt-4o": 5}

    def test_count_zero_raises(self):
        """Count after ':' must be positive; 0 should raise ValueError."""
        judge_model = _setup_judge_model_arg(["-j", "gpt-4o:0"])
        with pytest.raises(ValueError, match="must be positive"):
            parse_judge_models(judge_model)

    def test_count_negative_raises(self):
        """Count after ':' must be positive; negative should raise ValueError."""
        judge_model = _setup_judge_model_arg(["-j", "gpt-4o:-1"])
        with pytest.raises(ValueError, match="must be positive"):
            parse_judge_models(judge_model)

    def test_count_float_raises(self):
        """Count after ':' must be an integer; float string should raise ValueError."""
        judge_model = _setup_judge_model_arg(["-j", "gpt-4o:2.5"])
        with pytest.raises(ValueError, match="must be an integer"):
            parse_judge_models(judge_model)

    def test_count_empty_raises(self):
        """Count after ':' cannot be empty; `model:` should raise ValueError."""
        judge_model = _setup_judge_model_arg(["-j", "gpt-4o:"])
        with pytest.raises(ValueError, match="must be an integer"):
            parse_judge_models(judge_model)

    def test_count_non_numeric_raises(self):
        """Count after ':' must be numeric; otherwise should raise ValueError."""
        judge_model = _setup_judge_model_arg(["-j", "gpt-4o:abc"])
        with pytest.raises(ValueError, match="must be an integer"):
            parse_judge_models(judge_model)

    def test_count_alphanumeric_raises(self):
        """Count after ':' must be integer only; otherwise should raise ValueError."""
        judge_model = _setup_judge_model_arg(["-j", "gpt-4o:2x"])
        with pytest.raises(ValueError, match="must be an integer"):
            parse_judge_models(judge_model)
