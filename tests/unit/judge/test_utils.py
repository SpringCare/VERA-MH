"""Unit tests for judge utility functions."""

from unittest.mock import patch

import pytest

from judge.utils import (
    extract_model_names_from_path,
    extract_persona_name_from_filename,
    load_rubric_structure,
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
            "Question ID\tDimension\tRisk Type\tQuestion\tExamples\tSeverity\tAnswer\tGOTO\n"
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

    def test_extract_persona_handles_exception_gracefully(self):
        """Test that exceptions are handled gracefully with None return."""
        # Pass None to trigger an exception
        with patch("builtins.print"):  # Suppress error message output
            result = extract_persona_name_from_filename(None)

        assert result is None
