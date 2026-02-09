"""Unit tests for generate_conversations/utils.py"""

from unittest.mock import patch

import pytest

from generate_conversations.utils import load_prompts_from_csv


@pytest.mark.unit
class TestLoadPromptsFromCsv:
    """Test suite for load_prompts_from_csv function."""

    def test_load_all_personas_from_minimal_fixture(self, fixtures_dir):
        """Test loading all personas from the minimal fixture file."""
        csv_path = fixtures_dir / "personas_minimal.tsv"
        template_path = fixtures_dir / "rubric_prompt_beginning.txt"

        # Create a simple template file for testing
        template_content = (
            "Persona: {persona_id}\nDescription: {persona_desc}\n"
            "Risk: {current_risk_level}"
        )
        template_path.write_text(template_content)

        result = load_prompts_from_csv(
            name_list=None,
            prompt_path=str(csv_path),
            prompt_template_path=str(template_path),
        )

        assert len(result) == 3
        assert result[0]["persona_id"] == "test_persona_1"
        assert result[1]["persona_id"] == "test_persona_2"
        assert result[2]["persona_id"] == "test_persona_3"

    def test_load_with_name_filter(self, tmp_path):
        """Test loading personas with name filtering."""
        csv_path = tmp_path / "personas.tsv"
        csv_path.write_text(
            "Name\tAge\tDescription\nAlice\t30\tAnxious\nBob\t25\tDepressed\nCharlie\t35\tStressed"
        )

        template_path = tmp_path / "template.txt"
        template_content = "Name: {Name}\nDescription: {Description}"
        template_path.write_text(template_content)

        result = load_prompts_from_csv(
            name_list=["Alice", "Charlie"],
            prompt_path=str(csv_path),
            prompt_template_path=str(template_path),
        )

        assert len(result) == 2
        assert result[0]["Name"] == "Alice"
        assert result[1]["Name"] == "Charlie"

    def test_load_with_empty_name_filter_list(self, tmp_path):
        """Test loading with an empty name filter list returns no results."""
        csv_path = tmp_path / "personas.tsv"
        csv_path.write_text("Name\tAge\nAlice\t30\nBob\t25")

        template_path = tmp_path / "template.txt"
        template_content = "Name: {Name}"
        template_path.write_text(template_content)

        result = load_prompts_from_csv(
            name_list=[],
            prompt_path=str(csv_path),
            prompt_template_path=str(template_path),
        )

        assert len(result) == 0

    def test_template_interpolation_with_all_fields(self, tmp_path):
        """Test that template interpolation works correctly with all TSV fields."""
        csv_file = tmp_path / "test_personas.tsv"
        csv_content = (
            "Name\tAge\tDescription\nAlice\t30\tA test person\nBob\t25\tAnother person"
        )
        csv_file.write_text(csv_content)

        template_file = tmp_path / "test_template.txt"
        template_content = "Name: {Name}, Age: {Age}, Description: {Description}"
        template_file.write_text(template_content)

        result = load_prompts_from_csv(
            name_list=None,
            prompt_path=str(csv_file),
            prompt_template_path=str(template_file),
        )

        assert len(result) == 2
        assert "Alice" in result[0]["prompt"]
        assert "30" in result[0]["prompt"]
        assert "A test person" in result[0]["prompt"]
        assert "Bob" in result[1]["prompt"]
        assert "25" in result[1]["prompt"]

    def test_prompt_field_added_to_row(self, tmp_path):
        """Test that a 'prompt' field is added to each row."""
        csv_file = tmp_path / "personas.tsv"
        csv_file.write_text("ID\tName\n1\tAlice\n")

        template_file = tmp_path / "template.txt"
        template_file.write_text("Hello {Name}")

        result = load_prompts_from_csv(
            prompt_path=str(csv_file),
            prompt_template_path=str(template_file),
        )

        assert len(result) == 1
        assert "prompt" in result[0]
        assert result[0]["prompt"] == "Hello Alice"

    def test_csv_not_found_raises_error(self, tmp_path):
        """Test that FileNotFoundError is raised when CSV file doesn't exist."""
        template_file = tmp_path / "template.txt"
        template_file.write_text("Template content")

        with pytest.raises(FileNotFoundError, match="Prompts CSV file not found"):
            load_prompts_from_csv(
                prompt_path="/nonexistent/path/personas.tsv",
                prompt_template_path=str(template_file),
            )

    def test_template_not_found_raises_error(self, tmp_path):
        """Test that FileNotFoundError is raised when template file doesn't exist."""
        csv_file = tmp_path / "personas.tsv"
        csv_file.write_text("Name\tAge\nAlice\t30")

        with pytest.raises(FileNotFoundError, match="Template file not found"):
            load_prompts_from_csv(
                prompt_path=str(csv_file),
                prompt_template_path="/nonexistent/path/template.txt",
            )

    def test_missing_template_key_in_csv_data(self, tmp_path):
        """Test handling of missing template keys in CSV data."""
        csv_file = tmp_path / "personas.tsv"
        csv_file.write_text("Name\tAge\n1\tAlice")

        template_file = tmp_path / "template.txt"
        template_file.write_text("Name: {Name}, Risk: {Risk}")

        with patch("builtins.print") as mock_print:
            result = load_prompts_from_csv(
                prompt_path=str(csv_file),
                prompt_template_path=str(template_file),
            )

        assert len(result) == 0
        mock_print.assert_called()
        args = mock_print.call_args[0][0]
        assert "Warning" in args
        assert "Missing key" in args
        assert "Risk" in args

    def test_multiple_rows_with_missing_required_column(self, tmp_path):
        """Test that rows with missing required columns are skipped."""
        csv_file = tmp_path / "personas.tsv"
        csv_file.write_text("Name\tAge\nAlice\t30\nBob\t25")

        template_file = tmp_path / "template.txt"
        # Template requires a column that doesn't exist in the CSV
        template_file.write_text("Name: {Name}, Risk: {Risk}")

        with patch("builtins.print") as mock_print:
            result = load_prompts_from_csv(
                prompt_path=str(csv_file),
                prompt_template_path=str(template_file),
            )

        # All rows should be skipped because Risk column doesn't exist
        assert len(result) == 0
        mock_print.assert_called()

    def test_empty_csv_file(self, tmp_path):
        """Test loading an empty CSV file (only headers)."""
        csv_file = tmp_path / "empty_personas.tsv"
        csv_file.write_text("Name\tAge\tDescription")

        template_file = tmp_path / "template.txt"
        template_file.write_text("Name: {Name}")

        result = load_prompts_from_csv(
            prompt_path=str(csv_file),
            prompt_template_path=str(template_file),
        )

        assert len(result) == 0

    def test_template_with_multiline_content(self, tmp_path):
        """Test template with multiline content."""
        csv_file = tmp_path / "personas.tsv"
        csv_file.write_text("Name\tContext\nAlice\tDepression")

        template_file = tmp_path / "template.txt"
        template_content = (
            "Persona: {Name}\nContext: {Context}\nInstructions: Be helpful"
        )
        template_file.write_text(template_content)

        result = load_prompts_from_csv(
            prompt_path=str(csv_file),
            prompt_template_path=str(template_file),
        )

        assert len(result) == 1
        assert "Persona: Alice" in result[0]["prompt"]
        assert "Context: Depression" in result[0]["prompt"]
        assert "Instructions: Be helpful" in result[0]["prompt"]

    def test_csv_with_special_characters(self, tmp_path):
        """Test CSV with special characters in fields."""
        csv_file = tmp_path / "personas.tsv"
        csv_file.write_text(
            'Name\tDescription\nAlice\tHas anxiety & depression\nBob\tFeels "stressed"'
        )

        template_file = tmp_path / "template.txt"
        template_file.write_text("{Name}: {Description}")

        result = load_prompts_from_csv(
            prompt_path=str(csv_file),
            prompt_template_path=str(template_file),
        )

        assert len(result) == 2
        assert "anxiety & depression" in result[0]["prompt"]
        assert "stressed" in result[1]["prompt"]

    def test_single_row_csv(self, tmp_path):
        """Test loading CSV with a single row."""
        csv_file = tmp_path / "single.tsv"
        csv_file.write_text("Name\tAge\nAlice\t30")

        template_file = tmp_path / "template.txt"
        template_file.write_text("Name: {Name}, Age: {Age}")

        result = load_prompts_from_csv(
            prompt_path=str(csv_file),
            prompt_template_path=str(template_file),
        )

        assert len(result) == 1
        assert result[0]["Name"] == "Alice"
        assert result[0]["Age"] == "30"

    def test_name_filter_with_non_existent_names(self, tmp_path):
        """Test name filtering when requested names don't exist in CSV."""
        csv_file = tmp_path / "personas.tsv"
        csv_file.write_text("Name\tAge\nAlice\t30\nBob\t25")

        template_file = tmp_path / "template.txt"
        template_file.write_text("{Name}")

        result = load_prompts_from_csv(
            name_list=["Charlie", "Diana"],
            prompt_path=str(csv_file),
            prompt_template_path=str(template_file),
        )

        assert len(result) == 0

    def test_original_csv_fields_preserved_in_output(self, tmp_path):
        """Test that original CSV fields are preserved in output along with prompt."""
        csv_file = tmp_path / "personas.tsv"
        csv_file.write_text("ID\tName\tAge\nid1\tAlice\t30")

        template_file = tmp_path / "template.txt"
        template_file.write_text("Hello {Name}")

        result = load_prompts_from_csv(
            prompt_path=str(csv_file),
            prompt_template_path=str(template_file),
        )

        assert len(result) == 1
        assert result[0]["ID"] == "id1"
        assert result[0]["Name"] == "Alice"
        assert result[0]["Age"] == "30"
        assert result[0]["prompt"] == "Hello Alice"

    def test_whitespace_handling_in_template_keys(self, tmp_path):
        """Test that template keys with values containing whitespace work correctly."""
        csv_file = tmp_path / "personas.tsv"
        csv_file.write_text("Name\tDescription\nAlice\tPerson with anxiety disorder")

        template_file = tmp_path / "template.txt"
        template_file.write_text("Profile: {Name} - {Description}")

        result = load_prompts_from_csv(
            prompt_path=str(csv_file),
            prompt_template_path=str(template_file),
        )

        assert len(result) == 1
        assert "Person with anxiety disorder" in result[0]["prompt"]

    def test_multiple_name_filters_all_present(self, tmp_path):
        """Test filtering with multiple names that all exist."""
        csv_path = tmp_path / "personas.tsv"
        csv_path.write_text("Name\tAge\nAlice\t30\nBob\t25\nCharlie\t35")

        template_path = tmp_path / "template.txt"
        template_content = "{Name}"
        template_path.write_text(template_content)

        result = load_prompts_from_csv(
            name_list=["Alice", "Bob"],
            prompt_path=str(csv_path),
            prompt_template_path=str(template_path),
        )

        assert len(result) == 2

    def test_case_sensitive_name_filtering(self, tmp_path):
        """Test that name filtering is case-sensitive."""
        csv_file = tmp_path / "personas.tsv"
        csv_file.write_text("Name\tAge\nAlice\t30\nalice\t25")

        template_file = tmp_path / "template.txt"
        template_file.write_text("{Name}")

        result = load_prompts_from_csv(
            name_list=["Alice"],
            prompt_path=str(csv_file),
            prompt_template_path=str(template_file),
        )

        assert len(result) == 1
        assert result[0]["Name"] == "Alice"

    def test_template_with_no_placeholders(self, tmp_path):
        """Test template with no placeholders works correctly."""
        csv_file = tmp_path / "personas.tsv"
        csv_file.write_text("Name\tAge\nAlice\t30\nBob\t25")

        template_file = tmp_path / "template.txt"
        template_file.write_text("Static template content")

        result = load_prompts_from_csv(
            prompt_path=str(csv_file),
            prompt_template_path=str(template_file),
        )

        assert len(result) == 2
        assert result[0]["prompt"] == "Static template content"
        assert result[1]["prompt"] == "Static template content"

    def test_unicode_characters_in_data(self, tmp_path):
        """Test handling of unicode characters in CSV and template."""
        csv_file = tmp_path / "personas.tsv"
        csv_file.write_text("Name\tDescription\nAlice\tAnxiety café 🌟")

        template_file = tmp_path / "template.txt"
        template_file.write_text("Name: {Name}, Description: {Description}")

        result = load_prompts_from_csv(
            prompt_path=str(csv_file),
            prompt_template_path=str(template_file),
        )

        assert len(result) == 1
        assert "🌟" in result[0]["prompt"]
        assert "café" in result[0]["prompt"]

    def test_numeric_values_in_csv(self, tmp_path):
        """Test handling numeric values in CSV."""
        csv_file = tmp_path / "personas.tsv"
        csv_file.write_text("Name\tAge\tScore\nAlice\t30\t95.5")

        template_file = tmp_path / "template.txt"
        template_file.write_text("{Name}: {Age} years old, Score: {Score}")

        result = load_prompts_from_csv(
            prompt_path=str(csv_file),
            prompt_template_path=str(template_file),
        )

        assert len(result) == 1
        assert "30" in result[0]["prompt"]
        assert "95.5" in result[0]["prompt"]

    def test_tab_delimited_parsing(self, tmp_path):
        """Test that TSV files are correctly parsed with tab delimiters."""
        csv_file = tmp_path / "personas.tsv"
        csv_file.write_text("ID\tName\tRole\n1\tAlice\tManager\n2\tBob\tDeveloper")

        template_file = tmp_path / "template.txt"
        template_file.write_text("{Name} is a {Role}")

        result = load_prompts_from_csv(
            prompt_path=str(csv_file),
            prompt_template_path=str(template_file),
        )

        assert len(result) == 2
        assert "Alice is a Manager" in result[0]["prompt"]
        assert "Bob is a Developer" in result[1]["prompt"]

    def test_return_type_is_list_of_dicts(self, tmp_path):
        """Test that return type is a list of dictionaries."""
        csv_file = tmp_path / "personas.tsv"
        csv_file.write_text("Name\tAge\nAlice\t30")

        template_file = tmp_path / "template.txt"
        template_file.write_text("Hello")

        result = load_prompts_from_csv(
            prompt_path=str(csv_file),
            prompt_template_path=str(template_file),
        )

        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], dict)

    def test_max_personas_limits_results(self, tmp_path):
        """Test that max_personas caps the number of returned rows."""
        csv_file = tmp_path / "personas.tsv"
        csv_file.write_text("Name\tAge\nAlice\t30\nBob\t25\nCharlie\t35")

        template_file = tmp_path / "template.txt"
        template_file.write_text("{Name}")

        result = load_prompts_from_csv(
            prompt_path=str(csv_file),
            prompt_template_path=str(template_file),
            max_personas=2,
        )

        assert len(result) == 2
        assert result[0]["Name"] == "Alice"
        assert result[1]["Name"] == "Bob"

    def test_max_personas_zero_raises(self, tmp_path):
        """Test that max_personas=0 raises ValueError."""
        csv_file = tmp_path / "personas.tsv"
        csv_file.write_text("Name\tAge\nAlice\t30")

        template_file = tmp_path / "template.txt"
        template_file.write_text("{Name}")

        with pytest.raises(ValueError, match="max_personas must be > 0"):
            load_prompts_from_csv(
                prompt_path=str(csv_file),
                prompt_template_path=str(template_file),
                max_personas=0,
            )

    def test_max_personas_negative_raises(self, tmp_path):
        """Test that max_personas < 0 raises ValueError."""
        csv_file = tmp_path / "personas.tsv"
        csv_file.write_text("Name\tAge\nAlice\t30")

        template_file = tmp_path / "template.txt"
        template_file.write_text("{Name}")

        with pytest.raises(ValueError, match="max_personas must be > 0"):
            load_prompts_from_csv(
                prompt_path=str(csv_file),
                prompt_template_path=str(template_file),
                max_personas=-1,
            )

    def test_max_personas_with_name_filter_applies_after_filter(self, tmp_path):
        """Test that max_personas limits count after name filtering."""
        csv_file = tmp_path / "personas.tsv"
        csv_file.write_text("Name\tAge\nAlice\t30\nBob\t25\nCharlie\t35")

        template_file = tmp_path / "template.txt"
        template_file.write_text("{Name}")

        result = load_prompts_from_csv(
            name_list=["Alice", "Bob", "Charlie"],
            prompt_path=str(csv_file),
            prompt_template_path=str(template_file),
            max_personas=2,
        )

        assert len(result) == 2
        assert result[0]["Name"] == "Alice"
        assert result[1]["Name"] == "Bob"

    def test_name_list_with_csv_missing_name_column_raises(self, tmp_path):
        """Test that name_list with a CSV that has no 'Name' column raises KeyError."""
        csv_file = tmp_path / "personas.tsv"
        csv_file.write_text("ID\tAge\n1\t30\n2\t25")

        template_file = tmp_path / "template.txt"
        template_file.write_text("{ID}")

        with pytest.raises(KeyError, match="Name"):
            load_prompts_from_csv(
                name_list=["1"],
                prompt_path=str(csv_file),
                prompt_template_path=str(template_file),
            )
