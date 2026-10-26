"""Unit tests for utils/judge_results.py: TSV parsing and results.csv rebuild."""

from unittest.mock import patch

import pandas as pd
import pytest

from vera_mh.judge.rubric_config import (
    CONFIRMS_RISK,
    DETECTS_POTENTIAL_RISK,
    FOLLOWS_AI_BOUNDARIES,
    GUIDES_TO_HUMAN_CARE,
    SUPPORTIVE_CONVERSATION,
)
from vera_mh.utils.judge_results import (
    build_dataframe_from_tsv_files,
    build_results_csv_from_tsv_files,
    extract_conversation_filename_from_tsv,
    extract_persona_name_from_filename,
    load_rubric_structure,
    parse_evaluation_filename,
    parse_judge_metadata_from_evaluation_tsv_filename,
)
from vera_mh.utils.ratings import BEST_PRACTICE, DAMAGING, NEUTRAL

# ============================================================================
# Filename Parsing Tests
# ============================================================================


@pytest.mark.unit
def test_parse_evaluation_filename_valid():
    """Test parsing a valid evaluation filename."""
    filename = "000682_Alix_gemini-3-pro-preview_run22_gpt-4o_i1.tsv"
    result = parse_evaluation_filename(filename)
    assert result["id"] == "000682"
    assert result["persona"] == "Alix"
    assert result["user_model"] == "gemini-3-pro-preview"
    assert result["run"] == 22
    assert result["judge_model"] == "gpt-4o"
    assert result["judge_iteration"] == 1


@pytest.mark.unit
def test_parse_evaluation_filename_invalid():
    """Test parsing an invalid filename returns empty dict."""
    result = parse_evaluation_filename("invalid_filename.txt")
    assert result == {}


@pytest.mark.unit
def test_parse_evaluation_filename_no_extension():
    """Test parsing filename without extension."""
    filename = "000682_Alix_gemini-3-pro-preview_run22_gpt-4o_i1"
    result = parse_evaluation_filename(filename)
    assert result["id"] == "000682"
    assert result["persona"] == "Alix"


@pytest.mark.unit
def test_parse_judge_metadata_from_evaluation_tsv_filename_structured():
    """Structured basename matches parse_evaluation_filename."""
    filename = "000682_Alix_gemini-3-pro-preview_run22_gpt-4o_i1.tsv"
    judge_model, judge_instance = parse_judge_metadata_from_evaluation_tsv_filename(
        filename
    )
    assert judge_model == "gpt-4o"
    assert judge_instance == 1


@pytest.mark.unit
def test_parse_judge_metadata_from_evaluation_tsv_filename_batch_fallback():
    """Batch basename from judge_evaluation_tsv_filename (single judge token, no _)."""
    filename = "conv1_mock-judge_i1.tsv"
    judge_model, judge_instance = parse_judge_metadata_from_evaluation_tsv_filename(
        filename
    )
    assert judge_model == "mock-judge"
    assert judge_instance == 1


@pytest.mark.unit
def test_parse_judge_metadata_from_evaluation_tsv_filename_unknown():
    """Unparseable basenames yield empty judge and instance 0."""
    assert parse_judge_metadata_from_evaluation_tsv_filename("test.tsv") == ("", 0)
    assert parse_judge_metadata_from_evaluation_tsv_filename("nope.txt") == ("", 0)


@pytest.mark.unit
def test_build_results_csv_from_tsv_files_batch_judge_metadata(tmp_path):
    """build_results_csv_from_tsv_files sets judge columns for batch-style TSV names."""
    eval_dir = tmp_path / "evaluations"
    eval_dir.mkdir()

    tsv_content = (
        "Dimension\tScore\tJustification\n"
        f"{DETECTS_POTENTIAL_RISK}\tBest Practice\tGood\n"
    )
    (eval_dir / "conv1_mock-judge_i1.tsv").write_text(tsv_content)

    df = build_results_csv_from_tsv_files(eval_dir)

    assert len(df) == 1
    assert df.iloc[0]["judge_model"] == "mock-judge"
    assert df.iloc[0]["judge_instance"] == 1
    assert df.iloc[0]["judge_id"] == 0


# ============================================================================
# Conversation Filename Extraction Tests
# ============================================================================
# Tests for extract_conversation_filename_from_tsv function covering:
# - Primary path: structured format parsing (parse_evaluation_filename)
# - First fallback: regex pattern r"(.+?)_[^_]+_i\d+$" (line 204)
# - Alternative fallback: regex r"(.+)_i\d+$" + validation (lines 211-218)
# - Final fallback: simple extension replacement (line 221)
# - Edge cases: malformed filenames, complex model names, boundary conditions


@pytest.mark.unit
def test_extract_conversation_filename_structured_format():
    """Test extraction using structured format parsing (primary path)."""
    # Standard structured format that parse_evaluation_filename can handle
    filename = "000682_Alix_gemini-3-pro-preview_run22_gpt-4o_i1.tsv"
    result = extract_conversation_filename_from_tsv(filename)
    expected = "000682_Alix_gemini-3-pro-preview_run22.txt"
    assert result == expected


@pytest.mark.unit
def test_extract_conversation_filename_structured_format_various_models():
    """Test structured format with different user models."""
    test_cases = [
        (
            "abc123_John_claude-3-5-sonnet_run1_gpt-4o_i1.tsv",
            "abc123_John_claude-3-5-sonnet_run1.txt",
        ),
        (
            "def456_Sarah_gpt-4_run10_claude-sonnet_i2.tsv",
            "def456_Sarah_gpt-4_run10.txt",
        ),
        (
            "789xyz_Mike_llama-2-70b-chat_run5_claude-3-haiku_i3.tsv",
            "789xyz_Mike_llama-2-70b-chat_run5.txt",
        ),
    ]

    for input_filename, expected in test_cases:
        result = extract_conversation_filename_from_tsv(input_filename)
        assert result == expected, (
            f"Failed for {input_filename}: got {result}, expected {expected}"
        )


@pytest.mark.unit
def test_extract_conversation_filename_fallback_regex_pattern():
    """Test fallback regex pattern _{judge_model}_i{instance} (line 204)."""
    # Cases where structured parsing fails, but fallback regex works
    test_cases = [
        ("3ea338_Lena_g5_run4_uuid_gpt-4o_i1.tsv", "3ea338_Lena_g5_run4_uuid.txt"),
        ("conversation_name_claude-3-sonnet_i5.tsv", "conversation_name.txt"),
        (
            "complex_name_with_underscores_gpt-4_i2.tsv",
            "complex_name_with_underscores.txt",
        ),
        ("short_name_mixtral-8x7b_i10.tsv", "short_name.txt"),
    ]

    for input_filename, expected in test_cases:
        result = extract_conversation_filename_from_tsv(input_filename)
        assert result == expected, (
            f"Failed for {input_filename}: got {result}, expected {expected}"
        )


@pytest.mark.unit
def test_extract_conversation_filename_alternative_regex_pattern():
    """Test alternative regex for judge models without underscores.

    Covers score_utils lines 211-218.
    """
    # Cases where judge model doesn't have underscores in name
    test_cases = [
        ("conversation_name_gpt4o_i1.tsv", "conversation_name.txt"),
        ("test_file_claude3_i2.tsv", "test_file.txt"),
        ("another_test_gemini_i5.tsv", "another_test.txt"),
    ]

    for input_filename, expected in test_cases:
        result = extract_conversation_filename_from_tsv(input_filename)
        assert result == expected, (
            f"Failed for {input_filename}: got {result}, expected {expected}"
        )


@pytest.mark.unit
def test_extract_conversation_filename_edge_cases():
    """Test edge cases and malformed filenames."""
    # Edge cases that should fall through to the final fallback (line 221)
    test_cases = [
        # No _i pattern at all
        ("simple_filename.tsv", "simple_filename.txt"),
        # _i pattern but no preceding underscore context
        ("nounderscore_i1.tsv", "nounderscore_i1.txt"),
        # Multiple _i patterns - the regex will match the appropriate pattern
        # The first regex r"(.+?)_[^_]+_i\d+$" matches "file_i2_more_content_i3"
        # It captures "file_i2_more" (everything before "_content_i3")
        ("file_i2_more_content_i3.tsv", "file_i2_more.txt"),
        # Empty filename
        (".tsv", ".txt"),
        # Just extension
        ("tsv", "tsv.txt"),
    ]

    for input_filename, expected in test_cases:
        result = extract_conversation_filename_from_tsv(input_filename)
        assert result == expected, (
            f"Failed for {input_filename}: got {result}, expected {expected}"
        )


@pytest.mark.unit
def test_extract_conversation_filename_no_tsv_extension():
    """Test behavior with files that don't have .tsv extension."""
    test_cases = [
        # Should still work, just converts the extension
        (
            "000682_Alix_gemini-3-pro-preview_run22_gpt-4o_i1",
            "000682_Alix_gemini-3-pro-preview_run22.txt",
        ),
        ("conversation_name_claude-3-sonnet_i5", "conversation_name.txt"),
        ("simple_filename", "simple_filename.txt"),
    ]

    for input_filename, expected in test_cases:
        result = extract_conversation_filename_from_tsv(input_filename)
        assert result == expected, (
            f"Failed for {input_filename}: got {result}, expected {expected}"
        )


@pytest.mark.unit
def test_extract_conversation_filename_complex_model_names():
    """Test with complex model names containing hyphens and numbers."""
    test_cases = [
        # Complex judge model names
        ("conv_name_claude-3-5-sonnet-20241022_i1.tsv", "conv_name.txt"),
        ("test_gpt-4o-2024-05-13_i2.tsv", "test.txt"),
        ("example_llama-2-70b-chat-hf_i3.tsv", "example.txt"),
        # Complex conversation names with model-like patterns
        (
            "gpt-4_test_conversation_claude-3-sonnet_i1.tsv",
            "gpt-4_test_conversation.txt",
        ),
        ("claude-test_run_gpt-4o_i2.tsv", "claude-test_run.txt"),
    ]

    for input_filename, expected in test_cases:
        result = extract_conversation_filename_from_tsv(input_filename)
        assert result == expected, (
            f"Failed for {input_filename}: got {result}, expected {expected}"
        )


@pytest.mark.unit
def test_extract_conversation_filename_regex_boundaries():
    """Test regex boundary conditions and potential false matches."""
    test_cases = [
        # Instance numbers with multiple digits
        ("conversation_name_gpt-4o_i123.tsv", "conversation_name.txt"),
        ("test_claude-3-sonnet_i9999.tsv", "test.txt"),
        # Edge case: _i followed by non-digits (should fall through to final fallback)
        ("conversation_name_gpt-4o_iabc.tsv", "conversation_name_gpt-4o_iabc.txt"),
        ("test_file_judge_inotdigit.tsv", "test_file_judge_inotdigit.txt"),
        # Mixed cases with valid _i patterns - the regex will match appropriately
        # The pattern r"(.+?)_[^_]+_i\d+$" will match the last valid _model_i5 pattern
        ("valid_pattern_invalid_itext_valid_i5.tsv", "valid_pattern_invalid_itext.txt"),
    ]

    for input_filename, expected in test_cases:
        result = extract_conversation_filename_from_tsv(input_filename)
        assert result == expected, (
            f"Failed for {input_filename}: got {result}, expected {expected}"
        )


# ============================================================================
# TSV File Operations Tests
# ============================================================================


@pytest.mark.unit
def test_load_evaluation_tsv(tmp_path):
    """Test loading a single evaluation TSV file."""
    tsv_path = tmp_path / "test.tsv"
    tsv_content = (
        "Dimension\tScore\tJustification\nDetects Potential Risk\tBest Practice\tGood\n"
    )
    tsv_path.write_text(tsv_content)

    df = pd.read_csv(tsv_path, sep="\t")

    assert len(df) == 1
    assert "Dimension" in df.columns
    assert "Score" in df.columns


@pytest.mark.unit
def test_build_results_csv_from_tsv_files(tmp_path):
    """Test building results CSV from TSV files."""
    eval_dir = tmp_path / "evaluations"
    eval_dir.mkdir()

    tsv_content = (
        "Dimension\tScore\tJustification\n"
        f"{DETECTS_POTENTIAL_RISK}\tBest Practice\tGood\n"
        f"{CONFIRMS_RISK}\tSuboptimal but Low Potential for Harm\tSome probing\n"
    )
    (eval_dir / "test.tsv").write_text(tsv_content)

    df = build_results_csv_from_tsv_files(eval_dir)

    assert len(df) == 1
    assert "filename" in df.columns
    assert "run_id" in df.columns
    assert DETECTS_POTENTIAL_RISK in df.columns
    assert df.iloc[0][DETECTS_POTENTIAL_RISK] == BEST_PRACTICE


@pytest.mark.unit
def test_build_results_csv_from_tsv_files_no_files(tmp_path):
    """Test error when no TSV files found."""
    eval_dir = tmp_path / "empty"
    eval_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="No TSV files found"):
        build_results_csv_from_tsv_files(eval_dir)


# ============================================================================
# build_dataframe_from_tsv_files() Tests
# ============================================================================


@pytest.mark.unit
def test_build_dataframe_from_tsv_files_basic(tmp_path):
    """Test building dataframe from TSV files with basic valid data."""
    eval_dir = tmp_path / "evaluations"
    eval_dir.mkdir()

    # Create sample TSV file
    tsv_content = (
        "Dimension\tScore\tJustification\n"
        f"{DETECTS_POTENTIAL_RISK}\tBest Practice\tGood detection\n"
        f"{CONFIRMS_RISK}\tSuboptimal but Low Potential for Harm\tSome probing\n"
        f"{GUIDES_TO_HUMAN_CARE}\tBest Practice\t"
        "Good resources\n"
        f"{SUPPORTIVE_CONVERSATION}\tNot Relevant\tN/A\n"
        f"{FOLLOWS_AI_BOUNDARIES}\tBest Practice\tGood boundaries\n"
    )
    (eval_dir / "abc123_test.tsv").write_text(tsv_content)

    df = build_dataframe_from_tsv_files(eval_dir)

    assert len(df) == 1
    assert "filename" in df.columns
    assert "run_id" in df.columns
    assert DETECTS_POTENTIAL_RISK in df.columns
    assert df.iloc[0][DETECTS_POTENTIAL_RISK] == BEST_PRACTICE
    assert df.iloc[0][CONFIRMS_RISK] == NEUTRAL


@pytest.mark.unit
def test_build_dataframe_from_tsv_files_multiple_files(tmp_path):
    """Test building dataframe from multiple TSV files."""
    eval_dir = tmp_path / "evaluations"
    eval_dir.mkdir()

    # Create two TSV files
    for i, filename in enumerate(["file1.tsv", "file2.tsv"]):
        tsv_content = (
            "Dimension\tScore\tJustification\n"
            f"{DETECTS_POTENTIAL_RISK}\tBest Practice\tFile {i + 1}\n"
            f"{CONFIRMS_RISK}\tHigh Potential for Harm\tFile {i + 1}\n"
            f"{GUIDES_TO_HUMAN_CARE}\t"
            f"Best Practice\tFile {i + 1}\n"
            f"{SUPPORTIVE_CONVERSATION}\tNot Relevant\t"
            f"File {i + 1}\n"
            f"{FOLLOWS_AI_BOUNDARIES}\tBest Practice\tFile {i + 1}\n"
        )
        (eval_dir / filename).write_text(tsv_content)

    df = build_dataframe_from_tsv_files(eval_dir)

    assert len(df) == 2
    assert all(df[DETECTS_POTENTIAL_RISK] == BEST_PRACTICE)
    assert all(df[CONFIRMS_RISK] == DAMAGING)


@pytest.mark.unit
def test_build_dataframe_from_tsv_files_extracts_run_id(tmp_path):
    """Test extracting run_id from directory name."""
    eval_dir = tmp_path / "j_model1__p_model2__a_model3__run_12345"
    eval_dir.mkdir()

    tsv_content = (
        "Dimension\tScore\tJustification\n"
        f"{DETECTS_POTENTIAL_RISK}\tBest Practice\tTest\n"
        f"{CONFIRMS_RISK}\tBest Practice\tTest\n"
        f"{GUIDES_TO_HUMAN_CARE}\tBest Practice\t"
        "Test\n"
        f"{SUPPORTIVE_CONVERSATION}\tBest Practice\tTest\n"
        f"{FOLLOWS_AI_BOUNDARIES}\tBest Practice\tTest\n"
    )
    (eval_dir / "test.tsv").write_text(tsv_content)

    df = build_dataframe_from_tsv_files(eval_dir)

    assert df.iloc[0]["run_id"] == "run_12345"


@pytest.mark.unit
def test_build_dataframe_from_tsv_files_missing_dimensions(tmp_path):
    """Test handling of TSV files with missing dimensions."""
    eval_dir = tmp_path / "evaluations"
    eval_dir.mkdir()

    # Create TSV with only some dimensions
    tsv_content = (
        "Dimension\tScore\tJustification\n"
        f"{DETECTS_POTENTIAL_RISK}\tBest Practice\tOnly one dimension\n"
    )
    (eval_dir / "partial.tsv").write_text(tsv_content)

    df = build_dataframe_from_tsv_files(eval_dir)

    assert len(df) == 1
    assert df.iloc[0][DETECTS_POTENTIAL_RISK] == BEST_PRACTICE
    # Missing dimensions should be filled with empty strings
    assert df.iloc[0][CONFIRMS_RISK] == ""


@pytest.mark.unit
def test_build_dataframe_from_tsv_files_no_tsv_files_raises_error(tmp_path):
    """Test error when no TSV files found in directory."""
    eval_dir = tmp_path / "empty_evaluations"
    eval_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="No TSV files found"):
        build_dataframe_from_tsv_files(eval_dir)


@pytest.mark.unit
def test_build_dataframe_from_tsv_files_malformed_tsv(tmp_path):
    """Test handling of malformed TSV files - missing dimensions filled."""
    eval_dir = tmp_path / "evaluations"
    eval_dir.mkdir()

    # Create a TSV with missing Dimension column (truly malformed)
    (eval_dir / "bad.tsv").write_text("InvalidColumn\tValue\n")

    # Create a good TSV file so we don't hit FileNotFoundError
    tsv_content = (
        "Dimension\tScore\tJustification\n"
        f"{DETECTS_POTENTIAL_RISK}\tBest Practice\tGood\n"
        f"{CONFIRMS_RISK}\tBest Practice\tGood\n"
        f"{GUIDES_TO_HUMAN_CARE}\tBest Practice\t"
        "Good\n"
        f"{SUPPORTIVE_CONVERSATION}\tBest Practice\tGood\n"
        f"{FOLLOWS_AI_BOUNDARIES}\tBest Practice\tGood\n"
    )
    (eval_dir / "good.tsv").write_text(tsv_content)

    df = build_dataframe_from_tsv_files(eval_dir)

    # Malformed file creates row with empty dimension values
    assert len(df) == 2
    # Good file should have all dimensions populated
    good_row = df[df["filename"] == "good.txt"].iloc[0]
    assert good_row[DETECTS_POTENTIAL_RISK] == BEST_PRACTICE


# ============================================================================
# build_results_csv_from_tsv_files() - yes_question_id and yes_reasoning Tests
# ============================================================================


@pytest.mark.unit
def test_build_results_csv_with_reasoning_parsing_single_question(tmp_path):
    """Test Reasoning parsing with a single Q{id}: pattern."""
    eval_dir = tmp_path / "evaluations"
    eval_dir.mkdir()

    tsv_content = (
        "Dimension\tScore\tReasoning\n"
        "Detects Potential Risk\tBest Practice\t"
        "Q1: The agent correctly identifies potential suicide risk "
        "based on the user's statements about feeling hopeless.\n"
    )
    (eval_dir / "test.tsv").write_text(tsv_content)

    df = build_results_csv_from_tsv_files(eval_dir)

    assert len(df) == 1
    assert "Detects Potential Risk_yes_question_id" in df.columns
    assert "Detects Potential Risk_yes_reasoning" in df.columns
    assert df.iloc[0]["Detects Potential Risk_yes_question_id"] == "1"
    assert df.iloc[0]["Detects Potential Risk_yes_reasoning"] == (
        "The agent correctly identifies potential suicide risk "
        "based on the user's statements about feeling hopeless."
    )


@pytest.mark.unit
def test_build_results_csv_with_reasoning_parsing_multiple_questions(tmp_path):
    """Test Reasoning with multiple Q patterns; first match is used."""
    eval_dir = tmp_path / "evaluations"
    eval_dir.mkdir()

    tsv_content = (
        "Dimension\tScore\tReasoning\n"
        "Detects Potential Risk\tBest Practice\t"
        "Q3: Agent identifies risk factors well; "
        "Q7: Follow-up questions show good probing; "
        "Q12: Safety planning initiated appropriately.\n"
    )
    (eval_dir / "test.tsv").write_text(tsv_content)

    df = build_results_csv_from_tsv_files(eval_dir)

    assert len(df) == 1
    assert df.iloc[0]["Detects Potential Risk_yes_question_id"] == "3"
    assert (
        df.iloc[0]["Detects Potential Risk_yes_reasoning"]
        == "Agent identifies risk factors well"
    )


@pytest.mark.unit
def test_build_results_csv_with_alphanumeric_question_ids(tmp_path):
    """Question IDs are opaque strings rather than numeric values."""
    eval_dir = tmp_path / "evaluations"
    eval_dir.mkdir()

    tsv_content = (
        "Dimension\tScore\tReasoning\n"
        "Detects Potential Risk\tHigh Potential for Harm\t"
        "Q1a: Agent missed the initial risk signal; "
        "Q2-beta: Agent also missed the follow-up signal.\n"
    )
    (eval_dir / "test.tsv").write_text(tsv_content)

    df = build_results_csv_from_tsv_files(eval_dir)

    assert df.iloc[0]["Detects Potential Risk_yes_question_id"] == "1a"
    assert df.iloc[0]["Detects Potential Risk_yes_reasoning"] == (
        "Agent missed the initial risk signal"
    )


@pytest.mark.unit
def test_build_results_csv_with_reasoning_parsing_no_q_pattern(tmp_path):
    """Test parsing when Reasoning column has no Q{id}: pattern."""
    eval_dir = tmp_path / "evaluations"
    eval_dir.mkdir()

    tsv_content = (
        "Dimension\tScore\tReasoning\n"
        "Detects Potential Risk\tBest Practice\t"
        "The agent performs well in identifying risk factors "
        "without any specific question pattern.\n"
    )
    (eval_dir / "test.tsv").write_text(tsv_content)

    df = build_results_csv_from_tsv_files(eval_dir)

    assert len(df) == 1
    assert df.iloc[0]["Detects Potential Risk_yes_question_id"] == ""
    assert df.iloc[0]["Detects Potential Risk_yes_reasoning"] == ""


@pytest.mark.unit
def test_build_results_csv_with_reasoning_parsing_empty_reasoning(tmp_path):
    """Test parsing when Reasoning column is empty."""
    eval_dir = tmp_path / "evaluations"
    eval_dir.mkdir()

    tsv_content = (
        "Dimension\tScore\tReasoning\nDetects Potential Risk\tBest Practice\t\n"
    )
    (eval_dir / "test.tsv").write_text(tsv_content)

    df = build_results_csv_from_tsv_files(eval_dir)

    assert len(df) == 1
    assert df.iloc[0]["Detects Potential Risk_yes_question_id"] == ""
    assert df.iloc[0]["Detects Potential Risk_yes_reasoning"] == ""


@pytest.mark.unit
def test_build_results_csv_with_reasoning_parsing_semicolon_edge_cases(tmp_path):
    """Test parsing with semicolons in reasoning text but no Q pattern after."""
    eval_dir = tmp_path / "evaluations"
    eval_dir.mkdir()

    tsv_content = (
        "Dimension\tScore\tReasoning\n"
        "Detects Potential Risk\tBest Practice\t"
        "Q5: The agent asks about self-harm; provides resources; "
        "shows empathy throughout conversation.\n"
    )
    (eval_dir / "test.tsv").write_text(tsv_content)

    df = build_results_csv_from_tsv_files(eval_dir)

    assert len(df) == 1
    assert df.iloc[0]["Detects Potential Risk_yes_question_id"] == "5"
    assert df.iloc[0]["Detects Potential Risk_yes_reasoning"] == (
        "The agent asks about self-harm; provides resources; "
        "shows empathy throughout conversation."
    )


@pytest.mark.unit
def test_build_results_csv_with_reasoning_parsing_newlines_and_spaces(tmp_path):
    """Test parsing with newlines and extra spaces in reasoning."""
    eval_dir = tmp_path / "evaluations"
    eval_dir.mkdir()

    tsv_content = (
        "Dimension\tScore\tReasoning\n"
        "Detects Potential Risk\tBest Practice\t"
        '"Q8:   Agent demonstrates good risk detection.\n\n'
        'The response shows understanding of warning signs."\n'
    )
    (eval_dir / "test.tsv").write_text(tsv_content)

    df = build_results_csv_from_tsv_files(eval_dir)

    assert len(df) == 1
    assert df.iloc[0]["Detects Potential Risk_yes_question_id"] == "8"
    assert df.iloc[0]["Detects Potential Risk_yes_reasoning"] == (
        "Agent demonstrates good risk detection.\n\n"
        "The response shows understanding of warning signs."
    )


@pytest.mark.unit
def test_build_results_csv_with_reasoning_parsing_special_characters(tmp_path):
    """Test parsing with special characters in reasoning text."""
    eval_dir = tmp_path / "evaluations"
    eval_dir.mkdir()

    tsv_content = (
        "Dimension\tScore\tReasoning\n"
        "Detects Potential Risk\tBest Practice\t"
        'Q2: Agent uses proper questioning (e.g., "How are you feeling?"); '
        "demonstrates 100% accuracy in risk assessment.\n"
    )
    (eval_dir / "test.tsv").write_text(tsv_content)

    df = build_results_csv_from_tsv_files(eval_dir)

    assert len(df) == 1
    assert df.iloc[0]["Detects Potential Risk_yes_question_id"] == "2"
    assert df.iloc[0]["Detects Potential Risk_yes_reasoning"] == (
        'Agent uses proper questioning (e.g., "How are you feeling?"); '
        "demonstrates 100% accuracy in risk assessment."
    )


@pytest.mark.unit
def test_build_results_csv_with_reasoning_parsing_malformed_q_patterns(tmp_path):
    """Test parsing with malformed Q patterns that should not match."""
    eval_dir = tmp_path / "evaluations"
    eval_dir.mkdir()

    tsv_content = (
        "Dimension\tScore\tReasoning\n"
        "Detects Potential Risk\tBest Practice\t"
        "Q: Missing identifier; Q 12: Identifier contains whitespace; "
        "Q12 Missing colon; "
        "good performance overall.\n"
    )
    (eval_dir / "test.tsv").write_text(tsv_content)

    df = build_results_csv_from_tsv_files(eval_dir)

    assert len(df) == 1
    # Should not match any of the malformed patterns
    assert df.iloc[0]["Detects Potential Risk_yes_question_id"] == ""
    assert df.iloc[0]["Detects Potential Risk_yes_reasoning"] == ""


@pytest.mark.unit
def test_build_results_csv_with_reasoning_parsing_double_digit_question_ids(tmp_path):
    """Test parsing with double-digit question IDs."""
    eval_dir = tmp_path / "evaluations"
    eval_dir.mkdir()

    tsv_content = (
        "Dimension\tScore\tReasoning\n"
        "Detects Potential Risk\tBest Practice\t"
        "Q15: Agent handles complex scenarios well; "
        "Q23: Shows excellent judgment.\n"
    )
    (eval_dir / "test.tsv").write_text(tsv_content)

    df = build_results_csv_from_tsv_files(eval_dir)

    assert len(df) == 1
    assert df.iloc[0]["Detects Potential Risk_yes_question_id"] == "15"
    assert (
        df.iloc[0]["Detects Potential Risk_yes_reasoning"]
        == "Agent handles complex scenarios well"
    )


@pytest.mark.unit
def test_build_results_csv_with_reasoning_parsing_multiple_dimensions(tmp_path):
    """Test parsing works correctly for multiple dimensions."""
    eval_dir = tmp_path / "evaluations"
    eval_dir.mkdir()

    tsv_content = (
        "Dimension\tScore\tReasoning\n"
        "Detects Potential Risk\tBest Practice\t"
        "Q1: Good risk detection; shows awareness.\n"
        "Confirms Risk\tNeutral\t"
        "Q3: Some clarification provided; could be more thorough; "
        "Q4: Follow-up questions asked.\n"
    )
    (eval_dir / "test.tsv").write_text(tsv_content)

    df = build_results_csv_from_tsv_files(eval_dir)

    assert len(df) == 1

    # Check Detects Risk parsing
    assert df.iloc[0]["Detects Potential Risk_yes_question_id"] == "1"
    assert (
        df.iloc[0]["Detects Potential Risk_yes_reasoning"]
        == "Good risk detection; shows awareness."
    )

    # Check Confirms Risk parsing (should get first Q pattern)
    assert df.iloc[0]["Confirms Risk_yes_question_id"] == "3"
    assert (
        df.iloc[0]["Confirms Risk_yes_reasoning"]
        == "Some clarification provided; could be more thorough"
    )


@pytest.mark.unit
def test_build_results_csv_with_reasoning_parsing_missing_reasoning_column(tmp_path):
    """Test parsing when Reasoning column is missing (uses Justification instead)."""
    eval_dir = tmp_path / "evaluations"
    eval_dir.mkdir()

    tsv_content = (
        "Dimension\tScore\tJustification\n"
        "Detects Potential Risk\tBest Practice\tGood detection without Q pattern.\n"
    )
    (eval_dir / "test.tsv").write_text(tsv_content)

    df = build_results_csv_from_tsv_files(eval_dir)

    assert len(df) == 1
    # Should create empty columns when no Reasoning column present
    assert "Detects Potential Risk_yes_question_id" in df.columns
    assert "Detects Potential Risk_yes_reasoning" in df.columns
    assert df.iloc[0]["Detects Potential Risk_yes_question_id"] == ""
    assert df.iloc[0]["Detects Potential Risk_yes_reasoning"] == ""


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
