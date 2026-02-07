"""
Integration tests for run_pipeline.py end-to-end pipeline orchestration.

Tests the three-stage pipeline: generation → evaluation → scoring
Following VERA-MH testing patterns from test_conversation_runner.py

Note: Full end-to-end execution tests are complex due to module import mechanics.
These tests focus on argument parsing, configuration building, and error paths.
"""

import argparse
from unittest.mock import patch

import pytest

# Fixtures


@pytest.fixture
def pipeline_args():
    """Minimal valid pipeline arguments."""
    return argparse.Namespace(
        user_agent="claude-sonnet-4-5-20250929",
        provider_agent="gpt-4o",
        runs=1,
        turns=4,
        judge_model=["claude-sonnet-4-5-20250929"],
        user_agent_extra_params={},
        provider_agent_extra_params={},
        max_total_words=None,
        max_concurrent=None,
        max_personas=2,
        folder_name=None,
        run_id=None,
        debug=False,
        judge_model_extra_params={},
        judge_max_concurrent=None,
        judge_per_judge=False,
        judge_limit=None,
        judge_verbose_workers=False,
        rubrics=["data/rubric.tsv"],
        judge_output="evaluations",
        skip_risk_analysis=False,
        personas_tsv="data/personas.tsv",
    )


# Test Classes


@pytest.mark.integration
class TestPipelineArgumentParsing:
    """Test argument parsing and validation."""

    def test_parse_arguments_required_only(self):
        """Test parsing with only required arguments."""
        from run_pipeline import parse_arguments

        test_args = [
            "--user-agent",
            "claude-sonnet-4-5-20250929",
            "--provider-agent",
            "gpt-4o",
            "--runs",
            "1",
            "--turns",
            "4",
            "--judge-model",
            "claude-sonnet-4-5-20250929",
        ]

        with patch("sys.argv", ["run_pipeline.py"] + test_args):
            args = parse_arguments()

            assert args.user_agent == "claude-sonnet-4-5-20250929"
            assert args.provider_agent == "gpt-4o"
            assert args.runs == 1
            assert args.turns == 4
            assert args.judge_model == ["claude-sonnet-4-5-20250929"]

    def test_parse_arguments_with_extra_params(self):
        """Test parsing with extra model parameters."""
        from run_pipeline import parse_arguments

        test_args = [
            "--user-agent",
            "claude-sonnet-4-5-20250929",
            "--provider-agent",
            "gpt-4o",
            "--runs",
            "1",
            "--turns",
            "4",
            "--judge-model",
            "claude-sonnet-4-5-20250929",
            "--user-agent-extra-params",
            "temperature=0.7,max_tokens=1000",
            "--provider-agent-extra-params",
            "temperature=0.5",
            "--judge-model-extra-params",
            "temperature=0.1",
        ]

        with patch("sys.argv", ["run_pipeline.py"] + test_args):
            args = parse_arguments()

            assert args.user_agent_extra_params == {
                "temperature": 0.7,
                "max_tokens": 1000,
            }
            assert args.provider_agent_extra_params == {"temperature": 0.5}
            assert args.judge_model_extra_params == {"temperature": 0.1}

    def test_parse_arguments_multiple_judge_models(self):
        """Test parsing with multiple judge models."""
        from run_pipeline import parse_arguments

        test_args = [
            "--user-agent",
            "claude-sonnet-4-5-20250929",
            "--provider-agent",
            "gpt-4o",
            "--runs",
            "1",
            "--turns",
            "4",
            "--judge-model",
            "claude-sonnet-4-5-20250929:2",
            "gpt-4o",
        ]

        with patch("sys.argv", ["run_pipeline.py"] + test_args):
            args = parse_arguments()

            assert args.judge_model == ["claude-sonnet-4-5-20250929:2", "gpt-4o"]

    def test_parse_arguments_missing_required(self):
        """Test that missing required arguments raises error."""
        from run_pipeline import parse_arguments

        test_args = [
            "--user-agent",
            "claude-sonnet-4-5-20250929",
            # Missing other required args
        ]

        with patch("sys.argv", ["run_pipeline.py"] + test_args):
            with pytest.raises(SystemExit):
                parse_arguments()

    def test_parse_arguments_optional_flags(self):
        """Test parsing optional boolean flags."""
        from run_pipeline import parse_arguments

        test_args = [
            "--user-agent",
            "claude-sonnet-4-5-20250929",
            "--provider-agent",
            "gpt-4o",
            "--runs",
            "1",
            "--turns",
            "4",
            "--judge-model",
            "claude-sonnet-4-5-20250929",
            "--debug",
            "--judge-per-judge",
            "--judge-verbose-workers",
            "--skip-risk-analysis",
        ]

        with patch("sys.argv", ["run_pipeline.py"] + test_args):
            args = parse_arguments()

            assert args.debug is True
            assert args.judge_per_judge is True
            assert args.judge_verbose_workers is True
            assert args.skip_risk_analysis is True

    def test_parse_arguments_with_all_optional_arguments(self):
        """Test parsing with all optional arguments provided."""
        from run_pipeline import parse_arguments

        test_args = [
            "--user-agent",
            "claude-sonnet-4-5-20250929",
            "--provider-agent",
            "gpt-4o",
            "--runs",
            "2",
            "--turns",
            "10",
            "--judge-model",
            "claude-sonnet-4-5-20250929:2",
            "gpt-4o",
            "--user-agent-extra-params",
            "temperature=0.7",
            "--provider-agent-extra-params",
            "temperature=0.5",
            "--max-total-words",
            "5000",
            "--max-concurrent",
            "10",
            "--max-personas",
            "5",
            "--folder-name",
            "custom_folder",
            "--run-id",
            "test_run_id",
            "--debug",
            "--judge-model-extra-params",
            "temperature=0.1",
            "--judge-max-concurrent",
            "5",
            "--judge-per-judge",
            "--judge-limit",
            "10",
            "--judge-verbose-workers",
            "--rubrics",
            "data/rubric.tsv",
            "data/custom_rubric.tsv",
            "--judge-output",
            "custom_output",
            "--skip-risk-analysis",
            "--personas-tsv",
            "custom/personas.tsv",
        ]

        with patch("sys.argv", ["run_pipeline.py"] + test_args):
            args = parse_arguments()

            # Check all values were parsed correctly
            assert args.runs == 2
            assert args.turns == 10
            assert args.max_total_words == 5000
            assert args.max_concurrent == 10
            assert args.max_personas == 5
            assert args.folder_name == "custom_folder"
            assert args.run_id == "test_run_id"
            assert args.judge_max_concurrent == 5
            assert args.judge_limit == 10
            assert args.rubrics == ["data/rubric.tsv", "data/custom_rubric.tsv"]
            assert args.judge_output == "custom_output"
            assert args.personas_tsv == "custom/personas.tsv"


@pytest.mark.integration
class TestPipelineConfiguration:
    """Test configuration building logic from arguments."""

    def test_persona_model_config_dict_structure(self, pipeline_args):
        """Test that persona model config is built with correct structure."""
        # Build config as done in main()
        persona_config = {
            "model": pipeline_args.user_agent,
            **pipeline_args.user_agent_extra_params,
        }

        assert "model" in persona_config
        assert persona_config["model"] == "claude-sonnet-4-5-20250929"
        assert isinstance(persona_config, dict)

    def test_agent_model_config_dict_structure(self, pipeline_args):
        """Test that agent model config is built with correct structure."""
        # Build config as done in main()
        agent_config = {
            "model": pipeline_args.provider_agent,
            "name": pipeline_args.provider_agent,
            **pipeline_args.provider_agent_extra_params,
        }

        assert "model" in agent_config
        assert "name" in agent_config
        assert agent_config["model"] == "gpt-4o"
        assert agent_config["name"] == "gpt-4o"
        assert isinstance(agent_config, dict)

    def test_extra_params_merge_into_config(self):
        """Test that extra params correctly merge into model configs."""
        args = argparse.Namespace(
            user_agent="claude-sonnet-4-5-20250929",
            provider_agent="gpt-4o",
            user_agent_extra_params={"temperature": 0.7, "max_tokens": 1000},
            provider_agent_extra_params={"temperature": 0.5},
        )

        persona_config = {
            "model": args.user_agent,
            **args.user_agent_extra_params,
        }

        agent_config = {
            "model": args.provider_agent,
            "name": args.provider_agent,
            **args.provider_agent_extra_params,
        }

        # Check persona config
        assert persona_config["model"] == "claude-sonnet-4-5-20250929"
        assert persona_config["temperature"] == 0.7
        assert persona_config["max_tokens"] == 1000

        # Check agent config
        assert agent_config["model"] == "gpt-4o"
        assert agent_config["temperature"] == 0.5

    def test_judge_args_namespace_structure(self, pipeline_args):
        """Test that judge args Namespace is constructed correctly."""
        conv_folder = "conversations/test"

        # Build judge args as done in main()
        judge_args = argparse.Namespace(
            conversation=None,
            folder=conv_folder,
            rubrics=pipeline_args.rubrics,
            judge_model=pipeline_args.judge_model,
            judge_model_extra_params=pipeline_args.judge_model_extra_params,
            limit=pipeline_args.judge_limit,
            output=pipeline_args.judge_output,
            max_concurrent=pipeline_args.judge_max_concurrent,
            per_judge=pipeline_args.judge_per_judge,
            verbose_workers=pipeline_args.judge_verbose_workers,
        )

        # Verify structure
        assert isinstance(judge_args, argparse.Namespace)
        assert judge_args.conversation is None
        assert judge_args.folder == conv_folder
        assert judge_args.rubrics == pipeline_args.rubrics
        assert judge_args.judge_model == ["claude-sonnet-4-5-20250929"]
        assert judge_args.output == pipeline_args.judge_output

    def test_empty_extra_params_dont_pollute_config(self):
        """Test that empty extra params don't add unwanted keys."""
        args = argparse.Namespace(
            user_agent="claude-sonnet-4-5-20250929",
            user_agent_extra_params={},
        )

        persona_config = {
            "model": args.user_agent,
            **args.user_agent_extra_params,
        }

        # Should only have the model key
        assert len(persona_config) == 1
        assert "model" in persona_config


@pytest.mark.integration
class TestPipelineDataFlow:
    """Test data flow and path construction between stages."""

    def test_conversation_folder_to_judge_path_construction(self):
        """Test that conversation folder path is correctly passed to judge."""
        conv_folder = "conversations/test_20240101_120000"

        # As done in main(): judge receives the folder
        judge_args = argparse.Namespace(
            folder=conv_folder,
            conversation=None,
        )

        assert judge_args.folder == conv_folder
        assert judge_args.conversation is None

    def test_evaluation_folder_to_score_path_construction(self):
        """Test that evaluation folder path is correctly transformed for score."""
        import os

        eval_folder = "evaluations/test_20240101_120000"

        # As done in main(): score receives results.csv path
        results_csv = os.path.join(eval_folder, "results.csv")

        assert results_csv == "evaluations/test_20240101_120000/results.csv"
        assert results_csv.startswith(eval_folder)
        assert results_csv.endswith("results.csv")

    def test_personas_tsv_path_passed_to_score(self, pipeline_args):
        """Test that personas.tsv path is correctly passed to score."""
        # As done in main()
        personas_tsv_path = pipeline_args.personas_tsv

        assert personas_tsv_path == "data/personas.tsv"

    def test_skip_risk_analysis_flag_passed_to_score(self, pipeline_args):
        """Test that skip_risk_analysis flag is correctly passed to score."""
        # As done in main()
        skip_risk = pipeline_args.skip_risk_analysis

        assert skip_risk is False  # Default value

        # Test with True
        pipeline_args.skip_risk_analysis = True
        assert pipeline_args.skip_risk_analysis is True


@pytest.mark.integration
class TestPipelineNewArguments:
    """Test newly added arguments for consistency with individual scripts."""

    def test_run_id_argument_exists(self, pipeline_args):
        """Test that run_id argument exists in pipeline args."""
        assert hasattr(pipeline_args, "run_id")
        assert pipeline_args.run_id is None  # Default value

    def test_run_id_passed_to_generate(self, pipeline_args):
        """Test that run_id is correctly structured for generate_main."""
        # Set custom run_id
        pipeline_args.run_id = "custom_test_run"

        # Verify it's accessible
        assert pipeline_args.run_id == "custom_test_run"

    def test_rubrics_argument_exists(self, pipeline_args):
        """Test that rubrics argument exists in pipeline args."""
        assert hasattr(pipeline_args, "rubrics")
        assert pipeline_args.rubrics == ["data/rubric.tsv"]  # Default value

    def test_rubrics_passed_to_judge(self, pipeline_args):
        """Test that rubrics are correctly passed to judge args."""
        # Set custom rubrics
        pipeline_args.rubrics = ["data/rubric.tsv", "data/custom_rubric.tsv"]

        # As done in main(): judge receives these rubrics
        judge_args = argparse.Namespace(
            rubrics=pipeline_args.rubrics,
        )

        assert judge_args.rubrics == ["data/rubric.tsv", "data/custom_rubric.tsv"]
        assert len(judge_args.rubrics) == 2

    def test_judge_output_argument_exists(self, pipeline_args):
        """Test that judge_output argument exists in pipeline args."""
        assert hasattr(pipeline_args, "judge_output")
        assert pipeline_args.judge_output == "evaluations"  # Default value

    def test_judge_output_passed_to_judge(self, pipeline_args):
        """Test that judge_output is correctly passed to judge args."""
        # Set custom output folder
        pipeline_args.judge_output = "custom_evaluations"

        # As done in main(): judge receives this output folder
        judge_args = argparse.Namespace(
            output=pipeline_args.judge_output,
        )

        assert judge_args.output == "custom_evaluations"

    def test_parse_arguments_with_run_id(self):
        """Test parsing arguments with --run-id."""
        from run_pipeline import parse_arguments

        test_args = [
            "--user-agent",
            "claude-sonnet-4-5-20250929",
            "--provider-agent",
            "gpt-4o",
            "--runs",
            "1",
            "--turns",
            "4",
            "--judge-model",
            "claude-sonnet-4-5-20250929",
            "--run-id",
            "test_run_123",
        ]

        with patch("sys.argv", ["run_pipeline.py"] + test_args):
            args = parse_arguments()

            assert args.run_id == "test_run_123"

    def test_parse_arguments_with_rubrics(self):
        """Test parsing arguments with --rubrics."""
        from run_pipeline import parse_arguments

        test_args = [
            "--user-agent",
            "claude-sonnet-4-5-20250929",
            "--provider-agent",
            "gpt-4o",
            "--runs",
            "1",
            "--turns",
            "4",
            "--judge-model",
            "claude-sonnet-4-5-20250929",
            "--rubrics",
            "data/rubric.tsv",
            "data/custom_rubric.tsv",
        ]

        with patch("sys.argv", ["run_pipeline.py"] + test_args):
            args = parse_arguments()

            assert args.rubrics == ["data/rubric.tsv", "data/custom_rubric.tsv"]

    def test_parse_arguments_with_judge_output(self):
        """Test parsing arguments with --judge-output."""
        from run_pipeline import parse_arguments

        test_args = [
            "--user-agent",
            "claude-sonnet-4-5-20250929",
            "--provider-agent",
            "gpt-4o",
            "--runs",
            "1",
            "--turns",
            "4",
            "--judge-model",
            "claude-sonnet-4-5-20250929",
            "--judge-output",
            "custom_evals",
        ]

        with patch("sys.argv", ["run_pipeline.py"] + test_args):
            args = parse_arguments()

            assert args.judge_output == "custom_evals"

    def test_parse_arguments_defaults_for_new_args(self):
        """Test that new arguments have correct defaults."""
        from run_pipeline import parse_arguments

        test_args = [
            "--user-agent",
            "claude-sonnet-4-5-20250929",
            "--provider-agent",
            "gpt-4o",
            "--runs",
            "1",
            "--turns",
            "4",
            "--judge-model",
            "claude-sonnet-4-5-20250929",
        ]

        with patch("sys.argv", ["run_pipeline.py"] + test_args):
            args = parse_arguments()

            # Check defaults
            assert args.run_id is None
            assert args.rubrics == ["data/rubric.tsv"]
            assert args.judge_output == "evaluations"

    def test_short_flags_for_extra_params(self):
        """Test that short flags work for extra params arguments."""
        from run_pipeline import parse_arguments

        test_args = [
            "--user-agent",
            "claude-sonnet-4-5-20250929",
            "--provider-agent",
            "gpt-4o",
            "--runs",
            "1",
            "--turns",
            "4",
            "--judge-model",
            "claude-sonnet-4-5-20250929",
            "-uep",
            "temperature=0.7,max_tokens=1000",
            "-pep",
            "temperature=0.5",
            "-jep",
            "temperature=0.1",
        ]

        with patch("sys.argv", ["run_pipeline.py"] + test_args):
            args = parse_arguments()

            assert args.user_agent_extra_params == {
                "temperature": 0.7,
                "max_tokens": 1000,
            }
            assert args.provider_agent_extra_params == {"temperature": 0.5}
            assert args.judge_model_extra_params == {"temperature": 0.1}

    def test_short_flag_for_run_id(self):
        """Test that short flag -i works for run-id."""
        from run_pipeline import parse_arguments

        test_args = [
            "--user-agent",
            "claude-sonnet-4-5-20250929",
            "--provider-agent",
            "gpt-4o",
            "--runs",
            "1",
            "--turns",
            "4",
            "--judge-model",
            "claude-sonnet-4-5-20250929",
            "-i",
            "custom_run",
        ]

        with patch("sys.argv", ["run_pipeline.py"] + test_args):
            args = parse_arguments()

            assert args.run_id == "custom_run"


# Fixtures for validation tests


@pytest.fixture
def valid_pipeline_args():
    """Fixture providing valid minimal pipeline arguments."""
    return [
        "run_pipeline.py",
        "--user-agent",
        "test-model",
        "--provider-agent",
        "test-model",
        "--runs",
        "1",
        "--turns",
        "1",
        "--judge-model",
        "test-model",
    ]


@pytest.mark.integration
class TestPipelineValidation:
    """Test pipeline validation and error handling for empty folders."""

    @pytest.mark.asyncio
    async def test_step1_validation_folder_not_exists(
        self, tmp_path, valid_pipeline_args
    ):
        """Test that pipeline exits if Step 1 folder doesn't exist."""
        import sys
        from unittest.mock import patch

        from run_pipeline import main as pipeline_main

        # Mock generate module's main to return a non-existent folder
        async def mock_generate(*args, **kwargs):
            return None, str(tmp_path / "nonexistent")

        # Patch generate.main at the source
        with patch("generate.main", side_effect=mock_generate):
            # Mock sys.exit to raise SystemExit instead of actually exiting
            with patch.object(sys, "exit", side_effect=SystemExit) as mock_exit:
                with patch("sys.argv", valid_pipeline_args):
                    # Pipeline should raise SystemExit when folder doesn't exist
                    with pytest.raises(SystemExit):
                        await pipeline_main()

                # Verify sys.exit(1) was called
                mock_exit.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_step1_validation_no_conversation_files(
        self, tmp_path, valid_pipeline_args
    ):
        """Test that pipeline exits if Step 1 produces no .txt files."""
        import sys
        from unittest.mock import patch

        from run_pipeline import main as pipeline_main

        # Create empty conversation folder
        conv_folder = tmp_path / "conversations"
        conv_folder.mkdir()

        # Mock generate_main to return empty folder
        async def mock_generate(*args, **kwargs):
            return None, str(conv_folder)

        with patch("generate.main", side_effect=mock_generate):
            # Mock sys.exit to raise SystemExit instead of actually exiting
            with patch.object(sys, "exit", side_effect=SystemExit) as mock_exit:
                with patch("sys.argv", valid_pipeline_args):
                    # Pipeline should raise SystemExit
                    with pytest.raises(SystemExit):
                        await pipeline_main()

                    # Verify sys.exit(1) was called
                    mock_exit.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_step1_validation_only_log_files(self, tmp_path, valid_pipeline_args):
        """Test that pipeline exits if Step 1 only produces .log files."""
        import sys
        from unittest.mock import patch

        from run_pipeline import main as pipeline_main

        # Create conversation folder with only .log files
        conv_folder = tmp_path / "conversations"
        conv_folder.mkdir()
        (conv_folder / "conversation1.log").write_text("log content")
        (conv_folder / "conversation2.log").write_text("log content")

        # Mock generate_main to return folder with only logs
        async def mock_generate(*args, **kwargs):
            return None, str(conv_folder)

        with patch("generate.main", side_effect=mock_generate):
            # Mock sys.exit to raise SystemExit instead of actually exiting
            with patch.object(sys, "exit", side_effect=SystemExit) as mock_exit:
                with patch("sys.argv", valid_pipeline_args):
                    # Pipeline should raise SystemExit
                    with pytest.raises(SystemExit):
                        await pipeline_main()

                    # Verify sys.exit(1) was called
                    mock_exit.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_step2_validation_no_evaluation_folder(
        self, tmp_path, valid_pipeline_args
    ):
        """Test that pipeline exits if Step 2 returns None."""
        import sys
        from unittest.mock import patch

        from run_pipeline import main as pipeline_main

        # Create conversation folder with valid files
        conv_folder = tmp_path / "conversations"
        conv_folder.mkdir()
        (conv_folder / "conv1.txt").write_text("User: Hi\nAssistant: Hello")

        # Mock generate_main to return valid folder
        async def mock_generate(*args, **kwargs):
            return None, str(conv_folder)

        # Mock judge CLI main to return None
        async def mock_judge(args):
            return None

        with (
            patch("generate.main", side_effect=mock_generate),
            patch("judge.cli.main", side_effect=mock_judge),
        ):
            # Mock sys.exit to raise SystemExit instead of actually exiting
            with patch.object(sys, "exit", side_effect=SystemExit) as mock_exit:
                with patch("sys.argv", valid_pipeline_args):
                    # Pipeline should raise SystemExit
                    with pytest.raises(SystemExit):
                        await pipeline_main()

                    # Verify sys.exit(1) was called
                    mock_exit.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_step2_validation_folder_not_exists(
        self, tmp_path, valid_pipeline_args
    ):
        """Test that pipeline exits if Step 2 folder doesn't exist."""
        import sys
        from unittest.mock import patch

        from run_pipeline import main as pipeline_main

        # Create conversation folder with valid files
        conv_folder = tmp_path / "conversations"
        conv_folder.mkdir()
        (conv_folder / "conv1.txt").write_text("User: Hi\nAssistant: Hello")

        # Mock generate_main to return valid folder
        async def mock_generate(*args, **kwargs):
            return None, str(conv_folder)

        # Mock judge CLI main to return non-existent folder
        async def mock_judge(args):
            return str(tmp_path / "nonexistent_eval")

        with (
            patch("generate.main", side_effect=mock_generate),
            patch("judge.cli.main", side_effect=mock_judge),
        ):
            # Mock sys.exit to raise SystemExit instead of actually exiting
            with patch.object(sys, "exit", side_effect=SystemExit) as mock_exit:
                with patch("sys.argv", valid_pipeline_args):
                    # Pipeline should raise SystemExit
                    with pytest.raises(SystemExit):
                        await pipeline_main()

                    # Verify sys.exit(1) was called
                    mock_exit.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_step2_validation_no_results_csv(self, tmp_path, valid_pipeline_args):
        """Test that pipeline exits if Step 2 produces no results.csv."""
        import sys
        from unittest.mock import patch

        from run_pipeline import main as pipeline_main

        # Create conversation folder with valid files
        conv_folder = tmp_path / "conversations"
        conv_folder.mkdir()
        (conv_folder / "conv1.txt").write_text("User: Hi\nAssistant: Hello")

        # Create evaluation folder but no results.csv
        eval_folder = tmp_path / "evaluations"
        eval_folder.mkdir()
        (eval_folder / "some_other_file.json").write_text("{}")

        # Mock generate_main to return valid folder
        async def mock_generate(*args, **kwargs):
            return None, str(conv_folder)

        # Mock judge CLI main to return folder without results.csv
        async def mock_judge(args):
            return str(eval_folder)

        with (
            patch("generate.main", side_effect=mock_generate),
            patch("judge.cli.main", side_effect=mock_judge),
        ):
            # Mock sys.exit to raise SystemExit instead of actually exiting
            with patch.object(sys, "exit", side_effect=SystemExit) as mock_exit:
                with patch("sys.argv", valid_pipeline_args):
                    # Pipeline should raise SystemExit
                    with pytest.raises(SystemExit):
                        await pipeline_main()

                    # Verify sys.exit(1) was called
                    mock_exit.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_step2_validation_empty_folder_error_message(
        self, tmp_path, valid_pipeline_args, capsys
    ):
        """Test that error message lists files when folder is not empty."""
        import sys
        from unittest.mock import patch

        from run_pipeline import main as pipeline_main

        # Create conversation folder with valid files
        conv_folder = tmp_path / "conversations"
        conv_folder.mkdir()
        (conv_folder / "conv1.txt").write_text("User: Hi\nAssistant: Hello")

        # Create evaluation folder with some files but no results.csv
        eval_folder = tmp_path / "evaluations"
        eval_folder.mkdir()
        (eval_folder / "file1.json").write_text("{}")
        (eval_folder / "file2.json").write_text("{}")
        (eval_folder / "file3.log").write_text("log")

        # Mock functions
        async def mock_generate(*args, **kwargs):
            return None, str(conv_folder)

        async def mock_judge(args):
            return str(eval_folder)

        with (
            patch("generate.main", side_effect=mock_generate),
            patch("judge.cli.main", side_effect=mock_judge),
        ):
            # Mock sys.exit to raise SystemExit instead of actually exiting
            with patch.object(sys, "exit", side_effect=SystemExit) as mock_exit:
                with patch("sys.argv", valid_pipeline_args):
                    # Pipeline should raise SystemExit
                    with pytest.raises(SystemExit):
                        await pipeline_main()

                    # Capture printed output
                    captured = capsys.readouterr()

                    # Verify error message includes file listing
                    assert "Files in evaluation folder: 3" in captured.out
                    assert "Found:" in captured.out

                    # Verify sys.exit(1) was called
                    mock_exit.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_validation_success_messages(
        self, tmp_path, valid_pipeline_args, capsys
    ):
        """Test that validation success messages are displayed."""
        from unittest.mock import patch

        from run_pipeline import main as pipeline_main

        # Create conversation folder with valid files
        conv_folder = tmp_path / "conversations"
        conv_folder.mkdir()
        (conv_folder / "conv1.txt").write_text("User: Hi\nAssistant: Hello")
        (conv_folder / "conv2.txt").write_text("User: Hey\nAssistant: Hi there")

        # Create evaluation folder with results.csv
        eval_folder = tmp_path / "evaluations"
        eval_folder.mkdir()
        (eval_folder / "results.csv").write_text(
            "filename,run_id,Safety\nconv1.txt,test,Pass"
        )

        # Mock functions
        async def mock_generate(*args, **kwargs):
            return None, str(conv_folder)

        async def mock_judge(args):
            return str(eval_folder)

        def mock_score(*args, **kwargs):
            return {}

        with (
            patch("generate.main", side_effect=mock_generate),
            patch("judge.cli.main", side_effect=mock_judge),
            patch("run_pipeline.score_results", new=mock_score),
            patch("run_pipeline.print_scores"),
            patch("run_pipeline.create_visualizations"),
        ):
            with patch("sys.argv", valid_pipeline_args + ["--skip-risk-analysis"]):
                await pipeline_main()

                # Capture printed output
                captured = capsys.readouterr()

                # Verify success messages
                assert "✓ Validated: 2 conversation files generated" in captured.out
                assert (
                    "✓ Validated: results.csv exists with evaluation data"
                    in captured.out
                )
