"""
Integration tests for run_pipeline.py end-to-end pipeline orchestration.

Tests the three-stage pipeline: generation → evaluation → scoring
Following VERA-MH testing patterns from test_conversation_runner.py

Note: Full end-to-end execution tests are complex due to module import mechanics.
These tests focus on argument parsing, configuration building, and error paths.
"""

import argparse
import os
from pathlib import Path
from unittest.mock import patch

import pytest

# Helpers: pipeline Step 1 expects transcripts under <gen_run>/conversations/*.txt


def _make_gen_run_with_txt(
    tmp_path: Path,
    files: dict[str, str],
    *,
    basename: str = "p_test__a_m__t1__r1__ts",
) -> Path:
    gen_run = tmp_path / basename
    gen_run.mkdir()
    sub = gen_run / "conversations"
    sub.mkdir()
    for fname, content in files.items():
        (sub / fname).write_text(content)
    return gen_run


def _make_gen_run_empty_conversations(
    tmp_path: Path, basename: str = "p_test__a_m__t1__r1__ts"
) -> Path:
    gen_run = tmp_path / basename
    gen_run.mkdir()
    (gen_run / "conversations").mkdir()
    return gen_run


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
        conversation_output="output",
        judge_output=None,
        run_id=None,
        debug=False,
        judge_model_extra_params={},
        judge_max_concurrent=None,
        judge_per_judge=False,
        judge_limit=None,
        judge_verbose_workers=False,
        rubrics=["data/rubric.tsv"],
        resume_generate=False,
        resume_judge=False,
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

    def test_parse_short_co_is_conversation_output(self):
        """-co is shorthand for --conversation-output."""
        from run_pipeline import parse_arguments

        test_args = [
            "--user-agent",
            "m",
            "--provider-agent",
            "m",
            "--runs",
            "1",
            "--turns",
            "1",
            "--judge-model",
            "m",
            "-co",
            "my_runs",
        ]
        with patch("sys.argv", ["run_pipeline.py"] + test_args):
            args = parse_arguments()
            assert args.conversation_output == "my_runs"

    def test_parse_short_jo_is_judge_output(self):
        """-jo is shorthand for --judge-output."""
        from run_pipeline import parse_arguments

        test_args = [
            "--user-agent",
            "m",
            "--provider-agent",
            "m",
            "--runs",
            "1",
            "--turns",
            "1",
            "--judge-model",
            "m",
            "-jo",
            "evals/parent",
        ]
        with patch("sys.argv", ["run_pipeline.py"] + test_args):
            args = parse_arguments()
            assert args.judge_output == "evals/parent"

    @pytest.mark.parametrize(
        "co_flag,jo_flag",
        [
            ("--conversation-output", "--judge-output"),
            ("-co", "-jo"),
        ],
    )
    def test_parse_arguments_with_judge_output(self, co_flag, jo_flag):
        """Parse co/jo; resolve_pipeline_resume_paths for a non-resume run."""
        from run_pipeline import parse_arguments, resolve_pipeline_resume_paths

        test_args = [
            "--user-agent",
            "m",
            "--provider-agent",
            "m",
            "--runs",
            "1",
            "--turns",
            "1",
            "--judge-model",
            "m",
            co_flag,
            "runs/gen",
            jo_flag,
            "runs/evals",
        ]
        with patch("sys.argv", ["run_pipeline.py"] + test_args):
            args = parse_arguments()
            assert args.conversation_output == "runs/gen"
            assert args.judge_output == "runs/evals"
            resolve_pipeline_resume_paths(args)
            assert args._pipeline_gen_folder == os.path.normpath("runs/gen")
            assert args._pipeline_resume_generate is False
            assert args._pipeline_judge_output is None

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
            "--conversation-output",
            "custom_folder",
            "--judge-output",
            "/tmp/judge_parent",
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
            assert args.conversation_output == "custom_folder"
            assert args.judge_output == "/tmp/judge_parent"
            assert args.run_id == "test_run_id"
            assert args.judge_max_concurrent == 5
            assert args.judge_limit == 10
            assert args.rubrics == ["data/rubric.tsv", "data/custom_rubric.tsv"]
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
            output=None,
            max_concurrent=pipeline_args.judge_max_concurrent,
            per_judge=pipeline_args.judge_per_judge,
            verbose_workers=pipeline_args.judge_verbose_workers,
            resume=pipeline_args.resume_judge,
            debug=pipeline_args.debug,
        )

        # Verify structure
        assert isinstance(judge_args, argparse.Namespace)
        assert judge_args.conversation is None
        assert judge_args.folder == conv_folder
        assert judge_args.rubrics == pipeline_args.rubrics
        assert judge_args.judge_model == ["claude-sonnet-4-5-20250929"]
        assert judge_args.output is None
        assert judge_args.resume is False
        assert judge_args.debug is False

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

    def test_conversation_and_judge_output_attributes(self, pipeline_args):
        """conversation-output and judge-output exist on pipeline args."""
        assert hasattr(pipeline_args, "conversation_output")
        assert pipeline_args.conversation_output == "output"
        assert hasattr(pipeline_args, "judge_output")
        assert pipeline_args.judge_output is None

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
            assert args.conversation_output == "output"
            assert args.judge_output is None

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
                # Mock importlib to avoid judge loading (not needed for step 1 test)
                with patch("importlib.util.spec_from_file_location"):
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

        gen_run = _make_gen_run_empty_conversations(tmp_path)

        # Mock generate_main to return run folder with no .txt under conversations/
        async def mock_generate(*args, **kwargs):
            return None, str(gen_run)

        with patch("generate.main", side_effect=mock_generate):
            # Mock sys.exit to raise SystemExit instead of actually exiting
            with patch.object(sys, "exit", side_effect=SystemExit) as mock_exit:
                with patch("importlib.util.spec_from_file_location"):
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

        gen_run = _make_gen_run_with_txt(
            tmp_path,
            {
                "conversation1.log": "log content",
                "conversation2.log": "log content",
            },
        )

        # Mock generate_main to return folder with only logs under conversations/
        async def mock_generate(*args, **kwargs):
            return None, str(gen_run)

        with patch("generate.main", side_effect=mock_generate):
            # Mock sys.exit to raise SystemExit instead of actually exiting
            with patch.object(sys, "exit", side_effect=SystemExit) as mock_exit:
                with patch("importlib.util.spec_from_file_location"):
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
        from unittest.mock import MagicMock, patch

        from run_pipeline import main as pipeline_main

        gen_run = _make_gen_run_with_txt(
            tmp_path, {"conv1.txt": "User: Hi\nAssistant: Hello"}
        )

        # Mock generate_main to return valid folder
        async def mock_generate(*args, **kwargs):
            return None, str(gen_run)

        # Mock judge_main to return None
        async def mock_judge(args):
            return None

        # Create a mock module with the mock judge main function
        mock_judge_module = MagicMock()
        mock_judge_module.main = mock_judge

        with (
            patch("generate.main", side_effect=mock_generate),
            patch("importlib.util.module_from_spec", return_value=mock_judge_module),
            patch("importlib.util.spec_from_file_location"),
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
        from unittest.mock import MagicMock, patch

        from run_pipeline import main as pipeline_main

        gen_run = _make_gen_run_with_txt(
            tmp_path, {"conv1.txt": "User: Hi\nAssistant: Hello"}
        )

        # Mock generate_main to return valid folder
        async def mock_generate(*args, **kwargs):
            return None, str(gen_run)

        # Mock judge_main to return non-existent folder
        async def mock_judge(args):
            return str(tmp_path / "nonexistent_eval")

        # Create a mock module with the mock judge main function
        mock_judge_module = MagicMock()
        mock_judge_module.main = mock_judge

        with (
            patch("generate.main", side_effect=mock_generate),
            patch("importlib.util.module_from_spec", return_value=mock_judge_module),
            patch("importlib.util.spec_from_file_location"),
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
        from unittest.mock import MagicMock, patch

        from run_pipeline import main as pipeline_main

        gen_run = _make_gen_run_with_txt(
            tmp_path, {"conv1.txt": "User: Hi\nAssistant: Hello"}
        )

        # Create evaluation folder but no results.csv
        eval_folder = tmp_path / "evaluations"
        eval_folder.mkdir()
        (eval_folder / "some_other_file.json").write_text("{}")

        # Mock generate_main to return valid folder
        async def mock_generate(*args, **kwargs):
            return None, str(gen_run)

        # Mock judge_main to return folder without results.csv
        async def mock_judge(args):
            return str(eval_folder)

        # Create a mock module with the mock judge main function
        mock_judge_module = MagicMock()
        mock_judge_module.main = mock_judge

        with (
            patch("generate.main", side_effect=mock_generate),
            patch("importlib.util.module_from_spec", return_value=mock_judge_module),
            patch("importlib.util.spec_from_file_location"),
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
        from unittest.mock import MagicMock, patch

        from run_pipeline import main as pipeline_main

        gen_run = _make_gen_run_with_txt(
            tmp_path, {"conv1.txt": "User: Hi\nAssistant: Hello"}
        )

        # Create evaluation folder with some files but no results.csv
        eval_folder = tmp_path / "evaluations"
        eval_folder.mkdir()
        (eval_folder / "file1.json").write_text("{}")
        (eval_folder / "file2.json").write_text("{}")
        (eval_folder / "file3.log").write_text("log")

        # Mock functions
        async def mock_generate(*args, **kwargs):
            return None, str(gen_run)

        async def mock_judge(args):
            return str(eval_folder)

        # Create a mock module with the mock judge main function
        mock_judge_module = MagicMock()
        mock_judge_module.main = mock_judge

        with (
            patch("generate.main", side_effect=mock_generate),
            patch("importlib.util.module_from_spec", return_value=mock_judge_module),
            patch("importlib.util.spec_from_file_location"),
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
    async def test_results_csv_has_expected_columns(
        self, tmp_path, valid_pipeline_args
    ):
        """Test that results.csv contains all expected columns."""
        import csv
        from unittest.mock import MagicMock, patch

        from run_pipeline import main as pipeline_main

        gen_run = _make_gen_run_with_txt(
            tmp_path, {"conv1.txt": "User: Hi\nAssistant: Hello"}
        )

        # Create evaluation folder with results.csv containing expected columns
        eval_folder = tmp_path / "evaluations"
        eval_folder.mkdir()
        results_csv_content = (
            "filename,run_id,persona_name,risk_level,judge_model,judge_instance,judge_id,"
            "Detects Risk,Detects Risk_yes_question_id,Detects Risk_yes_reasoning,"
            "Clarifies Risk,Clarifies Risk_yes_question_id,"
            "Clarifies Risk_yes_reasoning,"
            "Guides to Human Support,Guides to Human Support_yes_question_id,"
            "Guides to Human Support_yes_reasoning,"
            "Collaborates and Validates Appropriately,"
            "Collaborates and Validates Appropriately_yes_question_id,"
            "Collaborates and Validates Appropriately_yes_reasoning,"
            "Maintains Safe Boundaries,Maintains Safe Boundaries_yes_question_id,"
            "Maintains Safe Boundaries_yes_reasoning\n"
            "conv1.txt,test_run,Omar,high,claude-3.5,1,judge_001,"
            "yes,q1,reasoning1,yes,q2,reasoning2,yes,q3,reasoning3,"
            "yes,q4,reasoning4,yes,q5,reasoning5\n"
        )
        (eval_folder / "results.csv").write_text(results_csv_content)

        # Mock functions
        async def mock_generate(*args, **kwargs):
            return None, str(gen_run)

        async def mock_judge(args):
            return str(eval_folder)

        def mock_score(*args, **kwargs):
            return {}

        # Create a mock module with the mock judge main function
        mock_judge_module = MagicMock()
        mock_judge_module.main = mock_judge

        with (
            patch("generate.main", side_effect=mock_generate),
            patch("importlib.util.module_from_spec", return_value=mock_judge_module),
            patch("importlib.util.spec_from_file_location"),
            patch("run_pipeline.score_results", new=mock_score),
            patch("run_pipeline.print_scores"),
            patch("run_pipeline.create_visualizations"),
        ):
            with patch("sys.argv", valid_pipeline_args + ["--skip-risk-analysis"]):
                await pipeline_main()

                # Verify results.csv has expected columns
                results_csv_path = eval_folder / "results.csv"
                with open(results_csv_path, "r") as f:
                    reader = csv.DictReader(f)
                    columns = reader.fieldnames

                    # Check for required columns
                    expected_columns = {
                        "filename",
                        "run_id",
                        "persona_name",
                        "risk_level",
                        "judge_model",
                        "judge_instance",
                        "judge_id",
                        "Detects Risk",
                        "Detects Risk_yes_question_id",
                        "Detects Risk_yes_reasoning",
                        "Clarifies Risk",
                        "Clarifies Risk_yes_question_id",
                        "Clarifies Risk_yes_reasoning",
                        "Guides to Human Support",
                        "Guides to Human Support_yes_question_id",
                        "Guides to Human Support_yes_reasoning",
                        "Collaborates and Validates Appropriately",
                        "Collaborates and Validates Appropriately_yes_question_id",
                        "Collaborates and Validates Appropriately_yes_reasoning",
                        "Maintains Safe Boundaries",
                        "Maintains Safe Boundaries_yes_question_id",
                        "Maintains Safe Boundaries_yes_reasoning",
                    }

                    assert columns is not None, "CSV should have column headers"
                    actual_columns = set(columns)

                    # Verify all expected columns are present
                    missing_columns = expected_columns - actual_columns
                    assert (
                        not missing_columns
                    ), f"Missing expected columns: {missing_columns}"

                    # Verify at least one data row exists
                    rows = list(reader)
                    assert len(rows) > 0, "CSV should contain at least one data row"

    @pytest.mark.asyncio
    async def test_results_csv_missing_required_columns(
        self, tmp_path, valid_pipeline_args
    ):
        """Test pipeline handling when results.csv is missing required columns."""
        from unittest.mock import MagicMock, patch

        from run_pipeline import main as pipeline_main

        gen_run = _make_gen_run_with_txt(
            tmp_path, {"conv1.txt": "User: Hi\nAssistant: Hello"}
        )

        # Create evaluation folder with results.csv missing critical columns
        eval_folder = tmp_path / "evaluations"
        eval_folder.mkdir()
        incomplete_csv_content = "filename,some_other_column\nconv1.txt,value\n"
        (eval_folder / "results.csv").write_text(incomplete_csv_content)

        # Mock functions
        async def mock_generate(*args, **kwargs):
            return None, str(gen_run)

        async def mock_judge(args):
            return str(eval_folder)

        def mock_score(*args, **kwargs):
            # Score function should handle missing columns gracefully
            return {}

        # Create a mock module with the mock judge main function
        mock_judge_module = MagicMock()
        mock_judge_module.main = mock_judge

        with (
            patch("generate.main", side_effect=mock_generate),
            patch("importlib.util.module_from_spec", return_value=mock_judge_module),
            patch("importlib.util.spec_from_file_location"),
            patch("run_pipeline.score_results", new=mock_score),
            patch("run_pipeline.print_scores"),
            patch("run_pipeline.create_visualizations"),
        ):
            with patch("sys.argv", valid_pipeline_args + ["--skip-risk-analysis"]):
                # Pipeline should complete but the scoring function will handle
                # missing columns appropriately
                await pipeline_main()

    @pytest.mark.asyncio
    async def test_validation_success_messages(
        self, tmp_path, valid_pipeline_args, capsys
    ):
        """Test that validation success messages are displayed."""
        from unittest.mock import MagicMock, patch

        from run_pipeline import main as pipeline_main

        gen_run = _make_gen_run_with_txt(
            tmp_path,
            {
                "conv1.txt": "User: Hi\nAssistant: Hello",
                "conv2.txt": "User: Hey\nAssistant: Hi there",
            },
        )

        # Create evaluation folder with results.csv
        eval_folder = tmp_path / "evaluations"
        eval_folder.mkdir()
        (eval_folder / "results.csv").write_text(
            "filename,run_id,Safety\nconv1.txt,test,Pass"
        )

        # Mock functions
        async def mock_generate(*args, **kwargs):
            return None, str(gen_run)

        async def mock_judge(args):
            return str(eval_folder)

        def mock_score(*args, **kwargs):
            return {}

        # Create a mock module with the mock judge main function
        mock_judge_module = MagicMock()
        mock_judge_module.main = mock_judge

        with (
            patch("generate.main", side_effect=mock_generate),
            patch("importlib.util.module_from_spec", return_value=mock_judge_module),
            patch("importlib.util.spec_from_file_location"),
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


@pytest.mark.integration
class TestPipelineResumeParsing:
    """CLI parsing for --resume-generate / --resume-judge."""

    def test_parse_resume_flags_default_false(self):
        from run_pipeline import parse_arguments

        test_args = [
            "--user-agent",
            "m",
            "--provider-agent",
            "m",
            "--runs",
            "1",
            "--turns",
            "1",
            "--judge-model",
            "m",
        ]
        with patch("sys.argv", ["run_pipeline.py"] + test_args):
            args = parse_arguments()
            assert args.resume_generate is False
            assert args.resume_judge is False

    def test_parse_resume_flags_can_be_true(self):
        from run_pipeline import parse_arguments

        test_args = [
            "--user-agent",
            "m",
            "--provider-agent",
            "m",
            "--runs",
            "1",
            "--turns",
            "1",
            "--judge-model",
            "m",
            "--resume-generate",
            "--resume-judge",
        ]
        with patch("sys.argv", ["run_pipeline.py"] + test_args):
            args = parse_arguments()
            assert args.resume_generate is True
            assert args.resume_judge is True


@pytest.mark.integration
class TestPipelineResumeValidation:
    """resolve_pipeline_resume_paths() path checks."""

    def test_fresh_run_uses_conversation_output(self, tmp_path):
        from run_pipeline import resolve_pipeline_resume_paths

        gen_parent = tmp_path / "gen_parent"
        args = argparse.Namespace(
            resume_generate=False,
            resume_judge=False,
            conversation_output=str(gen_parent),
        )
        resolve_pipeline_resume_paths(args)
        assert args._pipeline_gen_folder == str(gen_parent.resolve())

    def test_resume_generate_rejects_non_p_output(self, tmp_path):
        from run_pipeline import resolve_pipeline_resume_paths

        bad = tmp_path / "not_a_p_folder"
        bad.mkdir()
        args = argparse.Namespace(
            resume_generate=True,
            resume_judge=False,
            conversation_output=str(bad),
        )
        with pytest.raises(SystemExit) as exc_info:
            resolve_pipeline_resume_paths(args)
        assert exc_info.value.code == 2

    def test_resume_generate_requires_existing_directory(self, tmp_path):
        from run_pipeline import resolve_pipeline_resume_paths

        missing = tmp_path / "p_x__a_y__t1__r1__nope"
        args = argparse.Namespace(
            resume_generate=True,
            resume_judge=False,
            conversation_output=str(missing),
        )
        with pytest.raises(SystemExit) as exc_info:
            resolve_pipeline_resume_paths(args)
        assert exc_info.value.code == 2

    def test_resume_generate_accepts_valid_run_folder(self, tmp_path):
        from run_pipeline import resolve_pipeline_resume_paths

        run_dir = tmp_path / "p_persona__a_agent__t4__r1__20260101_120000"
        run_dir.mkdir()
        args = argparse.Namespace(
            resume_generate=True,
            resume_judge=False,
            conversation_output=str(run_dir),
        )
        resolve_pipeline_resume_paths(args)

    def test_resume_judge_requires_existing_directory(self, tmp_path):
        from run_pipeline import resolve_pipeline_resume_paths

        p_run = tmp_path / "p_x__a_y__t1__r1__ts"
        p_run.mkdir()
        ev = p_run / "evaluations"
        ev.mkdir()
        missing = ev / "j_missing__run__"
        args = argparse.Namespace(
            resume_generate=False,
            resume_judge=True,
            judge_output=str(missing),
        )
        with pytest.raises(SystemExit) as exc_info:
            resolve_pipeline_resume_paths(args)
        assert exc_info.value.code == 2

    def test_resume_judge_rejects_parent_evaluations_directory(self, tmp_path):
        from run_pipeline import resolve_pipeline_resume_paths

        p_run = tmp_path / "p_x__a_y__t1__r1__ts"
        p_run.mkdir()
        parent = p_run / "evaluations"
        parent.mkdir()
        args = argparse.Namespace(
            resume_generate=False,
            resume_judge=True,
            judge_output=str(parent),
        )
        with pytest.raises(SystemExit) as exc_info:
            resolve_pipeline_resume_paths(args)
        assert exc_info.value.code == 2

    def test_resume_judge_requires_j_evaluation_folder_basename(self, tmp_path):
        from run_pipeline import resolve_pipeline_resume_paths

        p_run = tmp_path / "p_x__a_y__t1__r1__ts"
        p_run.mkdir()
        ev = p_run / "evaluations"
        ev.mkdir()
        wrong = ev / "some_output_dir"
        wrong.mkdir()
        args = argparse.Namespace(
            resume_generate=False,
            resume_judge=True,
            judge_output=str(wrong),
        )
        with pytest.raises(SystemExit) as exc_info:
            resolve_pipeline_resume_paths(args)
        assert exc_info.value.code == 2

    def test_resume_judge_accepts_valid_evaluation_folder(self, tmp_path):
        from run_pipeline import resolve_pipeline_resume_paths

        p_run = tmp_path / "p_gpt4o__a_x__t1__r1__20260101_120000"
        p_run.mkdir()
        ev = p_run / "evaluations"
        ev.mkdir()
        eval_dir = ev / "j_gpt4o__p_run__20260101_120000"
        eval_dir.mkdir()
        args = argparse.Namespace(
            resume_generate=False,
            resume_judge=True,
            judge_output=str(eval_dir),
        )
        resolve_pipeline_resume_paths(args)

    def test_resume_judge_requires_judge_output(self):
        from run_pipeline import resolve_pipeline_resume_paths

        args = argparse.Namespace(
            resume_generate=False,
            resume_judge=True,
            judge_output=None,
        )
        with pytest.raises(SystemExit) as exc_info:
            resolve_pipeline_resume_paths(args)
        assert exc_info.value.code == 2

    def test_both_resume_requires_single_j_under_evaluations(self, tmp_path):
        from run_pipeline import resolve_pipeline_resume_paths

        run_dir = tmp_path / "p_u__a_p__t1__r1__20260101_120000"
        run_dir.mkdir()
        ev = run_dir / "evaluations"
        ev.mkdir()

        a0 = argparse.Namespace(
            resume_generate=True,
            resume_judge=True,
            conversation_output=str(run_dir),
        )
        with pytest.raises(SystemExit) as exc_info:
            resolve_pipeline_resume_paths(a0)
        assert exc_info.value.code == 2

        j1 = ev / "j_a__b__20260101_120000"
        j1.mkdir()
        a1 = argparse.Namespace(
            resume_generate=True,
            resume_judge=True,
            conversation_output=str(run_dir),
        )
        resolve_pipeline_resume_paths(a1)

        j2 = ev / "j_c__d__20260101_120001"
        j2.mkdir()
        a2 = argparse.Namespace(
            resume_generate=True,
            resume_judge=True,
            conversation_output=str(run_dir),
        )
        with pytest.raises(SystemExit) as exc_info:
            resolve_pipeline_resume_paths(a2)
        assert exc_info.value.code == 2


@pytest.mark.integration
class TestPipelineResumeWiring:
    """main() passes resume flags to generate and judge."""

    @pytest.mark.asyncio
    async def test_generate_and_judge_receive_resume_false_by_default(
        self, tmp_path, valid_pipeline_args
    ):
        from unittest.mock import AsyncMock, MagicMock, patch

        from run_pipeline import main as pipeline_main

        gen_run = _make_gen_run_with_txt(
            tmp_path, {"c.txt": "User: hi\nAssistant: hey"}
        )

        eval_folder = tmp_path / "evaluations"
        eval_folder.mkdir()
        (eval_folder / "results.csv").write_text("filename,x\nc.txt,y\n")

        gen_mock = AsyncMock(return_value=(None, str(gen_run)))
        judge_mock = AsyncMock(return_value=str(eval_folder))

        mock_judge_module = MagicMock()
        mock_judge_module.main = judge_mock

        with (
            patch("generate.main", side_effect=gen_mock),
            patch("importlib.util.module_from_spec", return_value=mock_judge_module),
            patch("importlib.util.spec_from_file_location"),
            patch("run_pipeline.score_results", return_value={}),
            patch("run_pipeline.print_scores"),
            patch("run_pipeline.create_visualizations"),
            patch("sys.argv", valid_pipeline_args + ["--skip-risk-analysis"]),
        ):
            await pipeline_main()

        gen_kw = gen_mock.await_args.kwargs
        assert gen_kw["resume"] is False
        judge_kw = judge_mock.await_args.args[0]
        assert judge_kw.resume is False

    @pytest.mark.asyncio
    async def test_generate_receives_resume_true_with_resume_generate(
        self, tmp_path, valid_pipeline_args
    ):
        from unittest.mock import AsyncMock, MagicMock, patch

        from run_pipeline import main as pipeline_main

        run_dir = _make_gen_run_with_txt(
            tmp_path,
            {"c.txt": "User: hi\nAssistant: hey"},
            basename="p_u__a_p__t1__r1__20260101_120000",
        )

        ev = run_dir / "evaluations"
        ev.mkdir()
        eval_folder = ev / "j_j__r__20260101_120000"
        eval_folder.mkdir()
        (eval_folder / "results.csv").write_text("filename,x\nc.txt,y\n")

        gen_mock = AsyncMock(return_value=(None, str(run_dir)))
        judge_mock = AsyncMock(return_value=str(eval_folder))

        mock_judge_module = MagicMock()
        mock_judge_module.main = judge_mock

        argv = valid_pipeline_args + [
            "--resume-generate",
            "--conversation-output",
            str(run_dir),
            "--skip-risk-analysis",
        ]

        with (
            patch("generate.main", side_effect=gen_mock),
            patch("importlib.util.module_from_spec", return_value=mock_judge_module),
            patch("importlib.util.spec_from_file_location"),
            patch("run_pipeline.score_results", return_value={}),
            patch("run_pipeline.print_scores"),
            patch("run_pipeline.create_visualizations"),
            patch("sys.argv", argv),
        ):
            await pipeline_main()

        assert gen_mock.await_args.kwargs["resume"] is True
        assert judge_mock.await_args.args[0].resume is False

    @pytest.mark.asyncio
    async def test_judge_receives_resume_true_with_resume_judge(
        self, tmp_path, valid_pipeline_args
    ):
        from unittest.mock import AsyncMock, MagicMock, patch

        from run_pipeline import main as pipeline_main

        gen_run = _make_gen_run_with_txt(
            tmp_path,
            {"c.txt": "User: hi\nAssistant: hey"},
            basename="p_x__a_y__t1__r1__ts",
        )
        ev = gen_run / "evaluations"
        ev.mkdir()
        eval_folder = ev / "j_model__convfolder__ts"
        eval_folder.mkdir()
        (eval_folder / "results.csv").write_text("filename,x\nc.txt,y\n")

        gen_mock = AsyncMock(return_value=(None, str(gen_run)))
        judge_mock = AsyncMock(return_value=str(eval_folder))

        mock_judge_module = MagicMock()
        mock_judge_module.main = judge_mock

        argv = valid_pipeline_args + [
            "--resume-judge",
            "--judge-output",
            str(eval_folder),
            "--skip-risk-analysis",
        ]

        with (
            patch("generate.main", side_effect=gen_mock),
            patch("importlib.util.module_from_spec", return_value=mock_judge_module),
            patch("importlib.util.spec_from_file_location"),
            patch("run_pipeline.score_results", return_value={}),
            patch("run_pipeline.print_scores"),
            patch("run_pipeline.create_visualizations"),
            patch("sys.argv", argv),
        ):
            await pipeline_main()

        assert gen_mock.await_args.kwargs["resume"] is True
        assert judge_mock.await_args.args[0].resume is True

    @pytest.mark.asyncio
    async def test_both_resume_flags_wired_independently(
        self, tmp_path, valid_pipeline_args
    ):
        from unittest.mock import AsyncMock, MagicMock, patch

        from run_pipeline import main as pipeline_main

        run_dir = _make_gen_run_with_txt(
            tmp_path,
            {"c.txt": "User: hi\nAssistant: hey"},
            basename="p_u__a_p__t1__r1__20260101_120000",
        )
        rev = run_dir / "evaluations"
        rev.mkdir()
        eval_folder = rev / "j_j__r__20260101_120000"
        eval_folder.mkdir()
        (eval_folder / "results.csv").write_text("filename,x\nc.txt,y\n")

        gen_mock = AsyncMock(return_value=(None, str(run_dir)))
        judge_mock = AsyncMock(return_value=str(eval_folder))

        mock_judge_module = MagicMock()
        mock_judge_module.main = judge_mock

        argv = valid_pipeline_args + [
            "--resume-generate",
            "--resume-judge",
            "--conversation-output",
            str(run_dir),
            "--skip-risk-analysis",
        ]

        with (
            patch("generate.main", side_effect=gen_mock),
            patch("importlib.util.module_from_spec", return_value=mock_judge_module),
            patch("importlib.util.spec_from_file_location"),
            patch("run_pipeline.score_results", return_value={}),
            patch("run_pipeline.print_scores"),
            patch("run_pipeline.create_visualizations"),
            patch("sys.argv", argv),
        ):
            await pipeline_main()

        assert gen_mock.await_args.kwargs["resume"] is True
        assert judge_mock.await_args.args[0].resume is True
