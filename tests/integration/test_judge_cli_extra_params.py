"""Integration tests for judge CLI extra parameters."""

import argparse

import pytest


@pytest.mark.integration
class TestJudgeCLIExtraParams:
    """Test CLI argument parsing for judge model extra parameters."""

    def test_cli_accepts_judge_model_extra_params(self):
        """Test that CLI accepts --judge-model-extra-params argument."""
        # Import the parser setup from judge.py
        parser = argparse.ArgumentParser(
            description="Judge existing LLM conversations using rubrics"
        )

        # Add the same arguments as judge.py
        from utils.utils import parse_key_value_list

        source_group = parser.add_mutually_exclusive_group(required=False)
        source_group.add_argument(
            "--conversation", "-c", help="Path to a single conversation file to judge"
        )
        source_group.add_argument(
            "--folder",
            "-f",
            default="conversations",
            help="Folder containing conversation files",
        )

        parser.add_argument(
            "--judge-model",
            "-j",
            help="Model to use for judging",
            default="claude-3-7-sonnet",
        )

        parser.add_argument(
            "--judge-model-extra-params",
            "-jep",
            help="Extra parameters for the judge model",
            type=parse_key_value_list,
            default={},
        )

        # Test parsing with extra params
        args = parser.parse_args(
            [
                "--folder",
                "test_folder",
                "--judge-model",
                "claude-3-7-sonnet",
                "--judge-model-extra-params",
                "temperature=0.7,max_tokens=1000",
            ]
        )

        assert args.judge_model == "claude-3-7-sonnet"
        assert args.judge_model_extra_params == {"temperature": 0.7, "max_tokens": 1000}

    def test_cli_judge_model_extra_params_short_flag(self):
        """Test that -jep shorthand works for judge-model-extra-params."""
        from utils.utils import parse_key_value_list

        parser = argparse.ArgumentParser()
        parser.add_argument("--folder", "-f", default="conversations")
        parser.add_argument("--judge-model", "-j", default="claude-3-7-sonnet")
        parser.add_argument(
            "--judge-model-extra-params",
            "-jep",
            type=parse_key_value_list,
            default={},
        )

        # Test with short flag
        args = parser.parse_args(
            ["-f", "test_folder", "-j", "gpt-4o", "-jep", "temperature=0.5"]
        )

        assert args.judge_model == "gpt-4o"
        assert args.judge_model_extra_params == {"temperature": 0.5}

    def test_cli_judge_model_extra_params_defaults_to_empty_dict(self):
        """Test that judge-model-extra-params defaults to empty dict."""
        from utils.utils import parse_key_value_list

        parser = argparse.ArgumentParser()
        parser.add_argument("--folder", "-f", default="conversations")
        parser.add_argument("--judge-model", "-j", default="claude-3-7-sonnet")
        parser.add_argument(
            "--judge-model-extra-params",
            "-jep",
            type=parse_key_value_list,
            default={},
        )

        # Test without providing extra params
        args = parser.parse_args(["-f", "test_folder", "-j", "claude-3-7-sonnet"])

        assert args.judge_model_extra_params == {}

    def test_cli_judge_model_extra_params_multiple_values(self):
        """Test parsing multiple extra parameters."""
        from utils.utils import parse_key_value_list

        parser = argparse.ArgumentParser()
        parser.add_argument("--folder", "-f", default="conversations")
        parser.add_argument("--judge-model", "-j", default="claude-3-7-sonnet")
        parser.add_argument(
            "--judge-model-extra-params",
            "-jep",
            type=parse_key_value_list,
            default={},
        )

        args = parser.parse_args(
            [
                "-f",
                "test_folder",
                "-j",
                "claude-3-7-sonnet",
                "-jep",
                "temperature=0.8,max_tokens=2000,top_p=0.95",
            ]
        )

        assert args.judge_model_extra_params == {
            "temperature": 0.8,
            "max_tokens": 2000,
            "top_p": 0.95,
        }

    def test_cli_help_includes_judge_model_extra_params(self, capsys):
        """Test that --help output includes judge-model-extra-params."""
        from utils.utils import parse_key_value_list

        parser = argparse.ArgumentParser(prog="judge")
        parser.add_argument("--folder", "-f", default="conversations")
        parser.add_argument("--judge-model", "-j", default="claude-3-7-sonnet")
        parser.add_argument(
            "--judge-model-extra-params",
            "-jep",
            help=(
                "Extra parameters for the judge model. "
                "Examples: temperature=0.7, max_tokens=1000"
            ),
            type=parse_key_value_list,
            default={},
        )

        # Capture help output
        with pytest.raises(SystemExit):
            parser.parse_args(["--help"])

        captured = capsys.readouterr()
        help_text = captured.out

        # Verify help text includes the argument
        assert "--judge-model-extra-params" in help_text or "-jep" in help_text
        assert "Extra parameters" in help_text

    def test_cli_judge_model_extra_params_with_string_values(self):
        """Test that string values in extra params work correctly."""
        from utils.utils import parse_key_value_list

        parser = argparse.ArgumentParser()
        parser.add_argument("--folder", "-f", default="conversations")
        parser.add_argument("--judge-model", "-j", default="claude-3-7-sonnet")
        parser.add_argument(
            "--judge-model-extra-params",
            "-jep",
            type=parse_key_value_list,
            default={},
        )

        args = parser.parse_args(
            [
                "-f",
                "test_folder",
                "-j",
                "claude-3-7-sonnet",
                "-jep",
                "temperature=0.7,model_type=chat",
            ]
        )

        assert args.judge_model_extra_params["temperature"] == 0.7
        assert args.judge_model_extra_params["model_type"] == "chat"

    def test_cli_pattern_matches_generate_py(self):
        """Test that judge.py extra params follow same pattern as generate.py."""
        from utils.utils import parse_key_value_list

        # Test generate.py pattern
        generate_parser = argparse.ArgumentParser()
        generate_parser.add_argument("--user-agent", "-u", required=False)
        generate_parser.add_argument(
            "--user-agent-extra-params",
            "-uep",
            type=parse_key_value_list,
            default={},
        )

        generate_args = generate_parser.parse_args(
            ["-u", "claude-3-7-sonnet", "-uep", "temperature=0.7"]
        )

        # Test judge.py pattern
        judge_parser = argparse.ArgumentParser()
        judge_parser.add_argument("--judge-model", "-j", required=False)
        judge_parser.add_argument(
            "--judge-model-extra-params",
            "-jep",
            type=parse_key_value_list,
            default={},
        )

        judge_args = judge_parser.parse_args(
            ["-j", "claude-3-7-sonnet", "-jep", "temperature=0.7"]
        )

        # Both should produce the same result structure
        assert generate_args.user_agent_extra_params == {
            "temperature": 0.7
        }, "generate.py pattern"
        assert judge_args.judge_model_extra_params == {
            "temperature": 0.7
        }, "judge.py pattern"
        assert (
            generate_args.user_agent_extra_params == judge_args.judge_model_extra_params
        )
