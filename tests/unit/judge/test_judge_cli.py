"""Unit tests for judge.py CLI and main entrypoint."""

import importlib.util
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

# Load judge.py script (project root) so we can test get_parser and main
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_JUDGE_SCRIPT = _PROJECT_ROOT / "judge.py"
_spec = importlib.util.spec_from_file_location("judge_script", _JUDGE_SCRIPT)
assert _spec is not None and _spec.loader is not None
_judge_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_judge_script)
get_parser = _judge_script.get_parser
main = _judge_script.main


@pytest.mark.unit
class TestJudgeParser:
    """Test judge.py argument parser (get_parser())."""

    def test_requires_conversation_or_folder(self):
        """Parser requires exactly one of --conversation or --folder."""
        parser = get_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["-j", "gpt-4o"])
        with pytest.raises(SystemExit):
            parser.parse_args(["-j", "gpt-4o", "-c", "c.txt", "-f", "folder"])

    def test_requires_judge_model(self):
        """Parser requires --judge-model."""
        parser = get_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["-f", "some_folder"])

    def test_folder_with_judge_model(self):
        """Folder mode: -f and -j parse correctly."""
        parser = get_parser()
        args = parser.parse_args(["-f", "conversations/run1", "-j", "gpt-4o"])
        assert args.folder == "conversations/run1"
        assert args.conversation is None
        assert args.judge_model == ["gpt-4o"]

    def test_conversation_with_judge_model(self):
        """Single conversation mode: -c and -j parse correctly."""
        parser = get_parser()
        args = parser.parse_args(["-c", "path/to/conv.txt", "-j", "claude-3-7-sonnet"])
        assert args.conversation == "path/to/conv.txt"
        assert args.folder is None
        assert args.judge_model == ["claude-3-7-sonnet"]

    def test_defaults(self):
        """Optional args have expected defaults."""
        parser = get_parser()
        args = parser.parse_args(["-f", "folder", "-j", "gpt-4o"])
        assert args.rubrics == ["data/rubric.tsv"]
        assert args.output == "evaluations"
        assert args.limit is None
        assert args.max_concurrent is None
        assert args.per_judge is False
        assert args.verbose_workers is False
        assert args.judge_model_extra_params == {}

    def test_short_flags(self):
        """Short flags -c, -f, -j, -l, -o, -m work."""
        parser = get_parser()
        args = parser.parse_args(
            ["-f", "dir", "-j", "gpt-4o", "-l", "5", "-o", "out", "-m", "3"]
        )
        assert args.folder == "dir"
        assert args.judge_model == ["gpt-4o"]
        assert args.limit == 5
        assert args.output == "out"
        assert args.max_concurrent == 3

    def test_per_judge_and_verbose_workers(self):
        """-pj and -vw set store_true flags."""
        parser = get_parser()
        args = parser.parse_args(["-f", "dir", "-j", "gpt-4o", "-pj", "-vw"])
        assert args.per_judge is True
        assert args.verbose_workers is True

    def test_judge_model_extra_params_parsed(self):
        """--judge-model-extra-params uses parse_key_value_list."""
        parser = get_parser()
        args = parser.parse_args(
            [
                "-f",
                "dir",
                "-j",
                "gpt-4o",
                "--judge-model-extra-params",
                "temperature=0.7,max_tokens=1000",
            ]
        )
        assert args.judge_model_extra_params == {
            "temperature": 0.7,
            "max_tokens": 1000,
        }

    def test_judge_model_nargs_plus(self):
        """--judge-model accepts multiple values (nargs='+')."""
        parser = get_parser()
        args = parser.parse_args(
            [
                "-f",
                "dir",
                "-j",
                "gpt-4o",
                "claude-sonnet-4-5-20250929:2",
            ]
        )
        assert args.judge_model == ["gpt-4o", "claude-sonnet-4-5-20250929:2"]


@pytest.mark.unit
class TestJudgeMain:
    """Test main() entrypoint with mocks (single vs folder path and arg forwarding)."""

    @pytest.mark.asyncio
    async def test_main_single_conversation_calls_judge_single(self):
        """main() with args.conversation calls judge_single_conversation."""
        parser = get_parser()
        args = parser.parse_args(
            [
                "-c",
                "conv.txt",
                "-j",
                "gpt-4o",
            ]
        )
        with (
            patch.object(_judge_script, "RubricConfig") as RubricConfig,
            patch.object(_judge_script, "ConversationData") as ConversationData,
            patch.object(_judge_script, "LLMJudge") as LLMJudge,
            patch.object(
                _judge_script,
                "judge_single_conversation",
                new_callable=AsyncMock,
            ) as judge_single,
        ):
            RubricConfig.load = AsyncMock(return_value="rubric_config")
            ConversationData.load = AsyncMock(return_value="conversation_data")
            LLMJudge.return_value = "judge_instance"

            result = await main(args)

            RubricConfig.load.assert_called_once_with(rubric_folder="data")
            ConversationData.load.assert_called_once_with("conv.txt")
            LLMJudge.assert_called_once_with(
                judge_model="gpt-4o",
                rubric_config="rubric_config",
                judge_model_extra_params={},
            )
            judge_single.assert_awaited_once_with(
                "judge_instance", "conversation_data", "evaluations"
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_main_folder_calls_judge_conversations(self):
        """main() with args.folder calls load_conversations and judge_conversations."""
        parser = get_parser()
        args = parser.parse_args(
            [
                "-f",
                "conversations/run1",
                "-j",
                "gpt-4o:2",
                "-l",
                "10",
                "-o",
                "eval_out",
                "-m",
                "4",
                "-pj",
                "-vw",
            ]
        )
        with (
            patch.object(_judge_script, "RubricConfig") as RubricConfig,
            patch.object(
                _judge_script,
                "load_conversations",
                new_callable=AsyncMock,
            ) as load_convos,
            patch.object(
                _judge_script,
                "judge_conversations",
                new_callable=AsyncMock,
            ) as judge_convos,
        ):
            RubricConfig.load = AsyncMock(return_value="rubric_config")
            load_convos.return_value = []
            judge_convos.return_value = ([], "evaluations/run1_timestamp")

            result = await main(args)

            RubricConfig.load.assert_called_once_with(rubric_folder="data")
            load_convos.assert_called_once_with("conversations/run1", limit=10)
            judge_convos.assert_awaited_once()
            assert judge_convos.await_args is not None
            call_kw = judge_convos.await_args[1]
            assert call_kw["judge_models"] == {"gpt-4o": 2}
            assert call_kw["rubric_config"] == "rubric_config"
            assert call_kw["max_concurrent"] == 4
            assert call_kw["output_root"] == "eval_out"
            assert call_kw["conversation_folder_name"] == "run1"
            assert call_kw["verbose"] is True
            assert call_kw["judge_model_extra_params"] == {}
            assert call_kw["per_judge"] is True
            assert call_kw["verbose_workers"] is True
            assert result == "evaluations/run1_timestamp"
