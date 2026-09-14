"""Unit tests for judge.py CLI and main entrypoint."""

import importlib.util
from pathlib import Path
from unittest.mock import ANY, AsyncMock, patch

import pytest

import judge.run as _judge_run
from utils.conversation_layout import resolve_conversation_input

# Load judge.py script (project root) so we can test get_parser and main
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_JUDGE_SCRIPT = _PROJECT_ROOT / "judge.py"
_spec = importlib.util.spec_from_file_location("judge_script", _JUDGE_SCRIPT)
assert _spec is not None and _spec.loader is not None
_judge_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_judge_script)
get_parser = _judge_script.get_parser
main = _judge_script.main

# main() resolves whichever manifest --rubrics names into three concrete paths.
# Tests below drive that with a test manifest rather than the shipped
# data/SI/rubric_manifest.json, so renaming a real rubric file can't break them;
# that the default *is* data/SI/... is asserted once, in test_defaults.
FIXTURE_MANIFEST = "tests/fixtures/rubric_manifest_simple.json"
FIXTURE_RUBRIC_PATHS = {
    "rubric_file": "tests/fixtures/rubric_simple.tsv",
    "rubric_prompt_beginning_file": "tests/fixtures/rubric_prompt_beginning.txt",
    "question_prompt_file": "tests/fixtures/question_prompt.txt",
}


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
        args = parser.parse_args(["-c", "path/to/conv.txt", "-j", "claude-sonnet-4-5"])
        assert args.conversation == "path/to/conv.txt"
        assert args.folder is None
        assert args.judge_model == ["claude-sonnet-4-5"]

    def test_defaults(self):
        """Optional args have expected defaults."""
        parser = get_parser()
        args = parser.parse_args(["-f", "folder", "-j", "gpt-4o"])
        assert args.rubrics == ["data/SI/rubric_manifest.json"]
        assert args.output is None
        assert args.limit is None
        assert args.max_concurrent is None
        assert args.per_judge is False
        assert args.verbose_workers is False
        assert args.resume is False
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
                "-r",
                FIXTURE_MANIFEST,
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
            RubricConfig.from_paths = AsyncMock(return_value="rubric_config")
            ConversationData.load = AsyncMock(return_value="conversation_data")
            LLMJudge.return_value = "judge_instance"

            result = await main(args)

            # main() resolves the manifest to concrete paths, then the rubric is
            # built from those paths rather than from the manifest itself.
            RubricConfig.from_paths.assert_called_once_with(**FIXTURE_RUBRIC_PATHS)
            ConversationData.load.assert_called_once_with("conv.txt")
            LLMJudge.assert_called_once_with(
                judge_model="gpt-4o",
                rubric_config="rubric_config",
                judge_model_extra_params={},
                log_file=ANY,
            )
            judge_single.assert_awaited_once()
            ja = judge_single.await_args[0]
            assert ja[0] == "judge_instance"
            assert ja[1] == "conversation_data"
            out_run = Path(ja[2])
            assert out_run.name.startswith("single_")
            assert out_run.name.endswith("__conv")
            assert result == ja[2]

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
                "-r",
                FIXTURE_MANIFEST,
            ]
        )
        # Loading and dispatch now happen inside `judge.run.run_judging`, so the
        # seams are patched there; main()'s job is to resolve and delegate.
        with (
            patch.object(_judge_run, "RubricConfig") as RubricConfig,
            patch.object(
                _judge_run,
                "load_conversations",
                new_callable=AsyncMock,
            ) as load_convos,
            patch.object(
                _judge_run,
                "judge_conversations",
                new_callable=AsyncMock,
            ) as judge_convos,
        ):
            RubricConfig.from_paths = AsyncMock(return_value="rubric_config")
            load_convos.return_value = []
            judge_convos.return_value = ([], "evaluations/run1_timestamp")

            result = await main(args)

            RubricConfig.from_paths.assert_called_once_with(**FIXTURE_RUBRIC_PATHS)
            expected_dir, _, _ = resolve_conversation_input("conversations/run1")
            load_convos.assert_called_once_with(expected_dir, limit=10)
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
            assert call_kw["resume"] is False
            assert result == "evaluations/run1_timestamp"

    @pytest.mark.asyncio
    async def test_main_folder_resume_uses_output_folder(self, tmp_path: Path):
        """main() with --resume passes output_folder instead of output_root."""
        eval_folder = tmp_path / "j_eval__run1"
        eval_folder.mkdir(parents=True, exist_ok=True)

        parser = get_parser()
        args = parser.parse_args(
            [
                "-f",
                "conversations/run1",
                "-j",
                "gpt-4o:2",
                "-o",
                str(eval_folder),
                "-r",
                FIXTURE_MANIFEST,
                "--resume",
            ]
        )
        with (
            patch.object(_judge_run, "RubricConfig") as RubricConfig,
            patch.object(
                _judge_run, "load_conversations", new_callable=AsyncMock
            ) as load_convos,
            patch.object(
                _judge_run, "judge_conversations", new_callable=AsyncMock
            ) as judge_convos,
        ):
            RubricConfig.from_paths = AsyncMock(return_value="rubric_config")
            load_convos.return_value = []
            judge_convos.return_value = ([], str(eval_folder))

            result = await main(args)

            call_kw = judge_convos.await_args[1]
            assert "output_root" not in call_kw
            assert call_kw["output_folder"] == str(eval_folder)
            assert call_kw["resume"] is True
            assert result == str(eval_folder)

    @pytest.mark.asyncio
    async def test_main_loads_distinct_rubric_bundles_end_to_end(self):
        """--rubrics actually drives which bundle is loaded, not a hardcoded one.

        Uses the real RubricConfig.load_bundle() against two different test
        fixture manifests and checks the parsed rubrics differ, proving the
        flag is live rather than a no-op.
        """
        parser = get_parser()

        async def load_rubric_config(rubrics_arg):
            args = parser.parse_args(
                ["-f", "conversations/run1", "-j", "gpt-4o", "-r", rubrics_arg]
            )
            with (
                patch.object(
                    _judge_run, "load_conversations", new_callable=AsyncMock
                ) as load_convos,
                patch.object(
                    _judge_run, "judge_conversations", new_callable=AsyncMock
                ) as judge_convos,
                patch.object(_judge_run, "RubricConfig") as RubricConfigMock,
            ):
                from judge.rubric_config import RubricConfig as RealRubricConfig

                # Real parsing, so the two fixtures must produce different
                # rubrics -- proving --rubrics is live, not a no-op.
                RubricConfigMock.from_paths = RealRubricConfig.from_paths
                load_convos.return_value = []
                judge_convos.return_value = ([], "evaluations/run1_timestamp")

                await main(args)
                return judge_convos.await_args[1]["rubric_config"]

        simple = await load_rubric_config("tests/fixtures/rubric_manifest_simple.json")
        multi_row = await load_rubric_config(
            "tests/fixtures/rubric_manifest_multi_row.json"
        )

        assert simple.question_order != multi_row.question_order

    @pytest.mark.asyncio
    async def test_main_warns_on_multiple_rubrics(self, capsys):
        """Passing multiple --rubrics values warns and uses only the first."""
        parser = get_parser()
        args = parser.parse_args(
            [
                "-f",
                "conversations/run1",
                "-j",
                "gpt-4o",
                "-r",
                "tests/fixtures/rubric_manifest_simple.json",
                "tests/fixtures/rubric_manifest_multi_row.json",
            ]
        )
        with (
            patch.object(_judge_run, "RubricConfig") as RubricConfig,
            patch.object(
                _judge_run, "load_conversations", new_callable=AsyncMock
            ) as load_convos,
            patch.object(
                _judge_run, "judge_conversations", new_callable=AsyncMock
            ) as judge_convos,
        ):
            RubricConfig.from_paths = AsyncMock(return_value="rubric_config")
            load_convos.return_value = []
            judge_convos.return_value = ([], "evaluations/run1_timestamp")

            await main(args)

            # Only the first manifest's files are resolved and loaded.
            RubricConfig.from_paths.assert_called_once_with(
                rubric_file="tests/fixtures/rubric_simple.tsv",
                rubric_prompt_beginning_file=(
                    "tests/fixtures/rubric_prompt_beginning.txt"
                ),
                question_prompt_file="tests/fixtures/question_prompt.txt",
            )
            captured = capsys.readouterr()
            assert "Warning" in captured.err
            assert "rubric_manifest_simple.json" in captured.err
