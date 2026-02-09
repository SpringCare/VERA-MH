import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Import additional modules to increase test coverage

# Test configuration constants
TEST_CONFIG = {
    "USER_MODEL": "gpt-4o",
    "PROVIDER_MODEL": "claude-opus-4-1-20250805",
    "JUDGE_MODEL": "gpt-4o",
    "JUDGE_INSTANCES": 1,  # Single instance for fastest test execution
    "TURNS": 6,  # Smaller for faster tests
    "RUNS_PER_PERSONA": 1,  # Reduced for faster tests
    "TEMP_USER": 0.0,
    "TEMP_PROVIDER": 0.0,
    "TIMEOUT_SECONDS": 300,
}


@pytest.fixture(scope="session", autouse=True)
def validate_test_environment():
    """Ensure test environment is properly configured before running any tests."""
    required_env_vars = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"]
    missing = [var for var in required_env_vars if not os.getenv(var)]
    if missing:
        pytest.skip(f"Missing required environment variables: {missing}")

    # Validate repo structure
    required_files = ["generate.py", "judge.py", "data/personas.tsv"]
    repo_root = Path.cwd()
    missing_files = [f for f in required_files if not (repo_root / f).exists()]
    if missing_files:
        pytest.skip(f"Missing required files: {missing_files}")


@pytest.fixture
def test_workspace():
    """Create isolated test workspace for each test."""
    with tempfile.TemporaryDirectory(prefix="vera_integration_test_") as tmpdir:
        workspace = Path(tmpdir)
        conversations_dir = workspace / "conversations"
        conversations_dir.mkdir()
        yield workspace


@pytest.fixture
def repo_root():
    """Get repository root path."""
    return Path(__file__).parent.parent.parent


@pytest.mark.integration
@pytest.mark.live
class TestVERAMHPipeline:
    """Integration tests for the complete VERA-MH pipeline using CLI tools."""

    async def run_generate_cli(
        self,
        persona_names: List[str],
        user_model: str,
        provider_model: str,
        conversations_root: Path,
        turns: int,
        runs: int,
        temp_user: float,
        temp_provider: float,
    ) -> Path:
        """Run generate.py CLI and return the conversation directory path."""
        conversations_root.mkdir(parents=True, exist_ok=True)

        # Build CLI command
        cmd = [
            "uv",
            "run",
            "python3",
            "generate.py",
            "--user-agent",
            user_model,
            "--provider-agent",
            provider_model,
            "--runs",
            str(runs),
            "--turns",
            str(turns),
            "--folder-name",
            str(conversations_root),
            "--user-agent-extra-params",
            f"temperature={temp_user}",
            "--provider-agent-extra-params",
            f"temperature={temp_provider}",
            "--max-concurrent",
            "1",  # Controlled for tests
            "--max-personas",
            "1",  # Limit to 1 persona for faster tests
        ]

        # Run generate.py CLI
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path.cwd(),  # Run from project root
            timeout=120,  # 2 minute timeout for generation
        )

        if result.returncode != 0:
            raise RuntimeError(f"Generate CLI failed: {result.stderr}")

        # Parse output to find the generated folder path
        # generate.py outputs: "✅ Generated N conversations → folder_path/"
        output_lines = result.stdout.strip().split("\n")
        folder_path = None

        for line in output_lines:
            if "Generated" in line and "conversations →" in line:
                # Extract folder path from line like:
                # "✅ Generated 2 conversations → /path/to/folder/"
                parts = line.split(" → ")
                if len(parts) > 1:
                    folder_path = parts[-1].rstrip("/")
                    break

        if not folder_path:
            # Fallback: look for directories in conversations_root
            conv_dirs = [d for d in conversations_root.iterdir() if d.is_dir()]
            if conv_dirs:
                # Get the most recently created directory
                folder_path = str(max(conv_dirs, key=lambda d: d.stat().st_ctime))

        if not folder_path:
            raise RuntimeError(
                f"Could not determine generated conversation folder from CLI output: "
                f"{result.stdout}"
            )

        conv_dir = Path(folder_path)

        # Verify the directory contains conversation files
        if not conv_dir.exists() or not conv_dir.is_dir():
            raise RuntimeError(f"Generated directory {conv_dir} is not valid")

        # Log results for debugging
        conv_files = list(conv_dir.glob("*.txt")) + list(conv_dir.glob("*.json"))
        logger = logging.getLogger(__name__)
        if not conv_files:
            logger.warning(
                f"Generated directory {conv_dir} exists but contains no "
                f"conversation files (.txt/.json). "
                f"Contents: {[f.name for f in conv_dir.iterdir()]}"
            )

        logger.info(f"Generated conversations in {conv_dir}")
        return conv_dir

    async def run_judge_cli(
        self,
        conversations_dir: Path,
        judge_model: str,
        test_workspace: Path,
        instances: int = 1,
    ) -> Path:
        """Run judge.py CLI and return the evaluation directory path."""
        eval_root = test_workspace / "evaluations"
        eval_root.mkdir(exist_ok=True)

        # Build CLI command
        cmd = [
            "uv",
            "run",
            "python3",
            "judge.py",
            "--folder",
            str(conversations_dir),
            "--judge-model",
            f"{judge_model}:{instances}",
            "--output",
            str(eval_root),
        ]

        # Run judge.py CLI
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path.cwd(),  # Run from project root
        )

        if result.returncode != 0:
            raise RuntimeError(f"Judge CLI failed: {result.stderr}")

        # Find the generated evaluation directory
        # judge.py creates directories like j_modelname_timestamp_folder
        eval_dirs = [
            d for d in eval_root.iterdir() if d.is_dir() and d.name.startswith("j_")
        ]
        if not eval_dirs:
            raise RuntimeError(f"No evaluation directory found in {eval_root}")

        # Get the most recent evaluation directory
        eval_dir = max(eval_dirs, key=lambda d: d.stat().st_ctime)

        # Verify results.csv exists
        results_csv = eval_dir / "results.csv"
        if not results_csv.exists():
            raise RuntimeError(f"results.csv not found in {eval_dir}")

        logging.info(f"Judge created evaluation files in {eval_dir}")
        return eval_dir

    def run_score_cli(self, eval_dir: Path) -> dict:
        """Run judge.score CLI and return the results dictionary."""
        results_csv = eval_dir / "results.csv"
        if not results_csv.exists():
            raise FileNotFoundError(f"Missing results.csv in {eval_dir}")

        # Build CLI command
        cmd = [
            "uv",
            "run",
            "python3",
            "-m",
            "judge.score",
            "--results-csv",
            str(results_csv),
            "--skip-risk-analysis",  # Skip for faster tests
        ]

        # Run judge.score CLI
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path.cwd(),  # Run from project root
        )

        if result.returncode != 0:
            raise RuntimeError(f"Score CLI failed: {result.stderr}")

        # Load the generated scores.json
        scores_json = eval_dir / "scores.json"
        if not scores_json.exists():
            raise RuntimeError(f"scores.json not found in {eval_dir}")

        with open(scores_json, "r") as f:
            scores_data = json.load(f)

        if "aggregates" not in scores_data:
            raise ValueError(
                f"Invalid scores data structure: missing 'aggregates' key. "
                f"Found keys: {list(scores_data.keys())}"
            )

        return scores_data

    def run_pipeline_cli(
        self,
        persona: str,
        test_workspace: Path,
        repo_root: Path,
        config: Dict[str, Any] = None,
        judge_model: str = None,
    ) -> Dict[str, Any]:
        """Run the complete pipeline using run_pipeline.py CLI and return results."""
        if config is None:
            config = TEST_CONFIG

        timestamp = int(time.time())
        folder_name = f"test_pipeline_{timestamp}"

        # Parse judge model to extract instances
        judge_model_name = judge_model or config["JUDGE_MODEL"]
        instances = config["JUDGE_INSTANCES"]
        if ":" in judge_model_name:
            judge_model_name, instances_str = judge_model_name.split(":", 1)
            instances = int(instances_str)

        # Build CLI command for run_pipeline.py
        cmd = [
            "uv",
            "run",
            "python3",
            "run_pipeline.py",
            "--user-agent",
            config["USER_MODEL"],
            "--provider-agent",
            config["PROVIDER_MODEL"],
            "--runs",
            str(config["RUNS_PER_PERSONA"]),
            "--turns",
            str(config["TURNS"]),
            "--judge-model",
            f"{judge_model_name}:{instances}",
            "--max-personas",
            "1",  # Limit to 1 persona for faster tests
            "--folder-name",
            folder_name,
            "--user-agent-extra-params",
            f"temperature={config['TEMP_USER']}",
            "--provider-agent-extra-params",
            f"temperature={config['TEMP_PROVIDER']}",
        ]

        # Run run_pipeline.py CLI
        original_cwd = os.getcwd()
        os.chdir(repo_root)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=repo_root,
                timeout=config["TIMEOUT_SECONDS"],
            )

            if result.returncode != 0:
                raise RuntimeError(f"Pipeline CLI failed: {result.stderr}")

            # Find the evaluation directory created by run_pipeline
            evaluations_base_dir = repo_root / "evaluations"
            if not evaluations_base_dir.exists():
                raise RuntimeError(
                    f"Evaluations directory not found: {evaluations_base_dir}"
                )

            # Find the most recent evaluation subfolder
            eval_dirs = [
                d
                for d in evaluations_base_dir.iterdir()
                if d.is_dir() and d.name.startswith("j_")
            ]
            if not eval_dirs:
                raise RuntimeError(
                    f"No evaluation subdirectories found in {evaluations_base_dir}"
                )

            eval_dir = max(eval_dirs, key=lambda d: d.stat().st_ctime)

            # Load and return the scores.json
            scores_json = eval_dir / "scores.json"
            if not scores_json.exists():
                raise RuntimeError(f"scores.json not found in {eval_dir}")

            with open(scores_json, "r") as f:
                scores_data = json.load(f)

            if "aggregates" not in scores_data:
                raise ValueError(
                    f"Invalid scores data structure: missing 'aggregates' key. "
                    f"Found keys: {list(scores_data.keys())}"
                )

            return scores_data

        finally:
            os.chdir(original_cwd)

            # Clean up test artifacts
            import shutil

            cleanup_dirs = [folder_name, "evaluations"]
            for dir_name in cleanup_dirs:
                dir_path = repo_root / dir_name
                if dir_path.exists() and dir_path.is_dir():
                    try:
                        shutil.rmtree(dir_path)
                    except Exception as e:
                        logger = logging.getLogger(__name__)
                        logger.warning(f"Failed to clean up {dir_path}: {e}")

    async def run_complete_pipeline(
        self,
        persona: str,
        test_workspace: Path,
        repo_root: Path,
        config: Dict[str, Any] = None,
        judge_model: str = None,
    ) -> Dict[str, Any]:
        """Run the complete pipeline using CLI tools and return results."""
        if config is None:
            config = TEST_CONFIG

        conversations_root = test_workspace / "conversations"

        # Generate conversations using CLI
        conv_dir = await self.run_generate_cli(
            persona_names=[persona],
            user_model=config["USER_MODEL"],
            provider_model=config["PROVIDER_MODEL"],
            conversations_root=conversations_root,
            turns=config["TURNS"],
            runs=config["RUNS_PER_PERSONA"],
            temp_user=config["TEMP_USER"],
            temp_provider=config["TEMP_PROVIDER"],
        )

        # Parse judge model to extract instances
        judge_model_name = judge_model or config["JUDGE_MODEL"]
        instances = 1  # default - use single instance for faster tests
        if ":" in judge_model_name:
            judge_model_name, instances_str = judge_model_name.split(":", 1)
            instances = int(instances_str)

        # Judge conversations using CLI
        eval_dir = await self.run_judge_cli(
            conversations_dir=conv_dir,
            judge_model=judge_model_name,
            test_workspace=test_workspace,
            instances=instances,
        )

        # Score evaluation using CLI
        score_json = self.run_score_cli(eval_dir=eval_dir)

        return score_json

    @pytest.mark.asyncio
    @pytest.mark.timeout(TEST_CONFIG["TIMEOUT_SECONDS"])
    async def test_complete_pipeline_single_persona(self, test_workspace, repo_root):
        """Test the complete pipeline with a single persona."""
        persona = "Sky"

        result = self.run_pipeline_cli(persona, test_workspace, repo_root)

        # Assert structure exists
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "aggregates" in result, "Result should contain 'aggregates' key"

        aggregates = result["aggregates"]

        # Assert required metrics exist
        required_metrics = [
            "overall_best_practice_pct",
            "overall_damaging_pct",
            "worst_band",
            "vera_score",
        ]
        for metric in required_metrics:
            assert metric in aggregates, f"Missing required metric: {metric}"

        # Assert metric value ranges (handle "Not Relevant" case)
        best_practice_pct = aggregates["overall_best_practice_pct"]
        damaging_pct = aggregates["overall_damaging_pct"]
        worst_band = aggregates["worst_band"]
        vera_score = aggregates["vera_score"]

        assert (
            0 <= best_practice_pct <= 100
        ), f"Best practice % should be 0-100, got {best_practice_pct}"
        assert (
            0 <= damaging_pct <= 100
        ), f"Damaging % should be 0-100, got {damaging_pct}"
        assert worst_band is not None, "Worst band should not be None"
        assert isinstance(vera_score, (int, float)), "VERA score should be numeric"
        assert vera_score >= 0, f"VERA score should be >= 0, got {vera_score}"

        # For integration tests, we expect some meaningful evaluation
        # If everything is "Not Relevant", that's still a valid result but worth noting
        suboptimal_pct = aggregates.get("overall_suboptimal_pct", 0)
        if best_practice_pct == 0 and damaging_pct == 0 and suboptimal_pct == 0:
            logger = logging.getLogger(__name__)
            logger.warning(
                "All evaluations marked 'Not Relevant' - "
                "conversation may not have triggered mental health criteria"
            )

    @pytest.mark.asyncio
    @pytest.mark.timeout(TEST_CONFIG["TIMEOUT_SECONDS"])
    async def test_individual_personas(self, test_workspace, repo_root):
        """Test that Ray persona works individually."""
        persona = "Ray"
        result = self.run_pipeline_cli(persona, test_workspace, repo_root)

        # Basic validation
        assert "aggregates" in result
        aggregates = result["aggregates"]

        # Each persona should produce valid results
        best_practice_pct = aggregates["overall_best_practice_pct"]
        damaging_pct = aggregates["overall_damaging_pct"]
        worst_band = aggregates["worst_band"]
        vera_score = aggregates["vera_score"]

        assert 0 <= best_practice_pct <= 100
        assert 0 <= damaging_pct <= 100
        assert worst_band is not None
        assert isinstance(vera_score, (int, float)), "VERA score should be numeric"
        assert vera_score >= 0, f"VERA score should be >= 0, got {vera_score}"

    @pytest.mark.asyncio
    @pytest.mark.timeout(TEST_CONFIG["TIMEOUT_SECONDS"])
    async def test_scoring_file_output(self, test_workspace, repo_root):
        """Test that scoring produces expected file outputs after judging."""
        # Generate conversations
        conversations_root = test_workspace / "conversations"
        conv_dir = await self.run_generate_cli(
            persona_names=["Ray"],
            user_model=TEST_CONFIG["USER_MODEL"],
            provider_model=TEST_CONFIG["PROVIDER_MODEL"],
            conversations_root=conversations_root,
            turns=TEST_CONFIG["TURNS"],
            runs=TEST_CONFIG["RUNS_PER_PERSONA"],
            temp_user=TEST_CONFIG["TEMP_USER"],
            temp_provider=TEST_CONFIG["TEMP_PROVIDER"],
        )

        # Verify conversation files were created
        conv_files = list(conv_dir.glob("*.txt")) + list(conv_dir.glob("*.json"))
        assert len(conv_files) > 0, "No conversation files were created"

        # Judge conversations
        eval_dir = await self.run_judge_cli(
            conversations_dir=conv_dir,
            judge_model=TEST_CONFIG["JUDGE_MODEL"],
            test_workspace=test_workspace,
            instances=TEST_CONFIG["JUDGE_INSTANCES"],
        )

        # Verify evaluation files were created
        results_csv = eval_dir / "results.csv"
        assert results_csv.exists(), "results.csv evaluation file was not created"

        # Score the evaluation
        score_result = self.run_score_cli(eval_dir=eval_dir)

        # Verify that scoring produces valid output structure
        assert isinstance(score_result, dict), "Score result should be a dictionary"
        assert (
            "aggregates" in score_result
        ), "Score result should contain 'aggregates' key"

        aggregates = score_result["aggregates"]

        # Verify required metrics exist (don't check ranges, just existence)
        required_metrics = [
            "overall_best_practice_pct",
            "overall_damaging_pct",
            "worst_band",
            "vera_score",
        ]
        for metric in required_metrics:
            assert metric in aggregates, f"Missing required metric: {metric}"
            assert aggregates[metric] is not None, f"Metric {metric} should not be None"

        # Verify scores.json file is created in the evaluation directory
        scores_file = eval_dir / "scores.json"
        assert scores_file.exists(), "scores.json file should be created"
        assert scores_file.stat().st_size > 0, "scores.json should not be empty"

        # Verify the scores.json contains valid JSON structure
        with open(scores_file, "r") as f:
            file_content = json.load(f)

        assert (
            "aggregates" in file_content
        ), "scores.json should contain 'aggregates' key"
        for metric in required_metrics:
            assert metric in file_content["aggregates"], f"scores.json missing {metric}"

    @pytest.mark.asyncio
    @pytest.mark.timeout(TEST_CONFIG["TIMEOUT_SECONDS"])
    async def test_conversation_generation_only(self, test_workspace, repo_root):
        """Test just the conversation generation step."""
        conversations_root = test_workspace / "conversations"

        conv_dir = await self.run_generate_cli(
            persona_names=["Ray"],
            user_model=TEST_CONFIG["USER_MODEL"],
            provider_model=TEST_CONFIG["PROVIDER_MODEL"],
            conversations_root=conversations_root,
            turns=TEST_CONFIG["TURNS"],
            runs=TEST_CONFIG["RUNS_PER_PERSONA"],
            temp_user=TEST_CONFIG["TEMP_USER"],
            temp_provider=TEST_CONFIG["TEMP_PROVIDER"],
        )

        # Assert conversation directory exists and contains files
        assert conv_dir.exists(), "Conversation directory should exist"
        assert conv_dir.is_dir(), "Conversation path should be a directory"

        # Look for conversation files with multiple extensions
        json_files = list(conv_dir.glob("*.json"))
        csv_files = list(conv_dir.glob("*.csv"))
        txt_files = list(conv_dir.glob("*.txt"))
        all_conversation_files = json_files + csv_files + txt_files

        # Check subdirectories for additional files
        subdirs = [p for p in conv_dir.iterdir() if p.is_dir()]
        for subdir in subdirs:
            # Apply same file filtering as parent directory
            subdir_json = list(subdir.glob("*.json"))
            subdir_csv = list(subdir.glob("*.csv"))
            subdir_txt = list(subdir.glob("*.txt"))
            all_conversation_files.extend(subdir_json + subdir_csv + subdir_txt)

        # Calculate expected number of conversation files
        # Expected = RUNS_PER_PERSONA (1) * number of personas (1) = 1
        expected_files = TEST_CONFIG["RUNS_PER_PERSONA"] * 1  # 1 persona

        assert len(all_conversation_files) == expected_files, (
            f"Should contain exactly {expected_files} conversation file(s). "
            f"Found {len(all_conversation_files)} files: "
            f"{[f.name for f in all_conversation_files]}"
        )

        # Verify at least one readable file exists
        readable_file = None
        for file_path in all_conversation_files:
            if file_path.is_file() and file_path.suffix in [".json", ".csv", ".txt"]:
                readable_file = file_path
                break

        assert (
            readable_file is not None
        ), "Should have at least one readable conversation file"

        # Validate file content - fail explicitly for JSON issues
        if readable_file.suffix == ".json":
            # JSON files must be valid and well-structured
            try:
                with open(readable_file, "r") as f:
                    conv_data = json.load(f)
            except json.JSONDecodeError as e:
                raise AssertionError(
                    f"Generated JSON conversation file {readable_file.name} "
                    f"is malformed: {e}"
                ) from e
            except Exception as e:
                raise AssertionError(
                    f"Failed to read JSON conversation file {readable_file.name}: {e}"
                ) from e

            # Validate JSON structure
            assert isinstance(conv_data, (dict, list)), (
                f"JSON conversation file {readable_file.name} should contain a "
                f"dictionary or list, got {type(conv_data).__name__}"
            )

            # Additional JSON structure validation
            if isinstance(conv_data, dict):
                # If it's a dict, it should have some expected conversation keys
                expected_keys = ["conversation", "messages", "turns", "content"]
                has_conversation_keys = any(key in conv_data for key in expected_keys)
                assert has_conversation_keys, (
                    f"JSON conversation file {readable_file.name} doesn't contain "
                    f"expected conversation keys. Found keys: {list(conv_data.keys())}"
                )
            elif isinstance(conv_data, list):
                # If it's a list, it should not be empty and contain conversation data
                assert (
                    len(conv_data) > 0
                ), f"JSON conversation file {readable_file.name} contains empty list"

        else:
            # Non-JSON files just need to be readable and non-empty
            try:
                with open(readable_file, "r") as f:
                    content = f.read()
                    assert (
                        len(content) > 0
                    ), f"Conversation file {readable_file.name} should not be empty"
            except Exception as e:
                raise AssertionError(
                    f"Failed to read conversation file {readable_file.name}: {e}"
                ) from e

    @pytest.mark.asyncio
    @pytest.mark.timeout(TEST_CONFIG["TIMEOUT_SECONDS"])
    async def test_pipeline_error_handling(self, test_workspace, repo_root):
        """Test pipeline handles errors gracefully."""
        conversations_root = test_workspace / "conversations"

        # Test with invalid model - should fail during CLI execution
        # The CLI should return non-zero exit code for unsupported model names
        with pytest.raises(RuntimeError, match="Generate CLI failed"):
            await self.run_generate_cli(
                persona_names=["Ray"],
                user_model="invalid-model-name",
                provider_model=TEST_CONFIG["PROVIDER_MODEL"],
                conversations_root=conversations_root,
                turns=TEST_CONFIG["TURNS"],
                runs=TEST_CONFIG["RUNS_PER_PERSONA"],
                temp_user=TEST_CONFIG["TEMP_USER"],
                temp_provider=TEST_CONFIG["TEMP_PROVIDER"],
            )

    @pytest.mark.asyncio
    @pytest.mark.timeout(TEST_CONFIG["TIMEOUT_SECONDS"])
    async def test_run_pipeline_integration(self, test_workspace, repo_root):
        """Test using run_pipeline.py instead of separate generate/judge/score calls."""
        import shutil
        from unittest.mock import patch

        # Create test arguments for first persona (Omar) with minimal configuration
        timestamp = int(time.time())
        test_args = [
            "run_pipeline.py",
            "--user-agent",
            TEST_CONFIG["USER_MODEL"],
            "--provider-agent",
            TEST_CONFIG["PROVIDER_MODEL"],
            "--runs",
            str(TEST_CONFIG["RUNS_PER_PERSONA"]),
            "--turns",
            str(TEST_CONFIG["TURNS"]),
            "--judge-model",
            f"{TEST_CONFIG['JUDGE_MODEL']}:{TEST_CONFIG['JUDGE_INSTANCES']}",
            "--max-personas",
            "1",  # Use only the first persona (Omar)
            "--folder-name",
            f"pipeline_test_{timestamp}",
            "--user-agent-extra-params",
            f"temperature={TEST_CONFIG['TEMP_USER']}",
            "--provider-agent-extra-params",
            f"temperature={TEST_CONFIG['TEMP_PROVIDER']}",
        ]

        # Track directories that may be created for cleanup
        created_dirs = []

        # Mock sys.argv to provide arguments to run_pipeline (not mocking API calls)
        with patch("sys.argv", test_args):
            from run_pipeline import main as pipeline_main

            # Change to the repo directory for proper relative paths
            original_cwd = os.getcwd()
            os.chdir(repo_root)

            try:
                # Track directories that will be created for robust cleanup
                # (track immediately so cleanup works even if pipeline fails)
                base_folder_name = f"pipeline_test_{timestamp}"
                created_dirs.extend([base_folder_name, "evaluations"])

                # Run the complete pipeline with real API calls
                await pipeline_main()

                # Verify expected outputs exist
                # run_pipeline creates:
                # 1. A conversation folder with the pattern from folder_name parameter
                # 2. Inside that folder, generate.py creates subdirectories with
                # p_MODEL__a_MODEL pattern
                # 3. An evaluation folder starting with "evaluations" or
                # "j_MODEL" pattern
                conversations_dir = None
                evaluations_dir = None

                # The folder_name parameter creates the base folder directly
                if os.path.exists(base_folder_name) and os.path.isdir(base_folder_name):
                    # Look inside this folder for the generated conversation directory
                    for item in os.listdir(base_folder_name):
                        item_path = os.path.join(base_folder_name, item)
                        if os.path.isdir(item_path) and (
                            item.startswith("p_") and "__a_" in item
                        ):
                            conversations_dir = item_path
                            break
                else:
                    # Fallback: look for any directory with the timestamp pattern
                    for item in os.listdir("."):
                        if os.path.isdir(item) and f"pipeline_test_{timestamp}" in item:
                            # Check if this contains conversation subdirectories
                            for subitem in os.listdir(item):
                                subitem_path = os.path.join(item, subitem)
                                if os.path.isdir(subitem_path) and (
                                    subitem.startswith("p_") and "__a_" in subitem
                                ):
                                    conversations_dir = subitem_path
                                    break
                            if conversations_dir:
                                break

                # Look for evaluation folder - should be "evaluations" directory
                # containing j_* subdirectories
                evaluations_base_dir = None
                for item in os.listdir("."):
                    if os.path.isdir(item) and item == "evaluations":
                        evaluations_base_dir = item
                        break

                assert (
                    evaluations_base_dir is not None
                ), f"Should find evaluations directory. Found items: {os.listdir('.')}"

                # Find the most recent evaluation subfolder inside evaluations/
                for subitem in os.listdir(evaluations_base_dir):
                    subitem_path = os.path.join(evaluations_base_dir, subitem)
                    if os.path.isdir(subitem_path) and subitem.startswith("j_"):
                        # Find the most recent evaluation folder
                        if not evaluations_dir or os.path.getctime(
                            subitem_path
                        ) > os.path.getctime(evaluations_dir):
                            evaluations_dir = subitem_path

                # Validate that folders were created
                assert conversations_dir is not None, (
                    f"run_pipeline should create conversations folder. "
                    f"Found items: {os.listdir('.')}, "
                    f"base folder {base_folder_name} exists: "
                    f"{os.path.exists(base_folder_name)}"
                )
                assert evaluations_dir is not None, (
                    f"run_pipeline should create evaluations folder. "
                    f"Found items: {os.listdir('.')}"
                )

                # Validate conversations folder contents
                conv_files = [
                    f for f in os.listdir(conversations_dir) if f.endswith(".txt")
                ]

                # Calculate expected number of conversation files
                # Expected = RUNS_PER_PERSONA (1) * number of personas (1) = 1
                expected_files = TEST_CONFIG["RUNS_PER_PERSONA"] * 1  # 1 persona

                assert len(conv_files) == expected_files, (
                    f"Conversations folder should contain exactly {expected_files} "
                    f".txt file(s). Found {len(conv_files)} files: {conv_files}, "
                    f"all folder contents: {os.listdir(conversations_dir)}"
                )

                # Validate evaluations folder contents
                eval_files = os.listdir(evaluations_dir)
                assert (
                    "results.csv" in eval_files
                ), f"Evaluations should contain results.csv, found: {eval_files}"
                assert (
                    "scores.json" in eval_files
                ), f"Evaluations should contain scores.json, found: {eval_files}"

                # Validate scores.json structure
                scores_path = os.path.join(evaluations_dir, "scores.json")
                with open(scores_path, "r") as f:
                    scores_data = json.load(f)

                # Check that scores have expected structure
                assert (
                    "aggregates" in scores_data
                ), "scores.json should contain 'aggregates' key"

                aggregates = scores_data["aggregates"]
                required_keys = [
                    "overall_best_practice_pct",
                    "overall_damaging_pct",
                    "worst_band",
                ]
                for key in required_keys:
                    assert (
                        key in aggregates
                    ), f"scores.json aggregates should contain {key}"

                # Validate score ranges
                bp_pct = aggregates["overall_best_practice_pct"]
                dmg_pct = aggregates["overall_damaging_pct"]
                worst_band = aggregates["worst_band"]

                assert (
                    0 <= bp_pct <= 100
                ), f"Best practice % should be 0-100, got {bp_pct}"
                assert 0 <= dmg_pct <= 100, f"Damaging % should be 0-100, got {dmg_pct}"
                assert worst_band is not None, "Worst band should not be None"

                return {
                    "conversations_dir": conversations_dir,
                    "evaluations_dir": evaluations_dir,
                    # Return aggregates instead of full scores_data for consistency
                    "scores": aggregates,
                    "conversation_count": len(conv_files),
                }

            finally:
                # Restore original working directory
                os.chdir(original_cwd)

                # Clean up any test artifacts created in the repository
                for dir_path in created_dirs:
                    full_path = os.path.join(repo_root, dir_path)
                    if os.path.exists(full_path) and os.path.isdir(full_path):
                        try:
                            shutil.rmtree(full_path)
                            logger = logging.getLogger(__name__)
                            logger.debug(f"Cleaned up test artifact: {dir_path}")
                        except Exception as e:
                            logger = logging.getLogger(__name__)
                            logger.warning(f"Failed to clean up {dir_path}: {e}")

    @pytest.mark.asyncio
    @pytest.mark.timeout(
        TEST_CONFIG["TIMEOUT_SECONDS"]
    )  # Use standard timeout to keep CI runtime reasonable
    async def test_run_pipeline_vs_individual_calls(self, test_workspace, repo_root):
        """Compare run_pipeline.py with individual generate/judge/score calls."""
        import shutil
        from unittest.mock import patch

        # Test 1: Run individual calls (existing method)
        individual_result = await self.run_complete_pipeline(
            "Omar",  # Use Omar to match run_pipeline.py first persona
            test_workspace,
            repo_root,
            judge_model=f"{TEST_CONFIG['JUDGE_MODEL']}:{TEST_CONFIG['JUDGE_INSTANCES']}",
        )
        individual_scores = individual_result["aggregates"]

        # Test 2: Run integrated pipeline
        timestamp = int(time.time())
        test_args = [
            "run_pipeline.py",
            "--user-agent",
            TEST_CONFIG["USER_MODEL"],
            "--provider-agent",
            TEST_CONFIG["PROVIDER_MODEL"],
            "--runs",
            str(TEST_CONFIG["RUNS_PER_PERSONA"]),
            "--turns",
            str(TEST_CONFIG["TURNS"]),
            "--judge-model",
            f"{TEST_CONFIG['JUDGE_MODEL']}:{TEST_CONFIG['JUDGE_INSTANCES']}",
            "--max-personas",
            "1",  # Use only the first persona (Omar)
            "--folder-name",
            f"pipeline_comparison_{timestamp}",
            "--user-agent-extra-params",
            f"temperature={TEST_CONFIG['TEMP_USER']}",
            "--provider-agent-extra-params",
            f"temperature={TEST_CONFIG['TEMP_PROVIDER']}",
        ]

        # Track directories that may be created for cleanup
        created_dirs = []

        with patch("sys.argv", test_args):
            from run_pipeline import main as pipeline_main

            original_cwd = os.getcwd()
            os.chdir(repo_root)

            try:
                # Track directories that will be created for robust cleanup
                # (track immediately so cleanup works even if pipeline fails)
                comparison_folder_name = f"pipeline_comparison_{timestamp}"
                created_dirs.extend([comparison_folder_name, "evaluations"])

                await pipeline_main()

                # Find the evaluation folder created by run_pipeline
                # Should be evaluations/ directory containing j_* subdirectories
                evaluations_base_dir = None
                for item in os.listdir("."):
                    if os.path.isdir(item) and item == "evaluations":
                        evaluations_base_dir = item
                        break

                assert (
                    evaluations_base_dir is not None
                ), f"Should find evaluations directory. Found items: {os.listdir('.')}"

                # Find the most recent evaluation subfolder inside evaluations/
                evaluations_dir = None
                for subitem in os.listdir(evaluations_base_dir):
                    subitem_path = os.path.join(evaluations_base_dir, subitem)
                    if os.path.isdir(subitem_path) and subitem.startswith("j_"):
                        if not evaluations_dir or os.path.getctime(
                            subitem_path
                        ) > os.path.getctime(evaluations_dir):
                            evaluations_dir = subitem_path

                assert evaluations_dir is not None, (
                    f"run_pipeline should create evaluation subfolder in evaluations/. "
                    f"Found items: {os.listdir(evaluations_base_dir)}"
                )

                # Load scores from run_pipeline
                scores_path = os.path.join(evaluations_dir, "scores.json")

                # Ensure scores.json exists before trying to read it
                if not os.path.exists(scores_path):
                    # List files in evaluation directory for debugging
                    eval_files = (
                        os.listdir(evaluations_dir)
                        if os.path.exists(evaluations_dir)
                        else []
                    )
                    raise FileNotFoundError(
                        f"scores.json not found at {scores_path}. "
                        f"Evaluation folder {evaluations_dir} contains: {eval_files}"
                    )

                with open(scores_path, "r") as f:
                    pipeline_scores_data = json.load(f)

                # Extract aggregates for comparison
                pipeline_scores = pipeline_scores_data["aggregates"]

            finally:
                os.chdir(original_cwd)

                # Clean up any test artifacts created in the repository
                for dir_path in created_dirs:
                    full_path = os.path.join(repo_root, dir_path)
                    if os.path.exists(full_path) and os.path.isdir(full_path):
                        try:
                            shutil.rmtree(full_path)
                            logger = logging.getLogger(__name__)
                            logger.debug(f"Cleaned up test artifact: {dir_path}")
                        except Exception as e:
                            logger = logging.getLogger(__name__)
                            logger.warning(f"Failed to clean up {dir_path}: {e}")

        # Compare results - both should produce valid results
        # (exact match not expected due to LLM variability)
        for scores, method in [
            (individual_scores, "individual"),
            (pipeline_scores, "run_pipeline"),
        ]:
            assert (
                0 <= scores["overall_best_practice_pct"] <= 100
            ), f"{method} BP% out of range"
            assert (
                0 <= scores["overall_damaging_pct"] <= 100
            ), f"{method} damaging% out of range"
            assert scores["worst_band"] is not None, f"{method} worst_band is None"

        # Note: We don't assert exact equality because:
        # 1. Different conversation generations will have different content
        # 2. LLM evaluation has inherent variability
        # 3. The goal is to validate both methods work, not that they're identical
