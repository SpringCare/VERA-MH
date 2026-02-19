"""Coverage validation tests for LLM implementations.

This module ensures that:
1. All LLM implementations have corresponding test files
2. All LLM implementations test empty conversation history (start_conversation)
3. All JudgeLLM implementations test structured output generation
4. All test classes inherit from appropriate base classes

These tests run in CI to prevent incomplete test coverage for new LLM implementations.
"""

import ast
import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Dict, List, Set

import pytest

import llm_clients
from llm_clients.llm_interface import JudgeLLM, LLMInterface


def get_all_llm_classes() -> Dict[str, List[str]]:
    """Discover all concrete LLM implementation classes.

    Returns:
        Dict with keys 'LLMInterface' and 'JudgeLLM',
        each containing list of class names
    """
    llm_implementations = {"LLMInterface": [], "JudgeLLM": []}

    # Scan llm_clients package
    package_path = Path(llm_clients.__file__).parent

    # Skip these modules
    skip_modules = {"llm_interface", "llm_factory", "config", "__init__"}

    for module_info in pkgutil.iter_modules([str(package_path)]):
        if module_info.name in skip_modules:
            continue

        try:
            module = importlib.import_module(f"llm_clients.{module_info.name}")
        except ImportError:
            continue

        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Only include classes defined in this module
            if obj.__module__ != f"llm_clients.{module_info.name}":
                continue

            # Skip the base interface classes themselves
            if obj in (LLMInterface, JudgeLLM):
                continue

            # Include all classes that inherit from LLMInterface or JudgeLLM
            # This helps catch incomplete implementations
            if issubclass(obj, JudgeLLM):
                llm_implementations["JudgeLLM"].append(obj.__name__)
            elif issubclass(obj, LLMInterface):
                llm_implementations["LLMInterface"].append(obj.__name__)

    return llm_implementations


def get_test_files() -> Set[str]:
    """Get all test files in the llm_clients test directory.

    Returns:
        Set of test file names (without .py extension)
    """
    test_path = Path(__file__).parent
    test_files = set()

    for test_file in test_path.glob("test_*.py"):
        test_files.add(test_file.stem)

    return test_files


def check_file_contains_string(file_path: Path, search_string: str) -> bool:
    """Check if a file contains a specific string.

    Args:
        file_path: Path to file to search
        search_string: String to search for

    Returns:
        True if string found, False otherwise
    """
    try:
        content = file_path.read_text()
        return search_string in content
    except Exception:
        return False


def file_defines_test_function(file_path: Path, test_name: str) -> bool:
    """Check if the file defines an actual test function with the given name.

    Uses AST so that commented-out code does not count. Returns True only
    if there is a FunctionDef (at module level or inside a class) with
    that name.

    Args:
        file_path: Path to the test file
        test_name: Exact name of the test function (e.g. "test_foo")

    Returns:
        True if the function is defined, False otherwise
    """
    try:
        content = file_path.read_text()
        tree = ast.parse(content)

        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == test_name
            ):
                return True
        return False
    except Exception:
        return False


def get_test_class_inheritance(test_file_path: Path) -> Dict[str, List[str]]:
    """Parse test file to determine class inheritance.

    Args:
        test_file_path: Path to test file

    Returns:
        Dict mapping test class names to list of base class names
    """
    try:
        content = test_file_path.read_text()
        tree = ast.parse(content)

        inheritance = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                base_names = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        base_names.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        # Handle cases like module.ClassName
                        base_names.append(base.attr)

                inheritance[node.name] = base_names

        return inheritance
    except Exception:
        return {}


@pytest.mark.unit
class TestLLMCoverage:
    """Tests to ensure complete test coverage for all LLM implementations."""

    def test_all_llm_implementations_have_test_files(self):
        """Ensure every LLM implementation has a corresponding test file."""
        implementations = get_all_llm_classes()
        test_files = get_test_files()

        missing_tests = []

        # Check all LLM implementations
        all_implementations = (
            implementations["LLMInterface"] + implementations["JudgeLLM"]
        )

        for impl_name in all_implementations:
            # Convert class name to expected test file name
            # e.g., "ClaudeLLM" -> "test_claude_llm"
            # e.g., "OpenAILLM" -> "test_openai_llm"

            # Remove "LLM" suffix and convert to snake_case
            name_without_llm = impl_name.replace("LLM", "")

            # Convert CamelCase to snake_case
            # Special handling for common patterns
            import re

            snake_case = re.sub(r"(?<!^)(?=[A-Z])", "_", name_without_llm).lower()

            # Handle special cases like "OpenAI" -> "openai" instead of "open_a_i"
            snake_case = snake_case.replace("open_a_i", "openai")

            expected_test_file = f"test_{snake_case}_llm"

            if expected_test_file not in test_files:
                missing_tests.append((impl_name, expected_test_file + ".py"))

        assert not missing_tests, (
            "\n\nMissing test files for LLM implementations:\n"
            + "\n".join(
                f"  - {impl} should have {test_file}"
                for impl, test_file in missing_tests
            )
            + "\n\nAll LLM implementations must have corresponding test files."
        )

    def test_all_judge_llm_implementations_test_structured_output(self):
        """Ensure all JudgeLLM implementations have structured output tests."""
        implementations = get_all_llm_classes()
        test_path = Path(__file__).parent

        missing_structured_tests = []

        for impl_name in implementations["JudgeLLM"]:
            # Convert class name to test file name
            name_without_llm = impl_name.replace("LLM", "")
            import re

            snake_case = re.sub(r"(?<!^)(?=[A-Z])", "_", name_without_llm).lower()
            test_file_name = f"test_{snake_case}_llm.py"
            test_file_path = test_path / test_file_name

            if test_file_path.exists():
                # Check if file contains structured output tests
                if not check_file_contains_string(
                    test_file_path, "generate_structured_response"
                ):
                    missing_structured_tests.append(impl_name)

        assert not missing_structured_tests, (
            "\n\nMissing structured output tests for JudgeLLM implementations:\n"
            + "\n".join(f"  - {impl}" for impl in missing_structured_tests)
            + "\n\nAll JudgeLLM implementations must test "
            + "generate_structured_response()"
            + ".\n"
            + "These tests should inherit from TestJudgeLLMBase "
            + "or be implemented directly."
        )

    def test_all_llm_implementations_test_empty_conversation_history(self):
        """Ensure all LLM implementations have an empty-conversation-history test."""
        import re

        implementations = get_all_llm_classes()
        test_path = Path(__file__).parent
        all_implementations = (
            implementations["LLMInterface"] + implementations["JudgeLLM"]
        )
        missing_empty_history_tests = []

        for impl_name in all_implementations:
            name_without_llm = impl_name.replace("LLM", "")
            snake_case = re.sub(r"(?<!^)(?=[A-Z])", "_", name_without_llm).lower()
            snake_case = snake_case.replace("open_a_i", "openai")
            test_file_name = f"test_{snake_case}_llm.py"
            test_file_path = test_path / test_file_name

            if test_file_path.exists():
                if not file_defines_test_function(
                    test_file_path,
                    "test_generate_response_with_empty_conversation_history",
                ):
                    missing_empty_history_tests.append(impl_name)

        assert not missing_empty_history_tests, (
            "\n\nMissing empty conversation history tests for LLM implementations:\n"
            + "\n".join(f"  - {impl}" for impl in missing_empty_history_tests)
            + "\n\nAll LLM implementations must define "
            + "test_generate_response_with_empty_conversation_history "
            + "to verify start_conversation / default start_prompt with empty history."
        )

    def test_all_test_classes_inherit_from_appropriate_base(self):
        """Ensure all test classes inherit from TestLLMBase or TestJudgeLLMBase."""
        implementations = get_all_llm_classes()
        test_path = Path(__file__).parent

        incorrect_inheritance = []

        # Check each implementation
        for impl_type in ["LLMInterface", "JudgeLLM"]:
            for impl_name in implementations[impl_type]:
                # Get test file path
                name_without_llm = impl_name.replace("LLM", "")
                import re

                snake_case = re.sub(r"(?<!^)(?=[A-Z])", "_", name_without_llm).lower()
                test_file_name = f"test_{snake_case}_llm.py"
                test_file_path = test_path / test_file_name

                if not test_file_path.exists():
                    continue

                # Parse inheritance
                inheritance = get_test_class_inheritance(test_file_path)

                # Find the main test class (e.g., TestClaudeLLM)
                test_class_name = f"Test{impl_name}"

                if test_class_name in inheritance:
                    base_classes = inheritance[test_class_name]

                    # Determine expected base class
                    if impl_type == "JudgeLLM":
                        expected_base = "TestJudgeLLMBase"
                        acceptable_bases = {"TestJudgeLLMBase", "TestLLMBase"}
                    else:
                        expected_base = "TestLLMBase"
                        acceptable_bases = {"TestLLMBase"}

                    # Check if it inherits from an acceptable base
                    has_correct_inheritance = any(
                        base in acceptable_bases for base in base_classes
                    )

                    if not has_correct_inheritance:
                        incorrect_inheritance.append(
                            (impl_name, test_class_name, expected_base, base_classes)
                        )

        # This test is informational for now - we'll enforce it after migration
        if incorrect_inheritance:
            import warnings

            warning_msg = (
                "\n\nTest classes not yet migrated to base class inheritance:\n"
                + "\n".join(
                    f"  - {test_class} (for {impl}) should inherit from {expected} "
                    f"(currently: {bases})"
                    for impl, test_class, expected, bases in incorrect_inheritance
                )
                + "\n\nThis will become a hard requirement after migration is complete."
            )
            warnings.warn(warning_msg, UserWarning)

    def test_no_duplicate_llm_implementations(self):
        """Ensure LLM class names are unique."""
        implementations = get_all_llm_classes()

        all_names = implementations["LLMInterface"] + implementations["JudgeLLM"]
        unique_names = set(all_names)

        assert len(all_names) == len(unique_names), (
            f"Duplicate LLM implementation names found. "
            f"All implementations: {all_names}"
        )

    def test_llm_implementations_exist(self):
        """Ensure we can discover LLM implementations (sanity check)."""
        implementations = get_all_llm_classes()

        # We should have at least these implementations
        expected_implementations = {
            "OllamaLLM",  # LLMInterface only
            "ClaudeLLM",
            "OpenAILLM",
            "GeminiLLM",
            "AzureLLM",  # JudgeLLM
        }

        all_implementations = set(
            implementations["LLMInterface"] + implementations["JudgeLLM"]
        )

        missing = expected_implementations - all_implementations

        assert not missing, (
            f"Expected LLM implementations not found: {missing}\n"
            f"Found implementations: {all_implementations}"
        )


@pytest.mark.unit
class TestCoverageHelpers:
    """Tests for the coverage validation helper functions."""

    def test_get_all_llm_classes_returns_expected_structure(self):
        """Test that get_all_llm_classes returns correct structure."""
        implementations = get_all_llm_classes()

        assert isinstance(implementations, dict)
        assert "LLMInterface" in implementations
        assert "JudgeLLM" in implementations
        assert isinstance(implementations["LLMInterface"], list)
        assert isinstance(implementations["JudgeLLM"], list)

    def test_get_test_files_finds_test_files(self):
        """Test that get_test_files finds existing test files."""
        test_files = get_test_files()

        assert isinstance(test_files, set)
        # Should find at least some test files
        assert len(test_files) > 0
        # Should find this file
        assert "test_coverage" in test_files

    def test_check_file_contains_string(self):
        """Test string search in files."""
        # Test with test_helpers.py file (known to exist)
        test_helpers_path = Path(__file__).parent / "test_helpers.py"

        assert check_file_contains_string(
            test_helpers_path, "assert_metadata_structure"
        )
        assert not check_file_contains_string(
            test_helpers_path, "THIS_STRING_IS_NOT_IN_ANY_FILE"
        )

    def test_file_defines_test_function_requires_actual_definition(self, tmp_path):
        """file_defines_test_function uses AST; commented-out code does not count."""
        # String in comment only -> False
        commented = tmp_path / "commented.py"
        commented.write_text("# def test_foo(): pass\n")
        assert file_defines_test_function(commented, "test_foo") is False

        # Actual definition -> True
        defined = tmp_path / "defined.py"
        defined.write_text("def test_foo(): pass\n")
        assert file_defines_test_function(defined, "test_foo") is True

        # Async def also counts
        async_def = tmp_path / "async_def.py"
        async_def.write_text("async def test_foo(): pass\n")
        assert file_defines_test_function(async_def, "test_foo") is True

    def test_get_test_class_inheritance(self):
        """Test parsing of test class inheritance."""
        # Test with this file
        test_file = Path(__file__)

        inheritance = get_test_class_inheritance(test_file)

        assert isinstance(inheritance, dict)
        # This file should have TestLLMCoverage class
        assert "TestLLMCoverage" in inheritance
