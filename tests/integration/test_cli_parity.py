"""Parity between the legacy `generate.py` CLI and `vera.py generate`.

TEMPORARY — DELETE WITH `generate.py`.

This module exists only for the window in which both CLIs ship. It asserts that
adding `vera generate` did not change what reaches the generation domain, which
is the one claim worth proving while `generate.py` still has users. The roadmap
already names its own end: `docs/architecture.md` Phase 1 deletes
`generate.py`/`judge.py`/`run_pipeline.py` and makes `vera.py` the only entry
point. At that moment the comparison loses its second side, so delete this file
rather than adapting it. See the transitional-boundary section of
`docs/architecture.md` and the docstring on `generate.run_for_user_models`.

The seam is `generate_conversations.run_generation`: both CLIs converge on
exactly one call to it, so stubbing it and diffing the recorded kwargs compares
the two resolution paths without API keys, model calls, or written output.
"""

import runpy
import sys
from pathlib import Path
from typing import Any, Callable, Iterator
from unittest.mock import patch

import pytest

from llm_clients.llm_interface import DEFAULT_START_PROMPT

REPO_ROOT = Path(__file__).parents[2]

# Flags spelled identically by both CLIs. What differs is how each names the
# models, the persona source, and the repeat count.
SHARED_FLAGS = [
    "-t",
    "5",
    "-o",
    "output",
    "--max-concurrent",
    "3",
    "--max-total-words",
    "100",
    "--sessions",
    "intake,followup",
]

LEGACY_ARGV = [
    "generate.py",
    "-u",
    "u-model",
    "-p",
    "c-model",
    "-r",
    "2",
    "--rubric-manifest",
    "data/SI/rubric_manifest.json",
    *SHARED_FLAGS,
]

UNIFIED_ARGV = [
    "generate",
    "-c",
    "c-model",
    "-u",
    "u-model:2",
    "--target",
    "SI",
    *SHARED_FLAGS,
]


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[dict[str, Any]]]:
    """Stub the domain boundary and collect the kwargs each CLI sends it.

    Both patches are needed: `generate` bound `run_generation` at import time,
    while a fresh `runpy` exec of `generate.py` re-imports it from the package.

    `chdir` matters because the legacy CLI resolves paths against the working
    directory, so its manifest argument only resolves from the repository root.
    """
    import generate
    import generate_conversations

    recorded: list[dict[str, Any]] = []

    async def record(**kwargs: Any) -> tuple[list[dict[str, Any]], str]:
        recorded.append(kwargs)
        return ([], "output/stub-run")

    monkeypatch.chdir(REPO_ROOT)
    with (
        patch.object(generate, "run_generation", record),
        patch.object(generate_conversations, "run_generation", record),
    ):
        yield recorded


def run_legacy_cli() -> None:
    """Invoke `generate.py` as a script; its argv handling is under `__main__`."""
    sys.argv = list(LEGACY_ARGV)
    # Returns rather than exiting: the script only calls `sys.exit(1)` when every
    # conversation was skipped, and the stub reports no conversations at all.
    runpy.run_path(str(REPO_ROOT / "generate.py"), run_name="__main__")


def run_unified_cli() -> None:
    import vera

    assert vera.main(list(UNIFIED_ARGV)) == 0


def normalize(call: dict[str, Any]) -> dict[str, Any]:
    """Erase the differences that are equivalent by construction.

    Two survive resolution, and neither changes behavior:

    - Paths. The legacy CLI keeps them relative to the working directory; the
      unified CLI resolves them against the repository root. Same files.
    - `start_prompt`. The legacy CLI writes the default in explicitly; omitting
      it is identical, because `LLMInterface.__init__` substitutes the same
      constant when it is absent. It also reaches `*_extra_run_params`, which
      the runner only prints.

    Anything else that differs is a real divergence and should fail the test.
    """

    def absolute(value: str) -> str:
        return str(Path(value).resolve())

    def without_default_start_prompt(config: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in config.items()
            if not (key == "start_prompt" and value == DEFAULT_START_PROMPT)
        }

    normalized = dict(call)
    normalized["persona_files"] = [absolute(p) for p in call["persona_files"]]
    normalized["persona_context_template_path"] = absolute(
        call["persona_context_template_path"]
    )
    normalized["output_folder"] = absolute(call["output_folder"])
    for key in (
        "persona_model_config",
        "agent_model_config",
        "persona_extra_run_params",
        "agent_extra_run_params",
    ):
        normalized[key] = without_default_start_prompt(call[key])
    return normalized


def capture(calls: list[dict[str, Any]], invoke: Callable[[], None]) -> dict[str, Any]:
    """Run one CLI and return its single normalized domain call."""
    calls.clear()
    invoke()
    assert len(calls) == 1, f"expected one generation, got {len(calls)}"
    return normalize(calls[0])


@pytest.mark.integration
def test_both_clis_send_the_domain_the_same_request(
    calls: list[dict[str, Any]],
) -> None:
    """Equivalent argv through either CLI resolves to one identical run."""
    legacy = capture(calls, run_legacy_cli)
    unified = capture(calls, run_unified_cli)

    differences = {
        key: (legacy.get(key), unified.get(key))
        for key in legacy.keys() | unified.keys()
        if legacy.get(key) != unified.get(key)
    }
    assert not differences, f"CLI paths diverge: {differences}"


@pytest.mark.integration
def test_parity_covers_every_domain_parameter(calls: list[dict[str, Any]]) -> None:
    """Guard the test above: a renamed parameter must not silently drop coverage.

    Equality of two dicts says nothing about which keys they contain, so a
    parameter added to `run_generation` and passed by neither CLI would leave
    the comparison passing while covering less. Pin the parameter list instead.
    """
    expected = {
        "agent_extra_run_params",
        "agent_model_config",
        "max_concurrent",
        "max_personas",
        "max_total_words",
        "max_turns",
        "output_folder",
        "persona_context_template_path",
        "persona_extra_run_params",
        "persona_files",
        "persona_model_config",
        "persona_names",
        "persona_speaks_first",
        "resume",
        "run_id",
        "runs_per_prompt",
        "session_types",
        "verbose",
    }
    assert set(capture(calls, run_unified_cli)) == expected
