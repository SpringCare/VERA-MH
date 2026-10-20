"""`import score` must not load the judge or the LLM stack.

Import-linter checks import statements, but not what a package's `__init__.py`
pulls in when any of its submodules is imported; this checks what actually
gets loaded, in a fresh interpreter.
"""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("package", ["judge", "llm_clients"])
def test_importing_score_does_not_load(package: str):
    loaded = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, score, score.compare; "
            f"print(any(m == {package!r} or m.startswith({package + '.'!r}) "
            "for m in sys.modules))",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    assert loaded == "False", f"importing score loaded {package}"
