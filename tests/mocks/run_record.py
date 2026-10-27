"""A minimal run record for tests that start a run by calling the domain directly.

Every new run needs a record (`utils/run_record.py`), which `vera` commands build
from their resolved config. Tests that skip the command layer use this instead:
it has the one section the domain fills in (`provenance.started_at`), and
`create_run_root` writes it as the run's `config.json` like any other record.
"""

from __future__ import annotations

from typing import Any


def minimal_run_record() -> dict[str, Any]:
    return {"format_version": 1, "provenance": {"started_at": None}}
