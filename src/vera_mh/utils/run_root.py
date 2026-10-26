"""Create run folders: the one place a run's root directory comes into being.

Generation, judging, and pooling each start a run by minting a name and
creating a folder for it. Before this module each did that on its own, read the
clock more than once, and created the folder with `exist_ok=True`, so two runs
given the same name silently wrote into one folder. Now:

- `run_started_at` is the run's single clock read. Every timestamp the run
  writes (its folder name, its metadata) comes from that one value.
- `create_run_root` creates the folder and fails if it already exists (AD-24).

Continuing a run (`--into`) opens a folder that already exists, so it does not
come through here.

See `docs/design/run-lifecycle.md` (order of events, rule 3).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


class RunRootExistsError(FileExistsError):
    """A new run's folder already exists, so starting it would mix two runs."""


def run_started_at() -> datetime:
    """Read the clock once, at the start of a run."""
    return datetime.now()


def create_run_root(parent: str | Path, name: str) -> Path:
    """Create the folder for a new run and return its path.

    The parent folders are created as needed. The run folder itself must not
    exist yet: `mkdir` without `exist_ok` checks and creates in one step, so two
    processes starting the same run cannot both succeed.
    """
    root = Path(parent) / name
    root.parent.mkdir(parents=True, exist_ok=True)
    try:
        root.mkdir()
    except FileExistsError as error:
        raise RunRootExistsError(
            f"run folder already exists: {root}. Two runs got the same name; "
            "start the run again, or pass --into to continue that run."
        ) from error
    return root
