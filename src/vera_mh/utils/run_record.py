"""The run record: `config.json` in each run folder, saying exactly what ran.

A config file may leave behavior fields to their defaults, so the file alone no
longer pins a run down. The run record does: it is the resolved config with
every default written out, plus the checksums of the files the run read and
where the run came from. It is also a valid run config, so
`vera <command> --config <run>/config.json` runs the same thing again.

```json
{
  "format_version": 1,
  "generation": {"...": "every field, defaults written out"},
  "invocation": {"debug": false, "sample": null},
  "inputs": {"/abs/path/personas.tsv": "sha256:..."},
  "provenance": {"vera_version": "2.1.0", "git_commit": "...", "started_at": "..."}
}
```

`config.json.sha256` holds a hash of everything except `provenance` and
`invocation`: two runs with the same hash used the same settings and the same
input files. Design: `docs/design/config-and-manifest.md`, "Run record".
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from vera_mh.utils.config_schema import RunConfig

FORMAT_VERSION = 1
RECORD_FILE = "config.json"
HASH_FILE = "config.json.sha256"

# Top-level keys a run record adds to a run config. `--config` accepts them so
# a record can be run again.
RECORD_ONLY_KEYS = frozenset({"format_version", "inputs", "provenance"})

# Left out of the hash because they don't change what a run produces.
UNHASHED_KEYS = frozenset({"provenance", "invocation"})

_REPO_ROOT = Path(__file__).resolve().parents[3]


def build_run_record(run_config: RunConfig) -> dict[str, Any]:
    """Build the record for one run, before its folder exists.

    `started_at` is filled in by `with_start_time` once the run has read the
    clock, so the record and the folder name carry the same time.
    """
    config = run_config.to_dict()
    # `into` continues a folder that already exists, so it can't be part of
    # that folder's record (see `RunConfig`).
    config["invocation"].pop("into", None)
    return {
        "format_version": FORMAT_VERSION,
        **config,
        "inputs": {path: file_checksum(path) for path in _input_files(run_config)},
        "provenance": {
            "vera_version": _vera_version(),
            "git_commit": _git_commit(),
            "started_at": None,
        },
    }


def with_start_time(record: dict[str, Any], started_at: datetime) -> dict[str, Any]:
    """A copy of `record` stamped with the run's one clock read."""
    stamped = copy.deepcopy(record)
    stamped["provenance"]["started_at"] = started_at.isoformat(timespec="seconds")
    return stamped


def record_hash(record: dict[str, Any]) -> str:
    """sha256 of the record's settings and inputs, ignoring provenance."""
    hashed = {key: value for key, value in record.items() if key not in UNHASHED_KEYS}
    canonical = json.dumps(hashed, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_run_record(run_root: Path, record: dict[str, Any]) -> None:
    """Write `config.json` and `config.json.sha256` into a new run folder."""
    (run_root / RECORD_FILE).write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    (run_root / HASH_FILE).write_text(record_hash(record) + "\n", encoding="utf-8")


def check_record_inputs(config: dict[str, Any]) -> None:
    """Stop a rerun whose input files changed since the record was written.

    A plain run config has no `inputs` and passes. A record does, and running it
    against an edited rubric or personas file would be a different experiment
    under the old record's name, so each checksum must still match.
    """
    inputs = config.get("inputs", {})
    if not isinstance(inputs, dict):
        raise ValueError("inputs must be an object of path to checksum")
    for path, recorded in inputs.items():
        if not Path(path).is_file():
            raise ValueError(f"input file from the run record is missing: {path}")
        if file_checksum(path) != recorded:
            raise ValueError(
                f"input file changed since the run record was written: {path}. "
                "Rerunning it would not repeat the recorded run."
            )


def file_checksum(path: str | Path) -> str:
    """`sha256:<hex>` of a file's bytes."""
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _input_files(run_config: RunConfig) -> list[str]:
    """Every file the run reads, as named in its resolved config.

    Folders (conversations to judge, evaluations to pool) aren't listed: they
    are the output of an earlier run, which has its own record.
    """
    files: list[str] = []
    if run_config.generation is not None:
        files += run_config.generation.personas
        files.append(run_config.generation.persona_context_template)
    if run_config.judging is not None:
        for rubric in run_config.judging.rubrics:
            files += rubric.to_dict().values()
        files += run_config.judging.personas
    if run_config.scoring is not None:
        files.append(run_config.scoring.results)
        if run_config.scoring.personas is not None:
            files.append(run_config.scoring.personas)
    if run_config.pooling is not None and run_config.pooling.personas is not None:
        files.append(run_config.pooling.personas)
    return list(dict.fromkeys(files))


def _vera_version() -> str | None:
    try:
        return version("vera-mh")
    except PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    """The checkout's commit, or None outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except OSError, subprocess.CalledProcessError:
        return None
    return result.stdout.strip() or None
