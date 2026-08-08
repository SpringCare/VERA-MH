"""Shared config input and rendering for the unified CLI."""

from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

from utils.config_schema import RunConfig

ROOT = Path(__file__).resolve().parents[1]
VERA_RUN_CONFIG_ENV = "VERA_RUN_CONFIG"


class ConfigError(ValueError):
    """Raised when CLI/config input cannot produce a valid invocation."""


def path_from_root(path: str) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return str(candidate.resolve())


def existing_file(path: str, *, field: str) -> str:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ConfigError(f"{field} does not exist or is not a file: {resolved}")
    return str(resolved)


def load_config(config_path: str | None) -> dict[str, Any] | None:
    """Load JSON from a file, stdin, or ``VERA_RUN_CONFIG``."""
    env_config = os.environ.get(VERA_RUN_CONFIG_ENV)
    if config_path and env_config:
        raise ConfigError(f"--config and {VERA_RUN_CONFIG_ENV} are mutually exclusive")
    try:
        if config_path == "-":
            value = json.loads(sys.stdin.read())
        elif config_path:
            value = json.loads(Path(config_path).read_text(encoding="utf-8"))
        elif env_config:
            value = json.loads(env_config)
        else:
            return None
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError(f"could not load config: {error}") from error
    if not isinstance(value, dict):
        raise ConfigError("run config must be a JSON object")
    return value


def print_resolved_config(run_config: RunConfig) -> None:
    print(json.dumps(run_config.to_dict(), indent=2))


def render_invocation(
    run_config: RunConfig, *, command: str, sample: int | None, debug: bool
) -> str:
    compact = json.dumps(run_config.to_dict(), sort_keys=True, separators=(",", ":"))
    controls: list[str] = []
    if sample is not None:
        controls.extend(["--sample", str(sample)])
    if debug:
        controls.append("--debug")
    suffix = f" {shlex.join(controls)}" if controls else ""
    return (
        f"{VERA_RUN_CONFIG_ENV}={shlex.quote(compact)} "
        f"uv run python vera.py {command}{suffix}"
    )
