"""Shared config input and rendering for the unified CLI."""

from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

from utils.config_schema import InvocationConfig, ModelSpec, RunConfig

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


def required(data: dict[str, Any], field: str, *, section: str) -> Any:
    """Return a required config field with one consistent error shape."""
    if field not in data:
        raise ConfigError(f"{section} is missing required field: {field}")
    return data[field]


def model_from_config(value: Any, *, field: str) -> ModelSpec:
    if not isinstance(value, dict):
        raise ConfigError(f"{field} must be an object")
    return ModelSpec.from_dict(value)


def models_from_config(value: Any, *, field: str) -> list[ModelSpec]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ConfigError(f"{field} must be a list of objects")
    return [ModelSpec.from_dict(item) for item in value]


def resolve_input(
    args: Any,
    *,
    run_fields: tuple[str, ...],
    allowed_fields: set[str],
) -> tuple[dict[str, Any] | None, InvocationConfig]:
    """Load one input form and resolve controls shared by every command."""
    config = load_config(getattr(args, "config", None))
    supplied = [field for field in run_fields if hasattr(args, field)]
    if config is not None and supplied:
        flags = ", ".join(f"--{field.replace('_', '-')}" for field in supplied)
        raise ConfigError(
            f"config input cannot be combined with run-defining CLI flags: {flags}"
        )

    persisted: dict[str, Any] = {}
    if config is not None:
        unknown = set(config).difference(allowed_fields | {"invocation"})
        if unknown:
            raise ConfigError(
                f"unknown top-level config field(s): {', '.join(sorted(unknown))}"
            )
        value = config.get("invocation", {})
        if not isinstance(value, dict):
            raise ConfigError("invocation must be an object")
        unknown = set(value).difference({"debug", "sample"})
        if unknown:
            raise ConfigError(
                f"unknown invocation field(s): {', '.join(sorted(unknown))}"
            )
        persisted = value

    try:
        invocation = InvocationConfig(
            debug=getattr(args, "debug", persisted.get("debug", False)),
            sample=getattr(args, "sample", persisted.get("sample")),
        )
    except ValueError as error:
        raise ConfigError(str(error)) from error
    return config, invocation


def print_resolved_config(run_config: RunConfig) -> None:
    print(json.dumps(run_config.to_dict(), indent=2))


def render_invocation(run_config: RunConfig, *, command: str) -> str:
    compact = json.dumps(run_config.to_dict(), sort_keys=True, separators=(",", ":"))
    return (
        f"{VERA_RUN_CONFIG_ENV}={shlex.quote(compact)} uv run python vera.py {command}"
    )
