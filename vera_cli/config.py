"""Shared config input and rendering for the unified CLI."""

from __future__ import annotations

import dataclasses
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

from utils.config_schema import InvocationConfig, ModelSpec, RunConfig

ROOT = Path(__file__).resolve().parents[1]
VERA_RUN_CONFIG_ENV = "VERA_RUN_CONFIG"

# Not every attribute on the parsed namespace came from a flag the user typed.
# Dispatch adds two of its own: `command` (from `add_subparsers(dest="command")`
# in `vera.py`) and `handler` (from each command's `set_defaults(handler=...)`).
#
# `resolve_input` decides which run-defining flags the user supplied by looking
# at what is present on the namespace, so these two must be excluded or they
# would be counted as user input and make every run look CLI-defined.
DISPATCH_ATTRIBUTES = frozenset({"command", "handler"})


class ConfigError(ValueError):
    """Raised when CLI/config input cannot produce a valid invocation."""


def path_from_root(path: str) -> str:
    """Make a config-supplied path absolute, relative to the repository root.

    Config files are checked in and shared, so their relative paths mean
    "relative to the repo", not to whatever directory `vera` was invoked from.
    CLI paths deliberately differ: they resolve against the current directory,
    like every other command-line tool.
    """
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return str(candidate.resolve())


def existing_file(path: str, *, field: str) -> str:
    """Return `path` absolute, failing if it is not an existing file.

    Called during resolution so a missing input fails before any model is
    called, naming the field that referenced it.
    """
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
    """Build one `ModelSpec` from a config object, reporting the field on error."""
    if not isinstance(value, dict):
        raise ConfigError(f"{field} must be an object")
    return ModelSpec.from_dict(value)


def models_from_cli(
    tokens: list[str], role_params: dict[str, Any] | None
) -> list[ModelSpec]:
    """Build `ModelSpec`s from CLI `name[:repeats]` tokens plus role parameters.

    Provider parameters are supplied per *role* on the command line (one
    `--*-params` flag covering every model of that role), matching what the
    legacy scripts accepted. The resolved form stays per-model: each `ModelSpec`
    gets its own copy, so `--print` shows exactly what each model will use and a
    printed config can then be edited per model.

    Per-model differentiation is a config-only capability; the CLI shorthand has
    no room to express it.
    """
    params = dict(role_params or {})
    return [
        dataclasses.replace(ModelSpec.from_shorthand(token), extra_params=dict(params))
        for token in tokens
    ]


def models_from_config(value: Any, *, field: str) -> list[ModelSpec]:
    """Build a `ModelSpec` list from a config array of objects."""
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ConfigError(f"{field} must be a list of objects")
    return [ModelSpec.from_dict(item) for item in value]


def resolve_input(
    args: Any,
    *,
    invocation_only_flags: frozenset[str],
    allowed_config_fields: set[str],
) -> tuple[dict[str, Any] | None, InvocationConfig]:
    """Pick the single input form for this run and resolve shared controls.

    Enforces the one rule every command obeys: a run is defined by CLI flags or
    by a config file, never a mixture, so a resolved run always has one
    traceable origin.

    A flag is run-defining unless the command names it invocation-only, and the
    run-defining set is derived here by subtraction rather than listed. That
    makes the rule structural: a flag added to a command's parser is covered
    without being registered anywhere else. It works because run-defining flags
    use `argparse.SUPPRESS`, so a flag reaches the namespace only when the user
    actually passed it (see `vera_cli/generate.py:register`).

    Both parameters are caller-supplied because the *rule* is shared but the
    *fields* are per-command — `generate` and a future `judge` differ in both.
    `allowed_config_fields` lists the top-level config keys this command
    understands; anything else is rejected rather than ignored, so a typo or a
    section belonging to another command fails loudly instead of silently doing
    nothing.
    """
    config = load_config(getattr(args, "config", None))
    supplied = sorted(
        set(vars(args)) - invocation_only_flags - DISPATCH_ATTRIBUTES,
    )
    if config is not None and supplied:
        flags = ", ".join(f"--{field.replace('_', '-')}" for field in supplied)
        raise ConfigError(
            f"config input cannot be combined with run-defining CLI flags: {flags}"
        )

    persisted: dict[str, Any] = {}
    if config is not None:
        unknown = set(config).difference(allowed_config_fields | {"invocation"})
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
    """Echo the resolved run before executing it, so runs are self-documenting."""
    print(json.dumps(run_config.to_dict(), indent=2))


def render_invocation(run_config: RunConfig, *, command: str) -> str:
    """Render a resolved run as a copy-pasteable command that reproduces it.

    This is what `--print` emits. The resolved config travels in the
    environment variable rather than a temp file so the output is a single
    self-contained line.
    """
    compact = json.dumps(run_config.to_dict(), sort_keys=True, separators=(",", ":"))
    return (
        f"{VERA_RUN_CONFIG_ENV}={shlex.quote(compact)} uv run python vera.py {command}"
    )
