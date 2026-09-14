"""Shared config input, path resolution, and rendering for the unified CLI.

Three groups, in order: loading a config document and reading fields off it;
turning config-supplied strings into verified absolute paths; and rendering a
resolved run back out.

The path-resolution group is here rather than in `targets.py` because it has
nothing to do with target manifests -- `config_path` and friends apply one
rule, that a config path is repo-relative and must exist, to whatever field a
command hands them. `targets.py` is about turning a target *name* into a
bundle, and it builds on these rather than owning them.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

from utils.config_schema import InvocationConfig, ModelSpec, RubricFiles, RunConfig

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


def config_path(value: object, *, field: str) -> str:
    """Resolve one explicit config path against the repository root, verifying it.

    Shared by every command's explicit-component branch, so "a config path is
    repo-relative and must exist" is stated once.
    """
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{field} must be a path")
    return existing_file(path_from_root(value), field=field)


def config_paths(value: object, *, field: str) -> list[str]:
    """Resolve a non-empty list of explicit config paths, verifying each."""
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise ConfigError(f"{field} must be a non-empty list of paths")
    return [config_path(item, field=field) for item in value]


def config_dir(value: object, *, field: str) -> str:
    """Resolve one explicit config *directory* against the repository root.

    Sibling of `config_path`/`config_paths`, which verify files. A conversations
    folder is a directory, so it needs its own existence check — but it obeys the
    same rule, so it lives beside them rather than in the one command that
    happens to be the only caller today.
    """
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{field} must be a path")
    resolved = Path(path_from_root(value))
    if not resolved.is_dir():
        raise ConfigError(f"{field} does not exist or is not a directory: {resolved}")
    return str(resolved)


def rubrics_from_config(value: object) -> list[RubricFiles]:
    """Resolve explicit `judging.rubrics` config entries into resolved rubrics.

    Lives here, not in the `judge` command, because the rule it applies is the
    shared one every command's explicit-component branch uses: a config path is
    repo-relative and must exist. That is the same rule as `config_path` beside
    it. What stays with `judge` is the *decision* — target or explicit — not the
    path handling; the command receives rubrics already resolved.

    It also cannot live with `RubricFiles` in `utils/config_schema.py`, tidy as
    that would be: resolution needs `path_from_root`, and `utils/` is the leaf
    layer, so it must not import `vera_cli`.

    `RubricFiles` is constructed exactly once per entry, already resolved. The
    previous version built it twice — once from raw config strings, then again
    from those resolved — which left the type transiently holding unresolved
    paths despite its docstring defining it as the resolved form.
    """
    if not isinstance(value, list) or not value:
        raise ConfigError("judging.rubrics must be a non-empty list of objects")
    rubrics = []
    for entry in value:
        if not isinstance(entry, dict):
            raise ConfigError("judging.rubrics entries must be objects")
        # `from_dict` on the raw entry would give an unresolved `RubricFiles`,
        # so borrow only its field validation and resolve before constructing.
        RubricFiles.validate_dict(entry)
        rubrics.append(
            RubricFiles(
                **{
                    field: config_path(entry[field], field=f"judging.rubrics.{field}")
                    for field in RubricFiles.field_names()
                }
            )
        )
    return rubrics


def flag_value(args: Any, field: str, *, defaults: dict[str, Any]) -> Any:
    """Read a run-defining flag, falling back to the command's CLI default.

    Needed because commands register run-defining flags with
    `argparse.SUPPRESS` so that flag *presence* stays detectable (see
    `vera_cli/generate.py:register`). Suppressed flags never reach the
    namespace, so the default cannot come from the parser and is applied here.

    `defaults` is a parameter rather than a module constant because each command
    owns its own defaults, beside its own flag definitions. That is the only
    thing that differed between the two identical copies this replaces.
    """
    return getattr(args, field, defaults[field])


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
