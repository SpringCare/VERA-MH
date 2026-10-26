"""Shared config input, path resolution, and rendering for the unified CLI.

Four groups, in order: loading a config document and reading fields off it;
turning config-supplied strings into verified absolute paths; rendering a
resolved run back out; and the command skeleton every `vera` command shares
(`run_command`, `resolve_with`, `add_common_flags`).

The path-resolution group is here rather than in `targets.py` because it has
nothing to do with target manifests -- `config_path` and friends apply one
rule, that a config path is repo-relative and must exist, to whatever field a
command hands them. `targets.py` is about turning a target *name* into a
bundle, and it builds on these rather than owning them.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import inspect
import json
import os
import shlex
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol, TypeVar

from vera_mh.utils.config_schema import (
    InvocationConfig,
    ModelSpec,
    RubricFiles,
    RunConfig,
)
from vera_mh.utils.debug import set_debug

ROOT = Path(__file__).resolve().parents[3]
VERA_RUN_CONFIG_ENV = "VERA_RUN_CONFIG"

# Not every attribute on the parsed namespace came from a flag the user typed.
# Dispatch adds two of its own: `command` (from `add_subparsers(dest="command")`
# in `cli.py`) and `handler` (from each command's `set_defaults(handler=...)`).
#
# `resolve_input` decides which run-defining flags the user supplied by looking
# at what is present on the namespace, so these two must be excluded or they
# would be counted as user input and make every run look CLI-defined.
DISPATCH_ATTRIBUTES = frozenset({"command", "handler"})

# Flags that do not define the run: they change how one invocation executes or
# is presented, so they are the only ones that may accompany `--config`.
#
# Not a hand-written list, and not per-command. The behaviour half is exactly
# the fields of `InvocationConfig` — that dataclass is what "invocation-only"
# means, and each of its fields is the dest of the flag with the same name.
# Added to it are the two flags that carry no invocation field at all:
# `--config` names the input form itself, and `--print` exits before a run
# exists to describe.
#
# This is the only flag classification written down. Every other flag is
# run-defining by subtraction (`resolve_input` derives it from the parsed
# namespace), so a newly added flag is subject to the config-or-flags rule
# automatically rather than needing to be listed somewhere second.
INVOCATION_ONLY_FLAGS = InvocationConfig.field_names() | {"config", "print_only"}


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


def config_section(
    config: dict[str, Any], name: str, *, command: str
) -> dict[str, Any]:
    """Return a copy of the config section `name`, which must be an object."""
    value = config.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"{command} requires a {name} config object")
    return dict(value)


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


def string_list(value: object, *, field: str) -> list[str]:
    """Validate a config value is a non-empty list of non-empty strings."""
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise ConfigError(f"{field} must be a non-empty list of paths")
    return value


def config_paths(value: object, *, field: str) -> list[str]:
    """Resolve a non-empty list of explicit config paths, verifying each."""
    return [config_path(item, field=field) for item in string_list(value, field=field)]


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


def _resolved_into(*, value: object, from_config: bool) -> str | None:
    """Resolve `--into`/`invocation.into` to an existing run folder, or `None`.

    Follows the same split as every other path: a CLI value resolves against the
    working directory, a config-stated one against the repository root. The
    folder must already exist — continuing a run that was never started is a
    typo, not a request to create one.

    `value` is typed `object` because it arrives unvalidated from two different
    places: `argparse` (always a `str` when present) and a parsed JSON document
    (any JSON type at all). Narrowing it is this function's job, so it cannot
    claim `str` on the way in.
    """
    # Three ways to mean "no folder to continue": the flag was omitted, so the
    # namespace lookup fell through to `None`; the config states `"into": null`;
    # or it states `""`, which names nothing. All resolve to no continuation
    # rather than an error, so omitting the field behaves exactly as before it
    # existed.
    if not value:
        return None
    # Reachable only from a config document — argparse cannot produce a
    # non-string here — so the message names the config field.
    if not isinstance(value, str):
        raise ConfigError("invocation.into must be null or a run folder path")
    resolved = Path(path_from_root(value)) if from_config else Path(value).resolve()
    # Checked at resolution time, before any model is called: a folder that is
    # missing (or is a file) means the run being continued does not exist, and
    # creating it would silently start a fresh run under a name the user
    # expected to already hold results.
    if not resolved.is_dir():
        raise ConfigError(f"--into is not an existing run folder: {resolved}")
    return str(resolved)


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

    `allowed_config_fields` is caller-supplied because the *rule* is shared but
    the *sections* are per-command: it lists the top-level config keys this
    command understands, and anything else is rejected rather than ignored, so
    a typo or a section belonging to another command fails loudly instead of
    silently doing nothing. The invocation-only set is deliberately *not* a
    parameter — it is uniform across commands by construction (see
    `INVOCATION_ONLY_FLAGS`), and passing it per command would invite the two
    to drift apart.
    """
    config = load_config(getattr(args, "config", None))
    supplied = sorted(
        set(vars(args)) - INVOCATION_ONLY_FLAGS - DISPATCH_ATTRIBUTES,
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
        unknown = set(value).difference(InvocationConfig.field_names())
        if unknown:
            raise ConfigError(
                f"unknown invocation field(s): {', '.join(sorted(unknown))}"
            )
        persisted = value

    try:
        # Each control is read from the namespace first and falls back to the
        # stored `invocation` object. The two are never both populated: the
        # config-or-flags check above has already rejected that combination for
        # run-defining flags, and for these the namespace only carries what the
        # user typed on *this* invocation, which is what should win.
        into = getattr(args, "into", persisted.get("into"))
        invocation = InvocationConfig(
            debug=getattr(args, "debug", persisted.get("debug", False)),
            sample=getattr(args, "sample", persisted.get("sample")),
            # A CLI path, so it resolves against the working directory like
            # `--output` does; a config-stated one against the repository root.
            into=_resolved_into(value=into, from_config=config is not None),
        )
    except ValueError as error:
        raise ConfigError(str(error)) from error
    return config, invocation


def print_resolved_config(run_config: RunConfig) -> None:
    """Echo the resolved run before executing it, so runs are self-documenting."""
    print(json.dumps(run_config.to_dict(), indent=2))


class _Renderable(Protocol):
    """Any resolved run that can serialize itself: has a ``to_dict()``.

    A typing-only description, so `render_invocation` (and `run_command`'s
    `--print` path) accept both a `RunConfig` (generate, judge, score, pool)
    and a pipeline's `PipelineRun`, which share no base class but both
    serialize to the config dict `--print` emits.
    """

    def to_dict(self) -> dict[str, Any]: ...


def render_invocation(run_config: _Renderable, *, command: str) -> str:
    """Render a resolved run as a copy-pasteable command that reproduces it.

    This is what `--print` emits. The resolved config travels in the
    environment variable rather than a temp file so the output is a single
    self-contained line.
    """
    compact = json.dumps(run_config.to_dict(), sort_keys=True, separators=(",", ":"))
    return f"{VERA_RUN_CONFIG_ENV}={shlex.quote(compact)} uv run vera {command}"


# --- The command skeleton -------------------------------------------------
#
# Every `vera` command follows the same four steps (vera_cli/README.md): `run`
# resolves the input, honors `--print` and `--debug`, echoes the resolved run,
# then executes it; `resolve_configs` picks config or flags; `_execute` calls
# the domain; `register` declares the flags. Only the per-command parts
# (fields, defaults, the domain call, run-defining flags) live in the command
# modules; the identical parts live here.

# The type of one resolved run, as a command's `resolve_configs` returns it:
# `RunConfig` for generate, judge, score, and pool; `PipelineRun` for pipeline.
# A type variable rather than a fixed type, so the type checker can confirm that
# `resolve`'s output is what `execute`, `echo`, and `debug_of` receive.
ResolvedRun = TypeVar("ResolvedRun", bound=_Renderable)


def run_command(
    args: argparse.Namespace,
    *,
    command: str,
    resolve: Callable[[argparse.Namespace], Sequence[ResolvedRun]],
    execute: Callable[[list[ResolvedRun]], Any],
    echo: Callable[[ResolvedRun], Any] | None = None,
    debug_of: Callable[[ResolvedRun], bool] | None = None,
) -> int:
    """Resolve, then print-only or execute: the body of every command's `run`.

    Resolution happens up front and completely, so bad input fails before any
    model is called or file written. `execute` may be async (generate, judge,
    pipeline), in which case it is run to completion and the exit code is 0;
    a sync `execute` (score, pool) returns the exit code itself.

    `echo` and `debug_of` default to a `RunConfig`'s; `pipeline` passes its
    own because a `PipelineRun` holds its invocation inside `generation`.
    """
    runs = list(resolve(args))
    if args.print_only:
        for run in runs:
            print(render_invocation(run, command=command))
        return 0

    debug = debug_of or (lambda run: run.invocation.debug)  # type: ignore[attr-defined]
    if any(debug(run) for run in runs):
        set_debug(True)
    for run in runs:
        (echo or print_resolved_config)(run)  # type: ignore[arg-type]
    result = execute(runs)
    if inspect.iscoroutine(result):
        asyncio.run(result)
        return 0
    return result


def resolve_with(
    args: argparse.Namespace,
    *,
    allowed_config_fields: set[str],
    label: str,
    from_config: Callable[[dict[str, Any], InvocationConfig], list[ResolvedRun]],
    from_cli: Callable[[argparse.Namespace, InvocationConfig], list[ResolvedRun]],
) -> list[ResolvedRun]:
    """Resolve config JSON or CLI flags into canonical runs.

    `ConfigError` passes through untouched, because it already carries a
    message written for the user; that arm must come first, since `ConfigError`
    is itself a `ValueError` and would otherwise be wrapped a second time.
    Anything else a section's validation raises (`TypeError`/`ValueError` from
    a dataclass `__post_init__`) is wrapped as
    `invalid <label> config: ...`, so every bad input reaches `cli.py` as the
    same kind of error.
    """
    try:
        config, invocation = resolve_input(
            args, allowed_config_fields=allowed_config_fields
        )
        return (
            from_config(config, invocation)
            if config is not None
            else from_cli(args, invocation)
        )
    except ConfigError:
        raise
    except (TypeError, ValueError) as error:
        raise ConfigError(f"invalid {label} config: {error}") from error


def add_common_flags(
    parser: argparse.ArgumentParser,
    *,
    handler: Callable[[argparse.Namespace], int],
    sample_help: str | None = None,
    into_group: Any = None,
) -> None:
    """Add the invocation flags every command shares, then attach `handler`.

    `--config`, `-d/--debug` and `--print` exist on every command. `--sample`
    is added only when a command has work to cap (`sample_help` says what it
    caps), and `--into` only into the mutually exclusive group it shares with
    `-o/--output`. Flags are added in one fixed order so `-h` output stays
    stable across commands.
    """
    if into_group is not None:
        into_group.add_argument(
            "--into",
            default=argparse.SUPPRESS,
            metavar="<run folder>",
            help=(
                "Resume: continue an existing run folder, skipping work already on "
                "disk (replaces --output, which mints a new run instead)"
            ),
        )
    parser.add_argument("--config", help="JSON path or '-' for stdin")
    if sample_help is not None:
        parser.add_argument(
            "--sample",
            type=int,
            default=argparse.SUPPRESS,
            help=sample_help,
        )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Enable debug logging",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        dest="print_only",
        help="Print the resolved invocation without executing it",
    )
    # `cli.py` dispatches with `args.handler(args)`.
    parser.set_defaults(handler=handler)
