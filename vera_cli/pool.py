"""The ``vera pool`` command: flags, resolution, and the workflow call.

Structured like `vera_cli/score.py`, which follows the command contract in
`vera_cli/README.md`:

1. `run` is the entry point `vera.py` dispatches to.
2. `resolve_configs` picks one input form and produces canonical `RunConfig`s.
3. `_execute` hands the resolved run to the scoring domain's `run_pooling`.
4. `register` declares the flags and attaches `run` as the subparser's handler.

Pooling merges two or more judge evaluations into one ``results.csv`` and
scores it. It is a separate command from `score` (and `judge` never pools),
per `docs/architecture.md`. Like `score`, it calls no model, so `_execute` is
synchronous, and a missing `--personas` skips the risk-level breakdown rather
than defaulting to `data/SI/personas.tsv` as the legacy script does.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from score import run_pooling
from utils.config_schema import InvocationConfig, PoolingConfig, RunConfig
from utils.debug import set_debug

from .config import (
    ConfigError,
    config_path,
    existing_file,
    flag_value,
    path_from_root,
    print_resolved_config,
    render_invocation,
    required,
    resolve_input,
)

# CLI behavior defaults, applied during resolution by `flag_value` rather than
# by the parser; see `vera_cli/generate.py` for why the parser cannot hold them.
DEFAULTS: dict[str, Any] = {
    "output": "output",
    "personas": None,
    "skip_risk_analysis": False,
}

# Top-level config keys `pool` accepts.
ALLOWED_CONFIG_FIELDS = {"pooling"}


def run(args: argparse.Namespace) -> int:
    """Resolve the requested run and execute it.

    Resolution happens up front and completely, so a missing evaluation or
    personas file fails before anything is written.
    """
    run_configs = resolve_configs(args)
    if args.print_only:
        for run_config in run_configs:
            print(render_invocation(run_config, command="pool"))
        return 0

    if any(config.invocation.debug for config in run_configs):
        set_debug(True)
    for run_config in run_configs:
        print_resolved_config(run_config)
    return _execute(run_configs)


def resolve_configs(args: argparse.Namespace) -> list[RunConfig]:
    """Resolve either config JSON or CLI flags into exactly one canonical run."""
    try:
        config, invocation = resolve_input(
            args,
            allowed_config_fields=ALLOWED_CONFIG_FIELDS,
        )
        return (
            _from_config(config, invocation)
            if config is not None
            else _from_cli(args, invocation)
        )
    except ConfigError:
        raise
    except (TypeError, ValueError) as error:
        raise ConfigError(f"invalid pooling config: {error}") from error


def _existing_evaluation(path: str, *, field: str) -> str:
    """Return `path` absolute, failing unless it is a folder or a results.csv.

    An evaluation is passed either as its ``j_*`` folder or as the
    ``results.csv`` inside it, the two forms `run_pooling` accepts.
    """
    resolved = Path(path).resolve()
    if resolved.is_dir():
        return str(resolved)
    if resolved.is_file() and resolved.name == "results.csv":
        return str(resolved)
    raise ConfigError(
        f"{field} must be an evaluation folder or its results.csv: {resolved}"
    )


def _from_cli(
    args: argparse.Namespace, invocation: InvocationConfig
) -> list[RunConfig]:
    """Resolve CLI flags into one canonical run, applying CLI defaults.

    Paths resolve against the working directory, like any command-line tool.
    """
    evaluations: list[str] | None = getattr(args, "evaluations", None)
    # Not `required=True` on the parser: a config may supply the value instead.
    if not evaluations:
        raise ConfigError("pool requires --evaluations")
    if len(evaluations) < 2:
        raise ConfigError("pool needs at least two --evaluations to merge")

    personas = flag_value(args, "personas", defaults=DEFAULTS)
    return [
        _run_config(
            invocation=invocation,
            evaluations=[
                _existing_evaluation(path, field="--evaluations")
                for path in evaluations
            ],
            output=str(Path(flag_value(args, "output", defaults=DEFAULTS)).resolve()),
            personas=(
                existing_file(str(Path(personas).resolve()), field="--personas")
                if personas is not None
                else None
            ),
            skip_risk_analysis=flag_value(
                args, "skip_risk_analysis", defaults=DEFAULTS
            ),
        )
    ]


def _from_config(
    config: dict[str, Any], invocation: InvocationConfig
) -> list[RunConfig]:
    """Resolve a config object into one canonical run.

    Every field is required, as in the other commands: a stored config fully
    describes a run. Paths resolve against the repository root, and explicit
    `null` is how a config says "no personas file".
    """
    value = config.get("pooling")
    if not isinstance(value, dict):
        raise ConfigError("pool requires a pooling config object")
    pooling = dict(value)

    evaluations = required(pooling, "evaluations", section="pooling config")
    if not isinstance(evaluations, list) or not all(
        isinstance(path, str) and path for path in evaluations
    ):
        raise ConfigError("pooling.evaluations must be a list of paths")
    output = required(pooling, "output", section="pooling config")
    if not isinstance(output, str) or not output:
        raise ConfigError("pooling.output must be a path string")
    personas = required(pooling, "personas", section="pooling config")

    return [
        _run_config(
            invocation=invocation,
            evaluations=[
                _existing_evaluation(path_from_root(path), field="pooling.evaluations")
                for path in evaluations
            ],
            output=path_from_root(output),
            personas=(
                config_path(personas, field="pooling.personas")
                if personas is not None
                else None
            ),
            skip_risk_analysis=required(
                pooling, "skip_risk_analysis", section="pooling config"
            ),
        )
    ]


def _run_config(
    *,
    invocation: InvocationConfig,
    evaluations: list[str],
    output: str,
    personas: str | None,
    skip_risk_analysis: bool,
) -> RunConfig:
    """Assemble one canonical `RunConfig` holding a pooling section.

    Type enforcement happens at runtime in `PoolingConfig.__post_init__`.
    """
    return RunConfig(
        invocation=invocation,
        pooling=PoolingConfig(
            evaluations=evaluations,
            output=output,
            personas=personas,
            skip_risk_analysis=skip_risk_analysis,
        ),
    )


def _execute(run_configs: list[RunConfig]) -> int:
    """Hand each resolved run to the scoring domain.

    `run_pooling` raises on unusable input (no rows, a missing folder); that is
    translated to the `ConfigError` every other bad input produces.
    """
    for run_config in run_configs:
        pooling = run_config.pooling
        if pooling is None:  # pragma: no cover - resolve_configs always sets it
            raise ConfigError("pool produced a run with no pooling section")
        try:
            run_pooling(
                evaluations=list(pooling.evaluations),
                output_parent=pooling.output,
                personas_tsv=pooling.personas,
                skip_risk_analysis=pooling.skip_risk_analysis,
            )
        except (FileNotFoundError, ValueError) as error:
            raise ConfigError(str(error)) from error
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register ``pool`` with the root parser.

    Uses the same `argparse.SUPPRESS` convention as the other commands, and
    likewise omits `--sample` and `--into`: pooling reads finished results in
    one pass, so there is no work to cap or continue.
    """
    parser = subparsers.add_parser(
        "pool",
        help="Merge two or more judge evaluations into one scored result",
    )
    parser.add_argument(
        "--evaluations",
        nargs="+",
        metavar="<evaluation>",
        default=argparse.SUPPRESS,
        help=(
            "Two or more judge evaluations to merge, each an evaluation folder "
            "(j_*) or the results.csv inside it"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="<dir>",
        default=argparse.SUPPRESS,
        help=(
            "Folder the pooled j_* folder is created in (default: output/ in "
            "the working directory)"
        ),
    )
    parser.add_argument(
        "--personas",
        metavar="<personas.tsv>",
        default=argparse.SUPPRESS,
        help=(
            "Personas file backing the risk-level breakdown "
            "(default: none, which skips that breakdown)"
        ),
    )
    parser.add_argument(
        "--skip-risk-analysis",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Skip the risk-level breakdown even when --personas is given",
    )
    parser.add_argument("--config", help="JSON path or '-' for stdin")
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
    parser.set_defaults(handler=run)
