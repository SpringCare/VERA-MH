"""The ``vera score`` command: flags, resolution, and the workflow call.

Structured exactly like `vera_cli/judge.py` and `vera_cli/generate.py`, which
are the reference implementations of the command contract in
`vera_cli/README.md`:

1. `run` is the entry point `vera.py` dispatches to.
2. `resolve_configs` picks one input form and produces canonical `RunConfig`s.
3. `_execute` hands the resolved run to the scoring domain.
4. `register` declares the flags and attaches `run` as the subparser's handler.

Two structural differences from its siblings, both following from what scoring
is rather than from a different opinion about the contract:

- **No `--target`.** Scoring reads a `results.csv` that judging already wrote;
  it needs no rubric, no personas set, and no prompts. `docs/vera-cli-use-cases.md`
  gives `score` exactly one required input, the results path.
- **`_execute` is synchronous.** No model is called, so there is no event loop
  to start.

One deliberate behavior difference from legacy `judge/score.py`, recorded in
CHANGELOG.md: with no `--personas`, risk-level analysis is skipped rather than
run against a default `data/SI/personas.tsv`. The lookup behind it joins on a
column literally named "Short Current Suicide Risk Level", so against any other
target's personas file it produced "Unknown" for every row and an empty
`scores_by_risk.json`. Skipping explicitly beats emitting output that looks
like a result. Passing `--personas` keeps the legacy behavior exactly.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from judge.score import run_scoring
from utils.config_schema import InvocationConfig, RunConfig, ScoringConfig
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
# by the parser — see `vera_cli/generate.py` for why the parser cannot hold them.
#
# `output` has no static default: `run_scoring` writes `scores/scores.json`
# beside the results CSV when it is None. `personas` defaults to None rather
# than to a personas file; see the module docstring.
DEFAULTS: dict[str, Any] = {
    "output": None,
    "personas": None,
    "skip_risk_analysis": False,
}

# Top-level config keys `score` accepts. `generation` and `judging` are absent
# for the same reason `judge` rejects `generation`: a key this command would
# ignore is better rejected than accepted. A later `pipeline` accepts all three.
ALLOWED_CONFIG_FIELDS = {"scoring"}


def run(args: argparse.Namespace) -> int:
    """Resolve the requested run and execute it.

    Resolution happens up front and completely, so a missing results CSV or
    personas file fails before anything is written.
    """
    run_configs = resolve_configs(args)
    if args.print_only:
        for run_config in run_configs:
            print(render_invocation(run_config, command="score"))
        return 0

    if any(config.invocation.debug for config in run_configs):
        set_debug(True)
    for run_config in run_configs:
        print_resolved_config(run_config)
    return _execute(run_configs)


def resolve_configs(args: argparse.Namespace) -> list[RunConfig]:
    """Resolve either config JSON or CLI flags into canonical runs.

    Always returns exactly one `RunConfig`. The list shape matches the sibling
    commands so `run` reads the same in all three; only `generate` can ever
    return more than one, via `--target all`.
    """
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
        raise ConfigError(f"invalid scoring config: {error}") from error


def _from_cli(
    args: argparse.Namespace, invocation: InvocationConfig
) -> list[RunConfig]:
    """Resolve CLI flags into one canonical run, applying CLI defaults."""
    results: str | None = getattr(args, "results", None)

    # The parser cannot enforce this with `required=True`: a config may supply
    # the value instead, and the flag uses SUPPRESS so absence is
    # indistinguishable from a default.
    if not results:
        raise ConfigError("score requires -r/--results")

    personas = flag_value(args, "personas", defaults=DEFAULTS)
    output = flag_value(args, "output", defaults=DEFAULTS)

    return [
        _run_config(
            invocation=invocation,
            # Resolved against the working directory, like every other
            # command-line tool, and verified so a typo fails before any file
            # is written. The config form of this same field resolves against
            # the repository root instead -- see `_from_config`, which is the
            # only place a config is read.
            results=existing_file(str(Path(results).resolve()), field="--results"),
            output=str(Path(output).resolve()) if output is not None else None,
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

    Every behavior field is required rather than defaulted, matching `generate`
    and `judge`: a stored config is a complete, reproducible description of a
    run, so a value it does not state is an error rather than something this
    code fills in. `output` and `personas` are required to be *present*, and
    explicit `null` is how a config says "no value".
    """
    value = config.get("scoring")
    if not isinstance(value, dict):
        raise ConfigError("score requires a scoring config object")
    scoring = dict(value)

    output = required(scoring, "output", section="scoring config")
    if output is not None and (not isinstance(output, str) or not output):
        raise ConfigError("scoring.output must be null or a path string")
    personas = required(scoring, "personas", section="scoring config")

    return [
        _run_config(
            invocation=invocation,
            results=config_path(
                required(scoring, "results", section="scoring config"),
                field="scoring.results",
            ),
            output=path_from_root(output) if output is not None else None,
            personas=(
                config_path(personas, field="scoring.personas")
                if personas is not None
                else None
            ),
            skip_risk_analysis=required(
                scoring, "skip_risk_analysis", section="scoring config"
            ),
        )
    ]


def _run_config(
    *,
    invocation: InvocationConfig,
    results: str,
    output: str | None,
    personas: str | None,
    skip_risk_analysis: bool,
) -> RunConfig:
    """Assemble and validate one canonical `RunConfig` holding a scoring section.

    Fields are named rather than forwarded as opaque keywords so this signature
    states what a scoring run consists of. Type enforcement happens at runtime
    in `ScoringConfig.__post_init__`.
    """
    return RunConfig(
        invocation=invocation,
        scoring=ScoringConfig(
            results=results,
            output=output,
            personas=personas,
            skip_risk_analysis=skip_risk_analysis,
        ),
    )


def _execute(run_configs: list[RunConfig]) -> int:
    """Hand each resolved run to the scoring domain.

    Synchronous, unlike `generate` and `judge`: scoring reads a CSV that already
    exists and calls no model, so there is nothing to await.

    `run_scoring` raises when there is nothing scoreable. That is a property of
    the input rather than of the command line, so it is translated to the same
    `ConfigError` every other bad input produces and `vera.py` presents it as a
    standard CLI error.
    """
    for run_config in run_configs:
        scoring = run_config.scoring
        if scoring is None:  # pragma: no cover - resolve_configs always sets it
            raise ConfigError("score produced a run with no scoring section")
        try:
            run_scoring(
                results_csv=scoring.results,
                output_json=scoring.output,
                personas_tsv=scoring.personas,
                skip_risk_analysis=scoring.skip_risk_analysis,
            )
        except (FileNotFoundError, ValueError) as error:
            raise ConfigError(str(error)) from error
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register ``score`` with the root parser.

    Uses the same `argparse.SUPPRESS` convention as `generate` and `judge`:
    run-defining flags are absent from the namespace unless the user passed
    them, which is what makes the config-or-flags rule enforceable.

    `--sample` and `--into` are deliberately not offered. Both are invocation
    controls that cap or continue *work*, and scoring performs none: it reads
    one finished CSV in one pass. Omitting them keeps `-h` honest rather than
    advertising two flags that would have nothing to act on.
    """
    parser = subparsers.add_parser(
        "score", help="Score judged results and draw the score visualizations"
    )
    parser.add_argument(
        "-r",
        "--results",
        metavar="<results.csv>",
        default=argparse.SUPPRESS,
        help=(
            "Per-question evaluation CSV written by a judge run, found at "
            "<evaluation run>/results.csv. Scoring aggregates it; it is the "
            "one required input"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="<scores.json>",
        default=argparse.SUPPRESS,
        help=(
            "Path for the scores JSON "
            "(default: scores/scores.json beside the results CSV)"
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

    # `run` is this module's handler, defined at the top of the module. `vera.py`
    # dispatches to whatever a subparser records here.
    parser.set_defaults(handler=run)
