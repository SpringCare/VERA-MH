"""The ``vera score`` command: flags, resolution, and the workflow call.

Structured exactly like `vera_cli/judge.py` and `vera_cli/generate.py`, which
are the reference implementations of the command contract in
`vera_cli/README.md`:

1. `run` is the entry point `cli.py` dispatches to.
2. `resolve_configs` picks one input form and produces canonical `RunConfig`s.
3. `_execute` hands the resolved run to the scoring domain.
4. `register` declares the flags and attaches `run` as the subparser's handler.

`run`, `resolve_configs` and the shared flags in `register` delegate to
`run_command`, `resolve_with` and `add_common_flags` in `vera_cli/config.py`;
this module supplies only what is specific to the command.

Two structural differences from its siblings, both following from what scoring
is rather than from a different opinion about the contract:

- **No `--target`.** Scoring reads a `results.csv` that judging already wrote;
  it needs no rubric, no personas set, and no prompts. `docs/vera-cli-use-cases.md`
  gives `score` exactly one required input, the results path.
- **`_execute` is synchronous.** No model is called, so there is no event loop
  to start.

One deliberate behavior difference from the pre-2.0 `score.py`, recorded in
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

from vera_mh.score import run_scoring
from vera_mh.utils.config_schema import InvocationConfig, RunConfig, ScoringConfig

from .config import (
    ConfigError,
    add_common_flags,
    config_path,
    config_section,
    existing_file,
    flag_value,
    path_from_root,
    required,
    resolve_with,
    run_command,
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
    return run_command(args, command="score", resolve=resolve_configs, execute=_execute)


def resolve_configs(args: argparse.Namespace) -> list[RunConfig]:
    """Resolve either config JSON or CLI flags into canonical runs.

    Always returns exactly one `RunConfig`. The list shape matches the sibling
    commands so `run` reads the same in all three; only `generate` can ever
    return more than one, via `--target all`.
    """
    return resolve_with(
        args,
        allowed_config_fields=ALLOWED_CONFIG_FIELDS,
        label="scoring",
        from_config=_from_config,
        from_cli=_from_cli,
    )


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
        RunConfig(
            invocation=invocation,
            scoring=ScoringConfig(
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
            ),
        )
    ]


def _from_config(
    config: dict[str, Any], invocation: InvocationConfig
) -> list[RunConfig]:
    """Resolve a config object into one canonical run.

    `results`, `personas`, and `skip_risk_analysis` are required: the last two
    decide whether the risk-level breakdown exists, so a config must say, with
    an explicit `null` for no personas file. `output` only says where scores go,
    so it takes the CLI default when left out.
    """
    scoring = config_section(config, "scoring", command="score")

    output = scoring.get("output", DEFAULTS["output"])
    if output is not None and (not isinstance(output, str) or not output):
        raise ConfigError("scoring.output must be null or a path string")
    personas = required(scoring, "personas", section="scoring config")

    return [
        RunConfig(
            invocation=invocation,
            scoring=ScoringConfig(
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
            ),
        )
    ]


def _execute(run_configs: list[RunConfig]) -> int:
    """Hand each resolved run to the scoring domain.

    Synchronous, unlike `generate` and `judge`: scoring reads a CSV that already
    exists and calls no model, so there is nothing to await.

    `run_scoring` raises when there is nothing scoreable. That is a property of
    the input rather than of the command line, so it is translated to the same
    `ConfigError` every other bad input produces and `cli.py` presents it as a
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
    add_common_flags(
        parser,
        handler=run,
    )
