"""The ``vera judge`` command: flags, resolution, and the workflow call.

Structured exactly like `vera_cli/generate.py`, which is the reference
implementation of the command contract in `vera_cli/README.md`:

1. `run` is the entry point `cli.py` dispatches to.
2. `resolve_configs` picks one input form and produces canonical `RunConfig`s.
3. `_execute` hands each resolved run to the judging domain.
4. `register` declares the flags and attaches `run` as the subparser's handler.

`run`, `resolve_configs` and the shared flags in `register` delegate to
`run_command`, `resolve_with` and `add_common_flags` in `vera_cli/config.py`;
this module supplies only what is specific to the command.

Steps 1-3 are the path one run takes, in the order it takes them. `register`
comes last for the reason given in `generate.py`: it is a declaration rather
than a step, and its ~95 lines of flag definitions sit between the reader and
the logic if they come first.

Nothing after `resolve_configs` reads an `argparse.Namespace`, and nothing
before `_execute` touches the judging domain.

One deliberate difference from legacy `judge.py`, recorded in
docs/vera-cli-use-cases.md: no single-conversation mode. Judge a folder
containing one conversation instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from vera_mh.judge import run_judging
from vera_mh.utils.config_schema import (
    InvocationConfig,
    JudgingConfig,
    RubricFiles,
    RunConfig,
)
from vera_mh.utils.conversation_layout import resolve_conversation_input
from vera_mh.utils.run_record import build_run_record
from vera_mh.utils.utils import parse_key_value_list

from .config import (
    ConfigError,
    add_common_flags,
    config_dir,
    config_path,
    config_section,
    flag_value,
    models_from_cli,
    models_from_config,
    path_from_root,
    required,
    resolve_with,
    rubrics_from_config,
    run_command,
    string_list,
)
from .targets import (
    ResolvedTarget,
    load_target,
    resolve_target_manifest,
    targets_from_config,
)

# CLI behavior defaults, applied during resolution by `flag_value` rather than by
# the parser — see `vera_cli/generate.py` for why the parser cannot hold them.
#
# `output` has no static default: it is derived from the conversations folder,
# landing beside the transcripts it evaluates. `-h` says so.
DEFAULTS: dict[str, Any] = {
    "output": None,
    "max_concurrent": None,
    "per_judge": False,
}

# Top-level config keys `judge` accepts. `generation` is absent for the same
# reason `generate` rejects `judging`: a key this command would ignore is worse
# rejected than accepted. A later `pipeline` accepts both.
ALLOWED_CONFIG_FIELDS = {"judging", "target"}


def run(args: argparse.Namespace) -> int:
    """Resolve the requested run(s) and execute them.

    Resolution happens up front and completely, so an invalid target, missing
    rubric file, or underivable output location fails before any model is called.
    """
    return run_command(args, command="judge", resolve=resolve_configs, execute=_execute)


def resolve_configs(args: argparse.Namespace) -> list[RunConfig]:
    """Resolve either config JSON or CLI flags into canonical runs.

    Always returns exactly one `RunConfig`. Unlike `generate`, `--target all` is
    rejected rather than fanned out — see `_target_selection`.
    """
    return resolve_with(
        args,
        allowed_config_fields=ALLOWED_CONFIG_FIELDS,
        label="judging",
        from_config=_from_config,
        from_cli=_from_cli,
    )


def _from_cli(
    args: argparse.Namespace, invocation: InvocationConfig
) -> list[RunConfig]:
    """Resolve CLI flags into one canonical run, applying CLI defaults.

    `--target` and `--rubric` differ only in intent, exactly as `--target` and
    `--personas` do for `generate`: the first names a whole bundle, the second
    names the rubric component explicitly. Both resolve through the same manifest
    to the same three files.
    """
    models: list[str] | None = getattr(args, "judge", None)
    conversations: list[str] | None = getattr(args, "conversations", None)
    target: str | None = getattr(args, "target", None)
    rubric: str | None = getattr(args, "rubric", None)

    # The parser cannot enforce these: a config may supply the same values, and
    # the flags use SUPPRESS so absence is indistinguishable from a default. The
    # target/rubric group enforces "not both" but cannot require one.
    if not models:
        raise ConfigError("judge requires at least one -j/--judge model")
    if not conversations:
        raise ConfigError("judge requires --conversations")
    # Both routes resolve through the same manifest to the same run, which
    # `test_target_and_rubric_resolve_the_same_rubric` pins: they differ in
    # intent, not in result. Persona annotation therefore comes from whichever
    # was named rather than being a privilege of `--target`.
    selection = target or rubric
    if not selection:
        raise ConfigError("judge requires --target or --rubric")
    resolved = _target_from_selection(selection)
    rubric_files = _rubric_files(resolved)
    personas = list(resolved.personas)
    annotation_columns = list(resolved.persona_annotation_columns)

    folders = [str(Path(folder).resolve()) for folder in conversations]
    return [
        RunConfig(
            invocation=invocation,
            judging=JudgingConfig(
                models=models_from_cli(models, getattr(args, "judge_params", None)),
                conversations=folders,
                rubrics=[rubric_files],
                output=_output_root(
                    folders[0], flag_value(args, "output", defaults=DEFAULTS)
                ),
                max_concurrent=flag_value(args, "max_concurrent", defaults=DEFAULTS),
                per_judge=flag_value(args, "per_judge", defaults=DEFAULTS),
                personas=personas,
                persona_annotation_columns=annotation_columns,
            ),
        )
    ]


def _from_config(
    config: dict[str, Any], invocation: InvocationConfig
) -> list[RunConfig]:
    """Resolve a config object into one canonical run.

    `models`, `conversations`, and the rubric (or a target) are required. The
    behavior fields take the CLI defaults when left out, as in `generate`, and a
    missing `output` lands beside the conversations, as `--output` does.
    """
    judging = config_section(config, "judging", command="judge")
    models = models_from_config(
        required(judging, "models", section="judging config"),
        field="judging.models",
    )
    conversations = [
        config_dir(folder, field="judging.conversations")
        for folder in string_list(
            required(judging, "conversations", section="judging config"),
            field="judging.conversations",
        )
    ]

    targets = targets_from_config(
        config=config,
        section=judging,
        explicit_fields=("rubrics",),
        section_name="judging",
    )
    if targets is not None:
        if len(targets) != 1:
            raise ConfigError(
                "judge does not support target 'all' yet: evaluations for "
                "different rubrics would share one output folder"
            )
        # The one-element list is AD-20's shape, not this branch's limitation: `rubrics`
        # is list-shaped from day one while `JudgingConfig` accepts only length 1
        # (several rubrics are evaluated as one target each, see docs/roadmap.md). It is
        # independent of the guard above — per AD-21 the `target all` lift produces one
        # invocation *per target*, so this line stays a one-element list inside a
        # comprehension rather than growing a second entry.
        rubrics = [_rubric_files(targets[0])]
        personas = list(targets[0].personas)
        annotation_columns = list(targets[0].persona_annotation_columns)
    else:
        rubrics = rubrics_from_config(
            required(judging, "rubrics", section="judging config")
        )
        # Unlike the behavior fields, these two are optional rather than
        # `required`: they annotate output rather than defining the run, and a
        # stored config that predates them still describes its run completely.
        # Reading them back is what makes `--print` round-trip.
        # `string_list` rejects an empty list, but empty is the meaningful
        # "annotate nothing" value here, so absent and empty are both accepted
        # and only a non-empty value is shape-checked.
        raw_personas = judging.get("personas") or []
        personas = (
            [
                config_path(path, field="judging.personas")
                for path in string_list(raw_personas, field="judging.personas")
            ]
            if raw_personas
            else []
        )
        raw_columns = judging.get("persona_annotation_columns") or []
        annotation_columns = (
            string_list(raw_columns, field="judging.persona_annotation_columns")
            if raw_columns
            else []
        )

    output = judging.get("output", DEFAULTS["output"])
    if output is not None and (not isinstance(output, str) or not output):
        raise ConfigError("judging.output must be null or a path string")
    output_root = _output_root(
        conversations[0], path_from_root(output) if output is not None else None
    )

    return [
        RunConfig(
            invocation=invocation,
            judging=JudgingConfig(
                models=models,
                conversations=conversations,
                rubrics=rubrics,
                output=output_root,
                max_concurrent=judging.get(
                    "max_concurrent", DEFAULTS["max_concurrent"]
                ),
                per_judge=judging.get("per_judge", DEFAULTS["per_judge"]),
                personas=personas,
                persona_annotation_columns=annotation_columns,
            ),
        )
    ]


def _reject_target_all(selection: str) -> str:
    """Reject `all` for judging, which cannot yet attribute its output.

    Judging every target means evaluating the same conversations under N rubrics.
    That resolves cleanly, but every run would land in the same
    `<run>/evaluations/` distinguishable only by timestamp, because the judge run
    folder name encodes the judge model and time, not the rubric. Erroring beats
    writing output nobody can attribute. Lifted in the Traceability phase, whose
    `output/<target>/` root attributes each run (see docs/roadmap.md).
    """
    if selection.casefold() == "all":
        raise ConfigError(
            "judge does not support --target all yet: evaluations for different "
            "rubrics would share one output folder and could not be told apart. "
            "Judge one target at a time."
        )
    return selection


def _rubric_files(target: ResolvedTarget) -> RubricFiles:
    """Project a resolved target onto the three rubric fields judging needs.

    A target carries generation fields too, so judging takes a projection rather
    than the whole bundle. Shared by both routes that start from a target — the
    `--target`/`--rubric` flags and a config's `target` key — which otherwise
    spell the same three-field rename twice.

    `load_target` has already resolved and verified every path, so this only
    renames; it is not a second validation step.
    """
    return RubricFiles(
        rubric_file=target.rubric,
        rubric_prompt_beginning_file=target.rubric_prompt_beginning,
        question_prompt_file=target.question_prompt,
    )


def _target_from_selection(selection: str) -> ResolvedTarget:
    """Resolve a target name or manifest path to its validated target."""
    return load_target(resolve_target_manifest(_reject_target_all(selection)))


def _output_root(conversations: str, output: str | None) -> str:
    """Decide where the evaluation run folder goes.

    Defaults beside the transcripts being judged, at
    `<conversation run>/evaluations/`, so the output records what produced it.

    When the input is not a recognizable generation run — a legacy flat folder of
    `.txt` files — there is nothing to derive from, and `--output` is required. Legacy
    `judge.py` instead wrote to `evaluations/` relative to the working directory;
    that silently detached results from their input and is not carried over. See
    the breaking-change note in CHANGELOG.md.
    """
    if output is not None:
        return str(Path(output).resolve())

    _, generation_run, _ = resolve_conversation_input(conversations)
    if generation_run is None:
        raise ConfigError(
            f"cannot derive an output location from {conversations}: it is not a "
            "generation run folder. Pass --output to say where evaluations "
            "should go."
        )
    return str((Path(generation_run) / "evaluations").resolve())


async def _execute(run_configs: list[RunConfig]) -> None:
    """Hand each resolved run to the judging domain."""
    for run_config in run_configs:
        judging = run_config.judging
        if judging is None:  # pragma: no cover - resolve_configs always sets it
            raise ConfigError("judge produced a run with no judging section")
        into = run_config.invocation.into
        # Exactly one of each, enforced by `JudgingConfig.__post_init__`; the
        # unpacking fails loudly rather than judging only the first if that
        # changes.
        (conversations,) = judging.conversations
        (rubric,) = judging.rubrics

        # Discovery of the transcripts directory is idempotent, so deriving it
        # here keeps the resolved config stating the folder the user named rather
        # than an internal subdirectory.
        transcripts_dir, _, folder_name = resolve_conversation_input(conversations)
        await run_judging(
            judge_models={model.name: model.repeats for model in judging.models},
            rubric_file=rubric.rubric_file,
            rubric_prompt_beginning_file=rubric.rubric_prompt_beginning_file,
            question_prompt_file=rubric.question_prompt_file,
            transcripts_dir=transcripts_dir,
            conversation_folder_name=folder_name,
            limit=run_config.invocation.sample,
            # Without `--into`, a parent to mint a new `j_*` run under; with
            # it, the exact existing run folder to land back in.
            output_dir=into or judging.output,
            is_existing_run=into is not None,
            # Every judge model shares one parameter dict
            # (`JudgingSpec.__post_init__` rejects anything else), so the first
            # model's stands for all of them.
            judge_model_extra_params=dict(judging.models[0].extra_params),
            max_concurrent=judging.max_concurrent,
            per_judge=judging.per_judge,
            verbose_workers=False,
            verbose=True,
            resume=into is not None,
            personas=list(judging.personas),
            persona_annotation_columns=list(judging.persona_annotation_columns),
            # A continued run keeps the record it started with.
            run_record=build_run_record(run_config) if into is None else None,
        )


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register ``judge`` with the root parser.

    Uses the same `argparse.SUPPRESS` convention as `generate`: run-defining
    flags are absent from the namespace unless the user passed them, which is
    what makes the config-or-flags rule enforceable.

    Note `-c` is deliberately *not* accepted. Judging is decoupled from chatbot
    selection by design, and in legacy `judge.py` `-c` meant `--conversation`,
    which this command does not have.
    """
    parser = subparsers.add_parser("judge", help="Evaluate conversations")
    parser.add_argument(
        "-j",
        "--judge",
        nargs="+",
        metavar="<model>[:<instances>]",
        default=argparse.SUPPRESS,
        help="Judge model(s) and how many instances of each to run",
    )
    parser.add_argument(
        "--conversations",
        nargs="+",
        metavar="<folder>",
        default=argparse.SUPPRESS,
        help=(
            "Conversation run folder to judge (exactly one; judge folders "
            "separately and combine with 'vera pool')"
        ),
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--target",
        default=argparse.SUPPRESS,
        help="Complete target name or manifest path supplying the rubric",
    )
    target.add_argument(
        "--rubric",
        default=argparse.SUPPRESS,
        help="Target name or manifest path whose rubric and prompts should be used",
    )
    # See `generate.register`: `--output` is a parent to mint a new run under
    # and is run-defining; `--into` names an existing run to continue and is
    # invocation-only. Exactly one may be given.
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument(
        "-o",
        "--output",
        default=argparse.SUPPRESS,
        help=(
            "Parent directory for the evaluation run folder "
            "(default: <conversation run>/evaluations/)"
        ),
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=argparse.SUPPRESS,
        help="Maximum concurrent judge workers (default: unlimited)",
    )
    parser.add_argument(
        "--per-judge",
        action="store_true",
        default=argparse.SUPPRESS,
        help=(
            "Apply --max-concurrent per judge model rather than across all "
            "(default: across all)"
        ),
    )
    parser.add_argument(
        "--judge-params",
        type=parse_key_value_list,
        default=argparse.SUPPRESS,
        metavar="k=v[,k=v...]",
        help="Provider parameters applied to every -j model (default: none)",
    )
    add_common_flags(
        parser,
        handler=run,
        sample_help="Debug-only cap on conversations judged",
        into_group=destination,
    )
