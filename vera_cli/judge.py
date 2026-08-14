"""The ``vera judge`` command: flags, resolution, and the workflow call.

Structured exactly like `vera_cli/generate.py`, which is the reference
implementation of the command contract in `vera_cli/README.md`:

1. `register` declares the flags and attaches `run` as the subparser's handler.
2. `run` is the entry point `vera.py` dispatches to.
3. `resolve_configs` picks one input form and produces canonical `RunConfig`s.
4. `_execute` hands each resolved run to the judging domain.

Nothing below step 3 reads an `argparse.Namespace`, and nothing above it touches
the judging domain.

Two deliberate differences from legacy `judge.py`, both recorded in
docs/vera-cli-use-cases.md:

- No `--resume`. The resume contract is deferred, so resuming stays available
  only through legacy `judge.py` until `vera resume` exists.
- No single-conversation mode. Judge a folder containing one conversation.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

from judge import run_judging
from utils.config_schema import (
    InvocationConfig,
    JudgingConfig,
    ModelSpec,
    RubricFiles,
    RunConfig,
)
from utils.conversation_layout import resolve_conversation_input
from utils.debug import set_debug
from utils.utils import parse_key_value_list

from .config import (
    ConfigError,
    models_from_cli,
    models_from_config,
    path_from_root,
    print_resolved_config,
    render_invocation,
    required,
    resolve_input,
)
from .targets import (
    config_path,
    load_target,
    resolve_target_manifest,
    targets_from_config,
)

# CLI behavior defaults, applied during resolution by `_value` rather than by the
# parser — see `vera_cli/generate.py` for why the parser cannot hold them.
#
# `output` has no static default: it is derived from the conversations folder,
# landing beside the transcripts it evaluates. `-h` says so.
DEFAULTS: dict[str, Any] = {
    "output": None,
    "max_concurrent": None,
    "per_judge": False,
}

# Flags that do not define the run. Note `--sample` doubles as the debug cap for
# both commands: it caps personas per file for `generate` and conversations
# loaded for `judge`, so `InvocationConfig` stays uniform across commands rather
# than growing a second per-command cap.
INVOCATION_ONLY_FLAGS = frozenset({"config", "sample", "debug", "print_only"})

# Top-level config keys `judge` accepts. `generation` is absent for the same
# reason `generate` rejects `judging`: a key this command would ignore is worse
# rejected than accepted. A later `pipeline` accepts both.
ALLOWED_CONFIG_FIELDS = {"judging", "target"}


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
    parser.add_argument(
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
    parser.add_argument("--config", help="JSON path or '-' for stdin")
    parser.add_argument(
        "--sample",
        type=int,
        default=argparse.SUPPRESS,
        help="Debug-only cap on conversations judged",
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

    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    """Resolve the requested run(s) and execute them.

    Resolution happens up front and completely, so an invalid target, missing
    rubric file, or underivable output location fails before any model is called.
    """
    run_configs = resolve_configs(args)
    if args.print_only:
        for run_config in run_configs:
            print(render_invocation(run_config, command="judge"))
        return 0

    if any(config.invocation.debug for config in run_configs):
        set_debug(True)
    for run_config in run_configs:
        print_resolved_config(run_config)
    asyncio.run(_execute(run_configs))
    return 0


def resolve_configs(args: argparse.Namespace) -> list[RunConfig]:
    """Resolve either config JSON or CLI flags into canonical runs.

    Always returns exactly one `RunConfig`. Unlike `generate`, `--target all` is
    rejected rather than fanned out — see `_target_selection`.
    """
    try:
        config, invocation = resolve_input(
            args,
            invocation_only_flags=INVOCATION_ONLY_FLAGS,
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
        raise ConfigError(f"invalid judging config: {error}") from error


def _reject_target_all(selection: str) -> str:
    """Reject `all` for judging, which cannot yet attribute its output.

    Judging every target means evaluating the same conversations under N rubrics.
    That resolves cleanly, but every run would land in the same
    `<run>/evaluations/` distinguishable only by timestamp, because the judge run
    folder name encodes the judge model and time, not the rubric. Erroring beats
    writing output nobody can attribute. Lifted in Phase 4, which adds the
    `evaluations/<target>/` segment (see docs/architecture.md).
    """
    if selection.casefold() == "all":
        raise ConfigError(
            "judge does not support --target all yet: evaluations for different "
            "rubrics would share one output folder and could not be told apart. "
            "Judge one target at a time."
        )
    return selection


def _rubric_from_target(selection: str) -> RubricFiles:
    """Resolve a target name or manifest path to its three rubric files."""
    target = load_target(resolve_target_manifest(_reject_target_all(selection)))
    return RubricFiles(
        rubric_file=target.rubric,
        rubric_prompt_beginning_file=target.rubric_prompt_beginning,
        question_prompt_file=target.question_prompt,
    )


def _output_root(conversations: str, output: str | None) -> str:
    """Decide where the evaluation run folder goes.

    Defaults beside the transcripts being judged, at
    `<conversation run>/evaluations/`, so the output records what produced it.

    When the input is not a recognizable generation run — a legacy flat folder of
    `.txt` files — there is nothing to derive from, and `-o` is required. Legacy
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
            "generation run folder. Pass -o/--output to say where evaluations "
            "should go."
        )
    return str((Path(generation_run) / "evaluations").resolve())


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
    if target:
        rubric_files = _rubric_from_target(target)
    elif rubric:
        rubric_files = _rubric_from_target(rubric)
    else:
        raise ConfigError("judge requires --target or --rubric")

    folders = [str(Path(folder).resolve()) for folder in conversations]
    return [
        _run_config(
            invocation,
            models=models_from_cli(models, getattr(args, "judge_params", None)),
            conversations=folders,
            rubrics=[rubric_files],
            output=_output_root(folders[0], _value(args, "output")),
            max_concurrent=_value(args, "max_concurrent"),
            per_judge=_value(args, "per_judge"),
        )
    ]


def _from_config(
    config: dict[str, Any], invocation: InvocationConfig
) -> list[RunConfig]:
    """Resolve a config object into one canonical run.

    Every behavior field is required rather than defaulted, matching `generate`:
    a stored config is a complete, reproducible description of a run, so a value
    it does not state is an error rather than something this code fills in.
    """
    value = config.get("judging")
    if not isinstance(value, dict):
        raise ConfigError("judge requires a judging config object")
    judging = dict(value)
    models = models_from_config(
        required(judging, "models", section="judging config"),
        field="judging.models",
    )
    conversations = [
        config_dir(folder, field="judging.conversations")
        for folder in _string_list(
            required(judging, "conversations", section="judging config"),
            field="judging.conversations",
        )
    ]

    targets = targets_from_config(
        config,
        judging,
        explicit_fields=("rubrics",),
        section_name="judging",
    )
    if targets is not None:
        if len(targets) != 1:
            raise ConfigError(
                "judge does not support target 'all' yet: evaluations for "
                "different rubrics would share one output folder"
            )
        rubrics = [
            RubricFiles(
                rubric_file=targets[0].rubric,
                rubric_prompt_beginning_file=targets[0].rubric_prompt_beginning,
                question_prompt_file=targets[0].question_prompt,
            )
        ]
    else:
        rubrics = _rubrics_from_config(
            required(judging, "rubrics", section="judging config")
        )

    output = required(judging, "output", section="judging config")
    if not isinstance(output, str) or not output:
        raise ConfigError("judging.output must be a path string")

    return [
        _run_config(
            invocation,
            models=models,
            conversations=conversations,
            rubrics=rubrics,
            output=path_from_root(output),
            max_concurrent=required(
                judging, "max_concurrent", section="judging config"
            ),
            per_judge=required(judging, "per_judge", section="judging config"),
        )
    ]


def _string_list(value: Any, *, field: str) -> list[str]:
    """Validate a config value is a non-empty list of non-empty strings."""
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise ConfigError(f"{field} must be a non-empty list of paths")
    return value


def config_dir(value: str, *, field: str) -> str:
    """Resolve a config-supplied directory against the repository root.

    The path-list helpers in `targets` verify *files*; a conversations folder is
    a directory, so it gets its own check.
    """
    resolved = Path(path_from_root(value))
    if not resolved.is_dir():
        raise ConfigError(f"{field} does not exist or is not a directory: {resolved}")
    return str(resolved)


def _rubrics_from_config(value: Any) -> list[RubricFiles]:
    """Build `RubricFiles` from explicit config entries, resolving each path."""
    if not isinstance(value, list) or not value:
        raise ConfigError("judging.rubrics must be a non-empty list of objects")
    rubrics = []
    for entry in value:
        if not isinstance(entry, dict):
            raise ConfigError("judging.rubrics entries must be objects")
        files = RubricFiles.from_dict(entry)
        rubrics.append(
            RubricFiles(
                **{
                    field: config_path(
                        getattr(files, field), field=f"judging.rubrics.{field}"
                    )
                    for field in (
                        "rubric_file",
                        "rubric_prompt_beginning_file",
                        "question_prompt_file",
                    )
                }
            )
        )
    return rubrics


def _value(args: argparse.Namespace, field: str) -> Any:
    """Read a run-defining flag, falling back to its CLI default."""
    return getattr(args, field, DEFAULTS[field])


def _run_config(
    invocation: InvocationConfig,
    *,
    models: list[ModelSpec],
    conversations: list[str],
    rubrics: list[RubricFiles],
    output: str,
    max_concurrent: int | None,
    per_judge: bool,
) -> RunConfig:
    """Assemble and validate one canonical `RunConfig` holding a judging section.

    Fields are named rather than forwarded as opaque keywords so this signature
    states what a judging run consists of. Type enforcement happens at runtime in
    `JudgingConfig.__post_init__`.
    """
    return RunConfig(
        invocation=invocation,
        judging=JudgingConfig(
            models=models,
            conversations=conversations,
            rubrics=rubrics,
            output=output,
            max_concurrent=max_concurrent,
            per_judge=per_judge,
        ),
    )


async def _execute(run_configs: list[RunConfig]) -> None:
    """Hand each resolved run to the judging domain."""
    for run_config in run_configs:
        judging = run_config.judging
        if judging is None:  # pragma: no cover - resolve_configs always sets it
            raise ConfigError("judge produced a run with no judging section")
        rubric = judging.rubrics[0]

        # Discovery of the transcripts directory is idempotent, so deriving it
        # here keeps the resolved config stating the folder the user named rather
        # than an internal subdirectory.
        transcripts_dir, _, folder_name = resolve_conversation_input(
            judging.conversations[0]
        )
        await run_judging(
            judge_models={model.name: model.repeats for model in judging.models},
            rubric_file=rubric.rubric_file,
            rubric_prompt_beginning_file=rubric.rubric_prompt_beginning_file,
            question_prompt_file=rubric.question_prompt_file,
            transcripts_dir=transcripts_dir,
            conversation_folder_name=folder_name,
            limit=run_config.invocation.sample,
            output_root=judging.output,
            output_folder=None,
            judge_model_extra_params=dict(judging.models[0].extra_params),
            max_concurrent=judging.max_concurrent,
            per_judge=judging.per_judge,
            verbose_workers=False,
            verbose=True,
            resume=False,
        )
