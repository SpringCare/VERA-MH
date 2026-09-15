"""The ``vera pipeline`` command: generate, then judge, then score.

A **shim**, deliberately. It owns no generation, judging, or scoring logic and
no defaults of its own: it resolves input with the same helpers the three
single-stage commands use, then calls `vera_cli.generate._execute`,
`judge.run.run_judging`, and `judge.score.run_scoring` in order, passing each
stage's output to the next.

It is exactly those three stages and no more. `docs/architecture.md` defines
this command as "Full workflow for one chatbot; passes paths between steps", and
lists `vera pool` as its own subcommand owned by `score.pool` — the same doc
states the principle directly: "Judge never auto-scores — `vera score`/`vera
pool` are separate subcommands." An earlier revision of this module pooled
implicitly whenever a run used more than one `-u` model, which contradicted
that and quietly produced an artifact the naming scheme had no home for.
Pooling the per-user-model evaluations into the headline score is therefore a
second, explicit command; `_execute` prints the evaluation folders so it needs
no log-scraping.

That chaining is the entire reason the command exists. Running the three
commands by hand works, but the caller has to copy the generation run folder
into the judge invocation and the `results.csv` path into the score
invocation — which is what `scripts/run_recommended_vera_pipeline.sh` used to
do by scraping them out of a log.

## Why the CLI shorthand is only `-c`, `-u`, `-j`, `--target`

`generate` and `judge` both define `--target`, `-o/--output`, and
`--max-concurrent` with different meanings per stage, so a flat union of their
flags would be ambiguous — `--max-concurrent 10` would have no way to say which
stage it throttles. Legacy `run_pipeline.py` solved this by prefixing one
side (`--judge-max-concurrent`, `--judge-limit`, ...), which doubles the flag
surface and invents a second spelling for every judging flag.

`docs/vera-cli-use-cases.md` chose the other answer, and this follows it: the
shorthand covers only the flags with one unambiguous meaning across the whole
pipeline — the three model roles and the target — and anything stage-specific
comes from `--config`. A caller who needs per-stage knobs is already writing a
config; a caller who does not gets a four-flag command line.

## Why judging is resolved in two halves

`JudgingConfig` requires `conversations`, and at resolve time a pipeline has
none — generation has not run yet. This is not a gap in the schema but a
property of the input format: per `docs/vera-cli-use-cases.md`, a pipeline
config *omits* `judging.conversations` precisely because the generation stage
supplies it.

So `PipelineRun` below holds the generation `RunConfig` plus the judging and
scoring inputs that are knowable up front, and the `JudgingConfig` is built
per generated run folder inside `_execute`, once the folder exists. Everything
resolvable before any model is called still is, so a bad rubric or an unknown
target fails before spending the budget.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import shlex
from pathlib import Path
from typing import Any

from judge import run_judging
from judge.score import run_scoring
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

from . import generate as generate_command
from . import judge as judge_command
from .config import (
    VERA_RUN_CONFIG_ENV,
    ConfigError,
    config_path,
    models_from_cli,
    models_from_config,
    print_resolved_config,
    required,
    resolve_input,
    rubrics_from_config,
)
from .targets import load_target, resolve_target_manifest

# Top-level config keys `pipeline` accepts. It is the one command that owns
# more than one stage, so unlike `generate` and `judge` it accepts every
# section. `scoring` is optional: with no section, scoring runs with no
# personas file, which skips the risk-level breakdown (see `vera_cli/score.py`).
ALLOWED_CONFIG_FIELDS = {"generation", "judging", "scoring", "target"}


@dataclasses.dataclass(frozen=True)
class PipelineRun:
    """One resolved pipeline: generation now, judging and scoring afterwards.

    `generation` is a complete `RunConfig` because that stage can be fully
    described up front and is handed straight to `generate`'s own executor.
    The judging fields are loose rather than a `JudgingConfig` for the reason
    in the module docstring — the conversations folder does not exist yet.
    """

    generation: RunConfig
    judge_models: list[ModelSpec]
    rubric: RubricFiles
    judge_max_concurrent: int | None
    per_judge: bool
    scoring_personas: str | None
    skip_risk_analysis: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize back into the pipeline config input format.

        `judging.conversations` is absent by design, so this round-trips: the
        output is a valid `vera pipeline --config` document, not a valid
        `vera judge` one.
        """
        config = self.generation.to_dict()
        config["judging"] = {
            "models": [model.to_dict() for model in self.judge_models],
            "rubrics": [self.rubric.to_dict()],
            "max_concurrent": self.judge_max_concurrent,
            "per_judge": self.per_judge,
        }
        config["scoring"] = {
            "personas": self.scoring_personas,
            "skip_risk_analysis": self.skip_risk_analysis,
        }
        return config


def run(args: argparse.Namespace) -> int:
    """Resolve the requested pipeline(s) and execute them end to end."""
    runs = resolve_configs(args)
    if args.print_only:
        for pipeline_run in runs:
            print(_render(pipeline_run))
        return 0

    if any(run.generation.invocation.debug for run in runs):
        set_debug(True)
    for pipeline_run in runs:
        print_resolved_config(pipeline_run.generation)
    asyncio.run(_execute(runs))
    return 0


def resolve_configs(args: argparse.Namespace) -> list[PipelineRun]:
    """Resolve either config JSON or CLI flags into canonical pipelines.

    `--target all` fans out here exactly as it does for `generate`, producing
    one pipeline per target. That is safe in a way `vera judge --target all` is
    not: each target gets its own generation run, so its evaluations land under
    that run rather than sharing one folder with nothing to tell them apart.
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
        raise ConfigError(f"invalid pipeline config: {error}") from error


def _from_cli(
    args: argparse.Namespace, invocation: InvocationConfig
) -> list[PipelineRun]:
    """Resolve CLI flags into canonical pipelines.

    The generation half delegates to `generate`'s own resolver rather than
    re-deriving it, so the two commands cannot drift on defaults, path
    resolution, or `--target all` expansion. Only the judging half is read
    here, and only the three values the shorthand carries.
    """
    judge_models: list[str] | None = getattr(args, "judge", None)
    if not judge_models:
        raise ConfigError("pipeline requires at least one -j/--judge model")

    target: str | None = getattr(args, "target", None)
    if not target:
        raise ConfigError("pipeline requires --target")

    # `generate._from_cli`, not its `resolve_configs`: the latter re-enters
    # `resolve_input`, which would reject pipeline's extra sections and, with
    # `--config -`, try to read stdin a second time. The generation flags are
    # spelled identically here, so the namespace is the one it expects. It also
    # owns `--target all` fan-out, which is why this returns one `RunConfig`
    # per target rather than exactly one.
    generation_runs = generate_command._from_cli(args, invocation)

    # Resolved once and shared: `--target` names one bundle per run, and for a
    # pipeline the rubric always comes from the same target the personas did.
    return [
        PipelineRun(
            generation=generation_run,
            judge_models=models_from_cli(
                judge_models, getattr(args, "judge_params", None)
            ),
            rubric=_rubric_for(generation_run, target),
            judge_max_concurrent=judge_command.DEFAULTS["max_concurrent"],
            per_judge=judge_command.DEFAULTS["per_judge"],
            scoring_personas=None,
            skip_risk_analysis=False,
        )
        for generation_run in generation_runs
    ]


def _from_config(
    config: dict[str, Any], invocation: InvocationConfig
) -> list[PipelineRun]:
    """Resolve a config object into canonical pipelines.

    As with the CLI path, the generation half is handed to `generate`'s own
    resolver. Judging behavior fields are required rather than defaulted,
    matching every other config-mode command: a stored config is a complete
    description of a run.
    """
    value = config.get("judging")
    if not isinstance(value, dict):
        raise ConfigError("pipeline requires a judging config object")
    judging = dict(value)
    if "conversations" in judging:
        raise ConfigError(
            "pipeline config must not set judging.conversations: the generation "
            "stage supplies the conversations this run judges"
        )

    # Same reasoning as the CLI path: the already-loaded config document goes
    # straight to generate's config resolver, never back through `resolve_input`.
    generation_runs = generate_command._from_config(config, invocation)

    models = models_from_config(
        required(judging, "models", section="judging config"),
        field="judging.models",
    )
    max_concurrent = required(judging, "max_concurrent", section="judging config")
    per_judge = required(judging, "per_judge", section="judging config")

    scoring = config.get("scoring") or {}
    if not isinstance(scoring, dict):
        raise ConfigError("scoring must be an object")
    personas = scoring.get("personas")
    skip_risk = scoring.get("skip_risk_analysis", False)
    if not isinstance(skip_risk, bool):
        raise ConfigError("scoring.skip_risk_analysis must be a boolean")

    target = config.get("target")
    return [
        PipelineRun(
            generation=generation_run,
            judge_models=models,
            rubric=_rubric_from_config(judging, target, generation_run),
            judge_max_concurrent=max_concurrent,
            per_judge=per_judge,
            scoring_personas=(
                config_path(personas, field="scoring.personas")
                if personas is not None
                else None
            ),
            skip_risk_analysis=skip_risk,
        )
        for generation_run in generation_runs
    ]


def _rubric_for(generation_run: RunConfig, target: str) -> RubricFiles:
    """Resolve the rubric for one generated run from its own target.

    With `--target all`, each run must take the rubric from *its* target, not
    from a single shared resolution — otherwise every target's conversations
    would be judged against whichever manifest happened to resolve first. The
    run's persona file identifies which target produced it.
    """
    if target.casefold() != "all":
        resolved = load_target(resolve_target_manifest(target))
    else:
        generation = generation_run.generation
        assert generation is not None  # generate.resolve_configs always sets it
        manifest = Path(generation.personas[0]).parent / "manifest.json"
        resolved = load_target(manifest)
    return RubricFiles(
        rubric_file=resolved.rubric,
        rubric_prompt_beginning_file=resolved.rubric_prompt_beginning,
        question_prompt_file=resolved.question_prompt,
    )


def _rubric_from_config(
    judging: dict[str, Any], target: object, generation_run: RunConfig
) -> RubricFiles:
    """Take the rubric from an explicit `judging.rubrics` entry, or the target.

    Mirrors `vera judge`: a top-level `target` and an explicit `rubrics` list
    are mutually exclusive, because a target already determines the rubric.
    """
    explicit = "rubrics" in judging
    if explicit and target is not None:
        raise ConfigError(
            "target is mutually exclusive with explicit judging fields: rubrics"
        )
    if explicit:
        rubrics = rubrics_from_config(judging["rubrics"])
        if len(rubrics) != 1:
            raise ConfigError("judging.rubrics must contain exactly one rubric")
        return rubrics[0]
    if not isinstance(target, str) or not target:
        raise ConfigError("pipeline requires a target or explicit judging.rubrics")
    return _rubric_for(generation_run, target)


async def _execute(runs: list[PipelineRun]) -> None:
    """Run each pipeline's three stages in order, feeding each into the next.

    Sequential for the same reason `generate._execute` is: concurrency caps
    apply within a stage, so overlapping stages would silently multiply a
    caller's cap against the same provider.

    A run with more than one `-u` model produces one generation run, one
    evaluation and one score *per user model*, and the per-model scores are not
    the headline number — the pooled score across them is. Pooling is a
    separate command by design (see the module docstring), so this ends by
    naming the evaluation folders it produced, which is what that command needs
    as input.
    """
    for pipeline_run in runs:
        run_folders = await generate_command._execute([pipeline_run.generation])
        evaluations = [
            await _judge_and_score(pipeline_run, run_folder)
            for run_folder in run_folders
        ]
        if len(evaluations) > 1:
            print(
                f"\n{len(evaluations)} evaluations written. The headline score "
                "pools them; these are the folders to pool:"
            )
            for evaluation in evaluations:
                print(f"  {evaluation}")


async def _judge_and_score(pipeline_run: PipelineRun, run_folder: str) -> str:
    """Judge one generated run folder, score it, and say where it landed."""
    conversations = str(Path(run_folder) / "conversations")
    judging = JudgingConfig(
        models=pipeline_run.judge_models,
        conversations=[conversations],
        rubrics=[pipeline_run.rubric],
        # Evaluations land beside the transcripts that produced them, the same
        # default `vera judge` applies when `--output` is omitted.
        output=str(Path(run_folder) / "evaluations"),
        max_concurrent=pipeline_run.judge_max_concurrent,
        per_judge=pipeline_run.per_judge,
    )
    transcripts_dir, _, folder_name = resolve_conversation_input(
        judging.conversations[0]
    )
    _, evaluation_folder = await run_judging(
        judge_models={model.name: model.repeats for model in judging.models},
        rubric_file=judging.rubrics[0].rubric_file,
        rubric_prompt_beginning_file=judging.rubrics[0].rubric_prompt_beginning_file,
        question_prompt_file=judging.rubrics[0].question_prompt_file,
        transcripts_dir=transcripts_dir,
        conversation_folder_name=folder_name,
        limit=pipeline_run.generation.invocation.sample,
        output_dir=judging.output,
        is_existing_run=False,
        judge_model_extra_params=dict(judging.models[0].extra_params),
        max_concurrent=judging.max_concurrent,
        per_judge=judging.per_judge,
        verbose_workers=False,
        verbose=True,
        resume=False,
    )

    run_scoring(
        results_csv=str(Path(evaluation_folder) / "results.csv"),
        output_json=None,
        personas_tsv=pipeline_run.scoring_personas,
        skip_risk_analysis=pipeline_run.skip_risk_analysis,
    )
    return evaluation_folder


def _render(pipeline_run: PipelineRun) -> str:
    """Render a resolved pipeline as a command that reproduces it."""
    compact = json.dumps(pipeline_run.to_dict(), sort_keys=True, separators=(",", ":"))
    return (
        f"{VERA_RUN_CONFIG_ENV}={shlex.quote(compact)} uv run python vera.py pipeline"
    )


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register ``pipeline`` with the root parser.

    The generation flags are spelled exactly as `generate` spells them, because
    `_from_cli` hands the same namespace to `generate.resolve_configs`. `-j` and
    `--judge-params` are spelled exactly as `judge` spells them, for symmetry.

    Stage-specific knobs are deliberately absent — see the module docstring.
    """
    parser = subparsers.add_parser(
        "pipeline", help="Generate, judge, and score in one run"
    )
    parser.add_argument(
        "-c",
        "--chatbot",
        default=argparse.SUPPRESS,
        help="Chatbot model under test",
    )
    parser.add_argument(
        "-u",
        "--user",
        nargs="+",
        metavar="<model>[:<repeats>]",
        default=argparse.SUPPRESS,
        help="User-side model(s) and full persona-set repeats",
    )
    parser.add_argument(
        "-j",
        "--judge",
        nargs="+",
        metavar="<model>[:<instances>]",
        default=argparse.SUPPRESS,
        help="Judge model(s) and how many instances of each to run",
    )
    parser.add_argument(
        "--target",
        default=argparse.SUPPRESS,
        help=(
            "Complete target name or manifest path supplying both personas and "
            "rubric; use 'all' to run every target"
        ),
    )
    parser.add_argument(
        "--user-params",
        type=parse_key_value_list,
        default=argparse.SUPPRESS,
        metavar="k=v[,k=v...]",
        help="Provider parameters applied to every -u model (default: none)",
    )
    parser.add_argument(
        "--chatbot-params",
        type=parse_key_value_list,
        default=argparse.SUPPRESS,
        metavar="k=v[,k=v...]",
        help="Provider parameters applied to the -c model (default: none)",
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
        help="Debug-only cap on personas loaded per file",
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
