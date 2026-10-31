"""The ``vera pipeline`` command: generate, judge, score, then pool.

A **shim**, deliberately. It owns no generation, judging, scoring, or pooling
logic and no defaults of its own: it resolves input with the same helpers the
single-stage commands use, then calls `vera_cli.generate._execute`,
`judge.run.run_judging`, `score.run_scoring`, and `score.run_pooling` in order,
passing each stage's output to the next.

A run with more than one `-u` model produces one evaluation per user model,
and the headline number is the pooled score across them, so the pipeline ends
by pooling those evaluations (with one user model there is nothing to pool).
This does not contradict "judge never auto-scores" in `docs/architecture.md`:
that rule keeps `vera judge` non-scoring, while `pipeline` exists to run every
stage. `vera pool` remains its own command for pooling evaluations judged
separately. Until the Traceability phase (`docs/roadmap.md`) builds the
target-rooted layout, the pooled folder is written under the generation
output folder, where `vera pool` writes by default.

That chaining is the entire reason the command exists. Running the three
commands by hand works, but the caller has to copy the generation run folder
into the judge invocation and the `results.csv` path into the score
invocation — which is what the pre-2.0 `run_recommended_vera_pipeline.sh` does by
scraping them out of a log.

`docs/pipeline.md` covers what makes this command's input different from the
single-stage ones: the paths that chain the stages together cannot be known
before the run starts, so a pipeline config states fewer fields than a judge or
score config, not more.

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

## Why judging and scoring are resolved as specs

`JudgingConfig` requires `conversations`, and at resolve time a pipeline has
none — generation has not run yet. This is not a gap in the schema but a
property of the input format: per `docs/vera-cli-use-cases.md`, a pipeline
config *omits* `judging.conversations` precisely because the generation stage
supplies it. `ScoringConfig.results` is the same, one stage later.

So `PipelineRun` below holds a `JudgingSpec` and a `ScoringSpec` — the
base classes of those configs, holding everything but the folders — and
`_judge_and_score` calls `complete` on each once the folder exists. Every
field is validated at resolve time, so a bad rubric, an unknown target or a
malformed `max_concurrent` fails before spending the budget.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Any

from vera_mh.judge import run_judging
from vera_mh.score import run_pooling, run_scoring
from vera_mh.utils.config_schema import (
    InvocationConfig,
    JudgingSpec,
    PoolingConfig,
    RubricFiles,
    RunConfig,
    ScoringSpec,
)
from vera_mh.utils.run_record import build_run_record
from vera_mh.utils.utils import parse_key_value_list

from . import generate as generate_command
from . import judge as judge_command
from .config import (
    ConfigError,
    add_common_flags,
    config_path,
    config_section,
    models_from_cli,
    models_from_config,
    print_resolved_config,
    required,
    resolve_with,
    rubrics_from_config,
    run_command,
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
    `judging` and `scoring` are specs rather than configs for the reason in the
    module docstring — the folders they would name do not exist yet.
    """

    generation: RunConfig
    judging: JudgingSpec
    scoring: ScoringSpec

    def to_dict(self) -> dict[str, Any]:
        """Serialize back into the pipeline config input format.

        `judging.conversations` is absent by design, so this round-trips: the
        output is a valid `vera pipeline --config` document, not a valid
        `vera judge` one.
        """
        return {
            **self.generation.to_dict(),
            "judging": self.judging.to_dict(),
            "scoring": self.scoring.to_dict(),
        }


def run(args: argparse.Namespace) -> int:
    """Resolve the requested pipeline(s) and execute them end to end.

    Debug is per invocation, not per stage: `set_debug` flips a process global
    (`utils/debug.py`) and `debug` lives on `InvocationConfig`, which is one per
    run. Whether per-stage debug is worth having is a Backlog item in
    `docs/roadmap.md`.
    """
    return run_command(
        args,
        command="pipeline",
        resolve=resolve_configs,
        execute=_execute,
        echo=lambda pipeline_run: print_resolved_config(pipeline_run.generation),
        debug_of=lambda pipeline_run: pipeline_run.generation.invocation.debug,
    )


def resolve_configs(args: argparse.Namespace) -> list[PipelineRun]:
    """Resolve either config JSON or CLI flags into canonical pipelines.

    Returns a list because `generate`'s resolver does; `--target all` is
    rejected for now (see `_reject_target_all`), so today it always holds
    exactly one pipeline.
    """
    return resolve_with(
        args,
        allowed_config_fields=ALLOWED_CONFIG_FIELDS,
        label="pipeline",
        from_config=_from_config,
        from_cli=_from_cli,
    )


def _from_cli(
    args: argparse.Namespace, invocation: InvocationConfig
) -> list[PipelineRun]:
    """Resolve CLI flags into canonical pipelines.

    The generation half delegates to `generate`'s own resolver rather than
    re-deriving it, so the two commands cannot drift on defaults or path
    resolution. Only the judging half is read here, and only the three values
    the shorthand carries.
    """
    # Both checks run before `generate._from_cli`, deliberately: that call
    # resolves a whole generation config — target resolution, manifest I/O —
    # and would report its own missing flags first, so a `vera pipeline -c X
    # -u Y` with no `-j` would do all that work and then complain about
    # something else. Checking here also has no "below" that could replace it:
    # `generate` has no concept of judge models, so if this does not check
    # `-j`, nothing does.
    judge_models: list[str] | None = getattr(args, "judge", None)
    if not judge_models:
        raise ConfigError("pipeline requires at least one -j/--judge model")

    target: str | None = getattr(args, "target", None)
    if not target:
        raise ConfigError("pipeline requires --target")
    _reject_target_all(target)

    # `generate._from_cli`, not its `resolve_configs`: the latter re-enters
    # `resolve_input`, which would reject pipeline's extra sections and, with
    # `--config -`, try to read stdin a second time. The generation flags are
    # spelled identically here, so the namespace is the one it expects. It
    # returns a list because it owns `--target all` fan-out; with `all`
    # rejected above, that list always has exactly one element.
    generation_runs = generate_command._from_cli(args, invocation)

    # Resolved once and shared: `--target` names one bundle per run, and for a
    # pipeline the rubric always comes from the same target the personas did.
    judging = JudgingSpec(
        models=models_from_cli(judge_models, getattr(args, "judge_params", None)),
        rubrics=[_rubric_for(target)],
        max_concurrent=judge_command.DEFAULTS["max_concurrent"],
        per_judge=judge_command.DEFAULTS["per_judge"],
    )
    return [
        PipelineRun(
            generation=generation_run,
            judging=judging,
            scoring=ScoringSpec(personas=None, skip_risk_analysis=False),
        )
        for generation_run in generation_runs
    ]


def _from_config(
    config: dict[str, Any], invocation: InvocationConfig
) -> list[PipelineRun]:
    """Resolve a config object into canonical pipelines.

    As with the CLI path, the generation half is handed to `generate`'s own
    resolver. Judging behavior fields take `judge`'s defaults when left out,
    as in every other config-mode command.
    """
    judging = config_section(config, "judging", command="pipeline")
    if "conversations" in judging:
        raise ConfigError(
            "pipeline config must not set judging.conversations: the generation "
            "stage supplies the conversations this run judges"
        )

    target = config.get("target")
    if isinstance(target, str):
        _reject_target_all(target)

    # Same reasoning as the CLI path: the already-loaded config document goes
    # straight to generate's config resolver, never back through `resolve_input`.
    generation_runs = generate_command._from_config(config, invocation)

    judging_spec = JudgingSpec(
        models=models_from_config(
            required(judging, "models", section="judging config"),
            field="judging.models",
        ),
        rubrics=[_rubric_from_config(judging, target)],
        max_concurrent=judging.get(
            "max_concurrent", judge_command.DEFAULTS["max_concurrent"]
        ),
        per_judge=judging.get("per_judge", judge_command.DEFAULTS["per_judge"]),
    )

    scoring = config.get("scoring") or {}
    if not isinstance(scoring, dict):
        raise ConfigError("scoring must be an object")
    personas = scoring.get("personas")
    # `ScoringSpec` validates the type; a `TypeError`/`ValueError` from it is
    # normalized to a `ConfigError` by `resolve_configs`.
    scoring_spec = ScoringSpec(
        personas=(
            config_path(personas, field="scoring.personas")
            if personas is not None
            else None
        ),
        skip_risk_analysis=scoring.get("skip_risk_analysis", False),
    )

    return [
        PipelineRun(
            generation=generation_run,
            judging=judging_spec,
            scoring=scoring_spec,
        )
        for generation_run in generation_runs
    ]


def _reject_target_all(selection: str) -> None:
    """Refuse `--target all` until there is a concrete use for it.

    `vera judge` rejects it too (`judge._reject_target_all`), and a pipeline
    could in principle support it — each target would generate its own
    conversations, so each evaluation would land under its own run folder
    rather than sharing one. But nobody has asked to run every target in one
    invocation, and supporting it means carrying a second rubric-resolution
    path that walks back from a run's persona file to find which target
    produced it. Deferred rather than built on spec; the flag can be widened
    later without breaking anyone.
    """
    if selection.casefold() == "all":
        raise ConfigError(
            "pipeline does not support --target all: run one target at a time."
        )


def _rubric_for(target: str) -> RubricFiles:
    """Resolve a target name or manifest path to the three rubric files.

    The projection is `judge`'s (`judge._rubric_files`) rather than a second
    copy of it — a target carries generation fields too, so judging takes the
    three it needs.
    """
    return judge_command._rubric_files(load_target(resolve_target_manifest(target)))


def _rubric_from_config(judging: dict[str, Any], target: object) -> RubricFiles:
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
    _reject_target_all(target)
    return _rubric_for(target)


async def _execute(runs: list[PipelineRun]) -> None:
    """Run each pipeline's three stages in order, feeding each into the next.

    Sequential for the same reason `generate._execute` is: concurrency caps
    apply within a stage, so overlapping stages would silently multiply a
    caller's cap against the same provider.

    A run with more than one `-u` model produces one generation run, one
    evaluation and one score *per user model*, and the per-model scores are not
    the headline number — the pooled score across them is. So it ends by
    pooling them, per `PipelineRun` and so never across targets, whose rubrics
    differ. The evaluation folders are printed before pooling, so they are known
    even if pooling fails; a failure propagates like any other stage's.
    """
    for pipeline_run in runs:
        run_folders = await generate_command._execute([pipeline_run.generation])
        evaluations = [
            await _judge_and_score(pipeline_run, run_folder)
            for run_folder in run_folders
        ]
        if len(evaluations) > 1:
            print(f"\n{len(evaluations)} evaluations written:")
            for evaluation in evaluations:
                print(f"  {evaluation}")
            generation = pipeline_run.generation.generation
            assert generation is not None  # generate.resolve_configs always sets it
            pooling = PoolingConfig(
                evaluations=evaluations,
                output=generation.output,
                personas=pipeline_run.scoring.personas,
                skip_risk_analysis=pipeline_run.scoring.skip_risk_analysis,
            )
            pooled = run_pooling(
                evaluations=list(pooling.evaluations),
                output_parent=pooling.output,
                personas_tsv=pooling.personas,
                skip_risk_analysis=pooling.skip_risk_analysis,
                run_record=build_run_record(
                    RunConfig(
                        invocation=pipeline_run.generation.invocation,
                        pooling=pooling,
                    )
                ),
            )
            print(f"\nHeadline (pooled) score: {pooled}")


def _persona_annotation(generation_run: RunConfig) -> tuple[list[str], list[str]]:
    """Find the personas file this run used and the columns to annotate with.

    Derived from the run's own persona file rather than from a shared target
    resolution, for the same reason `_rubric_for` does it that way: under
    `--target all` each run must consult the target that produced it.

    A run whose personas sit outside a target bundle has no manifest to read, so
    it annotates nothing -- the personas are still known, but which columns are
    worth comparing is the manifest's statement to make.
    """
    generation = generation_run.generation
    assert generation is not None  # generate.resolve_configs always sets it
    personas = list(generation.personas)
    manifest = Path(personas[0]).parent / "manifest.json"
    if not manifest.is_file():
        return personas, []
    return personas, list(load_target(manifest).persona_annotation_columns)


async def _judge_and_score(pipeline_run: PipelineRun, run_folder: str) -> str:
    """Judge one generated run folder, score it, and say where it landed."""
    personas, annotation_columns = _persona_annotation(pipeline_run.generation)
    judging = dataclasses.replace(
        pipeline_run.judging,
        personas=personas,
        persona_annotation_columns=annotation_columns,
    ).complete(
        conversations=[str(Path(run_folder) / "conversations")],
        # Evaluations land beside the transcripts that produced them, the same
        # default `vera judge` applies when `--output` is omitted.
        output=str(Path(run_folder) / "evaluations"),
    )
    _, evaluation_folder = await run_judging(
        judging,
        sample=pipeline_run.generation.invocation.sample,
        run_record=build_run_record(
            RunConfig(invocation=pipeline_run.generation.invocation, judging=judging)
        ),
    )

    scoring = pipeline_run.scoring.complete(
        results=str(Path(evaluation_folder) / "results.csv"), output=None
    )
    run_scoring(
        results_csv=scoring.results,
        output_json=scoring.output,
        personas_tsv=scoring.personas,
        skip_risk_analysis=scoring.skip_risk_analysis,
    )
    return evaluation_folder


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
            "rubric ('all' is not supported: run one target at a time)"
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
    add_common_flags(
        parser,
        handler=run,
        sample_help="Debug-only cap on personas loaded per file",
    )
