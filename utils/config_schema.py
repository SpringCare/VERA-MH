"""The canonical resolved form of one unified-CLI run.

`vera` accepts a run in two input forms — CLI flags, or a JSON config file —
and they are strictly either/or for a single run. Both forms converge on the
types in this module before anything executes, so the rest of the codebase has
exactly one shape to read.

What "resolved" means here: every value is final. Names are looked up, relative
paths are absolute, target manifests are expanded into concrete persona files,
and defaults are already applied. Nothing downstream re-interprets a field.

This module validates and serializes that form and does nothing else. It does
not parse arguments, read files, resolve paths, or define defaults — CLI
behavior defaults live beside the flags in `vera_cli/generate.py`, and
config-driven runs must state every behavior field explicitly. `to_dict` is the
inverse of the config input format, which is what makes a resolved run
round-trippable: `--print` emits a config that reproduces the same run.

The two halves of `RunConfig` split along one axis — whether a value defines
*which run this is*. See `GenerationConfig` and `InvocationConfig`.
"""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    """One model selection and its number of full persona-set repeats.

    `repeats` is how many times the entire persona set runs against this model,
    not a per-persona count. `extra_params` holds provider-specific request
    parameters (temperature, thinking budget, ...) passed through untouched.

    This is the single model type for every entity in the CLI vocabulary —
    user, chatbot, and later judge — so all three commands describe models the
    same way. `from_shorthand` parses the CLI form (`gpt-5:3`), `from_dict` the
    config form; both produce this.
    """

    name: str
    repeats: int
    extra_params: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("model name cannot be empty")
        if isinstance(self.repeats, bool) or not isinstance(self.repeats, int):
            raise ValueError("model repeats must be an integer")
        if self.repeats < 1:
            raise ValueError("model repeats must be at least 1")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelSpec":
        values = dict(data)
        try:
            name = values.pop("name")
            repeats = values.pop("repeats")
        except KeyError as error:
            raise ValueError(
                f"model is missing required field: {error.args[0]}"
            ) from error
        if not isinstance(name, str):
            raise ValueError("model name must be a string")
        return cls(name=name, repeats=repeats, extra_params=values)

    @classmethod
    def from_shorthand(cls, token: str) -> "ModelSpec":
        name, separator, repeat_text = token.rpartition(":")
        if not separator or not repeat_text.isdigit():
            name = token
            repeat_text = "1"
        return cls(name=name, repeats=int(repeat_text), extra_params={})

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "repeats": self.repeats, **self.extra_params}


@dataclasses.dataclass(frozen=True)
class RubricFiles:
    """The three resolved files that make up one rubric.

    They travel together because a rubric is not usable without all three, and
    they are named as files rather than as a manifest path because this is the
    resolved form — nothing downstream re-reads a manifest to find them.

    "Resolved" is enforced by construction rather than by convention. There is
    deliberately no `from_dict`: a config entry's paths are repo-relative and
    unverified, so building an instance straight from one would put this type in
    the state its own name rules out. Callers check the entry's shape with
    `validate_dict`, resolve the paths, and only then instantiate — see
    `vera_cli.targets.rubrics_from_config`, the one place that happens.
    """

    rubric_file: str
    rubric_prompt_beginning_file: str
    question_prompt_file: str

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"judging.rubrics {field.name} must be a path")

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        """The three field names, in declaration order."""
        return tuple(field.name for field in dataclasses.fields(cls))

    @classmethod
    def validate_dict(cls, data: dict[str, Any]) -> None:
        """Check a mapping has exactly this rubric's fields, nothing more.

        A classmethod rather than free-standing validation in the caller so the
        field list stays derived from the dataclass: adding a fourth rubric file
        cannot leave a hand-written check behind.
        """
        names = set(cls.field_names())
        missing = sorted(names.difference(data))
        if missing:
            raise ValueError(
                f"rubric is missing required field(s): {', '.join(missing)}"
            )
        unknown = sorted(set(data).difference(names))
        if unknown:
            raise ValueError(f"rubric has unknown field(s): {', '.join(unknown)}")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class GenerationConfig:
    """What run to perform — the run-defining half of a `RunConfig`.

    Every field here is part of the run's identity: change one and it is a
    different run producing different output. Because these values define the
    run, they must come from exactly one input form — supplying any of them as
    a CLI flag alongside `--config` is an error, not a merge (see
    `vera_cli.config.resolve_input`).

    Contrast `InvocationConfig`, which describes how a single invocation is
    executed and is not part of run identity.
    """

    chatbot: ModelSpec
    user: list[ModelSpec]
    personas: list[str]
    turns: int
    output: str
    max_concurrent: int | None
    max_total_words: int | None
    persona_speaks_first: bool
    sessions: list[str] | None
    persona_context_template: str

    def __post_init__(self) -> None:
        if self.chatbot.repeats != 1:
            raise ValueError("generation.chatbot repeats must be 1")
        if not self.user:
            raise ValueError("generation.user must contain at least one model")
        if not self.personas:
            raise ValueError("generation.personas must contain at least one file")
        if isinstance(self.turns, bool) or not isinstance(self.turns, int):
            raise ValueError("generation.turns must be an integer")
        if self.turns < 1:
            raise ValueError("generation.turns must be at least 1")
        if not self.output:
            raise ValueError("generation.output cannot be empty")
        if self.max_concurrent is not None:
            if isinstance(self.max_concurrent, bool) or not isinstance(
                self.max_concurrent, int
            ):
                raise ValueError("generation.max_concurrent must be null or an integer")
            if self.max_concurrent < 0:
                raise ValueError(
                    "generation.max_concurrent must be null, 0, or a positive integer"
                )
        if self.max_total_words is not None:
            if isinstance(self.max_total_words, bool) or not isinstance(
                self.max_total_words, int
            ):
                raise ValueError(
                    "generation.max_total_words must be null or an integer"
                )
            if self.max_total_words < 1:
                raise ValueError("generation.max_total_words must be at least 1")
        if not isinstance(self.persona_speaks_first, bool):
            raise ValueError("generation.persona_speaks_first must be a boolean")
        if self.sessions is not None:
            if not self.sessions:
                raise ValueError("generation.sessions cannot be empty")
            if not all(
                isinstance(session, str) and session for session in self.sessions
            ):
                raise ValueError(
                    "generation.sessions entries must be non-empty strings"
                )
        if not self.persona_context_template:
            raise ValueError("generation.persona_context_template is required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "chatbot": self.chatbot.to_dict(),
            "user": [model.to_dict() for model in self.user],
            "personas": list(self.personas),
            "turns": self.turns,
            "output": self.output,
            "max_concurrent": self.max_concurrent,
            "max_total_words": self.max_total_words,
            "persona_speaks_first": self.persona_speaks_first,
            "sessions": list(self.sessions) if self.sessions is not None else None,
            "persona_context_template": self.persona_context_template,
        }


@dataclasses.dataclass(frozen=True)
class InvocationConfig:
    """How this one invocation is executed — the non-run-defining half.

    These controls change what you observe while a run executes, not which run
    it is: `debug` adds logging, `sample` caps personas loaded per file for
    quick smoke checks. Because they are not part of run identity, they are the
    only fields that may accompany `--config` on the command line — you can
    replay a stored run with `--debug` without editing the config.

    `into` belongs here for the same reason, though it is the least obvious of
    the three. It names an existing run folder to continue, skipping work whose
    output is already on disk. That is a statement about *this execution*, not
    about which run it is: a run completed in one go and the same run finished
    across two invocations are the same run, which is the whole point of being
    able to continue one. Making it run-defining would also mean every stored
    config had to state it, and a config naming a folder would be usable exactly
    once — the second time, the folder is already complete.

    `into` is *not* `vera resume`. It is the stateless per-stage skip the legacy
    scripts spelled `--resume`, re-deriving progress from the files already
    written. `vera resume` reads `state.json` after verifying `config.json`,
    works across stages, and is deferred to a later phase (AD-18/23/24).

    Contrast `GenerationConfig`, which defines the run itself.
    """

    debug: bool
    sample: int | None
    into: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.debug, bool):
            raise ValueError("invocation.debug must be a boolean")
        if self.sample is not None:
            if isinstance(self.sample, bool) or not isinstance(self.sample, int):
                raise ValueError("invocation.sample must be null or an integer")
            if self.sample < 1:
                raise ValueError("--sample must be at least 1")
        if self.into is not None and (not isinstance(self.into, str) or not self.into):
            raise ValueError("invocation.into must be null or a run folder path")

    def to_dict(self) -> dict[str, Any]:
        return {"debug": self.debug, "into": self.into, "sample": self.sample}


@dataclasses.dataclass(frozen=True)
class JudgingConfig:
    """What judging run to perform — the run-defining half for `vera judge`.

    The counterpart of `GenerationConfig`, and subject to the same rule: every
    field here is part of the run's identity, so all of them must come from one
    input form.

    `rubrics` is list-shaped from day one per AD-20 while only length 1 is
    accepted, so lifting the multi-rubric restriction later is not a schema
    break. Each entry holds the three resolved rubric files rather than a
    manifest path, because the resolved form names concrete files.
    `conversations` is list-shaped for the same reason and likewise length 1.
    """

    models: list[ModelSpec]
    conversations: list[str]
    rubrics: list[RubricFiles]
    output: str
    max_concurrent: int | None
    per_judge: bool

    def __post_init__(self) -> None:
        if not self.models:
            raise ValueError("judging.models must contain at least one model")
        names = [model.name for model in self.models]
        if len(set(names)) != len(names):
            raise ValueError(
                "judging.models must not repeat a model name; use repeats to run "
                "several instances of the same model"
            )
        # The judging domain takes one provider-parameter dict for the whole run,
        # so per-model parameters cannot be honored yet. Reject them rather than
        # silently applying one model's parameters to all. Drop this check when
        # the domain accepts per-model parameters.
        distinct_params = {
            tuple(sorted(model.extra_params.items())) for model in self.models
        }
        if len(distinct_params) > 1:
            raise ValueError(
                "judging.models must all use the same provider parameters; "
                "per-model judge parameters are not supported yet"
            )
        if len(self.conversations) != 1:
            raise ValueError(
                "judging.conversations must contain exactly one folder; judge "
                "each folder separately and combine the results with vera pool"
            )
        if not all(isinstance(folder, str) and folder for folder in self.conversations):
            raise ValueError("judging.conversations entries must be non-empty paths")
        if len(self.rubrics) != 1:
            raise ValueError(
                "judging.rubrics must contain exactly one rubric; multi-rubric "
                "support is not implemented yet"
            )
        if not self.output:
            raise ValueError("judging.output cannot be empty")
        if self.max_concurrent is not None:
            if isinstance(self.max_concurrent, bool) or not isinstance(
                self.max_concurrent, int
            ):
                raise ValueError("judging.max_concurrent must be null or an integer")
            if self.max_concurrent < 0:
                raise ValueError(
                    "judging.max_concurrent must be null, 0, or a positive integer"
                )
        if not isinstance(self.per_judge, bool):
            raise ValueError("judging.per_judge must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "models": [model.to_dict() for model in self.models],
            "conversations": list(self.conversations),
            "rubrics": [rubric.to_dict() for rubric in self.rubrics],
            "output": self.output,
            "max_concurrent": self.max_concurrent,
            "per_judge": self.per_judge,
        }


@dataclasses.dataclass(frozen=True)
class RunConfig:
    """One fully resolved `vera` run, ready to execute.

    Holds one section per command taking part in the run. `generate` populates
    `generation`, `judge` populates `judging`, and a later `pipeline` populates
    both; at least one is required.

    `invocation` is a different kind of thing from those two, and the split is
    about lifetime rather than which command owns it. `generation`/`judging`
    describe **what run to perform** — the reproducible, identity-defining
    values that get hashed into the run id. `invocation` describes **how this
    particular execution behaved** (`debug`, `sample`). Two runs with identical
    `generation` sections are the same run even if one was invoked with
    `--debug`, which is why that flag cannot live in the section.

    Both are still persisted into the run's `config.json`, because how a run
    executed is part of its record (AD-17 in `docs/ARCHITECTURE-SPINE.md`).
    `--sample` is the one flag that both alters behavior and is persisted, and
    it is named there as a deliberate exception rather than a precedent.

    `--target all` resolves to one `RunConfig` per target; every other input
    resolves to exactly one.
    """

    invocation: InvocationConfig
    generation: GenerationConfig | None = None
    judging: JudgingConfig | None = None

    def __post_init__(self) -> None:
        if self.generation is None and self.judging is None:
            raise ValueError("a run must define generation, judging, or both")

    def to_dict(self) -> dict[str, Any]:
        """Serialize, omitting sections this run does not define.

        Absent sections are left out rather than emitted as null. That keeps the
        output a valid input config: a command rejects top-level fields it does
        not own, so a `null` judging section would make `vera generate --print`
        emit something `vera generate` itself refuses.
        """
        config: dict[str, Any] = {"invocation": self.invocation.to_dict()}
        if self.generation is not None:
            config["generation"] = self.generation.to_dict()
        if self.judging is not None:
            config["judging"] = self.judging.to_dict()
        return config
