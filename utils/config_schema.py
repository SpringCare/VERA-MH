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

    Contrast `GenerationConfig`, which defines the run itself.
    """

    debug: bool
    sample: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.debug, bool):
            raise ValueError("invocation.debug must be a boolean")
        if self.sample is not None:
            if isinstance(self.sample, bool) or not isinstance(self.sample, int):
                raise ValueError("invocation.sample must be null or an integer")
            if self.sample < 1:
                raise ValueError("--sample must be at least 1")

    def to_dict(self) -> dict[str, Any]:
        return {"debug": self.debug, "sample": self.sample}


@dataclasses.dataclass(frozen=True)
class RunConfig:
    """One fully resolved `vera generate` run, ready to execute.

    `--target all` resolves to one `RunConfig` per target; every other input
    resolves to exactly one.
    """

    invocation: InvocationConfig
    generation: GenerationConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "invocation": self.invocation.to_dict(),
            "generation": self.generation.to_dict(),
        }
