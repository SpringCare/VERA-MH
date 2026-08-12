"""Canonical run configuration shared by unified CLI input forms."""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    """One model selection and its number of full persona-set repeats."""

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
    """Fully resolved generation values passed to the generation command."""

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
    """Controls that describe how this resolved run is executed."""

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
    """Canonical resolved form of one ``vera generate`` invocation."""

    invocation: InvocationConfig
    generation: GenerationConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "invocation": self.invocation.to_dict(),
            "generation": self.generation.to_dict(),
        }
