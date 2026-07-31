"""Centralized `config.json` schema for `vera.py`.

Single source of truth for the run-config shape described in
docs/vera-cli-use-cases.md ("config.json shape") and docs/architecture.md
("Rubric bundle manifest"). CLI flags and `--config` both resolve into a
`RunConfig` here so there is exactly one canonical representation of "what
this run does," regardless of which input form produced it.

Per docs/architecture.md's "Stable interfaces" section, this file's schema
is a stable interface once Phase 3 formalizes it — see the ESCALATE section
there before changing the shape of `RunConfig`/`GenerationConfig`/
`JudgingConfig`. Phase 1 (this file, as first written) is explicitly the
*informal* shape.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional


@dataclasses.dataclass
class ModelSpec:
    """One model entry in a `generation.user`/`generation.chatbot`/`judging.models`.

    `name` is always a specific model identifier (e.g. "claude-sonnet-2026xxxx"),
    never a bare provider name. `extra_params` holds bespoke sampling knobs
    (temperature, top_p, max_tokens, ...) -- config-only, never expressible via
    `-u`/`-c`/`-j` shorthand.
    """

    name: str
    repeats: int = 1
    extra_params: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("model name cannot be empty")
        if isinstance(self.repeats, bool) or not isinstance(self.repeats, int):
            raise ValueError("model repeats must be an integer")
        if self.repeats < 1:
            raise ValueError("model repeats must be at least 1")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "repeats": self.repeats, **self.extra_params}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelSpec":
        data = dict(data)
        name = data.pop("name")
        repeats = data.pop("repeats", 1)
        return cls(name=name, repeats=repeats, extra_params=data)

    @classmethod
    def from_shorthand(cls, token: str) -> "ModelSpec":
        """Parse `-u`/`-c`/`-j` shorthand: "<model>[:<repeats>]"."""
        name, sep, repeats_str = token.rpartition(":")
        if not sep or not repeats_str.isdigit():
            name = token
            repeats_str = "1"
        if not name:
            raise ValueError(f"invalid model shorthand: {token!r}")
        repeats = int(repeats_str)
        if repeats < 1:
            raise ValueError(f"model repeats must be at least 1: {token!r}")
        return cls(name=name, repeats=repeats)


@dataclasses.dataclass
class RubricSpec:
    """One entry in `judging.rubrics[]`.

    `name` resolves to a rubric bundle manifest (see
    docs/architecture.md#rubric-bundle-manifest) via `--target`/manifest
    lookup, not a bare `.tsv` path. `models` optionally overrides
    `judging.models` for this rubric only.
    """

    name: str
    models: list[ModelSpec] = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name}
        if self.models:
            d["models"] = [m.to_dict() for m in self.models]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RubricSpec":
        return cls(
            name=data["name"],
            models=[ModelSpec.from_dict(m) for m in data.get("models", [])],
        )


@dataclasses.dataclass
class GenerationConfig:
    """`generation` block. Orthogonal to `JudgingConfig` -- see RunConfig."""

    chatbot: Optional[ModelSpec] = None
    user: list[ModelSpec] = dataclasses.field(default_factory=list)
    personas: list[str] = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.chatbot is not None:
            d["chatbot"] = self.chatbot.to_dict()
        if self.user:
            d["user"] = [m.to_dict() for m in self.user]
        if self.personas:
            d["personas"] = self.personas
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GenerationConfig":
        chatbot = data.get("chatbot")
        return cls(
            chatbot=ModelSpec.from_dict(chatbot) if chatbot else None,
            user=[ModelSpec.from_dict(m) for m in data.get("user", [])],
            personas=list(data.get("personas", [])),
        )


@dataclasses.dataclass
class JudgingConfig:
    """`judging` block. Orthogonal to `GenerationConfig` -- see RunConfig.

    `rubrics` is a list from day one (per docs/architecture.md's migration
    Phase 0-4 notes) even though only a length-1 list is supported/validated
    until Phase 4.
    """

    models: list[ModelSpec] = dataclasses.field(default_factory=list)
    rubrics: list[RubricSpec] = dataclasses.field(default_factory=list)
    conversations: list[str] = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.models:
            d["models"] = [m.to_dict() for m in self.models]
        if self.rubrics:
            d["rubrics"] = [r.to_dict() for r in self.rubrics]
        if self.conversations:
            d["conversations"] = self.conversations
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JudgingConfig":
        return cls(
            models=[ModelSpec.from_dict(m) for m in data.get("models", [])],
            rubrics=[RubricSpec.from_dict(r) for r in data.get("rubrics", [])],
            conversations=list(data.get("conversations", [])),
        )


@dataclasses.dataclass
class RunConfig:
    """The canonical resolved form of a `vera.py` invocation.

    Whether a run was invoked via CLI shorthand or `--config`, it always
    resolves to exactly one `RunConfig` -- printed at run start for
    terminal/CI-log visibility (docs/vera-cli-use-cases.md#config-mechanism).

    `target` mirrors the `--target <name>` shorthand: set only when the
    invocation used `--target` (or the input config's own top-level `target`
    field) instead of independently specifying `generation.personas` and
    `judging.rubrics`. Setting `target` alongside explicit
    `generation.personas`/`judging.rubrics` is an error -- see
    `docs/architecture.md#rubric-bundle-manifest`.
    """

    generation: Optional[GenerationConfig] = None
    judging: Optional[JudgingConfig] = None
    target: Optional[str] = None

    def __post_init__(self) -> None:
        if self.target is not None and self.generation and self.generation.personas:
            raise ValueError(
                "target is mutually exclusive with generation.personas "
                "-- target expands to both personas and rubrics itself"
            )
        if self.target is not None and self.judging and self.judging.rubrics:
            raise ValueError(
                "target is mutually exclusive with judging.rubrics "
                "-- target expands to both personas and rubrics itself"
            )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.generation is not None:
            d["generation"] = self.generation.to_dict()
        if self.judging is not None:
            d["judging"] = self.judging.to_dict()
        if self.target is not None:
            d["target"] = self.target
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunConfig":
        generation = data.get("generation")
        judging = data.get("judging")
        return cls(
            generation=GenerationConfig.from_dict(generation) if generation else None,
            judging=JudgingConfig.from_dict(judging) if judging else None,
            target=data.get("target"),
        )


@dataclasses.dataclass
class RubricBundleManifest:
    """A rubric bundle manifest (docs/architecture.md#rubric-bundle-manifest).

    Paths (`rubric_file`, `rubric_prompt_beginning_file`, `question_prompt_file`,
    entries in `personas`) resolve relative to the manifest's own folder --
    never relative to `$ROOT` or the CLI's working directory. `personas` is
    informational-only except when resolved via `--target`, per the
    docs/architecture.md `--target` note.
    """

    rubric_file: str
    rubric_prompt_beginning_file: str
    question_prompt_file: str
    personas: list[str] = dataclasses.field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RubricBundleManifest":
        return cls(
            rubric_file=data["rubric_file"],
            rubric_prompt_beginning_file=data["rubric_prompt_beginning_file"],
            question_prompt_file=data["question_prompt_file"],
            personas=list(data.get("personas", [])),
        )
