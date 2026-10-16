"""Record real LLM calls once, then replay them so a whole run is deterministic.

`recording_llm_factory` wraps the real clients `LLMFactory.create_llm` returns
and appends every call to a JSON Lines file: the model, its system prompt,
exactly what was sent, and what came back. `replay_llm_factory` stands in for
`LLMFactory.create_llm` and answers each call from that file.

A call is matched on everything that was sent, so a replay only succeeds when
the code sends the models exactly what it sent while recording. A changed
prompt, history, or judge question raises `NotInRecording` naming the call.
See docs/safe-changes.md.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

from llm_clients.llm_factory import LLMFactory
from llm_clients.llm_interface import JudgeLLM, Role

T = TypeVar("T")


class NotInRecording(AssertionError):
    """The code sent a model something it never sent while recording."""


def _what_the_model_sees(conversation_history: Optional[List[Dict[str, Any]]]):
    """The history without each turn's `logging` entry.

    That entry holds timings, token counts, and request IDs, which differ on
    every run and are never sent to the provider.
    """
    if conversation_history is None:
        return None
    return [
        {key: value for key, value in turn.items() if key != "logging"}
        for turn in conversation_history
    ]


def _call_key(model: str, system_prompt: str, method: str, sent: Any) -> str:
    return json.dumps(
        {
            "model": model,
            "system_prompt": system_prompt,
            "method": method,
            "sent": sent,
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def recording_llm_factory(recording: Path) -> Callable[..., Any]:
    """A stand-in for `LLMFactory.create_llm` that records every real call."""
    real_create_llm = LLMFactory.create_llm

    def append(model: str, system_prompt: str, method: str, sent: Any, reply: Any):
        line = {
            "model": model,
            "system_prompt": system_prompt,
            "method": method,
            "sent": sent,
            "reply": reply,
        }
        with open(recording, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    def create_llm(**kwargs: Any) -> Any:
        llm = real_create_llm(**kwargs)
        model = kwargs["model_name"]
        generate_response = llm.generate_response

        async def recorded_generate_response(conversation_history=None):
            reply = await generate_response(conversation_history)
            sent = _what_the_model_sees(conversation_history)
            append(model, llm.system_prompt, "text", sent, reply)
            return reply

        llm.generate_response = recorded_generate_response

        if isinstance(llm, JudgeLLM):
            generate_structured_response = llm.generate_structured_response

            async def recorded_structured_response(message, response_model):
                reply = await generate_structured_response(message, response_model)
                append(
                    model, llm.system_prompt, "structured", message, reply.model_dump()
                )
                return reply

            llm.generate_structured_response = recorded_structured_response
        return llm

    return create_llm


def replay_llm_factory(recording: Path) -> Callable[..., "ReplayLLM"]:
    """A stand-in for `LLMFactory.create_llm` that answers from a recording.

    Identical calls (same model, prompt, and input) are answered in the order
    they were recorded, which is why recording and replay run one job at a time.
    """
    replies: Dict[str, List[Any]] = defaultdict(list)
    for line in recording.read_text(encoding="utf-8").splitlines():
        call = json.loads(line)
        key = _call_key(
            call["model"], call["system_prompt"], call["method"], call["sent"]
        )
        replies[key].append(call["reply"])

    misses: List[str] = []

    def create_llm(**kwargs: Any) -> ReplayLLM:
        return ReplayLLM(replies, misses, **kwargs)

    # Exposed so a test can check that every recorded reply was used: a reply
    # left over means the code made fewer calls than when it was recorded.
    create_llm.unused_replies = replies  # type: ignore[attr-defined]
    # Generation skips a conversation whose call fails, so a miss would otherwise
    # surface later as "no transcripts". Keeping them lets a test report the cause.
    create_llm.misses = misses  # type: ignore[attr-defined]
    return create_llm


class ReplayLLM(JudgeLLM):
    """An LLM client whose every answer comes from a recording."""

    def __init__(
        self,
        replies: Dict[str, List[Any]],
        misses: List[str],
        *,
        model_name: str,
        name: str,
        role: Role,
        system_prompt: Optional[str] = None,
        first_message: Optional[str] = None,
        start_prompt: Optional[str] = None,
        **provider_settings: Any,
    ):
        super().__init__(
            name,
            role,
            system_prompt,
            first_message=first_message,
            start_prompt=start_prompt,
        )
        self.replies = replies
        self.misses = misses
        self.model_name = model_name
        # Generation logs print these; they don't affect what's replayed.
        self.temperature = provider_settings.get("temperature")
        self.max_tokens = provider_settings.get("max_tokens")

    def _next_reply(self, method: str, sent: Any) -> Any:
        key = _call_key(self.model_name, self.system_prompt, method, sent)
        if not self.replies.get(key):
            last_sent = sent[-1] if isinstance(sent, list) and sent else sent
            message = (
                f"{self.model_name} was sent something the recording doesn't have.\n"
                f"Last input: {str(last_sent)[:500]}"
            )
            self.misses.append(message)
            raise NotInRecording(message)
        self._set_response_metadata("replay")
        return self.replies[key].pop(0)

    async def start_conversation(self) -> str:
        if self.first_message is not None:
            return self.first_message
        return await self.generate_response(self.get_initial_prompt_turns())

    async def generate_response(
        self, conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        return self._next_reply("text", _what_the_model_sees(conversation_history))

    async def generate_structured_response(
        self, message: Optional[str], response_model: Type[T]
    ) -> T:
        return response_model.model_validate(self._next_reply("structured", message))  # type: ignore[attr-defined]

    def set_system_prompt(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt
