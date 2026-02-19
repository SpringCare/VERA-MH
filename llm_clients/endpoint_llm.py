import time
from typing import Any, Dict, List, Optional

import aiohttp

from utils.conversation_utils import build_langchain_messages

from .config import Config
from .llm_interface import LLMInterface, Role


class EndpointLLM(LLMInterface):
    """Chat-only LLM that calls a custom POST /api/chat endpoint.

    The API manages conversation history server-side via conversation_id.
    This implementation does not support structured output and cannot be used
    as a judge. For judge operations, use Claude, OpenAI, Gemini, or Azure.

    System prompt: This class accepts system_prompt (from LLMInterface) for
    interface consistency and as an example for subclasses. By default we do
    not send it to the endpoint as custom APIs typically manage system context
    themselves. To apply it (e.g. prefix first user message with
    \"System: ...\"), override generate_response or _build_body in a subclass.
    """

    def __init__(
        self,
        name: str,
        role: Role,
        system_prompt: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        first_message = kwargs.pop("first_message", None)
        start_prompt = kwargs.pop("start_prompt", None)
        super().__init__(
            name,
            role,
            system_prompt,
            first_message=first_message,
            start_prompt=start_prompt,
        )

        cfg = Config.get_endpoint_config()
        self._api_key = api_key or cfg["api_key"]
        self._base_url = base_url or cfg["base_url"]
        self._start_url = cfg.get("start_url", None)

        # NOTE: if start_url is set, we don't need to use the start_prompt
        # unless the developer wants to utilize it
        if self._start_url is not None:
            self.start_prompt = None

        if model_name and model_name.lower().startswith("endpoint-"):
            self._api_model = model_name[len("endpoint-") :].strip() or cfg["model"]
        else:
            self._api_model = cfg["model"]
        self.model_name = model_name or "endpoint"
        self.temperature = kwargs.pop("temperature", None)
        self.max_tokens = kwargs.pop("max_tokens", None)

    def __getattr__(self, name):
        """Delegate to self.llm when present; else return self's attribute or None.

        Only uses __dict__ lookups to avoid recursion. Attributes like
        temperature and max_tokens are on self; unknown names return None.
        """
        if "llm" in self.__dict__ and hasattr(self.__dict__["llm"], name):
            return getattr(self.__dict__["llm"], name)
        if name in self.__dict__:
            return self.__dict__[name]
        return None

    async def start_conversation(self) -> str:
        """Produce the first conversational turn:
        - static first_message if set, or
        - API call to start_url if set, or
        - API call to /api/chat with start_prompt if neither is set.
        """
        if self.first_message is not None:
            self._set_response_metadata("endpoint", static_first_message=True)
            return self.first_message
        elif self._start_url is not None:
            start_time = time.time()
            resp_data = await self._ainvoke(self._start_url, self.start_prompt)
            return self._process_chat_response(
                resp_data, round(time.time() - start_time, 3)
            )
        else:
            return await self.generate_response(self.get_initial_prompt_turns())

    def _default_headers(self) -> Dict[str, str]:
        """Default request headers (API key and content type)."""
        return {
            "X-API-Key": self._api_key,
            "Content-Type": "application/json",
        }

    def _process_chat_response(
        self, resp_data: Dict[str, Any], response_time_seconds: float
    ) -> str:
        """Extract message text from API response and set metadata. Return content."""
        msg_data = resp_data.get("message") or {}
        msg_text: str = msg_data.get("content", "")

        usage = {}
        if resp_data.get("prompt_eval_count") is not None:
            usage["prompt_tokens"] = resp_data.get("prompt_eval_count", 0)
        if resp_data.get("eval_count") is not None:
            usage["completion_tokens"] = resp_data.get("eval_count", 0)
        if usage:
            usage.setdefault("prompt_tokens", 0)
            usage.setdefault("completion_tokens", 0)
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

        self._set_response_metadata(
            "endpoint",
            model=resp_data.get("model", self._api_model),
            response_id=msg_data.get("id"),
            usage=usage,
            conversation_id=resp_data.get("conversation_id"),
            response_time_seconds=response_time_seconds,
            total_duration=resp_data.get("total_duration"),
            load_duration=resp_data.get("load_duration"),
            prompt_eval_count=resp_data.get("prompt_eval_count"),
            prompt_eval_duration=resp_data.get("prompt_eval_duration"),
            eval_count=resp_data.get("eval_count"),
            eval_duration=resp_data.get("eval_duration"),
        )
        self._update_conversation_id_from_metadata()
        return msg_text

    def _build_body(self, content: str) -> Dict[str, Any]:
        """Body: model, messages (user content only), stream, conversation_id.
        System prompt is not included; see class docstring.
        """
        return {
            "model": self._api_model,
            "messages": [{"role": "user", "content": content}],
            "stream": False,
            "conversation_id": self.conversation_id,
        }

    async def _ainvoke(
        self,
        url: str,
        content: str,
        *,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """POST to url with body built from content; return parsed JSON.
        Body: model, messages (single user message), stream=False, conversation_id.
        Default headers when headers is None. Raises RuntimeError on non-200.
        """
        req_headers = headers if headers is not None else self._default_headers()
        body = self._build_body(content)
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=req_headers, json=body) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"Endpoint returned {resp.status}: {text[:500]}")
                return await resp.json()

    async def generate_response(
        self,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Generate a response via POST /api/chat with server-side conversation_id.

        Only the latest user content is sent; self.system_prompt is not included
        in the request (see class docstring for rationale).
        """
        if not conversation_history or len(conversation_history) == 0:
            return await self.start_conversation()

        messages = build_langchain_messages(self.role, conversation_history)
        last_message = messages[-1].text  # no system_prompt in payload by design

        try:
            start_time = time.time()
            resp_data = await self._ainvoke(self._base_url, last_message)
            return self._process_chat_response(
                resp_data, round(time.time() - start_time, 3)
            )
        except Exception as e:
            self._set_response_metadata("endpoint", error=str(e))
            self._update_conversation_id_from_metadata()
            return f"Error generating response: {str(e)}"

    def set_system_prompt(self, system_prompt: str) -> None:
        """Set or update the system prompt."""
        self.system_prompt = system_prompt
