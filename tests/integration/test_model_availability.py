"""Live checks that the model IDs in ``Config`` still exist at the provider.

Why this exists: a provider retirement is silent until something calls the
retired ID, and then it surfaces as a generic transport or API error several
layers down a generation run. PR #218 spent its length tracing exactly that --
``claude-opus-4-1-20250805`` had retired, and the failure reached the test
output as a bare ``RuntimeError: Generate CLI failed:``.

These tests fail with the dead ID named instead. They are cheap (one GET per
provider, no generation) but they are ``live``, so they are excluded from the
default CI run and need real keys.

Deliberately not covered:

- **Azure.** ``DEFAULT_AZURE_MODEL`` names a *deployment*, which is specific to
  the resource the key points at, not a catalog entry. There is no portable
  listing to check it against.
- **Ollama / custom endpoints.** Locally or privately hosted; availability is a
  property of the host, not of a provider catalog.

Model IDs are queried rather than asserted against a hardcoded list on purpose:
a hardcoded list is another pin to rot, which is the problem this file exists
to catch.
"""

import json
import re
import urllib.error
import urllib.request
from typing import List, Optional, Set

import pytest

from llm_clients.config import Config

# One GET against a models endpoint; generous enough for a proxied gateway.
_TIMEOUT_SECONDS = 30

pytestmark = [pytest.mark.live, pytest.mark.enable_socket]


def _redact(url: str) -> str:
    """Mask the value of a ``key`` query parameter.

    Google authenticates by query parameter rather than header, so the raw URL
    carries a live credential. Everything this module reports on failure goes
    through here first, because pytest output lands in CI logs.
    """
    return re.sub(r"([?&]key=)[^&]*", r"\1REDACTED", url)


def _get_json(url: str, headers: dict) -> dict:
    """GET ``url`` and parse the JSON body, failing the test on an HTTP error.

    Provider errors are reported with the body included: a 401 from a bad key
    and a 404 from a retired model need to be told apart, and the status alone
    does not do that.
    """
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        pytest.fail(
            f"GET {_redact(url)} failed: HTTP {exc.code} {exc.reason}\n"
            f"--- body ---\n{body}\n"
            "A 401/403 here means the key is wrong or lacks permission, and a "
            "404 on a gateway means it does not expose this provider's "
            "catalog -- both are environment problems rather than a retired "
            "model."
        )
    except urllib.error.URLError as exc:
        pytest.skip(
            f"Cannot reach {_redact(url)} ({exc.reason}); treating as an "
            "environment problem rather than a missing model."
        )


def _resolves(model_id: str, available: Set[str]) -> bool:
    """Whether ``model_id`` names something in ``available``.

    An exact hit is the normal case. The prefix fallback covers the split
    between an undated alias (``claude-sonnet-5``) and the dated snapshots a
    catalog may list instead (``claude-sonnet-5-20260115``), and Google's
    ``-preview`` suffixing. Without it this test would report a live model as
    retired purely because the catalog names it more specifically.
    """
    if model_id in available:
        return True
    return any(listed.startswith(f"{model_id}-") for listed in available)


def _assert_resolves(model_id: str, available: Set[str], provider: str) -> None:
    assert _resolves(model_id, available), (
        f"{provider} model {model_id!r} is configured in llm_clients/config.py "
        f"but the provider does not list it -- it is most likely retired or "
        f"renamed. Update the corresponding Config.DEFAULT_*_MODEL.\n"
        f"Closest listed IDs: {sorted(_closest(model_id, available))[:10]}"
    )


def _closest(model_id: str, available: Set[str]) -> List[str]:
    """Listed IDs sharing a leading token with ``model_id``, to aid the fix."""
    family = model_id.split("-")[0]
    return [listed for listed in available if listed.startswith(family)] or sorted(
        available
    )[:10]


def _require_key(key: Optional[str], env_var: str) -> str:
    if not key:
        pytest.skip(f"{env_var} is not set")
    return key


def _openai_style_catalog(base: str, key: str) -> Set[str]:
    """Model IDs from an OpenAI-shaped ``/v1/models`` listing."""
    payload = _get_json(f"{base}/v1/models", {"Authorization": f"Bearer {key}"})
    return {entry["id"] for entry in payload.get("data", [])}


def _catalog(provider: str, key: str) -> Set[str]:
    """Model IDs available to this project for ``provider``.

    When a base URL override is set, traffic goes through a gateway (LiteLLM),
    and *that* catalog is the one that matters: a model the provider still
    serves is unusable here if the gateway in front of it does not expose it.
    A gateway also presents one OpenAI-compatible listing for every provider it
    fronts, so the provider-native endpoints below are only correct when
    calling the provider directly.
    """
    override = Config.get_base_url(provider)
    if override:
        return _openai_style_catalog(override, key)

    if provider == "anthropic":
        payload = _get_json(
            "https://api.anthropic.com/v1/models?limit=1000",
            {"x-api-key": key, "anthropic-version": "2023-06-01"},
        )
        return {entry["id"] for entry in payload.get("data", [])}

    if provider == "openai":
        return _openai_style_catalog("https://api.openai.com", key)

    if provider == "google":
        # Google takes the key as a query parameter, so the URL must never be
        # echoed -- see _redact, which _get_json applies to everything it
        # reports.
        payload = _get_json(
            "https://generativelanguage.googleapis.com/v1beta/models"
            f"?key={key}&pageSize=1000",
            {},
        )
        # Google returns fully qualified names ("models/gemini-3-pro").
        return {
            entry["name"].removeprefix("models/") for entry in payload.get("models", [])
        }

    raise AssertionError(f"No catalog endpoint known for provider {provider!r}")


class TestConfiguredModelsExist:
    """Each Config default resolves against its provider's catalog."""

    def test_claude_default_model_exists(self):
        key = _require_key(Config.ANTHROPIC_API_KEY, "ANTHROPIC_API_KEY")
        available = _catalog("anthropic", key)
        assert available, "Anthropic returned an empty model catalog"
        _assert_resolves(Config.DEFAULT_CLAUDE_MODEL, available, "Anthropic")

    def test_openai_default_model_exists(self):
        key = _require_key(Config.OPENAI_API_KEY, "OPENAI_API_KEY")
        available = _catalog("openai", key)
        assert available, "OpenAI returned an empty model catalog"
        _assert_resolves(Config.DEFAULT_OPENAI_MODEL, available, "OpenAI")

    def test_gemini_default_model_exists(self):
        key = _require_key(Config.GOOGLE_API_KEY, "GOOGLE_API_KEY")
        available = _catalog("google", key)
        assert available, "Google returned an empty model catalog"
        _assert_resolves(Config.DEFAULT_GEMINI_MODEL, available, "Google")
