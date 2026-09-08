"""Engine that speaks the OpenAI-compatible HTTP API shape.

Drives every provider that implements `POST /v1/chat/completions`
with `Authorization: Bearer <key>`:

  - OpenAI itself (`https://api.openai.com`)
  - xAI (`https://api.x.ai`)
  - OpenRouter, Together, Groq, Fireworks, DeepInfra, MiniMax, …
  - Local OpenAI-compatible servers (Ollama, vLLM, LM Studio,
    llama.cpp's server)

This is a *transport* — the user-facing "provider" picker decides
which `Provider` from the registry is in play, and the registry
hands this engine a `default_base_url`, an optional
`deny_pattern`, and the `backend_kind` to report. New providers
that share this protocol = a new entry in `providers.py`, not a
new engine file.

Auth: `Authorization: Bearer <api_key>` header. The API key is
read from config (`apiKey`, sensitive) with a fallback to the
`OPENAI_API_KEY` env var so existing setups still work.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import AsyncIterator
from typing import Any

import httpx

from .._generated.models import (
    BackendKind,
    ConfigField,
    ConfigFieldShowWhen,
    ConfigValueType,
    FinishReason,
    GenerateRequest,
    GenerateResponse,
    Role,
    Usage,
)
from ._subprocess import CliError
from ._thinking import apply_thinking_mode, strip_thinking_blocks

log = logging.getLogger(__name__)

# Default deny pattern for OpenAI proper (`provider: openai` →
# OpenAiCompatibleHttpEngine + this pattern). Catches every gpt-5 family
# member that uses either `-` or `.` after the family name, plus the
# o-series, while explicitly NOT matching hypothetical `gpt-50` /
# `gpt-5o` style names that aren't actually 5.x. Other providers can
# pass their own pattern (or None) via the registry.
OPENAI_DENY_PATTERN: re.Pattern[str] = re.compile(
    r"^(?:o\d+|gpt-5)(?:[-.]|$)",
    re.IGNORECASE,
)

# OpenAI's `/v1/models` returns every model on the account — embeddings,
# image, audio, moderation, retired completion-only models, etc. — and
# the API doesn't tag them by capability. Heuristic-filter to chat-likely
# IDs so the dropdown isn't 80 entries of `text-embedding-3-large`.
_NON_CHAT_PREFIXES = (
    "text-embedding-",
    "text-similarity-",
    "text-search-",
    "code-search-",
    "dall-e-",
    "whisper-",
    "tts-",
    "omni-moderation-",
    "babbage",
    "davinci",
    "ada-",
    "curie-",
    "computer-use-",
    "codex-",  # the standalone Codex models — different surface from chatgpt_subscription
)

_CHAT_MODEL_PREFIXES = (
    "gpt-",
    "chatgpt-",
    "claude-",
    "llama",
    "mistral",
    "qwen",
    "deepseek",
    "grok",
    "abab",  # MiniMax
)


def _is_plausible_chat_model(model_id: str) -> bool:
    lowered = model_id.lower()
    if lowered.startswith(_NON_CHAT_PREFIXES):
        return False
    if "embedding" in lowered or "moderation" in lowered:
        return False
    if "audio" in lowered or "realtime" in lowered or "transcribe" in lowered or "tts" in lowered:
        return False
    if "image" in lowered or "vision-preview" in lowered:
        # Pure image-gen models. Vision-capable chat models (e.g. gpt-4o)
        # don't carry "image" in the id, so they're unaffected.
        return False
    return lowered.startswith(_CHAT_MODEL_PREFIXES)


_FINISH_REASON_MAP = {
    "stop": FinishReason.stop,
    "length": FinishReason.length,
    "content_filter": FinishReason.error,
    "tool_calls": FinishReason.stop,
    "function_call": FinishReason.stop,
}


def _max_tokens_field_for(base_url: str) -> str:
    """Pick the right output-cap field name for this base URL.

    OpenAI's chat-completions API now requires `max_completion_tokens`
    for newer models and explicitly rejects `max_tokens`. Self-hosted
    OpenAI-compatible servers (Ollama, vLLM, LM Studio, llama.cpp)
    still implement the older spec and only understand `max_tokens`.
    Pick by base URL: openai.com → new field; anything else → legacy.
    """
    return "max_completion_tokens" if "openai.com" in base_url.lower() else "max_tokens"


class OpenAiCompatibleHttpEngine:
    """OpenAI-compatible HTTP engine. Provider-agnostic."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com",
        model_id: str = "gpt-4o",
        timeout_seconds: float = 120.0,
        deny_pattern: re.Pattern[str] | None = None,
        backend_kind: BackendKind = BackendKind.openai_api,
        thinking_mode: str = "auto",
        auth_required: bool = True,
        filter_models: bool = True,
    ) -> None:
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if auth_required and not resolved_key:
            raise CliError(
                "openai_compat_http engine has no API key — set `apiKey` in "
                "config or export OPENAI_API_KEY in the environment."
            )
        if deny_pattern is not None and deny_pattern.match(model_id):
            raise CliError(
                f"This provider rejects model {model_id!r} because it doesn't "
                "support a tunable `temperature` parameter (e.g. OpenAI's "
                "o-series rejects temperature outright; the gpt-5 family "
                "schema-accepts it but only the default value of 1). Eugene "
                "Plexus uses temperature as the per-pass divergence knob "
                "between hemispheres, and v0.2's NT system modulates it "
                "per-driver — a model that won't move with temperature has "
                "no place in the bicameral loop. Pick a non-reasoning chat "
                "model with controllable temperature from the dropdown."
            )
        self._api_key = resolved_key
        self._base_url = base_url.rstrip("/")
        self._model_id = model_id
        self._timeout_seconds = timeout_seconds
        self._deny_pattern = deny_pattern
        self.backend_kind = backend_kind
        self._thinking_mode = thinking_mode or "auto"
        self._filter_models = filter_models

    @classmethod
    def field_specs(cls, *, applicable_providers: list[str]) -> list[ConfigField]:
        """Config fields this engine reads. The schema builder shows
        each one only when one of `applicable_providers` is selected."""
        show_when = ConfigFieldShowWhen(key="provider", equals=applicable_providers)
        return [
            ConfigField(
                key="apiKey",
                label="API Key",
                description=(
                    "Secret key sent as the `Authorization: Bearer ...` "
                    "header on every API call. Get one from the provider's "
                    "console (OpenAI, xAI, OpenRouter, MiniMax, your "
                    "self-hosted server, etc.). If left blank, the driver "
                    "falls back to the `OPENAI_API_KEY` environment variable "
                    "in the shell that started it. Stored on disk in plain "
                    "text in v0.1; at-rest encryption is on the v0.2 list."
                ),
                category="adapter",
                valueType=ConfigValueType.secret,
                sensitive=True,
                requiresRestart=True,
                showWhen=show_when,
            ),
        ]

    @classmethod
    def from_config(
        cls,
        get: Any,
        *,
        default_base_url: str | None,
        deny_pattern: re.Pattern[str] | None,
        backend_kind: BackendKind,
        auth_required: bool = True,
        filter_models: bool = True,
    ) -> OpenAiCompatibleHttpEngine:
        # User's `baseUrl` wins (set only for openai_compat_custom);
        # otherwise the provider's default applies.
        base_url = str(get("baseUrl") or default_base_url or "").strip()
        if not base_url:
            raise CliError(
                "OpenAI-compatible engine has no base URL. For the custom "
                "provider, set `baseUrl` in config. For named providers, "
                "this is a registry bug — file an issue."
            )
        return cls(
            api_key=str(get("apiKey") or "") or None,
            base_url=base_url,
            model_id=str(get("modelId") or "gpt-4o"),
            timeout_seconds=float(get("requestTimeoutSeconds") or 120),
            deny_pattern=deny_pattern,
            backend_kind=backend_kind,
            thinking_mode=str(get("thinkingMode") or "auto"),
            auth_required=auth_required,
            filter_models=filter_models,
        )

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        # Apply the operator's thinkingMode by mutating the system
        # message before role-coercion. See engines/_thinking.py for
        # the per-mode directives — `off` is the one that suppresses
        # inline `<think>` blocks leaking into chat responses.
        messages = apply_thinking_mode(list(request.messages), self._thinking_mode)
        payload: dict[str, Any] = {
            "model": self._model_id,
            "messages": _to_openai_messages(messages),
        }
        if request.maxTokens is not None:
            payload[_max_tokens_field_for(self._base_url)] = request.maxTokens
        if request.temperature is not None:
            payload["temperature"] = float(request.temperature)
        if request.stop:
            payload["stop"] = list(request.stop)

        # DEBUG-level full-payload trace. The gateway's copy-trace
        # captures what we sent it; this captures what WE send upstream
        # (post role-coercion, post-thinking-directive injection,
        # post-param-shaping). When operators flip to DEBUG to chase
        # "is the LLM actually seeing what I think it's seeing", this
        # is the load-bearing log line. Auth header omitted on purpose.
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "openai_compat_http → POST %s/v1/chat/completions\n%s",
                self._base_url,
                json.dumps(payload, indent=2, ensure_ascii=False),
            )

        async with httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(self._timeout_seconds, connect=10.0),
        ) as client:
            try:
                response = await client.post(
                    "/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Accept": "application/json",
                    },
                    json=payload,
                )
            except httpx.HTTPError as e:
                raise CliError(f"openai_compat_http request failed: {e}") from e

        if response.status_code >= 400:
            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    "openai_compat_http ← HTTP %d (%dms) body:\n%s",
                    response.status_code,
                    int(response.elapsed.total_seconds() * 1000),
                    _redact(response.text[:4000]),
                )
            raise CliError(
                f"openai_compat_http returned {response.status_code}: "
                f"{_redact(response.text[:500])}"
            )

        try:
            body = response.json()
        except ValueError as e:
            raise CliError(f"openai_compat_http returned non-JSON: {response.text[:200]!r}") from e

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "openai_compat_http ← HTTP %d (%dms) body:\n%s",
                response.status_code,
                int(response.elapsed.total_seconds() * 1000),
                json.dumps(body, indent=2, ensure_ascii=False),
            )

        choices = body.get("choices") or []
        if not choices:
            raise CliError(f"openai_compat_http returned no choices: {body!r}")

        first = choices[0] or {}
        message = first.get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            raise CliError(f"openai_compat_http response missing string content: {body!r}")
        # Defensive strip of <think>...</think> when the operator opted
        # out of thinking but the model emitted tags anyway. See
        # _thinking.strip_thinking_blocks for the why.
        if self._thinking_mode == "off":
            content = strip_thinking_blocks(content)

        return GenerateResponse(
            content=content,
            finishReason=_FINISH_REASON_MAP.get(
                str(first.get("finish_reason") or "stop"), FinishReason.stop
            ),
            usage=_usage_from_envelope(body.get("usage") or {}),
            requestId=request.requestId,
            backend=self.backend_kind,
            modelId=str(body.get("model") or self._model_id),
            latencyMs=int(response.elapsed.total_seconds() * 1000),
        )

    async def stream(self, request: GenerateRequest) -> AsyncIterator[object]:
        # SSE streaming via stream=true is supported by the wire shape
        # but no consumer of inference-driver streaming exists yet.
        raise NotImplementedError("OpenAiCompatibleHttpEngine.stream not implemented in v0.1")
        yield  # pragma: no cover

    async def list_models(self) -> list[str]:
        """Live-fetch the model catalog from the configured base URL.

        OpenAI proper and most OpenAI-compatible providers expose
        `GET /v1/models` returning `{"data": [{"id": "<model>", ...}]}`.
        We filter out:

        - non-chat models (embeddings, dall-e, whisper, tts, moderation,
          codex-only, audio) — heuristic by id prefix / substring
        - models the configured `deny_pattern` would refuse at construction

        Returns `[]` on transport / parse failure so the schema endpoint
        can fall back to free-text input. Driver stays up either way.
        """
        headers: dict[str, str] = {"Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            async with httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(15.0, connect=5.0),
            ) as client:
                response = await client.get("/v1/models", headers=headers)
            if response.status_code >= 400:
                return []
            body = response.json()
        except (httpx.HTTPError, ValueError):
            return []
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list):
            return []
        ids: list[str] = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            mid = entry.get("id")
            if not isinstance(mid, str):
                continue
            # Local providers (Ollama, LM Studio) list ONLY what the
            # operator explicitly pulled — model ids include publisher
            # prefixes like `huihui_ai/dolphin3-abliterated:8b-...` that
            # the chat-prefix heuristic rejects. Skip filtering for
            # those; trust the operator's choice. The deny-pattern still
            # applies because temperature compatibility is a wire-level
            # concern, not a discovery one.
            if self._filter_models and not _is_plausible_chat_model(mid):
                continue
            if self._deny_pattern is not None and self._deny_pattern.match(mid):
                continue
            ids.append(mid)
        ids.sort()
        return ids


def _to_openai_messages(messages: list[Any]) -> list[dict[str, str]]:
    """Map our Message[] to OpenAI chat-completions messages."""
    out: list[dict[str, str]] = []
    for m in messages:
        if m.role == Role.system:
            out.append({"role": "system", "content": m.content})
        elif m.role == Role.user:
            out.append({"role": "user", "content": m.content})
        elif m.role == Role.assistant or m.role == Role.hemisphere:
            out.append({"role": "assistant", "content": m.content})
        else:
            out.append({"role": "user", "content": m.content})
    return out


def _usage_from_envelope(usage: dict[str, Any]) -> Usage | None:
    if not usage:
        return None
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    total = usage.get("total_tokens")
    if prompt is None and completion is None:
        return None
    return Usage(
        promptTokens=prompt or 0,
        completionTokens=completion or 0,
        totalTokens=total if total is not None else (prompt or 0) + (completion or 0),
    )


def _redact(text: str) -> str:
    """Best-effort redaction of API keys that might appear in error bodies."""
    return text.replace("Bearer ", "Bearer <redacted>")
