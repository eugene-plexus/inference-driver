"""Strata's OpenAI HTTP endpoint: text chat, tool calls and structured output.

Reuse transport, incremental reasoning, usage and cancellation. Do not infer
llama.cpp semantics from Strata's compatibility /props or /slots endpoints.

**Tools are assumed, as for every engine** (Troy, 2026-10-12,
`docs/design/tools-assumed.md`). Until that day this engine refused tools and
structured output because nobody had tried them. Measured on Troy's running Strata,
2026-10-12 (`qwen3.8-flash-next-iq2_xs`, its own OpenAI endpoint; its `/health`
names no version, so none is claimed), it calls tools correctly -- one call, two at
once, streamed, and the round trip with the tool's result -- and answers
`response_format` natively (`json_schema` and `json_object`). Three things it does
not do. The first two are sent anyway and named, never refused
(`settings_not_honoured`, reported as `capabilities.settingsNotHonoured` and, per
request, on `x-eugene-plexus-ignored-settings`):

* **It ignores `tool_choice`**, all of it: `none` still called the tool,
  `required` answered "Hello!", and a named function called another one. So
  `none` is honoured here by sending no tools, and `required` or a named
  function is sent as Strata takes it and named (`toolChoice`).
* **It ignores `parallel_tool_calls: false`.** Measured 2026-10-12 on Troy's
  running Strata (`qwen3.8-flash-next-iq2_xs`, its own OpenAI endpoint, directly),
  with a question that wants two calls ("the weather in Paris and in Tokyo", one
  tool, temperature 0), the field absent, `true` and `false`, two runs each:
  every one of the six answered both calls (Paris and Tokyo, `finish_reason`
  `tool_calls`, HTTP 200), in 484 to 502 ms (1202 ms for the first, cold).
  `false` with no tools at all is a 200 too (a plain answer, 3016 ms), not a 400.
  So `true` is what it does anyway, and `false` is accepted and not obeyed: sent,
  and named (`parallelToolCalls`). **Refused is wrong for it**: Codex sends
  `parallel_tool_calls: false` on every request, so a refusal made Strata unusable
  from Codex, which is what it was until this round.
* **`response_format` together with tools is its own 400**
  ("structured response_format with tools/MCP is not supported"), which this
  engine does not second-guess: the request goes through and the caller gets
  Strata's words.
"""

from __future__ import annotations

from typing import Any

import httpx

from .._generated.models import (
    GenerateRequest,
    RerankPassagesRequest,
    RerankPassagesResponse,
    SettingNotHonoured,
    Tool,
)
from .base import RerankUnsupported
from .openai_compat_http import OpenAiCompatibleHttpEngine, _Target, unread_tool_settings

#: What Strata's endpoint can carry, of the request's settings. `tools`,
#: `toolChoice`, `parallelToolCalls` and `responseFormat` are here since
#: 2026-10-12; the rest is what it always took. `seed` and the logprob family are
#: not: nothing has shown Strata honours them, and an explicit one is refused
#: rather than dropped (A2). (`toolChoice` and `parallelToolCalls` are listed AND
#: partly ignored: a setting Strata is measured not to obey is sent and named,
#: not left out, because a refusal is for what cannot be sent at all.)
_SUPPORTED = frozenset(
    {
        "maxTokens",
        "temperature",
        "topP",
        "topK",
        "minP",
        "stop",
        "reasoningEffort",
        "tools",
        "toolChoice",
        "parallelToolCalls",
        "responseFormat",
    }
)


class StrataHttpEngine(OpenAiCompatibleHttpEngine):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._slot_pinning = False

    def _supported_for(self, target: _Target) -> list[str]:
        return [key for key in super()._supported_for(target) if key in _SUPPORTED]

    def _unsupported_settings(self, target: _Target) -> frozenset[str]:
        from .openai_compat_http import _ALL_ENGINE_SETTINGS

        return super()._unsupported_settings(target) | (_ALL_ENGINE_SETTINGS - _SUPPORTED)

    def settings_not_honoured(self) -> list[SettingNotHonoured]:
        """What Strata is sent and does not honour (measured 2026-10-12):
        `toolChoice` `required` and a named function, and `parallelToolCalls`
        `false`. Everything else it reads: `auto` and `true` are what it does
        anyway, and `none` is honoured by sending no tools (`_payload_for`)."""
        return unread_tool_settings()

    async def _forced_tool(self, request: GenerateRequest) -> Tool | None:
        """Never: a named `tool_choice` is sent to Strata as it is.

        The base class answers a named choice on llama-server by turning the
        tool's parameters into a JSON-schema `response_format` and taking the
        tools off the request (`_force_by_schema`). That route is for an engine
        that ignores `tool_choice` and enforces a grammar, which Strata does
        not claim to be: the request goes to Strata as it came, `toolChoice` is
        named as ignored, and `response_format` is only ever the caller's own,
        which Strata answers natively. `_answers_as_llama_cpp` is already False
        here; this makes the route unreachable by construction, not by a sniff
        of `/props`.
        """
        return None

    async def _answers_as_llama_cpp(self) -> bool:
        return False

    async def _ctx_llama_cpp(self, client: httpx.AsyncClient) -> int | None:
        self._llama_cpp = False
        response = await client.get("/health", headers=self._headers(), timeout=2)
        body = response.json()
        if (
            response.status_code != 200
            or not isinstance(body, dict)
            or body.get("service") != "strata"
            or body.get("loaded") is not True
        ):
            return None
        value = body.get("max_context")
        return value if type(value) is int and value > 0 else None

    async def probe_completion(self) -> tuple[bool, bool]:
        return False, False

    async def probe_embeddings(self) -> bool:
        return False

    async def probe_rerank(self) -> bool | None:
        # Not asked: the inherited probe would send `/v1/rerank` to a subset
        # that serves text chat only, and say nothing it could be believed on.
        return False

    async def rerank(self, request: RerankPassagesRequest) -> RerankPassagesResponse:
        raise RerankUnsupported("Strata serves text chat only")

    async def probe_image_input(self) -> bool:
        return False

    async def probe_audio_input(self) -> bool:
        return False
