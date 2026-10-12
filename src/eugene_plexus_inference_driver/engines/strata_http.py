"""Strata's OpenAI HTTP endpoint: text chat, tool calls and structured output.

Reuse transport, incremental reasoning, usage and cancellation. Do not infer
llama.cpp semantics from Strata's compatibility /props or /slots endpoints.

**Tools are assumed, as for every engine** (Troy, 2026-10-12,
`docs/design/tools-assumed.md`). Until that day this engine refused tools and
structured output because nobody had tried them. Measured on Strata v0.1.39
(`qwen3.8-flash-next-iq2_xs`, its own OpenAI endpoint), it calls tools correctly
-- one call, two at once, streamed, and the round trip with the tool's result --
and answers `response_format` natively (`json_schema` and `json_object`). Two
things it does not do:

* **It ignores `tool_choice`**, all of it: `none` still called the tool,
  `required` answered "Hello!", and a named function called another one. So
  `none` is honoured here by sending no tools, and `required` or a named
  function is sent as Strata takes it and named on
  `x-eugene-plexus-ignored-settings` (`toolChoice`), not refused.
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
    Tool,
    ToolChoice,
)
from ._subprocess import CliError
from .base import RerankUnsupported
from .openai_compat_http import OpenAiCompatibleHttpEngine, _Target

#: What Strata's endpoint can carry, of the request's settings. `tools`,
#: `toolChoice` and `responseFormat` are here since 2026-10-12; the rest is
#: what it always took. `parallelToolCalls`, `seed` and the logprob family are
#: not: nothing has shown Strata honours them, and an explicit one is refused
#: rather than dropped (A2).
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

    def ignored_settings(self, request: GenerateRequest) -> list[str]:
        """`toolChoice`, when it is a value Strata ignores (measured 2026-10-12).

        `auto` is what it does anyway, and `none` is honoured by sending no
        tools (`_payload_for`); `required` and a named function are sent, and
        Strata answers as if they were `auto`.
        """
        choice = request.toolChoice
        if choice is None or choice in (ToolChoice.none, ToolChoice.auto):
            return []
        return ["toolChoice"]

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

    def _payload_for(
        self, request: GenerateRequest, target: _Target, forced: Tool | None = None
    ) -> dict[str, Any]:
        if any(
            isinstance(m.content, list)
            and any(getattr(p, "type", None) != "text" for p in m.content)
            for m in request.messages
        ):
            raise CliError("Experimental Strata supports text messages only.", upstream_status=400)
        if request.toolChoice == ToolChoice.none:
            # Honoured as Strata can: it ignores `tool_choice: none` (measured, it
            # called the tool), so the tools are not sent at all. Nothing is left
            # on the request for it to ignore.
            request = request.model_copy(update={"tools": None, "toolChoice": None})
        return super()._payload_for(request, target, forced)

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
