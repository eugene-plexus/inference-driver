"""Strata v0.1.39's experimental text-only subset of OpenAI HTTP.

Reuse transport, incremental reasoning, usage and cancellation. Do not infer
llama.cpp semantics from Strata's compatibility /props or /slots endpoints.
"""

from __future__ import annotations

from typing import Any

import httpx

from .._generated.models import GenerateRequest, Tool
from ._subprocess import CliError
from .openai_compat_http import OpenAiCompatibleHttpEngine, _Target

_SUPPORTED = frozenset(
    {"maxTokens", "temperature", "topP", "topK", "minP", "stop", "reasoningEffort"}
)


class StrataHttpEngine(OpenAiCompatibleHttpEngine):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.supports_tool_calling = False
        self._slot_pinning = False

    def _chat_tools_unavailable(self, upstream: str) -> bool:
        return True

    def _supported_for(self, target: _Target) -> list[str]:
        return [key for key in super()._supported_for(target) if key in _SUPPORTED]

    def _unsupported_settings(self, target: _Target) -> frozenset[str]:
        from .openai_compat_http import _ALL_ENGINE_SETTINGS

        return super()._unsupported_settings(target) | (_ALL_ENGINE_SETTINGS - _SUPPORTED)

    def _payload_for(
        self, request: GenerateRequest, target: _Target, forced: Tool | None = None
    ) -> dict[str, Any]:
        if request.tools or request.toolChoice is not None or request.responseFormat is not None:
            raise CliError(
                "Experimental Strata supports text chat only; tools and structured output "
                "are not supported yet.",
                upstream_status=400,
            )
        if any(
            isinstance(m.content, list)
            and any(getattr(p, "type", None) != "text" for p in m.content)
            for m in request.messages
        ):
            raise CliError("Experimental Strata supports text messages only.", upstream_status=400)
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

    async def probe_image_input(self) -> bool:
        return False

    async def probe_audio_input(self) -> bool:
        return False
