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
`fixed_temperature_pattern`, and the `backend_kind` to report. New providers
that share this protocol = a new entry in `providers.py`, not a
new engine file.

Auth: `Authorization: Bearer <api_key>` header. The API key is
read from config (`apiKey`, sensitive) with a fallback to the
`OPENAI_API_KEY` env var so existing setups still work.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from .._generated.models import (
    BackendKind,
    ConfigField,
    ConfigFieldShowWhen,
    ConfigValueType,
    EmbedResponse,
    FinishReason,
    Function1,
    FunctionCall,
    GenerateRequest,
    GenerateResponse,
    Role,
    ToolCall,
    ToolCallDelta,
    Usage,
)
from .._http import client_for
from ..images import content_wire, has_images
from ._subprocess import BackendTimeout, CliError
from ._thinking import ThinkingFilter, apply_thinking_mode, strip_thinking_blocks
from .base import DEFAULT_REQUEST_TIMEOUT_SECONDS, Chunk, refuse_unsupported_settings

log = logging.getLogger(__name__)
_IMAGE_ERROR_HINT = (
    "Image request refused by the backend. Check the loaded vision model/projector "
    "and available context; try a smaller image or shorter conversation. "
    "Upstream body omitted to protect attachment data."
)

# How long a discovered context window is trusted before the backend is
# asked again. Generous because the number only moves when an engine is
# restarted with different flags, and the cost of asking is three HTTP
# GETs inside the gateway's routing refresh.
_CONTEXT_TTL_SECONDS = 300.0

# How long a *miss* is remembered. Much shorter, because the common miss
# is an Ollama with nothing loaded yet: the window appears the moment a
# first request loads the model, and waiting five minutes to notice
# would mean a fresh install advertises no window through its whole
# first conversation.
_CONTEXT_MISS_TTL = 30.0

# Total wall-clock the three probes share. `/v1/info` is polled by the
# gateway's routing refresh, which runs inside whatever request
# triggered it, so this is latency an end user can feel -- and a backend
# that answers none of the three is usually a hosted provider that will
# 404 instantly anyway.
_CONTEXT_PROBE_BUDGET_SECONDS = 2.0

# Per-request deadline for the three context probes. They share the
# engine's one client, whose default budget is the generation timeout
# (120 s), so the short deadline has to ride on each request rather
# than on the client -- otherwise one client would have to carry four
# different budgets and the shortest would become everyone's.
_CONTEXT_PROBE_TIMEOUT = httpx.Timeout(_CONTEXT_PROBE_BUDGET_SECONDS, connect=2.0)


def _only_or_named(
    entries: list[dict[str, Any]], model_id: str, *, keys: tuple[str, ...]
) -> dict[str, Any] | None:
    """The entry describing `model_id`, or the sole entry if there is one.

    Matching by name first matters when a server hosts several models:
    taking the first card would report one model's window for another,
    and a window that is wrong is worse than one that is missing. The
    single-entry fallback covers the servers that do not echo the id we
    configured in the form we configured it.
    """
    named = [e for e in entries if any(e.get(k) == model_id for k in keys)]
    if named:
        return named[0]
    return entries[0] if len(entries) == 1 else None


# Models whose `temperature` is not tunable: OpenAI's o-series rejects
# the parameter outright, and the gpt-5 family schema-accepts it but
# errors on any value other than the default. Sending it is a 400, so
# the adapter drops it and warns rather than refusing the model — see
# `_temperature_for`. Catches every gpt-5 family member that uses
# either `-` or `.` after the family name, plus the o-series, while
# explicitly NOT matching hypothetical `gpt-50` / `gpt-5o` style names
# that aren't actually 5.x. Other providers pass their own pattern (or
# None) via the registry.
OPENAI_FIXED_TEMPERATURE_PATTERN: re.Pattern[str] = re.compile(
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

# The o-series doesn't share a prefix with anything else on the list, so
# it needs its own test. It was absent while reasoning models were
# refused outright and never reached the dropdown; now that they do, an
# allow-list that silently omits `o3` is the same refusal by another
# route. Anchored digits so a future `omni-` model doesn't match.
_O_SERIES_RE = re.compile(r"^o\d+(?:[-.]|$)", re.IGNORECASE)


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
    return lowered.startswith(_CHAT_MODEL_PREFIXES) or bool(_O_SERIES_RE.match(lowered))


_FINISH_REASON_MAP = {
    "stop": FinishReason.stop,
    "length": FinishReason.length,
    # **Its own value since 2026-09-19**, having been folded into
    # `error` since M0. Two different states with two different
    # remedies: `error` means the generation was truncated because
    # something broke and a retry may work, `content_filter` means a
    # classifier stopped it on purpose and a retry will not. Folded
    # together the gateway could only render the pair as `stop`, so a
    # refusal reached the caller as a natural end.
    "content_filter": FinishReason.content_filter,
    # Both were `FinishReason.stop` until tool calling landed, which is
    # the shape of the whole gap: a backend that HAD made a tool call
    # reported a clean natural stop, and the calls themselves were
    # dropped one line later. `function_call` is OpenAI's deprecated
    # spelling and means the same thing.
    "tool_calls": FinishReason.tool_calls,
    "function_call": FinishReason.tool_calls,
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


def _sse_data(line: str) -> str | None:
    """The payload of one SSE `data:` line, or None for anything else.

    Deliberately narrow. This reads *upstream's* SSE (llama.cpp, vLLM,
    Ollama, OpenAI and the rest all frame it the same way): one JSON
    object per `data:` line, blank lines between events, `:` comments
    used as keep-alives. Event names, multi-line data and `id:`/`retry:`
    fields do not appear on this wire and are ignored rather than
    half-supported.
    """
    if not line or line.startswith(":"):
        return None
    if not line.startswith("data:"):
        return None
    return line[5:].strip()


def _timed_out(exc: httpx.TimeoutException, limit_seconds: float, what: str) -> BackendTimeout:
    """The one failure whose cause we know exactly, said in words.

    `str(httpx.ReadTimeout(""))` is the empty string -- httpx raises
    these with no message -- so the driver's own error used to read
    ``"openai_compat_http request failed: "`` and stop. Everything a
    reader needs is here instead: which deadline fired, what it was set
    to, what to turn, and that the engine is almost certainly still
    computing rather than broken.

    A **connect** timeout is not one of these. Nothing was ever handed
    to the engine, so it is a dead host -- exactly what failover is for
    -- and it stays an ordinary `CliError` on the cascading path.
    """
    return BackendTimeout(
        f"openai_compat_http {what} did not answer within {limit_seconds:g}s "
        f"({type(exc).__name__}). The backend was still working, not broken: a large model "
        f"on CPU or a long answer can take minutes. Raise requestTimeoutSeconds on this "
        f"driver (and on the gateway, which holds the shorter deadline of the two).",
        limit_seconds=limit_seconds,
    )


class OpenAiCompatibleHttpEngine:
    """OpenAI-compatible HTTP engine. Provider-agnostic."""

    #: This engine can front a runtime the agent supervises, addressed by
    #: name rather than by URL. The CLI engines cannot — a subscription
    #: is not a runtime — so `app.build_engine_with` consults this before
    #: resolving `runtimeName` at all.
    follows_runtimes = True

    #: Upstream SSE delivers real per-token deltas, so `stream()`
    #: yields text as the model produces it.
    supports_streaming = True
    supports_tool_calling = True
    #: Set from `probe_embeddings()` at first ask, not declared --
    #: the backend is the only thing that knows, and it only knows
    #: when asked. False until then, which is honest: an
    #: unprobed backend has made no promise.
    supports_embeddings = False

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com",
        model_id: str = "gpt-4o",
        timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        fixed_temperature_pattern: re.Pattern[str] | None = None,
        backend_kind: BackendKind = BackendKind.openai_api,
        thinking_mode: str = "auto",
        auth_required: bool = True,
        filter_models: bool = True,
        runtime: str | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if auth_required and not resolved_key:
            raise CliError(
                "openai_compat_http engine has no API key — set `apiKey` in "
                "config or export OPENAI_API_KEY in the environment."
            )
        self._api_key = resolved_key
        self._base_url = base_url.rstrip("/")
        self._model_id = model_id
        self._timeout_seconds = timeout_seconds
        self._fixed_temperature_pattern = fixed_temperature_pattern
        self._warned_dropped: set[str] = set()
        self._temperature_is_fixed = fixed_temperature_pattern is not None and bool(
            fixed_temperature_pattern.match(model_id)
        )
        if self._temperature_is_fixed:
            log.warning(
                "model %r does not accept a tunable `temperature`; the driver "
                "will omit the parameter on every request and let the model "
                "use its own default. Everything else works normally.",
                model_id,
            )
        self.backend_kind = backend_kind
        self._thinking_mode = thinking_mode or "auto"
        self._filter_models = filter_models
        #: The supervised runtime this engine follows, when `base_url` was
        #: resolved from one. Reported on `/v1/info` so the gateway and
        #: the UI can show which engine process is behind this driver.
        self.runtime = runtime
        #: Last window the backend admitted to, and when we last asked.
        #: Cached on the engine rather than the app because the engine is
        #: rebuilt on every config change, which is exactly when a stale
        #: window would be wrong.
        self._context_window: int | None = None
        self._context_window_checked_at: float | None = None
        #: Whether the backend will embed. Determined by trying, once,
        #: and cached for the life of the engine -- which is the right
        #: lifetime: it cannot change without the backend restarting,
        #: and the engine is rebuilt on every config change.
        self._embeddings: bool | None = None
        #: This engine's one HTTP client, built on first use by
        #: `_client()`. See that method for why it is lazy and why the
        #: per-call deadlines do not live on it.
        self._http_client: httpx.AsyncClient | None = None

    def _client(self) -> httpx.AsyncClient:
        """This engine's HTTP client, built once and reused.

        **Never construct a client per call.** Doing so parses certifi's
        PEM bundle on the event loop -- 104-136 ms of synchronous CPU
        measured in this repo's own venv on the Python the installers
        provision -- which was the whole of the ~116 ms of control-plane
        overhead this project carried as unexplained from M8. It also
        stalls every *concurrent* stream, because the parse is CPU and
        the loop cannot interleave it.

        Lazy rather than built in `__init__` because `/v1/config/test`
        and the schema endpoint construct a throwaway engine to validate
        a PATCH; an eager client would leak a connection pool per PATCH.
        Whoever builds an engine closes it -- `aclose()`.

        The client carries the base URL and the default request budget;
        the probes that need a shorter deadline pass `timeout=` per
        request, because four different budgets share one client and
        the shortest of them must not become everyone's.
        """
        if self._http_client is None:
            self._http_client = client_for(
                self._base_url,
                base_url=self._base_url,
                timeout=httpx.Timeout(self._timeout_seconds, connect=10.0),
            )
        return self._http_client

    async def aclose(self) -> None:
        """Release the connection pool. Idempotent."""
        client, self._http_client = self._http_client, None
        if client is not None:
            await client.aclose()

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
        fixed_temperature_pattern: re.Pattern[str] | None,
        backend_kind: BackendKind,
        auth_required: bool = True,
        filter_models: bool = True,
        runtime_url: str | None = None,
        runtime_name: str | None = None,
    ) -> OpenAiCompatibleHttpEngine:
        # Precedence: a resolved runtime URL (the caller turned
        # `runtimeName` into one via the agent), then the operator's
        # literal `baseUrl` (set only for openai_compat_custom), then the
        # provider's built-in default. `runtimeName` wins when both are
        # set — the contract says so, and the reason is that the literal
        # URL is the one that goes stale.
        base_url = str(runtime_url or get("baseUrl") or default_base_url or "").strip()
        if not base_url:
            raise CliError(
                "OpenAI-compatible engine has no backend. For the custom provider, "
                "set `runtimeName` to a runtime the agent supervises, or `baseUrl` "
                "for an endpoint that is not one. For named providers this is a "
                "registry bug — file an issue."
            )
        return cls(
            api_key=str(get("apiKey") or "") or None,
            base_url=base_url,
            model_id=str(get("modelId") or "gpt-4o"),
            timeout_seconds=float(get("requestTimeoutSeconds") or DEFAULT_REQUEST_TIMEOUT_SECONDS),
            fixed_temperature_pattern=fixed_temperature_pattern,
            backend_kind=backend_kind,
            thinking_mode=str(get("thinkingMode") or "auto"),
            auth_required=auth_required,
            filter_models=filter_models,
            runtime=runtime_name if runtime_url else None,
        )

    def _payload_for(self, request: GenerateRequest) -> dict[str, Any]:
        """The chat-completions body for this request.

        Shared by `generate` and `stream` so the two cannot drift on
        param shaping -- which would be an especially quiet bug, since
        the only symptom would be a streamed answer differing from a
        non-streamed one for the same request.
        """
        if self._temperature_is_fixed:
            refuse_unsupported_settings(request, unsupported={"temperature", "topP"})
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
        if request.temperature is not None and not self._temperature_is_fixed:
            payload["temperature"] = float(request.temperature)
        # **Carried since 2026-09-19.** `GenerateRequest` had no field
        # for either until then, so `gateway.yaml`'s promise that both
        # were passed through to backends that support them was
        # unfulfillable and nothing was logged. They ride through
        # untouched, exactly as `temperature` does and for the same
        # reason: the gateway owns every parameter that changes what
        # the model says, and a driver never substitutes one.
        #
        # `top_p` is dropped by the same flag that drops `temperature`,
        # and only by that flag. OpenAI's reasoning models reject both
        # -- the sampler is not the caller's to tune there -- so
        # sending it would be the 400 that flag exists to avoid. The
        # seed is NOT dropped with them: those models accept it, and
        # widening the drop to every sampling parameter because two
        # travel together would be the over-correction.
        if request.topP is not None and not self._temperature_is_fixed:
            payload["top_p"] = float(request.topP)
        elif request.topP is not None:
            self._warn_dropped("top_p")
        # `is not None` and not truthiness: **`seed=0` is a real seed**
        # and a falsy one, and dropping it would answer a request for a
        # reproducible result with a different answer every time --
        # which is the whole of what this field was doing before today.
        if request.seed is not None:
            payload["seed"] = int(request.seed)
        if request.stop:
            payload["stop"] = list(request.stop)
        # Tools ride through untouched. We do not validate the JSON
        # Schema in `parameters`, rewrite dialects, or reorder the list:
        # a backend that rejects a construct rejects it in its own
        # words, which is more use to the caller than a guess of ours
        # made one hop earlier.
        if request.tools:
            payload["tools"] = [t.model_dump(mode="json", exclude_none=True) for t in request.tools]
        if request.toolChoice is not None:
            choice = request.toolChoice
            payload["tool_choice"] = (
                str(choice.value)
                if hasattr(choice, "value")
                else choice.model_dump(mode="json", exclude_none=True)
                if hasattr(choice, "model_dump")
                else choice
            )
        if request.responseFormat is not None:
            payload["response_format"] = request.responseFormat.model_dump(
                mode="json", by_alias=True, exclude_none=True
            )
        return payload

    def _warn_dropped(self, field: str) -> None:
        """Say it once per field per engine, not once per request.

        The contract promises a parameter we cannot carry is "dropped
        with a warning". A warning on every request would be a log line
        per token-generating call on a busy backend, which is how a
        real warning becomes something an operator filters out.
        """
        if field in self._warned_dropped:
            return
        self._warned_dropped.add(field)
        log.warning(
            "model %r does not accept `%s`; the driver is omitting it on every "
            "request and letting the model use its own default. This is said "
            "once per parameter for the life of this engine.",
            self._model_id,
            field,
        )

    def _headers(self) -> dict[str, str]:
        # No key means no header, not `Bearer None`. Providers that need
        # auth reject a missing header with a readable 401; a literal
        # "None" is a malformed credential, and some servers answer that
        # with a 400 that reads like our payload was wrong.
        headers = {"Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        # **Brackets the same span `stream()` does, deliberately.** This
        # used to report `response.elapsed`, which httpx starts *after*
        # the client is built and stops when the body is read -- so the
        # driver's largest cost sat outside the driver's own measurement,
        # and the ~116 ms gap between what the gateway timed and what the
        # driver reported read as unexplained overhead for a week. One
        # instrument, one span, both paths. `perf_counter` and never
        # `monotonic`: on Windows/CPython 3.12, the Python both
        # installers provision, `monotonic()` is `GetTickCount64` with a
        # 15.6 ms grid -- 20 distinct values in 300 ms -- which is coarser
        # than the thing being measured.
        started = time.perf_counter()
        payload = self._payload_for(request)
        image_request = has_images(request.messages)

        # DEBUG-level full-payload trace. The gateway's copy-trace
        # captures what we sent it; this captures what WE send upstream
        # (post role-coercion, post-thinking-directive injection,
        # post-param-shaping). When operators flip to DEBUG to chase
        # "is the LLM actually seeing what I think it's seeing", this
        # is the load-bearing log line. Auth header omitted on purpose.
        if log.isEnabledFor(logging.DEBUG) and not image_request:
            log.debug(
                "openai_compat_http → POST %s/v1/chat/completions\n%s",
                self._base_url,
                json.dumps(payload, indent=2, ensure_ascii=False),
            )

        headers = self._headers()

        client = self._client()
        try:
            response = await client.post(
                "/v1/chat/completions",
                headers=headers,
                json=payload,
            )
        except httpx.ConnectTimeout as e:
            # A host that never accepted the connection is a DEAD host,
            # not a slow one: nothing was handed to an engine, so the
            # next backend is a real rescue. Deliberately below the
            # `TimeoutException` branch it would otherwise match.
            raise CliError(f"openai_compat_http could not connect: {e!r}") from e
        except httpx.TimeoutException as e:
            raise _timed_out(e, self._timeout_seconds, "the completion") from e
        except httpx.HTTPError as e:
            raise CliError(f"openai_compat_http request failed: {e!r}") from e

        if response.status_code >= 400:
            if log.isEnabledFor(logging.DEBUG) and not image_request:
                log.debug(
                    "openai_compat_http ← HTTP %d (%dms) body:\n%s",
                    response.status_code,
                    int((time.perf_counter() - started) * 1000),
                    _redact(response.text[:4000]),
                )
            # **The status is the payload.** An over-long prompt comes
            # back from llama.cpp as a 400 naming both numbers, and the
            # route needs the 4xx/5xx distinction to decide whether the
            # gateway should cascade past it. Flattening it here is what
            # made an exact refusal look like a broken backend.
            raise CliError(
                f"openai_compat_http returned {response.status_code}: "
                f"{_redact(response.text[:500]) if not image_request else _IMAGE_ERROR_HINT}",
                upstream_status=response.status_code,
            )

        try:
            body = response.json()
        except ValueError as e:
            raise CliError("openai_compat_http returned non-JSON") from e

        if log.isEnabledFor(logging.DEBUG) and not image_request:
            log.debug(
                "openai_compat_http ← HTTP %d (%dms) body:\n%s",
                response.status_code,
                int((time.perf_counter() - started) * 1000),
                json.dumps(body, indent=2, ensure_ascii=False),
            )

        choices = body.get("choices") or []
        if not choices:
            raise CliError("openai_compat_http returned no choices")

        first = choices[0] or {}
        message = first.get("message") or {}
        content = message.get("content")
        tool_calls = _tool_calls_from_wire(message.get("tool_calls"))
        # **`content` is null on a tool-call-only turn**, and this used
        # to raise on exactly that response -- the concrete way a tool
        # call failed here before the contract carried one. Text is
        # still required when there are no calls: a response with
        # neither is a backend malfunction, not an empty answer.
        if not isinstance(content, str):
            if not tool_calls:
                raise CliError("openai_compat_http response missing string content")
            content = None
        elif self._thinking_mode == "off":
            # Defensive strip of <think>...</think> when the operator opted
            # out of thinking but the model emitted tags anyway. See
            # _thinking.strip_thinking_blocks for the why.
            content = strip_thinking_blocks(content)

        return GenerateResponse(
            content=content,
            toolCalls=tool_calls or None,
            finishReason=_FINISH_REASON_MAP.get(
                str(first.get("finish_reason") or "stop"), FinishReason.stop
            ),
            usage=_usage_from_envelope(body.get("usage") or {}),
            requestId=request.requestId,
            backend=self.backend_kind,
            modelId=str(body.get("model") or self._model_id),
            latencyMs=int((time.perf_counter() - started) * 1000),
        )

    async def stream(self, request: GenerateRequest) -> AsyncGenerator[Chunk, None]:
        """Token-by-token, over upstream's own SSE.

        The wire shape is OpenAI's: `data:` lines carrying a chunk whose
        `choices[0].delta.content` is the new text, terminated by a
        literal `data: [DONE]`. Deltas that carry only a role are
        normal and yield nothing.

        Two things here are not obvious:

        **`thinkingMode: off` needs an incremental filter.** The batch
        path strips `<think>...</think>` from a finished string, which
        a stream cannot do -- a block already forwarded cannot be
        un-sent. `ThinkingFilter` withholds text that might still turn
        out to be a tag. Without it this mode would be silently weaker
        for exactly the clients that stream.

        **`usage` usually is not there.** Most servers omit it on a
        streamed response unless asked; llama.cpp and vLLM both honour
        `stream_options.include_usage`, so we ask, and tolerate its
        absence rather than failing the request over accounting.

        The response is closed by the `async with`, including when the
        consumer abandons the generator -- which is what a client
        disconnect looks like from here.
        """
        payload = self._payload_for(request)
        image_request = has_images(request.messages)
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}

        started = time.perf_counter()
        filtered = ThinkingFilter() if self._thinking_mode == "off" else None
        emitted: list[str] = []
        # Did upstream actually say it was finished? Iterating a closed
        # connection simply ends, so without one of these two markers a
        # backend that died mid-answer is indistinguishable from one that
        # completed -- see the raise below.
        saw_terminator = False
        finish_reason = "stop"
        usage_payload: dict[str, Any] = {}
        # Tool-call fragments accumulated by `index`, so the terminal
        # `done` can carry whole calls. A model may interleave fragments
        # of two calls, which is why the index is the key and not the
        # arrival order.
        call_parts: dict[int, dict[str, str]] = {}
        served_model = self._model_id

        client = self._client()
        try:
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                headers={**self._headers(), "Accept": "text/event-stream"},
                json=payload,
            ) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    # Nothing has been streamed yet, so this can
                    # still be a status code -- see the same raise in
                    # `generate`.
                    detail = (
                        _IMAGE_ERROR_HINT
                        if image_request
                        else _redact(body.decode("utf-8", "replace")[:500])
                    )
                    raise CliError(
                        f"openai_compat_http returned {response.status_code}: {detail}",
                        upstream_status=response.status_code,
                    )
                async for line in response.aiter_lines():
                    data = _sse_data(line)
                    if data is None:
                        continue
                    if data == "[DONE]":
                        saw_terminator = True
                        break
                    try:
                        event = json.loads(data)
                    except ValueError:
                        # A malformed frame mid-stream is not worth
                        # failing a half-delivered answer over.
                        log.debug("openai_compat_http: unparseable SSE frame (contents omitted)")
                        continue
                    served_model = str(event.get("model") or served_model)
                    if event.get("usage"):
                        usage_payload = event["usage"]
                    for choice in event.get("choices") or []:
                        if choice.get("finish_reason"):
                            finish_reason = str(choice["finish_reason"])
                            saw_terminator = True
                        delta = choice.get("delta") or {}
                        # Tool-call fragments ride their own frame.
                        # Forwarded rather than accumulated here: the
                        # gateway has to emit them as OpenAI deltas
                        # anyway, and buffering them to the end of
                        # the stream would defeat the point of
                        # streaming a call the caller wants to start
                        # dispatching. We also accumulate a copy so
                        # the terminal `done` carries whole calls,
                        # for the non-streaming half of the contract.
                        raw_calls = delta.get("tool_calls")
                        if raw_calls:
                            fragments = _tool_call_deltas_from_wire(raw_calls)
                            if fragments:
                                _accumulate_tool_calls(call_parts, raw_calls)
                                yield Chunk(toolCalls=fragments)
                        text = (delta.get("content")) or ""
                        if not text:
                            continue
                        visible = filtered.feed(text) if filtered is not None else text
                        if visible:
                            emitted.append(visible)
                            yield Chunk(text=visible)
        except httpx.ConnectTimeout as e:
            raise CliError(f"openai_compat_http could not connect: {e!r}") from e
        except httpx.TimeoutException as e:
            raise _timed_out(e, self._timeout_seconds, "the stream") from e
        except httpx.HTTPError as e:
            raise CliError(f"openai_compat_http stream failed: {e!r}") from e

        if not saw_terminator:
            # **Found by M10's acceptance run, which fixtures could not
            # produce.** A backend that vanishes mid-answer closes the
            # connection, `aiter_lines()` ends without raising, and this
            # method used to fall through and emit a `done` carrying
            # whatever had arrived -- reporting a truncated answer as a
            # completed one, with a 200. Upstream marks the end with
            # `data: [DONE]` or a `finish_reason` (llama.cpp, vLLM and
            # Ollama all send at least one); neither means the stream was
            # cut, and saying so is what lets the gateway emit an error
            # frame instead of presenting half an answer as whole.
            raise CliError(
                "openai_compat_http stream ended without [DONE] or a finish_reason "
                f"after {len(''.join(emitted))} characters: the backend closed the "
                "connection mid-answer"
            )

        if filtered is not None:
            tail = filtered.flush()
            if tail:
                emitted.append(tail)
                yield Chunk(text=tail)

        content = "".join(emitted)
        tool_calls = _finish_tool_calls(call_parts)
        yield Chunk(
            done=True,
            result=GenerateResponse(
                # None rather than "" when the turn was only tool calls,
                # so the streamed and non-streamed shapes agree.
                content=content if (content or not tool_calls) else None,
                toolCalls=tool_calls or None,
                finishReason=_FINISH_REASON_MAP.get(finish_reason, FinishReason.stop),
                usage=_usage_from_envelope(usage_payload),
                requestId=request.requestId,
                backend=self.backend_kind,
                modelId=served_model,
                latencyMs=int((time.perf_counter() - started) * 1000),
            ),
        )

    async def embed(self, inputs: list[str]) -> EmbedResponse:
        """`POST /v1/embeddings` upstream, in the shape OpenAI defined.

        **Always asks for floats.** The base64 encoding the OpenAI SDKs
        request by default is applied by the gateway instead, so a
        caller sees identical behaviour whether or not this particular
        backend implements `encoding_format` -- and several do not.
        """
        started = time.perf_counter()
        payload: dict[str, Any] = {"model": self._model_id, "input": inputs}

        client = self._client()
        try:
            response = await client.post("/v1/embeddings", headers=self._headers(), json=payload)
        except httpx.ConnectTimeout as e:
            raise CliError(f"openai_compat_http could not connect: {e!r}") from e
        except httpx.TimeoutException as e:
            raise _timed_out(e, self._timeout_seconds, "the embeddings request") from e
        except httpx.HTTPError as e:
            raise CliError(f"openai_compat_http embeddings request failed: {e!r}") from e

        if response.status_code >= 400:
            raise CliError(
                f"openai_compat_http returned {response.status_code} for embeddings: "
                f"{_redact(response.text[:500])}",
                upstream_status=response.status_code,
            )
        try:
            body = response.json()
        except ValueError as e:
            raise CliError(
                f"openai_compat_http returned non-JSON for embeddings: {response.text[:200]!r}"
            ) from e

        vectors = _vectors_from_envelope(body, expected=len(inputs))
        return EmbedResponse(
            embeddings=vectors,
            modelId=str(body.get("model") or self._model_id),
            backend=self.backend_kind,
            usage=_usage_from_envelope(body.get("usage") or {}),
            latencyMs=int((time.perf_counter() - started) * 1000),
        )

    async def probe_embeddings(self) -> bool:
        """Whether this backend will embed, determined by asking it to.

        **There is no read-only signal, and that was measured rather
        than assumed.** `llama-server` b9846's `/props` exposes no
        pooling or embedding field; Ollama's OpenAI-compatible surface
        says nothing either, and its `/api/ps` reports a context length
        but not a task. The capability also belongs to the *runner*
        rather than the model -- an Ollama runner started for chat
        refuses to embed the very model it is serving.

        So: send the smallest possible request and see. The negative
        case costs nothing measurable -- llama.cpp rejects a non-pooling
        model in 45 ms, before any compute, and Ollama answers
        immediately -- while the positive case costs one tiny embedding
        against a model this driver exists to serve.

        Cached for the engine's lifetime once there is a definite
        answer. A transport failure is **not** a definite answer and is
        not cached, or a backend that happened to be down at startup
        would be recorded as incapable forever.
        """
        if self._embeddings is not None:
            return self._embeddings
        try:
            await self.embed(["1"])
        except CliError as e:
            status = getattr(e, "upstream_status", None)
            if status is None:
                log.debug("embeddings probe inconclusive (transport): %s", e)
                return False
            # A definite "no" from the backend. 4xx and 5xx both count:
            # a server that errors on a one-character embed is not one
            # to advertise an embeddings surface for.
            log.info("backend at %s does not serve embeddings (HTTP %s)", self._base_url, status)
            self._embeddings = False
            return False
        except (httpx.HTTPError, ValueError) as e:
            log.debug("embeddings probe inconclusive: %s", e)
            return False
        log.info("backend at %s serves embeddings", self._base_url)
        self._embeddings = True
        return True

    async def probe_image_input(self) -> bool:
        """Confirm the loaded llama.cpp model without caching across restarts."""
        try:
            async with asyncio.timeout(2.0):
                client = self._client()
                response = await client.get("/props", headers=self._headers(), timeout=2.0)
                if response.status_code != 200:
                    return False
                props = response.json()
                if not isinstance(props, dict):
                    return False
                modalities = props.get("modalities")
                if not isinstance(modalities, dict) or modalities.get("vision") is not True:
                    return False
                response = await client.get("/v1/models", headers=self._headers(), timeout=2.0)
                if response.status_code != 200:
                    return False
                body = response.json()
                models = body.get("data") if isinstance(body, dict) else None
                return bool(
                    isinstance(models, list)
                    and len(models) == 1
                    and isinstance(models[0], dict)
                    and models[0].get("id") == self._model_id
                )
        except (httpx.HTTPError, ValueError, TimeoutError):
            return False

    async def context_window(self) -> int | None:
        """The context window this backend resolved, read back from it.

        Reported as `capabilities.maxContextTokens` on `/v1/info`, where
        it was contracted at M0 and populated by nothing until now --
        `capabilities.streaming`'s M10 story, one field over. The
        consequence was concrete: every backend the install does not
        supervise, which is every Ollama and LM Studio anyone points us
        at, advertised no window at all, so `GET /v1/models` reported
        `context_length: null` for the most common local setup there is.

        **The resolved window, never the trained one.** Each source
        below reports what the server actually allocated, which is what
        a caller needs; a model's trained maximum is an upper bound that
        overstates whenever the operator or the server itself picked
        something smaller, and overstating is the one direction that
        hurts -- a harness that fills an advertised window it does not
        have is the silent-truncation failure this is here to expose.

        Three sources, in the order that short-circuits soonest for the
        engine most likely to be behind an `openai_compat_http` driver:

        * `GET /props` -- llama.cpp, `default_generation_settings.n_ctx`
        * `GET /v1/models` -- vLLM, `max_model_len` on the model's card
        * `GET /api/ps` -- Ollama, `context_length` on a **loaded**
          model. Ollama's OpenAI-compatible surface carries none of
          this, and its `/api/show` carries only the trained maximum;
          `/api/ps` is the one place the number it actually chose
          appears. Measured on 0.34.0: 131072, matching what it
          auto-sized to.

        Absent means unknown and stays unknown. A hosted provider has
        nothing to read, a model Ollama has idled out is not in
        `/api/ps` until the next request loads it, and in both cases
        guessing would be worse than silence -- `_smallest_context` in
        the gateway simply skips a backend that reports nothing.

        Deliberately not on the request path. `/v1/info` is what the
        gateway polls to build its routing table, and a probe there
        happens inside the refresh that a request triggered, so the
        whole cycle shares one short deadline and the answer is cached.
        """
        now = time.perf_counter()
        if self._context_window_checked_at is not None:
            ttl = _CONTEXT_TTL_SECONDS if self._context_window is not None else _CONTEXT_MISS_TTL
            if now - self._context_window_checked_at < ttl:
                return self._context_window

        deadline = now + _CONTEXT_PROBE_BUDGET_SECONDS
        found: int | None = None
        try:
            client = self._client()
            for source in (self._ctx_llama_cpp, self._ctx_vllm, self._ctx_ollama):
                if time.perf_counter() >= deadline:
                    break
                try:
                    found = await source(client)
                except (httpx.HTTPError, ValueError, TypeError, KeyError):
                    found = None
                if found is not None:
                    break
        except (httpx.HTTPError, ValueError):
            found = None

        self._context_window_checked_at = time.perf_counter()
        # A probe that found nothing does not erase what an earlier one
        # found. Ollama drops a model out of `/api/ps` the moment it
        # idles out, and the window it will get on the next load is
        # overwhelmingly the same one -- reporting `null` in between
        # would make the advertised window flicker with the engine's
        # idle timer rather than with anything a caller did.
        if found is not None:
            if found != self._context_window:
                log.info(
                    "backend at %s reports a context window of %d tokens",
                    self._base_url,
                    found,
                )
            self._context_window = found
        return self._context_window

    async def _ctx_llama_cpp(self, client: httpx.AsyncClient) -> int | None:
        """`GET /props` -- what `llama-server` resolved, post-clamp.

        The agent reads the same field off supervised runtimes; this
        reads it from wherever the driver points, supervised or not.
        """
        response = await client.get(
            "/props", headers=self._headers(), timeout=_CONTEXT_PROBE_TIMEOUT
        )
        if response.status_code >= 400:
            return None
        props = response.json()
        if not isinstance(props, dict):
            return None
        settings = props.get("default_generation_settings")
        value = settings.get("n_ctx") if isinstance(settings, dict) else None
        if value is None:
            value = props.get("n_ctx")
        return value if isinstance(value, int) and value > 0 else None

    async def _ctx_vllm(self, client: httpx.AsyncClient) -> int | None:
        """`GET /v1/models` -- vLLM puts `max_model_len` on each card.

        Matched to our own `modelId` rather than taken from the first
        entry, because a vLLM serving several models would otherwise
        hand back a window belonging to a different one.
        """
        response = await client.get(
            "/v1/models", headers=self._headers(), timeout=_CONTEXT_PROBE_TIMEOUT
        )
        if response.status_code >= 400:
            return None
        body = response.json()
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list):
            return None
        cards = [c for c in data if isinstance(c, dict)]
        card = _only_or_named(cards, self._model_id, keys=("id",))
        value = card.get("max_model_len") if card else None
        return value if isinstance(value, int) and value > 0 else None

    async def _ctx_ollama(self, client: httpx.AsyncClient) -> int | None:
        """`GET /api/ps` -- the window Ollama chose for a loaded model.

        Native API, not the OpenAI-compatible one, because the number
        does not exist on the compatible surface. Empty while nothing is
        loaded, which is why a miss is cached briefly and never
        overwrites a previous hit.
        """
        response = await client.get(
            "/api/ps",
            headers={"Accept": "application/json"},
            timeout=_CONTEXT_PROBE_TIMEOUT,
        )
        if response.status_code >= 400:
            return None
        body = response.json()
        models = body.get("models") if isinstance(body, dict) else None
        if not isinstance(models, list):
            return None
        entries = [m for m in models if isinstance(m, dict)]
        entry = _only_or_named(entries, self._model_id, keys=("name", "model"))
        value = entry.get("context_length") if entry else None
        return value if isinstance(value, int) and value > 0 else None

    async def list_models(self) -> list[str]:
        """Live-fetch the model catalog from the configured base URL.

        OpenAI proper and most OpenAI-compatible providers expose
        `GET /v1/models` returning `{"data": [{"id": "<model>", ...}]}`.
        We filter out non-chat models (embeddings, dall-e, whisper, tts,
        moderation, codex-only, audio) — heuristic by id prefix /
        substring. Reasoning models stay in the list: they are chat
        models the operator owns, and the only thing special about them
        is a parameter the adapter already drops for them.

        Returns `[]` on transport / parse failure so the schema endpoint
        can fall back to free-text input. Driver stays up either way.
        """
        headers: dict[str, str] = {"Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            response = await self._client().get(
                "/v1/models", headers=headers, timeout=httpx.Timeout(15.0, connect=5.0)
            )
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
            # those; trust the operator's choice.
            if self._filter_models and not _is_plausible_chat_model(mid):
                continue
            ids.append(mid)
        ids.sort()
        return ids


def _to_openai_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """Map our Message[] to OpenAI chat-completions messages.

    The `tool` role and an assistant turn's `toolCalls` are what make an
    agent loop possible: the harness sends back the assistant message
    that *asked* for the calls plus one `tool` message per result, and a
    backend that does not receive both has no idea its own request was
    answered.

    Before tool calling landed the final branch here coerced every
    unrecognised role to `user`. That was safe while `Role` had three
    values; with `tool` in the enum it would have turned a tool result
    into something the model reads as the human talking, which is a
    plausible-looking transcript that quietly breaks the loop.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        role = getattr(m, "role", None)
        content = content_wire(getattr(m, "content", None))
        if role == Role.system:
            out.append({"role": "system", "content": content or ""})
        elif role == Role.user:
            out.append({"role": "user", "content": content or ""})
        elif role == Role.assistant:
            msg: dict[str, Any] = {"role": "assistant", "content": content}
            calls = getattr(m, "toolCalls", None)
            if calls:
                msg["tool_calls"] = [_tool_call_to_wire(c) for c in calls]
            out.append(msg)
        elif role == Role.tool:
            out.append(
                {
                    "role": "tool",
                    "content": content or "",
                    "tool_call_id": getattr(m, "toolCallId", None) or "",
                }
            )
        else:
            out.append({"role": "user", "content": content or ""})
    return out


def _tool_call_to_wire(call: Any) -> dict[str, Any]:
    """One of our tool calls as OpenAI sends them back up.

    Accepts a dict as well as a model, because `common.yaml` types
    `Message.toolCalls` loosely on purpose -- the shared schema refuses
    to be a third definition of OpenAI's object.
    """
    if isinstance(call, dict):
        fn = call.get("function") or {}
        return {
            "id": call.get("id") or "",
            "type": "function",
            "function": {
                "name": (fn.get("name") if isinstance(fn, dict) else None) or "",
                "arguments": (fn.get("arguments") if isinstance(fn, dict) else None) or "",
            },
        }
    return {
        "id": getattr(call, "id", "") or "",
        "type": "function",
        "function": {
            "name": getattr(call.function, "name", "") or "",
            "arguments": getattr(call.function, "arguments", "") or "",
        },
    }


def _tool_call_deltas_from_wire(raw: Any) -> list[ToolCallDelta]:
    """One frame's worth of tool-call fragments, forwarded as-is.

    A fragment is not a call and is not parseable on its own: `id` and
    `function.name` arrive once, and `function.arguments` arrives as a
    string split at arbitrary points -- `{"loc` in one frame and
    `ation": "NYC"}` in the next is normal. Anything that tries to read
    a single fragment as JSON will fail on almost every stream.
    """
    if not isinstance(raw, list):
        return []
    out: list[ToolCallDelta] = []
    for position, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        raw_fn = item.get("function")
        fn: dict[str, Any] = raw_fn if isinstance(raw_fn, dict) else {}
        out.append(
            ToolCallDelta(
                index=int(index) if isinstance(index, int) else position,
                id=item.get("id"),
                # `function` is the only const in the contract, so it is
                # always that rather than an echo of whatever upstream
                # sent -- a value we cannot represent is a value we must
                # not invent.
                type="function",
                function=Function1(name=fn.get("name"), arguments=fn.get("arguments"))
                if fn
                else None,
            )
        )
    return out


def _accumulate_tool_calls(parts: dict[int, dict[str, str]], raw: Any) -> None:
    """Fold one frame's fragments into the calls being assembled."""
    if not isinstance(raw, list):
        return
    for position, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        key = int(index) if isinstance(index, int) else position
        slot = parts.setdefault(key, {"id": "", "name": "", "arguments": ""})
        if item.get("id"):
            slot["id"] = str(item["id"])
        fn = item.get("function")
        if isinstance(fn, dict):
            if fn.get("name"):
                slot["name"] = str(fn["name"])
            if fn.get("arguments"):
                slot["arguments"] += str(fn["arguments"])


def _finish_tool_calls(parts: dict[int, dict[str, str]]) -> list[ToolCall]:
    """The assembled calls, in index order.

    A fragment set with no name is dropped rather than emitted with an
    empty one: a call nothing can dispatch is worse than no call, because
    a harness will try.
    """
    calls: list[ToolCall] = []
    for key in sorted(parts):
        slot = parts[key]
        if not slot.get("name"):
            continue
        calls.append(
            ToolCall(
                id=slot.get("id") or f"call_{key}",
                type="function",
                function=FunctionCall(name=slot["name"], arguments=slot.get("arguments") or ""),
            )
        )
    return calls


def _tool_calls_from_wire(raw: Any) -> list[ToolCall]:
    """Parse a backend's `tool_calls` array into our shape.

    `arguments` stays a string. A model can emit invalid JSON and
    OpenAI's contract preserves what it actually said rather than
    failing the whole response; parsing here would move that failure to
    the one place least able to report it usefully.
    """
    if not isinstance(raw, list):
        return []
    calls: list[ToolCall] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        fn = item.get("function") or {}
        name = fn.get("name") if isinstance(fn, dict) else None
        if not name:
            continue
        args = fn.get("arguments") if isinstance(fn, dict) else None
        calls.append(
            ToolCall(
                id=str(item.get("id") or f"call_{i}"),
                type="function",
                function=FunctionCall(name=str(name), arguments=str(args or "")),
            )
        )
    return calls


def _vectors_from_envelope(body: Any, *, expected: int) -> list[list[float]]:
    """`data[].embedding`, ordered by `index`.

    **Sorted by `index` rather than trusted in arrival order.** OpenAI
    documents that `data` may come back out of order, and the caller has
    no other way to match a vector to its input -- an embedding carries
    no identity. Getting this wrong would silently pair every vector
    with the wrong text, which is the failure that looks like a working
    system producing bad search results.
    """
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list) or not data:
        raise CliError(f"openai_compat_http returned no embeddings: {body!r}")
    rows: list[tuple[int, list[float]]] = []
    for position, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise CliError(f"openai_compat_http returned a malformed embedding entry: {entry!r}")
        vector = entry.get("embedding")
        if isinstance(vector, str):
            # The backend answered base64 even though we asked for
            # floats. Refused rather than decoded: we never requested it,
            # so something is translating the request, and quietly
            # coping would hide that.
            raise CliError(
                "openai_compat_http returned a base64 embedding for a float request; "
                "the driver asks for floats and the gateway does any encoding"
            )
        if not isinstance(vector, list) or not vector:
            raise CliError(f"openai_compat_http returned a malformed embedding: {entry!r}")
        index = entry.get("index")
        rows.append((index if isinstance(index, int) else position, [float(x) for x in vector]))
    rows.sort(key=lambda r: r[0])
    if len(rows) != expected:
        raise CliError(
            f"openai_compat_http returned {len(rows)} embeddings for {expected} inputs; "
            "order and count are the only way a caller can match vectors to text"
        )
    return [v for _, v in rows]


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
