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
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from .._generated.models import (
    BackendKind,
    ChatAnnotation,
    ChatLogprobs,
    ChatTokenLogprob,
    ConfigField,
    ConfigFieldShowWhen,
    ConfigValueType,
    DriverModel,
    EmbedResponse,
    FinishReason,
    Function1,
    FunctionCall,
    GenerateRequest,
    GenerateResponse,
    ImagePartial,
    ImageRequest,
    ImageResponse,
    ModerateRequest,
    ModerateResponse,
    ModerationPartType,
    Role,
    SpeakRequest,
    SpeechFormat,
    Stage,
    StreamProgress,
    ToolCall,
    ToolCallDelta,
    TranscribeRequest,
    TranscribeResponse,
    Usage,
    VideoJob,
    VideoRequest,
)
from .._http import client_for
from ..audio_out import AudioAssembly
from ..failures import request_id, retry_after
from ..images import attachment_kinds, content_wire
from ..images_out import ImageRefusal as ImageOutRefusal
from ..images_out import (
    completed_from,
    event_kind,
    format_name,
    partial_from,
)
from ..images_out import response_from as images_response_from
from ..speech import (
    ALL_FORMATS,
    OPENROUTER_FORMATS,
    SpeechRefusal,
    refuse_format,
    streaming_wav_header,
)
from ..transcription import response_from
from ..videos_out import VideoRefusal
from ..videos_out import job_from as video_job_from
from ._catalogue import (
    _LIST_TIMEOUT,
    _SHOW_CONCURRENCY,
    _SHOW_TIMEOUT,
    Catalogue,
    CatalogueError,
    EngineDefaults,
    from_lmstudio,
    from_openai_list,
    from_openrouter,
    ollama_entry,
    upstream_words,
    with_openrouter_images,
    with_openrouter_videos,
)
from ._subprocess import BackendTimeout, CliError
from ._thinking import ThinkingFilter, apply_thinking_mode, strip_thinking_blocks
from .base import (
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_STREAM_STALL_SECONDS,
    Chunk,
    ModelNotServed,
    ModelRequired,
    TokenCountUnsupported,
    refuse_unsupported_settings,
    resolve_single_model,
)

# A count is two small HTTP calls handled beside the slots, never a
# prefill, so it gets a short deadline of its own rather than the
# minutes a generation is allowed.
_COUNT_TIMEOUT_SECONDS = 30.0

log = logging.getLogger(__name__)
_IMAGE_ERROR_HINT = (
    "Image request refused by the backend. Check the loaded vision model/projector "
    "and available context; try a smaller image or shorter conversation. "
    "Upstream body omitted to protect attachment data."
)
_AUDIO_ERROR_HINT = (
    "Audio request refused by the backend. Check that the model takes audio input and "
    "the clip's format; try a shorter clip or conversation. "
    "Upstream body omitted to protect attachment data."
)
_FILE_ERROR_HINT = (
    "File request refused by the backend. Check that the model reads PDFs; try a smaller "
    "file or shorter conversation. Upstream body omitted to protect attachment data."
)


def _attachment_hint(kinds: frozenset[str]) -> str | None:
    """What to say instead of an upstream error body that may echo an attachment."""
    if "image" in kinds:
        return _IMAGE_ERROR_HINT
    if "audio" in kinds:
        return _AUDIO_ERROR_HINT
    if "file" in kinds:
        return _FILE_ERROR_HINT
    return None


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

# The fewest seconds between two `working` progress frames made from SSE
# keepalive comments. OpenRouter sends one every few seconds while a
# model is queued or thinking; forwarding each would be a frame per
# comment for a signal that only needs to say "still there".
_KEEPALIVE_PROGRESS_SECONDS = 2.0

# The answers that mean "this server has no such endpoint", as opposed to
# "not now". Only these settle that a backend is not `llama-server`.
_NO_SUCH_PATH = frozenset({404, 405, 410, 501})


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
    return "max_completion_tokens" if _is_openai_endpoint(base_url) else "max_tokens"


def _is_openai_endpoint(base_url: str) -> bool:
    """OpenAI's own API, rather than something that speaks its shape.

    The one backend behind this engine known to REJECT the local-engine
    extensions: it answers `top_k` with "Unrecognized request argument
    supplied", and a history message carrying `reasoning_content` with
    an unexpected-property 400. Everything else -- llama.cpp, vLLM,
    Ollama, LM Studio -- either reads them or ignores them.
    """
    return "openai.com" in base_url.lower()


# Settings OpenAI's own endpoint does not take. Refused when explicit and
# left out of `supported_settings`, so the gateway routes around this
# backend instead of trying it and failing.
_LOCAL_ENGINE_SETTINGS = frozenset({"topK", "minP"})

# P2c (2026-09-28): settings only a hosted API is known to take -- OpenAI's
# own, or an OpenRouter model whose listing names them (186 list
# `reasoning_effort`, 149 `logprobs`, 141 `logit_bias`, measured). A local
# engine or a provider whose list says nothing per model is not claimed
# for any of them, so a request that sets one is routed where it will be
# honoured rather than sent where it may be dropped.
_HOSTED_SETTINGS = frozenset(
    {"logprobs", "logitBias", "reasoningEffort", "verbosity", "prediction", "webSearchOptions"}
)

# Hints: they change where or how cheaply an answer is made, never what it
# says. Carried to OpenAI's own API, the only backend known to take them
# (no OpenRouter model lists any, measured), and dropped elsewhere with a
# warning once, rather than refused.
_HINTS: tuple[tuple[str, str], ...] = (
    ("promptCacheKey", "prompt_cache_key"),
    ("promptCacheRetention", "prompt_cache_retention"),
    ("serviceTier", "service_tier"),
    ("safetyIdentifier", "safety_identifier"),
)

# Settings a fixed-sampler model rejects along with `temperature`: OpenAI's
# reasoning models refuse the penalties as they refuse the sampler itself.
_FIXED_SAMPLER_SETTINGS = frozenset({"temperature", "topP", "frequencyPenalty", "presencePenalty"})

# Every request setting this engine can put on the wire at all, in the
# order `supported_settings` reports them.
_ENGINE_SETTINGS: tuple[str, ...] = (
    "maxTokens",
    "temperature",
    "topP",
    "seed",
    "stop",
    "tools",
    "toolChoice",
    "responseFormat",
    "topK",
    "minP",
    "frequencyPenalty",
    "presencePenalty",
    "parallelToolCalls",
    "logprobs",
    "logitBias",
    "reasoningEffort",
    "verbosity",
    "prediction",
    "webSearchOptions",
)
_ALL_ENGINE_SETTINGS = frozenset(_ENGINE_SETTINGS)


def _logprobs_of(raw: Any) -> ChatLogprobs | None:
    """A choice's `logprobs`, OpenAI's shape (measured through OpenRouter,
    2026-09-28), or None when there is none or it is malformed. A bad
    `logprobs` is not worth failing an answer over."""
    if not isinstance(raw, dict) or not raw.get("content"):
        return None
    try:
        return ChatLogprobs.model_validate(raw)
    except ValueError:
        log.debug("openai_compat_http: logprobs in an unexpected shape (omitted)")
        return None


def _annotations_of(raw: Any) -> list[ChatAnnotation]:
    """The `url_citation` annotations on a message or delta. OpenRouter's
    `file` annotations are its own cache of a parsed PDF, not a citation,
    and no OpenAI client reads them, so they are not carried."""
    out: list[ChatAnnotation] = []
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, dict) and item.get("type") == "url_citation":
            try:
                out.append(ChatAnnotation.model_validate(item))
            except ValueError:
                log.debug("openai_compat_http: a citation in an unexpected shape (omitted)")
    return out


def _reasoning_of(obj: dict[str, Any]) -> str | None:
    """A message's or delta's separately-reported reasoning, if any.

    **Two spellings, both measured.** llama.cpp b10948 sends
    `reasoning_content` (DeepSeek's name) and vLLM 0.29 sends
    `reasoning` -- it renamed the field and now treats the old one as a
    deprecated alias on input. A reader of only one of them discards
    the other engine's thinking exactly as this adapter discarded both
    before 2026-09-23.
    """
    for key in ("reasoning_content", "reasoning"):
        value = obj.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _stop_sequence(choice: dict[str, Any], request: GenerateRequest) -> str | None:
    """Which requested stop string ended the answer, if the backend says.

    vLLM puts it in `stop_reason`; **llama.cpp b10948 does not report it
    at all** (measured: its choice carries `finish_reason`, `index` and
    `message`), so behind llama.cpp this is always None. A `stop_reason`
    that is a token id -- vLLM's EOS case -- or a string the caller never
    asked for is not a stop sequence, and naming it as one would put a
    value in the caller's hands that matches nothing it sent.
    """
    reason = choice.get("stop_reason")
    if isinstance(reason, str) and request.stop and reason in request.stop:
        return reason
    return None


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


def _prompt_progress(read: dict[str, Any]) -> StreamProgress | None:
    """llama.cpp's `prompt_progress` in the contract's words, or None.

    Captured from b11215: `{"total", "cache", "processed", "time_ms"}`,
    `processed` counting the cached tokens too, so it runs from `cache`
    to `total`. A frame missing either end is not reported rather than
    reported as zero.
    """

    def count(key: str) -> int | None:
        value = read.get(key)
        return int(value) if isinstance(value, int | float) and value >= 0 else None

    total, processed = count("total"), count("processed")
    if total is None or processed is None:
        return None
    return StreamProgress(
        stage=Stage.prompt,
        promptTokens=total,
        cachedTokens=count("cache"),
        processedTokens=processed,
        elapsedMs=count("time_ms"),
    )


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


def _stalled(stall_seconds: float, emitted_chars: int) -> BackendTimeout:
    """The backend WAS answering and went silent on an open socket.

    Not `_timed_out`'s case: there the whole request took too long and
    the engine is presumed still computing. Here the engine had started
    streaming -- it owed the next token within `streamStallSeconds` and
    sent nothing, no token, no finish reason, no close. Holding on
    costs the caller the rest of the request budget for an answer that
    is not coming; ending it names what happened.

    A `BackendTimeout` on purpose: mid-stream the gateway is committed
    (M10), so this can only ever surface as the stream's terminal
    `error` frame with the 504 identity -- never as a natural end, and
    never as a 502 inviting a cascade that cannot happen.
    """
    return BackendTimeout(
        f"openai_compat_http stream went silent for {stall_seconds:g}s mid-answer, "
        f"after {emitted_chars} characters. The backend stopped producing tokens without "
        f"closing the stream or sending a finish reason. If this backend legitimately "
        f"pauses that long between tokens, raise streamStallSeconds on this driver "
        f"(0 disables the check).",
        limit_seconds=stall_seconds,
    )


def classify_openai_model(model_id: str) -> list[str]:
    """Sort an id from OpenAI's own `/v1/models` into the surfaces it answers.

    `api.openai.com` lists every model on the account with nothing per model
    (call P1-3), so an account over it can only go by the id -- the same
    heuristic the model dropdown has used since M0, turned from a filter
    into a sorter so a speech or image model is kept, under the surface its
    door will serve (P1-4), instead of dropped. An id it cannot place
    serves nothing rather than being guessed into chat.
    """
    lowered = model_id.lower()
    if "embedding" in lowered or lowered.startswith(("text-similarity-", "text-search-")):
        return ["embeddings"]
    if "moderation" in lowered:
        return ["moderation"]
    if lowered.startswith("tts-") or "-tts" in lowered:
        return ["speech"]
    # Only whisper translates: OpenAI answers `/v1/audio/translations` 404 for
    # `gpt-4o-mini-transcribe` and `gpt-4o-transcribe` (measured 2026-09-28).
    if lowered.startswith("whisper-"):
        return ["transcription", "translation"]
    if "transcribe" in lowered:
        return ["transcription"]
    # `chatgpt-image-latest` is an image model too (OpenAI's edit models;
    # on the account Troy gave, measured 2026-09-28) and filed as nothing.
    if lowered.startswith(("dall-e-", "gpt-image-", "chatgpt-image-")):
        return ["image"]
    if lowered.startswith("sora"):
        return ["video"]
    return ["chat"] if _is_plausible_chat_model(model_id) else []


@dataclass(frozen=True)
class _Target:
    """The model one request is for: what callers see, what the backend is
    asked for, and the one per-model fact the payload shaping reads."""

    id: str
    upstream: str
    temperature_fixed: bool
    #: The account's catalogue entry; None for a single-model driver.
    entry: DriverModel | None = None


class OpenAiCompatibleHttpEngine:
    """OpenAI-compatible HTTP engine. Provider-agnostic."""

    #: This engine can front a runtime the agent supervises, addressed by
    #: name rather than by URL. The CLI engines cannot — a subscription
    #: is not a runtime — so `app.build_engine_with` consults this before
    #: resolving `runtimeName` at all.
    follows_runtimes = True

    #: With no `modelId`, this engine is a provider ACCOUNT (P1) serving
    #: every model its backend lists. `app.build_engine_with` hands such an
    #: engine the provider key and where to keep its model list.
    serves_accounts = True

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
        model_id: str | None = "gpt-4o",
        upstream_model_id: str | None = None,
        timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        stall_seconds: float = DEFAULT_STREAM_STALL_SECONDS,
        fixed_temperature_pattern: re.Pattern[str] | None = None,
        backend_kind: BackendKind = BackendKind.openai_api,
        thinking_mode: str = "auto",
        auth_required: bool = True,
        filter_models: bool = True,
        runtime: str | None = None,
        catalogue_source: str = "openai",
        require_parameters: bool = False,
        get: Callable[[str], Any] | None = None,
        catalogue_path: Path | None = None,
        provider: str | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if auth_required and not resolved_key:
            raise CliError(
                "openai_compat_http engine has no API key — set `apiKey` in "
                "config or export OPENAI_API_KEY in the environment."
            )
        self._api_key = resolved_key
        self._base_url = base_url.rstrip("/")
        #: None makes this driver a provider ACCOUNT (P1): it serves every
        #: model its backend lists, and each request names one. A single
        #: model is the degenerate case and behaves exactly as before.
        self._model_id = model_id
        #: What the backend is actually asked for. Resolved ONCE, here,
        #: and used at exactly the wire boundary (`_payload_for`, the
        #: embed payload, and the probes that match a backend's own
        #: model listing). Everything a caller sees — /v1/info,
        #: GenerateResponse, the stream's terminal frame, EmbedResponse —
        #: reports `_model_id`, the public identity the gateway routes
        #: on. Defaulting to the public id keeps every install that
        #: predates the split byte-identical on the wire.
        self._upstream_model_id = upstream_model_id or model_id or ""
        self._timeout_seconds = timeout_seconds
        self._stall_seconds = stall_seconds
        self._fixed_temperature_pattern = fixed_temperature_pattern
        self._warned_dropped: set[tuple[str, str]] = set()
        # Matched against the upstream id — the pattern describes what
        # the BACKEND rejects, and the backend only ever sees that name.
        self._temperature_is_fixed = model_id is not None and self._fixes_temperature(
            self._upstream_model_id
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
        #: OpenRouter only: ask it to route to a provider that honours every
        #: setting sent. Its per-model `supported_parameters` is a union
        #: over providers (measured: gpt-oss-20b has 12 endpoints, one takes
        #: tools but not response_format, another the reverse), so without
        #: this a setting the caller asked for can be dropped by whichever
        #: provider it picks -- exactly what A2 forbids.
        self._require_parameters = require_parameters
        self._catalogue_source = catalogue_source
        #: The account's model list, or None for a single-model driver.
        self.catalogue: Catalogue | None = None
        if model_id is None:
            self.catalogue = Catalogue(
                source=catalogue_source,
                origin=f"{provider or catalogue_source}|{self._base_url}",
                fetch=self._fetch_catalogue,
                get=get or (lambda _key: None),
                path=catalogue_path,
            )
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
        #: Whether the backend answered as `llama-server`, the one backend
        #: that says how far it has read a prompt. Learned from `/props`
        #: by the context probe (or on the first stream that asks), and
        #: None until something definite came back: a hosted API refuses
        #: `return_progress` as an unknown field, so it is sent only on a
        #: yes.
        self._llama_cpp: bool | None = None
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
            ConfigField(
                key="streamStallSeconds",
                label="Stream stall timeout",
                description=(
                    "How long a streamed answer may go silent between "
                    "tokens before this driver declares the backend "
                    "stalled and ends the stream with an error. The "
                    "clock starts at the first token — the quiet model "
                    "load and prompt reading before it are covered by "
                    "the backend timeout instead — so a slow machine is "
                    "not punished for a slow start, only for going "
                    "quiet mid-answer. Be generous for a model running "
                    "on the processor, where tokens can be seconds "
                    "apart. 0 turns the check off. Set per driver: a "
                    "fast card and a CPU box can each hold their own "
                    "number."
                ),
                category="network",
                valueType=ConfigValueType.duration,
                default=DEFAULT_STREAM_STALL_SECONDS,
                minimum=0,
                maximum=600,
                requiresRestart=True,
                showWhen=show_when,
            ),
            ConfigField(
                key="catalogueInclude",
                label="Models to use",
                description=(
                    "When Model is left empty this driver uses every model "
                    "the provider lists. Keep only the ones matching these "
                    "patterns: `*` matches anything, `/` included, so "
                    "`anthropic/*` keeps one vendor's models on OpenRouter "
                    "and `qwen3*` every Qwen 3 in Ollama. The default `*` "
                    "keeps everything. Takes effect without a restart."
                ),
                category="adapter",
                valueType=ConfigValueType.string_list,
                default=["*"],
                requiresRestart=False,
                showWhen=show_when,
            ),
            ConfigField(
                key="catalogueExclude",
                label="Models to leave out",
                description=(
                    "Patterns for models to leave out even when they match "
                    "Models to use, for example `*:free` or `*-preview`. "
                    "Takes effect without a restart."
                ),
                category="adapter",
                valueType=ConfigValueType.string_list,
                default=[],
                requiresRestart=False,
                showWhen=show_when,
            ),
            ConfigField(
                key="catalogueRefreshMinutes",
                label="Model list refresh",
                description=(
                    "How often, in minutes, this driver reads the "
                    "provider's model list again when Model is left empty. "
                    "A failed read keeps the list it already has."
                ),
                category="adapter",
                valueType=ConfigValueType.integer,
                default=60,
                minimum=1,
                maximum=1440,
                requiresRestart=False,
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
        catalogue_source: str = "openai",
        require_parameters: bool = False,
        catalogue_path: Path | None = None,
        provider: str | None = None,
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
        # `0 or DEFAULT` is DEFAULT -- the seed=0 mistake (R3.4). Zero
        # here means "the operator turned the stall check off" and must
        # arrive as zero, so only absence falls back.
        raw_stall = get("streamStallSeconds")
        return cls(
            api_key=str(get("apiKey") or "") or None,
            base_url=base_url,
            # **Empty is an account, not "gpt-4o"** (P1-2). The old default
            # sent a model nobody chose to whatever the key belonged to;
            # an empty model now means "serve every model this backend
            # lists", which is what leaving it empty on an aggregator meant.
            model_id=str(get("modelId") or "").strip() or None,
            upstream_model_id=str(get("upstreamModelId") or "") or None,
            timeout_seconds=float(get("requestTimeoutSeconds") or DEFAULT_REQUEST_TIMEOUT_SECONDS),
            stall_seconds=(
                DEFAULT_STREAM_STALL_SECONDS
                if raw_stall is None or raw_stall == ""
                else float(raw_stall)
            ),
            fixed_temperature_pattern=fixed_temperature_pattern,
            backend_kind=backend_kind,
            thinking_mode=str(get("thinkingMode") or "auto"),
            auth_required=auth_required,
            filter_models=filter_models,
            runtime=runtime_name if runtime_url else None,
            catalogue_source=catalogue_source,
            require_parameters=require_parameters,
            get=get,
            catalogue_path=catalogue_path,
            provider=provider,
        )

    # -- which model a request is for (P1) ---------------------------------

    @property
    def model_id(self) -> str | None:
        """The one model a single-model driver serves; None for an account."""
        return self._model_id

    def _fixes_temperature(self, upstream: str) -> bool:
        pattern = self._fixed_temperature_pattern
        return pattern is not None and bool(pattern.match(upstream))

    def resolve_model(self, requested: str | None) -> _Target:
        """The model a request is for, or `ModelNotServed` / `ModelRequired`.

        Called before any backend work, by the routes (for the 404) and by
        every engine method (for the wire), so the two cannot disagree.
        """
        if self.catalogue is None:
            resolve_single_model(self._model_id, requested)
            return _Target(
                id=self._model_id or "",
                upstream=self._upstream_model_id,
                temperature_fixed=self._temperature_is_fixed,
            )
        if not requested:
            raise ModelRequired(
                "This driver is a provider account serving "
                f"{len(self.catalogue.exposed())} models; a request must name one "
                "in `model`."
            )
        entry = self.catalogue.find(requested)
        if entry is None:
            raise ModelNotServed(requested, served=len(self.catalogue.exposed()))
        upstream = entry.upstreamId or entry.id
        return _Target(
            id=entry.id,
            upstream=upstream,
            temperature_fixed=self._fixes_temperature(upstream),
            entry=entry,
        )

    def engine_defaults(self, upstream: str = "") -> EngineDefaults:
        """What a model inherits where its provider's listing says nothing."""
        return EngineDefaults(
            supported_settings=self._supported_for(
                _Target(
                    id=upstream,
                    upstream=upstream,
                    temperature_fixed=self._fixes_temperature(upstream),
                )
            ),
            tool_calling=self.supports_tool_calling,
            streaming=self.supports_streaming,
        )

    async def _fetch_catalogue(self) -> list[DriverModel]:
        """Read this account's list from the provider's own listing."""
        client = self._client()
        headers = self._headers()
        source = self._catalogue_source

        async def read(path: str) -> Any:
            response = await client.get(path, headers=headers, timeout=_LIST_TIMEOUT)
            if response.status_code >= 400:
                raise CatalogueError(
                    f"{source} refused the model list ({response.status_code}): "
                    f"{upstream_words(response)}"
                )
            try:
                return response.json()
            except ValueError as e:
                raise CatalogueError(f"{source}'s model list was not JSON") from e

        if source == "openrouter":
            # The ACCOUNT's list, every modality. `/v1/models` is the
            # public 458 that output text; this is the 625 this key can
            # call (measured 2026-09-27).
            models = from_openrouter(await read("/v1/models/user?output_modalities=all"))
            # P4: the image settings live only on the images listing, and it
            # is supplementary: a failed read of it must not unroute the
            # account's chat models, which on a first boot, with no last good
            # list, would leave it serving nothing (the P2 gate's fixture
            # has no images listing, and went dark). Without it, an
            # image-only model keeps `image` from its output modalities, with
            # no capabilities: the backend checks its settings, and a stream
            # or a mask, which must be confirmed, is routed nowhere.
            try:
                images = await read("/v1/images/models")
            except (CatalogueError, httpx.HTTPError) as e:
                log.warning(
                    "OpenRouter's image model list could not be read (%s); image models are "
                    "served without their listed settings until the next refresh",
                    e,
                )
            else:
                models = with_openrouter_images(models, images)
            # P5: the video settings too, and supplementary in the same way.
            try:
                videos = await read("/v1/videos/models")
            except (CatalogueError, httpx.HTTPError) as e:
                log.warning(
                    "OpenRouter's video model list could not be read (%s); video models are "
                    "served without their listed settings until the next refresh",
                    e,
                )
            else:
                models = with_openrouter_videos(models, videos)
            return [self._with_fixed_temperature(m) for m in models]
        if source == "ollama":
            body = await read("/api/tags")
            listed = body.get("models") if isinstance(body, dict) else None
            names = [
                m.get("name") or m.get("model")
                for m in (listed if isinstance(listed, list) else [])
                if isinstance(m, dict)
            ]
            gate = asyncio.Semaphore(_SHOW_CONCURRENCY)

            async def show(name: str) -> dict[str, Any] | None:
                async with gate:
                    try:
                        answer = await client.post(
                            "/api/show",
                            headers=headers,
                            json={"model": name},
                            timeout=_SHOW_TIMEOUT,
                        )
                        if answer.status_code >= 400:
                            return None
                        shown = answer.json()
                        return shown if isinstance(shown, dict) else None
                    except (httpx.HTTPError, ValueError):
                        return None

            valid = [n for n in names if isinstance(n, str) and n]
            shows = await asyncio.gather(*(show(n) for n in valid))
            return [
                ollama_entry(name, shown, self.engine_defaults(name))
                for name, shown in zip(valid, shows, strict=True)
            ]
        if source == "lmstudio":
            try:
                return from_lmstudio(await read("/api/v0/models"), self.engine_defaults())
            except CatalogueError:
                # An LM Studio without the v0 REST API still speaks OpenAI's.
                pass
        body = await read("/v1/models")
        classify = classify_openai_model if self._filter_models else None
        return [
            self._with_fixed_temperature(m)
            for m in from_openai_list(body, self.engine_defaults(), classify=classify)
        ]

    def _with_fixed_temperature(self, model: DriverModel) -> DriverModel:
        """Drop the sampler settings from a model whose provider rejects them."""
        if model.capabilities is None or not self._fixes_temperature(model.upstreamId or model.id):
            return model
        caps = model.capabilities
        caps.supportedSettings = [
            s for s in (caps.supportedSettings or []) if s not in _FIXED_SAMPLER_SETTINGS
        ]
        return model

    @property
    def supported_settings(self) -> list[str]:
        return self._supported_for(
            _Target(
                id=self._model_id or "",
                upstream=self._upstream_model_id,
                temperature_fixed=self._temperature_is_fixed,
            )
        )

    def _supported_for(self, target: _Target) -> list[str]:
        """The settings this engine can carry for one model.

        For an account over OpenRouter it is what that model's listing
        names, intersected with what this engine can send at all.
        """
        unsupported = self._unsupported_settings(target)
        return [field for field in _ENGINE_SETTINGS if field not in unsupported]

    def _unsupported_settings(self, target: _Target) -> frozenset[str]:
        """What this backend cannot carry, whoever asks.

        One answer read by both `supported_settings` (so the gateway
        routes around this backend) and `_payload_for` (so an explicit
        request that reaches it anyway is refused before any HTTP). Two
        separate lists would be two chances to disagree.
        """
        unsupported: set[str] = set()
        if target.temperature_fixed:
            unsupported |= _FIXED_SAMPLER_SETTINGS
        if _is_openai_endpoint(self._base_url):
            unsupported |= _LOCAL_ENGINE_SETTINGS
        caps = target.entry.capabilities if target.entry is not None else None
        listed_by_provider = self._catalogue_source == "openrouter" and caps is not None
        if not _is_openai_endpoint(self._base_url) and not listed_by_provider:
            unsupported |= _HOSTED_SETTINGS
        if listed_by_provider and caps is not None:
            # The listing is the authority for what this model accepts; a
            # setting it does not name is one OpenRouter would drop.
            listed = set(caps.supportedSettings or [])
            unsupported |= _ALL_ENGINE_SETTINGS - listed
        return frozenset(unsupported)

    def _payload_for(self, request: GenerateRequest, target: _Target) -> dict[str, Any]:
        """The chat-completions body for this request.

        Shared by `generate` and `stream` so the two cannot drift on
        param shaping -- which would be an especially quiet bug, since
        the only symptom would be a streamed answer differing from a
        non-streamed one for the same request.
        """
        unsupported = self._unsupported_settings(target)
        if unsupported:
            refuse_unsupported_settings(request, unsupported=set(unsupported))
        # Apply the operator's thinkingMode by mutating the system
        # message before role-coercion. See engines/_thinking.py for
        # the per-mode directives — `off` is the one that suppresses
        # inline `<think>` blocks leaking into chat responses.
        messages = apply_thinking_mode(list(request.messages), self._thinking_mode)
        payload: dict[str, Any] = {
            "model": target.upstream,
            "messages": _to_openai_messages(
                messages, send_reasoning=not _is_openai_endpoint(self._base_url)
            ),
        }
        if request.maxTokens is not None:
            payload[_max_tokens_field_for(self._base_url)] = request.maxTokens
        if request.temperature is not None and not target.temperature_fixed:
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
        if request.topP is not None and not target.temperature_fixed:
            payload["top_p"] = float(request.topP)
        elif request.topP is not None:
            self._warn_dropped("top_p", target)
        # `is not None` and not truthiness: **`seed=0` is a real seed**
        # and a falsy one, and dropping it would answer a request for a
        # reproducible result with a different answer every time --
        # which is the whole of what this field was doing before today.
        if request.seed is not None:
            payload["seed"] = int(request.seed)
        if request.stop:
            payload["stop"] = list(request.stop)
        # **Carried since 2026-09-23**, with no field for any of them
        # before -- so the gateway refused all five with a 400, although
        # llama.cpp accepted every one of them on the capture that
        # scoped this change. `is not None` throughout, for the seed=0
        # reason: `top_k: 0` disables the cut and `parallel_tool_calls:
        # false` asks for one call a turn, and both are falsy.
        #
        # What reaches here for a backend that cannot take one was not
        # explicit (an explicit one was refused above), so it is omitted
        # and said once -- the `temperature` rule, applied to its
        # neighbours.
        for field, key, value in (
            ("topK", "top_k", request.topK),
            ("minP", "min_p", request.minP),
            ("frequencyPenalty", "frequency_penalty", request.frequencyPenalty),
            ("presencePenalty", "presence_penalty", request.presencePenalty),
            ("parallelToolCalls", "parallel_tool_calls", request.parallelToolCalls),
        ):
            if value is None:
                continue
            if field in unsupported:
                self._warn_dropped(key, target)
                continue
            payload[key] = value
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
        if self._require_parameters and request.callerSettings:
            payload["provider"] = {"require_parameters": True}
        if request.logprobs is not None:
            payload["logprobs"] = request.logprobs
            if request.topLogprobs is not None:
                payload["top_logprobs"] = request.topLogprobs
        if request.logitBias is not None:
            payload["logit_bias"] = dict(request.logitBias)
        if request.reasoningEffort is not None:
            payload["reasoning_effort"] = request.reasoningEffort.value
        if request.verbosity is not None:
            payload["verbosity"] = request.verbosity.value
        if request.prediction is not None:
            payload["prediction"] = request.prediction.model_dump(mode="json", exclude_none=True)
        if request.webSearchOptions is not None:
            payload["web_search_options"] = request.webSearchOptions.model_dump(
                mode="json", exclude_none=True
            )
        for ours, theirs in _HINTS:
            value = getattr(request, ours)
            if value is None:
                continue
            if _is_openai_endpoint(self._base_url):
                payload[theirs] = getattr(value, "value", value)
            else:
                self._warn_dropped(theirs, target)
        if request.audioOutput is not None:
            # **Always `pcm16`, whatever was asked** (P2b): every
            # audio-output model behind an account answers audio only on
            # a stream and only as `pcm16` (measured 2026-09-28), and
            # Lyria sends MP3 whatever it is asked. The caller's format
            # decides what `generate` makes of the stream, not what is
            # asked of the backend.
            payload["modalities"] = ["text", "audio"]
            payload["audio"] = {"voice": request.audioOutput.voice, "format": "pcm16"}
        return payload

    def _public_model_id(self, reported: object, target: _Target) -> str:
        """The model identity a caller sees on a response.

        When this engine translates (`upstreamModelId` set and
        different), the backend's own name for itself — `default_model`,
        an absolute path — is exactly what must NOT be echoed: two MLX
        runtimes serving different models would collapse into one name,
        which is the collision the split exists to prevent. So a
        translating engine always answers with the public id.

        When nothing is translated, the backend's echo is kept, as it
        always was: a backend that reports serving something other than
        what was configured is telling the truth about what answered,
        and hiding that would un-diagnose a misconfigured endpoint.
        """
        if target.entry is not None:
            # **An account always answers with the id that was asked for.**
            # OpenRouter answers an alias under its target's id
            # (`~z-ai/glm-flash-latest` came back as `z-ai/glm-5.3-flash`,
            # measured), and echoing that would name a model the gateway
            # does not route and the caller did not ask for.
            return target.id
        if target.upstream != target.id:
            return target.id
        return str(reported or target.id)

    def _warn_dropped(self, field: str, target: _Target) -> None:
        """Say it once per field per engine, not once per request.

        The contract promises a parameter we cannot carry is "dropped
        with a warning". A warning on every request would be a log line
        per token-generating call on a busy backend, which is how a
        real warning becomes something an operator filters out.
        """
        if (target.id, field) in self._warned_dropped:
            return
        self._warned_dropped.add((target.id, field))
        log.warning(
            "model %r does not accept `%s`; the driver is omitting it on every "
            "request and letting the model use its own default. This is said "
            "once per parameter for the life of this engine.",
            target.id,
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
        if request.audioOutput is not None:
            return await self._assembled(request)
        started = time.perf_counter()
        target = self.resolve_model(request.model)
        payload = self._payload_for(request, target)
        hint = _attachment_hint(attachment_kinds(request.messages))
        attachment_request = hint is not None

        # DEBUG-level full-payload trace. The gateway's copy-trace
        # captures what we sent it; this captures what WE send upstream
        # (post role-coercion, post-thinking-directive injection,
        # post-param-shaping). When operators flip to DEBUG to chase
        # "is the LLM actually seeing what I think it's seeing", this
        # is the load-bearing log line. Auth header omitted on purpose.
        if log.isEnabledFor(logging.DEBUG) and not attachment_request:
            log.debug(
                "openai_compat_http → POST %s/v1/chat/completions\n%s",
                self._base_url,
                json.dumps(payload, indent=2, ensure_ascii=False),
            )

        headers = {
            **self._headers(),
            **({"X-Request-ID": str(request.requestId)} if request.requestId else {}),
        }

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
            if log.isEnabledFor(logging.DEBUG) and not attachment_request:
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
                f"{_redact(response.text[:500]) if hint is None else hint}",
                upstream_status=response.status_code,
                retry_after_seconds=retry_after(response.headers.get("Retry-After")),
            )

        try:
            body = response.json()
        except ValueError as e:
            raise CliError("openai_compat_http returned non-JSON") from e

        if log.isEnabledFor(logging.DEBUG) and not attachment_request:
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
        reasoning = _reasoning_of(message)
        # **`content` is null on a tool-call-only turn**, and this used
        # to raise on exactly that response -- the concrete way a tool
        # call failed here before the contract carried one. **And on a
        # reasoning-only turn**: vLLM gives a model that thought until
        # `max_tokens` `content: null` with everything in `reasoning`,
        # which this raised on until 2026-09-23 -- a 502 the gateway
        # cascaded on, for a reply that was perfectly well formed. A
        # response with no text, no calls and no reasoning is still a
        # backend malfunction, not an empty answer.
        if not isinstance(content, str):
            if not tool_calls and not reasoning:
                raise CliError("openai_compat_http response missing string content")
            content = None
        elif self._thinking_mode == "off":
            # Defensive strip of <think>...</think> when the operator opted
            # out of thinking but the model emitted tags anyway. See
            # _thinking.strip_thinking_blocks for the why.
            content = strip_thinking_blocks(content)

        stop_sequence = _stop_sequence(first, request)
        message = first.get("message")
        citations = _annotations_of(
            message.get("annotations") if isinstance(message, dict) else None
        )
        return GenerateResponse(
            content=content,
            logprobs=_logprobs_of(first.get("logprobs")),
            annotations=citations or None,
            # `off` withholds it on this path exactly as the stream
            # withholds its frames: the operator said not to show
            # reasoning, and a separate field is still showing it.
            reasoning=reasoning if self._thinking_mode != "off" else None,
            toolCalls=tool_calls or None,
            finishReason=FinishReason.stop_sequence
            if stop_sequence is not None
            else _FINISH_REASON_MAP.get(
                str(first.get("finish_reason") or "stop"), FinishReason.stop
            ),
            stopSequence=stop_sequence,
            usage=_usage_from_envelope(body.get("usage") or {}),
            requestId=request.requestId,
            backend=self.backend_kind,
            modelId=self._public_model_id(body.get("model"), target),
            latencyMs=int((time.perf_counter() - started) * 1000),
        )

    async def _assembled(self, request: GenerateRequest) -> GenerateResponse:
        """A non-streamed answer with audio: the stream, assembled (P2b).

        The backend answers audio only on a stream (OpenRouter refuses
        anything else before any provider sees it, measured), so the batch
        path is the stream path with the fragments kept.
        """
        result: GenerateResponse | None = None
        async for chunk in self.stream(request, assemble_audio=True):
            if chunk.done:
                result = chunk.result
        if result is None:
            raise CliError("openai_compat_http stream ended without a result")
        return result

    async def stream(
        self, request: GenerateRequest, *, assemble_audio: bool = False
    ) -> AsyncGenerator[Chunk, None]:
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
        target = self.resolve_model(request.model)
        payload = self._payload_for(request, target)
        hint = _attachment_hint(attachment_kinds(request.messages))
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
        report = bool(request.reportProgress)
        if report and await self._answers_as_llama_cpp():
            # llama.cpp's own flag: a `prompt_progress` object on a frame
            # per batch it reads. Measured on b11215: without it, 21 s of
            # prompt reading on the processor produced no frame at all.
            payload["return_progress"] = True
        last_keepalive = 0.0

        started = time.perf_counter()
        filtered = ThinkingFilter() if self._thinking_mode == "off" else None
        emitted: list[str] = []
        # Reasoning the backend streamed on its own channel. Forwarded
        # frame by frame and kept for the terminal `done`, unless the
        # operator's thinkingMode is `off`, in which case it is read
        # past exactly as it was for everyone before 2026-09-23.
        show_reasoning = self._thinking_mode != "off"
        reasoned: list[str] = []
        stop_sequence: str | None = None
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
        served_model = target.id
        # A spoken answer's fragments (P2b). Kept whole only for
        # `generate`, which returns one clip; a streamed caller has been
        # sent them already, so the terminal frame does not repeat them.
        audio = AudioAssembly(keep=assemble_audio) if request.audioOutput is not None else None
        # P2c: every token's log probability, for the terminal `done`, and
        # every citation the provider's search produced.
        token_logprobs: list[ChatTokenLogprob] = []
        citations: list[ChatAnnotation] = []

        client = self._client()
        try:
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                headers={
                    **self._headers(),
                    "Accept": "text/event-stream",
                    **({"X-Request-ID": str(request.requestId)} if request.requestId else {}),
                },
                json=payload,
            ) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    # Nothing has been streamed yet, so this can
                    # still be a status code -- see the same raise in
                    # `generate`.
                    detail = (
                        hint if hint is not None else _redact(body.decode("utf-8", "replace")[:500])
                    )
                    raise CliError(
                        f"openai_compat_http returned {response.status_code}: {detail}",
                        upstream_status=response.status_code,
                        retry_after_seconds=retry_after(response.headers.get("Retry-After")),
                    )
                if report:
                    # The response opened: the service has the request.
                    # The one thing every HTTP backend says, a hosted API
                    # included, and until the first token often the only
                    # thing -- a cloud reasoning model can be silent for a
                    # minute after this.
                    yield Chunk(progress=StreamProgress(stage=Stage.working))
                lines = response.aiter_lines()
                # The stall clock arms at the first DATA frame, not the
                # first line: before a token arrives the silence is a
                # legitimate model load plus prompt reading -- minutes
                # on a CPU box -- and the request budget governs it.
                # Any line at all resets the clock afterwards, so a
                # backend emitting keepalives is never called stalled.
                # A frame that only reports prompt progress does not arm
                # it: it is still the prompt being read, and a batch on a
                # large model on the processor can take longer than the
                # stall window.
                saw_first_data = False
                while True:
                    try:
                        if saw_first_data and self._stall_seconds > 0:
                            try:
                                async with asyncio.timeout(self._stall_seconds):
                                    line = await anext(lines)
                            except TimeoutError as stall:
                                if saw_terminator:
                                    # The answer is complete -- a
                                    # finish_reason arrived and only the
                                    # [DONE] marker is being withheld.
                                    # Failing a whole answer over a
                                    # missing goodbye would be the
                                    # over-correction.
                                    log.debug(
                                        "stream stalled after its finish_reason; "
                                        "treating as complete"
                                    )
                                    break
                                raise _stalled(
                                    self._stall_seconds,
                                    len("".join(emitted)) + len("".join(reasoned)),
                                ) from stall
                        else:
                            line = await anext(lines)
                    except StopAsyncIteration:
                        break
                    if report and line.startswith(":"):
                        # An SSE keepalive comment: the service saying it
                        # is still working (OpenRouter's `: OPENROUTER
                        # PROCESSING`). Said at most every couple of
                        # seconds, however often the service says it.
                        now = time.perf_counter()
                        if now - last_keepalive >= _KEEPALIVE_PROGRESS_SECONDS:
                            last_keepalive = now
                            yield Chunk(progress=StreamProgress(stage=Stage.working))
                        continue
                    data = _sse_data(line)
                    if data is None:
                        continue
                    if data == "[DONE]":
                        saw_first_data = True
                        saw_terminator = True
                        break
                    try:
                        event = json.loads(data)
                    except ValueError:
                        # A malformed frame mid-stream is not worth
                        # failing a half-delivered answer over.
                        saw_first_data = True
                        log.debug("openai_compat_http: unparseable SSE frame (contents omitted)")
                        continue
                    read = event.get("prompt_progress") if isinstance(event, dict) else None
                    if isinstance(read, dict):
                        if report:
                            progress = _prompt_progress(read)
                            if progress is not None:
                                yield Chunk(progress=progress)
                    else:
                        saw_first_data = True
                    served_model = self._public_model_id(event.get("model") or served_model, target)
                    if event.get("usage"):
                        usage_payload = event["usage"]
                    for choice in event.get("choices") or []:
                        if choice.get("finish_reason"):
                            finish_reason = str(choice["finish_reason"])
                            saw_terminator = True
                            stop_sequence = _stop_sequence(choice, request)
                        delta = choice.get("delta") or {}
                        # Logprobs ride on the choice beside `delta`, not
                        # in it (measured through OpenRouter), on the
                        # frames that carry tokens.
                        frame_logprobs = _logprobs_of(choice.get("logprobs"))
                        if frame_logprobs is not None:
                            token_logprobs.extend(frame_logprobs.content or [])
                        cited = _annotations_of(delta.get("annotations"))
                        if cited:
                            citations.extend(cited)
                            yield Chunk(annotations=cited)
                        # Reasoning rides its own frames, ahead of the
                        # answer. Before 2026-09-23 this loop looked at
                        # `content` alone, and a model that thought until
                        # `max_tokens` produced a stream with nothing in it.
                        thought = _reasoning_of(delta)
                        if thought and show_reasoning:
                            reasoned.append(thought)
                            yield Chunk(reasoning=thought)
                        # Tool-call fragments ride their own frame.
                        # Forwarded rather than accumulated here: the
                        # gateway has to emit them as OpenAI deltas
                        # anyway, and buffering them to the end of
                        # the stream would defeat the point of
                        # streaming a call the caller wants to start
                        # dispatching. We also accumulate a copy so
                        # the terminal `done` carries whole calls,
                        # for the non-streaming half of the contract.
                        if audio is not None and delta.get("audio") is not None:
                            try:
                                fragment = audio.feed(delta["audio"])
                            except ValueError as e:
                                raise CliError(f"openai_compat_http: {e}") from e
                            if fragment is not None:
                                yield Chunk(audio=fragment)
                        raw_calls = delta.get("tool_calls")
                        if raw_calls:
                            fragments = _tool_call_deltas_from_wire(raw_calls)
                            if fragments:
                                _accumulate_tool_calls(call_parts, raw_calls)
                                yield Chunk(toolCalls=fragments)
                        text = (delta.get("content")) or ""
                        if not text:
                            if frame_logprobs is not None:
                                yield Chunk(logprobs=frame_logprobs)
                            continue
                        visible = filtered.feed(text) if filtered is not None else text
                        if visible:
                            emitted.append(visible)
                            yield Chunk(text=visible, logprobs=frame_logprobs)
                        elif frame_logprobs is not None:
                            yield Chunk(logprobs=frame_logprobs)
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
                f"after {len(''.join(emitted)) + len(''.join(reasoned))} characters: "
                "the backend closed the "
                "connection mid-answer"
            )

        if filtered is not None:
            tail = filtered.flush()
            if tail:
                emitted.append(tail)
                yield Chunk(text=tail)

        content = "".join(emitted)
        tool_calls = _finish_tool_calls(call_parts)
        spoken = audio is not None and audio.heard
        clip = (
            audio.result(request.audioOutput.format)
            if assemble_audio and audio is not None and request.audioOutput is not None
            else None
        )
        yield Chunk(
            done=True,
            result=GenerateResponse(
                # None rather than "" when the turn was only tool calls or
                # only speech, so the streamed and non-streamed shapes
                # agree -- and a spoken answer's text is its transcript.
                content=content if (content or not (tool_calls or spoken)) else None,
                audio=clip,
                logprobs=ChatLogprobs(content=token_logprobs) if token_logprobs else None,
                annotations=citations or None,
                reasoning="".join(reasoned) or None,
                toolCalls=tool_calls or None,
                finishReason=FinishReason.stop_sequence
                if stop_sequence is not None
                else _FINISH_REASON_MAP.get(finish_reason, FinishReason.stop),
                stopSequence=stop_sequence,
                usage=_usage_from_envelope(usage_payload),
                requestId=request.requestId,
                backend=self.backend_kind,
                modelId=served_model,
                latencyMs=int((time.perf_counter() - started) * 1000),
            ),
        )

    def speech_formats(self, target: _Target) -> tuple[SpeechFormat, ...]:
        """The formats this model can be given in (P3a): the listing's, else
        OpenRouter's measured two plus a WAV made from pcm, else all six --
        OpenAI's own API, or a local server that will refuse what it lacks."""
        caps = target.entry.capabilities if target.entry is not None else None
        if caps is not None and caps.speechFormats:
            return tuple(caps.speechFormats)
        if self._catalogue_source == "openrouter":
            return OPENROUTER_FORMATS
        return ALL_FORMATS

    async def speak(self, request: SpeakRequest) -> AsyncGenerator[bytes, None]:
        """`POST /v1/audio/speech` upstream, streamed back as it arrives (P3a).

        **`wav` is made here from `pcm` on OpenRouter**, whose speech route
        takes `mp3` and `pcm` only (measured), with a streaming WAV header
        written before the first sample. Every other format is asked of the
        backend as it is, and **the format is always sent**: OpenRouter's
        default is `pcm` where OpenAI's is `mp3` (measured).
        """
        target = self.resolve_model(request.model)
        if target.entry is not None and "speech" not in (target.entry.surfaces or []):
            raise SpeechRefusal(
                f"{target.id!r} does not speak; choose a model with the speech surface"
            )
        formats = self.speech_formats(target)
        asked = request.format or SpeechFormat.mp3
        refuse_format(asked, formats)
        made_here = asked is SpeechFormat.wav and self._catalogue_source == "openrouter"
        payload: dict[str, Any] = {
            "model": target.upstream,
            "input": request.input,
            "voice": request.voice,
            "response_format": (SpeechFormat.pcm if made_here else asked).value,
        }
        if request.speed is not None:
            payload["speed"] = request.speed
        if request.instructions:
            payload["instructions"] = request.instructions
        client = self._client()
        try:
            async with client.stream(
                "POST",
                "/v1/audio/speech",
                headers={
                    **self._headers(),
                    "Accept": "application/octet-stream",
                    **({"X-Request-ID": str(request.requestId)} if request.requestId else {}),
                },
                json=payload,
            ) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    raise CliError(
                        f"openai_compat_http returned {response.status_code} for speech: "
                        f"{_redact(body.decode('utf-8', 'replace')[:500])}",
                        upstream_status=response.status_code,
                        retry_after_seconds=retry_after(response.headers.get("Retry-After")),
                    )
                header_sent = not made_here
                async for chunk in response.aiter_raw():
                    if not chunk:
                        continue
                    if not header_sent:
                        header_sent = True
                        yield streaming_wav_header()
                    yield chunk
        except httpx.ConnectTimeout as e:
            raise CliError(f"openai_compat_http could not connect: {e!r}") from e
        except httpx.TimeoutException as e:
            raise _timed_out(e, self._timeout_seconds, "the speech request") from e
        except httpx.HTTPError as e:
            raise CliError(f"openai_compat_http speech failed: {e!r}") from e

    async def transcribe(self, request: TranscribeRequest, audio: bytes) -> TranscribeResponse:
        """`POST /v1/audio/transcriptions` upstream, in OpenAI's multipart
        form, which OpenRouter and `llama-server` both take (measured).

        **`response_format` is sent only for `verbose`**: `json` is every
        backend's default, and `llama-server` refuses every other value, so
        leaving it out is the one request all three answer the same way.

        **`translate` asks `/v1/audio/translations` instead** (P3-4), the
        same form without `language` or timestamps, which the route has
        already refused with it.
        """
        started = time.perf_counter()
        target = self.resolve_model(request.model)
        fields: dict[str, Any] = {"model": target.upstream}
        if request.verbose:
            fields["response_format"] = "verbose_json"
        if request.language:
            fields["language"] = request.language
        if request.prompt:
            fields["prompt"] = request.prompt
        if request.temperature is not None:
            fields["temperature"] = str(request.temperature)
        if request.timestampGranularities:
            fields["timestamp_granularities[]"] = [g.value for g in request.timestampGranularities]
        what = "translation" if request.translate else "transcription"
        upload = (
            request.audio.filename,
            audio,
            request.audio.mediaType or "application/octet-stream",
        )
        client = self._client()
        try:
            response = await client.post(
                f"/v1/audio/{what}s",
                headers={
                    **self._headers(),
                    **({"X-Request-ID": str(request.requestId)} if request.requestId else {}),
                },
                data=fields,
                files={"file": upload},
            )
        except httpx.ConnectTimeout as e:
            raise CliError(f"openai_compat_http could not connect: {e!r}") from e
        except httpx.TimeoutException as e:
            raise _timed_out(e, self._timeout_seconds, f"the {what}") from e
        except httpx.HTTPError as e:
            raise CliError(f"openai_compat_http {what} failed: {e!r}") from e
        if response.status_code >= 400:
            raise CliError(
                f"openai_compat_http returned {response.status_code} for a {what}: "
                f"{_redact(response.text[:500])}",
                upstream_status=response.status_code,
                retry_after_seconds=retry_after(response.headers.get("Retry-After")),
            )
        try:
            body = response.json()
            return response_from(
                body,
                model_id=body.get("model") if isinstance(body.get("model"), str) else target.id,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
        except (ValueError, AttributeError) as e:
            raise CliError(f"openai_compat_http {what} answer was not usable: {e}") from e

    async def moderate(self, request: ModerateRequest) -> ModerateResponse:
        """`POST /v1/moderations` upstream (P6), in OpenAI's own shape: only
        OpenAI's API serves it (OpenRouter answers 404, measured).

        `texts` go as a list, one result each; `parts` as one multimodal
        input, one result. The results come back as the backend sent them.
        """
        started = time.perf_counter()
        target = self.resolve_model(request.model)
        wire: list[Any]
        if request.parts:
            wire = [
                {"type": "image_url", "image_url": {"url": part.image}}
                if part.type is ModerationPartType.image
                else {"type": "text", "text": part.text or ""}
                for part in request.parts
            ]
        else:
            # Each text is generated as a `RootModel[str]` (its maxLength).
            wire = [getattr(text, "root", text) for text in request.texts or []]
        client = self._client()
        try:
            response = await client.post(
                "/v1/moderations",
                headers={
                    **self._headers(),
                    **({"X-Request-ID": str(request.requestId)} if request.requestId else {}),
                },
                json={"model": target.upstream, "input": wire},
            )
        except httpx.ConnectTimeout as e:
            raise CliError(f"openai_compat_http could not connect: {e!r}") from e
        except httpx.TimeoutException as e:
            raise _timed_out(e, self._timeout_seconds, "the moderation") from e
        except httpx.HTTPError as e:
            raise CliError(f"openai_compat_http moderation failed: {e!r}") from e
        if response.status_code >= 400:
            raise CliError(
                f"openai_compat_http returned {response.status_code} for a moderation: "
                f"{_redact(response.text[:500])}",
                upstream_status=response.status_code,
                retry_after_seconds=retry_after(response.headers.get("Retry-After")),
            )
        try:
            body = response.json()
            results = body["results"]
            if not isinstance(results, list):
                raise TypeError("results is not a list")
        except (ValueError, KeyError, TypeError) as e:
            raise CliError(f"openai_compat_http moderation answer was not usable: {e}") from e
        return ModerateResponse(
            id=body.get("id") if isinstance(body.get("id"), str) else None,
            results=[r for r in results if isinstance(r, dict)],
            modelId=self._public_model_id(body.get("model"), target),
            latencyMs=int((time.perf_counter() - started) * 1000),
        )

    def _image_target(self, request: ImageRequest, mask: bytes | None) -> _Target:
        """The model an image request is for, refused before any upload
        leaves: one without the `image` surface, and a `mask` for a backend
        that would ignore it (OpenRouter does, measured) -- a silently
        ignored mask is an edit of the whole picture."""
        target = self.resolve_model(request.model)
        if target.entry is not None and "image" not in (target.entry.surfaces or []):
            raise ImageOutRefusal(
                f"{target.id!r} does not make images; choose a model with the image surface"
            )
        caps = (
            target.entry.capabilities.image if target.entry and target.entry.capabilities else None
        )
        if mask is not None and caps is not None and not caps.mask:
            raise ImageOutRefusal(
                f"mask: {target.id!r} does not honour a mask (only OpenAI's own API does), so "
                "the edit would change the whole image. Send it without mask, or to an OpenAI "
                "account's image model."
            )
        return target

    def _image_request(
        self,
        target: _Target,
        request: ImageRequest,
        uploads: list[bytes],
        mask: bytes | None,
        *,
        stream: bool,
    ) -> tuple[str, dict[str, Any]]:
        """The upstream path and `httpx` arguments for one image request.

        **OpenRouter has no edit route** (404, measured): an edit is a
        generation with `input_references`, which must be objects -- the
        plain strings its guide shows are a 400 (measured). **OpenAI** takes
        an edit on `/v1/images/edits` in the multipart form its SDK sends,
        and answers `dall-e-*` with a URL unless asked for `b64_json`; its
        GPT image models always answer base64 and take no
        `response_format`.
        """
        settings: list[tuple[str, Any]] = [
            ("n", request.n),
            ("size", request.size),
            ("quality", request.quality),
            ("background", request.background),
            ("output_format", request.outputFormat),
            ("output_compression", request.outputCompression),
            ("moderation", request.moderation),
            ("style", request.style),
            ("user", request.user),
            ("input_fidelity", request.inputFidelity),
        ]
        if stream:
            settings.append(("stream", True))
            settings.append(("partial_images", request.partialImages))
        chosen = {key: value for key, value in settings if value is not None}
        dall_e = target.upstream.lower().startswith("dall-e-")
        if self._catalogue_source == "openrouter" or not request.references:
            payload: dict[str, Any] = {"model": target.upstream, "prompt": request.prompt, **chosen}
            if request.references:
                payload["input_references"] = [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{ref.mediaType};base64,{ref.data}"},
                    }
                    for ref in request.references
                ]
            if dall_e:
                payload["response_format"] = "b64_json"
            return "/v1/images/generations", {"json": payload}
        fields: dict[str, Any] = {"model": target.upstream, "prompt": request.prompt}
        for key, value in chosen.items():
            fields[key] = "true" if value is True else str(value)
        if dall_e:
            fields["response_format"] = "b64_json"
        name = "image" if len(uploads) == 1 else "image[]"
        files: list[tuple[str, tuple[str, bytes, str]]] = []
        for index, (ref, raw) in enumerate(zip(request.references, uploads, strict=True)):
            ext = format_name(ref.mediaType)
            files.append((name, (f"image{index}.{ext}", raw, ref.mediaType)))
        if mask is not None and request.mask is not None:
            files.append(
                (
                    "mask",
                    (f"mask.{format_name(request.mask.mediaType)}", mask, request.mask.mediaType),
                )
            )
        return "/v1/images/edits", {"data": fields, "files": files}

    async def image(
        self, request: ImageRequest, uploads: list[bytes], mask: bytes | None
    ) -> ImageResponse:
        """Images made or edited, answered as base64 with the media type
        their bytes carry (P4). `uploads` are `request.references` decoded,
        in order; the route has checked each is an image."""
        started = time.perf_counter()
        target = self._image_target(request, mask)
        path, sending = self._image_request(target, request, uploads, mask, stream=False)
        client = self._client()
        try:
            response = await client.post(
                path,
                headers={
                    **self._headers(),
                    **({"X-Request-ID": str(request.requestId)} if request.requestId else {}),
                },
                **sending,
            )
        except httpx.ConnectTimeout as e:
            raise CliError(f"openai_compat_http could not connect: {e!r}") from e
        except httpx.TimeoutException as e:
            raise _timed_out(e, self._timeout_seconds, "the image request") from e
        except httpx.HTTPError as e:
            raise CliError(f"openai_compat_http image request failed: {e!r}") from e
        if response.status_code >= 400:
            raise CliError(
                f"openai_compat_http returned {response.status_code} for an image request: "
                f"{_redact(response.text[:500])}",
                upstream_status=response.status_code,
                retry_after_seconds=retry_after(response.headers.get("Retry-After")),
            )
        try:
            return images_response_from(
                response.json(),
                model_id=target.id,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
        except (ValueError, AttributeError) as e:
            raise CliError(f"openai_compat_http image answer was not usable: {e}") from e

    async def image_stream(
        self, request: ImageRequest, uploads: list[bytes], mask: bytes | None
    ) -> AsyncGenerator[ImagePartial | ImageResponse, None]:
        """Partial renders as they arrive, then the final `ImageResponse`.

        Reads both backends' streams: OpenAI's `event:`-named frames and
        OpenRouter's bare `data:` frames with `: ` keepalives between them
        (measured). **A backend that answers plain JSON although asked to
        stream** -- OpenRouter does, for a model that cannot -- is refused:
        the model's listing said it streams, and a stream with its partials
        missing is not what was asked for (P4-3).
        """
        started = time.perf_counter()
        target = self._image_target(request, mask)
        caps = (
            target.entry.capabilities.image if target.entry and target.entry.capabilities else None
        )
        if caps is not None and not caps.streaming:
            raise ImageOutRefusal(f"stream: {target.id!r} does not stream partial images")
        path, sending = self._image_request(target, request, uploads, mask, stream=True)
        client = self._client()
        completed: list[dict[str, Any]] = []
        partials = 0
        try:
            async with client.stream(
                "POST",
                path,
                headers={
                    **self._headers(),
                    "Accept": "text/event-stream",
                    **({"X-Request-ID": str(request.requestId)} if request.requestId else {}),
                },
                **sending,
            ) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    raise CliError(
                        f"openai_compat_http returned {response.status_code} for an image "
                        f"stream: {_redact(body.decode('utf-8', 'replace')[:500])}",
                        upstream_status=response.status_code,
                        retry_after_seconds=retry_after(response.headers.get("Retry-After")),
                    )
                if "text/event-stream" not in response.headers.get("content-type", ""):
                    await response.aread()
                    raise ImageOutRefusal(
                        f"stream: {target.id!r} answered one JSON document although asked to "
                        "stream, so it gives no partial images. Ask without stream."
                    )
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                    except ValueError as e:
                        raise CliError(
                            f"openai_compat_http image stream sent a bad frame: {e}"
                        ) from e
                    if isinstance(event, dict) and (
                        event.get("type") == "error" or "error" in event
                    ):
                        raise CliError(
                            "openai_compat_http image stream failed: "
                            f"{_redact(json.dumps(event.get('error', event))[:500])}"
                        )
                    kind = event_kind(event)
                    try:
                        if kind == "partial":
                            yield partial_from(event, partials)
                            partials += 1
                        elif kind == "completed":
                            completed.append(event)
                    except ValueError as e:
                        raise CliError(
                            f"openai_compat_http image stream frame was not usable: {e}"
                        ) from e
        except httpx.ConnectTimeout as e:
            raise CliError(f"openai_compat_http could not connect: {e!r}") from e
        except httpx.TimeoutException as e:
            raise _timed_out(e, self._timeout_seconds, "the image stream") from e
        except httpx.HTTPError as e:
            raise CliError(f"openai_compat_http image stream failed: {e!r}") from e
        try:
            yield completed_from(
                completed,
                model_id=target.id,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
        except ValueError as e:
            raise CliError(f"openai_compat_http image stream ended badly: {e}") from e

    def _video_source(self) -> None:
        """Only OpenRouter makes videos here: OpenAI's video API shut down on
        2026-09-24 (measured), and no other provider speaks a video shape."""
        if self._catalogue_source != "openrouter":
            raise VideoRefusal(
                "This backend makes no videos: only an OpenRouter account does (OpenAI's own "
                "video API shut down on 2026-09-24)."
            )

    async def video(self, request: VideoRequest, first_frame: bytes | None) -> VideoJob:
        """Submit one video job to OpenRouter, in its own shape (P5).

        `duration` is an integer and `size` one the model lists (another is
        its 400, measured); the first frame is `frame_images` with
        `frame_type: first_frame`. The answer is the job, `pending`.
        """
        self._video_source()
        target = self.resolve_model(request.model)
        if target.entry is not None and "video" not in (target.entry.surfaces or []):
            raise VideoRefusal(
                f"{target.id!r} makes no videos; choose a model with the video surface"
            )
        caps = (
            target.entry.capabilities.video if target.entry and target.entry.capabilities else None
        )
        if (
            request.firstFrame is not None
            and first_frame is not None
            and (caps is None or not caps.firstFrame)
        ):
            raise VideoRefusal(
                f"input_reference: {target.id!r} takes no first frame, so the image would be "
                "ignored"
            )
        payload: dict[str, Any] = {"model": target.upstream, "prompt": request.prompt}
        if request.seconds is not None:
            payload["duration"] = request.seconds
        if request.size is not None:
            payload["size"] = request.size
        frame = request.firstFrame
        if frame is not None:
            payload["frame_images"] = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{frame.mediaType};base64,{frame.data}"},
                    "frame_type": "first_frame",
                }
            ]
        client = self._client()
        try:
            response = await client.post(
                "/v1/videos",
                headers={
                    **self._headers(),
                    **({"X-Request-ID": str(request.requestId)} if request.requestId else {}),
                },
                json=payload,
            )
        except httpx.ConnectTimeout as e:
            raise CliError(f"openai_compat_http could not connect: {e!r}") from e
        except httpx.TimeoutException as e:
            raise _timed_out(e, self._timeout_seconds, "the video submit") from e
        except httpx.HTTPError as e:
            raise CliError(f"openai_compat_http video submit failed: {e!r}") from e
        if response.status_code >= 400:
            raise CliError(
                f"openai_compat_http returned {response.status_code} for a video submit: "
                f"{_redact(response.text[:500])}",
                upstream_status=response.status_code,
                retry_after_seconds=retry_after(response.headers.get("Retry-After")),
            )
        try:
            return video_job_from(response.json(), model_id=target.id)
        except (ValueError, AttributeError) as e:
            raise CliError(f"openai_compat_http video submit answer was not usable: {e}") from e

    async def video_job(self, job_id: str) -> VideoJob:
        """Poll one job. An unknown job is the backend's 404, relayed as one."""
        self._video_source()
        client = self._client()
        try:
            response = await client.get(
                f"/v1/videos/{quote(job_id, safe='')}",
                headers=self._headers(),
                timeout=_LIST_TIMEOUT,
            )
        except httpx.HTTPError as e:
            raise CliError(f"openai_compat_http video poll failed: {e!r}") from e
        if response.status_code >= 400:
            raise CliError(
                f"openai_compat_http returned {response.status_code} for a video poll: "
                f"{_redact(response.text[:500])}",
                upstream_status=response.status_code,
            )
        try:
            return video_job_from(response.json(), model_id=None)
        except (ValueError, AttributeError) as e:
            raise CliError(f"openai_compat_http video poll answer was not usable: {e}") from e

    async def video_content(self, job_id: str) -> AsyncGenerator[bytes, None]:
        """A finished job's MP4, streamed as OpenRouter sends it (chunked, no
        length; `Range` is ignored, measured). A refusal raises before any
        byte is yielded."""
        self._video_source()
        client = self._client()
        try:
            async with client.stream(
                "GET",
                f"/v1/videos/{quote(job_id, safe='')}/content",
                params={"index": 0},
                headers={**self._headers(), "Accept": "video/mp4"},
            ) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    raise CliError(
                        f"openai_compat_http returned {response.status_code} for a video's "
                        "content: "
                        f"{_redact(body.decode('utf-8', 'replace')[:500])}",
                        upstream_status=response.status_code,
                    )
                async for chunk in response.aiter_raw():
                    if chunk:
                        yield chunk
        except httpx.ConnectTimeout as e:
            raise CliError(f"openai_compat_http could not connect: {e!r}") from e
        except httpx.TimeoutException as e:
            raise _timed_out(e, self._timeout_seconds, "the video download") from e
        except httpx.HTTPError as e:
            raise CliError(f"openai_compat_http video download failed: {e!r}") from e

    async def embed(self, inputs: list[str], *, model: str | None = None) -> EmbedResponse:
        """`POST /v1/embeddings` upstream, in the shape OpenAI defined.

        **Always asks for floats.** The base64 encoding the OpenAI SDKs
        request by default is applied by the gateway instead, so a
        caller sees identical behaviour whether or not this particular
        backend implements `encoding_format` -- and several do not.
        """
        started = time.perf_counter()
        target = self.resolve_model(model)
        payload: dict[str, Any] = {"model": target.upstream, "input": inputs}

        client = self._client()
        try:
            response = await client.post(
                "/v1/embeddings",
                headers={
                    **self._headers(),
                    **({"X-Request-ID": str(request_id.get())} if request_id.get() else {}),
                },
                json=payload,
            )
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
                retry_after_seconds=retry_after(response.headers.get("Retry-After")),
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
            modelId=self._public_model_id(body.get("model"), target),
            backend=self.backend_kind,
            usage=_usage_from_envelope(body.get("usage") or {}),
            latencyMs=int((time.perf_counter() - started) * 1000),
        )

    async def count_prompt_tokens(self, request: GenerateRequest) -> int:
        """The prompt this request would send, counted by the backend itself.

        **The same payload `generate` sends** -- `_payload_for`, so the
        thinking directive and every shaped setting are in it -- rendered
        by llama.cpp's `/apply-template` and counted by its `/tokenize`
        with the special tokens the completion path adds. Measured exact
        against a one-token generation's `prompt_tokens` on Gemma 4 E4B
        and Qwen3 (see the test module). Nothing is generated.

        Raises `TokenCountUnsupported` where the count could not be exact:
        an image (the template renders a marker; the projector decides
        the cost), OpenAI's own API, and any backend without the template
        endpoint. A backend that is down or rejects the request raises
        `CliError`, as `generate` would.
        """
        kinds = attachment_kinds(request.messages)
        if "image" in kinds:
            raise TokenCountUnsupported(
                "an image's token cost is decided by the projector when it encodes the "
                "picture, and the chat template renders only a marker for it"
            )
        if kinds:
            raise TokenCountUnsupported(
                "an attachment's token cost is decided by the encoder that reads it, and "
                "the chat template renders only a marker for it"
            )
        if _is_openai_endpoint(self._base_url):
            raise TokenCountUnsupported("OpenAI's own API cannot count a chat prompt here")
        payload = self._payload_for(request, self.resolve_model(request.model))
        client = self._client()
        try:
            rendered = await client.post(
                "/apply-template",
                headers=self._headers(),
                json=payload,
                timeout=_COUNT_TIMEOUT_SECONDS,
            )
            if rendered.status_code in (404, 405, 501):
                raise TokenCountUnsupported(
                    f"this backend has no /apply-template (HTTP {rendered.status_code}); "
                    "only llama.cpp's llama-server can count a chat prompt without generating"
                )
            if rendered.status_code >= 400:
                raise CliError(
                    f"openai_compat_http /apply-template returned {rendered.status_code}: "
                    f"{_redact(rendered.text[:500])}",
                    upstream_status=rendered.status_code,
                )
            prompt = rendered.json().get("prompt")
            if not isinstance(prompt, str):
                raise CliError("openai_compat_http /apply-template returned no prompt")
            counted = await client.post(
                "/tokenize",
                headers=self._headers(),
                json={"content": prompt, "add_special": True},
                timeout=_COUNT_TIMEOUT_SECONDS,
            )
            if counted.status_code >= 400:
                raise CliError(
                    f"openai_compat_http /tokenize returned {counted.status_code}: "
                    f"{_redact(counted.text[:500])}",
                    upstream_status=counted.status_code,
                )
            tokens = counted.json().get("tokens")
        except httpx.TimeoutException as e:
            raise _timed_out(e, _COUNT_TIMEOUT_SECONDS, "the token count") from e
        except httpx.HTTPError as e:
            raise CliError(f"openai_compat_http token count failed: {e!r}") from e
        except ValueError as e:
            raise CliError("openai_compat_http token count returned non-JSON") from e
        if not isinstance(tokens, list):
            raise CliError("openai_compat_http /tokenize returned no tokens")
        return len(tokens)

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
        return await self._probe_modality("vision")

    async def probe_audio_input(self) -> bool:
        """`llama-server`'s `/props` `modalities.audio`, for a model whose
        projector hears (Voxtral, Qwen2.5-Omni, Gemma 3n). Same checks as
        the image probe, so an Ollama or a hosted API reports false."""
        return await self._probe_modality("audio")

    async def _probe_modality(self, modality: str) -> bool:
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
                if not isinstance(modalities, dict) or modalities.get(modality) is not True:
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
                    and models[0].get("id") == self._upstream_model_id
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
        if response.status_code in _NO_SUCH_PATH:
            # A definite no: whatever this is, it has no `/props`.
            self._llama_cpp = False
            return None
        if response.status_code >= 400:
            # **Not a no.** llama-server answers 503 while it is still
            # loading its model, and the companion driver starts before
            # that is done -- so this was the FIRST thing a fresh driver
            # heard, and caching it as "not llama.cpp" switched progress
            # off for the life of the engine. Found by the live run.
            return None
        props = response.json()
        if not isinstance(props, dict):
            self._llama_cpp = False
            return None
        settings = props.get("default_generation_settings")
        value = settings.get("n_ctx") if isinstance(settings, dict) else None
        if value is None:
            value = props.get("n_ctx")
        found = value if isinstance(value, int) and value > 0 else None
        self._llama_cpp = found is not None
        return found

    async def _answers_as_llama_cpp(self) -> bool:
        """Whether `return_progress` may be sent, asking `/props` once if unknown.

        Usually answered already: the context probe behind `/v1/info`,
        which the gateway polls before it routes anything here, reads the
        same endpoint. Asked here only for a stream that reached a fresh
        driver first. A transport failure leaves the answer unknown, and
        unknown is no -- the flag is only ever sent on a yes.
        """
        if self._llama_cpp is None:
            try:
                await self._ctx_llama_cpp(self._client())
            except (httpx.HTTPError, ValueError):
                log.debug("could not tell whether %s is llama.cpp", self._base_url)
        return self._llama_cpp is True

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
        card = _only_or_named(cards, self._upstream_model_id, keys=("id",))
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
        entry = _only_or_named(entries, self._upstream_model_id, keys=("name", "model"))
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


def _to_openai_messages(
    messages: list[Any], *, send_reasoning: bool = True
) -> list[dict[str, Any]]:
    """Map our Message[] to OpenAI chat-completions messages.

    An assistant turn's `reasoning` goes back up as `reasoning_content`
    -- the name llama.cpp reads (measured: it ignored `reasoning` on the
    same request) and vLLM accepts as an alias -- unless the backend is
    OpenAI's own API (`send_reasoning=False`), which refuses the
    property.

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
            reasoning = getattr(m, "reasoning", None)
            if send_reasoning and reasoning:
                msg["reasoning_content"] = reasoning
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
        # OpenAI's detail objects, which llama.cpp (cached) and vLLM
        # (both) fill in. Absent stays absent: a zero here would claim
        # the backend counted and found none.
        cachedPromptTokens=_detail(usage, "prompt_tokens_details", "cached_tokens"),
        reasoningTokens=_detail(usage, "completion_tokens_details", "reasoning_tokens"),
    )


def _detail(usage: dict[str, Any], group: str, key: str) -> int | None:
    details = usage.get(group)
    value = details.get(key) if isinstance(details, dict) else None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _redact(text: str) -> str:
    """Best-effort redaction of API keys that might appear in error bodies."""
    return text.replace("Bearer ", "Bearer <redacted>")
