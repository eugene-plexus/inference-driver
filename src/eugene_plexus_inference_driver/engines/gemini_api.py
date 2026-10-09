"""Google's Gemini API, natively: `generateContent` and its kin (gemini-provider.md).

One driver per Google account (P1): an empty `modelId` serves every model the
key may use, each from Google's own listing plus our table of what a family
takes. The native API, not Google's OpenAI-compatible endpoint (call G1): the
compatible one silently ignores any parameter it does not list, which is the
one thing a driver here must not do, and it has no place for thought
signatures.

* **Chat** both ways, streamed or not: messages to `contents`, tools to
  `functionDeclarations`, `response_format` to a response schema,
  `reasoning_effort` to `thinkingConfig`, thoughts back as reasoning.
* **Thought signatures** are remembered here by tool-call id (G2, G7).
* **A setting Gemini cannot honour is refused** (A2), never sent and ignored.
* **Embeddings, images, Veo video, speech (WAV or PCM, G5) and transcription**
  through the routes the other media engines use.

The key rides in `x-goog-api-key` and nowhere else: not in a URL, not in a
log, and scrubbed from any text echoed back (`_scrub`).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
import re
import time
from collections.abc import AsyncGenerator, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlsplit

import httpx

from .._generated.models import (
    BackendKind,
    ConfigField,
    ConfigFieldShowWhen,
    ConfigValueType,
    DriverModel,
    EmbedResponse,
    FinishReason,
    Function1,
    GeneratedImage,
    GenerateRequest,
    GenerateResponse,
    ImageRequest,
    ImageResponse,
    ImageUsage,
    SpeakRequest,
    SpeechFormat,
    Stage,
    StreamProgress,
    ToolCall,
    ToolCallDelta,
    TranscribeRequest,
    TranscribeResponse,
    TranscriptionUsage,
    Usage,
    VideoJob,
    VideoJobStatus,
    VideoRequest,
)
from .._http import client_for
from ..images_out import ImageRefusal, sniff
from ..speech import SpeechRefusal, refuse_format
from ..transcription import TranscriptionRefusal
from ..videos_out import VideoRefusal
from . import _gemini_wire as wire
from ._catalogue import _LIST_TIMEOUT, Catalogue, CatalogueError
from ._subprocess import BackendTimeout, CliError
from .base import (
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_STREAM_STALL_SECONDS,
    Chunk,
    ModelNotServed,
    ModelRequired,
    refuse_unsupported_settings,
)

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
_KEEPALIVE_PROGRESS_SECONDS = 2.0
_EMBED_BATCH = 100
_LIST_PAGES = 20

#: Every request setting the contract names; what a model's listing lacks is refused.
_ALL_SETTINGS = frozenset(
    {
        *wire.CHAT_SETTINGS,
        "minP",
        "logprobs",
        "logitBias",
        "reasoningEffort",
        "verbosity",
        "prediction",
        "webSearchOptions",
    }
)

_AUDIO_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mp3",
    "mpeg": "audio/mp3",
    "aac": "audio/aac",
    "m4a": "audio/aac",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
    "oga": "audio/ogg",
    "aiff": "audio/aiff",
    "aif": "audio/aiff",
}
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_./-]+$")


@dataclass(frozen=True)
class _Resolved:
    """The model one request is for: what callers see, what Google is asked for."""

    id: str
    upstream: str
    entry: DriverModel


class GeminiApiEngine:
    """Google's Gemini API, as a provider account."""

    backend_kind = BackendKind.gemini_api
    supports_tool_calling = True
    supports_streaming = True
    supports_embeddings = True
    serves_accounts = True
    follows_runtimes = False
    #: What `/v1/speak` makes when no format is asked for (G5).
    default_speech_format = SpeechFormat.wav
    #: The settings this engine can carry at all (per model, the listing narrows it).
    supported_settings: ClassVar[list[str]] = list(wire.CHAT_SETTINGS)

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        stall_seconds: float = DEFAULT_STREAM_STALL_SECONDS,
        model_id: str | None = None,
        get: Any = None,
        catalogue_path: Path | None = None,
        provider: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._stall_seconds = stall_seconds
        self._only = wire.bare_id(model_id) if model_id else None
        self._http: httpx.AsyncClient | None = None
        self._signatures = wire.SignatureCache()
        self._warned: set[tuple[str, str]] = set()
        self.catalogue = Catalogue(
            source="gemini",
            origin=f"{provider or 'gemini'}|{self._base_url}",
            fetch=self._fetch_catalogue,
            get=get or (lambda _key: None),
            path=catalogue_path,
        )

    # -- construction -------------------------------------------------------

    @classmethod
    def field_specs(cls, *, applicable_providers: list[str]) -> list[ConfigField]:
        """The account fields this engine reads, shown for Gemini. Keys shared
        with the other HTTP engines are merged by the schema builder."""
        show_when = ConfigFieldShowWhen(key="provider", equals=applicable_providers)
        return [
            ConfigField(
                key="apiKey",
                label="API Key",
                description=(
                    "The provider's secret key, sent as it requires: "
                    "`Authorization: Bearer` for OpenAI-compatible providers, "
                    "`xi-api-key` for ElevenLabs, `x-goog-api-key` for Gemini. If left "
                    "blank the driver falls back to `OPENAI_API_KEY`, or "
                    "`ELEVENLABS_API_KEY` for ElevenLabs, or `GEMINI_API_KEY` for "
                    "Gemini, in the environment that started it."
                ),
                category="adapter",
                valueType=ConfigValueType.secret,
                sensitive=True,
                requiresRestart=True,
                showWhen=show_when,
            ),
            ConfigField(
                key="catalogueInclude",
                label="Models to use",
                description=(
                    "Keep only the provider's models matching these patterns: "
                    "`*` matches anything, `/` included. The default `*` keeps "
                    "everything. Takes effect without a restart."
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
                description="Patterns for models to leave out. Takes effect without a restart.",
                category="adapter",
                valueType=ConfigValueType.string_list,
                default=[],
                requiresRestart=False,
                showWhen=show_when,
            ),
        ]

    @classmethod
    def from_config(
        cls,
        get: Any,
        *,
        default_base_url: str = DEFAULT_BASE_URL,
        auth_required: bool = True,
        catalogue_path: Path | None = None,
        provider: str | None = None,
        backend_kind: BackendKind = BackendKind.gemini_api,
    ) -> GeminiApiEngine:
        del backend_kind  # one protocol, one kind; kept for registry symmetry
        api_key = str(get("apiKey") or "") or os.environ.get("GEMINI_API_KEY") or None
        if auth_required and not api_key:
            raise CliError(
                "Gemini needs an API key: set apiKey on this driver, or "
                "GEMINI_API_KEY in the environment that starts it."
            )
        raw_stall = get("streamStallSeconds")
        return cls(
            api_key=api_key,
            base_url=str(get("baseUrl") or default_base_url),
            timeout_seconds=float(get("requestTimeoutSeconds") or DEFAULT_REQUEST_TIMEOUT_SECONDS),
            stall_seconds=(
                DEFAULT_STREAM_STALL_SECONDS
                if raw_stall is None or raw_stall == ""
                else float(raw_stall)
            ),
            model_id=str(get("modelId") or "").strip() or None,
            get=get,
            catalogue_path=catalogue_path,
            provider=provider,
        )

    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = client_for(
                self._base_url,
                base_url=self._base_url,
                timeout=httpx.Timeout(self._timeout_seconds, connect=10.0),
            )
        return self._http

    async def aclose(self) -> None:
        client, self._http = self._http, None
        if client is not None:
            await client.aclose()

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self._api_key:
            headers["x-goog-api-key"] = self._api_key
        return headers

    def _scrub(self, text: str) -> str:
        """Nothing this driver says may carry the key, whatever echoed it."""
        if self._api_key:
            text = text.replace(self._api_key, "<redacted>")
        return re.sub(r"(?i)(key=)[^&\s\"']+", r"\1<redacted>", text)

    @contextlib.contextmanager
    def _guard(self, what: str) -> Iterator[None]:
        try:
            yield
        except httpx.ConnectTimeout as e:
            raise CliError(f"Gemini could not be reached for {what}: {type(e).__name__}") from e
        except httpx.TimeoutException as e:
            raise BackendTimeout(
                f"Gemini did not answer {what} within {self._timeout_seconds:g}s "
                f"({type(e).__name__}). The backend was still working, not broken. Raise "
                "requestTimeoutSeconds on this driver (and on the gateway).",
                limit_seconds=self._timeout_seconds,
            ) from e
        except httpx.HTTPError as e:
            raise CliError(self._scrub(f"Gemini failed {what}: {type(e).__name__}: {e}")) from e

    def _failure(self, response: httpx.Response, what: str) -> CliError:
        return wire.failure_from(response, what, scrub=self._scrub)

    def _warn_dropped(self, model: str, field: str) -> None:
        if (model, field) in self._warned:
            return
        self._warned.add((model, field))
        log.warning(
            "Gemini model %r cannot carry `%s`; it is omitted on every request (it was not "
            "an explicit setting). Said once per parameter for the life of this engine.",
            model,
            field,
        )

    # -- the account -----------------------------------------------------------

    async def _fetch_catalogue(self) -> list[DriverModel]:
        client = self._client()
        models: list[DriverModel] = []
        token: str | None = None
        for _ in range(_LIST_PAGES):
            params: dict[str, Any] = {"pageSize": 1000}
            if token:
                params["pageToken"] = token
            response = await client.get(
                "/models", params=params, headers=self._headers(), timeout=_LIST_TIMEOUT
            )
            if response.status_code >= 400:
                raise CatalogueError(
                    f"Gemini refused the model list ({response.status_code}): "
                    f"{self._failure(response, 'the model list')}"
                )
            try:
                body = response.json()
            except ValueError as e:
                raise CatalogueError("Gemini's model list was not JSON") from e
            listed = body.get("models") if isinstance(body, dict) else None
            if not isinstance(listed, list):
                raise CatalogueError("Gemini's model list had no `models` array")
            for entry in listed:
                if isinstance(entry, dict) and (model := wire.model_from_listing(entry)):
                    models.append(model)
            token = (
                body.get("nextPageToken") if isinstance(body.get("nextPageToken"), str) else None
            )
            if not token:
                break
        if self._only:
            models = [m for m in models if m.id == self._only]
        return models

    async def list_models(self) -> list[str]:
        return [m.id for m in self.catalogue.exposed()]

    def resolve_model(self, requested: str | None) -> _Resolved:
        if not requested:
            raise ModelRequired(
                "This driver is a Gemini account serving "
                f"{len(self.catalogue.exposed())} models; a request must name one in `model`."
            )
        entry = self.catalogue.find(wire.bare_id(requested))
        if entry is None:
            raise ModelNotServed(requested, served=len(self.catalogue.exposed()))
        return _Resolved(id=entry.id, upstream=entry.id, entry=entry)

    def _surface(self, target: _Resolved, surface: str, what: str) -> None:
        if surface not in target.entry.surfaces:
            raise CliError(
                f"{target.id!r} does not serve {what} (it serves "
                f"{', '.join(target.entry.surfaces) or 'nothing Eugene routes'}).",
                upstream_status=400,
            )

    async def context_window(self) -> int | None:
        return None

    # -- chat ----------------------------------------------------------------------

    def _unsupported(self, target: _Resolved) -> set[str]:
        caps = target.entry.capabilities
        return set(_ALL_SETTINGS - set(caps.supportedSettings or [] if caps else []))

    def _chat_body(
        self, request: GenerateRequest, target: _Resolved
    ) -> tuple[dict[str, Any], bool]:
        """`(the generateContent body, whether it replays a call with no signature)`."""
        self._surface(target, "chat", "chat")
        if request.audioOutput is not None or request.completion is not None:
            raise CliError(
                "audioOutput and raw completion are not served by Gemini's chat; speech has "
                "its own surface.",
                upstream_status=400,
            )
        unsupported = self._unsupported(target)
        refuse_unsupported_settings(request, unsupported=unsupported)
        explicit = set(request.callerSettings or [])
        if request.parallelToolCalls is False:
            if "parallelToolCalls" in explicit:
                raise CliError(
                    "parallel_tool_calls: Gemini may make several calls in one turn and has "
                    "no setting to prevent it, so false cannot be honoured; remove it.",
                    upstream_status=400,
                )
            self._warn_dropped(target.id, "parallel_tool_calls")
        for field, value in (
            ("minP", request.minP),
            ("logprobs", request.logprobs),
            ("logitBias", request.logitBias),
            ("verbosity", request.verbosity),
            ("prediction", request.prediction),
            ("webSearchOptions", request.webSearchOptions),
            ("reasoningEffort", request.reasoningEffort),
        ):
            if value is not None and field in unsupported:
                self._warn_dropped(target.id, field)
        gemma = target.id.lower().startswith("gemma")
        contents, system, unsigned = wire.contents_from(
            list(request.messages), self._signatures, fold_system=gemma
        )
        if not contents:
            raise CliError("messages: nothing to send to Gemini.", upstream_status=400)
        body: dict[str, Any] = {"contents": contents}
        if system is not None:
            body["systemInstruction"] = system
        if request.tools:
            body["tools"] = wire.declarations_from(request.tools)
        config = wire.tool_config_from(request.toolChoice)
        if config is not None and request.tools:
            body["toolConfig"] = config
        gen: dict[str, Any] = {}
        for key, value in (
            ("temperature", request.temperature),
            ("topP", request.topP),
            ("topK", request.topK),
            ("maxOutputTokens", request.maxTokens),
            ("seed", request.seed),
            ("presencePenalty", request.presencePenalty),
            ("frequencyPenalty", request.frequencyPenalty),
        ):
            if value is not None:
                gen[key] = value
        if request.stop:
            gen["stopSequences"] = list(request.stop)
        fmt = request.responseFormat
        if fmt is not None:
            kind = fmt.type.value
            if kind in ("json_object", "json_schema"):
                gen["responseMimeType"] = "application/json"
            if kind == "json_schema" and fmt.json_schema is not None:
                gen["responseJsonSchema"] = fmt.json_schema.schema_
        if "reasoningEffort" not in unsupported:
            gen["thinkingConfig"] = wire.thinking_config(target.upstream, request.reasoningEffort)
        if gen:
            body["generationConfig"] = gen
        return body, unsigned

    def _result(
        self,
        body: dict[str, Any],
        parsed: wire.Parsed,
        candidate: dict[str, Any],
        *,
        request: GenerateRequest,
        target: _Resolved,
        started: float,
    ) -> GenerateResponse:
        finish = wire.finish_of(candidate.get("finishReason"), called=bool(parsed.calls))
        content: str | None = parsed.text or None
        if finish is FinishReason.content_filter and not parsed.text and not parsed.calls:
            content = wire.refusal_note(candidate)
        return GenerateResponse(
            content=content,
            reasoning=parsed.reasoning or None,
            toolCalls=parsed.calls or None,
            finishReason=finish,
            usage=wire.usage_of(body.get("usageMetadata")),
            requestId=request.requestId,
            backend=self.backend_kind,
            modelId=target.id,
            latencyMs=int((time.perf_counter() - started) * 1000),
        )

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        started = time.perf_counter()
        target = self.resolve_model(request.model)
        payload, unsigned = self._chat_body(request, target)
        try:
            with self._guard("the completion"):
                response = await self._client().post(
                    f"/models/{target.upstream}:generateContent",
                    headers=self._headers(),
                    json=payload,
                )
            if response.status_code >= 400:
                raise self._failure(response, "the completion")
            try:
                body = response.json()
            except ValueError as e:
                raise CliError("Gemini returned non-JSON for the completion") from e
        except CliError as e:
            raise wire.explain_signature_refusal(e, replayed_unsigned=unsigned) from e
        candidates = body.get("candidates") if isinstance(body, dict) else None
        if not candidates:
            raise wire.blocked_prompt(body) or CliError("Gemini returned no candidates")
        candidate = candidates[0] if isinstance(candidates[0], dict) else {}
        parsed = wire.read_parts((candidate.get("content") or {}).get("parts"), self._signatures)
        if not (parsed.text or parsed.calls or parsed.reasoning) and candidate.get(
            "finishReason"
        ) in (None, "STOP"):
            raise CliError("Gemini returned an answer with no text, reasoning or tool call")
        return self._result(
            body, parsed, candidate, request=request, target=target, started=started
        )

    async def stream(self, request: GenerateRequest) -> AsyncGenerator[Chunk, None]:
        started = time.perf_counter()
        target = self.resolve_model(request.model)
        payload, unsigned = self._chat_body(request, target)
        report = bool(request.reportProgress)
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        calls: list[ToolCall] = []
        usage_raw: dict[str, Any] = {}
        finish_raw: Any = None
        last_candidate: dict[str, Any] = {}
        finished = False
        saw_first = False
        last_keepalive = 0.0
        try:
            with self._guard("the stream"):
                async with self._client().stream(
                    "POST",
                    f"/models/{target.upstream}:streamGenerateContent",
                    params={"alt": "sse"},
                    headers={**self._headers(), "Accept": "text/event-stream"},
                    json=payload,
                ) as response:
                    if response.status_code >= 400:
                        await response.aread()
                        raise self._failure(response, "the stream")
                    if report:
                        yield Chunk(progress=StreamProgress(stage=Stage.working))
                    lines = response.aiter_lines()
                    while True:
                        try:
                            if saw_first and self._stall_seconds > 0:
                                try:
                                    async with asyncio.timeout(self._stall_seconds):
                                        line = await anext(lines)
                                except TimeoutError as stall:
                                    if finished:
                                        break
                                    raise BackendTimeout(
                                        f"Gemini's stream went silent for "
                                        f"{self._stall_seconds:g}s mid-answer, after "
                                        f"{len(''.join(text_parts + reasoning_parts))} "
                                        "characters. If it legitimately pauses that long, "
                                        "raise streamStallSeconds on this driver (0 disables).",
                                        limit_seconds=self._stall_seconds,
                                    ) from stall
                            else:
                                line = await anext(lines)
                        except StopAsyncIteration:
                            break
                        if report and line.startswith(":"):
                            now = time.perf_counter()
                            if now - last_keepalive >= _KEEPALIVE_PROGRESS_SECONDS:
                                last_keepalive = now
                                yield Chunk(progress=StreamProgress(stage=Stage.working))
                            continue
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if not data or data == "[DONE]":
                            continue
                        saw_first = True
                        try:
                            event = json.loads(data)
                        except ValueError:
                            log.debug("gemini_api: unparseable SSE frame (contents omitted)")
                            continue
                        if not isinstance(event, dict):
                            continue
                        if isinstance(event.get("error"), dict):
                            error = event["error"]
                            code = error.get("code")
                            raise CliError(
                                self._scrub(
                                    f"Gemini's stream failed ({error.get('status')}): "
                                    f"{error.get('message')}"
                                ),
                                upstream_status=code if isinstance(code, int) else None,
                            )
                        if isinstance(event.get("usageMetadata"), dict):
                            usage_raw = event["usageMetadata"]
                        found = event.get("candidates")
                        if not found:
                            blocked = wire.blocked_prompt(event)
                            if blocked is not None:
                                raise blocked
                            continue
                        candidate = found[0] if isinstance(found[0], dict) else {}
                        last_candidate = candidate
                        parsed = wire.read_parts(
                            (candidate.get("content") or {}).get("parts"), self._signatures
                        )
                        if parsed.reasoning:
                            reasoning_parts.append(parsed.reasoning)
                            yield Chunk(reasoning=parsed.reasoning)
                        if parsed.text:
                            text_parts.append(parsed.text)
                            yield Chunk(text=parsed.text)
                        for call in parsed.calls:
                            calls.append(call)
                            yield Chunk(
                                toolCalls=[
                                    ToolCallDelta(
                                        index=len(calls) - 1,
                                        id=call.id,
                                        type="function",
                                        function=Function1(
                                            name=call.function.name,
                                            arguments=call.function.arguments,
                                        ),
                                    )
                                ]
                            )
                        if candidate.get("finishReason"):
                            finish_raw = candidate["finishReason"]
                            finished = True
        except CliError as e:
            raise wire.explain_signature_refusal(e, replayed_unsigned=unsigned) from e
        if not finished:
            raise CliError(
                "Gemini's stream ended without a finishReason after "
                f"{len(''.join(text_parts + reasoning_parts))} characters: the backend closed "
                "the connection mid-answer"
            )
        finish = wire.finish_of(finish_raw, called=bool(calls))
        content = "".join(text_parts) or None
        if finish is FinishReason.content_filter and not content and not calls:
            note = wire.refusal_note(last_candidate)
            content = note
            yield Chunk(text=note)
        yield Chunk(
            done=True,
            result=GenerateResponse(
                content=content,
                reasoning="".join(reasoning_parts) or None,
                toolCalls=calls or None,
                finishReason=finish,
                usage=wire.usage_of(usage_raw),
                requestId=request.requestId,
                backend=self.backend_kind,
                modelId=target.id,
                latencyMs=int((time.perf_counter() - started) * 1000),
            ),
        )

    # -- embeddings --------------------------------------------------------------

    async def embed(
        self, inputs: list[str], *, model: str | None = None, dimensions: int | None = None
    ) -> EmbedResponse:
        """`batchEmbedContents`, in batches of 100, vectors in input order.
        `dimensions` is Google's `outputDimensionality`; the driver's embed
        route carries none today."""
        started = time.perf_counter()
        target = self.resolve_model(model)
        self._surface(target, "embeddings", "embeddings")
        vectors: list[list[float]] = []
        tokens = 0
        for start in range(0, len(inputs), _EMBED_BATCH):
            batch = inputs[start : start + _EMBED_BATCH]
            requests: list[dict[str, Any]] = []
            for text in batch:
                item: dict[str, Any] = {
                    "model": f"models/{target.upstream}",
                    "content": {"parts": [{"text": text}]},
                }
                if dimensions is not None:
                    item["outputDimensionality"] = dimensions
                requests.append(item)
            with self._guard("the embeddings request"):
                response = await self._client().post(
                    f"/models/{target.upstream}:batchEmbedContents",
                    headers=self._headers(),
                    json={"requests": requests},
                )
            if response.status_code >= 400:
                raise self._failure(response, "embeddings")
            try:
                body = response.json()
                rows = body["embeddings"]
                got = [[float(x) for x in row["values"]] for row in rows]
            except (ValueError, KeyError, TypeError) as e:
                raise CliError(f"Gemini's embeddings answer was not usable: {e!r}") from e
            if len(got) != len(batch) or not all(got):
                raise CliError(
                    f"Gemini returned {len(got)} embeddings for {len(batch)} inputs; order and "
                    "count are the only way a caller can match vectors to text"
                )
            vectors.extend(got)
            meta = body.get("usageMetadata")
            if isinstance(meta, dict) and isinstance(meta.get("promptTokenCount"), int):
                tokens += meta["promptTokenCount"]
        return EmbedResponse(
            embeddings=vectors,
            modelId=target.id,
            backend=self.backend_kind,
            usage=Usage(promptTokens=tokens, totalTokens=tokens) if tokens else None,
            latencyMs=int((time.perf_counter() - started) * 1000),
        )

    # -- images ------------------------------------------------------------------

    async def image(
        self, request: ImageRequest, uploads: list[bytes], mask: bytes | None
    ) -> ImageResponse:
        """An image made or edited by `generateContent` with an image answer."""
        started = time.perf_counter()
        target = self.resolve_model(request.model)
        if "image" not in target.entry.surfaces:
            raise ImageRefusal(
                f"{target.id!r} does not make images; choose a model with the image surface"
            )
        if mask is not None:
            raise ImageRefusal(
                "mask: Gemini edits by instruction and takes no mask, so the edit would change "
                "the whole image. Send it without a mask."
            )
        for field, value in (
            ("n", request.n if request.n not in (None, 1) else None),
            ("size", request.size),
            ("quality", request.quality),
            ("background", request.background),
            ("output_format", request.outputFormat),
            ("output_compression", request.outputCompression),
            ("moderation", request.moderation),
            ("style", request.style),
            ("input_fidelity", request.inputFidelity),
            ("partial_images", request.partialImages),
        ):
            if value is not None:
                raise ImageRefusal(
                    f"{field}: Gemini's image models take none of this setting and would ignore "
                    "it (and make one image per request); remove it."
                )
        parts: list[dict[str, Any]] = [{"text": request.prompt}]
        for ref, raw in zip(request.references or [], uploads, strict=True):
            parts.append(
                {
                    "inlineData": {
                        "mimeType": sniff(raw) or ref.mediaType,
                        "data": base64.b64encode(raw).decode("ascii"),
                    }
                }
            )
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
        }
        with self._guard("the image request"):
            response = await self._client().post(
                f"/models/{target.upstream}:generateContent",
                headers=self._headers(),
                json=payload,
            )
        if response.status_code >= 400:
            raise self._failure(response, "an image request")
        try:
            body = response.json()
        except ValueError as e:
            raise CliError("Gemini returned non-JSON for an image request") from e
        candidates = body.get("candidates") if isinstance(body, dict) else None
        if not candidates:
            raise wire.blocked_prompt(body) or CliError("Gemini returned no candidates")
        candidate = candidates[0] if isinstance(candidates[0], dict) else {}
        parsed = wire.read_parts(
            (candidate.get("content") or {}).get("parts"), self._signatures, remember=False
        )
        if not parsed.images:
            said = f" It said: {parsed.text[:300]}" if parsed.text else ""
            raise ImageRefusal(
                f"Gemini made no image (finishReason={candidate.get('finishReason')}).{said}"
            )
        images: list[GeneratedImage] = []
        for inline in parsed.images:
            raw = wire.decode_b64(str(inline["data"]), "image")
            media = sniff(raw)
            if media is None:
                raise CliError("an image in Gemini's answer was not PNG, JPEG, WebP, GIF or SVG")
            images.append(GeneratedImage(data=str(inline["data"]), mediaType=media))
        meta = body.get("usageMetadata") if isinstance(body.get("usageMetadata"), dict) else {}
        usage = wire.usage_of(meta)
        return ImageResponse(
            images=images,
            usage=ImageUsage(
                inputTokens=usage.promptTokens,
                outputTokens=usage.completionTokens,
                totalTokens=usage.totalTokens,
            )
            if usage
            else None,
            modelId=target.id,
            latencyMs=int((time.perf_counter() - started) * 1000),
        )

    # -- video (Veo) ----------------------------------------------------------------

    @staticmethod
    def _job_id(operation: str) -> str:
        return base64.urlsafe_b64encode(operation.encode()).decode().rstrip("=")

    @staticmethod
    def _operation(job_id: str) -> str:
        try:
            name = base64.urlsafe_b64decode(job_id + "=" * (-len(job_id) % 4)).decode()
        except (ValueError, UnicodeDecodeError):
            raise CliError("No such video job.", upstream_status=404) from None
        if not _SAFE_NAME.match(name) or ".." in name or name.startswith("/"):
            raise CliError("No such video job.", upstream_status=404)
        return name

    async def video(self, request: VideoRequest, first_frame: bytes | None) -> VideoJob:
        """Submit one Veo job (`predictLongRunning`); the answer is the job."""
        target = self.resolve_model(request.model)
        if "video" not in target.entry.surfaces:
            raise VideoRefusal(
                f"{target.id!r} makes no videos; choose a model with the video surface"
            )
        caps = target.entry.capabilities.video if target.entry.capabilities else None
        parameters: dict[str, Any] = {}
        resolution: str | None = None
        if request.size is not None:
            listed = caps.sizes if caps and caps.sizes else list(wire.VIDEO_SIZES)
            if request.size not in listed or request.size not in wire.VIDEO_SIZES:
                raise VideoRefusal(
                    f"size: {target.id!r} makes {', '.join(listed)}; {request.size!r} is not one"
                )
            parameters["aspectRatio"], resolution = wire.VIDEO_SIZES[request.size]
            parameters["resolution"] = resolution
        if request.seconds is not None:
            allowed = caps.durations if caps and caps.durations else list(wire.VIDEO_SECONDS)
            if request.seconds not in allowed:
                raise VideoRefusal(
                    f"seconds: Veo makes {', '.join(str(s) for s in allowed)} second videos, "
                    f"not {request.seconds}"
                )
            parameters["durationSeconds"] = str(request.seconds)
        if resolution in ("1080p", "4k") and request.seconds not in (None, 8):
            raise VideoRefusal(
                f"seconds: Veo makes {resolution} only at 8 seconds, not {request.seconds}"
            )
        instance: dict[str, Any] = {"prompt": request.prompt}
        frame = request.firstFrame
        if frame is not None and first_frame is not None:
            instance["image"] = {
                "inlineData": {
                    "mimeType": sniff(first_frame) or frame.mediaType,
                    "data": base64.b64encode(first_frame).decode("ascii"),
                }
            }
        payload: dict[str, Any] = {"instances": [instance]}
        if parameters:
            payload["parameters"] = parameters
        with self._guard("the video submit"):
            response = await self._client().post(
                f"/models/{target.upstream}:predictLongRunning",
                headers=self._headers(),
                json=payload,
            )
        if response.status_code >= 400:
            raise self._failure(response, "a video submit")
        try:
            name = response.json()["name"]
            if not isinstance(name, str) or not name:
                raise TypeError("name is not a string")
        except (ValueError, KeyError, TypeError) as e:
            raise CliError(f"Gemini's video submit answer was not usable: {e!r}") from e
        return VideoJob(jobId=self._job_id(name), status=VideoJobStatus.queued, modelId=target.id)

    async def _poll(self, job_id: str) -> dict[str, Any]:
        name = self._operation(job_id)
        with self._guard("the video poll"):
            response = await self._client().get(
                f"/{name}", headers=self._headers(), timeout=_LIST_TIMEOUT
            )
        if response.status_code >= 400:
            raise self._failure(response, "a video poll")
        try:
            body = response.json()
        except ValueError as e:
            raise CliError("Gemini's video poll answer was not JSON") from e
        if not isinstance(body, dict):
            raise CliError("Gemini's video poll answer was not an object")
        return body

    @staticmethod
    def _samples(body: dict[str, Any]) -> list[dict[str, Any]]:
        response = body.get("response")
        generated = (
            (response or {}).get("generateVideoResponse") if isinstance(response, dict) else None
        )
        samples = (generated or {}).get("generatedSamples") if isinstance(generated, dict) else None
        return [s for s in samples if isinstance(s, dict)] if isinstance(samples, list) else []

    async def video_job(self, job_id: str) -> VideoJob:
        body = await self._poll(job_id)
        if not body.get("done"):
            return VideoJob(jobId=job_id, status=VideoJobStatus.in_progress)
        error = body.get("error")
        if isinstance(error, dict):
            message = self._scrub(str(error.get("message") or error.get("status") or "failed"))
            return VideoJob(jobId=job_id, status=VideoJobStatus.failed, error=f"Gemini: {message}")
        if not self._samples(body):
            generated = (body.get("response") or {}).get("generateVideoResponse") or {}
            reasons = generated.get("raiMediaFilteredReasons")
            said = (
                "; ".join(str(r) for r in reasons)
                if isinstance(reasons, list)
                else "no reason given"
            )
            return VideoJob(
                jobId=job_id,
                status=VideoJobStatus.failed,
                error=f"Gemini made no video (filtered by its safety checks: {said})",
            )
        return VideoJob(jobId=job_id, status=VideoJobStatus.completed, progress=100)

    async def video_content(self, job_id: str) -> AsyncGenerator[bytes, None]:
        """A finished job's video, downloaded with the key from the URI Google
        named. The key is sent only to Google's own API host, and a redirect
        (to its storage) is followed without it."""
        body = await self._poll(job_id)
        samples = self._samples(body) if body.get("done") else []
        uri = ((samples[0].get("video") or {}).get("uri")) if samples else None
        if not isinstance(uri, str) or not uri:
            raise CliError("This video job has no finished video yet.", upstream_status=404)
        # The key goes only where `baseUrl` already sends it: the same scheme,
        # host and port (https for Google; a fixture or proxy may be http).
        base = urlsplit(self._base_url)
        named = urlsplit(uri)
        if (named.scheme, named.hostname, named.port) != (base.scheme, base.hostname, base.port):
            raise CliError(
                "Gemini named a video address outside its own API host; the key was not sent."
            )
        safe_schemes = {"https", base.scheme}
        with self._guard("the video download"):
            async with self._client().stream(
                "GET", uri, headers={**self._headers(), "Accept": "video/mp4"}
            ) as response:
                if response.is_redirect:
                    target = response.headers.get("location", "")
                    await response.aread()
                    if urlsplit(target).scheme not in safe_schemes:
                        raise CliError("Gemini redirected the video download somewhere unsafe.")
                    async with self._client().stream("GET", target) as followed:
                        if followed.status_code >= 400:
                            await followed.aread()
                            raise self._failure(followed, "the video download")
                        async for chunk in followed.aiter_raw():
                            if chunk:
                                yield chunk
                    return
                if response.status_code >= 400:
                    await response.aread()
                    raise self._failure(response, "the video download")
                async for chunk in response.aiter_raw():
                    if chunk:
                        yield chunk

    # -- speech ------------------------------------------------------------------------

    def resolve_speech(self, requested: str | None) -> _Resolved:
        target = self.resolve_model(requested)
        if "speech" not in target.entry.surfaces:
            raise SpeechRefusal(
                f"{target.id!r} does not speak; choose a model with the speech surface"
            )
        return target

    async def speak(self, request: SpeakRequest) -> AsyncGenerator[bytes, None]:
        """Speech by `generateContent` with an audio answer: PCM, served as WAV
        or raw PCM and nothing else (G5)."""
        target = self.resolve_speech(request.model)
        asked = request.format or self.default_speech_format
        refuse_format(asked, wire.SPEECH_FORMATS)
        if request.speed is not None:
            raise SpeechRefusal(
                "speed: Gemini's speech takes no speed and would ignore it; remove it"
            )
        if request.instructions:
            raise SpeechRefusal(
                "instructions: Gemini speaks the text it is given and takes style only as part "
                "of that text, so instructions would be dropped; remove them"
            )
        payload = {
            "contents": [{"role": "user", "parts": [{"text": request.input}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": request.voice}}
                },
            },
        }
        with self._guard("the speech request"):
            response = await self._client().post(
                f"/models/{target.upstream}:generateContent",
                headers=self._headers(),
                json=payload,
            )
        if response.status_code >= 400:
            raise self._failure(response, "speech")
        try:
            body = response.json()
        except ValueError as e:
            raise CliError("Gemini returned non-JSON for speech") from e
        candidates = body.get("candidates") if isinstance(body, dict) else None
        if not candidates:
            raise wire.blocked_prompt(body) or CliError("Gemini returned no candidates")
        candidate = candidates[0] if isinstance(candidates[0], dict) else {}
        parsed = wire.read_parts(
            (candidate.get("content") or {}).get("parts"), self._signatures, remember=False
        )
        if not parsed.audio:
            raise CliError(f"Gemini made no audio (finishReason={candidate.get('finishReason')})")
        inline = parsed.audio[0]
        raw = wire.decode_b64(str(inline["data"]), "audio")
        rate = wire.pcm_rate(str(inline.get("mimeType") or ""))
        has_header = raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"
        if asked is SpeechFormat.wav:
            yield raw if has_header else wire.wav_of(raw, rate)
        else:
            yield raw[44:] if has_header else raw

    # -- transcription ---------------------------------------------------------------

    async def transcribe(self, request: TranscribeRequest, audio: bytes) -> TranscribeResponse:
        """A transcript by `generateContent` with the audio inline and an
        instruction to transcribe it."""
        started = time.perf_counter()
        target = self.resolve_model(request.model)
        if "transcription" not in target.entry.surfaces:
            raise TranscriptionRefusal(f"{target.id!r} does not transcribe")
        if request.timestampGranularities:
            raise TranscriptionRefusal(
                "timestamp_granularities: Gemini returns text without timings, so word or "
                "segment timestamps would be invented; remove them"
            )
        mime = self._audio_mime(request)
        instruction = (
            "Transcribe this audio exactly as spoken. Reply with the transcript text only, "
            "with no commentary, headings or translation."
        )
        if request.language:
            instruction += f" The language spoken is {request.language}."
        if request.prompt:
            instruction += f" Context and spellings to expect: {request.prompt}"
        payload: dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": mime,
                                "data": base64.b64encode(audio).decode("ascii"),
                            }
                        },
                        {"text": instruction},
                    ],
                }
            ]
        }
        if request.temperature is not None:
            payload["generationConfig"] = {"temperature": request.temperature}
        with self._guard("the transcription"):
            response = await self._client().post(
                f"/models/{target.upstream}:generateContent",
                headers=self._headers(),
                json=payload,
            )
        if response.status_code >= 400:
            raise self._failure(response, "a transcription")
        try:
            body = response.json()
        except ValueError as e:
            raise CliError("Gemini returned non-JSON for a transcription") from e
        candidates = body.get("candidates") if isinstance(body, dict) else None
        if not candidates:
            raise wire.blocked_prompt(body) or CliError("Gemini returned no candidates")
        candidate = candidates[0] if isinstance(candidates[0], dict) else {}
        parsed = wire.read_parts(
            (candidate.get("content") or {}).get("parts"), self._signatures, remember=False
        )
        if not parsed.text.strip():
            raise CliError(
                "Gemini returned no transcript (finishReason="
                f"{candidate.get('finishReason')}); the audio may be silent or blocked"
            )
        usage = wire.usage_of(body.get("usageMetadata"))
        return TranscribeResponse(
            text=parsed.text.strip(),
            usage=TranscriptionUsage(
                inputTokens=usage.promptTokens,
                outputTokens=usage.completionTokens,
                totalTokens=usage.totalTokens,
            )
            if usage
            else None,
            modelId=target.id,
            latencyMs=int((time.perf_counter() - started) * 1000),
        )

    @staticmethod
    def _audio_mime(request: TranscribeRequest) -> str:
        declared = (request.audio.mediaType or "").split(";")[0].strip().lower()
        extension = request.audio.filename.rsplit(".", 1)[-1].lower()
        for key in (declared.removeprefix("audio/"), declared.removeprefix("audio/x-"), extension):
            if key in _AUDIO_TYPES:
                return _AUDIO_TYPES[key]
        raise TranscriptionRefusal(
            "audio: Gemini reads wav, mp3, aac, flac, ogg and aiff; this file is "
            f"{declared or extension!r}"
        )
