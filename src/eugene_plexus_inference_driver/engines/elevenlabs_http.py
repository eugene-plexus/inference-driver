"""ElevenLabs' own API, for speech (P3a, 2026-09-28).

Nothing about ElevenLabs is OpenAI-shaped (design §0.4): the key rides in
`xi-api-key`, the voice is in the path, the output format is a query
parameter, and the body is `text`, `model_id` and `voice_settings`. This
engine is the translation behind `POST /v1/speak`, so the gateway's
`/v1/audio/speech` -- and the OpenAI SDK pointed at it -- speaks through
ElevenLabs unchanged.

**An account, and it needs `models_read` (call P3-2).** The models are the
account's own list (`GET /v1/models`, the ones that can do text to speech).
A key without that permission offers no models, and `/v1/info`'s catalogue
error says which permission is missing: ElevenLabs' scoped keys refuse with
a message naming it (measured), which is the sentence an operator needs.
Voices are read from `GET /v1/voices` when `voices_read` allows and are
passed through either way (P3-3).

Speech only: no chat, no embeddings, no transcription (P3-1 defers
ElevenLabs' speech-to-text).
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import quote

import httpx

from .._generated.models import (
    BackendKind,
    Capabilities,
    ConfigField,
    ConfigFieldShowWhen,
    ConfigValueType,
    DriverModel,
    EmbedResponse,
    GenerateRequest,
    GenerateResponse,
    SpeakRequest,
    SpeechFormat,
)
from .._http import client_for
from ..failures import retry_after
from ..speech import ELEVENLABS_FORMATS, SpeechRefusal, refuse_format, streaming_wav_header
from ._catalogue import Catalogue, CatalogueError, upstream_words
from ._subprocess import BackendTimeout, CliError
from .base import DEFAULT_REQUEST_TIMEOUT_SECONDS, Chunk, ModelNotServed, ModelRequired

log = logging.getLogger(__name__)

#: OpenAI's format -> ElevenLabs' `output_format`. `wav` is its `pcm_24000`
#: with a header made here, because ElevenLabs' own WAV is Pro-tier only.
_OUTPUT_FORMATS: dict[SpeechFormat, str] = {
    SpeechFormat.mp3: "mp3_44100_128",
    SpeechFormat.opus: "opus_48000_64",
    SpeechFormat.pcm: "pcm_24000",
    SpeechFormat.wav: "pcm_24000",
}

_LIST_TIMEOUT = httpx.Timeout(20.0, connect=5.0)


@dataclass(frozen=True)
class _Resolved:
    entry: DriverModel


class ElevenLabsHttpEngine:
    """Speech through ElevenLabs' text-to-speech route."""

    backend_kind = BackendKind.elevenlabs_http
    chat_capable = False
    supports_tool_calling = False
    supports_streaming = False
    #: No chat settings: this engine does not chat.
    supported_settings: ClassVar[list[str]] = []
    #: A provider account: the model list is the account's own.
    serves_accounts = True
    follows_runtimes = False

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str,
        timeout_seconds: float,
        get: Any = None,
        catalogue_path: Path | None = None,
        provider: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._http: httpx.AsyncClient | None = None
        self.catalogue = Catalogue(
            source="elevenlabs",
            origin=f"{provider or 'elevenlabs'}|{self._base_url}",
            fetch=self._fetch_catalogue,
            get=get or (lambda _key: None),
            path=catalogue_path,
        )
        #: Said once: a key that cannot list voices still speaks.
        self._voices_note_logged = False

    # -- construction -------------------------------------------------------

    @classmethod
    def field_specs(cls, *, applicable_providers: list[str]) -> list[ConfigField]:
        """The account fields this engine reads, shown for ElevenLabs. They
        share their keys with the OpenAI-compatible engine's, and the schema
        builder merges the two `showWhen` lists."""
        show_when = ConfigFieldShowWhen(key="provider", equals=applicable_providers)
        return [
            ConfigField(
                key="apiKey",
                label="API Key",
                description=(
                    "The provider's secret key, sent as it requires: "
                    "`Authorization: Bearer` for OpenAI-compatible providers, "
                    "`xi-api-key` for ElevenLabs. If left blank the driver falls "
                    "back to `OPENAI_API_KEY`, or `ELEVENLABS_API_KEY` for "
                    "ElevenLabs, in the environment that started it."
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
        default_base_url: str,
        auth_required: bool = True,
        catalogue_path: Path | None = None,
        provider: str | None = None,
        backend_kind: BackendKind = BackendKind.elevenlabs_http,
    ) -> ElevenLabsHttpEngine:
        del backend_kind  # one protocol, one kind; kept for registry symmetry
        api_key = str(get("apiKey") or "") or os.environ.get("ELEVENLABS_API_KEY") or None
        if auth_required and not api_key:
            raise CliError(
                "ElevenLabs needs an API key: set apiKey on this driver, or "
                "ELEVENLABS_API_KEY in the environment that starts it."
            )
        return cls(
            api_key=api_key,
            base_url=str(get("baseUrl") or default_base_url),
            timeout_seconds=float(get("requestTimeoutSeconds") or DEFAULT_REQUEST_TIMEOUT_SECONDS),
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
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    def _headers(self) -> dict[str, str]:
        return {"xi-api-key": self._api_key} if self._api_key else {}

    # -- the account -------------------------------------------------------

    async def _fetch_catalogue(self) -> list[DriverModel]:
        client = self._client()
        response = await client.get("/v1/models", headers=self._headers(), timeout=_LIST_TIMEOUT)
        if response.status_code >= 400:
            # A scoped key names the permission it lacks (measured:
            # "missing the permission models_read"), which is the fix.
            raise CatalogueError(
                f"ElevenLabs refused the model list ({response.status_code}): "
                f"{upstream_words(response)}"
            )
        try:
            listed = response.json()
        except ValueError as e:
            raise CatalogueError("ElevenLabs' model list was not JSON") from e
        voices = await self._voices()
        models: list[DriverModel] = []
        for entry in listed if isinstance(listed, list) else []:
            if not isinstance(entry, dict) or not isinstance(entry.get("model_id"), str):
                continue
            if entry.get("can_do_text_to_speech") is False:
                continue
            models.append(
                DriverModel(
                    id=entry["model_id"],
                    name=entry.get("name") if isinstance(entry.get("name"), str) else None,
                    surfaces=["speech"],
                    voices=voices,
                    capabilities=Capabilities(
                        supportedSettings=[],
                        speechFormats=list(ELEVENLABS_FORMATS),
                    ),
                )
            )
        return models

    async def _voices(self) -> list[str] | None:
        """The account's voice ids, or None when the key cannot list them.
        None is not "no voices": any id is passed through (P3-3)."""
        try:
            response = await self._client().get(
                "/v1/voices", headers=self._headers(), timeout=_LIST_TIMEOUT
            )
        except httpx.HTTPError as e:
            log.info("ElevenLabs voices could not be read (%s); voices pass through", e)
            return None
        if response.status_code >= 400:
            if not self._voices_note_logged:
                self._voices_note_logged = True
                log.info(
                    "ElevenLabs voices are not listed (%s): %s. Any voice id is passed "
                    "through; this is said once.",
                    response.status_code,
                    upstream_words(response),
                )
            return None
        try:
            listed = response.json().get("voices")
        except (ValueError, AttributeError):
            return None
        return [v["voice_id"] for v in listed or [] if isinstance(v, dict) and v.get("voice_id")]

    async def list_models(self) -> list[str]:
        return [m.id for m in self.catalogue.exposed()]

    def resolve_model(self, requested: str | None) -> _Resolved:
        """The shape the routes' model check reads, so a chat request naming
        one of these models is told it answers speech, not a 404."""
        return _Resolved(self.resolve_speech(requested))

    def resolve_speech(self, requested: str | None) -> DriverModel:
        if not requested:
            raise ModelRequired("name one of this ElevenLabs account's models")
        entry = self.catalogue.find(requested)
        if entry is None:
            raise ModelNotServed(requested, served=len(self.catalogue.exposed()))
        return entry

    # -- speech ---------------------------------------------------------------

    async def speak(self, request: SpeakRequest) -> AsyncGenerator[bytes, None]:
        """The audio, streamed from ElevenLabs' `/stream` route."""
        entry = self.resolve_speech(request.model)
        asked = request.format or SpeechFormat.mp3
        refuse_format(asked, ELEVENLABS_FORMATS)
        if request.instructions:
            raise SpeechRefusal(
                "instructions: ElevenLabs takes no speaking instructions, so they "
                "would be dropped; remove them, or choose a model that takes them"
            )
        body: dict[str, Any] = {"text": request.input, "model_id": entry.id}
        if request.speed is not None:
            body["voice_settings"] = {"speed": request.speed}
        path = f"/v1/text-to-speech/{quote(request.voice, safe='')}/stream"
        started = time.perf_counter()
        client = self._client()
        try:
            async with client.stream(
                "POST",
                path,
                params={"output_format": _OUTPUT_FORMATS[asked]},
                headers=self._headers(),
                json=body,
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise CliError(
                        f"ElevenLabs returned {response.status_code}: {upstream_words(response)}",
                        upstream_status=response.status_code,
                        retry_after_seconds=retry_after(response.headers.get("Retry-After")),
                    )
                header_sent = asked is not SpeechFormat.wav
                async for chunk in response.aiter_raw():
                    if not chunk:
                        continue
                    if not header_sent:
                        header_sent = True
                        yield streaming_wav_header()
                    yield chunk
        except httpx.ConnectTimeout as e:
            raise CliError(f"ElevenLabs could not be reached: {e!r}") from e
        except httpx.TimeoutException as e:
            raise BackendTimeout(
                f"ElevenLabs did not answer within {self._timeout_seconds:g}s "
                f"({type(e).__name__}). Raise requestTimeoutSeconds on this driver.",
                limit_seconds=self._timeout_seconds,
            ) from e
        except httpx.HTTPError as e:
            raise CliError(f"ElevenLabs speech failed: {e!r}") from e
        log.debug(
            "ElevenLabs spoke %d characters in %.2fs",
            len(request.input),
            time.perf_counter() - started,
        )

    # -- what this engine does not do --------------------------------------------

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        raise CliError("ElevenLabs serves speech only; send chat to a chat model")

    async def stream(self, request: GenerateRequest) -> AsyncGenerator[Chunk, None]:
        raise CliError("ElevenLabs serves speech only; send chat to a chat model")
        yield  # pragma: no cover - makes this an async generator

    async def embed(self, inputs: list[str], *, model: str | None = None) -> EmbedResponse:
        raise CliError("ElevenLabs serves speech only; it does not embed")

    async def context_window(self) -> int | None:
        return None
