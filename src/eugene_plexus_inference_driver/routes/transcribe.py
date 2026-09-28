"""POST /v1/transcribe: audio in, text out (P3b, 2026-09-28), and in
English whatever was spoken with `translate` (P3-4)."""

from __future__ import annotations

import base64
import binascii
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status

from .._generated.models import Problem, TranscribeRequest, TranscribeResponse
from ..disconnect import ClientGone, serve_while_connected
from ..engines._subprocess import CliError
from ..failures import request_id
from ..locality import enforce
from ..transcription import TranscriptionRefusal
from .generate import _backend_error, _client_gone, _not_configured, _resolve_model

router = APIRouter(tags=["inference"])

log = logging.getLogger(__name__)

#: OpenAI's upload limit. The route's body limit is larger, for base64.
MAX_AUDIO_BYTES = 25 * 1024 * 1024


def _refused(title: str, detail: str, kind: str, *, code: int = 400) -> HTTPException:
    return HTTPException(
        status_code=code,
        detail=Problem(
            type=f"https://github.com/eugene-plexus/inference-driver#{kind}",
            title=title,
            status=code,
            detail=detail,
            component="inference-driver",
        ).model_dump(exclude_none=True),
    )


async def transcribes(engine: Any, surfaces: list[str] | None, *, translate: bool = False) -> bool:
    """Whether this driver's backend transcribes this model -- or, with
    `translate`, translates it: an account's catalogue says per model; a
    single `llama-server` transcribes only with a projector that hears
    (measured), which `/props` reports, and never translates (its
    `/v1/audio/translations` is a 404, measured)."""
    if getattr(engine, "transcribe", None) is None:
        return False
    if surfaces is not None:
        return ("translation" if translate else "transcription") in surfaces
    if translate:
        return False
    probe = getattr(engine, "probe_audio_input", None)
    try:
        return bool(probe is not None and await probe() is True)
    except Exception:
        return False


@router.post("/v1/transcribe", response_model=TranscribeResponse)
async def transcribe(request: Request, body: TranscribeRequest) -> TranscribeResponse:
    """The transcript, from whichever backend this driver fronts.

    **Refused, never guessed at**: a backend that does not transcribe (or,
    for `translate`, translate) gets a 400 naming itself before any audio
    is sent.
    """
    engine: Any = request.app.state.adapter
    if engine is None:
        raise _not_configured(getattr(request.app.state, "adapter_error", None))
    entry = _resolve_model(engine, body.model)
    enforce(engine, body.localOnly)
    kind_label = getattr(engine.backend_kind, "value", str(engine.backend_kind))
    surfaces = entry.surfaces if entry is not None else None
    if body.translate and not await transcribes(engine, surfaces, translate=True):
        raise _refused(
            "This backend does not translate",
            f"This driver's backend ({kind_label}) does not translate this model, so no audio "
            "was sent. Only OpenAI's whisper models translate; GET /v1/info reports each "
            "model's surfaces.",
            "translation-unsupported",
        )
    if not body.translate and not await transcribes(engine, surfaces):
        raise _refused(
            "This backend does not transcribe",
            f"This driver's backend ({kind_label}) does not transcribe this model, so no audio "
            "was sent. GET /v1/info reports each model's surfaces; a llama-server "
            "transcribes only with a projector that hears.",
            "transcription-unsupported",
        )
    if body.translate and (body.language or body.timestampGranularities):
        field = "language" if body.language else "timestampGranularities"
        raise _refused(
            "Not a translation setting",
            f"{field}: a translation takes neither a language (the text is English) nor "
            "timestamp granularities, as OpenAI's does not.",
            "bad-request",
        )
    try:
        audio = base64.b64decode(body.audio.data, validate=True)
    except (binascii.Error, ValueError):
        raise _refused(
            "Audio is not base64", "audio.data: must be base64 with no data: prefix.", "bad-audio"
        ) from None
    if len(audio) > MAX_AUDIO_BYTES:
        raise _refused(
            "Audio too large",
            f"audio.data: {len(audio)} bytes is over the 25 MiB limit.",
            "audio-too-large",
            code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        )
    if body.timestampGranularities and not body.verbose:
        raise _refused(
            "Timestamps need verbose",
            "timestampGranularities: only with verbose, as OpenAI requires.",
            "bad-request",
        )
    token = request_id.set(str(body.requestId) if body.requestId else None)
    try:
        answer: TranscribeResponse = await serve_while_connected(
            request, engine.transcribe(body, audio), what="a transcription"
        )
        return answer
    except ClientGone as e:
        raise _client_gone() from e
    except TranscriptionRefusal as e:
        raise _refused(
            "Transcription request refused", f"{e}. No audio was sent.", "transcription-refused"
        ) from None
    except CliError as e:
        log.warning("transcription failed: %s", e)
        raise _backend_error(e, kind_label) from e
    finally:
        request_id.reset(token)
