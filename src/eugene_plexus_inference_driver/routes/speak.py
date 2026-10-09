"""POST /v1/speak: text in, audio bytes out, streamed (P3a, 2026-09-28)."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any, cast

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from .._generated.models import Problem, RetryDisposition, SpeakRequest, SpeechFormat
from ..disconnect import ClientGone, serve_while_connected
from ..engines._subprocess import CliError
from ..engines.base import ModelNotServed, ModelRequired
from ..locality import enforce
from ..speech import MEDIA_TYPES, SpeechRefusal
from .generate import _backend_error, _client_gone, _not_configured

router = APIRouter(tags=["inference"])

log = logging.getLogger(__name__)


def _refused(title: str, detail: str, kind: str, *, code: int = 400) -> HTTPException:
    return HTTPException(
        status_code=code,
        detail=Problem(
            type=f"https://github.com/eugene-plexus/inference-driver#{kind}",
            title=title,
            status=code,
            detail=detail,
            component="inference-driver",
            retryDisposition=RetryDisposition.safe if code == 404 else RetryDisposition.terminal,
        ).model_dump(exclude_none=True),
    )


@router.post("/v1/speak")
async def speak(request: Request, body: SpeakRequest) -> StreamingResponse:
    """The audio, streamed as the backend makes it.

    **Where the status code stops being available**, as for
    `/v1/generate/stream`: everything that can fail cleanly -- no engine,
    a model this driver does not serve, a format it cannot make, the
    backend refusing before its first byte -- is awaited before the
    response starts, so it is still a status code. After the first byte a
    failure can only end the stream.
    """
    engine: Any = request.app.state.adapter
    if engine is None:
        raise _not_configured(getattr(request.app.state, "adapter_error", None))
    kind_label = getattr(engine.backend_kind, "value", str(engine.backend_kind))
    speaker = getattr(engine, "speak", None)
    if speaker is None:
        raise _refused(
            "This backend does not speak",
            f"This driver's backend ({kind_label}) has no speech; send speech to a model "
            "with the speech surface.",
            "speech-unsupported",
        )
    enforce(engine, body.localOnly)
    fmt: SpeechFormat = body.format or cast(
        SpeechFormat, getattr(engine, "default_speech_format", SpeechFormat.mp3)
    )
    chunks = speaker(body)
    try:
        first = await serve_while_connected(request, anext(chunks), what="speech")
    except StopAsyncIteration:
        first = b""
    except ClientGone as e:
        await chunks.aclose()
        raise _client_gone() from e
    except SpeechRefusal as e:
        raise _refused(
            "Speech request refused", f"{e}. No backend was called.", "speech-refused"
        ) from None
    except ModelNotServed as e:
        raise _refused(
            "Model not served by this driver",
            f"{e} No backend was called.",
            "model-not-served",
            code=status.HTTP_404_NOT_FOUND,
        ) from None
    except ModelRequired as e:
        raise _refused("A model is required", str(e), "model-required") from None
    except CliError as e:
        log.warning("speech failed before its first byte: %s", e)
        raise _backend_error(e, kind_label) from e

    async def audio() -> AsyncIterator[bytes]:
        try:
            if first:
                yield first
            async for chunk in chunks:
                yield chunk
        except CliError as e:
            # The 200 and some audio are out; ending the stream is all that
            # is left. The gateway sees a short body, the log says why.
            log.warning("speech failed mid-stream: %s", e)
        finally:
            await chunks.aclose()

    return StreamingResponse(audio(), media_type=MEDIA_TYPES[fmt])
