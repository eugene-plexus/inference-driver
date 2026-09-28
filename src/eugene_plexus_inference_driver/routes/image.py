"""POST /v1/image and /v1/image/stream: images made or edited (P4, 2026-09-28)."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from .._generated.models import ImagePartial, ImageRequest, ImageResponse, Problem
from ..disconnect import ClientGone, serve_while_connected
from ..engines._subprocess import CliError
from ..failures import request_id
from ..images_out import MAX_UPLOAD_BYTES, ImageRefusal, decode_upload
from ..locality import enforce
from .generate import _backend_error, _client_gone, _not_configured, _resolve_model

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
        ).model_dump(exclude_none=True),
    )


def _prepare(request: Request, body: ImageRequest) -> tuple[Any, list[bytes], bytes | None]:
    """Everything that can fail before a backend is called, as a status code:
    no engine, a model not served, one that makes no images, `localOnly`,
    and uploads that are not images or are over 25 MiB in all."""
    engine: Any = request.app.state.adapter
    if engine is None:
        raise _not_configured(getattr(request.app.state, "adapter_error", None))
    entry = _resolve_model(engine, body.model)
    enforce(engine, body.localOnly)
    kind_label = getattr(engine.backend_kind, "value", str(engine.backend_kind))
    makes = getattr(engine, "image", None) is not None and (
        entry is not None and "image" in entry.surfaces
    )
    if not makes:
        raise _refused(
            "This backend does not make images",
            f"This driver's backend ({kind_label}) does not make images with this model, so "
            "nothing was sent. GET /v1/info reports each model's surfaces.",
            "image-unsupported",
        )
    try:
        uploads = [
            decode_upload(ref.data, f"references[{i}]")
            for i, ref in enumerate(body.references or [])
        ]
        mask = decode_upload(body.mask.data, "mask") if body.mask is not None else None
    except ImageRefusal as e:
        raise _refused("Not an image", str(e), "bad-image") from None
    total = sum(len(u) for u in uploads) + (len(mask) if mask is not None else 0)
    if total > MAX_UPLOAD_BYTES:
        raise _refused(
            "Images too large",
            f"references and mask: {total} bytes in all is over the 25 MiB limit.",
            "image-too-large",
            code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        )
    return engine, uploads, mask


@router.post("/v1/image", response_model=ImageResponse, response_model_exclude_none=True)
async def image(request: Request, body: ImageRequest) -> ImageResponse:
    """The images, from whichever backend this driver fronts."""
    engine, uploads, mask = _prepare(request, body)
    kind_label = getattr(engine.backend_kind, "value", str(engine.backend_kind))
    token = request_id.set(str(body.requestId) if body.requestId else None)
    try:
        answer: ImageResponse = await serve_while_connected(
            request, engine.image(body, uploads, mask), what="an image request"
        )
        return answer
    except ClientGone as e:
        raise _client_gone() from e
    except ImageRefusal as e:
        raise _refused("Image request refused", str(e), "image-refused") from None
    except CliError as e:
        log.warning("image request failed: %s", e)
        raise _backend_error(e, kind_label) from e
    finally:
        request_id.reset(token)


def _frame(item: ImagePartial | ImageResponse) -> str:
    event = "partial" if isinstance(item, ImagePartial) else "done"
    return f"event: {event}\ndata: {item.model_dump_json(exclude_none=True)}\n\n"


@router.post("/v1/image/stream")
async def image_stream(request: Request, body: ImageRequest) -> StreamingResponse:
    """`partial` events, then `done`, or `error`.

    **The first event is awaited before the 200 is committed**, as
    `/v1/generate/stream` does, so every failure that can be a status code
    is one: a model that does not stream, a backend that answered JSON
    instead of a stream, a backend refusal before it rendered anything.
    """
    engine, uploads, mask = _prepare(request, body)
    kind_label = getattr(engine.backend_kind, "value", str(engine.backend_kind))
    stream_of = getattr(engine, "image_stream", None)
    if stream_of is None:
        raise _refused(
            "This backend does not stream images",
            f"This driver's backend ({kind_label}) does not stream partial images.",
            "image-refused",
        )
    stream = stream_of(body, uploads, mask)
    token = request_id.set(str(body.requestId) if body.requestId else None)
    try:
        first = await serve_while_connected(request, anext(stream), what="an image stream")
    except StopAsyncIteration:
        first = None
    except ClientGone as e:
        await stream.aclose()
        raise _client_gone() from e
    except ImageRefusal as e:
        await stream.aclose()
        raise _refused("Image request refused", str(e), "image-refused") from None
    except CliError as e:
        log.warning("image stream failed before it opened: %s", e)
        await stream.aclose()
        raise _backend_error(e, kind_label) from e
    finally:
        request_id.reset(token)

    async def events() -> AsyncIterator[str]:
        try:
            if first is not None:
                yield _frame(first)
            async for item in stream:
                yield _frame(item)
        except (CliError, ImageRefusal) as e:
            log.warning("image stream failed mid-stream: %s", e)
            detail = (
                _backend_error(e, kind_label).detail
                if isinstance(e, CliError)
                else _refused("Image request refused", str(e), "image-refused").detail
            )
            yield "event: error\ndata: " + json.dumps(detail) + "\n\n"
        finally:
            await stream.aclose()

    return StreamingResponse(events(), media_type="text/event-stream")
