"""POST /v1/video, GET /v1/video/{jobId} and its content (P5, 2026-09-28)."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from .._generated.models import Problem, VideoJob, VideoRequest
from ..disconnect import ClientGone, serve_while_connected
from ..engines._subprocess import CliError
from ..failures import request_id
from ..images_out import MAX_UPLOAD_BYTES, ImageRefusal, decode_upload
from ..locality import enforce
from ..videos_out import VideoRefusal
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


def _engine(request: Request) -> Any:
    engine: Any = request.app.state.adapter
    if engine is None:
        raise _not_configured(getattr(request.app.state, "adapter_error", None))
    if getattr(engine, "video", None) is None:
        kind = getattr(engine.backend_kind, "value", str(engine.backend_kind))
        raise _refused(
            "This backend makes no videos",
            f"This driver's backend ({kind}) makes no videos.",
            "video-unsupported",
        )
    return engine


def _failed(e: CliError, engine: Any) -> HTTPException:
    """An unknown job is the backend's 404 and stays one: the gateway reads
    it as *no such job*, not as a request to fix."""
    if getattr(e, "upstream_status", None) == 404:
        return _refused("No such video job", str(e), "video-not-found", code=404)
    return _backend_error(e, getattr(engine.backend_kind, "value", str(engine.backend_kind)))


@router.post("/v1/video", response_model=VideoJob, response_model_exclude_none=True)
async def video(request: Request, body: VideoRequest) -> VideoJob:
    """Submit one job to the backend this driver fronts; the answer is the job."""
    engine = _engine(request)
    entry = _resolve_model(engine, body.model)
    enforce(engine, body.localOnly)
    if entry is not None and "video" not in entry.surfaces:
        raise _refused(
            "This model makes no videos",
            f"{entry.id!r} answers {', '.join(entry.surfaces) or 'nothing'}, not video.",
            "video-unsupported",
        )
    first_frame: bytes | None = None
    if body.firstFrame is not None:
        try:
            first_frame = decode_upload(body.firstFrame.data, "firstFrame")
        except ImageRefusal as e:
            raise _refused("Not an image", str(e), "bad-image") from None
        if len(first_frame) > MAX_UPLOAD_BYTES:
            raise _refused(
                "Image too large",
                f"firstFrame: {len(first_frame)} bytes is over the 25 MiB limit.",
                "image-too-large",
                code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            )
    token = request_id.set(str(body.requestId) if body.requestId else None)
    try:
        job: VideoJob = await serve_while_connected(
            request, engine.video(body, first_frame), what="a video submit"
        )
        return job
    except ClientGone as e:
        raise _client_gone() from e
    except VideoRefusal as e:
        raise _refused("Video request refused", str(e), "video-refused") from None
    except CliError as e:
        log.warning("video submit failed: %s", e)
        raise _failed(e, engine) from e
    finally:
        request_id.reset(token)


@router.get("/v1/video/{job_id}", response_model=VideoJob, response_model_exclude_none=True)
async def video_job(request: Request, job_id: str) -> VideoJob:
    """The job as the backend reports it now."""
    engine = _engine(request)
    try:
        job: VideoJob = await serve_while_connected(
            request, engine.video_job(job_id), what="a video poll"
        )
        return job
    except ClientGone as e:
        raise _client_gone() from e
    except VideoRefusal as e:
        raise _refused("Video request refused", str(e), "video-refused") from None
    except CliError as e:
        raise _failed(e, engine) from e


@router.get("/v1/video/{job_id}/content")
async def video_content(request: Request, job_id: str) -> StreamingResponse:
    """The MP4, streamed. The first chunk is awaited before the 200, so a
    backend's refusal is still a status code."""
    engine = _engine(request)
    chunks = engine.video_content(job_id)
    try:
        first = await serve_while_connected(request, anext(chunks), what="a video download")
    except StopAsyncIteration:
        first = b""
    except ClientGone as e:
        await chunks.aclose()
        raise _client_gone() from e
    except VideoRefusal as e:
        await chunks.aclose()
        raise _refused("Video request refused", str(e), "video-refused") from None
    except CliError as e:
        await chunks.aclose()
        raise _failed(e, engine) from e

    async def body() -> AsyncIterator[bytes]:
        try:
            if first:
                yield first
            async for chunk in chunks:
                yield chunk
        except CliError as e:
            # The 200 and some of the video are out; the download just ends.
            log.warning("video download failed mid-stream: %s", e)
        finally:
            await chunks.aclose()

    return StreamingResponse(body(), media_type="video/mp4")
