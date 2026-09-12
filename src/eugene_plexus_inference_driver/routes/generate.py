"""POST /v1/generate and POST /v1/generate/stream, both real since M10."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from .._generated.models import GenerateRequest, GenerateResponse, Problem
from ..engines._subprocess import CliError

if TYPE_CHECKING:
    from ..engines.base import BackendEngine

router = APIRouter(tags=["inference"])

log = logging.getLogger(__name__)


@router.post("/v1/generate", response_model=GenerateResponse)
async def generate(request: Request, body: GenerateRequest) -> GenerateResponse:
    engine: BackendEngine | None = request.app.state.adapter
    if engine is None:
        raise _not_configured(getattr(request.app.state, "adapter_error", None))
    _refuse_unsupported_tools(engine, body)
    try:
        return await engine.generate(body)
    except CliError as e:
        log.warning("backend invocation failed: %s", e)
        # `backend_kind` is BackendKind in production but tests may stub
        # it as a plain string — accept either via getattr.
        kind_label = getattr(engine.backend_kind, "value", str(engine.backend_kind))
        raise _backend_error(e, kind_label) from e


@router.post("/v1/generate/stream")
async def generate_stream(request: Request, body: GenerateRequest) -> StreamingResponse:
    """The contracted SSE stream: `token` events, then `done`, or `error`.

    **Where the status code stops being available.** A driver with no
    engine, or one whose backend refuses before producing anything, can
    still answer with a real HTTP status — so those paths raise exactly
    as `/v1/generate` does. Once the first byte of the stream is out the
    200 is committed, and a failure can only be an `event: error` frame.
    The generator below is therefore split deliberately: everything that
    can fail cleanly happens before it is handed to `StreamingResponse`.

    That is the same rule the gateway applies one layer up, where it has
    sharper teeth: there, a failure before the first token can still
    cascade to another backend, and after it cannot.
    """
    engine: BackendEngine | None = request.app.state.adapter
    if engine is None:
        raise _not_configured(getattr(request.app.state, "adapter_error", None))
    _refuse_unsupported_tools(engine, body)

    stream = engine.stream(body)
    kind_label = getattr(engine.backend_kind, "value", str(engine.backend_kind))

    try:
        first = await anext(stream)
    except StopAsyncIteration:
        first = None
    except CliError as e:
        # Nothing has been sent, so this can still be a status code.
        log.warning("backend invocation failed before the stream opened: %s", e)
        await stream.aclose()
        raise _backend_error(e, kind_label) from e

    async def events() -> AsyncIterator[str]:
        try:
            if first is not None:
                yield _frame(first)
            async for chunk in stream:
                yield _frame(chunk)
        except CliError as e:
            # The 200 is already sent; an error can only be a frame now.
            log.warning("backend failed mid-stream: %s", e)
            yield _error_frame(str(e), kind_label)
        finally:
            # A client that disconnects abandons this generator, and the
            # engine's own `finally` is what kills the subprocess or
            # releases the upstream response. Closing explicitly means
            # that happens here rather than whenever the loop is
            # collected.
            await stream.aclose()

    return StreamingResponse(events(), media_type="text/event-stream")


def _frame(chunk: Any) -> str:
    """One SSE event, framed as `inference-driver.yaml` specifies."""
    if getattr(chunk, "done", False):
        result = getattr(chunk, "result", None)
        payload = result.model_dump(exclude_none=True, mode="json") if result else {}
        return f"event: done\ndata: {json.dumps(payload)}\n\n"
    # A token frame carries text or tool-call fragments, never both:
    # upstream sends them in separate deltas, and merging them here
    # would invent a shape no backend produces and no client expects.
    calls = getattr(chunk, "toolCalls", None)
    if calls:
        fragments = [c.model_dump(exclude_none=True, mode="json") for c in calls]
        return f"event: token\ndata: {json.dumps({'toolCalls': fragments})}\n\n"
    return f"event: token\ndata: {json.dumps({'text': getattr(chunk, 'text', '')})}\n\n"


def _refuse_unsupported_tools(engine: BackendEngine, body: GenerateRequest) -> None:
    """400 when the caller sent tools and this backend cannot carry them.

    **Never silently strip.** A harness that receives a plain answer
    cannot tell "the model chose not to call anything" from "nobody ever
    offered it the tools", and the second is a bug wearing the first
    one's clothes -- it reads as a model being unhelpful, which is where
    days go. `capabilities.toolCalling` exists so the question is
    answerable before a request is ever sent.

    400 and not 502: nothing is wrong with the backend, the request is
    asking it for something it does not do.
    """
    if not body.tools:
        return
    if getattr(engine, "supports_tool_calling", False):
        return
    kind_label = getattr(engine.backend_kind, "value", str(engine.backend_kind))
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=Problem(
            type="https://github.com/eugene-plexus/inference-driver#tools-unsupported",
            title="Tools not supported by this backend",
            status=400,
            detail=(
                f"This driver's backend ({kind_label}) cannot carry tool definitions, "
                "so the request was refused rather than answered without them. "
                "GET /v1/info reports capabilities.toolCalling; the gateway reports "
                "the same per model as x_eugene_plexus.tool_calling on GET /v1/models."
            ),
            component=f"inference-driver:{kind_label}",
        ).model_dump(exclude_none=True),
    )


def _error_frame(detail: str, kind_label: str) -> str:
    problem = Problem(
        type="https://github.com/eugene-plexus/inference-driver#backend-error",
        title="Backend error",
        status=502,
        detail=detail,
        component=f"inference-driver:{kind_label}",
    ).model_dump(exclude_none=True, mode="json")
    return f"event: error\ndata: {json.dumps(problem)}\n\n"


def _not_configured(adapter_error: str | None) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=Problem(
            type="https://github.com/eugene-plexus/inference-driver#engine-not-configured",
            title="Engine not configured",
            status=503,
            detail=(
                f"This driver has no working engine. {adapter_error or 'Unknown error.'} "
                "Update the configuration via PATCH /v1/config and restart the driver."
            ),
            component="inference-driver:degraded",
        ).model_dump(exclude_none=True),
    )


def _backend_error(e: Exception, kind_label: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail=Problem(
            type="https://github.com/eugene-plexus/inference-driver#backend-error",
            title="Backend error",
            status=502,
            detail=str(e),
            component=f"inference-driver:{kind_label}",
        ).model_dump(exclude_none=True),
    )
