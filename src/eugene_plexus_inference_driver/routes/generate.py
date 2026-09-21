"""POST /v1/generate and POST /v1/generate/stream, both real since M10."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from .._generated.models import (
    EmbedRequest,
    EmbedResponse,
    GenerateRequest,
    GenerateResponse,
    Problem,
)
from ..disconnect import ClientGone, serve_while_connected
from ..engines._subprocess import BackendTimeout, CliError
from ..images import ImageRefusal, has_images, validate_messages

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
    await _validate_content(engine, body)
    try:
        return await serve_while_connected(request, engine.generate(body), what="a generation")
    except ClientGone as e:
        raise _client_gone() from e
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
    await _validate_content(engine, body)

    stream = engine.stream(body)
    kind_label = getattr(engine.backend_kind, "value", str(engine.backend_kind))

    try:
        # **The window Starlette does not cover.** Everything below this
        # point is inside `StreamingResponse`, which races the body
        # against `http.disconnect` itself. This await is not: it is
        # deliberately before the response so an early failure can still
        # be a status code, and on a cold engine it is the whole model
        # load plus the prefill -- minutes, on exactly the box where a
        # caller gives up.
        first = await serve_while_connected(request, anext(stream), what="a streamed prefill")
    except StopAsyncIteration:
        first = None
    except ClientGone as e:
        await stream.aclose()
        raise _client_gone() from e
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
            yield _error_frame(str(e), kind_label, timed_out=isinstance(e, BackendTimeout))
        finally:
            # A client that disconnects abandons this generator, and the
            # engine's own `finally` is what kills the subprocess or
            # releases the upstream response. Closing explicitly means
            # that happens here rather than whenever the loop is
            # collected.
            await stream.aclose()

    return StreamingResponse(events(), media_type="text/event-stream")


@router.post("/v1/embed", response_model=EmbedResponse)
async def embed(request: Request, body: EmbedRequest) -> EmbedResponse:
    """Text in, vectors out, in the order the text arrived.

    **Refused, never substituted.** A backend that cannot embed gets a
    400 naming itself, rather than anything that might be mistaken for
    an embedding. That is the same rule tool calling landed on and for
    a sharper reason: a caller cannot look at a vector and tell whether
    it is wrong, and if it reaches a vector store the mistake outlives
    the request.
    """
    engine: BackendEngine | None = request.app.state.adapter
    if engine is None:
        raise _not_configured(getattr(request.app.state, "adapter_error", None))

    kind_label = getattr(engine.backend_kind, "value", str(engine.backend_kind))
    probe = getattr(engine, "probe_embeddings", None)
    capable = (
        await probe() if probe is not None else bool(getattr(engine, "supports_embeddings", False))
    )
    if not capable:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=Problem(
                type="https://github.com/eugene-plexus/inference-driver#embeddings-unsupported",
                title="Embeddings not supported by this backend",
                status=400,
                detail=(
                    f"This driver's backend ({kind_label}) does not serve embeddings, so the "
                    "request was refused rather than answered with something that is not one. "
                    "GET /v1/info reports capabilities.embeddings; the gateway reports the same "
                    "per model as x_eugene_plexus.surfaces on GET /v1/models. A local engine "
                    "must be started in embedding mode -- it is a property of the running "
                    "backend, not of the model."
                ),
                component=f"inference-driver:{kind_label}",
            ).model_dump(exclude_none=True),
        )

    try:
        return await serve_while_connected(
            request, engine.embed(list(body.input)), what="an embedding"
        )
    except ClientGone as e:
        raise _client_gone() from e
    except CliError as e:
        log.warning("embeddings invocation failed: %s", e)
        raise _backend_error(e, kind_label) from e


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


def _error_frame(detail: str, kind_label: str, *, timed_out: bool = False) -> str:
    """The failure as a frame, once the 200 is already on the wire.

    Carries the same 504/502 split the status codes do, because a
    caller that reads the frame is reading the only description of
    the failure it will ever get -- and "still computing" and
    "broken" are different instructions to whoever is watching.
    """
    problem = Problem(
        type=(
            "https://github.com/eugene-plexus/inference-driver#backend-timeout"
            if timed_out
            else "https://github.com/eugene-plexus/inference-driver#backend-error"
        ),
        title="Backend did not finish in time" if timed_out else "Backend error",
        status=504 if timed_out else 502,
        detail=detail,
        component=f"inference-driver:{kind_label}",
    ).model_dump(exclude_none=True, mode="json")
    return f"event: error\ndata: {json.dumps(problem)}\n\n"


def _client_gone() -> HTTPException:
    """499, the status nginx invented for exactly this and nobody
    standardised. Nothing will read it -- the socket is closed -- but it
    is what the access log records, and "the caller left" and "we failed"
    must not look the same there."""
    return HTTPException(
        status_code=499,
        detail=Problem(
            type="https://github.com/eugene-plexus/inference-driver#client-disconnected",
            title="Client disconnected",
            status=499,
            detail="The caller went away while this request was running; the backend "
            "call was cancelled rather than left to finish into a closed socket.",
            component="inference-driver",
        ).model_dump(exclude_none=True),
    )


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


# Backend 4xx codes that say "not now" rather than "not ever". These
# keep the 502 they have always had, so the gateway's priority-list
# cascade still fires on them -- a rate-limited cloud provider falling
# through to a local engine is the case failover was built for, and the
# smoke test that motivated it was exactly an OpenRouter 429.
_RETRYABLE_BACKEND_STATUSES = frozenset({408, 409, 425, 429})


def _backend_error(e: Exception, kind_label: str) -> HTTPException:
    """A backend failure, as the status the layer above should act on.

    **Two buckets, and the split is the whole point.** The gateway
    cascades a 5xx to the next backend in the slot and hard-fails a 4xx,
    because a request the backend rejected is one the next backend would
    reject identically. That rule is only as good as this function: when
    every backend failure became a 502, the gateway cascaded through
    every replica and every tier of a request that could not succeed
    anywhere, then handed the caller a retryable error.

    The case that matters is an over-long prompt. `llama-server` answers
    one with a 400 naming both numbers -- `n_prompt_tokens` and `n_ctx`
    -- which is a better answer than anything this layer could compute,
    and it used to arrive as "every backend serving this model failed".
    A harness reading a 502 retries the same prompt; a harness reading a
    400 fixes it. Deciding which one it sees is this function's job.

    `upstream_status` is None for a subprocess backend and for a
    transport failure, where there is no status to carry and 502 is
    right.
    """
    if isinstance(e, BackendTimeout):
        # **Not 502, and the distinction is the whole of R2.5.** A 502
        # tells the gateway "this backend is broken, the next one may
        # not be", and the gateway acts on that by sending the same
        # prompt to the next replica and then the next tier -- each
        # taking the same time to do the same work, so a 30B on CPU was
        # declared a total failure at the sum of the deadlines with two
        # engines having computed the answer. A 504 says a deadline
        # fired, which is a fact about the clock rather than about this
        # backend, and is equally true of every other backend serving
        # the model.
        return HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=Problem(
                type="https://github.com/eugene-plexus/inference-driver#backend-timeout",
                title="Backend did not finish in time",
                status=504,
                detail=str(e),
                component=f"inference-driver:{kind_label}",
            ).model_dump(exclude_none=True),
        )

    upstream = getattr(e, "upstream_status", None)
    rejected = (
        isinstance(upstream, int)
        and 400 <= upstream < 500
        and upstream not in _RETRYABLE_BACKEND_STATUSES
    )
    if rejected:
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=Problem(
                type="https://github.com/eugene-plexus/inference-driver#backend-rejected-request",
                title="Backend rejected the request",
                status=400,
                detail=(
                    f"The backend refused this request with HTTP {upstream}, so it was not "
                    f"retried against another backend -- the next one would refuse it too. "
                    f"A prompt longer than the context window is the usual cause and the "
                    f"backend's own message below says so exactly. {e}"
                ),
                component=f"inference-driver:{kind_label}",
            ).model_dump(exclude_none=True),
        )
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


async def _validate_content(engine: Any, body: GenerateRequest) -> None:
    try:
        await run_in_threadpool(validate_messages, body.messages)
    except ImageRefusal as exc:
        raise HTTPException(
            status_code=400,
            detail={"title": "Invalid image input", "status": 400, "detail": str(exc)},
        ) from None
    if not has_images(body.messages):
        return
    probe = getattr(engine, "probe_image_input", None)
    if probe is None or not await probe():
        raise HTTPException(
            status_code=400,
            detail={
                "title": "Image input not supported",
                "status": 400,
                "detail": "This backend has no confirmed vision model loaded. Select a vision "
                "model with its projector loaded; capabilities.imageInput must be true.",
            },
        )
