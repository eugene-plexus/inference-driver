"""POST /v1/generate and POST /v1/generate/stream, both real since M10."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, cast

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from .._generated.models import (
    AudioOutputFormat,
    DecisionRequest,
    DecisionResponse,
    DriverModel,
    EmbedRequest,
    EmbedResponse,
    GenerateRequest,
    GenerateResponse,
    Problem,
    RetryDisposition,
    TokenCount,
)
from ..audio_out import BATCH_FORMATS, STREAM_FORMATS
from ..disconnect import ClientGone, serve_while_connected
from ..engines._subprocess import BackendTimeout, CliError
from ..engines.base import (
    ModelNotServed,
    ModelRequired,
    TokenCountUnsupported,
    resolve_single_model,
)
from ..engines.systemone_http import validate_questions
from ..failures import credential_refused, disposition, request_id
from ..images import ImageRefusal, attachment_kinds, validate_messages
from ..locality import enforce

if TYPE_CHECKING:
    from ..engines.base import BackendEngine

router = APIRouter(tags=["inference"])

log = logging.getLogger(__name__)


@router.post("/v1/generate", response_model=GenerateResponse)
async def generate(request: Request, body: GenerateRequest) -> GenerateResponse:
    engine: BackendEngine | None = request.app.state.adapter
    if engine is None:
        raise _not_configured(getattr(request.app.state, "adapter_error", None))
    entry = _resolve_model(engine, body.model)
    _refuse_non_chat(engine, entry)
    enforce(engine, body.localOnly)
    _refuse_unsupported_tools(engine, body, entry)
    _refuse_audio_output(engine, body, entry, formats=BATCH_FORMATS)
    await _validate_content(engine, body, entry)
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

    **With `reportProgress` the first frame can be progress**, so the 200
    commits while the backend is still reading the prompt and a failure
    after that is an `event: error` frame. The gateway cascades on that
    frame exactly as on a status code, because progress is not output --
    which is why the caller has to ask for it.
    """
    engine: BackendEngine | None = request.app.state.adapter
    if engine is None:
        raise _not_configured(getattr(request.app.state, "adapter_error", None))
    entry = _resolve_model(engine, body.model)
    _refuse_non_chat(engine, entry)
    enforce(engine, body.localOnly)
    _refuse_unsupported_tools(engine, body, entry)
    _refuse_audio_output(engine, body, entry, formats=STREAM_FORMATS)
    await _validate_content(engine, body, entry)

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
            yield "event: error\ndata: " + json.dumps(_backend_error(e, kind_label).detail) + "\n\n"
        finally:
            # A client that disconnects abandons this generator, and the
            # engine's own `finally` is what kills the subprocess or
            # releases the upstream response. Closing explicitly means
            # that happens here rather than whenever the loop is
            # collected.
            await stream.aclose()

    return StreamingResponse(events(), media_type="text/event-stream")


@router.post("/v1/generate/count", response_model=TokenCount)
async def count_prompt_tokens(request: Request, body: GenerateRequest) -> TokenCount:
    """The prompt `/v1/generate` would send, counted by the backend, generating nothing.

    The same refusals as a generation, in the same order, so a count is
    never answered for a request that would not be served. What only a
    count can say -- *this backend cannot count exactly* -- is a 501,
    which the gateway turns into "count it some other way" rather than
    a failure, because nothing failed.
    """
    engine: BackendEngine | None = request.app.state.adapter
    if engine is None:
        raise _not_configured(getattr(request.app.state, "adapter_error", None))
    entry = _resolve_model(engine, body.model)
    _refuse_non_chat(engine, entry)
    enforce(engine, body.localOnly)
    _refuse_unsupported_tools(engine, body, entry)
    kind_label = getattr(engine.backend_kind, "value", str(engine.backend_kind))
    counter = getattr(engine, "count_prompt_tokens", None)
    if counter is None:
        raise _cannot_count(f"{kind_label} has no prompt to count without generating", kind_label)
    try:
        counted = await serve_while_connected(request, counter(body), what="a token count")
    except TokenCountUnsupported as e:
        raise _cannot_count(str(e), kind_label) from e
    except ClientGone as e:
        raise _client_gone() from e
    except CliError as e:
        log.warning("token count failed: %s", e)
        raise _backend_error(e, kind_label) from e
    return TokenCount(promptTokens=counted)


def _cannot_count(reason: str, kind_label: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=Problem(
            type="https://github.com/eugene-plexus/inference-driver#token-count-unsupported",
            title="This backend cannot count this prompt without generating",
            status=501,
            detail=f"Cannot count exactly: {reason}. Nothing was sent to the model.",
            component=f"inference-driver:{kind_label}",
        ).model_dump(exclude_none=True),
    )


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
    entry = _resolve_model(engine, body.model)
    enforce(engine, body.localOnly)

    kind_label = getattr(engine.backend_kind, "value", str(engine.backend_kind))
    if entry is not None:
        # An account's model says what it is on its own catalogue entry.
        capable = "embeddings" in entry.surfaces
    else:
        probe = getattr(engine, "probe_embeddings", None)
        capable = (
            await probe()
            if probe is not None
            else bool(getattr(engine, "supports_embeddings", False))
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
                    "GET /v1/info reports each model's surfaces; the gateway reports the same "
                    "per model as x_eugene_plexus.surfaces on GET /v1/models. A local engine "
                    "must be started in embedding mode -- it is a property of the running "
                    "backend, not of the model."
                ),
                component=f"inference-driver:{kind_label}",
            ).model_dump(exclude_none=True),
        )

    token = request_id.set(str(body.requestId) if body.requestId else None)
    try:
        return await serve_while_connected(
            request,
            cast(Any, engine).embed(list(body.input), model=body.model),
            what="an embedding",
        )
    except ClientGone as e:
        raise _client_gone() from e
    except CliError as e:
        log.warning("embeddings invocation failed: %s", e)
        raise _backend_error(e, kind_label) from e
    finally:
        request_id.reset(token)


@router.post("/v1/decide", response_model=DecisionResponse)
async def decide(request: Request, body: DecisionRequest) -> DecisionResponse:
    """One state, named typed questions, validated structured answers.

    **Deliberately NOT wrapped in `serve_while_connected`.** For chat, a
    departed caller means cancelling the backend call — the socket
    closes and an interruptible engine stops. A decision backend is the
    opposite case: Kev's server is single-slot and cannot shed work, so
    cancelling our HTTP request frees nothing — the backend keeps
    computing — while making the driver LOOK free. Capacity must remain
    occupied while known local work continues, so the call runs to
    completion and the answer is discarded with the connection, which is
    exactly what over-admission protection requires.
    """
    engine: BackendEngine | None = request.app.state.adapter
    if engine is None:
        raise _not_configured(getattr(request.app.state, "adapter_error", None))
    entry = _resolve_model(engine, body.model)
    kind_label = getattr(engine.backend_kind, "value", str(engine.backend_kind))
    if entry is not None or not getattr(engine, "decision_kinds", None):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=Problem(
                type="https://github.com/eugene-plexus/inference-driver#decisions-unsupported",
                title="This backend does not answer typed decisions",
                status=400,
                detail=(
                    f"This driver's backend ({kind_label}) serves chat, not the System One "
                    "decision protocol — use POST /v1/generate for conversations, or point "
                    "the decision at a model served by a systemone_http driver."
                ),
                component=f"inference-driver:{kind_label}",
            ).model_dump(exclude_none=True),
        )
    enforce(engine, body.localOnly)

    # The pinned protocol refuses unknown fields rather than dropping
    # them, and pydantic has already shed any by the time `body` exists —
    # so the check runs against the RAW question objects.
    raw = await request.json()
    violations = validate_questions(raw.get("questions"))
    if violations:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=Problem(
                type="https://github.com/eugene-plexus/inference-driver#decision-protocol",
                title="Decision request violates the pinned protocol",
                status=400,
                detail="; ".join(violations),
                component=f"inference-driver:{kind_label}",
            ).model_dump(exclude_none=True),
        )

    token = request_id.set(str(body.requestId) if body.requestId else None)
    try:
        # `decide` is not on the BackendEngine protocol — only the
        # System One engine has it, and the capability gate above is the
        # runtime proof. A protocol method would force a stub on engines
        # that must refuse instead.
        result: DecisionResponse = await cast(Any, engine).decide(body)
        return result
    except CliError as e:
        log.warning("decision invocation failed: %s", e)
        raise _backend_error(e, kind_label) from e
    finally:
        request_id.reset(token)


def _resolve_model(engine: BackendEngine, requested: str | None) -> DriverModel | None:
    """Which model this request is for, refused before any backend work.

    Returns the account's catalogue entry, whose surfaces and capabilities
    the gates below read; None for a single-model driver, whose gates read
    the engine as they always have. A model this driver does not serve is
    **404** and never answered by whatever it happens to hold -- answering
    a request for model A with model B is the substitution `models[]`
    exists to rule out.
    """
    try:
        resolver = getattr(engine, "resolve_model", None)
        if resolver is not None:
            return getattr(resolver(requested), "entry", None)
        resolve_single_model(getattr(engine, "_model_id", None), requested)
        return None
    except ModelNotServed as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=Problem(
                type="https://github.com/eugene-plexus/inference-driver#model-not-served",
                title="Model not served by this driver",
                status=404,
                detail=str(e) + " No backend was called.",
                component="inference-driver",
                retryDisposition=RetryDisposition.safe,
            ).model_dump(exclude_none=True),
        ) from None
    except ModelRequired as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=Problem(
                type="https://github.com/eugene-plexus/inference-driver#model-required",
                title="This driver needs to be told which model",
                status=400,
                detail=str(e),
                component="inference-driver",
            ).model_dump(exclude_none=True),
        ) from None


def _refuse_non_chat(engine: BackendEngine, entry: DriverModel | None = None) -> None:
    """A model that does not chat refuses chat with a sentence, not a
    protocol error from inside a backend that never spoke it."""
    if entry is not None:
        if "chat" in entry.surfaces:
            return
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=Problem(
                type="https://github.com/eugene-plexus/inference-driver#chat-unsupported",
                title="This model does not answer chat",
                status=400,
                detail=(
                    f"{entry.id!r} answers "
                    f"{', '.join(entry.surfaces) or 'nothing Eugene serves yet'}, not chat. "
                    "GET /v1/info lists what each model answers."
                ),
                component="inference-driver",
            ).model_dump(exclude_none=True),
        )
    if getattr(engine, "chat_capable", True):
        return
    kind_label = getattr(engine.backend_kind, "value", str(engine.backend_kind))
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=Problem(
            type="https://github.com/eugene-plexus/inference-driver#chat-unsupported",
            title="This backend answers typed decisions, not chat",
            status=400,
            detail=(
                f"This driver's backend ({kind_label}) serves the System One decision "
                "protocol. Send decisions to POST /v1/systemone on the gateway; for a "
                "conversation, name a chat model instead."
            ),
            component=f"inference-driver:{kind_label}",
        ).model_dump(exclude_none=True),
    )


def _frame(chunk: Any) -> str:
    """One SSE event, framed as `inference-driver.yaml` specifies."""
    if getattr(chunk, "done", False):
        result = getattr(chunk, "result", None)
        payload = result.model_dump(exclude_none=True, mode="json") if result else {}
        return f"event: done\ndata: {json.dumps(payload)}\n\n"
    # Not a token at all: what the backend is doing while it produces
    # nothing. Its own event name, so a consumer that counts tokens, or
    # takes the first one as the commit point, cannot mistake it for one.
    progress = getattr(chunk, "progress", None)
    if progress is not None:
        return f"event: progress\ndata: {progress.model_dump_json(exclude_none=True)}\n\n"
    # A token frame carries text, reasoning or tool-call fragments, one
    # kind only: upstream sends them in separate deltas, and merging them
    # here would invent a shape no backend produces and no client expects.
    calls = getattr(chunk, "toolCalls", None)
    if calls:
        fragments = [c.model_dump(exclude_none=True, mode="json") for c in calls]
        return f"event: token\ndata: {json.dumps({'toolCalls': fragments})}\n\n"
    audio = getattr(chunk, "audio", None)
    if audio is not None:
        fragment = audio.model_dump(exclude_none=True, mode="json")
        return f"event: token\ndata: {json.dumps({'audio': fragment})}\n\n"
    reasoning = getattr(chunk, "reasoning", "")
    if reasoning:
        return f"event: token\ndata: {json.dumps({'reasoning': reasoning})}\n\n"
    return f"event: token\ndata: {json.dumps({'text': getattr(chunk, 'text', '')})}\n\n"


def _refuse_audio_output(
    engine: BackendEngine,
    body: GenerateRequest,
    entry: DriverModel | None,
    *,
    formats: frozenset[AudioOutputFormat],
) -> None:
    """400 when the caller asked for a spoken answer this driver cannot give.

    Two ways (P2b): the model does not confirm audio output -- its
    listing's `output_modalities` did not say `audio`, which is every
    local engine and CLI -- or the format is one the backend's `pcm16`
    stream cannot become without a transcoder (P2-1). Both before
    anything is forwarded: a text model asked to speak would answer in
    text, and a caller who asked for MP3 would be handed PCM.
    """
    asked = body.audioOutput
    if asked is None:
        return
    kind_label = getattr(engine.backend_kind, "value", str(engine.backend_kind))
    confirmed = (
        entry is not None
        and entry.capabilities is not None
        and entry.capabilities.audioOutput is True
    )
    if not confirmed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=Problem(
                type="https://github.com/eugene-plexus/inference-driver#audio-output-unsupported",
                title="Audio output not supported",
                status=400,
                detail=(
                    "This model is not confirmed to answer with audio. Select a model that "
                    "speaks; capabilities.audioOutput must be true."
                ),
                component=f"inference-driver:{kind_label}",
            ).model_dump(exclude_none=True),
        )
    if asked.format not in formats:
        served = " or ".join(sorted(f.value for f in formats))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=Problem(
                type="https://github.com/eugene-plexus/inference-driver#audio-format-unsupported",
                title="Audio format not supported",
                status=400,
                detail=(
                    f"audioOutput.format {asked.format.value!r} cannot be served here: the "
                    f"backend streams pcm16 only, so this route can answer in {served}."
                ),
                component=f"inference-driver:{kind_label}",
            ).model_dump(exclude_none=True),
        )


def _refuse_unsupported_tools(
    engine: BackendEngine, body: GenerateRequest, entry: DriverModel | None = None
) -> None:
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
    if entry is not None:
        if entry.capabilities is not None and entry.capabilities.toolCalling:
            return
    elif getattr(engine, "supports_tool_calling", False):
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
            retryDisposition=RetryDisposition.safe,
            status=503,
            detail=(
                f"This driver has no working engine. {adapter_error or 'Unknown error.'} "
                "Update the configuration via PATCH /v1/config and restart the driver."
            ),
            component="inference-driver:degraded",
        ).model_dump(exclude_none=True),
    )


def _backend_error(e: Exception, kind_label: str) -> HTTPException:
    refused = credential_refused(e)
    if refused is not None:
        # The backend refused OUR key, not the caller's request. A 400 here
        # told the caller to fix a request that was fine (measured live
        # against OpenRouter). 502, because the fault is upstream of the
        # caller; `terminal`, because the 4xx non-cascade rule is older
        # than this and changing what we SAY must not change what we DO.
        what = "the account behind its API key has no credit" if refused == 402 else "its API key"
        step = "Add credit to that account, or set" if refused == 402 else "Set"
        wait = getattr(e, "retry_after_seconds", None)
        return HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            headers={"Retry-After": str(int(wait + 0.999))} if wait is not None else None,
            detail=Problem(
                type="https://github.com/eugene-plexus/inference-driver#backend-credential-refused",
                title="Backend refused this driver's credential",
                status=502,
                detail=(
                    f"The backend refused this driver's credential ({what}; HTTP {refused}): {e} "
                    f"Nothing is wrong with the request. {step} a working API key on this driver."
                ),
                component=f"inference-driver:{kind_label}",
                retryDisposition=RetryDisposition.terminal,
                retryAfterSeconds=wait,
            ).model_dump(exclude_none=True),
        )
    outcome = disposition(e)
    code = 504 if isinstance(e, BackendTimeout) else 400 if outcome == "terminal" else 502
    delay = getattr(e, "retry_after_seconds", None)
    message = str(e)
    if outcome == "indeterminate":
        message += (
            " Outcome unknown: work may have occurred. "
            "Eugene will not automatically replay this request."
        )
    return HTTPException(
        status_code=code,
        headers={"Retry-After": str(int(delay + 0.999))} if delay is not None else None,
        detail=Problem(
            type="https://github.com/eugene-plexus/inference-driver#backend-rejected-request"
            if outcome == "terminal"
            else "https://github.com/eugene-plexus/inference-driver#backend-error",
            title="Backend rejected the request"
            if outcome == "terminal"
            else "Backend did not finish in time"
            if code == 504
            else "Backend error",
            status=code,
            detail=message,
            component=f"inference-driver:{kind_label}",
            retryDisposition=RetryDisposition(outcome),
            retryAfterSeconds=delay,
        ).model_dump(exclude_none=True),
    )


async def _validate_content(
    engine: Any, body: GenerateRequest, entry: DriverModel | None = None
) -> None:
    try:
        await run_in_threadpool(validate_messages, body.messages)
    except ImageRefusal as exc:
        raise HTTPException(
            status_code=400,
            detail={"title": "Invalid attachment", "status": 400, "detail": str(exc)},
        ) from None
    for kind in sorted(attachment_kinds(body.messages)):
        flag, title, detail = _UNCONFIRMED[kind]
        if entry is not None:
            # An account's model: its catalogue entry is the confirmation.
            confirmed = entry.capabilities is not None and getattr(entry.capabilities, flag) is True
        else:
            probe = getattr(engine, f"probe_{kind}_input", None)
            confirmed = probe is not None and bool(await probe())
        if not confirmed:
            raise HTTPException(
                status_code=400, detail={"title": title, "status": 400, "detail": detail}
            )


#: Per attachment kind: the capability that confirms it, and the refusal
#: when nothing does. A model that cannot take the input is never sent it.
_UNCONFIRMED = {
    "image": (
        "imageInput",
        "Image input not supported",
        "This backend has no confirmed vision model loaded. Select a vision model with its "
        "projector loaded; capabilities.imageInput must be true.",
    ),
    "audio": (
        "audioInput",
        "Audio input not supported",
        "This backend has no model confirmed to take audio. Select a model that hears audio; "
        "capabilities.audioInput must be true.",
    ),
    "file": (
        "fileInput",
        "File input not supported",
        "This backend has no model confirmed to read files. Select a model that reads PDFs; "
        "capabilities.fileInput must be true.",
    ),
}
