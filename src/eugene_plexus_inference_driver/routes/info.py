"""GET /v1/info — driver metadata."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status

from .. import __version__
from .._generated.models import (
    BackendKind,
    Capabilities,
    DecisionCapability,
    DriverInfo,
    DriverModel,
    Problem,
)
from ..config import ConfigStore
from ..locality import engine_locality

router = APIRouter(tags=["meta"])

log = logging.getLogger(__name__)


@router.get("/v1/info", response_model=DriverInfo, response_model_exclude_none=True)
async def info(request: Request, models: bool = True, model: str | None = None) -> DriverInfo:
    store: ConfigStore = request.app.state.config_store
    provider_key = str(store.get("provider") or "") or None

    # The configured backend is determined by the engine the registry
    # picks for the configured provider — so prefer the live engine's
    # `backend_kind` over re-deriving it from config. Falls back to
    # config-only inference when the engine isn't constructed
    # (degraded mode), so /v1/info stays useful for ops.
    # The runtime this driver follows, when it was configured with
    # `runtimeName`. Not the driver's own address — that rule stands — but
    # *what it serves*, which is what /v1/info is for: a model that is
    # loaded but unroutable becomes diagnosable from the routing table.
    runtime = str(store.get("runtimeName") or "").strip() or None

    engine = request.app.state.adapter
    if engine is not None:
        catalogue = getattr(engine, "catalogue", None)
        if not models:
            served = None
        elif catalogue is not None:
            # A provider account: its catalogue, filtered by its patterns.
            # No per-model probes -- the listing is the answer (P1-3).
            served = catalogue.exposed()
        else:
            served = await _single_model(engine)
        if served is not None and model is not None:
            # One candidate's entry, for the gateway's per-request re-check.
            served = [m for m in served if m.id == model]
        return DriverInfo(
            locality=engine_locality(engine),
            localOnlyEnforced=True,
            backend=engine.backend_kind,
            provider=provider_key,
            models=served,
            catalogue=catalogue.summary() if catalogue is not None else None,
            # Off the live engine: only set when the URL was genuinely
            # resolved from a runtime, so a stale name on a CLI provider
            # does not claim a runtime it is not fronting.
            runtime=getattr(engine, "runtime", None),
            version=__version__,
        )

    # Degraded mode: derive backend from the provider registry if we
    # can, otherwise 503 with the same error message we always have.
    try:
        from ..providers import get_provider

        provider = get_provider(provider_key) if provider_key else None
        if provider is not None:
            backend = BackendKind(
                provider.engine_kwargs.get("backend_kind") or BackendKind.openai_compat_http
            )
        else:
            backend = BackendKind.claude_code_cli
    except (KeyError, ValueError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=Problem(
                type="https://github.com/eugene-plexus/inference-driver#config-invalid",
                title="Configuration invalid",
                status=503,
                detail=f"provider config value is invalid: {e}",
                component="inference-driver:degraded",
            ).model_dump(exclude_none=True),
        ) from e

    return DriverInfo(
        localOnlyEnforced=True,
        backend=backend,
        provider=provider_key,
        # Degraded: nothing is served, and an empty list says so -- which is
        # not an absent one, which would read as a driver from before P1.
        models=[] if models else None,
        # Degraded: the configured intent, so a driver that failed to
        # resolve its runtime still says which one it was meant to front.
        runtime=runtime,
        version=__version__,
    )


async def _single_model(engine: Any) -> list[DriverModel]:
    """A single-model driver's one entry, probed as `/v1/info` always was.

    Empty when no model is configured: a subscription CLI left on its
    default, or a decision engine with none chosen, serves nothing
    routable -- as a driver with no `modelId` did before P1.
    """
    model_id = getattr(engine, "model_id", None) or getattr(engine, "_model_id", None)
    if not model_id:
        return []
    # The gateway reconfirms locality/settings within four seconds before
    # waking a stopped runtime. Optional backend probes must not serialize
    # their timeouts or let an embedding probe use the generation deadline.
    # Keep known engine policy available; unconfirmed capabilities stay
    # conservative, and cancelling a probe does not cache a false answer.
    image_input, audio_input, context_window, embeddings, completion = await asyncio.gather(
        _bounded_probe(_image_input(engine), False),
        _bounded_probe(_modality_input(engine, "audio"), False),
        _bounded_probe(_context_window(engine), None),
        _bounded_probe(_embeddings(engine), None),
        _bounded_probe(_completion(engine), (False, False)),
    )
    decision = _decision_capability(engine)
    # `surfaces` replaces `capabilities.embeddings` and `chatCapable`, with
    # the gateway's old reading of them kept exactly: a backend that embeds
    # is an embeddings backend, a decision engine answers decisions alone,
    # and everything else -- every driver that predates both -- is chat.
    if decision is not None or not getattr(engine, "chat_capable", True):
        surfaces = ["decisions"]
    elif embeddings is True:
        surfaces = ["embeddings"]
    else:
        surfaces = ["chat"]
        # P3b: `llama-server`'s `/v1/audio/transcriptions` answers only
        # with a projector that hears, which is what `/props` reports as
        # audio (measured), so the same probe says it transcribes.
        if audio_input is True and getattr(engine, "transcribe", None) is not None:
            surfaces.append("transcription")
        # P6: a local `llama-server` or vLLM continues raw text.
        if completion[0]:
            surfaces.append("completion")
    upstream = getattr(engine, "_upstream_model_id", None)
    return [
        DriverModel(
            id=model_id,
            upstreamId=upstream if upstream and upstream != model_id else None,
            surfaces=surfaces,
            capabilities=Capabilities(
                supportedSettings=list(getattr(engine, "supported_settings", [])),
                imageInput=image_input,
                audioInput=audio_input,
                fillInMiddle=completion[1],
                # No local engine reads a PDF part today; an account says
                # per model from its listing instead.
                fileInput=False,
                # Nor speaks (P2b).
                audioOutput=False,
                # Contracted since M0 and populated since M10, when there
                # was finally something true to say: `streaming` means
                # "emits genuinely incremental tokens", which is False for
                # a backend that streams one whole message (codex).
                streaming=bool(getattr(engine, "supports_streaming", False)),
                toolCalling=bool(getattr(engine, "supports_tool_calling", False)),
                # Probed from the backend and cached by the engine; None
                # stays None rather than becoming a guess (step 7).
                maxContextTokens=context_window,
                decision=decision,
            ),
        )
    ]


async def _bounded_probe(probe: Awaitable[Any], fallback: Any) -> Any:
    try:
        async with asyncio.timeout(2.5):
            return await probe
    except TimeoutError:
        return fallback


async def _context_window(engine: Any) -> int | None:
    """The engine's resolved window, and never an exception.

    `/v1/info` is what the gateway polls to build its routing table, and
    a driver that 500s here drops out of routing entirely. A window is
    the least important thing this endpoint reports, so it is the first
    thing to give up: any failure is an absent window, which the
    contract already defines as "unknown".
    """
    probe = getattr(engine, "context_window", None)
    if probe is None:
        return None
    try:
        value = await probe()
    except Exception:  # see docstring: /v1/info must not fail over a window
        log.debug("context-window probe failed; reporting unknown", exc_info=True)
        return None
    return value if isinstance(value, int) and value > 0 else None


async def _completion(engine: Any) -> tuple[bool, bool]:
    """`(continues raw text, fills in the middle)` (P6), never an
    exception, for the reason `_embeddings` gives."""
    probe = getattr(engine, "probe_completion", None)
    if probe is None:
        return (False, False)
    try:
        completes, fills = await probe()
        return (bool(completes), bool(fills))
    except Exception:  # see _embeddings: /v1/info must not fail over a capability
        log.debug("completion probe failed; reporting none", exc_info=True)
        return (False, False)


async def _embeddings(engine: Any) -> bool | None:
    """Whether the backend embeds, and never an exception.

    Same rule as `_context_window`: `/v1/info` is what the gateway polls
    to build its routing table, so a driver that 500s here drops out of
    routing entirely. A capability flag is not worth that, and `None`
    already means "unknown" in the contract.
    """
    probe = getattr(engine, "probe_embeddings", None)
    if probe is None:
        declared = getattr(engine, "supports_embeddings", None)
        return bool(declared) if declared is not None else None
    try:
        return bool(await probe())
    except Exception:  # see docstring: /v1/info must not fail over a capability
        log.debug("embeddings probe failed; reporting unknown", exc_info=True)
        return None


async def _image_input(engine: Any) -> bool:
    return await _modality_input(engine, "image")


async def _modality_input(engine: Any, kind: str) -> bool:
    probe = getattr(engine, f"probe_{kind}_input", None)
    if probe is None:
        return False
    try:
        return await probe() is True
    except Exception:
        return False


def _decision_capability(engine: Any) -> DecisionCapability | None:
    """Present iff this engine decides — its absence is how the gateway
    knows not to route `/v1/systemone` here."""
    kinds = getattr(engine, "decision_kinds", None)
    if not kinds:
        return None
    return DecisionCapability(
        kinds=list(kinds),
        maxOptions=255,
        maxConcurrent=getattr(engine, "decision_max_concurrent", None),
    )
