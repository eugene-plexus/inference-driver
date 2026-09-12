"""GET /v1/info — driver metadata."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status

from .. import __version__
from .._generated.models import BackendKind, Capabilities, DriverInfo, Problem
from ..config import ConfigStore

router = APIRouter(tags=["meta"])

log = logging.getLogger(__name__)


@router.get("/v1/info", response_model=DriverInfo)
async def info(request: Request) -> DriverInfo:
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
        backend = engine.backend_kind
        return DriverInfo(
            # Contracted since M0 and populated since M10, when there was
            # finally something true to say: `streaming` means "emits
            # genuinely incremental tokens", which is False for a backend
            # that streams one whole message (codex) even though its
            # endpoint works. A flag that said True for everything would
            # tell a UI nothing.
            capabilities=Capabilities(
                streaming=bool(getattr(engine, "supports_streaming", False)),
                toolCalling=bool(getattr(engine, "supports_tool_calling", False)),
                # Contracted at M0, populated by nothing until step 7 --
                # `streaming`'s own story, one field over, and with the
                # same consequence: every backend the install does not
                # supervise advertised no context window at all, so the
                # gateway published `context_length: null` for the most
                # ordinary local setup there is. Probed from the backend
                # and cached by the engine; None stays None rather than
                # becoming a guess.
                maxContextTokens=await _context_window(engine),
                # The third capability flag this project contracted and
                # left unpopulated -- `streaming` was the first (M10),
                # `maxContextTokens` the second (step 7). Determined by
                # asking the backend, because nothing exposes it: see
                # `probe_embeddings`.
                embeddings=await _embeddings(engine),
            ),
            backend=backend,
            provider=provider_key,
            modelId=store.get("modelId") or None,
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
        backend=backend,
        provider=provider_key,
        modelId=store.get("modelId") or None,
        # Degraded: the configured intent, so a driver that failed to
        # resolve its runtime still says which one it was meant to front.
        runtime=runtime,
        version=__version__,
    )


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
