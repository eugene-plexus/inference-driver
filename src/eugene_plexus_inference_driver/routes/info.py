"""GET /v1/info — driver metadata."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from .. import __version__
from .._generated.models import BackendKind, Capabilities, DriverInfo, Problem
from ..config import ConfigStore

router = APIRouter(tags=["meta"])


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
            capabilities=Capabilities(streaming=bool(getattr(engine, "supports_streaming", False))),
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
