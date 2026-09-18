"""Config protocol routes: GET, PATCH, schema, test."""

from __future__ import annotations

import time
from typing import Any

import anyio.to_thread
from fastapi import APIRouter, Request

from .._generated.models import (
    ConfigDocument,
    ConfigSchema,
    ConfigTestRequest,
    ConfigTestResult,
    ConfigUpdateRequest,
    ConfigUpdateResult,
    GenerateRequest,
    Message,
    Role,
)
from ..config import ConfigStore, as_schema

router = APIRouter(tags=["config"])


@router.get("/v1/config", response_model=ConfigDocument)
async def get_config(request: Request) -> ConfigDocument:
    store: ConfigStore = request.app.state.config_store
    return store.as_document()


@router.get("/v1/config/schema", response_model=ConfigSchema)
async def get_config_schema(request: Request) -> ConfigSchema:
    available_models = getattr(request.app.state, "available_models", None) or []
    return as_schema(available_models=available_models)


@router.patch("/v1/config", response_model=ConfigUpdateResult)
async def patch_config(
    request: Request,
    body: ConfigUpdateRequest,
) -> ConfigUpdateResult:
    store: ConfigStore = request.app.state.config_store
    return store.apply_patch(body)


@router.post("/v1/config/test", response_model=ConfigTestResult)
async def test_config(
    request: Request,
    body: ConfigTestRequest | None = None,
) -> ConfigTestResult:
    """Build a temporary engine from saved + override config and run a
    minimal generate() round-trip. Override values are NOT persisted —
    PATCH /v1/config is still required to commit them."""
    # Imported lazily to avoid a routes -> app -> routes circular dep.
    from ..app import build_engine_with, runtime_resolver_for

    start = time.perf_counter()
    store: ConfigStore = request.app.state.config_store
    overrides: dict[str, Any] = {}
    if body and body.overrides:
        overrides = body.overrides.model_dump(exclude_none=True)

    def get(key: str) -> Any:
        return overrides[key] if key in overrides else store.get(key)

    resolver = runtime_resolver_for(request.app)
    try:
        # Same resolver the lifespan uses, so a pending `runtimeName` can
        # be tested against the agent before it is saved. **In a thread**,
        # because resolving a `runtimeName` is a synchronous HTTP call
        # with a five-second cap (`runtime_lookup`, sync on purpose), and
        # on this path it would otherwise block the event loop -- and so
        # every other request this driver is serving -- for the whole of
        # a lookup against an agent that is down.
        engine = await anyio.to_thread.run_sync(
            lambda: build_engine_with(get, resolve_runtime=resolver)
        )
    except Exception as e:
        return ConfigTestResult(
            ok=False,
            component="inference-driver",
            latencyMs=int((time.perf_counter() - start) * 1000),
            error=f"engine construction failed: {e}",
        )

    test_request = GenerateRequest(
        messages=[Message(role=Role.user, content="Reply with exactly: PING")],
    )
    # This engine is a throwaway and owns its own connection pool, so it
    # is closed whichever way the test ends. Without this, every PATCH
    # validated through here would leak one pool for the life of the
    # process -- which is why `_client()` is lazy rather than eager.
    try:
        try:
            response = await engine.generate(test_request)
        except Exception as e:
            return ConfigTestResult(
                ok=False,
                component="inference-driver",
                latencyMs=int((time.perf_counter() - start) * 1000),
                error=f"{engine.backend_kind.value} generate failed: {e}",
            )

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return ConfigTestResult(
            ok=True,
            component="inference-driver",
            latencyMs=elapsed_ms,
            summary=f"{engine.backend_kind.value} responded in {response.latencyMs or 0}ms.",
            sampleOutput=(response.content or "")[:200],
        )
    finally:
        closer = getattr(engine, "aclose", None)
        if closer is not None:
            await closer()
