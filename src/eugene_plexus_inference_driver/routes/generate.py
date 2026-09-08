"""POST /v1/generate (real) and POST /v1/generate/stream (still 501)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request, status

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
        adapter_error = getattr(request.app.state, "adapter_error", None)
        raise HTTPException(
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
    try:
        return await engine.generate(body)
    except CliError as e:
        log.warning("backend invocation failed: %s", e)
        # `backend_kind` is BackendKind in production but tests may stub
        # it as a plain string — accept either via getattr.
        kind_label = getattr(engine.backend_kind, "value", str(engine.backend_kind))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=Problem(
                type="https://github.com/eugene-plexus/inference-driver#backend-error",
                title="Backend error",
                status=502,
                detail=str(e),
                component=f"inference-driver:{kind_label}",
            ).model_dump(exclude_none=True),
        ) from e


@router.post("/v1/generate/stream")
async def generate_stream(body: GenerateRequest) -> None:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=Problem(
            type="https://github.com/eugene-plexus/inference-driver#not-implemented",
            title="Not Implemented",
            status=501,
            detail=(
                "POST /v1/generate/stream is not yet wired up; will land "
                "alongside the gateway + UI consumers in v0.2."
            ),
            component="inference-driver",
        ).model_dump(exclude_none=True),
    )
