"""POST /v1/moderate: text or an image in, a verdict out (P6, 2026-09-28)."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from .._generated.models import ModerateRequest, ModerateResponse, ModerationPartType, Problem
from ..disconnect import ClientGone, serve_while_connected
from ..engines._subprocess import CliError
from ..failures import request_id
from ..locality import enforce
from .generate import _backend_error, _client_gone, _not_configured, _resolve_model

router = APIRouter(tags=["inference"])

log = logging.getLogger(__name__)


def _refused(title: str, detail: str, kind: str) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail=Problem(
            type=f"https://github.com/eugene-plexus/inference-driver#{kind}",
            title=title,
            status=400,
            detail=detail,
            component="inference-driver",
        ).model_dump(exclude_none=True),
    )


def _shape_refusal(body: ModerateRequest) -> str | None:
    """What is wrong with the input's shape, before any backend is asked."""
    if bool(body.texts) == bool(body.parts):
        return "texts, parts: exactly one of them, as OpenAI's input is strings or parts"
    for i, part in enumerate(body.parts or []):
        if part.type is ModerationPartType.text and part.text is None:
            return f"parts[{i}].text: a text part carries its text"
        if part.type is ModerationPartType.image and not (part.image or "").startswith("data:"):
            return f"parts[{i}].image: a data: URL; a remote image is never forwarded (A4)"
    return None


@router.post("/v1/moderate", response_model=ModerateResponse, response_model_exclude_none=True)
async def moderate(request: Request, body: ModerateRequest) -> ModerateResponse:
    """The verdict, from whichever backend this driver fronts.

    **Refused, never guessed at**: a model without the `moderation` surface
    gets a 400 naming this backend before anything is sent. Only an
    account lists one today; no local engine moderates.
    """
    engine: Any = request.app.state.adapter
    if engine is None:
        raise _not_configured(getattr(request.app.state, "adapter_error", None))
    entry = _resolve_model(engine, body.model)
    enforce(engine, body.localOnly)
    kind_label = getattr(engine.backend_kind, "value", str(engine.backend_kind))
    moderates = getattr(engine, "moderate", None) is not None and (
        entry is not None and "moderation" in entry.surfaces
    )
    if not moderates:
        raise _refused(
            "This backend does not moderate",
            f"This driver's backend ({kind_label}) does not moderate with this model, so "
            "nothing was sent. GET /v1/info reports each model's surfaces.",
            "moderation-unsupported",
        )
    wrong = _shape_refusal(body)
    if wrong is not None:
        raise _refused("Moderation request refused", f"{wrong}. Nothing was sent.", "bad-request")
    token = request_id.set(str(body.requestId) if body.requestId else None)
    try:
        answer: ModerateResponse = await serve_while_connected(
            request, engine.moderate(body), what="a moderation"
        )
        return answer
    except ClientGone as e:
        raise _client_gone() from e
    except CliError as e:
        log.warning("moderation failed: %s", e)
        raise _backend_error(e, kind_label) from e
    finally:
        request_id.reset(token)
