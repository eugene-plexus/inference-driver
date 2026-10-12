"""POST /v1/rerank: a question and passages in, the passages scored, best first (MR6)."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from .._generated.models import Problem, RerankPassagesRequest, RerankPassagesResponse
from ..disconnect import ClientGone, serve_while_connected
from ..engines._subprocess import CliError
from ..engines.base import RerankUnsupported
from ..failures import request_id
from ..locality import enforce
from .generate import _backend_error, _client_gone, _not_configured, _resolve_model

router = APIRouter(tags=["inference"])

log = logging.getLogger(__name__)


def _unsupported(reason: str, kind_label: str) -> HTTPException:
    """The one 400 for a backend that does not sort results, whoever found out:
    the route (an account, an engine with no rerank) or the backend itself."""
    return HTTPException(
        status_code=400,
        detail=Problem(
            type="https://github.com/eugene-plexus/inference-driver#rerank-unsupported",
            title="This backend does not sort results",
            status=400,
            detail=(
                f"This driver's backend ({kind_label}) does not sort results: {reason}. The "
                "request was refused rather than answered with something that is not a ranking. "
                "GET /v1/info reports each model's surfaces; the gateway reports the same per "
                "model as x_eugene_plexus.surfaces on GET /v1/models. A local llama-server sorts "
                "results only when it is started with --reranking."
            ),
            component=f"inference-driver:{kind_label}",
        ).model_dump(exclude_none=True),
    )


@router.post("/v1/rerank", response_model=RerankPassagesResponse, response_model_exclude_none=True)
async def rerank(request: Request, body: RerankPassagesRequest) -> RerankPassagesResponse:
    """The passages scored against the question, from whichever backend this
    driver fronts: scores 0 to 1 on one scale, best first.

    **Refused, never substituted**, as `/v1/embed` is: a backend that does not
    sort results gets a 400 naming itself, rather than anything that might be
    taken for a ranking. Every engine without a `rerank` is one (the CLI
    subscriptions, a decision engine, an account of Gemini or ElevenLabs), and
    so is every account, because none sorts results yet: an account never
    reports the `rerank` surface, whatever its listing says.
    """
    engine: Any = request.app.state.adapter
    if engine is None:
        raise _not_configured(getattr(request.app.state, "adapter_error", None))
    entry = _resolve_model(engine, body.model)
    enforce(engine, body.localOnly)
    kind_label = getattr(engine.backend_kind, "value", str(engine.backend_kind))
    if entry is not None:
        raise _unsupported(
            f"{entry.id!r} is a model on an account, and no account sorts results yet", kind_label
        )
    if getattr(engine, "rerank", None) is None:
        raise _unsupported("this kind of backend has no rerank", kind_label)
    token = request_id.set(str(body.requestId) if body.requestId else None)
    try:
        answer: RerankPassagesResponse = await serve_while_connected(
            request, engine.rerank(body), what="a rerank"
        )
        return answer
    except ClientGone as e:
        raise _client_gone() from e
    except RerankUnsupported as e:
        raise _unsupported(str(e), kind_label) from None
    except CliError as e:
        log.warning("rerank failed: %s", e)
        raise _backend_error(e, kind_label) from e
    finally:
        request_id.reset(token)
