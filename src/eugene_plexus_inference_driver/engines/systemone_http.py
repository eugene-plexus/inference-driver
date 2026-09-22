"""Engine that speaks the TypeSafe System One decision protocol.

Drives every backend that implements `POST /v1/systemone` — a
supervised Kev runtime, another System One-compatible server the
operator points us at, or TypeSafe's own hosted endpoint — the way
`openai_compat_http` drives every chat-completions backend. The
protocol is **pinned**: docs.typesafe.ai/api as read on 2026-09-22,
cross-checked against `kev/serve.py` at commit `1c35199` and a live
kev-0.8b run the same day (`specs/docs/design/decision-models.md` is
the ledger).

**Decisions, never text — and never invented.** The one rule this
module exists to enforce sits in `_validated_answers`: a backend that
answers with a missing question, a wrong type, an illegal choice, a
distribution that is not one, or a number that is not finite has
**failed**, and the caller gets a 502 naming what was malformed — not a
decision the driver made up on the backend's behalf. Provider
probabilities and `confidence` are preserved exactly as reported,
because they are the provider's calibration and are not comparable
across models.

This engine does not chat (`chat_capable = False`), does not stream
(the protocol is one round trip), and does not embed. Raw `state` is
never logged by default — it is the caller's record, often someone
else's ticket.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import AsyncGenerator
from typing import Any, TypeGuard

import httpx

from .._generated.models import (
    BackendKind,
    ConfigField,
    ConfigFieldShowWhen,
    ConfigValueType,
    DecisionAnswer,
    DecisionRequest,
    DecisionResponse,
    EmbedResponse,
    GenerateRequest,
    GenerateResponse,
    Usage,
)
from .._generated.models import (
    Kind as DecisionKind,
)
from .._http import client_for
from ..failures import retry_after
from ._subprocess import BackendTimeout, CliError
from .base import DEFAULT_REQUEST_TIMEOUT_SECONDS, Chunk

log = logging.getLogger(__name__)

#: The protocol's own bounds, from the pin. A backend may be stricter;
#: these are what the driver refuses before any backend work.
MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10

#: The fields the pinned protocol defines on a question. Anything else
#: is refused, never dropped: an unsupported knob silently removed
#: would answer a different question than the caller asked.
_QUESTION_FIELDS = frozenset({"type", "instructions", "criteria"})

#: How far a choice distribution may sum from 1 before it is malformed.
#: Providers round (Kev reports two decimals), so an exact test would
#: refuse honest answers; a distribution off by more than this is not a
#: rounding artifact.
_DISTRIBUTION_TOLERANCE = 0.05


def validate_questions(raw_questions: Any) -> list[str]:
    """Protocol-validate a raw `questions` map. Returns the violations.

    Takes the RAW dict (not the generated models) on purpose: pydantic's
    default is to ignore unknown fields, and the one thing the protocol
    pin demands is that an unknown field is refused rather than
    silently shed. Shared by this driver's route; the gateway carries
    its own copy of the same rules because bounds are enforced before
    any backend work and components share schemas, not code.
    """
    problems: list[str] = []
    if not isinstance(raw_questions, dict) or not raw_questions:
        return ["`questions` must be a non-empty object of named questions"]
    for name, question in raw_questions.items():
        where = f"questions[{name!r}]"
        if not isinstance(question, dict):
            problems.append(f"{where} must be an object")
            continue
        unknown = sorted(set(question) - _QUESTION_FIELDS)
        if unknown:
            problems.append(
                f"{where} carries fields the pinned protocol does not define: "
                f"{', '.join(unknown)} — refused rather than dropped"
            )
        kind = question.get("type")
        if kind not in ("noul", "choice", "score"):
            problems.append(f"{where}.type must be one of noul, choice, score")
            continue
        if "instructions" not in question or question["instructions"] in (None, ""):
            problems.append(f"{where}.instructions is required")
        criteria = question.get("criteria")
        if kind == "noul":
            if criteria is not None and (
                not isinstance(criteria, dict) or set(criteria) - {"true", "false"}
            ):
                problems.append(
                    f'{where}.criteria for noul is an optional object mapping "true" '
                    f'and "false" to outcome descriptions'
                )
        elif kind == "choice":
            if not isinstance(criteria, dict) or not criteria:
                problems.append(f"{where}.criteria for choice must map option names to rubrics")
            elif len(criteria) > MAX_CHOICE_OPTIONS:
                problems.append(
                    f"{where}.criteria has {len(criteria)} options; the protocol's "
                    f"ceiling is {MAX_CHOICE_OPTIONS}"
                )
        elif kind == "score":
            if not isinstance(criteria, list) or not all(isinstance(c, str) for c in criteria):
                problems.append(f"{where}.criteria for score must be an array of level strings")
            elif not (MIN_SCORE_LEVELS <= len(criteria) <= MAX_SCORE_LEVELS):
                problems.append(
                    f"{where}.criteria has {len(criteria)} levels; the protocol takes "
                    f"{MIN_SCORE_LEVELS}-{MAX_SCORE_LEVELS}, ordered lowest first"
                )
    return problems


class SystemOneHttpEngine:
    """The System One transport, one configured backend per driver."""

    backend_kind = BackendKind.systemone_http
    follows_runtimes = True
    supports_streaming = False
    supports_tool_calling = False
    supports_embeddings = False
    #: This backend decides; it does not chat. The generate route reads
    #: this and answers 400 naming /v1/systemone instead of handing a
    #: chat request to a backend that never spoke the protocol.
    chat_capable = False
    decision_kinds = (DecisionKind.noul, DecisionKind.choice, DecisionKind.score)

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str,
        model_id: str | None = None,
        upstream_model_id: str | None = None,
        timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        auth_required: bool = False,
        max_concurrent: int | None = None,
        runtime: str | None = None,
    ) -> None:
        if auth_required and not api_key:
            raise CliError(
                "systemone_http engine has no API key — the hosted provider "
                "requires one; set `apiKey` in config."
            )
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model_id = model_id
        #: Same split as every other engine since B1: the backend sees
        #: this, everyone else sees the public `_model_id`. Kev echoes
        #: whatever it is sent without validating (measured), so for a
        #: supervised runtime this is simply the alias.
        self._upstream_model_id = upstream_model_id or model_id
        self._timeout_seconds = timeout_seconds
        #: How many requests the backend can hold at once, advertised on
        #: /v1/info so the gateway does not over-admit. Kev's server
        #: handles exactly one (a lock, no cross-caller batching).
        self.decision_max_concurrent = max_concurrent
        self.runtime = runtime
        self._http_client: httpx.AsyncClient | None = None

    # --- construction -------------------------------------------------------

    @classmethod
    def field_specs(cls, *, applicable_providers: list[str]) -> list[ConfigField]:
        show_when = ConfigFieldShowWhen(key="provider", equals=applicable_providers)
        return [
            ConfigField(
                key="decisionMaxConcurrent",
                label="Backend concurrency",
                description=(
                    "How many decision requests the backend can hold at once, "
                    "advertised to the gateway so it never over-admits a "
                    "backend that cannot shed work. A supervised Kev runtime's "
                    "companion sets 1 automatically — Kev's server handles one "
                    "request at a time. Leave empty when unknown."
                ),
                category="adapter",
                valueType=ConfigValueType.integer,
                minimum=1,
                requiresRestart=True,
                showWhen=show_when,
            ),
        ]

    @classmethod
    def from_config(
        cls,
        get: Any,
        *,
        default_base_url: str | None,
        backend_kind: BackendKind = BackendKind.systemone_http,
        auth_required: bool = False,
        runtime_url: str | None = None,
        runtime_name: str | None = None,
    ) -> SystemOneHttpEngine:
        del backend_kind  # one protocol, one kind; kept for registry symmetry
        base_url = str(runtime_url or get("baseUrl") or default_base_url or "").strip()
        if not base_url:
            raise CliError(
                "systemone_http engine has no backend. Set `runtimeName` to a Kev "
                "runtime the agent supervises, or `baseUrl` for a System "
                "One-compatible server that is not one."
            )
        raw_concurrent = get("decisionMaxConcurrent")
        return cls(
            api_key=str(get("apiKey") or "") or None,
            base_url=base_url,
            model_id=str(get("modelId") or "") or None,
            upstream_model_id=str(get("upstreamModelId") or "") or None,
            timeout_seconds=float(get("requestTimeoutSeconds") or DEFAULT_REQUEST_TIMEOUT_SECONDS),
            auth_required=auth_required,
            max_concurrent=int(raw_concurrent) if raw_concurrent else None,
            runtime=runtime_name if runtime_url else None,
        )

    def _client(self) -> httpx.AsyncClient:
        """One client for the life of the engine — never per call (R1.1)."""
        if self._http_client is None:
            self._http_client = client_for(
                self._base_url,
                base_url=self._base_url,
                timeout=httpx.Timeout(self._timeout_seconds, connect=10.0),
            )
        return self._http_client

    async def aclose(self) -> None:
        client, self._http_client = self._http_client, None
        if client is not None:
            await client.aclose()

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    # --- the one real operation ---------------------------------------------

    async def decide(self, request: DecisionRequest) -> DecisionResponse:
        """One state, several typed questions, validated answers."""
        started = time.perf_counter()
        questions = {name: q.model_dump(exclude_none=True) for name, q in request.questions.items()}
        violations = validate_questions(questions)
        if violations:
            raise CliError(
                "decision request violates the pinned protocol: " + "; ".join(violations),
                upstream_status=400,
            )
        payload: dict[str, Any] = {
            # The sentinel the pin's own examples use when nothing is
            # configured; a supervised runtime's companion supplies the
            # alias, and the hosted provider a real versioned id.
            "model": self._upstream_model_id or "kev-latest",
            "state": request.state,
            "questions": questions,
        }
        try:
            response = await self._client().post(
                "/v1/systemone",
                headers={
                    **self._headers(),
                    **({"X-Request-ID": str(request.requestId)} if request.requestId else {}),
                },
                json=payload,
            )
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            raise CliError(f"systemone_http could not connect: {e!r}") from e
        except httpx.TimeoutException as e:
            raise BackendTimeout(
                f"systemone_http got no answer within {self._timeout_seconds:.0f}s "
                f"(`requestTimeoutSeconds`); the backend may still be computing "
                f"this decision",
                limit_seconds=self._timeout_seconds,
            ) from e
        except httpx.HTTPError as e:
            raise CliError(f"systemone_http request failed: {e!r}") from e

        if response.status_code >= 400:
            raise CliError(
                f"systemone_http returned {response.status_code}: {response.text[:500]}",
                upstream_status=response.status_code,
                retry_after_seconds=retry_after(response.headers.get("Retry-After")),
            )
        try:
            body = response.json()
        except ValueError as e:
            raise CliError("systemone_http returned non-JSON") from e

        answers = _validated_answers(questions, body)
        usage = body.get("usage") or {}
        return DecisionResponse(
            answers=answers,
            modelId=self._public_model_id(body.get("model")),
            reportedModel=str(body["model"]) if body.get("model") else None,
            backend=self.backend_kind,
            usage=_usage_from(usage),
            latencyMs=int((time.perf_counter() - started) * 1000),
            requestId=request.requestId,
        )

    def _public_model_id(self, reported: object) -> str | None:
        """Same rule as every engine since B1: a translating engine
        answers with the public id; otherwise the backend's echo is
        kept (Kev echoes the request's model verbatim, measured)."""
        if self._model_id and self._upstream_model_id != self._model_id:
            return self._model_id
        return str(reported) if reported else self._model_id

    # --- surfaces this backend does not serve --------------------------------

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        raise CliError(
            "this backend answers typed decisions on POST /v1/systemone, not chat; "
            "point conversations at a chat model",
            upstream_status=400,
        )

    async def stream(self, request: GenerateRequest) -> AsyncGenerator[Chunk, None]:
        raise CliError(
            "this backend answers typed decisions on POST /v1/systemone, not chat; "
            "decisions are non-streaming",
            upstream_status=400,
        )
        yield Chunk()  # pragma: no cover - unreachable; makes this a generator

    async def embed(self, inputs: list[str]) -> EmbedResponse:
        raise CliError(
            "this backend answers typed decisions, not embeddings",
            upstream_status=400,
        )

    async def context_window(self) -> int | None:
        return None

    async def list_models(self) -> list[str]:
        """What the backend says it serves — Kev's `GET /v1/models`
        answers `{models: [{id, ...}]}` (measured). Suggestions only;
        an unreachable backend is an empty list, not a failure."""
        try:
            response = await self._client().get("/v1/models", timeout=5.0)
            body = response.json()
        except (httpx.HTTPError, ValueError):
            return []
        models = body.get("models") if isinstance(body, dict) else None
        if not isinstance(models, list):
            return []
        return [str(m["id"]) for m in models if isinstance(m, dict) and m.get("id")]


def _usage_from(usage: dict[str, Any]) -> Usage | None:
    """TypeSafe's `input_tokens`/`output_tokens` into the shared Usage.

    Absent accounting stays absent — unavailable numbers are unknown,
    never zero, so a metering caller is not told a decision was free.
    """

    def _int(key: str) -> int | None:
        value = usage.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    prompt, completion = _int("input_tokens"), _int("output_tokens")
    if prompt is None and completion is None:
        return None
    total = (
        (prompt or 0) + (completion or 0)
        if (prompt is not None or completion is not None)
        else None
    )
    return Usage(promptTokens=prompt, completionTokens=completion, totalTokens=total)


def _finite(value: Any) -> TypeGuard[float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validated_answers(
    questions: dict[str, dict[str, Any]], body: dict[str, Any]
) -> dict[str, DecisionAnswer]:
    """Every question answered, every answer legal — or a backend error.

    Malformed output is refused with the exact defect named, because the
    alternative — passing through whatever arrived, or worse repairing
    it — is an invented decision with the backend's name on it.
    """
    raw = body.get("answers")
    if not isinstance(raw, dict):
        raise CliError("systemone_http backend answered without an `answers` object")
    missing = sorted(set(questions) - set(raw))
    extra = sorted(set(raw) - set(questions))
    if missing:
        raise CliError(f"systemone_http backend left questions unanswered: {', '.join(missing)}")
    if extra:
        raise CliError(
            f"systemone_http backend answered questions nobody asked: {', '.join(extra)}"
        )

    validated: dict[str, DecisionAnswer] = {}
    for name, question in questions.items():
        answer = raw[name]
        problem = _answer_problem(question, answer)
        if problem:
            raise CliError(f"systemone_http backend answer for {name!r} is malformed: {problem}")
        validated[name] = DecisionAnswer.model_validate(answer)
    return validated


def _answer_problem(question: dict[str, Any], answer: Any) -> str | None:
    if not isinstance(answer, dict):
        return "not an object"
    kind = question["type"]
    if answer.get("type") != kind:
        return f"type {answer.get('type')!r} does not match the question's {kind!r}"
    confidence = answer.get("confidence")
    if confidence is not None and not (_finite(confidence) and 0 <= confidence <= 1):
        return "confidence is not a finite number in [0, 1]"

    if kind == "noul":
        noul = answer.get("noul")
        if not (_finite(noul) and 0 <= noul <= 1):
            return "noul is not a finite probability in [0, 1]"
        return None

    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, dict) or not probabilities:
        return "probabilities missing"
    if not all(_finite(v) and 0 <= v <= 1 for v in probabilities.values()):
        return "probabilities carry a value outside [0, 1] or not finite"
    if abs(sum(probabilities.values()) - 1.0) > _DISTRIBUTION_TOLERANCE:
        return f"probabilities sum to {sum(probabilities.values()):.3f}, not 1"

    if kind == "choice":
        legal = set(question.get("criteria") or {})
        choice = answer.get("choice")
        if choice not in legal:
            return f"choice {choice!r} is not one of the request's options"
        if set(probabilities) - legal:
            return "probabilities name options the request never offered"
        return None

    # score
    levels = question.get("criteria") or []
    legal_indices = {str(i) for i in range(len(levels))}
    score = answer.get("score")
    if not (_finite(score) and 0 <= score <= max(len(levels) - 1, 0)):
        return f"score is not a finite number within the {len(levels)}-level scale"
    if set(probabilities) - legal_indices:
        return "probabilities name levels outside the request's scale"
    legend = answer.get("legend")
    if legend is not None and (not isinstance(legend, dict) or set(legend) - legal_indices):
        return "legend names levels outside the request's scale"
    return None
