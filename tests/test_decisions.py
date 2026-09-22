"""The System One engine and POST /v1/decide.

Fixture bodies are the MEASURED shapes from a real kev-0.8b server
(commit 1c35199, run 2026-09-22 on WSL CPU — see
specs/docs/design/decision-models.md), not invented: the mixed
noul/choice/score response below is byte-shaped like the live one,
including the float `latency_ms` and the echoed request model.

The module's one rule, tested from both directions: **malformed backend
output is a backend error, never an invented decision** — and a
well-formed answer passes through with its probabilities and confidence
exactly as reported.
"""

from __future__ import annotations

import json

import httpx
import respx
from fastapi.testclient import TestClient

from eugene_plexus_inference_driver._generated.models import (
    DecisionQuestion,
    DecisionRequest,
)
from eugene_plexus_inference_driver._generated.models import (
    Type1 as DecisionType,
)
from eugene_plexus_inference_driver.engines._subprocess import CliError
from eugene_plexus_inference_driver.engines.systemone_http import (
    SystemOneHttpEngine,
    validate_questions,
)

BASE = "http://127.0.0.1:8009"


def _engine(**overrides) -> SystemOneHttpEngine:
    kwargs = {
        "base_url": BASE,
        "model_id": "kev-tiny",
        "upstream_model_id": None,
        "timeout_seconds": 30.0,
        "max_concurrent": 1,
    }
    kwargs.update(overrides)
    return SystemOneHttpEngine(**kwargs)


def _request() -> DecisionRequest:
    return DecisionRequest(
        state="Customer: charged twice. Agent: refunded the duplicate.",
        questions={
            "refunded": DecisionQuestion(
                type=DecisionType.noul, instructions="Was a refund issued?"
            ),
            "route": DecisionQuestion(
                type=DecisionType.choice,
                instructions="Route this ticket.",
                criteria={"billing": "payment", "shipping": "delivery", "technical": "bugs"},
            ),
            "urgency": DecisionQuestion(
                type=DecisionType.score,
                instructions="How urgent now?",
                criteria=["low", "medium", "high"],
            ),
        },
    )


def _kev_body(model: str = "kev-tiny") -> dict:
    """Shaped like the measured live response, float latency included."""
    return {
        "model": model,
        "answers": {
            "refunded": {"type": "noul", "noul": 0.98},
            "route": {
                "type": "choice",
                "choice": "billing",
                "confidence": 0.76,
                "probabilities": {"billing": 0.84, "shipping": 0.07, "technical": 0.09},
            },
            "urgency": {
                "type": "score",
                "score": 0.75,
                "legend": {"0": "low", "1": "medium", "2": "high"},
                "probabilities": {"0": 0.41, "1": 0.43, "2": 0.16},
                "confidence": 0.72,
            },
        },
        "usage": {"input_tokens": 89, "output_tokens": 156},
        "latency_ms": 397.8,
    }


# --------------------------------------------------------------------------- #
# protocol validation, before any backend work
# --------------------------------------------------------------------------- #


def test_unknown_question_fields_are_refused_not_dropped() -> None:
    problems = validate_questions({"q": {"type": "noul", "instructions": "?", "temperature": 0.7}})
    assert len(problems) == 1
    assert "temperature" in problems[0]
    assert "refused rather than dropped" in problems[0]


def test_protocol_bounds_are_enforced() -> None:
    too_many_options = {str(i): None for i in range(256)}
    problems = validate_questions(
        {
            "a": {"type": "choice", "instructions": "?", "criteria": too_many_options},
            "b": {"type": "score", "instructions": "?", "criteria": ["only-one"]},
            "c": {"type": "score", "instructions": "?", "criteria": [str(i) for i in range(11)]},
        }
    )
    assert len(problems) == 3
    assert "255" in problems[0]
    assert "2-10" in problems[1] and "2-10" in problems[2]


def test_unknown_kind_and_missing_instructions_are_named() -> None:
    problems = validate_questions(
        {
            "bool": {"type": "boolean", "instructions": "?"},  # Vercel's dialect, not ours
            "empty": {"type": "noul"},
        }
    )
    assert any("noul, choice, score" in p for p in problems)
    assert any("instructions is required" in p for p in problems)


def test_noul_criteria_shape_is_checked() -> None:
    ok = validate_questions(
        {"q": {"type": "noul", "instructions": "?", "criteria": {"true": "a", "false": "b"}}}
    )
    assert ok == []
    bad = validate_questions(
        {"q": {"type": "noul", "instructions": "?", "criteria": {"maybe": "c"}}}
    )
    assert len(bad) == 1


# --------------------------------------------------------------------------- #
# the engine, against the measured Kev shape
# --------------------------------------------------------------------------- #


@respx.mock
async def test_decide_round_trip_preserves_the_provider_answers() -> None:
    route = respx.post(f"{BASE}/v1/systemone").mock(
        return_value=httpx.Response(200, json=_kev_body())
    )
    engine = _engine()

    result = await engine.decide(_request())

    sent = json.loads(route.calls[0].request.read())
    assert set(sent) == {"model", "state", "questions"}
    assert sent["model"] == "kev-tiny"
    assert set(sent["questions"]) == {"refunded", "route", "urgency"}

    assert result.answers["refunded"].noul == 0.98
    assert result.answers["route"].choice == "billing"
    assert result.answers["route"].probabilities["billing"] == 0.84
    # Preserved exactly as reported: provider calibration is not ours.
    assert result.answers["route"].confidence == 0.76
    assert result.answers["urgency"].score == 0.75
    assert result.answers["urgency"].legend == {"0": "low", "1": "medium", "2": "high"}
    assert result.usage is not None
    assert result.usage.promptTokens == 89
    assert result.usage.completionTokens == 156
    assert result.reportedModel == "kev-tiny"
    assert result.backend.value == "systemone_http"


@respx.mock
async def test_translating_engine_reports_public_and_sends_upstream() -> None:
    route = respx.post(f"{BASE}/v1/systemone").mock(
        return_value=httpx.Response(200, json=_kev_body(model="kev-latest"))
    )
    engine = _engine(model_id="decisions", upstream_model_id="kev-latest")

    result = await engine.decide(_request())

    sent = json.loads(route.calls[0].request.read())
    assert sent["model"] == "kev-latest"
    assert result.modelId == "decisions"
    assert result.reportedModel == "kev-latest"


@respx.mock
async def test_absent_usage_stays_absent() -> None:
    body = _kev_body()
    del body["usage"]
    respx.post(f"{BASE}/v1/systemone").mock(return_value=httpx.Response(200, json=body))

    result = await _engine().decide(_request())
    # Unknown accounting is unknown, never zero: a metering caller must
    # not be told a decision was free.
    assert result.usage is None


# --------------------------------------------------------------------------- #
# malformed backend output is a backend error
# --------------------------------------------------------------------------- #


async def _expect_malformed(body: dict, fragment: str) -> None:
    respx.post(f"{BASE}/v1/systemone").mock(return_value=httpx.Response(200, json=body))
    try:
        await _engine().decide(_request())
    except CliError as e:
        assert fragment in str(e), str(e)
        return
    raise AssertionError(f"malformed answer was accepted (wanted {fragment!r})")


@respx.mock
async def test_a_missing_answer_is_a_backend_error() -> None:
    body = _kev_body()
    del body["answers"]["urgency"]
    await _expect_malformed(body, "unanswered")


@respx.mock
async def test_an_uninvited_answer_is_a_backend_error() -> None:
    body = _kev_body()
    body["answers"]["extra"] = {"type": "noul", "noul": 0.5}
    await _expect_malformed(body, "nobody asked")


@respx.mock
async def test_an_illegal_choice_is_a_backend_error() -> None:
    body = _kev_body()
    body["answers"]["route"]["choice"] = "refunds"
    await _expect_malformed(body, "not one of the request's options")


@respx.mock
async def test_a_distribution_that_is_not_one_is_a_backend_error() -> None:
    body = _kev_body()
    body["answers"]["route"]["probabilities"] = {"billing": 0.5, "shipping": 0.1}
    await _expect_malformed(body, "sum to")


@respx.mock
async def test_a_non_finite_number_is_a_backend_error() -> None:
    body = _kev_body()
    body["answers"]["refunded"]["noul"] = 1.7
    await _expect_malformed(body, "finite probability")


@respx.mock
async def test_a_score_outside_the_scale_is_a_backend_error() -> None:
    body = _kev_body()
    body["answers"]["urgency"]["score"] = 5.0
    await _expect_malformed(body, "3-level scale")


@respx.mock
async def test_a_wrong_answer_type_is_a_backend_error() -> None:
    body = _kev_body()
    body["answers"]["refunded"] = {"type": "choice", "choice": "yes", "probabilities": {"yes": 1}}
    await _expect_malformed(body, "does not match")


@respx.mock
async def test_probabilities_naming_unoffered_options_are_a_backend_error() -> None:
    body = _kev_body()
    body["answers"]["route"]["probabilities"]["refunds"] = 0.0
    await _expect_malformed(body, "never offered")


# --------------------------------------------------------------------------- #
# the surfaces this backend does not serve
# --------------------------------------------------------------------------- #


def _decision_app_client(tmp_path, monkeypatch) -> TestClient:
    config = tmp_path / "driver.yaml"
    config.write_text(
        "provider: systemone_custom\n"
        f"baseUrl: {BASE}\n"
        "modelId: decisions\n"
        "backendLocality: local\n"
        "decisionMaxConcurrent: 1\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("EUGENE_PLEXUS_DRIVER_CONFIG_FILE", str(config))
    from eugene_plexus_inference_driver.app import create_app

    return TestClient(create_app())


def test_a_decision_driver_refuses_chat_naming_the_door(tmp_path, monkeypatch) -> None:
    with _decision_app_client(tmp_path, monkeypatch) as client:
        response = client.post(
            "/v1/generate",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 400
        assert "/v1/systemone" in response.text

        stream = client.post(
            "/v1/generate/stream",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert stream.status_code == 400


def test_info_advertises_the_decision_capability(tmp_path, monkeypatch) -> None:
    with _decision_app_client(tmp_path, monkeypatch) as client:
        info = client.get("/v1/info").json()
    assert info["capabilities"]["chatCapable"] is False
    assert info["capabilities"]["decision"]["kinds"] == ["noul", "choice", "score"]
    assert info["capabilities"]["decision"]["maxConcurrent"] == 1
    assert info["modelId"] == "decisions"


@respx.mock
def test_decide_route_refuses_unknown_fields_from_the_raw_body(tmp_path, monkeypatch) -> None:
    """Pydantic sheds unknown fields before a handler sees them, so the
    refusal must read the RAW body — this test is the proof it does."""
    with _decision_app_client(tmp_path, monkeypatch) as client:
        response = client.post(
            "/v1/decide",
            json={
                "state": "x",
                "questions": {
                    "q": {"type": "noul", "instructions": "?", "top_p": 0.9},
                },
            },
        )
        assert response.status_code == 400
        assert "top_p" in response.text


def test_a_chat_driver_refuses_decide_naming_the_gap(tmp_path, monkeypatch) -> None:
    config = tmp_path / "driver.yaml"
    config.write_text(
        f"provider: openai_compat_custom\nbaseUrl: {BASE}\nmodelId: qwen\nbackendLocality: local\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("EUGENE_PLEXUS_DRIVER_CONFIG_FILE", str(config))
    from eugene_plexus_inference_driver.app import create_app

    with TestClient(create_app()) as client:
        response = client.post(
            "/v1/decide",
            json={"state": "x", "questions": {"q": {"type": "noul", "instructions": "?"}}},
        )
    assert response.status_code == 400
    assert "does not answer typed decisions" in response.text


@respx.mock
def test_decide_round_trip_through_the_route(tmp_path, monkeypatch) -> None:
    respx.post(f"{BASE}/v1/systemone").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "decisions",
                "answers": {"q": {"type": "noul", "noul": 0.9}},
                "usage": {"input_tokens": 5, "output_tokens": 9},
                "latency_ms": 12.0,
            },
        )
    )
    with _decision_app_client(tmp_path, monkeypatch) as client:
        response = client.post(
            "/v1/decide",
            json={"state": "x", "questions": {"q": {"type": "noul", "instructions": "?"}}},
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answers"]["q"]["noul"] == 0.9
    assert body["modelId"] == "decisions"


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #


def test_the_hosted_provider_requires_a_key() -> None:
    try:
        SystemOneHttpEngine(base_url="https://api.typesafe.ai", auth_required=True)
    except CliError as e:
        assert "API key" in str(e)
        return
    raise AssertionError("hosted engine constructed with no key")


def test_from_config_with_no_backend_says_both_fixes() -> None:
    try:
        SystemOneHttpEngine.from_config({}.get, default_base_url=None)
    except CliError as e:
        assert "runtimeName" in str(e) and "baseUrl" in str(e)
        return
    raise AssertionError("engine constructed with no backend")


@respx.mock
async def test_list_models_reads_the_kev_shape() -> None:
    respx.get(f"{BASE}/v1/models").mock(
        return_value=httpx.Response(
            200,
            json={"models": [{"id": "kev-latest", "aliases": ["jev-latest"], "device": "cpu"}]},
        )
    )
    assert await _engine().list_models() == ["kev-latest"]
