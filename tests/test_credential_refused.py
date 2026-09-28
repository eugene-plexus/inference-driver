"""A backend refusing THIS DRIVER's own credential is not the caller's 400.

Measured live on 2026-09-28 against OpenRouter with an invalid key: the
backend's 401 (`{"error":{"message":"User not found.","code":401}}`) left
here as a 400 `#backend-rejected-request`, which the gateway reports as
the caller's `invalid_request_error`. The caller cannot fix a key only
the operator holds, and the same request with the same key fails the
same way. It is a 502 `#backend-credential-refused` now, `terminal` so
nothing cascades (the 4xx non-cascade rule is older and unchanged).
"""

from __future__ import annotations

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from eugene_plexus_inference_driver.engines._subprocess import CliError
from eugene_plexus_inference_driver.failures import credential_refused, disposition
from tests.test_context_window import _Refusing
from tests.test_decisions import BASE, _decision_app_client

OPENROUTER_401 = {"error": {"message": "User not found.", "code": 401}}
CHAT = {"messages": [{"role": "user", "content": "x"}]}
DECIDE = {"state": "x", "questions": {"q": {"type": "noul", "instructions": "?"}}}


@pytest.mark.parametrize(
    "status,words,expected",
    [
        (401, "User not found.", 401),
        (402, "Insufficient credits", 402),
        (403, "This key lacks permission for this model", 403),
        # OpenRouter's moderation refusal is a 403 about the CONTENT: the caller's.
        (403, "Your chosen model requires moderation and your input was flagged", None),
        (403, "Request violates the content policy", None),
        (400, "prompt is 15010 tokens, n_ctx is 512", None),
        (404, "no such model", None),
        (429, "rate limited", None),
        (None, "connection refused", None),
    ],
)
def test_which_refusals_are_the_drivers_own_credential(status, words, expected) -> None:
    assert (
        credential_refused(CliError(f"backend returned {status}: {words}", upstream_status=status))
        == expected
    )


def test_the_classification_changes_what_we_say_not_what_we_do() -> None:
    """Every credential refusal stays `terminal`, as every 4xx was: a
    change to the message must not quietly widen the cascade."""
    for status in (401, 402, 403):
        assert disposition(CliError("x", upstream_status=status)) == "terminal"


def _refused(client: TestClient, status: int, words: str) -> dict:
    client.app.state.adapter = _Refusing(  # type: ignore[attr-defined]
        CliError(f"openai_compat_http returned {status}: {words}", upstream_status=status)
    )
    response = client.post("/v1/generate", json=CHAT)
    assert response.status_code == 502, response.text
    return response.json()["detail"]


def test_a_401_is_a_502_naming_the_cause_and_the_next_step(client: TestClient) -> None:
    detail = _refused(client, 401, '{"error":{"message":"Incorrect API key provided"}}')
    assert detail["type"].endswith("#backend-credential-refused")
    assert detail["retryDisposition"] == "terminal"
    assert "HTTP 401" in detail["detail"]
    assert "Incorrect API key provided" in detail["detail"]  # the provider's own words
    assert "Nothing is wrong with the request" in detail["detail"]
    assert "Set a working API key on this driver" in detail["detail"]


def test_a_402_says_the_account_has_no_credit(client: TestClient) -> None:
    detail = _refused(client, 402, "Insufficient credits")
    assert detail["type"].endswith("#backend-credential-refused")
    assert "no credit" in detail["detail"]
    assert "Add credit to that account" in detail["detail"]


def test_a_403_about_the_content_stays_the_callers_400(client: TestClient) -> None:
    """The pair that tells the fix from the over-correction."""
    client.app.state.adapter = _Refusing(  # type: ignore[attr-defined]
        CliError(
            "openai_compat_http returned 403: your input was flagged by moderation",
            upstream_status=403,
        )
    )
    response = client.post("/v1/generate", json=CHAT)
    assert response.status_code == 400
    assert response.json()["detail"]["type"].endswith("#backend-rejected-request")


def test_a_credential_refusal_keeps_the_backends_retry_hint(client: TestClient) -> None:
    client.app.state.adapter = _Refusing(  # type: ignore[attr-defined]
        CliError("returned 402: out of credit", upstream_status=402, retry_after_seconds=30)
    )
    response = client.post("/v1/generate", json=CHAT)
    assert response.status_code == 502
    assert response.headers["Retry-After"] == "30"


class _RefusingStream(_Refusing):
    supports_streaming = True
    model_id = "the-model"
    runtime = None

    def stream(self, request):  # an async generator that fails before its first frame
        async def gen():
            raise self._error
            yield  # pragma: no cover

        return gen()


def test_the_streaming_door_refuses_before_the_first_frame_the_same_way(client: TestClient) -> None:
    """Streaming is what most clients use; the refusal comes before any
    frame, so it can still be a status and must be this one."""
    client.app.state.adapter = _RefusingStream(  # type: ignore[attr-defined]
        CliError("openai_compat_http returned 401: invalid key", upstream_status=401)
    )
    response = client.post("/v1/generate/stream", json=CHAT)
    assert response.status_code == 502, response.text
    assert response.json()["detail"]["type"].endswith("#backend-credential-refused")


@respx.mock
def test_the_decision_door_with_the_wire_shape_measured_live(tmp_path, monkeypatch) -> None:
    """The reproduction, through the real engine and route: OpenRouter's
    exact 401 body for an invalid key, sent to POST /v1/decide."""
    respx.post(f"{BASE}/v1/systemone").mock(return_value=httpx.Response(401, json=OPENROUTER_401))
    with _decision_app_client(tmp_path, monkeypatch) as client:
        response = client.post("/v1/decide", json=DECIDE)
    assert response.status_code == 502, response.text
    detail = response.json()["detail"]
    assert detail["type"].endswith("#backend-credential-refused")
    assert detail["retryDisposition"] == "terminal"
    assert "User not found." in detail["detail"]
    assert detail["component"] == "inference-driver:systemone_http"


@respx.mock
def test_the_decision_door_still_passes_a_real_rejection_as_400(tmp_path, monkeypatch) -> None:
    respx.post(f"{BASE}/v1/systemone").mock(
        return_value=httpx.Response(422, json={"error": {"message": "questions.q.type: invalid"}})
    )
    with _decision_app_client(tmp_path, monkeypatch) as client:
        response = client.post("/v1/decide", json=DECIDE)
    assert response.status_code == 400
    assert response.json()["detail"]["type"].endswith("#backend-rejected-request")
