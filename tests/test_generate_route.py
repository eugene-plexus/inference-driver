"""Tests for POST /v1/generate via FastAPI test client.

Patches the adapter at the FastAPI app-state boundary so we can exercise
the route without needing real CLIs installed on the test machine.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi.testclient import TestClient

from eugene_plexus_inference_driver._generated.models import (
    BackendKind,
    FinishReason,
    GenerateRequest,
    GenerateResponse,
)
from eugene_plexus_inference_driver.engines._subprocess import CliError


class _StaticAdapter:
    backend_kind = "claude_code_cli"

    def __init__(self, response: GenerateResponse | None = None, error: Exception | None = None):
        self._response = response
        self._error = error

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response

    async def stream(self, request: GenerateRequest) -> AsyncIterator[object]:
        raise NotImplementedError
        yield  # pragma: no cover


def test_generate_returns_adapter_response(client: TestClient) -> None:
    fake = GenerateResponse(
        content="hello",
        finishReason=FinishReason.stop,
        backend=BackendKind.claude_code_cli,
        modelId="claude-opus-4-7",
        latencyMs=150,
    )
    client.app.state.adapter = _StaticAdapter(response=fake)  # type: ignore[attr-defined]

    response = client.post(
        "/v1/generate",
        json={"messages": [{"role": "user", "content": "ping"}]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["content"] == "hello"
    assert body["finishReason"] == "stop"
    assert body["backend"] == "claude_code_cli"
    assert body["modelId"] == "claude-opus-4-7"
    assert body["latencyMs"] == 150


def test_generate_maps_cli_error_to_502(client: TestClient) -> None:
    client.app.state.adapter = _StaticAdapter(error=CliError("something broke"))  # type: ignore[attr-defined]

    response = client.post(
        "/v1/generate",
        json={"messages": [{"role": "user", "content": "ping"}]},
    )

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert detail["status"] == 502
    assert detail["title"] == "Backend error"
    assert "something broke" in detail["detail"]
    assert detail["component"].startswith("inference-driver:")


def test_generate_stream_still_returns_501(client: TestClient) -> None:
    response = client.post(
        "/v1/generate/stream",
        json={"messages": [{"role": "user", "content": "ping"}]},
    )
    assert response.status_code == 501
