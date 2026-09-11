"""Tests for POST /v1/generate via FastAPI test client.

Patches the adapter at the FastAPI app-state boundary so we can exercise
the route without needing real CLIs installed on the test machine.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi.testclient import TestClient

from eugene_plexus_inference_driver._generated.models import (
    BackendKind,
    FinishReason,
    GenerateRequest,
    GenerateResponse,
)
from eugene_plexus_inference_driver.engines._subprocess import CliError
from eugene_plexus_inference_driver.engines.base import Chunk


class _StaticAdapter:
    backend_kind = "claude_code_cli"

    def __init__(
        self,
        response: GenerateResponse | None = None,
        error: Exception | None = None,
        chunks: tuple[str, ...] = ("Hel", "lo"),
    ):
        self._response = response
        self._error = error
        self._chunks = chunks

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response

    async def stream(self, request: GenerateRequest) -> AsyncIterator[object]:
        if self._error is not None:
            raise self._error
        assert self._response is not None
        for piece in self._chunks:
            yield Chunk(text=piece)
        yield Chunk(done=True, result=self._response)


class _MidStreamFailure:
    """Emits a token, then breaks. The case the route must turn into an
    `event: error` frame rather than a status code -- by then the 200 is
    already on the wire."""

    backend_kind = "claude_code_cli"

    async def generate(self, request: GenerateRequest) -> GenerateResponse:  # pragma: no cover
        raise AssertionError("not used")

    async def stream(self, request: GenerateRequest) -> AsyncIterator[object]:
        yield Chunk(text="partial")
        raise CliError("backend died mid-answer")


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


def _events(raw: str) -> list[tuple[str, str]]:
    """Parse the SSE body into `(event, data)` pairs.

    Walks lines and flushes on a blank one, rather than splitting on a
    doubled newline, so it does not care whether the transport used
    LF or CRLF.
    """
    out: list[tuple[str, str]] = []
    name: str | None = None
    data: str | None = None
    for line in raw.splitlines():
        if not line.strip():
            if name and data is not None:
                out.append((name, data))
            name = data = None
            continue
        if line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:"):
            data = line[5:].strip()
    if name and data is not None:
        out.append((name, data))
    return out


def test_generate_stream_emits_token_events_then_done(client: TestClient) -> None:
    """The contract in `inference-driver.yaml`: `event: token` carrying
    `{"text": ...}`, terminated by `event: done` carrying the whole
    `GenerateResponse`. This endpoint was a 501 stub from M0 to M9."""
    fake = GenerateResponse(
        content="Hello",
        finishReason=FinishReason.stop,
        backend=BackendKind.claude_code_cli,
        modelId="m",
        latencyMs=5,
    )
    client.app.state.adapter = _StaticAdapter(response=fake)  # type: ignore[attr-defined]
    response = client.post(
        "/v1/generate/stream",
        json={"messages": [{"role": "user", "content": "ping"}]},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _events(response.text)
    assert [name for name, _ in events] == ["token", "token", "done"]
    assert [json.loads(data)["text"] for name, data in events if name == "token"] == ["Hel", "lo"]
    final = json.loads(events[-1][1])
    assert final["content"] == "Hello"


def test_a_backend_that_fails_before_the_first_token_is_still_a_status_code(
    client: TestClient,
) -> None:
    """Nothing has been sent yet, so the caller can still be told
    properly. Only once the stream is open does an error have to become
    a frame."""
    client.app.state.adapter = _StaticAdapter(error=CliError("nope"))  # type: ignore[attr-defined]
    response = client.post(
        "/v1/generate/stream",
        json={"messages": [{"role": "user", "content": "ping"}]},
    )
    assert response.status_code == 502


def test_a_backend_that_fails_after_the_first_token_becomes_an_error_frame(
    client: TestClient,
) -> None:
    """The 200 is already on the wire, so there is no status left to
    send. The partial answer is kept -- truncating is honest, discarding
    what the user already saw is not."""
    client.app.state.adapter = _MidStreamFailure()  # type: ignore[attr-defined]
    response = client.post(
        "/v1/generate/stream",
        json={"messages": [{"role": "user", "content": "ping"}]},
    )
    assert response.status_code == 200
    events = _events(response.text)
    assert [name for name, _ in events] == ["token", "error"]
    assert json.loads(events[0][1])["text"] == "partial"
    problem = json.loads(events[1][1])
    assert problem["status"] == 502
    assert "died mid-answer" in problem["detail"]


def test_stream_on_a_degraded_driver_is_503(client: TestClient) -> None:
    client.app.state.adapter = None  # type: ignore[attr-defined]
    response = client.post(
        "/v1/generate/stream",
        json={"messages": [{"role": "user", "content": "ping"}]},
    )
    assert response.status_code == 503
