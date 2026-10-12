"""`x-eugene-plexus-ignored-settings` on `POST /v1/generate` and its stream.

Troy, 2026-10-12: a setting the engine is measured not to honour is sent anyway and
NAMED, not refused; a refusal is for what cannot be sent at all. The engine says
what it ignores (an optional `ignored_settings(request)` hook, as `probe_embeddings`
is optional), in the contract's spelling (`toolChoice`), and the route returns it
as a comma-separated header on the 200 -- before the first frame of a stream --
and not at all when there is nothing to name. Strata is the first engine to use it
(`test_strata_tools.py`); these tests are the mechanism itself, with an engine that
names whatever it is told to. **What an engine names is only what the caller sent**: a
direct caller gets the header only for the settings it lists in `callerSettings` (an engine's
own tests: `test_strata_tools.py`, `test_openrouter_tool_gaps.py`).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_inference_driver._generated.models import (
    BackendKind,
    FinishReason,
    GenerateRequest,
    GenerateResponse,
)
from eugene_plexus_inference_driver.engines._subprocess import CliError
from eugene_plexus_inference_driver.engines.base import Chunk
from tests.test_generate_route import _events

HEADER = "x-eugene-plexus-ignored-settings"
CHAT = {"messages": [{"role": "user", "content": "ping"}]}


def _answer() -> GenerateResponse:
    return GenerateResponse(
        content="pong",
        finishReason=FinishReason.stop,
        backend=BackendKind.openai_compat_http,
        modelId="m",
        latencyMs=1,
    )


class _Naming:
    """An engine that names whatever it is given, and records what it was asked about."""

    backend_kind = BackendKind.openai_compat_http
    supports_tool_calling = True
    supports_streaming = True

    def __init__(self, names: list[str], *, fail_after_first_token: bool = False) -> None:
        self._names = names
        self._fail = fail_after_first_token
        self.asked: list[GenerateRequest] = []

    def ignored_settings(self, request: GenerateRequest) -> list[str]:
        self.asked.append(request)
        return list(self._names)

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        return _answer()

    async def stream(self, request: GenerateRequest) -> AsyncIterator[Chunk]:
        yield Chunk(text="po")
        if self._fail:
            raise CliError("the backend went away after its first token")
        yield Chunk(text="ng")
        yield Chunk(done=True, result=_answer())


class _Silent:
    """An engine from before the mechanism: no `ignored_settings` at all."""

    backend_kind = BackendKind.openai_compat_http
    supports_tool_calling = True
    supports_streaming = True

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        return _answer()

    async def stream(self, request: GenerateRequest) -> AsyncIterator[Chunk]:
        yield Chunk(text="pong")
        yield Chunk(done=True, result=_answer())


@pytest.mark.parametrize("path", ["/v1/generate", "/v1/generate/stream"])
def test_the_engine_names_what_its_backend_ignores_on_both_endpoints(
    client: TestClient, path: str
) -> None:
    client.app.state.adapter = _Naming(["toolChoice"])  # type: ignore[attr-defined]
    response = client.post(path, json=CHAT)
    assert response.status_code == 200, response.text
    assert response.headers[HEADER] == "toolChoice"


@pytest.mark.parametrize("path", ["/v1/generate", "/v1/generate/stream"])
def test_several_are_comma_separated_in_the_order_given_without_repeats(
    client: TestClient, path: str
) -> None:
    client.app.state.adapter = _Naming(["toolChoice", "topK", "toolChoice"])  # type: ignore[attr-defined]
    assert client.post(path, json=CHAT).headers[HEADER] == "toolChoice, topK"


@pytest.mark.parametrize("path", ["/v1/generate", "/v1/generate/stream"])
@pytest.mark.parametrize("engine", [_Naming([]), _Silent()], ids=["names-nothing", "no-hook"])
def test_nothing_to_name_is_no_header(client: TestClient, path: str, engine: Any) -> None:
    client.app.state.adapter = engine  # type: ignore[attr-defined]
    response = client.post(path, json=CHAT)
    assert response.status_code == 200, response.text
    assert HEADER not in response.headers


def test_a_stream_has_the_header_on_its_response_whatever_follows(client: TestClient) -> None:
    """It is known before the first frame, so a stream that fails after one still carries it."""
    client.app.state.adapter = _Naming(["toolChoice"], fail_after_first_token=True)  # type: ignore[attr-defined]
    response = client.post("/v1/generate/stream", json=CHAT)
    assert response.status_code == 200
    assert response.headers[HEADER] == "toolChoice"
    assert [name for name, _ in _events(response.text)] == ["token", "error"]


def test_the_engine_is_asked_about_the_request_it_will_serve(client: TestClient) -> None:
    """Which values of a setting are ignored is the engine's to say, so it sees the request."""
    engine = _Naming(["toolChoice"])
    client.app.state.adapter = engine  # type: ignore[attr-defined]
    client.post("/v1/generate", json={**CHAT, "toolChoice": "required"})
    [asked] = engine.asked
    assert str(asked.toolChoice.value) == "required"  # type: ignore[union-attr]


def test_a_request_the_driver_refuses_carries_no_header(client: TestClient) -> None:
    """Named on the 200 only: a refusal sent nothing, so nothing was ignored."""
    engine = _Naming(["toolChoice"])
    engine.supports_tool_calling = False
    client.app.state.adapter = engine  # type: ignore[attr-defined]
    tool = {"type": "function", "function": {"name": "t", "parameters": {"type": "object"}}}
    response = client.post("/v1/generate", json={**CHAT, "tools": [tool]})
    assert response.status_code == 400
    assert HEADER not in response.headers
    assert json.loads(response.text)["detail"]["type"].endswith("#tools-unsupported")
    assert engine.asked == []
