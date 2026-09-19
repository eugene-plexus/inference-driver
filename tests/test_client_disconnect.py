"""R2.5 — the driver keeps computing for a client that has gone.

The reproduction, written before the fix (roadmap §1). The gateway is a
client like any other: when *it* gives up — its own deadline, or its own
caller's closed tab — nothing here notices, and the engine holds the GPU
for the rest of the answer.

Three sites, and the third is the one reasoning alone gets wrong.
`StreamingResponse` already races the body iterator against
`http.disconnect`, so a stream in flight is cancelled by Starlette —
**but the route awaits the first chunk BEFORE handing the generator
over**, deliberately, so a failure that early can still be a status
code. On a cold `llama-server` that first await is the whole prefill,
which is exactly the window where a client gives up.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi import FastAPI, HTTPException
from starlette.requests import Request

from eugene_plexus_inference_driver._generated.models import (
    EmbedRequest,
    FinishReason,
    GenerateRequest,
    GenerateResponse,
)
from eugene_plexus_inference_driver.disconnect import _watch
from eugene_plexus_inference_driver.engines.base import Chunk
from eugene_plexus_inference_driver.routes.generate import embed, generate, generate_stream


class _HangingAdapter:
    """An engine that never answers, and remembers how it ended."""

    backend_kind = "openai_compat_http"
    supports_tool_calling = True
    supports_embeddings = True

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.cancelled = False
        self.finished = False

    async def _hang(self) -> None:
        self.entered.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        self.finished = True

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        await self._hang()
        raise AssertionError("unreachable")

    async def embed(self, texts: list[str]) -> Any:
        await self._hang()
        raise AssertionError("unreachable")

    async def probe_embeddings(self) -> bool:
        return True

    async def stream(self, request: GenerateRequest) -> AsyncIterator[Chunk]:
        await self._hang()
        yield Chunk(text="never")


_SCOPE: dict[str, Any] = {
    "type": "http",
    "asgi": {"version": "3.0", "spec_version": "2.3"},
    "http_version": "1.1",
    "method": "POST",
    "scheme": "http",
    "path": "/v1/generate",
    "raw_path": b"/v1/generate",
    "query_string": b"",
    "root_path": "",
    "headers": [],
    "client": ("127.0.0.1", 5555),
    "server": ("127.0.0.1", 8081),
}
"""The scope uvicorn builds, minus the app. `spec_version` 2.3 is what
uvicorn really advertises, and it decides which branch of Starlette's
`StreamingResponse` runs -- so it is part of the subject, not padding."""


def _request(app: FastAPI, gone: asyncio.Event) -> Request:
    sent_body = False

    async def receive() -> dict[str, Any]:
        # The shape uvicorn delivers: the body, then nothing until the
        # socket closes. FastAPI consumes the first message before the
        # route body parameter exists; the watcher blocks on the second.
        nonlocal sent_body
        if not sent_body:
            sent_body = True
            return {"type": "http.request", "body": b"{}", "more_body": False}
        await gone.wait()
        return {"type": "http.disconnect"}

    return Request(_SCOPE | {"app": app}, receive)


def _app(adapter: _HangingAdapter) -> FastAPI:
    app = FastAPI()
    app.state.adapter = adapter
    return app


async def _drive(adapter: _HangingAdapter, call: Any) -> None:
    app = _app(adapter)
    gone = asyncio.Event()
    request = _request(app, gone)
    route = asyncio.ensure_future(call(request))
    await asyncio.wait_for(adapter.entered.wait(), timeout=5)
    gone.set()
    # Short on purpose: see the gateway's twin of this helper. A
    # generous deadline masks a version that abandons the backend call
    # instead of cancelling it, because the deadline's own cancellation
    # does the cancelling for it and everything downstream then looks
    # right.
    started = asyncio.get_running_loop().time()
    try:
        await asyncio.wait_for(route, timeout=2)
    except asyncio.TimeoutError:  # noqa: UP041
        route.cancel()
        pytest.fail("the route kept the engine running for a caller that had gone")
    except HTTPException as e:
        # 499, and deliberately not 500: "the caller left" and "we
        # failed" must not read the same in an access log.
        assert e.status_code == 499, e.status_code
    except asyncio.CancelledError:
        return
    elapsed = asyncio.get_running_loop().time() - started
    assert elapsed < 1.5, f"the route took {elapsed:.1f}s to notice the caller had gone"


@pytest.mark.asyncio
async def test_a_gateway_that_gave_up_cancels_the_generation() -> None:
    adapter = _HangingAdapter()
    body = GenerateRequest(messages=[{"role": "user", "content": "hi"}])
    await _drive(adapter, lambda r: generate(r, body))
    assert adapter.cancelled, "the engine was left computing for nobody"
    assert not adapter.finished


@pytest.mark.asyncio
async def test_a_gateway_that_gave_up_cancels_the_embedding() -> None:
    adapter = _HangingAdapter()
    body = EmbedRequest(input=["hi"])
    await _drive(adapter, lambda r: embed(r, body))
    assert adapter.cancelled
    assert not adapter.finished


@pytest.mark.asyncio
async def test_a_disconnect_during_the_prefill_cancels_the_stream() -> None:
    """The window Starlette does not cover: before the generator is
    handed to `StreamingResponse`, which on a cold engine is the whole
    model load plus the prompt."""
    adapter = _HangingAdapter()
    body = GenerateRequest(messages=[{"role": "user", "content": "hi"}])
    await _drive(adapter, lambda r: generate_stream(r, body))
    assert adapter.cancelled
    assert not adapter.finished


@pytest.mark.asyncio
async def test_a_caller_that_stays_gets_its_answer() -> None:
    """The control: the watcher must not cancel a live request."""

    class _Fast(_HangingAdapter):
        async def generate(self, request: GenerateRequest) -> GenerateResponse:
            return GenerateResponse(
                content="the answer",
                finishReason=FinishReason.stop,
                backend="openai_compat_http",
                modelId="m",
                latencyMs=1,
            )

    adapter = _Fast()
    app = _app(adapter)
    gone = asyncio.Event()  # never set
    body = GenerateRequest(messages=[{"role": "user", "content": "hi"}])
    result = await asyncio.wait_for(generate(_request(app, gone), body), timeout=5)
    assert result.content == "the answer"


def test_the_watcher_does_not_hang_a_plain_request(client) -> None:  # type: ignore[no-untyped-def]
    """The trap this cost, pinned.

    The first version used `Request.is_disconnected()`, which peeks at
    the receive channel inside an already-cancelled `anyio.CancelScope`.
    That only behaves inside anyio's own task tree, and the watcher is a
    raw `asyncio.Task` -- so under `TestClient`, whose receive blocks
    until the response is complete, the peek never came back and the
    whole suite wedged on the first route test that reached it. Nothing
    asserted "an ordinary request still returns", so the failure looked
    like an infrastructure problem rather than like this change.
    """
    response = client.post(
        "/v1/generate",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code < 500, response.text


async def test_the_watcher_returns_only_for_a_DISCONNECT() -> None:
    """`_watch` must read the message type, not merely that a message came.

    The sabotage pass found this missing. Every other check here feeds a
    channel that goes quiet after the body and then delivers a
    disconnect, so a watcher that treated ANY message as "the client is
    gone" was never contradicted — the one message it would have
    misread never arrived. Here one does.
    """
    delivered: list[dict[str, Any]] = [
        {"type": "http.request", "body": b"{}", "more_body": True},
        {"type": "http.request", "body": b"", "more_body": False},
        {"type": "http.request", "body": b"", "more_body": False},
    ]
    seen = asyncio.Event()

    async def receive() -> dict[str, Any]:
        if delivered:
            message = delivered.pop(0)
            if not delivered:
                seen.set()
            return message
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")

    request = Request(_SCOPE | {"app": None}, receive)
    watcher = asyncio.ensure_future(_watch(request))
    await asyncio.wait_for(seen.wait(), timeout=5)
    await asyncio.sleep(0.05)
    assert not watcher.done(), "a message that is not a disconnect was read as one"
    watcher.cancel()


async def test_the_watcher_does_return_for_a_disconnect() -> None:
    """The twin, so the test above cannot pass by never returning."""

    async def receive() -> dict[str, Any]:
        return {"type": "http.disconnect"}

    request = Request(_SCOPE | {"app": None}, receive)
    await asyncio.wait_for(_watch(request), timeout=5)
