"""A backend that emits three tokens and goes silent is not still working.

R2.5 settled what a fired *request* deadline means; this is the case it
left open, and the one production gateways report most: the backend
starts answering and then stops -- no token, no finish reason, no close
-- on an open socket. Before this check the driver held that stream for
the full request budget (660 s), because httpx's read timeout IS the
request budget on this client.

Four properties, each pinned here:

1. Mid-answer silence longer than `streamStallSeconds` ends the stream
   as a `BackendTimeout` naming the knob -- within the stall window,
   not the request budget.
2. Silence BEFORE the first token is exempt: that is a model load plus
   prompt reading, minutes long on a CPU box, governed by
   `requestTimeoutSeconds`. The stall clock arms at the first frame.
3. A stall AFTER the finish_reason is a complete answer missing only
   its goodbye, and must not fail.
4. Zero disables the check and MUST arrive as zero -- `0 or DEFAULT`
   is DEFAULT, the seed=0 mistake (R3.4), and a truthiness read here
   would re-enable a check the operator turned off.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator

import httpx
import pytest
import respx
from fastapi import HTTPException

from eugene_plexus_inference_driver._generated.models import (
    BackendKind,
    GenerateRequest,
    Message,
    Role,
)
from eugene_plexus_inference_driver.engines._subprocess import BackendTimeout
from eugene_plexus_inference_driver.engines.base import DEFAULT_STREAM_STALL_SECONDS
from eugene_plexus_inference_driver.engines.openai_compat_http import (
    OpenAiCompatibleHttpEngine,
    _stalled,
)
from eugene_plexus_inference_driver.routes.generate import _backend_error

URL = "http://127.0.0.1:9/v1/chat/completions"


def _frame(payload: dict) -> bytes:
    return b"data: " + json.dumps(payload).encode() + b"\n\n"


TOKEN = _frame({"choices": [{"delta": {"content": "Hi"}}]})
FINISH = _frame({"choices": [{"delta": {}, "finish_reason": "stop"}]})
DONE = b"data: [DONE]\n\n"


class _PacedStream(httpx.AsyncByteStream):
    """SSE bytes delivered with real pauses between them.

    A stall cannot be mocked with `side_effect` -- the defect is the
    TIMING of a healthy-looking stream, so the fixture must actually
    go quiet on an open connection.
    """

    def __init__(self, parts: list[tuple[float, bytes]]) -> None:
        self._parts = parts

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for delay, chunk in self._parts:
            if delay:
                await asyncio.sleep(delay)
            if chunk:
                yield chunk

    async def aclose(self) -> None:
        return None


def _mock_stream(parts: list[tuple[float, bytes]]) -> None:
    respx.post(URL).mock(
        return_value=httpx.Response(
            200,
            stream=_PacedStream(parts),
            headers={"content-type": "text/event-stream"},
        )
    )


def _engine(stall_seconds: float) -> OpenAiCompatibleHttpEngine:
    return OpenAiCompatibleHttpEngine(
        base_url="http://127.0.0.1:9/",
        api_key=None,
        model_id="m",
        backend_kind=BackendKind.openai_compat_http,
        timeout_seconds=42.0,
        stall_seconds=stall_seconds,
        auth_required=False,
    )


def _request() -> GenerateRequest:
    return GenerateRequest(messages=[Message(role=Role.user, content="say PING")])


async def _collect(engine: OpenAiCompatibleHttpEngine) -> str:
    parts: list[str] = []
    async for chunk in engine.stream(_request()):
        if getattr(chunk, "text", None):
            parts.append(chunk.text)
    return "".join(parts)


@pytest.mark.asyncio
@respx.mock
async def test_a_mid_answer_stall_is_ended_within_the_stall_window() -> None:
    """One token, then silence. Before this check the caller waited out
    the whole 42 s request budget; now the stall clock answers."""
    _mock_stream([(0.0, TOKEN), (30.0, DONE)])
    started = time.perf_counter()
    with pytest.raises(BackendTimeout) as caught:
        await _collect(_engine(stall_seconds=0.15))
    elapsed = time.perf_counter() - started
    assert elapsed < 5.0, f"fired at {elapsed:.1f}s -- the request budget, not the stall clock"
    message = str(caught.value)
    assert "streamStallSeconds" in message, message
    assert "0.15" in message, message


@pytest.mark.asyncio
@respx.mock
async def test_silence_before_the_first_token_is_not_a_stall() -> None:
    """The cold-load exemption: a gap longer than the stall threshold is
    fine while nothing has been streamed yet."""
    _mock_stream([(0.5, TOKEN), (0.0, FINISH), (0.0, DONE)])
    text = await _collect(_engine(stall_seconds=0.15))
    assert text == "Hi"


@pytest.mark.asyncio
@respx.mock
async def test_a_stall_after_the_finish_reason_is_a_complete_answer() -> None:
    """finish_reason arrived, [DONE] never does. The answer is whole;
    failing it over a missing goodbye would be the over-correction."""
    _mock_stream([(0.0, TOKEN), (0.0, FINISH), (30.0, DONE)])
    text = await _collect(_engine(stall_seconds=0.15))
    assert text == "Hi"


@pytest.mark.asyncio
@respx.mock
async def test_zero_disables_the_check() -> None:
    _mock_stream([(0.0, TOKEN), (0.4, TOKEN), (0.0, FINISH), (0.0, DONE)])
    text = await _collect(_engine(stall_seconds=0.0))
    assert text == "HiHi"


@pytest.mark.asyncio
@respx.mock
async def test_the_zero_fixture_would_fail_with_the_check_on() -> None:
    """The disabled test proves nothing unless its gap actually trips
    an enabled check -- the fixture must be able to fail."""
    _mock_stream([(0.0, TOKEN), (0.4, TOKEN), (0.0, FINISH), (0.0, DONE)])
    with pytest.raises(BackendTimeout):
        await _collect(_engine(stall_seconds=0.15))


@pytest.mark.asyncio
@respx.mock
async def test_a_slow_consumer_never_trips_the_stall() -> None:
    """The timer measures the BACKEND's silence, not the caller's pace.
    It wraps only the upstream read, and between yields the consumer
    owns the clock -- so a browser that renders slowly cannot make a
    healthy backend read as stalled."""
    _mock_stream([(0.0, TOKEN), (0.0, TOKEN), (0.0, FINISH), (0.0, DONE)])
    engine = _engine(stall_seconds=0.15)
    parts: list[str] = []
    async for chunk in engine.stream(_request()):
        if getattr(chunk, "text", None):
            parts.append(chunk.text)
        await asyncio.sleep(0.3)  # slower than the stall threshold
    assert "".join(parts) == "HiHi"


def test_the_stall_answers_504_and_names_the_knob() -> None:
    """Mid-stream the gateway is committed (M10), so this surfaces as
    the terminal error frame with the 504 identity -- never a natural
    end, never a cascade-inviting 502."""
    problem = _backend_error(_stalled(30.0, 12), "openai_compat_http")
    assert isinstance(problem, HTTPException)
    assert problem.status_code == 504
    body = f"{problem.detail['title']} {problem.detail['detail']}"
    assert "streamStallSeconds" in body
    assert "12" in body


def test_the_schema_default_reads_the_one_constant() -> None:
    from eugene_plexus_inference_driver.config import FIELDS

    field = next(f for f in FIELDS if f.key == "streamStallSeconds")
    assert field.default == DEFAULT_STREAM_STALL_SECONDS
    assert field.minimum == 0, "0 must stay a legal value: it is the off switch"


def test_from_config_zero_is_zero_not_the_default() -> None:
    """`0 or DEFAULT` is DEFAULT -- the seed=0 mistake. An operator who
    turned the check off must get it off."""

    def build(stall: object) -> OpenAiCompatibleHttpEngine:
        values = {"streamStallSeconds": stall, "modelId": "m"}
        return OpenAiCompatibleHttpEngine.from_config(
            values.get,
            default_base_url="http://127.0.0.1:9/",
            fixed_temperature_pattern=None,
            backend_kind=BackendKind.openai_compat_http,
            auth_required=False,
        )

    assert build(0)._stall_seconds == 0.0
    assert build(None)._stall_seconds == DEFAULT_STREAM_STALL_SECONDS
    assert build(7)._stall_seconds == 7.0
