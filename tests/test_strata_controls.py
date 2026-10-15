"""Native thinking controls and stop strings missing from Strata's HTTP implementation."""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator

import httpx
import pytest
import respx

from eugene_plexus_inference_driver._generated.models import GenerateRequest
from eugene_plexus_inference_driver.engines.strata_http import StrataHttpEngine

BASE = "http://strata.test"


def request(**extra) -> GenerateRequest:
    return GenerateRequest.model_validate(
        {"messages": [{"role": "user", "content": "hi"}], **extra}
    )


def frame(delta, finish=None) -> bytes:
    return (
        "data: " + json.dumps({"choices": [{"delta": delta, "finish_reason": finish}]}) + "\n\n"
    ).encode()


class TrackedStream(httpx.AsyncByteStream):
    def __init__(self, frames: list[bytes]) -> None:
        self.frames = frames
        self.closed = False
        self.read = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for value in self.frames:
            self.read += 1
            yield value

    async def aclose(self) -> None:
        self.closed = True


async def answer(engine, ask, stream):
    if stream:
        chunks = [chunk async for chunk in engine.stream(ask)]
        result = chunks[-1].result
        assert "".join(chunk.text for chunk in chunks) == (result.content or "")
        return result
    return await engine.generate(ask)


@pytest.mark.anyio
@respx.mock
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "pieces,expected,matched",
    [
        (["hello ST", "OP hidden", "still hidden"], "hello ", "STOP"),
        (["hello ST", "UFF"], "hello STUFF", None),
        (["ST", "OP"], "", "STOP"),
        (["hello ST"], "hello ST", None),
        (["hello END and STOP"], "hello ", "END"),
        (["こんにちは S", "TOP 隠す"], "こんにちは ", "STOP"),
    ],
)
async def test_stop_strings_across_chunks_cancel_without_leaking_tail(
    stream, pieces, expected, matched
):
    upstream = TrackedStream(
        [*(frame({"content": p}) for p in pieces), frame({}, "stop"), b"data: [DONE]\n\n"]
    )
    respx.post(f"{BASE}/v1/chat/completions").respond(
        200, stream=upstream, headers={"content-type": "text/event-stream"}
    )
    engine = StrataHttpEngine(base_url=BASE, model_id="qwen", auth_required=False)
    try:
        result = await answer(
            engine, request(stop=["STOP", "END"], callerSettings=["stop"]), stream
        )
        assert result.content == expected
        assert result.stopSequence == matched
        assert upstream.closed
        if matched:
            assert upstream.read <= len(pieces)
            assert result.usage is None
            assert result.finishReason == "stop"
    finally:
        await engine.aclose()


@pytest.mark.anyio
@respx.mock
@pytest.mark.parametrize("stream", [False, True])
async def test_stop_does_not_modify_tool_arguments_or_reasoning(stream):
    upstream = TrackedStream(
        [
            frame({"reasoning_content": "STOP is a word"}),
            frame(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"query":"STOP"}'},
                        }
                    ]
                }
            ),
            frame({}, "tool_calls"),
            b"data: [DONE]\n\n",
        ]
    )
    respx.post(f"{BASE}/v1/chat/completions").respond(
        200, stream=upstream, headers={"content-type": "text/event-stream"}
    )
    engine = StrataHttpEngine(base_url=BASE, model_id="qwen", auth_required=False)
    try:
        result = await answer(engine, request(stop=["STOP"]), stream)
        assert result.reasoning == "STOP is a word"
        assert result.toolCalls[0].function.arguments == '{"query":"STOP"}'
        assert result.finishReason == "tool_calls" and result.stopSequence is None
    finally:
        await engine.aclose()


@pytest.mark.anyio
@respx.mock
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "mode,effort,expected",
    [
        ("off", None, "none"),
        ("low", None, "low"),
        ("medium", None, "medium"),
        ("high", None, "high"),
        ("auto", None, None),
        ("off", "high", "high"),
        ("high", "none", "none"),
        ("auto", "minimal", "minimal"),
    ],
)
async def test_native_thinking_defaults_and_request_override(mode, effort, expected, stream):
    route = respx.post(f"{BASE}/v1/chat/completions").respond(
        200,
        json={
            "choices": [
                {
                    "message": {"content": "hello", "reasoning_content": "thought"},
                    "finish_reason": "stop",
                }
            ]
        },
    )
    if stream:
        route.respond(
            200,
            stream=TrackedStream(
                [
                    frame({"reasoning_content": "thought"}),
                    frame({"content": "hello"}),
                    frame({}, "stop"),
                    b"data: [DONE]\n\n",
                ]
            ),
            headers={"content-type": "text/event-stream"},
        )
    engine = StrataHttpEngine(
        base_url=BASE, model_id="qwen", auth_required=False, thinking_mode=mode
    )
    try:
        result = await answer(engine, request(reasoningEffort=effort), stream)
        payload = json.loads(route.calls[-1].request.content)
        assert payload.get("reasoning_effort") == expected
        assert payload["messages"] == [{"role": "user", "content": "hi"}]
        assert result.reasoning == (None if expected in {"none", "minimal"} else "thought")
    finally:
        await engine.aclose()


@pytest.mark.anyio
@respx.mock
async def test_abandoning_a_stop_filtered_stream_closes_upstream():
    upstream = TrackedStream([frame({"content": "first"}), frame({"content": "more"})])
    respx.post(f"{BASE}/v1/chat/completions").respond(
        200, stream=upstream, headers={"content-type": "text/event-stream"}
    )
    engine = StrataHttpEngine(base_url=BASE, model_id="qwen", auth_required=False)
    try:
        async with contextlib.aclosing(engine.stream(request(stop=["END"]))) as chunks:
            assert (await anext(chunks)).text == "first"
        assert upstream.closed and upstream.read == 1
    finally:
        await engine.aclose()
