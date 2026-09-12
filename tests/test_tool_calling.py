"""Tool calls, end to end through the OpenAI-compatible engine.

Install-paths §9 step 6. Before this, `ChatCompletionRequest` had nine
fields and none was `tools`, the driver's `_FINISH_REASON_MAP` flattened
a backend's `tool_calls` into a plain `stop`, and the non-streaming
parser *raised* on `content: null` — which is the shape of every
tool-call-only turn. So the failure was not "tools work badly": a
backend that made a tool call reported a clean natural stop with the
calls discarded, or a 502.

Upstream HTTP is mocked with `respx`, as the neighbouring adapter tests
do. What is under test is our translation in both directions, which is
the only part we own: we do not execute tools and we do not parse the
arguments string.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from eugene_plexus_inference_driver._generated.models import (
    FinishReason,
    FunctionCall,
    FunctionDefinition,
    GenerateRequest,
    Message,
    Role,
    Tool,
    ToolCall,
)
from eugene_plexus_inference_driver.engines.openai_compat_http import OpenAiCompatibleHttpEngine

pytestmark = pytest.mark.anyio

BASE = "http://127.0.0.1:11434"

WEATHER_TOOL = Tool(
    type="function",
    function=FunctionDefinition(
        name="get_weather",
        description="Look up the current weather for a place.",
        parameters={
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    ),
)


def _engine(**kwargs: Any) -> OpenAiCompatibleHttpEngine:
    return OpenAiCompatibleHttpEngine(
        base_url=BASE,
        model_id="qwen3-coder:30b",
        auth_required=False,
        **kwargs,
    )


def _ask(**kwargs: Any) -> GenerateRequest:
    return GenerateRequest(
        messages=[Message(role=Role.user, content="What is the weather in Oslo?")],
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# down: definitions reach the backend untouched
# --------------------------------------------------------------------------- #


@respx.mock
async def test_tools_reach_the_backend_verbatim() -> None:
    """We are a protocol adapter, so the JSON Schema goes down unaltered.

    Asserted against the body actually sent rather than against our own
    call, because "passed through" is a claim about the wire.
    """
    route = respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_plain_reply("no idea"))
    )
    await _engine().generate(_ask(tools=[WEATHER_TOOL], toolChoice="auto"))

    sent = json.loads(route.calls[0].request.content)
    assert sent["tool_choice"] == "auto"
    assert sent["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Look up the current weather for a place.",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
    ]


@respx.mock
async def test_no_tool_keys_when_the_caller_sent_none() -> None:
    """The overwhelmingly common request must be byte-identical to before.

    A `tools: null` or `tool_choice: none` we invented would be a
    behaviour change for every non-agent caller, and some backends treat
    the presence of the key as meaningful.
    """
    route = respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_plain_reply("hello"))
    )
    await _engine().generate(_ask())

    sent = json.loads(route.calls[0].request.content)
    assert "tools" not in sent
    assert "tool_choice" not in sent
    assert "response_format" not in sent


@respx.mock
async def test_a_tool_result_goes_up_as_a_tool_message() -> None:
    """The half an agent loop dies without.

    The harness replays the assistant turn that *asked* for the call plus
    one `tool` message per result. Until `Role.tool` existed, the final
    branch of `_to_openai_messages` coerced every unrecognised role to
    `user` — so a tool result arrived looking like the human talking,
    with the assistant's own `tool_calls` dropped. That transcript is
    plausible enough to read fine and wrong enough that the model never
    learns its call was answered.
    """
    route = respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_plain_reply("It is 4C in Oslo."))
    )
    await _engine().generate(
        GenerateRequest(
            messages=[
                Message(role=Role.user, content="What is the weather in Oslo?"),
                Message(
                    role=Role.assistant,
                    content=None,
                    toolCalls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "Oslo"}',
                            },
                        }
                    ],
                ),
                Message(role=Role.tool, content='{"tempC": 4}', toolCallId="call_1"),
            ],
            tools=[WEATHER_TOOL],
        )
    )

    sent = json.loads(route.calls[0].request.content)
    assistant, tool = sent["messages"][1], sent["messages"][2]
    assert assistant["role"] == "assistant"
    assert assistant["content"] is None
    assert assistant["tool_calls"][0]["function"]["name"] == "get_weather"
    assert tool == {
        "role": "tool",
        "content": '{"tempC": 4}',
        "tool_call_id": "call_1",
    }


# --------------------------------------------------------------------------- #
# up: calls come back, and a null content is not an error
# --------------------------------------------------------------------------- #


@respx.mock
async def test_a_tool_call_comes_back_with_null_content() -> None:
    """The exact response that used to raise.

    `content` is null on a tool-call-only turn, and the parser required a
    string. Two things are asserted: the calls survive, and the finish
    reason is `tool_calls` rather than the `stop` the map used to return
    — a caller that trusted `stop` would end the loop right where it
    should have dispatched.
    """
    respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1,
                "model": "qwen3-coder:30b",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_abc",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"location": "Oslo"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        )
    )
    reply = await _engine().generate(_ask(tools=[WEATHER_TOOL]))

    assert reply.finishReason == FinishReason.tool_calls
    assert reply.content is None
    assert reply.toolCalls is not None
    call = reply.toolCalls[0]
    assert call.id == "call_abc"
    assert call.function.name == "get_weather"
    # A string, unparsed. A model can emit invalid JSON and OpenAI's
    # contract preserves what it said; parsing here would move that
    # failure to the layer least able to report it usefully.
    assert call.function.arguments == '{"location": "Oslo"}'


@respx.mock
async def test_a_reply_with_neither_text_nor_calls_is_still_an_error() -> None:
    """Nullable content is not "anything goes".

    The old check was load bearing for a real malfunction — a backend
    answering with no message at all — and relaxing it for tool calls
    must not relax it for that.
    """
    respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1,
                "model": "qwen3-coder:30b",
                "choices": [
                    {"index": 0, "message": {"role": "assistant"}, "finish_reason": "stop"}
                ],
            },
        )
    )
    with pytest.raises(Exception, match="missing string content"):
        await _engine().generate(_ask())


# --------------------------------------------------------------------------- #
# streaming: fragments, accumulated by index
# --------------------------------------------------------------------------- #


@respx.mock
async def test_streamed_tool_call_fragments_accumulate_by_index() -> None:
    """`arguments` arrives split at arbitrary points, and two calls interleave.

    Both properties are real and both break a naive reader: no single
    fragment is parseable JSON, and arrival order is not call order —
    which is why `index` is the key rather than a counter.
    """
    frames = [
        _delta({"role": "assistant"}),
        _delta({"tool_calls": [_frag(0, id="call_a", name="get_weather")]}),
        _delta({"tool_calls": [_frag(1, id="call_b", name="get_time")]}),
        _delta({"tool_calls": [_frag(0, args='{"loc')]}),
        _delta({"tool_calls": [_frag(1, args='{"tz": "UTC"}')]}),
        _delta({"tool_calls": [_frag(0, args='ation": "Oslo"}')]}),
        _delta({}, finish_reason="tool_calls"),
    ]
    body = "".join(f"data: {json.dumps(f)}\n\n" for f in frames) + "data: [DONE]\n\n"
    respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
    )

    forwarded: list[Any] = []
    final = None
    async for chunk in _engine().stream(_ask(tools=[WEATHER_TOOL])):
        if chunk.done:
            final = chunk.result
        elif chunk.toolCalls:
            forwarded.append(chunk.toolCalls)

    # Forwarded as they land, not buffered to the end: a caller that
    # streams wants to start dispatching, and there were five such frames.
    assert len(forwarded) == 5

    assert final is not None
    assert final.finishReason == FinishReason.tool_calls
    assert final.content is None
    assert final.toolCalls is not None
    by_name = {c.function.name: c.function.arguments for c in final.toolCalls}
    assert by_name == {
        "get_weather": '{"location": "Oslo"}',
        "get_time": '{"tz": "UTC"}',
    }
    assert [c.id for c in final.toolCalls] == ["call_a", "call_b"]


@respx.mock
async def test_a_fragment_set_with_no_name_is_dropped() -> None:
    """A call nothing can dispatch is worse than no call: a harness tries."""
    frames = [
        _delta({"tool_calls": [_frag(0, args='{"x": 1}')]}),
        _delta({}, finish_reason="tool_calls"),
    ]
    body = "".join(f"data: {json.dumps(f)}\n\n" for f in frames) + "data: [DONE]\n\n"
    respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
    )

    final = None
    async for chunk in _engine().stream(_ask()):
        if chunk.done:
            final = chunk.result
    assert final is not None
    assert final.toolCalls is None


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _plain_reply(text: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "qwen3-coder:30b",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
    }


def _delta(delta: dict[str, Any], finish_reason: str | None = None) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "qwen3-coder:30b",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _frag(
    index: int, *, id: str | None = None, name: str | None = None, args: str | None = None
) -> dict[str, Any]:
    out: dict[str, Any] = {"index": index}
    if id is not None:
        out["id"] = id
    fn: dict[str, Any] = {}
    if name is not None:
        fn["name"] = name
    if args is not None:
        fn["arguments"] = args
    if fn:
        out["function"] = fn
    return out


def test_tool_call_models_round_trip() -> None:
    """The generated shapes are the ones the wire needs."""
    call = ToolCall(
        id="call_1",
        type="function",
        function=FunctionCall(name="get_weather", arguments='{"location": "Oslo"}'),
    )
    assert call.model_dump(mode="json") == {
        "id": "call_1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"location": "Oslo"}'},
    }
