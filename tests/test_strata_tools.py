"""Strata takes tools, tool results and structured output (Troy, 2026-10-12).

Until that day the driver refused all three for Strata, because nobody had tried
them, and the gateway listed every Strata model without tools. Measured on
Strata v0.1.39 (`qwen3.8-flash-next-iq2_xs`, its own OpenAI endpoint;
`docs/design/tools-assumed.md` in specs): it calls tools correctly (one call,
two at once, streamed, and the round trip with the tool's result) and answers
`response_format` natively. What it does not do is read `tool_choice`: `none`
still called the tool, `required` answered "Hello!", and a named function called
another one. And `response_format` together with tools is its own 400.

The backend here is `respx`, answering in the shapes Strata sent, trimmed to
the fields the driver reads. Each test fails against the driver as it was:
every one of them was a 400 from the driver before any HTTP went out.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from eugene_plexus_inference_driver._generated.models import (
    FinishReason,
    FunctionDefinition,
    GenerateRequest,
    Message,
    NamedToolChoice,
    ResponseFormat,
    Role,
    Tool,
)
from eugene_plexus_inference_driver.app import create_app
from eugene_plexus_inference_driver.engines._subprocess import CliError
from eugene_plexus_inference_driver.engines.strata_http import StrataHttpEngine
from eugene_plexus_inference_driver.settings import Settings
from tests.test_generate_route import _events

BASE = "http://strata.local"
MODEL = "qwen3.8-flash-next-iq2_xs"
HEADER = "x-eugene-plexus-ignored-settings"
CHAT = f"{BASE}/v1/chat/completions"

WEATHER = Tool(
    type="function",
    function=FunctionDefinition(
        name="get_weather",
        description="Current weather for a city.",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    ),
)
TIME = Tool(
    type="function",
    function=FunctionDefinition(
        name="get_time",
        description="The current local time in a city.",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    ),
)
NAMED = {"type": "function", "function": {"name": "get_time"}}
SCHEMA_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}
#: Strata's own words (its `_openai`), measured: the one thing it refuses here.
STRATA_FORMAT_WITH_TOOLS = {
    "error": {
        "type": "invalid_request_error",
        "message": "structured response_format with tools/MCP is not supported",
    }
}


def _engine() -> StrataHttpEngine:
    return StrataHttpEngine(base_url=BASE, model_id=MODEL, auth_required=False)


def _ask(**fields: Any) -> GenerateRequest:
    return GenerateRequest(
        messages=[Message(role=Role.user, content="What's the weather in Paris right now?")],
        **fields,
    )


def _sent(route: respx.Route) -> dict[str, Any]:
    return json.loads(route.calls[-1].request.content)


def _text_reply(text: str = "Hello!") -> dict[str, Any]:
    return {
        "id": "chatcmpl-strata",
        "object": "chat.completion",
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text, "reasoning_content": "Say hi."},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 40, "completion_tokens": 3, "total_tokens": 43},
    }


def _call_reply(*calls: tuple[str, str]) -> dict[str, Any]:
    """What Strata answered for a weather question: the calls, reasoning apart."""
    return {
        "id": "chatcmpl-strata",
        "object": "chat.completion",
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "The user wants the weather, so call get_weather.",
                    "tool_calls": [
                        {
                            "id": f"call_{index:024x}",
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                        for index, (name, arguments) in enumerate(calls)
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100},
    }


def _sse(*frames: dict[str, Any]) -> str:
    return "".join(f"data: {json.dumps(f)}\n\n" for f in frames) + "data: [DONE]\n\n"


def _chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
    return {
        "id": "chatcmpl-strata",
        "object": "chat.completion.chunk",
        "model": MODEL,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def _streamed_call() -> str:
    """The call arriving in deltas, as Strata streamed it: valid JSON when joined."""
    return _sse(
        _chunk({"role": "assistant", "reasoning_content": "Call get_weather."}),
        _chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_0f9dbf7e3551468fa0b0c323",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": ""},
                    }
                ]
            }
        ),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"city":'}}]}),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '"Paris"}'}}]}),
        _chunk({}, finish="tool_calls"),
    )


def _client(tmp_path: Path) -> TestClient:
    """The real app, wired from a config the way an install configures a Strata driver."""
    config = tmp_path / "strata.yaml"
    config.write_text(
        json.dumps({"provider": "strata_local", "baseUrl": BASE, "modelId": MODEL}), "utf-8"
    )
    return TestClient(create_app(settings=Settings(config_file=config)))


def _body(**fields: Any) -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": "What's the weather in Paris right now?"}],
        **fields,
    }


def _tool_json(*tools: Tool) -> list[dict[str, Any]]:
    return [t.model_dump(mode="json", exclude_none=True) for t in tools]


# --------------------------------------------------------------------------- #
# what Strata reports
# --------------------------------------------------------------------------- #


def test_strata_reports_tools_and_offers_the_settings_that_carry_them() -> None:
    engine = _engine()
    assert engine.supports_tool_calling is True
    offered = set(engine.supported_settings)
    assert {"tools", "toolChoice", "responseFormat"} <= offered
    # Not claimed: nothing has shown Strata honours them, so an explicit one is refused (A2).
    assert not {"parallelToolCalls", "seed", "logprobs"} & offered


@respx.mock
def test_info_says_toolcalling_true_for_a_strata_driver(tmp_path: Path) -> None:
    """Through the registry's `strata_local`, as the agent runs it: the gateway reads this."""
    respx.get(f"{BASE}/health").mock(
        return_value=httpx.Response(
            200, json={"service": "strata", "loaded": True, "max_context": 131072}
        )
    )
    with _client(tmp_path) as client:
        (model,) = client.get("/v1/info").json()["models"]
    assert model["id"] == MODEL
    assert model["surfaces"] == ["chat"]
    caps = model["capabilities"]
    assert caps["toolCalling"] is True
    assert {"tools", "toolChoice", "responseFormat"} <= set(caps["supportedSettings"])
    assert caps["maxContextTokens"] == 131072


# --------------------------------------------------------------------------- #
# tools, the stream, and the round trip
# --------------------------------------------------------------------------- #


@respx.mock
@pytest.mark.anyio
async def test_tools_reach_strata_and_the_call_comes_back() -> None:
    route = respx.post(CHAT).mock(
        return_value=httpx.Response(200, json=_call_reply(("get_weather", '{"city":"Paris"}')))
    )
    engine = _engine()
    try:
        reply = await engine.generate(_ask(tools=[WEATHER]))
    finally:
        await engine.aclose()

    sent = _sent(route)
    assert [t["function"]["name"] for t in sent["tools"]] == ["get_weather"]
    assert "tool_choice" not in sent, "the caller chose nothing, so nothing is invented"
    assert reply.finishReason is FinishReason.tool_calls
    assert not reply.content, "Strata sends an empty content with a call, as measured"
    assert reply.reasoning == "The user wants the weather, so call get_weather."
    [call] = reply.toolCalls or []
    assert call.function.name == "get_weather"
    assert json.loads(call.function.arguments) == {"city": "Paris"}


@respx.mock
@pytest.mark.anyio
async def test_two_calls_in_one_turn_both_come_back() -> None:
    respx.post(CHAT).mock(
        return_value=httpx.Response(
            200,
            json=_call_reply(
                ("get_weather", '{"city":"Paris"}'), ("get_weather", '{"city":"Tokyo"}')
            ),
        )
    )
    engine = _engine()
    try:
        reply = await engine.generate(_ask(tools=[WEATHER]))
    finally:
        await engine.aclose()
    cities = [json.loads(c.function.arguments)["city"] for c in reply.toolCalls or []]
    assert cities == ["Paris", "Tokyo"]


@respx.mock
@pytest.mark.anyio
async def test_a_streamed_call_arrives_in_fragments_and_whole_at_the_end() -> None:
    route = respx.post(CHAT).mock(
        return_value=httpx.Response(
            200, text=_streamed_call(), headers={"content-type": "text/event-stream"}
        )
    )
    engine = _engine()
    try:
        chunks = [c async for c in engine.stream(_ask(tools=[WEATHER]))]
    finally:
        await engine.aclose()

    sent = _sent(route)
    assert sent["stream"] is True and [t["function"]["name"] for t in sent["tools"]] == [
        "get_weather"
    ]
    # Strata is not llama-server: nothing of its is sent.
    assert not {"id_slot", "return_progress", "timings_per_token"} & sent.keys()
    fragments = [f for c in chunks if c.toolCalls for f in c.toolCalls]
    assert fragments and fragments[0].function is not None
    assert fragments[0].function.name == "get_weather"
    done = chunks[-1].result
    assert done is not None and done.finishReason is FinishReason.tool_calls
    [call] = done.toolCalls or []
    assert json.loads(call.function.arguments) == {"city": "Paris"}


@respx.mock
@pytest.mark.anyio
async def test_the_round_trip_sends_the_result_back_and_gets_the_answer() -> None:
    """Call, then its result, then the answer: the loop an agent runs."""
    route = respx.post(CHAT).mock(
        return_value=httpx.Response(
            200, json=_text_reply("It's currently 17°C in Paris with light rain.")
        )
    )
    request = GenerateRequest(
        messages=[
            Message(role=Role.user, content="What's the weather in Paris right now?"),
            Message(
                role=Role.assistant,
                content=None,
                toolCalls=[
                    {
                        "id": "call_0f9dbf7e3551468fa0b0c323",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                    }
                ],
            ),
            Message(
                role=Role.tool,
                content='{"city":"Paris","temp_c":17,"sky":"light rain"}',
                toolCallId="call_0f9dbf7e3551468fa0b0c323",
            ),
        ],
        tools=[WEATHER],
    )
    engine = _engine()
    try:
        reply = await engine.generate(request)
    finally:
        await engine.aclose()

    messages = _sent(route)["messages"]
    assert messages[1]["role"] == "assistant"
    assert messages[1]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert messages[2] == {
        "role": "tool",
        "content": '{"city":"Paris","temp_c":17,"sky":"light rain"}',
        "tool_call_id": "call_0f9dbf7e3551468fa0b0c323",
    }
    assert reply.content == "It's currently 17°C in Paris with light rain."
    assert reply.toolCalls is None


# --------------------------------------------------------------------------- #
# tool_choice: none is honoured, the rest is sent and named
# --------------------------------------------------------------------------- #


@respx.mock
@pytest.mark.anyio
@pytest.mark.parametrize("stream", [False, True])
async def test_tool_choice_none_sends_no_tools_and_no_tool_choice(stream: bool) -> None:
    """Strata ignores `none` (measured: it called the tool), so nothing is left to ignore."""
    route = respx.post(CHAT).mock(
        return_value=httpx.Response(
            200,
            text=_sse(_chunk({"content": "Paris is rainy."}), _chunk({}, finish="stop")),
            headers={"content-type": "text/event-stream"},
        )
        if stream
        else httpx.Response(200, json=_text_reply("Paris is rainy."))
    )
    request = _ask(tools=[WEATHER], toolChoice="none", callerSettings=["tools", "toolChoice"])
    engine = _engine()
    try:
        if stream:
            _ = [c async for c in engine.stream(request)]
        else:
            await engine.generate(request)
    finally:
        await engine.aclose()

    sent = _sent(route)
    assert "tools" not in sent and "tool_choice" not in sent
    assert engine.ignored_settings(request) == [], "it was honoured, so there is nothing to name"


@respx.mock
@pytest.mark.anyio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("choice", ["required", NAMED], ids=["required", "named"])
async def test_required_and_a_named_function_are_sent_as_strata_takes_them(
    stream: bool, choice: Any
) -> None:
    route = respx.post(CHAT).mock(
        return_value=httpx.Response(
            200, text=_streamed_call(), headers={"content-type": "text/event-stream"}
        )
        if stream
        else httpx.Response(200, json=_call_reply(("get_weather", '{"city":"Paris"}')))
    )
    request = _ask(
        tools=[WEATHER, TIME],
        toolChoice=NamedToolChoice.model_validate(choice) if isinstance(choice, dict) else choice,
        callerSettings=["tools", "toolChoice"],
    )
    engine = _engine()
    try:
        if stream:
            _ = [c async for c in engine.stream(request)]
        else:
            await engine.generate(request)
    finally:
        await engine.aclose()

    sent = _sent(route)
    assert [t["function"]["name"] for t in sent["tools"]] == ["get_weather", "get_time"]
    assert sent["tool_choice"] == choice, "not refused for being a value Strata ignores"
    assert "response_format" not in sent
    assert engine.ignored_settings(request) == ["toolChoice"]


@pytest.mark.parametrize(
    ("choice", "named"),
    [
        (None, []),
        ("auto", []),
        ("none", []),
        ("required", ["toolChoice"]),
        (NAMED, ["toolChoice"]),
    ],
    ids=["absent", "auto", "none", "required", "named"],
)
def test_only_a_tool_choice_strata_ignores_is_named(choice: Any, named: list[str]) -> None:
    fields: dict[str, Any] = {"tools": [WEATHER, TIME]}
    if choice is not None:
        fields["toolChoice"] = (
            NamedToolChoice.model_validate(choice) if isinstance(choice, dict) else choice
        )
    assert _engine().ignored_settings(_ask(**fields)) == named


# --------------------------------------------------------------------------- #
# the header, through the real app, on both endpoints
# --------------------------------------------------------------------------- #


@respx.mock
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("choice", "header"),
    [
        ("required", "toolChoice"),
        (NAMED, "toolChoice"),
        ("auto", None),
        ("none", None),
        (None, None),
    ],
    ids=["required", "named", "auto", "none", "absent"],
)
def test_the_header_names_toolchoice_on_generate_and_on_the_stream(
    tmp_path: Path, stream: bool, choice: Any, header: str | None
) -> None:
    respx.post(CHAT).mock(
        return_value=httpx.Response(
            200, text=_streamed_call(), headers={"content-type": "text/event-stream"}
        )
        if stream
        else httpx.Response(200, json=_call_reply(("get_weather", '{"city":"Paris"}')))
    )
    fields: dict[str, Any] = {"tools": _tool_json(WEATHER, TIME)}
    if choice is not None:
        fields["toolChoice"] = choice
    with _client(tmp_path) as client:
        response = client.post(
            "/v1/generate/stream" if stream else "/v1/generate", json=_body(**fields)
        )
    assert response.status_code == 200, response.text
    assert response.headers.get(HEADER) == header
    if stream:
        # Known before the first frame: it is on the response that carries them.
        assert [name for name, _ in _events(response.text)][-1] == "done"


@respx.mock
def test_a_tools_round_trip_through_the_app_is_a_tool_call_not_a_refusal(tmp_path: Path) -> None:
    """The reproduction of Troy's report: a tools request to a Strata driver was a 400."""
    respx.post(CHAT).mock(
        return_value=httpx.Response(200, json=_call_reply(("get_weather", '{"city":"Paris"}')))
    )
    with _client(tmp_path) as client:
        response = client.post("/v1/generate", json=_body(tools=_tool_json(WEATHER)))
    assert response.status_code == 200, response.text
    reply = response.json()
    assert reply["finishReason"] == "tool_calls"
    assert reply["toolCalls"][0]["function"]["name"] == "get_weather"
    assert HEADER not in response.headers


# --------------------------------------------------------------------------- #
# structured output: Strata's own response_format, never a forced tool
# --------------------------------------------------------------------------- #


@respx.mock
@pytest.mark.anyio
@pytest.mark.parametrize("stream", [False, True])
async def test_response_format_alone_is_strata_native(stream: bool) -> None:
    route = respx.post(CHAT).mock(
        return_value=httpx.Response(
            200,
            text=_sse(_chunk({"content": '{"city":"Paris"}'}), _chunk({}, finish="stop")),
            headers={"content-type": "text/event-stream"},
        )
        if stream
        else httpx.Response(200, json=_text_reply('{"city":"Paris"}'))
    )
    request = _ask(responseFormat=ResponseFormat.model_validate(SCHEMA_FORMAT))
    engine = _engine()
    try:
        if stream:
            _ = [c async for c in engine.stream(request)]
        else:
            await engine.generate(request)
    finally:
        await engine.aclose()

    sent = _sent(route)
    assert sent["response_format"] == SCHEMA_FORMAT
    # Never emulated with a tool: no tool, no choice, and no line added to the prompt.
    assert "tools" not in sent and "tool_choice" not in sent
    assert sent["messages"] == [
        {"role": "user", "content": "What's the weather in Paris right now?"}
    ]
    assert engine.ignored_settings(request) == []


@respx.mock
@pytest.mark.anyio
async def test_response_format_with_tools_is_stratas_own_400_in_its_words() -> None:
    respx.post(CHAT).mock(return_value=httpx.Response(400, json=STRATA_FORMAT_WITH_TOOLS))
    request = _ask(tools=[WEATHER], responseFormat=ResponseFormat.model_validate(SCHEMA_FORMAT))
    engine = _engine()
    try:
        with pytest.raises(CliError) as refused:
            await engine.generate(request)
        assert refused.value.upstream_status == 400
        assert "structured response_format with tools/MCP is not supported" in str(refused.value)
        # Not evidence that Strata cannot call tools: it can, as the next request shows.
        assert engine.supports_tool_calling is True
    finally:
        await engine.aclose()


@respx.mock
@pytest.mark.parametrize("path", ["/v1/generate", "/v1/generate/stream"])
def test_strata_400_for_format_with_tools_is_the_callers_400(tmp_path: Path, path: str) -> None:
    """Relayed in Strata's words on both endpoints, and before any stream opens."""
    respx.post(CHAT).mock(return_value=httpx.Response(400, json=STRATA_FORMAT_WITH_TOOLS))
    with _client(tmp_path) as client:
        response = client.post(
            path, json=_body(tools=_tool_json(WEATHER), responseFormat=SCHEMA_FORMAT)
        )
        assert response.status_code == 400, response.text
        detail = response.json()["detail"]
        assert detail["type"].endswith("#backend-rejected-request")
        assert detail["retryDisposition"] == "terminal"
        assert "structured response_format with tools/MCP is not supported" in detail["detail"]
        # And tools stay on: the driver learned nothing from a 400 about format-with-tools.
        (model,) = client.get("/v1/info").json()["models"]
        assert model["capabilities"]["toolCalling"] is True


@respx.mock
@pytest.mark.anyio
async def test_a_named_choice_never_takes_the_forced_tool_route() -> None:
    """The base engine answers a named `tool_choice` on llama-server by trading the
    tools for a JSON-schema `response_format` (A5). That route is for an engine that
    ignores `tool_choice` and enforces a grammar; it must never be Strata's, not even
    when `/props` looks like llama-server's and the engine believes it."""
    respx.get(f"{BASE}/props").mock(
        return_value=httpx.Response(200, json={"default_generation_settings": {"n_ctx": 16384}})
    )
    route = respx.post(CHAT).mock(
        return_value=httpx.Response(200, json=_call_reply(("get_weather", '{"city":"Paris"}')))
    )
    request = _ask(tools=[WEATHER, TIME], toolChoice=NamedToolChoice.model_validate(NAMED))
    engine = _engine()

    async def _yes() -> bool:
        return True

    engine._answers_as_llama_cpp = _yes  # type: ignore[method-assign]
    try:
        reply = await engine.generate(request)
    finally:
        await engine.aclose()

    sent = _sent(route)
    assert sent["tool_choice"] == NAMED
    assert [t["function"]["name"] for t in sent["tools"]] == ["get_weather", "get_time"]
    assert "response_format" not in sent
    assert sent["messages"][0]["role"] == "user", "no system line added to force a function"
    [call] = reply.toolCalls or []
    assert call.function.name == "get_weather", "Strata's own choice, reported as it came"


@pytest.mark.anyio
async def test_strata_still_refuses_what_it_does_not_take() -> None:
    """The settings Strata has never carried are still refused when explicit."""
    engine = _engine()
    for field, value in (("seed", 1), ("parallelToolCalls", False)):
        request = _ask(tools=[WEATHER], callerSettings=[field], **{field: value})
        with pytest.raises(CliError) as refused:
            engine._payload_for(request, engine.resolve_model(None))
        assert refused.value.upstream_status == 400
    await engine.aclose()
