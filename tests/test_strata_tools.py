"""Strata takes tools, tool results and structured output (Troy, 2026-10-12).

Until that day the driver refused all three for Strata, because nobody had tried
them, and the gateway listed every Strata model without tools. Measured on Troy's
running Strata, 2026-10-12 (`qwen3.8-flash-next-iq2_xs`, its own OpenAI endpoint,
whose `/health` names no version; `docs/design/tools-assumed.md` in specs): it calls
tools correctly (one call, two at once, streamed, and the round trip with the tool's
result) and answers `response_format` natively. What it does not do is read
`tool_choice`: `none` still called the tool, `required` answered "Hello!", and a named
function called another one. Nor `parallel_tool_calls: false`: with a question that
wants two calls, the field absent, `true` and `false` all answered both (measured
again on 2026-10-12, six runs). And `response_format` together with tools is its own
400.

What it does not honour is **sent anyway and named**, not refused: Codex sends
`parallel_tool_calls: false` on every request, so a refusal made Strata unusable
from it. Named twice, on purpose: `capabilities.settingsNotHonoured` (static, so
the gateway can name it before a stream opens) and the per-request header.

The backend here is `respx`, answering in the shapes Strata sent, trimmed to
the fields the driver reads. Each test fails against the driver as it was:
every one of them was a 400 from the driver before any HTTP went out.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from PIL import Image

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
from tests.test_attachments import audio_part, file_part
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
    # `parallelToolCalls` is offered although Strata ignores `false`: a measured gap is named,
    # not refused, so the gateway does not route Codex (which sends it on every request) away.
    assert {"tools", "toolChoice", "parallelToolCalls", "responseFormat"} <= offered
    # Carried since round four: Strata's own source reads them per request (`test_strata.py`).
    assert {"seed", "frequencyPenalty", "presencePenalty"} <= offered
    # Not claimed: nothing has shown Strata returns them, so an explicit one is refused (A2).
    assert "logprobs" not in offered


#: What the contract says Strata is sent and does not honour (specs 6178255): `toolChoice` with
#: `required` and `function` (it honours `auto`; `none` is given as no tools), and
#: `parallelToolCalls` with `false`.
STRATA_GAPS = [
    {"setting": "toolChoice", "values": ["required", "function"]},
    {"setting": "parallelToolCalls", "values": [False]},
]


def test_strata_names_what_it_does_not_honour_by_value() -> None:
    engine = _engine()
    gaps = [g.model_dump(mode="json") for g in engine.settings_not_honoured()]
    assert gaps == STRATA_GAPS
    # A fresh list each time, so whoever holds one cannot change what the next is told.
    engine.settings_not_honoured()[0].values.append("auto")
    assert [g.model_dump(mode="json") for g in engine.settings_not_honoured()] == STRATA_GAPS


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
    assert {"tools", "toolChoice", "parallelToolCalls", "responseFormat"} <= set(
        caps["supportedSettings"]
    )
    assert caps["maxContextTokens"] == 131072
    # Static, per backend: what the gateway names on its own header before a stream opens.
    assert caps["settingsNotHonoured"] == STRATA_GAPS


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
    """Strata ignores `none` (measured: it called the tool), so nothing is left to ignore.
    What rides with the tools goes with them: `parallel_tool_calls: false` beside no tools
    would be a setting sent for nothing, and named as ignored for nothing."""
    route = respx.post(CHAT).mock(
        return_value=httpx.Response(
            200,
            text=_sse(_chunk({"content": "Paris is rainy."}), _chunk({}, finish="stop")),
            headers={"content-type": "text/event-stream"},
        )
        if stream
        else httpx.Response(200, json=_text_reply("Paris is rainy."))
    )
    request = _ask(
        tools=[WEATHER],
        toolChoice="none",
        parallelToolCalls=False,
        callerSettings=["tools", "toolChoice", "parallelToolCalls"],
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
    assert "tools" not in sent and "tool_choice" not in sent and "parallel_tool_calls" not in sent
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
    ("choice", "parallel", "named"),
    [
        (None, None, []),
        ("auto", None, []),
        ("none", None, []),
        ("required", None, ["toolChoice"]),
        (NAMED, None, ["toolChoice"]),
        (None, True, []),
        (None, False, ["parallelToolCalls"]),
        ("auto", True, []),
        ("auto", False, ["parallelToolCalls"]),
        ("required", False, ["toolChoice", "parallelToolCalls"]),
        (NAMED, False, ["toolChoice", "parallelToolCalls"]),
        # `none` is honoured as no tools, and `false` went with them: nothing was ignored.
        ("none", False, []),
    ],
    ids=[
        "absent",
        "auto",
        "none",
        "required",
        "named",
        "parallel-true",
        "parallel-false",
        "auto-parallel-true",
        "auto-parallel-false",
        "required-parallel-false",
        "named-parallel-false",
        "none-parallel-false",
    ],
)
def test_only_what_strata_ignores_is_named(
    choice: Any, parallel: bool | None, named: list[str]
) -> None:
    fields: dict[str, Any] = {"tools": [WEATHER, TIME]}
    caller = ["tools"]
    if choice is not None:
        fields["toolChoice"] = (
            NamedToolChoice.model_validate(choice) if isinstance(choice, dict) else choice
        )
        caller.append("toolChoice")
    if parallel is not None:
        fields["parallelToolCalls"] = parallel
        caller.append("parallelToolCalls")
    assert _engine().ignored_settings(_ask(callerSettings=caller, **fields)) == named


@pytest.mark.parametrize(
    ("choice", "parallel"),
    [("required", None), (NAMED, None), (None, False), ("required", False)],
    ids=["required", "named", "parallel-false", "required-parallel-false"],
)
def test_a_setting_the_caller_did_not_send_is_sent_but_not_named(
    choice: Any, parallel: bool | None
) -> None:
    """The header names what the caller sent (`callerSettings`), as the gateway does, so the two
    never disagree about a request. A setting that rides on the request without the caller
    having sent it (a profile's or an install's default) is still sent to Strata, which still
    ignores it, and is not named: the caller never asked for it, and the gateway's own header
    says nothing of it either.

    **A direct caller gets the header only for the settings it lists in `callerSettings`.** One
    that lists none, as a request made without the gateway often does, is sent exactly what it
    asked for and is named nothing: the provenance is the caller's to give, and without it every
    setting reads as one the caller never asked for."""
    fields: dict[str, Any] = {"tools": [WEATHER, TIME]}
    if choice is not None:
        fields["toolChoice"] = (
            NamedToolChoice.model_validate(choice) if isinstance(choice, dict) else choice
        )
    if parallel is not None:
        fields["parallelToolCalls"] = parallel
    engine = _engine()
    assert engine.ignored_settings(_ask(callerSettings=["tools"], **fields)) == []
    # Not named, though sent: the settings are on the wire as the caller left them, so the
    # reason there is nothing to name is the missing provenance, and not that nothing was sent.
    assert engine.ignored_settings(_ask(**fields)) == [], "no callerSettings: sent, and not named"
    assert engine.ignored_settings(_ask(callerSettings=[], **fields)) == []
    payload = engine._payload_for(_ask(**fields), engine.resolve_model(None))
    if choice is not None:
        assert "tool_choice" in payload, "sent: it is only the naming that has no provenance"
    if parallel is not None:
        assert payload["parallel_tool_calls"] is parallel


# --------------------------------------------------------------------------- #
# the header, through the real app, on both endpoints
# --------------------------------------------------------------------------- #


@respx.mock
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("choice", "parallel", "header"),
    [
        ("required", None, "toolChoice"),
        (NAMED, None, "toolChoice"),
        ("auto", None, None),
        ("none", None, None),
        (None, None, None),
        (None, True, None),
        (None, False, "parallelToolCalls"),
        ("auto", False, "parallelToolCalls"),
        ("required", False, "toolChoice, parallelToolCalls"),
        ("none", False, None),
    ],
    ids=[
        "required",
        "named",
        "auto",
        "none",
        "absent",
        "parallel-true",
        "parallel-false",
        "auto-parallel-false",
        "required-parallel-false",
        "none-parallel-false",
    ],
)
def test_the_header_names_what_strata_ignores_on_generate_and_on_the_stream(
    tmp_path: Path, stream: bool, choice: Any, parallel: bool | None, header: str | None
) -> None:
    respx.post(CHAT).mock(
        return_value=httpx.Response(
            200, text=_streamed_call(), headers={"content-type": "text/event-stream"}
        )
        if stream
        else httpx.Response(200, json=_call_reply(("get_weather", '{"city":"Paris"}')))
    )
    fields: dict[str, Any] = {"tools": _tool_json(WEATHER, TIME)}
    caller = ["tools"]
    if choice is not None:
        fields["toolChoice"] = choice
        caller.append("toolChoice")
    if parallel is not None:
        fields["parallelToolCalls"] = parallel
        caller.append("parallelToolCalls")
    with _client(tmp_path) as client:
        response = client.post(
            "/v1/generate/stream" if stream else "/v1/generate",
            json=_body(callerSettings=caller, **fields),
        )
    assert response.status_code == 200, response.text
    assert response.headers.get(HEADER) == header
    if stream:
        # Known before the first frame: it is on the response that carries them.
        assert [name for name, _ in _events(response.text)][-1] == "done"


@respx.mock
@pytest.mark.parametrize("stream", [False, True])
def test_a_setting_that_is_not_the_callers_is_sent_to_strata_and_not_named(
    tmp_path: Path, stream: bool
) -> None:
    """Through the app, on the wire and on the response: `callerSettings` lists only `tools`,
    so the `parallel_tool_calls: false` and `tool_choice: required` that ride on the request
    (a profile's, say) go to Strata as they always did, and neither is named."""
    route = respx.post(CHAT).mock(
        return_value=httpx.Response(
            200, text=_streamed_call(), headers={"content-type": "text/event-stream"}
        )
        if stream
        else httpx.Response(200, json=_call_reply(("get_weather", '{"city":"Paris"}')))
    )
    body = _body(
        tools=_tool_json(WEATHER, TIME),
        toolChoice="required",
        parallelToolCalls=False,
        callerSettings=["tools"],
    )
    with _client(tmp_path) as client:
        response = client.post("/v1/generate/stream" if stream else "/v1/generate", json=body)
    assert response.status_code == 200, response.text
    sent = _sent(route)
    assert sent["tool_choice"] == "required" and sent["parallel_tool_calls"] is False
    assert HEADER not in response.headers


@respx.mock
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("value", [False, True])
def test_parallel_tool_calls_is_sent_to_strata_whichever_way_it_is_set(
    tmp_path: Path, stream: bool, value: bool
) -> None:
    """Both values go on the wire (Strata honours `true` and ignores `false`; the driver says
    so and does not decide for it), and an explicit one is not the 400 it was."""
    route = respx.post(CHAT).mock(
        return_value=httpx.Response(
            200, text=_streamed_call(), headers={"content-type": "text/event-stream"}
        )
        if stream
        else httpx.Response(200, json=_call_reply(("get_weather", '{"city":"Paris"}')))
    )
    body = _body(
        tools=_tool_json(WEATHER),
        parallelToolCalls=value,
        callerSettings=["tools", "parallelToolCalls"],
    )
    with _client(tmp_path) as client:
        response = client.post("/v1/generate/stream" if stream else "/v1/generate", json=body)
    assert response.status_code == 200, response.text
    assert _sent(route)["parallel_tool_calls"] is value
    assert response.headers.get(HEADER) == (None if value else "parallelToolCalls")


@respx.mock
@pytest.mark.parametrize("stream", [False, True])
def test_a_codex_shaped_request_is_served_with_the_gaps_named(tmp_path: Path, stream: bool) -> None:
    """Codex sends `parallel_tool_calls: false` and `tool_choice: auto` on every request, tools
    or not, all of it explicit (`callerSettings`). Refusing either made Strata unusable from it
    (found by the first build round's review); both go through, and the gap is named."""
    route = respx.post(CHAT).mock(
        return_value=httpx.Response(
            200, text=_streamed_call(), headers={"content-type": "text/event-stream"}
        )
        if stream
        else httpx.Response(200, json=_call_reply(("get_weather", '{"city":"Paris"}')))
    )
    body = _body(
        tools=_tool_json(WEATHER, TIME),
        toolChoice="auto",
        parallelToolCalls=False,
        callerSettings=["tools", "toolChoice", "parallelToolCalls"],
    )
    with _client(tmp_path) as client:
        response = client.post("/v1/generate/stream" if stream else "/v1/generate", json=body)
    assert response.status_code == 200, response.text
    sent = _sent(route)
    assert sent["tool_choice"] == "auto" and sent["parallel_tool_calls"] is False
    assert response.headers[HEADER] == "parallelToolCalls"
    if not stream:
        assert response.json()["toolCalls"][0]["function"]["name"] == "get_weather"


@respx.mock
def test_parallel_false_with_no_tools_is_sent_and_named_and_not_refused(tmp_path: Path) -> None:
    """Codex sends it on a turn with no tools too. Strata answered that with a 200
    (measured, 2026-10-12), so it is not for this driver to refuse."""
    route = respx.post(CHAT).mock(return_value=httpx.Response(200, json=_text_reply("Hello!")))
    body = _body(parallelToolCalls=False, callerSettings=["parallelToolCalls"])
    with _client(tmp_path) as client:
        response = client.post("/v1/generate", json=body)
    assert response.status_code == 200, response.text
    assert _sent(route)["parallel_tool_calls"] is False
    assert response.headers[HEADER] == "parallelToolCalls"


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
    """The settings Strata has never carried are still refused when explicit. (A setting it is
    measured to ignore, `parallel_tool_calls: false`, is sent and named instead: above; and
    `seed` and the penalties, which its source reads per request, are carried since round four:
    `test_strata.py`.)"""
    engine = _engine()
    for field, value in (("logprobs", True), ("logitBias", {"1": 2})):
        request = _ask(tools=[WEATHER], callerSettings=[field], **{field: value})
        with pytest.raises(CliError) as refused:
            engine._payload_for(request, engine.resolve_model(None))
        assert refused.value.upstream_status == 400
    await engine.aclose()


# --------------------------------------------------------------------------- #
# what is not text is the route's to refuse, in words that name the cause
# --------------------------------------------------------------------------- #


def _png() -> str:
    """A real PNG, so that what refuses it is Strata's lack of image input and not a malformed
    attachment (each part below is valid for what it says it is)."""
    data = io.BytesIO()
    Image.new("RGB", (8, 8), "purple").save(data, format="PNG")
    return "data:image/png;base64," + base64.b64encode(data.getvalue()).decode()


@respx.mock
@pytest.mark.parametrize("path", ["/v1/generate", "/v1/generate/stream"])
@pytest.mark.parametrize(
    ("part", "title"),
    [
        ({"type": "image_url", "image_url": {"url": _png()}}, "Image input not supported"),
        (audio_part(), "Audio input not supported"),
        (file_part(), "File input not supported"),
    ],
    ids=["image", "audio", "file"],
)
def test_a_part_that_is_not_text_is_refused_before_strata_is_asked(
    tmp_path: Path, path: str, part: dict[str, Any], title: str
) -> None:
    """The check that used to sit in `StrataHttpEngine._payload_for` ("text messages only") was
    dead code: a message's content there is a typed object, never a Python `list`, so it
    never fired. The route is what refuses, and it does so for every engine whose backend has
    not confirmed the input: a 400 that says what the backend lacks, before inference is sent."""
    status = respx.get(f"{BASE}/v1/status").respond(
        200,
        json={
            "service": "strata",
            "model": MODEL,
            "loaded": True,
            "vision": {"enabled": False, "available": False},
        },
    )
    chat = respx.post(CHAT).mock(return_value=httpx.Response(200, json=_text_reply("Hello!")))
    body = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "What is this?"}, part]}
        ],
    }
    with _client(tmp_path) as client:
        response = client.post(path, json=body)
    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert detail["title"] == title
    assert "capabilities." in detail["detail"] and "must be true" in detail["detail"]
    assert status.called is (part["type"] == "image_url")
    assert not chat.called, "no inference was sent to Strata"


@respx.mock
def test_text_only_parts_are_flattened_and_reach_strata_as_text(tmp_path: Path) -> None:
    """The other half of the same question: a content array of text parts only is plain text
    to the route (`validate_messages` flattens it), so Strata gets what it always could."""
    route = respx.post(CHAT).mock(return_value=httpx.Response(200, json=_text_reply("Hello!")))
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is"},
                    {"type": "text", "text": " the weather?"},
                ],
            }
        ],
    }
    with _client(tmp_path) as client:
        response = client.post("/v1/generate", json=body)
    assert response.status_code == 200, response.text
    sent = _sent(route)["messages"]
    assert [m["role"] for m in sent] == ["user"]
    assert isinstance(sent[0]["content"], str) and "weather" in sent[0]["content"]
