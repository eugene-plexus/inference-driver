"""A named `tool_choice` on llama-server, answered by structured output (A5).

Measured 2026-09-30 (specs `docs/design/tool-call-repair.md`): llama-server
b11303 reads `tool_choice` as a string, so `{"type": "function",
"function": {"name": ...}}` falls back to `auto` with only a log warning,
and 22 of 30 forced answers across ten models came back as prose.
`"required"` with only that tool was not enough either: Qwen3.6-35B-A3B
wrote prose and degenerated to the token limit 3 times of 3. What the
engine does enforce is a JSON schema as the response format, which gave a
valid call there 3 of 3. So on llama-server the driver asks for the named
tool's arguments as structured output, and returns them as the call.

Every other backend gets the request exactly as before: OpenAI, vLLM and
the rest honour a named choice themselves.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from eugene_plexus_inference_driver._generated.models import (
    FinishReason,
    FunctionDefinition,
    GenerateRequest,
    Message,
    NamedToolChoice,
    Role,
    Tool,
)
from eugene_plexus_inference_driver.engines.openai_compat_http import OpenAiCompatibleHttpEngine

pytestmark = pytest.mark.anyio

BASE = "http://127.0.0.1:8090"
PARAMETERS = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}
WEATHER = Tool(
    type="function",
    function=FunctionDefinition(
        name="get_weather", description="Get the current weather for a city.", parameters=PARAMETERS
    ),
)
TIME = Tool(type="function", function=FunctionDefinition(name="get_time", parameters={}))
FORCED = NamedToolChoice.model_validate({"type": "function", "function": {"name": "get_weather"}})


def _engine() -> OpenAiCompatibleHttpEngine:
    return OpenAiCompatibleHttpEngine(base_url=BASE, model_id="qwen", auth_required=False)


def _ask(**kwargs: Any) -> GenerateRequest:
    return GenerateRequest(
        messages=[Message(role=Role.user, content="Tell me about Oslo.")],
        tools=[WEATHER, TIME],
        toolChoice=FORCED,
        **kwargs,
    )


def _llama_server() -> None:
    respx.get(f"{BASE}/props").mock(
        return_value=httpx.Response(200, json={"default_generation_settings": {"n_ctx": 16384}})
    )


def _reply(content: str, finish: str = "stop") -> dict[str, Any]:
    return {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish,
            }
        ],
        "model": "qwen",
    }


def _sse(*frames: dict[str, Any]) -> str:
    return "".join(f"data: {json.dumps(f)}\n\n" for f in frames) + "data: [DONE]\n\n"


@respx.mock
async def test_llama_server_is_asked_for_the_arguments_as_structured_output() -> None:
    _llama_server()
    route = respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_reply('{"city": "Oslo"}'))
    )
    result = await _engine().generate(_ask())

    sent = json.loads(route.calls[0].request.content)
    assert "tools" not in sent and "tool_choice" not in sent and "parallel_tool_calls" not in sent
    assert sent["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "get_weather", "schema": PARAMETERS},
    }
    system = sent["messages"][0]
    assert system["role"] == "system" and "get_weather" in system["content"]
    assert sent["messages"][1] == {"role": "user", "content": "Tell me about Oslo."}

    assert result.content is None
    assert result.finishReason is FinishReason.tool_calls
    [call] = result.toolCalls or []
    assert call.function.name == "get_weather"
    assert json.loads(call.function.arguments) == {"city": "Oslo"}
    assert call.id.startswith("call_") and len(call.id) >= 9


@respx.mock
async def test_an_existing_system_message_is_kept_and_told() -> None:
    _llama_server()
    route = respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_reply('{"city": "Oslo"}'))
    )
    request = GenerateRequest(
        messages=[
            Message(role=Role.system, content="You are terse."),
            Message(role=Role.user, content="Tell me about Oslo."),
        ],
        tools=[WEATHER],
        toolChoice=FORCED,
    )
    await _engine().generate(request)
    messages = json.loads(route.calls[0].request.content)["messages"]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"].startswith("You are terse.")
    assert "get_weather" in messages[0]["content"]


@respx.mock
async def test_a_streamed_call_arrives_as_one_call_and_reasoning_still_streams() -> None:
    _llama_server()
    respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            text=_sse(
                {"choices": [{"index": 0, "delta": {"reasoning_content": "Oslo, weather."}}]},
                {"choices": [{"index": 0, "delta": {"content": '{"ci'}}]},
                {"choices": [{"index": 0, "delta": {"content": 'ty": "Oslo"}'}}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            ),
        )
    )
    chunks = [chunk async for chunk in _engine().stream(_ask())]

    assert [c.reasoning for c in chunks if c.reasoning] == ["Oslo, weather."]
    assert not [c.text for c in chunks if c.text], "the arguments are not the answer's text"
    [fragments] = [c.toolCalls for c in chunks if c.toolCalls]
    [fragment] = fragments
    assert fragment.index == 0 and fragment.function is not None
    assert fragment.function.name == "get_weather"
    assert json.loads(fragment.function.arguments or "") == {"city": "Oslo"}
    done = chunks[-1].result
    assert done is not None and done.finishReason is FinishReason.tool_calls
    assert done.content is None
    [call] = done.toolCalls or []
    assert call.id == fragment.id and len(call.id) >= 9


@respx.mock
async def test_arguments_cut_off_by_the_budget_are_not_a_call() -> None:
    """The honest end of a model that ran out: `length`, and no call."""
    _llama_server()
    respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_reply('{"ci', finish="length"))
    )
    result = await _engine().generate(_ask())
    assert result.toolCalls is None
    assert result.finishReason is FinishReason.length
    assert result.content == '{"ci'


@respx.mock
async def test_another_backend_gets_the_named_choice_untouched() -> None:
    respx.get(f"{BASE}/props").mock(return_value=httpx.Response(404))
    route = respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_reply("ok"))
    )
    await _engine().generate(_ask())
    sent = json.loads(route.calls[0].request.content)
    assert sent["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}
    assert [t["function"]["name"] for t in sent["tools"]] == ["get_weather", "get_time"]
    assert "response_format" not in sent


@respx.mock
async def test_a_choice_naming_no_offered_tool_is_left_to_the_backend() -> None:
    _llama_server()
    route = respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_reply("ok"))
    )
    request = GenerateRequest(
        messages=[Message(role=Role.user, content="hi")], tools=[TIME], toolChoice=FORCED
    )
    await _engine().generate(request)
    sent = json.loads(route.calls[0].request.content)
    assert sent["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}


@respx.mock
async def test_a_streamed_call_with_no_id_gets_one_long_enough_for_any_template() -> None:
    """Mistral's template refuses a tool-call id under 9 characters, and
    `call_0` was what the driver made up when a backend sent none."""
    respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            text=_sse(
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"name": "get_time", "arguments": "{}"},
                                    }
                                ]
                            },
                        }
                    ]
                },
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
            ),
        )
    )
    request = GenerateRequest(messages=[Message(role=Role.user, content="time?")], tools=[TIME])
    chunks = [chunk async for chunk in _engine().stream(request)]
    [call] = chunks[-1].result.toolCalls or []  # type: ignore[union-attr]
    assert call.id.startswith("call_") and len(call.id) >= 9
