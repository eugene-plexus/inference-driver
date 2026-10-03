"""What an Ollama backend silently ignores, and what it reads differently.

Upstream drift audit, 2026-10-03, Ollama v0.35.1:

* Its OpenAI-compatible request (`openai/openai.go`) has no `tool_choice`,
  `top_k`, `min_p` or `parallel_tool_calls`, so they are dropped without a
  word. A setting the caller asked for must be refused rather than
  dropped (A2), so `top_k` and `min_p` are not offered for an Ollama
  backend at all, and `tool_choice` / `parallel_tool_calls` are refused for
  every value Ollama would not honour by ignoring it (`auto` and `true` it
  honours: that is what it does anyway, and Codex sends both on every
  request).
* It reads replayed reasoning only as `reasoning` (ollama#18534); we sent
  `reasoning_content`, which llama.cpp reads and Ollama drops.
* It matches tool results to calls by position, not by id (ollama#18762),
  so the results go back in the order the assistant made the calls.
* `/api/ps` was the one probe sent without the operator's key.

Each Ollama rule is paired with the same request to a llama-server-style
custom backend, whose behaviour must not move.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from eugene_plexus_inference_driver._generated.models import BackendKind, GenerateRequest
from eugene_plexus_inference_driver.engines._subprocess import CliError
from eugene_plexus_inference_driver.engines.openai_compat_http import OpenAiCompatibleHttpEngine

OLLAMA = "http://127.0.0.1:11434"
LLAMA = "http://127.0.0.1:8081"
OK = {
    "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
}
TOOLS = [
    {"type": "function", "function": {"name": "a", "parameters": {"type": "object"}}},
    {"type": "function", "function": {"name": "b", "parameters": {"type": "object"}}},
]


def _engine(source: str, api_key: str | None = None) -> OpenAiCompatibleHttpEngine:
    return OpenAiCompatibleHttpEngine(
        api_key=api_key,
        base_url=OLLAMA if source == "ollama" else LLAMA,
        model_id="qwen3:8b",
        backend_kind=BackendKind.openai_compat_http,
        auth_required=False,
        catalogue_source=source,
    )


def _ask(*, explicit: bool = True, **fields: Any) -> GenerateRequest:
    body: dict[str, Any] = {"messages": [{"role": "user", "content": "hi"}], **fields}
    if explicit:
        body["callerSettings"] = [k for k in fields if k != "tools"] or None
    return GenerateRequest.model_validate(body)


def _route(base: str) -> respx.Route:
    return respx.post(f"{base}/v1/chat/completions").mock(return_value=httpx.Response(200, json=OK))


def _sent(route: respx.Route) -> dict[str, Any]:
    return json.loads(route.calls[-1].request.content)


# --------------------------------------------------------------------------- #
# Settings Ollama drops
# --------------------------------------------------------------------------- #


def test_ollama_is_not_offered_for_the_samplers_it_drops() -> None:
    settings = set(_engine("ollama").supported_settings)
    assert not settings & {"topK", "minP"}
    # Listed, because `auto` and `true` are honoured; the other values are
    # refused per request below.
    assert {"toolChoice", "parallelToolCalls", "tools"} <= settings
    assert {"topK", "minP"} <= set(_engine("openai").supported_settings)


@pytest.mark.parametrize(("field", "value"), [("topK", 20), ("minP", 0.05)])
@respx.mock
async def test_an_explicit_sampler_ollama_drops_is_refused(field: str, value: Any) -> None:
    route = _route(OLLAMA)
    with pytest.raises(CliError) as refused:
        await _engine("ollama").generate(_ask(**{field: value}))
    assert refused.value.upstream_status == 400
    assert not route.called


@respx.mock
async def test_an_inherited_sampler_is_left_out_for_ollama_and_sent_to_llama_server() -> None:
    ollama, llama = _route(OLLAMA), _route(LLAMA)
    await _engine("ollama").generate(_ask(explicit=False, topK=20, minP=0.05))
    await _engine("openai").generate(_ask(explicit=False, topK=20, minP=0.05))
    assert "top_k" not in _sent(ollama) and "min_p" not in _sent(ollama)
    assert _sent(llama)["top_k"] == 20 and _sent(llama)["min_p"] == 0.05


@pytest.mark.parametrize(
    "choice", ["required", "none", {"type": "function", "function": {"name": "a"}}]
)
@respx.mock
async def test_a_tool_choice_ollama_would_ignore_is_refused(choice: Any) -> None:
    route = _route(OLLAMA)
    with pytest.raises(CliError) as refused:
        await _engine("ollama").generate(_ask(tools=TOOLS, toolChoice=choice))
    assert refused.value.upstream_status == 400
    assert "tool_choice" in str(refused.value)
    assert not route.called


@respx.mock
async def test_auto_and_parallel_true_are_what_ollama_does_and_still_work() -> None:
    route = _route(OLLAMA)
    await _engine("ollama").generate(_ask(tools=TOOLS, toolChoice="auto", parallelToolCalls=True))
    assert _sent(route)["tool_choice"] == "auto"


@respx.mock
async def test_one_call_a_turn_is_refused_for_ollama_and_sent_to_llama_server() -> None:
    ollama, llama = _route(OLLAMA), _route(LLAMA)
    with pytest.raises(CliError) as refused:
        await _engine("ollama").generate(_ask(tools=TOOLS, parallelToolCalls=False))
    assert "parallel_tool_calls" in str(refused.value)
    assert not ollama.called
    await _engine("openai").generate(_ask(tools=TOOLS, parallelToolCalls=False))
    assert _sent(llama)["parallel_tool_calls"] is False


@respx.mock
async def test_a_required_tool_choice_still_reaches_llama_server() -> None:
    route = _route(LLAMA)
    await _engine("openai").generate(_ask(tools=TOOLS, toolChoice="required"))
    assert _sent(route)["tool_choice"] == "required"


# --------------------------------------------------------------------------- #
# Replayed reasoning, and tool results in call order
# --------------------------------------------------------------------------- #

HISTORY = [
    {"role": "user", "content": "Weather in Oslo and Bergen?"},
    {
        "role": "assistant",
        "content": None,
        "reasoning": "Two cities, two calls.",
        "toolCalls": [
            {"id": "call_oslo", "type": "function", "function": {"name": "a", "arguments": "{}"}},
            {"id": "call_bergen", "type": "function", "function": {"name": "b", "arguments": "{}"}},
        ],
    },
    # The harness answered Bergen first.
    {"role": "tool", "toolCallId": "call_bergen", "content": "rain"},
    {"role": "tool", "toolCallId": "call_oslo", "content": "snow"},
    {"role": "user", "content": "And tomorrow?"},
]


def _history() -> GenerateRequest:
    return GenerateRequest.model_validate({"messages": HISTORY, "tools": TOOLS})


@respx.mock
async def test_ollama_is_given_reasoning_under_the_name_it_reads() -> None:
    ollama, llama = _route(OLLAMA), _route(LLAMA)
    await _engine("ollama").generate(_history())
    await _engine("openai").generate(_history())
    to_ollama = _sent(ollama)["messages"][1]
    assert to_ollama["reasoning"] == "Two cities, two calls."
    assert "reasoning_content" not in to_ollama
    to_llama = _sent(llama)["messages"][1]
    assert to_llama["reasoning_content"] == "Two cities, two calls."
    assert "reasoning" not in to_llama


@respx.mock
async def test_ollama_gets_tool_results_in_the_order_the_calls_were_made() -> None:
    ollama, llama = _route(OLLAMA), _route(LLAMA)
    await _engine("ollama").generate(_history())
    await _engine("openai").generate(_history())
    order = [m.get("tool_call_id") for m in _sent(ollama)["messages"] if m["role"] == "tool"]
    assert order == ["call_oslo", "call_bergen"]
    contents = [m["content"] for m in _sent(ollama)["messages"] if m["role"] == "tool"]
    assert contents == ["snow", "rain"]
    # Everything else stays where it was.
    assert [m["role"] for m in _sent(ollama)["messages"]] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "user",
    ]
    # llama-server matches by id: its order is the caller's.
    order = [m.get("tool_call_id") for m in _sent(llama)["messages"] if m["role"] == "tool"]
    assert order == ["call_bergen", "call_oslo"]


# --------------------------------------------------------------------------- #
# The window probe carries the key like every other call
# --------------------------------------------------------------------------- #


@respx.mock
async def test_the_ollama_window_probe_sends_the_operators_key() -> None:
    respx.get(f"{OLLAMA}/props").mock(return_value=httpx.Response(404))
    respx.get(f"{OLLAMA}/v1/models").mock(return_value=httpx.Response(200, json={"data": []}))
    ps = respx.get(f"{OLLAMA}/api/ps").mock(
        return_value=httpx.Response(
            200, json={"models": [{"name": "qwen3:8b", "context_length": 32768}]}
        )
    )
    window = await _engine("ollama", api_key="ollama-behind-a-proxy").context_window()
    assert window == 32768
    assert ps.calls[0].request.headers.get("Authorization") == "Bearer ollama-behind-a-proxy"
