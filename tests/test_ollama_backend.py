"""What an Ollama backend silently ignores, and what it reads differently.

Upstream drift audit, 2026-10-03, Ollama v0.35.1:

* Its OpenAI-compatible request (`openai/openai.go`) has no `tool_choice`,
  `top_k`, `min_p` or `parallel_tool_calls`, so they are dropped without a
  word. A setting the caller asked for must be refused rather than
  dropped (A2), so `top_k` and `min_p` are not offered for an Ollama
  backend at all. `tool_choice` and `parallel_tool_calls` are the other
  kind (Troy, 2026-10-12, `docs/design/tools-assumed.md`): `auto` and `true`
  are what Ollama does anyway, `none` is honoured by sending no tools, and
  every other value is **sent and named** (`capabilities.settingsNotHonoured`,
  `x-eugene-plexus-ignored-settings`), not refused. Codex sends both on every
  request, and until round two of that work Ollama refused it.
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
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from eugene_plexus_inference_driver._generated.models import BackendKind, GenerateRequest
from eugene_plexus_inference_driver.app import create_app
from eugene_plexus_inference_driver.engines._subprocess import CliError
from eugene_plexus_inference_driver.engines.openai_compat_http import OpenAiCompatibleHttpEngine
from eugene_plexus_inference_driver.settings import Settings
from tests.test_accounts import _wait_for_catalogue

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
    # Listed, because `auto` and `true` are honoured and every other value is sent and
    # named below (never refused: a refusal is for what cannot be sent at all).
    assert {"toolChoice", "parallelToolCalls", "tools"} <= settings
    assert {"topK", "minP"} <= set(_engine("openai").supported_settings)


def test_ollama_names_what_it_ignores_by_value_and_llama_server_names_nothing() -> None:
    """`capabilities.settingsNotHonoured`: static per backend, so the gateway can name it
    before a stream opens. Ollama's OpenAI-compatible request has no field for `tool_choice`
    or `parallel_tool_calls` (`openai/openai.go`, v0.35.1), so it reads neither; `auto` and
    `true` are what it does anyway."""
    gaps = [g.model_dump(mode="json") for g in _engine("ollama").settings_not_honoured()]
    assert gaps == [
        {"setting": "toolChoice", "values": ["required", "function"]},
        {"setting": "parallelToolCalls", "values": [False]},
    ]
    assert _engine("openai").settings_not_honoured() == []


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
    "choice",
    ["required", {"type": "function", "function": {"name": "a"}}],
    ids=["required", "named"],
)
@pytest.mark.parametrize("explicit", [True, False], ids=["explicit", "inherited"])
@respx.mock
async def test_a_tool_choice_ollama_ignores_is_sent_and_named_not_refused(
    choice: Any, explicit: bool
) -> None:
    """Refused, an explicit one blocked every caller that sends it, and an inherited one was
    left out without a word on the wire. A setting a backend is not read by is sent as it is
    and said to be ignored (`x-eugene-plexus-ignored-settings`, `settingsNotHonoured`) --
    **named only when the caller sent it** (`callerSettings`), as the gateway names it, so an
    inherited one is sent all the same and the header says nothing of it."""
    route = _route(OLLAMA)
    engine = _engine("ollama")
    request = _ask(explicit=explicit, tools=TOOLS, toolChoice=choice)
    await engine.generate(request)
    assert _sent(route)["tool_choice"] == choice
    assert [t["function"]["name"] for t in _sent(route)["tools"]] == ["a", "b"]
    assert engine.ignored_settings(request) == (["toolChoice"] if explicit else [])


@respx.mock
async def test_tool_choice_none_is_honoured_for_ollama_by_sending_no_tools() -> None:
    """Ollama does not read `tool_choice`, so `none` would be answered as `auto`: it is honoured
    the way it can be, by not offering the tools at all (as for Strata)."""
    route = _route(OLLAMA)
    engine = _engine("ollama")
    request = _ask(tools=TOOLS, toolChoice="none", parallelToolCalls=False)
    await engine.generate(request)
    sent = _sent(route)
    assert "tools" not in sent and "tool_choice" not in sent and "parallel_tool_calls" not in sent
    assert engine.ignored_settings(request) == [], "it was honoured, so nothing is named"


@respx.mock
async def test_tool_choice_none_still_reaches_llama_server_which_reads_it() -> None:
    route = _route(LLAMA)
    await _engine("openai").generate(_ask(tools=TOOLS, toolChoice="none"))
    assert _sent(route)["tool_choice"] == "none"
    assert len(_sent(route)["tools"]) == 2


@respx.mock
async def test_auto_and_parallel_true_are_what_ollama_does_and_still_work() -> None:
    route = _route(OLLAMA)
    engine = _engine("ollama")
    request = _ask(tools=TOOLS, toolChoice="auto", parallelToolCalls=True)
    await engine.generate(request)
    assert _sent(route)["tool_choice"] == "auto"
    assert _sent(route)["parallel_tool_calls"] is True
    assert engine.ignored_settings(request) == []


@respx.mock
async def test_one_call_a_turn_is_sent_to_both_and_named_for_ollama_only() -> None:
    ollama, llama = _route(OLLAMA), _route(LLAMA)
    request = _ask(tools=TOOLS, parallelToolCalls=False)
    ollama_engine, llama_engine = _engine("ollama"), _engine("openai")
    await ollama_engine.generate(request)
    await llama_engine.generate(request)
    assert _sent(ollama)["parallel_tool_calls"] is False
    assert _sent(llama)["parallel_tool_calls"] is False
    assert ollama_engine.ignored_settings(request) == ["parallelToolCalls"]
    assert llama_engine.ignored_settings(request) == []


@respx.mock
async def test_a_codex_shaped_request_is_served_by_ollama_with_its_gaps_named() -> None:
    """`tool_choice: auto` and `parallel_tool_calls: false` on every request, all explicit."""
    route = _route(OLLAMA)
    engine = _engine("ollama")
    request = _ask(tools=TOOLS, toolChoice="auto", parallelToolCalls=False)
    await engine.generate(request)
    assert route.called
    assert engine.ignored_settings(request) == ["parallelToolCalls"]


@respx.mock
def test_through_the_app_an_ollama_account_reports_and_names_what_it_ignores(
    tmp_path: Path,
) -> None:
    """The driver the agent runs for an Ollama runtime: `/v1/info` carries the static list on
    every chat model, and a request that trips one is answered with the header, not a 400."""
    respx.get(f"{OLLAMA}/api/tags").mock(
        return_value=httpx.Response(
            200, json={"models": [{"name": "qwen3-coder:30b"}, {"name": "nomic-embed-text:latest"}]}
        )
    )
    shows = {"qwen3-coder:30b": ["completion", "tools"], "nomic-embed-text:latest": ["embedding"]}
    respx.post(f"{OLLAMA}/api/show").mock(
        side_effect=lambda r: httpx.Response(
            200, json={"capabilities": shows[json.loads(r.content)["model"]]}
        )
    )
    chat = _route(OLLAMA)
    config = tmp_path / "ollama.yaml"
    config.write_text(json.dumps({"provider": "ollama_local"}), "utf-8")
    with TestClient(create_app(settings=Settings(config_file=config))) as client:
        info = _wait_for_catalogue(client)
        by_id = {m["id"]: m["capabilities"] for m in info["models"]}
        assert by_id["qwen3-coder:30b"]["settingsNotHonoured"] == [
            {"setting": "toolChoice", "values": ["required", "function"]},
            {"setting": "parallelToolCalls", "values": [False]},
        ]
        assert "settingsNotHonoured" not in by_id["nomic-embed-text:latest"], "it does not chat"
        response = client.post(
            "/v1/generate",
            json={
                "model": "qwen3-coder:30b",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": TOOLS,
                "toolChoice": "required",
                "parallelToolCalls": False,
                "callerSettings": ["tools", "toolChoice", "parallelToolCalls"],
            },
        )
    assert response.status_code == 200, response.text
    assert response.headers["x-eugene-plexus-ignored-settings"] == "toolChoice, parallelToolCalls"
    assert _sent(chat)["tool_choice"] == "required"
    assert _sent(chat)["parallel_tool_calls"] is False


@respx.mock
def test_through_the_app_an_ollama_model_without_tools_is_not_sent_what_rides_with_them(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """One rule for any model whose tools are off, learned or listed (round four). `/api/show`
    lists no `tools` among this model's capabilities, which is Ollama's own word and
    `toolCalling` false. Codex sends `tool_choice` and `parallel_tool_calls` on every request,
    tools or not; Ollama has no field for either (read from its source), so on a request that
    carries no tools they were sent for nothing. They are left out now, once per field, with
    the cause in words; a request that does carry tools is still the door's refusal."""
    caplog.set_level("WARNING", logger="eugene_plexus_inference_driver.engines.openai_compat_http")
    respx.get(f"{OLLAMA}/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": "gemma3:4b"}]})
    )
    respx.post(f"{OLLAMA}/api/show").mock(
        return_value=httpx.Response(200, json={"capabilities": ["completion"]})
    )
    chat = _route(OLLAMA)
    config = tmp_path / "ollama.yaml"
    config.write_text(json.dumps({"provider": "ollama_local"}), "utf-8")
    plain = {
        "model": "gemma3:4b",
        "messages": [{"role": "user", "content": "hi"}],
        "toolChoice": "auto",
        "parallelToolCalls": False,
        "callerSettings": ["toolChoice", "parallelToolCalls"],
    }
    with TestClient(create_app(settings=Settings(config_file=config))) as client:
        info = _wait_for_catalogue(client)
        (model,) = info["models"]
        assert model["capabilities"]["toolCalling"] is False
        for _ in range(2):
            served = client.post("/v1/generate", json=plain)
            assert served.status_code == 200, served.text
        assert not {"tools", "tool_choice", "parallel_tool_calls"} & _sent(chat).keys()

        calls = len(chat.calls)
        refused = client.post(
            "/v1/generate",
            json={**plain, "tools": TOOLS, "callerSettings": ["tools", "toolChoice"]},
        )
        assert refused.status_code == 400
        assert refused.json()["detail"]["type"].endswith("#tools-unsupported")
        assert len(chat.calls) == calls, "and nothing was sent"
    left_out = [r.getMessage() for r in caplog.records if "omitting `" in r.getMessage()]
    assert sum("`tool_choice`" in m for m in left_out) == 1, left_out
    assert sum("`parallel_tool_calls`" in m for m in left_out) == 1, left_out
    assert all("Ollama's own capabilities" in m and "do not list tools" in m for m in left_out)


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
