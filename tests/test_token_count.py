"""Counting a request's prompt with the backend's own tokenizer, 2026-09-23.

For the gateway's Anthropic `count_tokens`. Claude Code's `/context`
sends fourteen of them, and without an answer it counts by generating:
a real one-token request per category, each a full prefill. So this
counts WITHOUT generating, and only where the count is exact.

**Measured before it was written.** llama.cpp b10948's `/apply-template`
renders the chat-completions body into the prompt the model would see,
and `/tokenize` with `add_special: true` counts it; against the
`prompt_tokens` of a one-token generation of the same request that was
exact on two template families -- Gemma 4 E4B 27/27, 59/59, 95/95
(plain, tools, a tool loop) and Qwen3-0.6B 24/24, 147/147, 21/21 (plain,
tools, a history turn carrying reasoning). The fixtures are those raw
bodies. An image renders as a `<__media_...__>` marker (captured), so a
request with one cannot be counted this way and is refused.
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
    GenerateRequest,
    Message,
    Role,
    Tool,
)
from eugene_plexus_inference_driver.engines._subprocess import CliError
from eugene_plexus_inference_driver.engines.base import TokenCountUnsupported
from eugene_plexus_inference_driver.engines.openai_compat_http import OpenAiCompatibleHttpEngine

BASE = "http://127.0.0.1:9181"
FIXTURES = Path(__file__).parent / "fixtures"
TEMPLATE = json.loads((FIXTURES / "llama_cpp_apply_template_tools.json").read_text("utf-8"))
TOKENS = json.loads((FIXTURES / "llama_cpp_tokenize_tools.json").read_text("utf-8"))
WEATHER = Tool.model_validate(
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
)
PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg=="


def _engine(**kwargs: Any) -> OpenAiCompatibleHttpEngine:
    kwargs.setdefault("base_url", BASE)
    kwargs.setdefault("model_id", "gemma-4-e4b")
    return OpenAiCompatibleHttpEngine(api_key="sk-stub", **kwargs)


def _ask(**kwargs: Any) -> GenerateRequest:
    kwargs.setdefault("messages", [Message(role=Role.user, content="Weather in Oslo?")])
    kwargs.setdefault("tools", [WEATHER])
    return GenerateRequest(**kwargs)


def _body(route: respx.Route, index: int = 0) -> dict[str, Any]:
    return json.loads(route.calls[index].request.read())


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #


@respx.mock
async def test_the_count_is_the_backends_own_template_and_tokenizer() -> None:
    """The reproduction: before this, nothing here could count without generating."""
    template = respx.post(f"{BASE}/apply-template").mock(
        return_value=httpx.Response(200, json=TEMPLATE)
    )
    tokenize = respx.post(f"{BASE}/tokenize").mock(return_value=httpx.Response(200, json=TOKENS))
    chat = respx.post(f"{BASE}/v1/chat/completions")

    count = await _engine().count_prompt_tokens(_ask())

    assert count == len(TOKENS["tokens"]) == 59
    assert template.call_count == 1
    assert not chat.called, "a count must never generate"
    # The prompt the template rendered is what gets counted, with the
    # special tokens the completion path adds (BOS on Gemma: 58 without).
    assert _body(tokenize) == {"content": TEMPLATE["prompt"], "add_special": True}


@respx.mock
@pytest.mark.parametrize("thinking_mode", ["auto", "off"])
async def test_the_template_is_given_exactly_what_a_generation_would_send(
    thinking_mode: str,
) -> None:
    """Same payload, thinking directive and all -- a count of any other
    prompt is a count of a request nobody is going to make."""
    template = respx.post(f"{BASE}/apply-template").mock(
        return_value=httpx.Response(200, json=TEMPLATE)
    )
    respx.post(f"{BASE}/tokenize").mock(return_value=httpx.Response(200, json=TOKENS))
    engine = _engine(thinking_mode=thinking_mode)
    request = _ask(
        messages=[
            Message(role=Role.system, content="You are terse."),
            Message(role=Role.user, content="Weather in Oslo?"),
        ]
    )

    await engine.count_prompt_tokens(request)

    sent = _body(template)
    expected = engine._payload_for(request)
    assert sent["messages"] == expected["messages"]
    assert sent["tools"] == expected["tools"]


@respx.mock
async def test_an_image_is_not_counted_because_the_template_only_marks_it() -> None:
    template = respx.post(f"{BASE}/apply-template")
    request = _ask(
        messages=[
            Message.model_validate(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What colour?"},
                        {"type": "image_url", "image_url": {"url": PNG}},
                    ],
                }
            )
        ]
    )
    with pytest.raises(TokenCountUnsupported, match="image"):
        await _engine().count_prompt_tokens(request)
    assert not template.called


@respx.mock
@pytest.mark.parametrize("status", [404, 405])
async def test_a_backend_without_a_template_endpoint_cannot_count(status: int) -> None:
    """vLLM, Ollama, LM Studio: not asked to guess."""
    respx.post(f"{BASE}/apply-template").mock(return_value=httpx.Response(status, text="nope"))
    tokenize = respx.post(f"{BASE}/tokenize")
    with pytest.raises(TokenCountUnsupported):
        await _engine().count_prompt_tokens(_ask())
    assert not tokenize.called


@respx.mock
async def test_openais_own_endpoint_is_never_asked() -> None:
    template = respx.post("https://api.openai.com/apply-template")
    with pytest.raises(TokenCountUnsupported):
        await _engine(base_url="https://api.openai.com", model_id="gpt-4o").count_prompt_tokens(
            _ask()
        )
    assert not template.called


@respx.mock
async def test_a_backend_that_cannot_be_reached_is_a_backend_error() -> None:
    """Not "cannot count": the backend that would have counted is down,
    which the caller should hear as that."""
    respx.post(f"{BASE}/apply-template").mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(CliError) as caught:
        await _engine().count_prompt_tokens(_ask())
    assert not isinstance(caught.value, TokenCountUnsupported)


@respx.mock
async def test_a_template_refusal_is_the_backends_own_4xx() -> None:
    """A request the template rejects would be rejected by a generation
    too, in the same words -- it is not a backend that cannot count."""
    respx.post(f"{BASE}/apply-template").mock(
        return_value=httpx.Response(400, json={"error": {"message": "bad tool schema"}})
    )
    with pytest.raises(CliError) as caught:
        await _engine().count_prompt_tokens(_ask())
    assert not isinstance(caught.value, TokenCountUnsupported)
    assert caught.value.upstream_status == 400


# --------------------------------------------------------------------------- #
# The route
# --------------------------------------------------------------------------- #


class _CliAdapter:
    backend_kind = "claude_code_cli"
    supports_tool_calling = False


def test_the_route_answers_the_count(client: TestClient) -> None:
    with respx.mock:
        respx.post(f"{BASE}/apply-template").mock(return_value=httpx.Response(200, json=TEMPLATE))
        respx.post(f"{BASE}/tokenize").mock(return_value=httpx.Response(200, json=TOKENS))
        engine = _engine()
        engine.supports_tool_calling = True
        client.app.state.adapter = engine  # type: ignore[attr-defined]
        r = client.post(
            "/v1/generate/count",
            json=_ask().model_dump(mode="json", by_alias=True, exclude_none=True),
        )
    assert r.status_code == 200, r.text
    assert r.json() == {"promptTokens": 59}


def test_an_engine_with_no_counter_is_501(client: TestClient) -> None:
    client.app.state.adapter = _CliAdapter()  # type: ignore[attr-defined]
    r = client.post("/v1/generate/count", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 501, r.text
    assert r.json()["detail"]["status"] == 501


def test_a_backend_that_cannot_count_is_501_not_a_failure(client: TestClient) -> None:
    with respx.mock:
        respx.post(f"{BASE}/apply-template").mock(return_value=httpx.Response(404))
        client.app.state.adapter = _engine()  # type: ignore[attr-defined]
        r = client.post(
            "/v1/generate/count", json={"messages": [{"role": "user", "content": "hi"}]}
        )
    assert r.status_code == 501, r.text


def test_an_unreachable_backend_is_502(client: TestClient) -> None:
    with respx.mock:
        respx.post(f"{BASE}/apply-template").mock(side_effect=httpx.ConnectError("refused"))
        client.app.state.adapter = _engine()  # type: ignore[attr-defined]
        r = client.post(
            "/v1/generate/count", json={"messages": [{"role": "user", "content": "hi"}]}
        )
    assert r.status_code == 502, r.text


def test_local_only_is_enforced_before_anything_is_sent(client: TestClient) -> None:
    with respx.mock:
        template = respx.post("https://api.example.com/apply-template")
        client.app.state.adapter = _engine(base_url="https://api.example.com")  # type: ignore[attr-defined]
        r = client.post(
            "/v1/generate/count",
            json={"messages": [{"role": "user", "content": "hi"}], "localOnly": True},
        )
    assert r.status_code == 403, r.text
    assert not template.called
