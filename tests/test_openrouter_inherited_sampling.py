"""An inherited sampling setting an OpenRouter model's listing leaves out is dropped, not sent.

Found 2026-11-05 by an app (Knight Shift) on the Unraid install: every chat request to Claude
Haiku 5.5, Claude Sonnet 5.5 or GPT-6 Luna through OpenRouter that carried a setting of the
caller's (`max_tokens`, `tools`) failed with OpenRouter's 404 "No endpoints found that can handle
the requested parameters". The same request with no settings succeeded.

The cause: the caller's settings make Eugene ask OpenRouter for `require_parameters`
(`_asks_require_parameters`), so OpenRouter routes only to providers that take every parameter
in the request. These models' listings name no `temperature` or `top_p` (OpenRouter's public
list, 2026-11-06: Claude 5.5 also names no `seed`, GPT-6 no `stop`; 112 of 453 models name no
`temperature`). The gateway's default temperature arrives filled in, not in `callerSettings`, and
`_payload_for` checked only `temperature_fixed` before sending it, so the request carried a
parameter no provider of the model takes, and there was nowhere to route it.

The rule is the one the neighbours (`top_k`, `min_p`, the penalties) already follow: a setting
the listing leaves out is unsupported (`_unsupported_settings`). **Explicit** (in
`callerSettings`), it is refused with a 400 before anything is sent. **Inherited**, it is left
out of the payload and said once in the log. A model whose listing names the setting is sent it.
"""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from eugene_plexus_inference_driver._generated.models import BackendKind, GenerateRequest
from eugene_plexus_inference_driver.app import create_app
from eugene_plexus_inference_driver.engines.openai_compat_http import OpenAiCompatibleHttpEngine
from eugene_plexus_inference_driver.settings import Settings
from tests.test_accounts import _completion, _or_model, _wait_for_catalogue, _write

OPENROUTER = "https://openrouter.ai/api"
CHAT = f"{OPENROUTER}/v1/chat/completions"
ENGINE_LOG = "eugene_plexus_inference_driver.engines.openai_compat_http"
WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}
SSE_OK = (
    'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":null}]}\n\n'
    'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
    "data: [DONE]\n\n"
)
_COMMON = [
    "include_reasoning",
    "max_completion_tokens",
    "max_tokens",
    "reasoning",
    "reasoning_effort",
    "response_format",
    "structured_outputs",
    "tool_choice",
    "tools",
    "verbosity",
]
#: Each model's `supported_parameters`, as OpenRouter's public list gave them on 2026-11-06.
LISTED = {
    "anthropic/claude-haiku-5.5": [*_COMMON, "stop"],
    "openai/gpt-6-luna": [*_COMMON, "seed"],
    "anthropic/claude-sonnet-4.6": [*_COMMON, "stop", "temperature", "top_k", "top_p"],
}
#: What the gateway fills in from the install's defaults or a profile: none of it the caller's.
INHERITED = {"temperature": 0.7, "topP": 0.9, "seed": 7, "stop": ["\n\nUser:"]}


def _mock_listing() -> None:
    models = [_or_model(name, supported_parameters=params) for name, params in LISTED.items()]
    respx.get(f"{OPENROUTER}/v1/models/user").mock(
        return_value=httpx.Response(200, json={"data": models})
    )
    for kind in ("images", "videos"):
        respx.get(f"{OPENROUTER}/v1/{kind}/models").mock(
            return_value=httpx.Response(200, json={"data": []})
        )


def _answer(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    if body.get("stream"):
        return httpx.Response(200, text=SSE_OK, headers={"content-type": "text/event-stream"})
    return httpx.Response(200, json=_completion(body["model"]))


async def _account() -> OpenAiCompatibleHttpEngine:
    """An OpenRouter account as the registry builds one (`providers.py`), its list read."""
    _mock_listing()
    engine = OpenAiCompatibleHttpEngine(
        api_key="sk-or-test",
        base_url=OPENROUTER,
        model_id=None,
        backend_kind=BackendKind.openai_compat_http,
        dialect="openrouter",
        catalogue_source="openrouter",
        filter_models=False,
        require_parameters=True,
    )
    assert engine.catalogue is not None
    assert await engine.catalogue.refresh()
    return engine


def _ask(model: str, **fields: Any) -> GenerateRequest:
    return GenerateRequest.model_validate(
        {
            "model": model,
            "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
            "tools": [WEATHER],
            "maxTokens": 512,
            **INHERITED,
            "callerSettings": ["maxTokens", "tools"],
            **fields,
        }
    )


@respx.mock
@pytest.mark.anyio
@pytest.mark.parametrize(
    ("model", "left_out"),
    [
        ("anthropic/claude-haiku-5.5", {"temperature", "top_p", "seed"}),
        ("openai/gpt-6-luna", {"temperature", "top_p", "stop"}),
    ],
)
async def test_an_inherited_setting_the_listing_leaves_out_is_not_sent(
    model: str, left_out: set[str], caplog: pytest.LogCaptureFixture
) -> None:
    engine = await _account()
    try:
        with caplog.at_level(logging.WARNING, logger=ENGINE_LOG):
            payload = engine._payload_for(_ask(model), engine.resolve_model(model))
        assert payload["provider"] == {"require_parameters": True}
        assert not left_out & payload.keys(), f"sent what {model}'s listing leaves out"
        sent = {"temperature", "top_p", "seed", "stop"} - left_out
        assert sent <= payload.keys(), "a setting the listing names is still sent"
        assert payload["max_completion_tokens"] == 512 and payload["tools"]
        for key in left_out:
            assert any(f"`{key}`" in r.getMessage() for r in caplog.records), key
    finally:
        await engine.aclose()


@respx.mock
@pytest.mark.anyio
async def test_a_model_whose_listing_names_the_settings_is_sent_them() -> None:
    engine = await _account()
    try:
        model = "anthropic/claude-sonnet-4.6"
        payload = engine._payload_for(_ask(model), engine.resolve_model(model))
        assert payload["temperature"] == 0.7 and payload["top_p"] == 0.9
        assert payload["stop"] == ["\n\nUser:"]
        assert "seed" not in payload, "its listing names no seed"
    finally:
        await engine.aclose()


@respx.mock
@pytest.mark.anyio
@pytest.mark.parametrize(
    ("field", "wire"),
    [("temperature", "temperature"), ("topP", "top_p"), ("seed", "seed")],
)
async def test_an_explicit_setting_the_listing_leaves_out_is_still_refused(
    field: str, wire: str
) -> None:
    from eugene_plexus_inference_driver.engines._subprocess import CliError

    engine = await _account()
    try:
        model = "anthropic/claude-haiku-5.5"
        request = _ask(model, callerSettings=["maxTokens", "tools", field])
        with pytest.raises(CliError, match=wire) as refused:
            engine._payload_for(request, engine.resolve_model(model))
        assert refused.value.upstream_status == 400
    finally:
        await engine.aclose()


@contextlib.contextmanager
def _driver(tmp_path: Path) -> Iterator[tuple[TestClient, respx.Route]]:
    """The real app for an OpenRouter account (which asks for `require_parameters`)."""
    _mock_listing()
    chat = respx.post(CHAT).mock(side_effect=_answer)
    config = tmp_path / "openrouter.yaml"
    _write(config, {"provider": "openrouter", "apiKey": "sk-or-test"})
    with TestClient(create_app(settings=Settings(config_file=config))) as client:
        _wait_for_catalogue(client)
        yield client, chat


@respx.mock
@pytest.mark.parametrize("stream", [False, True])
def test_the_app_sends_no_inherited_temperature_with_require_parameters(
    tmp_path: Path, stream: bool
) -> None:
    """Knight Shift's request, as the gateway hands it on: streamed, the caller's `max_tokens`
    and `tools`, the install's default temperature filled in."""
    with _driver(tmp_path) as (client, chat):
        response = client.post(
            "/v1/generate/stream" if stream else "/v1/generate",
            json={
                "model": "anthropic/claude-haiku-5.5",
                "messages": [{"role": "user", "content": "Your move."}],
                "tools": [WEATHER],
                "maxTokens": 512,
                "temperature": 0.7,
                "callerSettings": ["maxTokens", "tools"],
            },
        )
        assert response.status_code == 200, response.text
        sent = json.loads(chat.calls[-1].request.content)
    assert sent["provider"] == {"require_parameters": True}
    assert "temperature" not in sent
