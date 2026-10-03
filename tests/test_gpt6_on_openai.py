"""GPT-6 on OpenAI's own API: a fixed sampler, and tools only on Responses.

Upstream drift audit, 2026-10-03 (Docs: OpenAI's "Using GPT-6"):

* `temperature` and `top_p` are a 400 on every GPT-6 model, as on the gpt-5
  family -- and `OPENAI_FIXED_TEMPERATURE_PATTERN` matched `gpt-5` alone, so
  the driver sent both. The pattern now covers gpt-5 through gpt-9.
* `logprobs` is accepted only with `reasoning_effort: none`, so for a
  fixed-sampler model it is refused when explicit (or dropped when it came
  from a default) unless the request itself says `none`.
* `gpt-6-astra*` and `gpt-6.1-sol*` take tools only on the Responses API,
  and this engine speaks Chat Completions, so neither is advertised as
  tool-capable on OpenAI direct. Through OpenRouter, whose listing says per
  model, nothing changes.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from eugene_plexus_inference_driver._generated.models import BackendKind, GenerateRequest
from eugene_plexus_inference_driver.engines._catalogue import from_openai_list
from eugene_plexus_inference_driver.engines._subprocess import CliError
from eugene_plexus_inference_driver.engines.openai_compat_http import (
    OPENAI_FIXED_TEMPERATURE_PATTERN,
    OpenAiCompatibleHttpEngine,
)

OPENAI = "https://api.openai.com"
OK = {
    "choices": [{"message": {"role": "assistant", "content": "Red"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
}
TOOL = {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}


def _openai(model_id: str | None) -> OpenAiCompatibleHttpEngine:
    return OpenAiCompatibleHttpEngine(
        api_key="sk-test",
        base_url=OPENAI,
        model_id=model_id,
        fixed_temperature_pattern=OPENAI_FIXED_TEMPERATURE_PATTERN,
        backend_kind=BackendKind.openai_api,
    )


def _ask(model: str, **fields: Any) -> GenerateRequest:
    explicit = [k for k in fields if k not in ("topLogprobs",)]
    return GenerateRequest.model_validate(
        {
            "model": model,
            "messages": [{"role": "user", "content": "Name a colour."}],
            "callerSettings": explicit or None,
            **fields,
        }
    )


@pytest.mark.parametrize(
    "model",
    [
        "gpt-6-astra",
        "gpt-6.1-sol",
        "gpt-6-sol",
        "gpt-6-luna",
        "gpt-7",
        "gpt-9.2-mini",
        "gpt-5.6-sol",
    ],
)
def test_gpt5_through_gpt9_fix_the_sampler(model: str) -> None:
    assert OPENAI_FIXED_TEMPERATURE_PATTERN.match(model)


@pytest.mark.parametrize("model", ["gpt-4o", "gpt-4.1", "gpt-50", "gpt-5o", "gpt-10", "gpt-60"])
def test_other_names_do_not(model: str) -> None:
    assert not OPENAI_FIXED_TEMPERATURE_PATTERN.match(model)


@respx.mock
async def test_gpt6_is_never_sent_temperature_or_top_p() -> None:
    route = respx.post(f"{OPENAI}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=OK)
    )
    engine = _openai("gpt-6-sol")
    # From a profile or the install default: not the caller's, so dropped.
    await engine.generate(
        GenerateRequest.model_validate(
            {"messages": [{"role": "user", "content": "hi"}], "temperature": 0.7, "topP": 0.9}
        )
    )
    sent = json.loads(route.calls[0].request.content)
    assert "temperature" not in sent and "top_p" not in sent
    assert "temperature" not in engine.supported_settings
    assert "topP" not in engine.supported_settings


@respx.mock
async def test_explicit_logprobs_on_gpt6_is_refused_unless_effort_is_none() -> None:
    route = respx.post(f"{OPENAI}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=OK)
    )
    engine = _openai("gpt-6-sol")
    with pytest.raises(CliError) as refused:
        await engine.generate(_ask("gpt-6-sol", logprobs=True, topLogprobs=2))
    assert refused.value.upstream_status == 400
    assert "logprobs" in str(refused.value) and "reasoning_effort" in str(refused.value)
    assert not route.called

    for effort in ("low", "max"):
        with pytest.raises(CliError):
            await engine.generate(_ask("gpt-6-sol", logprobs=True, reasoningEffort=effort))
    assert not route.called

    # The one combination OpenAI takes.
    await engine.generate(_ask("gpt-6-sol", logprobs=True, topLogprobs=2, reasoningEffort="none"))
    sent = json.loads(route.calls[0].request.content)
    assert sent["logprobs"] is True and sent["top_logprobs"] == 2
    assert sent["reasoning_effort"] == "none"


@respx.mock
async def test_implicit_logprobs_on_gpt6_is_dropped_not_sent() -> None:
    route = respx.post(f"{OPENAI}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=OK)
    )
    request = GenerateRequest.model_validate(
        {"messages": [{"role": "user", "content": "hi"}], "logprobs": True, "topLogprobs": 2}
    )
    await _openai("gpt-6-sol").generate(request)
    sent = json.loads(route.calls[0].request.content)
    assert "logprobs" not in sent and "top_logprobs" not in sent


@respx.mock
async def test_a_tunable_model_keeps_logprobs_whatever_its_effort() -> None:
    """The pair that tells the fix from the over-correction."""
    route = respx.post(f"{OPENAI}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=OK)
    )
    await _openai("gpt-4.1").generate(_ask("gpt-4.1", logprobs=True, topLogprobs=2))
    sent = json.loads(route.calls[0].request.content)
    assert sent["logprobs"] is True and sent["top_logprobs"] == 2


@pytest.mark.parametrize("model", ["gpt-6-astra", "gpt-6-astra-2026-09-30", "gpt-6.1-sol"])
def test_responses_only_tool_models_are_not_tool_capable_on_chat(model: str) -> None:
    engine = _openai(model)
    assert engine.supports_tool_calling is False
    settings = set(engine.supported_settings)
    assert not settings & {"tools", "toolChoice", "parallelToolCalls"}


@pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-luna", "gpt-5.6-sol", "gpt-4.1"])
def test_the_other_gpt_models_keep_their_tools(model: str) -> None:
    engine = _openai(model)
    assert engine.supports_tool_calling is True
    assert {"tools", "toolChoice"} <= set(engine.supported_settings)


async def test_an_openai_account_marks_each_model_by_its_own_name() -> None:
    engine = _openai(None)
    listing = {"data": [{"id": "gpt-6-astra"}, {"id": "gpt-6.1-sol"}, {"id": "gpt-6-sol"}]}
    respx_mock = respx.mock(assert_all_called=False)
    with respx_mock:
        respx_mock.get(f"{OPENAI}/v1/models").mock(return_value=httpx.Response(200, json=listing))
        models = {m.id: m for m in await engine._fetch_catalogue()}
    for name, tools in (("gpt-6-astra", False), ("gpt-6.1-sol", False), ("gpt-6-sol", True)):
        caps = models[name].capabilities
        assert caps is not None
        assert caps.toolCalling is tools, name
        assert ("tools" in (caps.supportedSettings or [])) is tools, name


def test_the_same_name_elsewhere_is_left_to_that_provider() -> None:
    """Only OpenAI's own Chat Completions lacks the tools: a custom
    endpoint is the operator's, and OpenRouter's listing says per model."""
    engine = OpenAiCompatibleHttpEngine(
        api_key=None,
        base_url="http://127.0.0.1:9",
        model_id="gpt-6-astra",
        auth_required=False,
    )
    assert engine.supports_tool_calling is True
    defaults = engine.engine_defaults("gpt-6-astra")
    model = from_openai_list({"data": [{"id": "gpt-6-astra"}]}, defaults, classify=None)[0]
    assert model.capabilities is not None and model.capabilities.toolCalling is True
