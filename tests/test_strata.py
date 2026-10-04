"""Strata must not acquire llama.cpp-only semantics through /props."""

from __future__ import annotations

import pytest
import respx

from eugene_plexus_inference_driver._generated.models import GenerateRequest, Message, Role
from eugene_plexus_inference_driver.engines._subprocess import CliError
from eugene_plexus_inference_driver.engines.openai_compat_http import OpenAiCompatibleHttpEngine
from eugene_plexus_inference_driver.engines.strata_http import StrataHttpEngine

BASE = "http://127.0.0.1:8095"


def test_supported_settings_exclude_unimplemented_features():
    engine = StrataHttpEngine(base_url=BASE, model_id="qwen", auth_required=False)
    assert not engine.supports_tool_calling
    assert {"temperature", "maxTokens", "topK"} <= set(engine.supported_settings)
    assert not {"tools", "toolChoice", "responseFormat", "seed", "logprobs"} & set(
        engine.supported_settings
    )


@pytest.mark.anyio
@respx.mock
async def test_context_and_response_with_cached_usage_and_identity():
    engine = StrataHttpEngine(base_url=BASE, model_id="qwen-small", auth_required=False)
    respx.get(f"{BASE}/health").respond(
        200, json={"service": "strata", "loaded": True, "max_context": 8192}
    )
    chat = respx.post(f"{BASE}/v1/chat/completions").respond(
        200,
        json={
            "id": "test",
            "model": "qwen-small",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "hello",
                        "reasoning_content": "thinking",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 4,
                "total_tokens": 16,
                "prompt_tokens_details": {"cached_tokens": 8},
            },
        },
    )
    assert await engine.context_window() == 8192
    assert not await engine._answers_as_llama_cpp()
    result = await engine.generate(
        GenerateRequest(messages=[Message(role=Role.user, content="hi")])
    )
    assert "hello" in str(result)
    import json

    sent = json.loads(chat.calls[0].request.content)
    assert sent["model"] == "qwen-small"
    assert not {"id_slot", "return_progress"} & sent.keys()
    await engine.aclose()


def test_explicit_seed_is_refused_before_http():
    engine = StrataHttpEngine(base_url=BASE, model_id="qwen", auth_required=False)
    request = GenerateRequest(
        messages=[Message(role=Role.user, content="hi")], seed=1, callerSettings=["seed"]
    )
    with pytest.raises(CliError):
        engine._payload_for(request, engine.resolve_model(None))


@pytest.mark.anyio
@respx.mock
async def test_generic_props_probe_does_not_mistake_strata_for_llama_cpp():
    respx.get(f"{BASE}/props").respond(
        200, json={"build_info": "Strata 0.1.39", "default_generation_settings": {"n_ctx": 8192}}
    )
    engine = OpenAiCompatibleHttpEngine(base_url=BASE, model_id="qwen", auth_required=False)
    assert not await engine._answers_as_llama_cpp()
    await engine.aclose()
