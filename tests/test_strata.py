"""Strata must not acquire llama.cpp-only semantics through /props."""

from __future__ import annotations

import json

import pytest
import respx

from eugene_plexus_inference_driver._generated.models import GenerateRequest, Message, Role
from eugene_plexus_inference_driver.engines._subprocess import CliError
from eugene_plexus_inference_driver.engines.openai_compat_http import OpenAiCompatibleHttpEngine
from eugene_plexus_inference_driver.engines.strata_http import StrataHttpEngine

BASE = "http://127.0.0.1:8095"


def test_supported_settings_are_what_strata_reads_and_not_what_it_has_never_shown():
    """What Strata does not take is still not offered (an explicit one is refused, A2).
    Tools, `tool_choice`, `parallel_tool_calls` and `response_format` are taken since
    2026-10-12: measured, it calls tools and answers a JSON schema (`test_strata_tools.py`).
    `parallel_tool_calls` is offered although Strata ignores `false`: that is named, not
    refused, because Codex sends it on every request. **`seed`, `frequency_penalty` and
    `presence_penalty` are taken since round four**: Strata's own source reads all three per
    request (`sampling_keys`; its documentation: "Temperature / top_p / top_k / min_p / seed are
    honored per request ... the penalties ride the same path"), and one running Strata answered
    a seed (`test_seed_and_the_penalties_are_sent_to_strata_as_asked`). `logprobs` is not."""
    engine = StrataHttpEngine(base_url=BASE, model_id="qwen", auth_required=False)
    assert engine.supports_tool_calling
    offered = set(engine.supported_settings)
    assert {
        "temperature",
        "maxTokens",
        "topK",
        "seed",
        "frequencyPenalty",
        "presencePenalty",
        "tools",
        "toolChoice",
        "parallelToolCalls",
        "responseFormat",
    } <= offered
    assert not {"logprobs", "logitBias"} & offered


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
    sent = json.loads(chat.calls[0].request.content)
    assert sent["model"] == "qwen-small"
    assert not {"id_slot", "return_progress"} & sent.keys()
    await engine.aclose()


_REPLY = {
    "id": "test",
    "model": "qwen",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
}
_ALL_THREE = ["seed", "frequencyPenalty", "presencePenalty"]


@pytest.mark.anyio
@respx.mock
async def test_seed_and_the_penalties_are_sent_to_strata_as_asked() -> None:
    """**Measured once on Troy's running Strata, 2026-10-12** (`qwen3.8-flash-next-iq2_xs`, its
    own OpenAI endpoint at 127.0.0.1:8092, directly; three requests, the same prompt, `temperature`
    0.8, `max_tokens` 48, `reasoning_effort` none): `seed` 4242 twice answered the same sentence
    both times ("The sea is a restless mirror, reflecting the sky while hiding its own secrets in
    the dark."), and `seed` 4243 another one ("The sea is a restless blue heart, beating against
    the shore with secrets it keeps for eternity."). That is one run of three requests: this
    Strata honoured a seed once, and Strata's documentation says a sampled result is not
    reproducible run to run on its default adaptive expert tier (`--adapt-every 100000` makes it
    so). Its source reads `seed`, `frequency_penalty` and `presence_penalty` per request, so they
    are carried and not refused, as an explicit setting an engine can carry must be (A2).

    `seed` 0 is carried too (R3.4: it is a real seed, and falsy). Strata's own source forwards a
    positive one only (`seed > 0`, read and not measured), so it takes a 0 as no seed: sent, and
    not nameable as not honoured, because `settingsNotHonoured` names values that are strings or
    booleans."""
    route = respx.post(f"{BASE}/v1/chat/completions").respond(200, json=_REPLY)
    engine = StrataHttpEngine(base_url=BASE, model_id="qwen", auth_required=False)
    try:
        for seed in (4242, 0):
            request = GenerateRequest(
                messages=[Message(role=Role.user, content="hi")],
                seed=seed,
                frequencyPenalty=0.4,
                presencePenalty=0.2,
                callerSettings=_ALL_THREE,
            )
            await engine.generate(request)
            sent = json.loads(route.calls[-1].request.content)
            assert sent["seed"] == seed
            assert sent["frequency_penalty"] == 0.4 and sent["presence_penalty"] == 0.2
            assert engine.ignored_settings(request) == []
    finally:
        await engine.aclose()


def test_logprobs_is_still_refused_before_http() -> None:
    """Nothing has shown Strata returns log probabilities, so an explicit one is refused (A2)
    and never sent and ignored."""
    engine = StrataHttpEngine(base_url=BASE, model_id="qwen", auth_required=False)
    request = GenerateRequest(
        messages=[Message(role=Role.user, content="hi")], logprobs=True, callerSettings=["logprobs"]
    )
    with pytest.raises(CliError) as refused:
        engine._payload_for(request, engine.resolve_model(None))
    assert refused.value.upstream_status == 400


@pytest.mark.anyio
@respx.mock
async def test_generic_props_probe_does_not_mistake_strata_for_llama_cpp():
    respx.get(f"{BASE}/props").respond(
        200, json={"build_info": "Strata 0.1.39", "default_generation_settings": {"n_ctx": 8192}}
    )
    engine = OpenAiCompatibleHttpEngine(base_url=BASE, model_id="qwen", auth_required=False)
    assert not await engine._answers_as_llama_cpp()
    await engine.aclose()
