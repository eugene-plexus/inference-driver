"""Strata must not acquire llama.cpp-only semantics through /props."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from eugene_plexus_inference_driver._generated.models import GenerateRequest, Message, Role
from eugene_plexus_inference_driver.app import create_app
from eugene_plexus_inference_driver.engines._subprocess import CliError
from eugene_plexus_inference_driver.engines.openai_compat_http import OpenAiCompatibleHttpEngine
from eugene_plexus_inference_driver.engines.strata_http import StrataHttpEngine
from eugene_plexus_inference_driver.settings import Settings

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

    Only a positive seed: Strata's own source forwards one only when it is a positive whole number
    (`seed > 0`, read and not measured), so a seed of 0 or below is refused in words that name it,
    and an inherited one is left out (`test_an_explicit_seed_strata_would_ignore_is_refused...`)."""
    route = respx.post(f"{BASE}/v1/chat/completions").respond(200, json=_REPLY)
    engine = StrataHttpEngine(base_url=BASE, model_id="qwen", auth_required=False)
    try:
        for seed in (4242, 1):
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


def _strata_app(tmp_path: Path) -> TestClient:
    """The real app, configured as an install configures a Strata driver."""
    config = tmp_path / "strata.yaml"
    config.write_text(
        json.dumps({"provider": "strata_local", "baseUrl": BASE, "modelId": "qwen"}), "utf-8"
    )
    return TestClient(create_app(settings=Settings(config_file=config)))


def _chat(request: httpx.Request) -> httpx.Response:
    """What Strata answers: a stream's frames when the request asked for one."""
    if json.loads(request.content).get("stream"):
        frames = (
            'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}\n\n'
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, text=frames, headers={"content-type": "text/event-stream"})
    return httpx.Response(200, json=_REPLY)


async def _serve(engine: StrataHttpEngine, request: GenerateRequest, stream: bool) -> None:
    if stream:
        _ = [chunk async for chunk in engine.stream(request)]
    else:
        await engine.generate(request)


@pytest.mark.anyio
@respx.mock
@pytest.mark.parametrize("stream", [False, True], ids=["generate", "stream"])
@pytest.mark.parametrize("seed", [0, -1, -4242])
async def test_an_explicit_seed_strata_would_ignore_is_refused_in_words_that_name_it(
    seed: int, stream: bool
) -> None:
    """Strata forwards a seed only when it is a positive whole number: its `sampling_keys` takes
    `seed` when `seed > 0`, and its own config check says "expected a positive integer". For 0 or
    below it answers WITHOUT one, and says nothing. A caller that asked for a seed asked for a
    reproducible answer and would be handed a different one each time, so an explicit one that
    Strata would ignore is refused, before anything is sent, naming the value (a seed is not a
    setting that can be "named as not honoured": that list names strings and booleans)."""
    route = respx.post(f"{BASE}/v1/chat/completions").mock(side_effect=_chat)
    engine = StrataHttpEngine(base_url=BASE, model_id="qwen", auth_required=False)
    request = GenerateRequest(
        messages=[Message(role=Role.user, content="hi")], seed=seed, callerSettings=["seed"]
    )
    try:
        with pytest.raises(CliError) as refused:
            await _serve(engine, request, stream)
    finally:
        await engine.aclose()
    assert refused.value.upstream_status == 400
    words = str(refused.value)
    assert f"seed: {seed} " in words, "the value is named"
    assert "positive whole number" in words and "would ignore" in words
    assert "remove the seed" in words, "and what to do"
    assert not route.called, "refused before anything was sent"


@respx.mock
@pytest.mark.parametrize("path", ["/v1/generate", "/v1/generate/stream"])
def test_through_the_app_a_seed_of_zero_is_the_callers_400_and_nothing_is_sent(
    tmp_path: Path, path: str
) -> None:
    route = respx.post(f"{BASE}/v1/chat/completions").mock(side_effect=_chat)
    body = {"messages": [{"role": "user", "content": "hi"}], "seed": 0, "callerSettings": ["seed"]}
    with _strata_app(tmp_path) as client:
        response = client.post(path, json=body)
        assert response.status_code == 400, response.text
        detail = response.json()["detail"]
        assert detail["type"].endswith("#backend-rejected-request")
        assert detail["retryDisposition"] == "terminal"
        assert "seed: 0 " in detail["detail"] and "positive whole number" in detail["detail"]
        # A positive one is served, on the same driver.
        served = client.post(path, json={**body, "seed": 7})
        assert served.status_code == 200, served.text
    assert route.call_count == 1
    assert json.loads(route.calls[0].request.content)["seed"] == 7


@pytest.mark.anyio
@respx.mock
@pytest.mark.parametrize("seed", [0, -7])
async def test_an_inherited_seed_strata_would_ignore_is_left_out_and_said_once(
    seed: int, caplog: pytest.LogCaptureFixture
) -> None:
    """A profile's seed was not asked for by this caller, so it is not refused (A2): it is left
    out, where Strata would have ignored it, and said once. A positive one is sent as ever."""
    caplog.set_level(logging.WARNING, logger="eugene_plexus_inference_driver.engines.strata_http")
    route = respx.post(f"{BASE}/v1/chat/completions").mock(side_effect=_chat)
    engine = StrataHttpEngine(base_url=BASE, model_id="qwen", auth_required=False)
    try:
        inherited = GenerateRequest(
            messages=[Message(role=Role.user, content="hi")], seed=seed, temperature=0.7
        )
        for _ in range(3):
            await engine.generate(inherited)
        assert route.call_count == 3
        assert all("seed" not in json.loads(c.request.content) for c in route.calls)
        assert json.loads(route.calls[-1].request.content)["temperature"] == 0.7

        await engine.generate(
            GenerateRequest(messages=[Message(role=Role.user, content="hi")], seed=4242)
        )
        assert json.loads(route.calls[-1].request.content)["seed"] == 4242
    finally:
        await engine.aclose()
    said = [r.getMessage() for r in caplog.records if "seed" in r.getMessage()]
    assert len(said) == 1, said
    assert str(seed) in said[0] and "positive whole number" in said[0]


@pytest.mark.anyio
@respx.mock
@pytest.mark.parametrize("stream", [False, True], ids=["generate", "stream"])
async def test_negative_penalties_are_sent_to_strata_as_asked(stream: bool) -> None:
    """Strata's own per-request path reads a penalty that is not zero, a negative one included
    (`sampling_keys`: `penalty_freq` and `penalty_present` follow `!= 0.0`). The "expected a
    number >= 0" in its source is the check of a default written in its CONFIG FILE, not of a
    request. So a negative penalty is sent, not refused here on a rule that is not Strata's."""
    route = respx.post(f"{BASE}/v1/chat/completions").mock(side_effect=_chat)
    engine = StrataHttpEngine(base_url=BASE, model_id="qwen", auth_required=False)
    request = GenerateRequest(
        messages=[Message(role=Role.user, content="hi")],
        frequencyPenalty=-0.5,
        presencePenalty=-1.25,
        callerSettings=["frequencyPenalty", "presencePenalty"],
    )
    try:
        await _serve(engine, request, stream)
    finally:
        await engine.aclose()
    sent = json.loads(route.calls[0].request.content)
    assert sent["frequency_penalty"] == -0.5 and sent["presence_penalty"] == -1.25
    assert engine.ignored_settings(request) == []


#: A stand-in for whatever Strata says if it does refuse a penalty: its words are not known here
#: (none was measured), and are not this driver's to write. Whatever they are, they are relayed.
STRATA_REFUSAL = {
    "error": {"type": "invalid_request_error", "message": "stand-in for Strata's own refusal"}
}


@respx.mock
@pytest.mark.parametrize("path", ["/v1/generate", "/v1/generate/stream"])
def test_a_refusal_strata_gives_of_a_penalty_is_relayed_in_its_words(
    tmp_path: Path, path: str
) -> None:
    respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(400, json=STRATA_REFUSAL)
    )
    body = {
        "messages": [{"role": "user", "content": "hi"}],
        "frequencyPenalty": -0.5,
        "callerSettings": ["frequencyPenalty"],
    }
    with _strata_app(tmp_path) as client:
        response = client.post(path, json=body)
    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert detail["type"].endswith("#backend-rejected-request")
    assert "stand-in for Strata's own refusal" in detail["detail"]


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
