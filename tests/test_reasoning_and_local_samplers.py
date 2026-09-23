"""Reasoning reaches the caller, and the local samplers reach the backend.

2026-09-23. Measured against llama.cpp b10948 launched with the agent's
own argv (`--model --host --port --alias`, nothing about reasoning): a
reasoning model's thinking arrives in `reasoning_content`, separately
from `content`, by default. This adapter read only `content`, so every
token of it was discarded -- and a Qwen3 that spent its `max_tokens`
thinking came back as an empty answer with `finish_reason: length` and
nothing anywhere to say why. vLLM 0.29 names the same field `reasoning`
(read from its `ChatMessage`/`DeltaMessage`), and additionally puts
`content: null` on a reasoning-only turn, which this adapter answered
with "missing string content" -- a 502 for a well-formed reply.

The llama.cpp fixtures under `tests/fixtures/` are raw bodies captured
from that server, not shapes written from the contract: a fixture
invented from a contract tests the contract.

And the other half: `top_k`, `min_p`, the two penalties and
`parallel_tool_calls` had nowhere to go on `GenerateRequest`, so the
gateway refused them. llama.cpp accepted all five on the same capture.
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
    BackendKind,
    FinishReason,
    GenerateRequest,
    GenerateResponse,
    Message,
    Role,
)
from eugene_plexus_inference_driver.engines._subprocess import CliError
from eugene_plexus_inference_driver.engines.base import Chunk, warn_dropped_sampling
from eugene_plexus_inference_driver.engines.openai_compat_http import (
    OPENAI_FIXED_TEMPERATURE_PATTERN,
    OpenAiCompatibleHttpEngine,
)

BASE = "http://127.0.0.1:9181"
FIXTURES = Path(__file__).parent / "fixtures"


def _engine(**kwargs: Any) -> OpenAiCompatibleHttpEngine:
    kwargs.setdefault("base_url", BASE)
    kwargs.setdefault("model_id", "qwen3-0.6b")
    return OpenAiCompatibleHttpEngine(api_key="sk-stub", **kwargs)


def _ask(**kwargs: Any) -> GenerateRequest:
    kwargs.setdefault("messages", [Message(role=Role.user, content="What is 2+2?")])
    return GenerateRequest(**kwargs)


def _sse(name: str) -> httpx.Response:
    return httpx.Response(
        200,
        text=(FIXTURES / name).read_text(encoding="utf-8"),
        headers={"content-type": "text/event-stream"},
    )


def _captured(name: str) -> tuple[str, str]:
    """(reasoning, content) exactly as the captured stream carried them."""
    reasoning = content = ""
    for line in (FIXTURES / name).read_text(encoding="utf-8").splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        for choice in json.loads(line[6:]).get("choices") or []:
            delta = choice.get("delta") or {}
            reasoning += delta.get("reasoning_content") or ""
            content += delta.get("content") or ""
    return reasoning, content


def _sent(route: respx.Route, index: int = 0) -> dict[str, Any]:
    return json.loads(route.calls[index].request.read())


async def _drain(engine: OpenAiCompatibleHttpEngine, request: GenerateRequest) -> list[Chunk]:
    return [chunk async for chunk in engine.stream(request)]


# --------------------------------------------------------------------------- #
# llama.cpp: reasoning_content, captured
# --------------------------------------------------------------------------- #


@respx.mock
async def test_llamacpp_reasoning_reaches_the_batch_result() -> None:
    """The captured reply: `content` is `""`, `finish_reason` is `length`,
    and the whole of what the model produced is in `reasoning_content`.
    Before the fix the result was an empty answer and nothing else."""
    body = json.loads((FIXTURES / "llamacpp_b10948_thinking_hits_length.json").read_text())
    respx.post(f"{BASE}/v1/chat/completions").mock(return_value=httpx.Response(200, json=body))

    result = await _engine().generate(_ask())

    assert result.reasoning == body["choices"][0]["message"]["reasoning_content"]
    assert result.reasoning and "2+2" in result.reasoning
    assert result.content == ""
    assert result.finishReason == FinishReason.length


@respx.mock
async def test_llamacpp_reasoning_streams_as_its_own_frames_before_the_answer() -> None:
    name = "llamacpp_b10948_thinking_then_answer.sse"
    respx.post(f"{BASE}/v1/chat/completions").mock(return_value=_sse(name))
    want_reasoning, want_content = _captured(name)

    chunks = await _drain(_engine(), _ask())
    events = [c for c in chunks if not c.done]
    kinds = ["reasoning" if c.reasoning else "text" if c.text else "other" for c in events]

    assert "".join(c.reasoning for c in events) == want_reasoning
    assert "".join(c.text for c in events) == want_content
    # A frame carries one kind, and all the thinking precedes the answer.
    assert not [c for c in events if c.reasoning and c.text]
    assert kinds.index("text") > max(i for i, k in enumerate(kinds) if k == "reasoning")
    final = chunks[-1].result
    assert final is not None
    assert final.reasoning == want_reasoning
    assert final.content == want_content


@respx.mock
async def test_thinking_that_runs_out_of_budget_is_not_an_empty_stream() -> None:
    """The measured failure, streamed: twenty-odd frames, every one of
    them reasoning, then `length`. Before the fix the caller received a
    stream with no output in it at all."""
    name = "llamacpp_b10948_thinking_hits_length.sse"
    respx.post(f"{BASE}/v1/chat/completions").mock(return_value=_sse(name))
    want_reasoning, want_content = _captured(name)
    assert want_content == "" and want_reasoning  # the capture is what we think it is

    chunks = await _drain(_engine(), _ask(maxTokens=24))

    assert "".join(c.reasoning for c in chunks if not c.done) == want_reasoning
    final = chunks[-1].result
    assert final is not None
    assert final.finishReason == FinishReason.length
    assert final.reasoning == want_reasoning


@respx.mock
async def test_cached_prompt_tokens_are_carried_and_absent_details_stay_absent() -> None:
    body = json.loads((FIXTURES / "llamacpp_b10948_thinking_hits_length.json").read_text())
    respx.post(f"{BASE}/v1/chat/completions").mock(return_value=httpx.Response(200, json=body))

    usage = (await _engine().generate(_ask())).usage

    assert usage is not None
    assert usage.cachedPromptTokens == body["usage"]["prompt_tokens_details"]["cached_tokens"]
    # llama.cpp reports no reasoning count, and none is invented.
    assert usage.reasoningTokens is None


# --------------------------------------------------------------------------- #
# vLLM: `reasoning`, a null content, stop_reason, both usage details
# --------------------------------------------------------------------------- #


def _vllm_reply(**message: Any) -> dict[str, Any]:
    """vLLM 0.29's `ChatMessage` serialised: every OpenAI field present,
    `tool_calls` an empty list by default, and `reasoning` beside them."""
    msg = {
        "role": "assistant",
        "content": None,
        "refusal": None,
        "annotations": None,
        "audio": None,
        "function_call": None,
        "tool_calls": [],
        "reasoning": None,
    }
    msg.update(message)
    finish = msg.pop("finish_reason", "length")
    stop_reason = msg.pop("stop_reason", None)
    return {
        "id": "chatcmpl-v",
        "object": "chat.completion",
        "created": 1,
        "model": "qwen3-0.6b",
        "choices": [
            {"index": 0, "message": msg, "finish_reason": finish, "stop_reason": stop_reason}
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 30,
            "total_tokens": 42,
            "prompt_tokens_details": {"cached_tokens": 8},
            "completion_tokens_details": {"reasoning_tokens": 30},
        },
    }


@respx.mock
async def test_a_vllm_reasoning_only_turn_is_an_answer_not_a_502() -> None:
    """`content: null`, no calls, only `reasoning` -- the shape vLLM
    gives a model that thought until `max_tokens`. It used to raise
    "missing string content", which the route reports as a backend
    failure and the gateway cascades on."""
    respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_vllm_reply(reasoning="VLLMTHOUGHT so far"))
    )

    result = await _engine().generate(_ask())

    assert result.reasoning == "VLLMTHOUGHT so far"
    assert result.content is None
    assert result.finishReason == FinishReason.length
    assert result.usage is not None
    assert result.usage.cachedPromptTokens == 8
    assert result.usage.reasoningTokens == 30


@respx.mock
async def test_a_reply_with_nothing_at_all_is_still_an_error() -> None:
    """The pair. Accepting a null content because reasoning was present
    must not become accepting a null content, full stop."""
    respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_vllm_reply())
    )
    with pytest.raises(CliError, match="missing string content"):
        await _engine().generate(_ask())


@respx.mock
async def test_vllm_streamed_reasoning_uses_its_own_field_name() -> None:
    frames = [
        {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]},
        {"choices": [{"index": 0, "delta": {"reasoning": "VLLM"}}]},
        {"choices": [{"index": 0, "delta": {"reasoning": "THOUGHT"}}]},
        {"choices": [{"index": 0, "delta": {"content": "4"}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    body = "".join(f"data: {json.dumps(f)}\n\n" for f in frames) + "data: [DONE]\n\n"
    respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
    )

    chunks = await _drain(_engine(), _ask())

    assert [c.reasoning for c in chunks if c.reasoning] == ["VLLM", "THOUGHT"]
    assert chunks[-1].result is not None
    assert chunks[-1].result.reasoning == "VLLMTHOUGHT"
    assert chunks[-1].result.content == "4"


@respx.mock
async def test_vllm_names_the_stop_sequence_that_ended_the_answer() -> None:
    respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json=_vllm_reply(content="1, 2, 3, 4, ", finish_reason="stop", stop_reason="5"),
        )
    )

    result = await _engine().generate(_ask(stop=["5", "END"]))

    assert result.finishReason == FinishReason.stop_sequence
    assert result.stopSequence == "5"


@pytest.mark.parametrize("stop_reason", [None, 151645, "NOT-REQUESTED"])
@respx.mock
async def test_a_stop_reason_that_names_no_requested_sequence_stays_a_stop(
    stop_reason: object,
) -> None:
    """A token id (vLLM's EOS case) or a string the caller never asked
    for is not a stop sequence, and claiming it was would put a value
    in `stop_sequence` that the caller cannot match to anything."""
    respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, json=_vllm_reply(content="done", finish_reason="stop", stop_reason=stop_reason)
        )
    )

    result = await _engine().generate(_ask(stop=["5"]))

    assert result.finishReason == FinishReason.stop
    assert result.stopSequence is None


@respx.mock
async def test_the_stop_sequence_is_named_on_the_streamed_path_too() -> None:
    frames = [
        {"choices": [{"index": 0, "delta": {"content": "1, 2"}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop", "stop_reason": "END"}]},
    ]
    body = "".join(f"data: {json.dumps(f)}\n\n" for f in frames) + "data: [DONE]\n\n"
    respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
    )

    chunks = await _drain(_engine(), _ask(stop=["END"]))

    final = chunks[-1].result
    assert final is not None
    assert final.finishReason == FinishReason.stop_sequence
    assert final.stopSequence == "END"


# --------------------------------------------------------------------------- #
# thinkingMode: off withholds it on both paths
# --------------------------------------------------------------------------- #


@respx.mock
async def test_thinking_mode_off_withholds_reasoning_on_both_paths() -> None:
    name = "llamacpp_b10948_thinking_then_answer.sse"
    route = respx.post(f"{BASE}/v1/chat/completions")
    route.side_effect = [
        _sse(name),
        httpx.Response(
            200,
            json=json.loads((FIXTURES / "llamacpp_b10948_thinking_hits_length.json").read_text()),
        ),
    ]
    _, want_content = _captured(name)
    engine = _engine(thinking_mode="off")

    chunks = await _drain(engine, _ask())
    batch = await engine.generate(_ask())

    assert [c for c in chunks if c.reasoning] == []
    assert chunks[-1].result is not None and chunks[-1].result.reasoning is None
    assert "".join(c.text for c in chunks if not c.done) == want_content
    assert batch.reasoning is None


# --------------------------------------------------------------------------- #
# a history turn's reasoning goes back up
# --------------------------------------------------------------------------- #


def _history() -> list[Message]:
    return [
        Message(role=Role.user, content="Weather in Oslo?", reasoning="NEVER-ON-A-USER"),
        Message(
            role=Role.assistant,
            content=None,
            reasoning="CANARY I should call get_weather",
            toolCalls=[
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"Oslo"}'},
                }
            ],
        ),
        Message(role=Role.tool, content="snow", toolCallId="c1"),
    ]


@respx.mock
async def test_an_assistant_turn_hands_its_reasoning_back() -> None:
    """Measured: llama.cpp renders it -- 172 prompt tokens without and
    184 with a twelve-token canary on the same tool loop. Sent as
    `reasoning_content`, the name llama.cpp reads (it ignored
    `reasoning` on the same request) and vLLM accepts as an alias."""
    route = respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_vllm_reply(content="snowing", finish_reason="stop"))
    )

    await _engine().generate(_ask(messages=_history()))

    user, assistant, tool = _sent(route)["messages"]
    assert assistant["reasoning_content"] == "CANARY I should call get_weather"
    assert "reasoning_content" not in user
    assert "reasoning_content" not in tool


@respx.mock
async def test_openai_itself_is_not_sent_reasoning_it_would_refuse() -> None:
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_vllm_reply(content="snowing", finish_reason="stop"))
    )

    await _engine(base_url="https://api.openai.com", model_id="gpt-4o").generate(
        _ask(messages=_history())
    )

    assert all("reasoning_content" not in m for m in _sent(route)["messages"])


# --------------------------------------------------------------------------- #
# the samplers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("stream", [False, True])
@respx.mock
async def test_the_local_samplers_reach_the_backend(stream: bool) -> None:
    route = respx.post(f"{BASE}/v1/chat/completions")
    route.side_effect = [
        _sse("llamacpp_b10948_thinking_hits_length.sse")
        if stream
        else httpx.Response(200, json=_vllm_reply(content="hi", finish_reason="stop"))
    ]
    request = _ask(
        topK=20, minP=0.05, frequencyPenalty=0.1, presencePenalty=-0.2, parallelToolCalls=True
    )

    if stream:
        await _drain(_engine(), request)
    else:
        await _engine().generate(request)

    payload = _sent(route)
    assert payload["top_k"] == 20
    assert payload["min_p"] == 0.05
    assert payload["frequency_penalty"] == 0.1
    assert payload["presence_penalty"] == -0.2
    assert payload["parallel_tool_calls"] is True


@respx.mock
async def test_falsy_values_are_values() -> None:
    """`top_k: 0` disables the cut and `parallel_tool_calls: false` asks
    for one call per turn; both are falsy. The seed=0 rule (R3.4)."""
    route = respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_vllm_reply(content="hi", finish_reason="stop"))
    )

    await _engine().generate(
        _ask(topK=0, minP=0.0, frequencyPenalty=0.0, presencePenalty=0.0, parallelToolCalls=False)
    )

    payload = _sent(route)
    assert payload["top_k"] == 0
    assert payload["min_p"] == 0.0
    assert payload["frequency_penalty"] == 0.0
    assert payload["presence_penalty"] == 0.0
    assert payload["parallel_tool_calls"] is False


@respx.mock
async def test_unset_samplers_are_absent() -> None:
    route = respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_vllm_reply(content="hi", finish_reason="stop"))
    )

    await _engine().generate(_ask())

    payload = _sent(route)
    for key in ("top_k", "min_p", "frequency_penalty", "presence_penalty", "parallel_tool_calls"):
        assert key not in payload


def test_a_local_backend_advertises_every_new_setting() -> None:
    settings = set(_engine().supported_settings)
    assert {"topK", "minP", "frequencyPenalty", "presencePenalty", "parallelToolCalls"} <= settings


@pytest.mark.parametrize("field,value", [("topK", 20), ("minP", 0.1)])
@respx.mock
async def test_openai_itself_refuses_the_local_only_samplers(field: str, value: object) -> None:
    """OpenAI answers `top_k` with "Unrecognized request argument". Said
    here, before any HTTP, and left out of supportedSettings so the
    gateway routes around this backend rather than trying it."""
    route = respx.post("https://api.openai.com/v1/chat/completions")
    engine = _engine(base_url="https://api.openai.com", model_id="gpt-4o")

    with pytest.raises(CliError) as caught:
        await engine.generate(_ask(**{field: value}, callerSettings=[field]))

    assert caught.value.upstream_status == 400
    assert not route.called
    assert field not in engine.supported_settings
    # The pair: OpenAI does take the penalties and parallel_tool_calls.
    assert {"frequencyPenalty", "presencePenalty", "parallelToolCalls"} <= set(
        engine.supported_settings
    )


@pytest.mark.parametrize("field", ["frequencyPenalty", "presencePenalty"])
@respx.mock
async def test_a_fixed_sampler_model_refuses_the_penalties(field: str) -> None:
    route = respx.post("https://api.openai.com/v1/chat/completions")
    engine = _engine(
        base_url="https://api.openai.com",
        model_id="o3-mini",
        fixed_temperature_pattern=OPENAI_FIXED_TEMPERATURE_PATTERN,
    )

    with pytest.raises(CliError):
        await engine.generate(_ask(**{field: 0.5}, callerSettings=[field]))

    assert not route.called
    assert field not in engine.supported_settings


@pytest.mark.parametrize(
    "field,value",
    [
        ("topK", 20),
        ("minP", 0.1),
        ("frequencyPenalty", 0.5),
        ("presencePenalty", 0.5),
        ("parallelToolCalls", False),
    ],
)
def test_the_cli_engines_refuse_every_new_explicit_setting(field: str, value: object) -> None:
    with pytest.raises(CliError) as caught:
        warn_dropped_sampling(
            _ask(**{field: value}, callerSettings=[field]),
            engine="claude_code_cli",
            model_id="claude-opus-4-7",
            warned=set(),
        )
    assert caught.value.upstream_status == 400


# --------------------------------------------------------------------------- #
# the route frames a reasoning token
# --------------------------------------------------------------------------- #


class _ThinkingAdapter:
    backend_kind = "openai_compat_http"

    async def generate(self, request: GenerateRequest) -> GenerateResponse:  # pragma: no cover
        raise AssertionError("not used")

    async def stream(self, request: GenerateRequest) -> Any:
        yield Chunk(reasoning="ROUTETHOUGHT")
        yield Chunk(text="answer")
        yield Chunk(
            done=True,
            result=GenerateResponse(
                content="answer",
                reasoning="ROUTETHOUGHT",
                finishReason=FinishReason.stop,
                backend=BackendKind.openai_compat_http,
                modelId="qwen3-0.6b",
                latencyMs=1,
            ),
        )


def test_the_stream_route_frames_reasoning_as_its_own_token(client: TestClient) -> None:
    client.app.state.adapter = _ThinkingAdapter()  # type: ignore[attr-defined]

    response = client.post(
        "/v1/generate/stream", json={"messages": [{"role": "user", "content": "ping"}]}
    )

    assert response.status_code == 200
    tokens = [
        json.loads(line[6:])
        for block in response.text.split("\n\n")
        if block.startswith("event: token")
        for line in block.splitlines()
        if line.startswith("data: ")
    ]
    assert tokens == [{"reasoning": "ROUTETHOUGHT"}, {"text": "answer"}]
    done = next(b for b in response.text.split("\n\n") if b.startswith("event: done"))
    assert json.loads(done.split("data: ", 1)[1])["reasoning"] == "ROUTETHOUGHT"
