"""A stream says what the backend is doing before it has anything to say.

2026-09-27. A tester sent a prompt to a model on the processor, saw
nothing for minutes, concluded it had failed and left; the answer was
there when he came back. Before the first token the stream carried no
frame at all -- measured on llama.cpp b11215: 21 s of prompt reading,
then the first token.

Every backend now reports what it can observe, and only when the caller
asked (`reportProgress`):

* llama.cpp: how far it has read the prompt (`return_progress`), from a
  capture of b11215 in `fixtures/`.
* any HTTP backend, a hosted API included: `working` when the response
  opens, and at SSE keepalive comments, throttled.
* Claude Code: `working` at start and at each request it sends, `tool`
  when it starts one of its tools, and its thinking as reasoning -- from
  a capture of the real 2.1.207 CLI in `fixtures/`, sanitized.
* Codex: `working` at `turn.started`, `tool` at an item that runs
  something, and the reason it failed, which it gives on stdout.

Progress is never output: it does not arm the stall clock, and a hosted
API is never sent llama.cpp's flag, which it would refuse.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
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
    Stage,
    StreamProgress,
)
from eugene_plexus_inference_driver.engines import claude_code_cli, codex_cli
from eugene_plexus_inference_driver.engines._subprocess import CliError, CliResult
from eugene_plexus_inference_driver.engines.base import Chunk
from eugene_plexus_inference_driver.engines.claude_code_cli import ClaudeCodeCliEngine
from eugene_plexus_inference_driver.engines.codex_cli import CodexCliEngine
from eugene_plexus_inference_driver.engines.openai_compat_http import OpenAiCompatibleHttpEngine

FIXTURES = Path(__file__).parent / "fixtures"
BASE = "http://127.0.0.1:9181"
LLAMA_SSE = "llamacpp_b11215_return_progress.sse"
# The real llama-server b10948 with `timings_per_token` and `return_progress`
# (Qwen3 1.7B, 40 tokens), its model path replaced (2026-10-10).
LLAMA_TIMINGS_SSE = "llamacpp_b10948_timings_per_token.sse"
CLAUDE_JSONL = "claude_code_2_1_stream_tool_use.jsonl"


def _ask(report: bool = True) -> GenerateRequest:
    return GenerateRequest(
        messages=[Message(role=Role.user, content="Which way? One word.")],
        reportProgress=report,
    )


def _engine(**kwargs: Any) -> OpenAiCompatibleHttpEngine:
    kwargs.setdefault("base_url", BASE)
    kwargs.setdefault("model_id", "qwen3-0.6b")
    kwargs.setdefault("auth_required", False)
    return OpenAiCompatibleHttpEngine(**kwargs)


def _sse(name: str) -> httpx.Response:
    return httpx.Response(
        200,
        text=(FIXTURES / name).read_text(encoding="utf-8"),
        headers={"content-type": "text/event-stream"},
    )


def _captured_progress(name: str) -> list[dict[str, int]]:
    reads = []
    for line in (FIXTURES / name).read_text(encoding="utf-8").splitlines():
        if line.startswith("data: {"):
            read = json.loads(line[6:]).get("prompt_progress")
            if read:
                reads.append(read)
    return reads


def _props_llama(route: respx.MockRouter | None = None) -> respx.Route:
    return respx.get(f"{BASE}/props").mock(
        return_value=httpx.Response(200, json={"default_generation_settings": {"n_ctx": 8192}})
    )


def _sent(route: respx.Route) -> dict[str, Any]:
    return json.loads(route.calls[0].request.read())


async def _drain(engine: Any, request: GenerateRequest) -> list[Chunk]:
    return [chunk async for chunk in engine.stream(request)]


def _progress(chunks: list[Chunk]) -> list[StreamProgress]:
    return [c.progress for c in chunks if c.progress is not None]


# --------------------------------------------------------------------------- #
# llama.cpp: prompt reading, from the capture
# --------------------------------------------------------------------------- #


@respx.mock
async def test_llamacpp_is_asked_for_progress_and_every_batch_is_reported() -> None:
    _props_llama()
    chat = respx.post(f"{BASE}/v1/chat/completions").mock(return_value=_sse(LLAMA_SSE))
    chunks = await _drain(_engine(), _ask())

    assert _sent(chat)["return_progress"] is True
    reads = [p for p in _progress(chunks) if p.stage == Stage.prompt]
    captured = _captured_progress(LLAMA_SSE)
    assert len(captured) == 16
    assert [(p.promptTokens, p.cachedTokens, p.processedTokens, p.elapsedMs) for p in reads] == [
        (r["total"], r["cache"], r["processed"], r["time_ms"]) for r in captured
    ]
    # All of it before the first word of output, which is the point.
    first_output = next(i for i, c in enumerate(chunks) if c.text or c.reasoning)
    last_read = max(i for i, c in enumerate(chunks) if c.progress and c.progress.stage == "prompt")
    assert last_read < first_output


@respx.mock
async def test_the_answer_is_the_same_with_or_without_progress() -> None:
    _props_llama()
    respx.post(f"{BASE}/v1/chat/completions").mock(return_value=_sse(LLAMA_SSE))
    asked = await _drain(_engine(), _ask(report=True))
    plain = await _drain(_engine(), _ask(report=False))

    def output(chunks: list[Chunk]) -> tuple[str, str, Any]:
        done = chunks[-1].result
        return (
            "".join(c.text for c in chunks),
            "".join(c.reasoning for c in chunks),
            done.model_dump() if done else None,
        )

    assert output(asked)[:2] == output(plain)[:2]
    assert output(asked)[0] or output(asked)[1]
    # Unasked, nothing that is not output comes out -- even from a backend
    # that sends progress anyway.
    assert _progress(plain) == []


@respx.mock
async def test_unasked_the_flag_is_not_sent() -> None:
    props = _props_llama()
    chat = respx.post(f"{BASE}/v1/chat/completions").mock(return_value=_sse(LLAMA_SSE))
    await _drain(_engine(), _ask(report=False))
    assert not {"return_progress", "timings_per_token"} & _sent(chat).keys()
    assert not props.called


@respx.mock
async def test_a_hosted_api_is_never_sent_the_flag_but_still_says_it_has_the_request() -> None:
    # A hosted API has no /props, and OpenAI refuses an unknown field with
    # a 400 -- so `return_progress` goes only to a backend that answered
    # as llama-server.
    respx.get(f"{BASE}/props").mock(return_value=httpx.Response(404))
    body = 'data: {"choices":[{"delta":{"content":"East"}}]}\n\n'
    body += 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
    chat = respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
    )
    chunks = await _drain(_engine(), _ask())

    assert not {"return_progress", "timings_per_token"} & _sent(chat).keys()
    assert [p.stage for p in _progress(chunks)] == [Stage.working]
    assert chunks[0].progress is not None  # before anything else


@respx.mock
async def test_a_backend_that_cannot_be_asked_is_not_sent_the_flag() -> None:
    # Unknown is no: a transport failure on /props must not become a yes.
    respx.get(f"{BASE}/props").mock(side_effect=httpx.ConnectError("refused"))
    chat = respx.post(f"{BASE}/v1/chat/completions").mock(return_value=_sse(LLAMA_SSE))
    await _drain(_engine(), _ask())
    assert "return_progress" not in _sent(chat)


# --------------------------------------------------------------------------- #
# llama.cpp: how far it has written, from the capture (2026-10-10)
# --------------------------------------------------------------------------- #


@respx.mock
async def test_llamacpp_is_asked_for_its_running_count(monkeypatch: pytest.MonkeyPatch) -> None:
    # Every frame's count, unthrottled: it rises, and the last one is the
    # usage's own count. Troy, 2026-10-10: Workbench said "Waiting for the
    # model" through minutes of reasoning.
    from eugene_plexus_inference_driver.engines import openai_compat_http

    monkeypatch.setattr(openai_compat_http, "_GENERATING_PROGRESS_SECONDS", 0.0)
    _props_llama()
    chat = respx.post(f"{BASE}/v1/chat/completions").mock(return_value=_sse(LLAMA_TIMINGS_SSE))
    chunks = await _drain(_engine(), _ask())

    assert _sent(chat)["timings_per_token"] is True
    writing = [p for p in _progress(chunks) if p.stage == Stage.generating]
    counts = [p.generatedTokens for p in writing]
    assert counts and counts == sorted(set(counts))
    done = chunks[-1].result
    assert done is not None and done.usage is not None
    assert counts[-1] == done.usage.completionTokens == 40
    assert writing[-1].tokensPerSecond is not None and writing[-1].tokensPerSecond > 0
    # Writing comes after reading: no count before the prompt is read.
    first_count = next(
        i for i, c in enumerate(chunks) if c.progress and c.progress.stage == "generating"
    )
    last_read = max(i for i, c in enumerate(chunks) if c.progress and c.progress.stage == "prompt")
    assert last_read < first_count


@respx.mock
async def test_the_running_count_is_said_at_most_twice_a_second() -> None:
    # The capture arrives at once, so the throttle lets one count through.
    _props_llama()
    respx.post(f"{BASE}/v1/chat/completions").mock(return_value=_sse(LLAMA_TIMINGS_SSE))
    chunks = await _drain(_engine(), _ask())
    assert [p.generatedTokens for p in _progress(chunks) if p.stage == Stage.generating] == [1]


@respx.mock
async def test_the_answer_is_the_same_with_or_without_the_running_count() -> None:
    _props_llama()
    respx.post(f"{BASE}/v1/chat/completions").mock(return_value=_sse(LLAMA_TIMINGS_SSE))
    asked = await _drain(_engine(), _ask(report=True))
    plain = await _drain(_engine(), _ask(report=False))
    assert "".join(c.reasoning for c in asked) == "".join(c.reasoning for c in plain)
    assert "".join(c.text for c in asked) == "".join(c.text for c in plain)
    assert "".join(c.reasoning for c in asked)
    assert _progress(plain) == []


def test_a_frame_without_a_count_says_nothing() -> None:
    from eugene_plexus_inference_driver.engines.openai_compat_http import _generating_progress

    assert _generating_progress(None) is None
    assert _generating_progress({"predicted_n": 0, "predicted_per_second": 0.0}) is None
    assert _generating_progress({"predicted_n": True}) is None
    said = _generating_progress({"predicted_n": 12, "predicted_per_second": 0.0})
    assert said is not None and said.generatedTokens == 12 and said.tokensPerSecond is None


@respx.mock
async def test_a_llamacpp_still_loading_is_asked_again() -> None:
    # Found by the live run: llama-server answers /props with 503 while it
    # loads, which is exactly when a companion driver first asks. That is
    # "not yet", not "not llama.cpp".
    props = respx.get(f"{BASE}/props")
    props.side_effect = [
        httpx.Response(503, json={"error": {"message": "Loading model"}}),
        httpx.Response(200, json={"default_generation_settings": {"n_ctx": 8192}}),
    ]
    # The context probe tries vLLM's and Ollama's endpoints on a miss.
    respx.get(f"{BASE}/v1/models").mock(return_value=httpx.Response(503))
    respx.get(f"{BASE}/api/ps").mock(return_value=httpx.Response(404))
    chat = respx.post(f"{BASE}/v1/chat/completions").mock(return_value=_sse(LLAMA_SSE))
    engine = _engine()
    assert await engine.context_window() is None
    await _drain(engine, _ask())
    assert _sent(chat)["return_progress"] is True
    assert props.call_count == 2


@respx.mock
async def test_the_context_probe_already_answers_so_props_is_read_once() -> None:
    props = _props_llama()
    respx.post(f"{BASE}/v1/chat/completions").mock(return_value=_sse(LLAMA_SSE))
    engine = _engine()
    assert await engine.context_window() == 8192
    await _drain(engine, _ask())
    await _drain(engine, _ask())
    assert props.call_count == 1


class _Paced(httpx.AsyncByteStream):
    def __init__(self, parts: list[tuple[float, bytes]]) -> None:
        self._parts = parts

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for delay, chunk in self._parts:
            if delay:
                await asyncio.sleep(delay)
            yield chunk

    async def aclose(self) -> None:
        return None


def _frame(payload: dict[str, Any]) -> bytes:
    return b"data: " + json.dumps(payload).encode() + b"\n\n"


def _read(processed: int) -> bytes:
    return _frame(
        {
            "choices": [{"delta": {"role": "assistant", "content": None}}],
            "prompt_progress": {
                "total": 1000,
                "cache": 0,
                "processed": processed,
                "time_ms": processed,
            },
        }
    )


@respx.mock
async def test_slow_batches_are_not_a_stalled_stream() -> None:
    # A batch on a large model on the processor can take longer than the
    # stall window. The prompt is still being read; the clock must not run.
    _props_llama()
    parts = [(0.0, _read(0)), (0.4, _read(500)), (0.4, _read(1000)), (0.4, b"")]
    parts += [(0.0, _frame({"choices": [{"delta": {"content": "East"}}]}))]
    parts += [(0.0, _frame({"choices": [{"delta": {}, "finish_reason": "stop"}]}))]
    parts += [(0.0, b"data: [DONE]\n\n")]
    respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, stream=_Paced(parts), headers={"content-type": "text/event-stream"}
        )
    )
    chunks = await _drain(_engine(stall_seconds=0.3), _ask())
    assert "".join(c.text for c in chunks) == "East"
    assert [p.processedTokens for p in _progress(chunks) if p.stage == Stage.prompt] == [
        0,
        500,
        1000,
    ]


@respx.mock
async def test_keepalive_comments_say_working_but_not_once_per_comment() -> None:
    # OpenRouter's `: OPENROUTER PROCESSING` while a model is queued.
    respx.get(f"{BASE}/props").mock(return_value=httpx.Response(404))
    parts = [(0.0, b": OPENROUTER PROCESSING\n\n") for _ in range(5)]
    parts += [(0.0, _frame({"choices": [{"delta": {"content": "Hi"}}]}))]
    parts += [(0.0, _frame({"choices": [{"delta": {}, "finish_reason": "stop"}]}))]
    parts += [(0.0, b"data: [DONE]\n\n")]
    respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, stream=_Paced(parts), headers={"content-type": "text/event-stream"}
        )
    )
    chunks = await _drain(_engine(), _ask())
    # One for the response opening, one for the burst of five comments.
    assert [p.stage for p in _progress(chunks)] == [Stage.working, Stage.working]


# --------------------------------------------------------------------------- #
# The route
# --------------------------------------------------------------------------- #


class _Engine:
    backend_kind = BackendKind.openai_compat_http
    supports_tool_calling = True

    async def generate(self, request: GenerateRequest) -> GenerateResponse:  # pragma: no cover
        raise AssertionError("not called")

    async def stream(self, request: GenerateRequest) -> AsyncIterator[Chunk]:
        yield Chunk(progress=StreamProgress(stage=Stage.prompt, promptTokens=9, processedTokens=4))
        yield Chunk(progress=StreamProgress(stage=Stage.tool, tool="Read"))
        yield Chunk(text="East")
        yield Chunk(
            done=True,
            result=GenerateResponse(content="East", finishReason=FinishReason.stop),
        )


def test_the_route_frames_progress_as_its_own_event(client: TestClient) -> None:
    client.app.state.adapter = _Engine()  # type: ignore[attr-defined]
    with client.stream(
        "POST",
        "/v1/generate/stream",
        json={"messages": [{"role": "user", "content": "x"}], "reportProgress": True},
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())
    events = [block.split("\n") for block in body.strip().split("\n\n")]
    assert [e[0] for e in events] == [
        "event: progress",
        "event: progress",
        "event: token",
        "event: done",
    ]
    assert json.loads(events[0][1][6:]) == {
        "stage": "prompt",
        "promptTokens": 9,
        "processedTokens": 4,
    }
    assert json.loads(events[1][1][6:]) == {"stage": "tool", "tool": "Read"}


# --------------------------------------------------------------------------- #
# Claude Code: from the capture
# --------------------------------------------------------------------------- #


def _claude(monkeypatch: pytest.MonkeyPatch) -> None:
    lines = (FIXTURES / CLAUDE_JSONL).read_text(encoding="utf-8").splitlines()

    async def fake(argv: list[str], **_: Any) -> AsyncIterator[str]:
        for line in lines:
            yield line

    monkeypatch.setattr(claude_code_cli, "stream_cli_lines", fake)


async def test_claude_says_when_it_starts_sends_and_runs_a_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _claude(monkeypatch)
    chunks = await _drain(ClaudeCodeCliEngine(), _ask())
    seen = [(p.stage, p.tool) for p in _progress(chunks)]
    # init, `requesting`, the Read tool, `requesting` again with its result.
    assert seen == [
        (Stage.working, None),
        (Stage.working, None),
        (Stage.tool, "Read"),
        (Stage.working, None),
    ]
    tool_at = next(i for i, c in enumerate(chunks) if c.progress and c.progress.tool == "Read")
    first_text = next(i for i, c in enumerate(chunks) if c.text)
    assert tool_at < first_text
    assert chunks[-1].result is not None
    assert chunks[-1].result.content == "File says hello."


async def test_claudes_thinking_is_reasoning_not_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    # 118 thinking deltas in the capture, every one read past until today.
    _claude(monkeypatch)
    chunks = await _drain(ClaudeCodeCliEngine(), _ask(report=False))
    thoughts = [c.reasoning for c in chunks if c.reasoning]
    assert len(thoughts) == 118
    assert chunks[-1].result is not None
    assert chunks[-1].result.reasoning == "".join(thoughts)
    assert _progress(chunks) == []


async def test_claudes_thinking_is_withheld_when_thinking_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _claude(monkeypatch)
    chunks = await _drain(ClaudeCodeCliEngine(thinking_mode="off"), _ask())
    assert not any(c.reasoning for c in chunks)
    assert chunks[-1].result is not None and chunks[-1].result.reasoning is None


# --------------------------------------------------------------------------- #
# Codex
# --------------------------------------------------------------------------- #

# Captured 2026-09-27 from Codex on a CLI too old for its account's model.
CODEX_TOO_OLD = [
    '{"type":"thread.started","thread_id":"t"}',
    '{"type":"turn.started"}',
    '{"type":"error","message":"{\\"type\\":\\"error\\",\\"status\\":400,\\"error\\":'
    '{\\"type\\":\\"invalid_request_error\\",\\"message\\":\\"The \'gpt-6-astra\' model '
    "requires a newer version of Codex. Please upgrade to the latest app or CLI and try "
    'again.\\"}}"}',
    '{"type":"turn.failed","error":{"message":"{\\"type\\":\\"error\\",\\"status\\":400,'
    '\\"error\\":{\\"type\\":\\"invalid_request_error\\",\\"message\\":\\"The \'gpt-6-astra\' '
    "model requires a newer version of Codex. Please upgrade to the latest app or CLI and "
    'try again.\\"}}"}}',
]
TOO_OLD = "requires a newer version of Codex"


def _is_the_apis_sentence(message: str) -> None:
    # The sentence two JSON levels down, not the JSON around it and not
    # stderr's "Reading additional input from stdin...".
    assert TOO_OLD in message
    assert "{" not in message and "stdin" not in message


def _codex_stream(
    monkeypatch: pytest.MonkeyPatch, lines: list[str], *, exit_error: bool = False
) -> None:
    async def fake(argv: list[str], **_: Any) -> AsyncIterator[str]:
        for line in lines:
            yield line
        if exit_error:
            raise CliError("codex exited 1: Reading additional input from stdin...")

    monkeypatch.setattr(codex_cli, "stream_cli_lines", fake)


async def test_codex_gives_its_own_reason_when_it_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    _codex_stream(monkeypatch, CODEX_TOO_OLD, exit_error=True)
    with pytest.raises(CliError) as failed:
        await _drain(CodexCliEngine(), _ask())
    _is_the_apis_sentence(str(failed.value))


async def test_codex_gives_its_reason_on_the_batch_path_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake(argv: list[str], **_: Any) -> CliResult:
        return CliResult(
            stdout="\n".join(CODEX_TOO_OLD).encode(),
            stderr=b"Reading additional input from stdin...\nERROR codex_models_manager: ...",
            returncode=1,
            elapsed_ms=900,
        )

    monkeypatch.setattr(codex_cli, "run_cli", fake)
    with pytest.raises(CliError) as failed:
        await CodexCliEngine().generate(_ask())
    _is_the_apis_sentence(str(failed.value))


async def test_codex_says_when_its_turn_starts_and_what_it_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Item shapes from Codex's `exec --json` documentation, not a capture:
    # the CLI here was too old to run a turn.
    lines = [
        '{"type":"turn.started"}',
        '{"type":"item.completed","item":{"id":"i0","type":"reasoning","text":"Listing files"}}',
        '{"type":"item.started","item":{"id":"i1","type":"command_execution","command":"dir"}}',
        '{"type":"item.completed","item":{"id":"i1","type":"command_execution","command":"dir"}}',
        '{"type":"item.started","item":{"id":"i2","type":"mcp_tool_call","tool":"search"}}',
        '{"type":"item.completed","item":{"id":"i3","type":"agent_message","text":"Two files."}}',
        '{"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":3}}',
    ]
    _codex_stream(monkeypatch, lines)
    chunks = await _drain(CodexCliEngine(), _ask())
    assert [(p.stage, p.tool) for p in _progress(chunks)] == [
        (Stage.working, None),
        (Stage.tool, "command"),
        (Stage.tool, "search"),
    ]
    assert [c.reasoning for c in chunks if c.reasoning] == ["Listing files"]
    assert chunks[-1].result is not None and chunks[-1].result.content == "Two files."

    _codex_stream(monkeypatch, lines)
    assert _progress(await _drain(CodexCliEngine(), _ask(report=False))) == []
