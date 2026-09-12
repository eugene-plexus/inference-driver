"""Step 7 — context-window honesty, driver half.

Two things live here because they are two halves of one decision
(`agent-clients-and-tool-calling.md` §6, open call #3, settled
2026-09-12 as *let the engine refuse*):

* **Advertising.** `capabilities.maxContextTokens` was contracted at M0
  and populated by nothing, so every backend the install does not
  supervise reported no window at all. It is probed from the backend now.
* **Not swallowing the refusal.** An engine that counts tokens exactly
  refuses an over-long prompt with both numbers; the driver used to
  flatten that into a 502, which made the gateway cascade a request no
  backend could serve and hand the caller a retryable error.

Nothing here counts a prompt, by decision.
"""

from __future__ import annotations

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from eugene_plexus_inference_driver._generated.models import (
    BackendKind,
    GenerateRequest,
    Message,
    Role,
)
from eugene_plexus_inference_driver.engines._subprocess import CliError
from eugene_plexus_inference_driver.engines.openai_compat_http import OpenAiCompatibleHttpEngine

BASE = "http://127.0.0.1:8081"


def _engine(model_id: str = "test-model") -> OpenAiCompatibleHttpEngine:
    return OpenAiCompatibleHttpEngine(
        api_key=None,
        base_url=BASE,
        model_id=model_id,
        backend_kind=BackendKind.openai_compat_http,
        auth_required=False,
    )


def _request(prompt: str = "ping") -> GenerateRequest:
    return GenerateRequest(messages=[Message(role=Role.user, content=prompt)])


# --------------------------------------------------------------------- #
# the probe: three real wire shapes, and an honest miss
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
@respx.mock
async def test_llama_cpp_window_comes_from_props() -> None:
    """`/props` is what the agent reads off a supervised runtime; the
    driver reads it from wherever it points, supervised or not."""
    respx.get(f"{BASE}/props").mock(
        return_value=httpx.Response(
            200, json={"default_generation_settings": {"n_ctx": 4096}, "n_ctx": 99999}
        )
    )

    assert await _engine().context_window() == 4096


@pytest.mark.asyncio
@respx.mock
async def test_vllm_window_matches_our_model_not_the_first_card() -> None:
    """A server hosting several models must not hand back the wrong
    one's window. A window that is wrong is worse than one missing."""
    respx.get(f"{BASE}/props").mock(return_value=httpx.Response(404))
    respx.get(f"{BASE}/v1/models").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {"id": "someone-else", "max_model_len": 2048},
                    {"id": "test-model", "max_model_len": 32768},
                ]
            },
        )
    )

    assert await _engine("test-model").context_window() == 32768


@pytest.mark.asyncio
@respx.mock
async def test_ollama_window_comes_from_api_ps() -> None:
    """Ollama's OpenAI-compatible surface carries no window at all and
    `/api/show` carries only the trained maximum. `/api/ps` is the one
    place the number it actually chose appears — measured 131072 on
    0.34.0, which is what it auto-sized to."""
    respx.get(f"{BASE}/props").mock(return_value=httpx.Response(404))
    respx.get(f"{BASE}/v1/models").mock(return_value=httpx.Response(200, json={"data": []}))
    respx.get(f"{BASE}/api/ps").mock(
        return_value=httpx.Response(
            200,
            json={"models": [{"name": "qwen3-coder:30b", "context_length": 131072}]},
        )
    )

    assert await _engine("qwen3-coder:30b").context_window() == 131072


@pytest.mark.asyncio
@respx.mock
async def test_a_backend_that_answers_none_of_them_reports_unknown() -> None:
    """A hosted provider 404s all three. Unknown is a real answer and the
    gateway skips it — the alternative, a plausible-looking number nobody
    read from the backend, is the thing the field exists to prevent."""
    respx.get(f"{BASE}/props").mock(return_value=httpx.Response(404))
    respx.get(f"{BASE}/v1/models").mock(return_value=httpx.Response(404))
    respx.get(f"{BASE}/api/ps").mock(return_value=httpx.Response(404))

    assert await _engine().context_window() is None


@pytest.mark.asyncio
@respx.mock
async def test_a_later_miss_does_not_erase_an_earlier_hit() -> None:
    """Ollama drops a model out of `/api/ps` the moment it idles out. The
    advertised window must not flicker with the engine's idle timer."""
    engine = _engine("qwen3-coder:30b")
    respx.get(f"{BASE}/props").mock(return_value=httpx.Response(404))
    respx.get(f"{BASE}/v1/models").mock(return_value=httpx.Response(200, json={"data": []}))
    ps = respx.get(f"{BASE}/api/ps").mock(
        return_value=httpx.Response(
            200, json={"models": [{"name": "qwen3-coder:30b", "context_length": 131072}]}
        )
    )
    assert await engine.context_window() == 131072

    ps.mock(return_value=httpx.Response(200, json={"models": []}))
    engine._context_window_checked_at = None  # force a re-probe past the TTL
    assert await engine.context_window() == 131072


@pytest.mark.asyncio
@respx.mock
async def test_the_window_is_cached_rather_than_probed_per_call() -> None:
    """`/v1/info` is polled by the gateway's routing refresh, inside
    whatever request triggered it. Three GETs per poll would be latency
    an end user feels."""
    route = respx.get(f"{BASE}/props").mock(
        return_value=httpx.Response(200, json={"default_generation_settings": {"n_ctx": 8192}})
    )
    engine = _engine()

    assert await engine.context_window() == 8192
    assert await engine.context_window() == 8192
    assert await engine.context_window() == 8192
    assert route.call_count == 1


@pytest.mark.asyncio
async def test_a_cli_subscription_reports_no_window() -> None:
    """The harness on the other side of the pipe owns the window, picks
    the model and does its own compaction. Any number would be a guess."""
    from eugene_plexus_inference_driver.engines.claude_code_cli import ClaudeCodeCliEngine

    assert await ClaudeCodeCliEngine(binary_path="claude").context_window() is None


# --------------------------------------------------------------------- #
# not swallowing the engine's own refusal
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
@respx.mock
async def test_an_over_long_prompt_carries_llama_cpps_status_and_both_numbers() -> None:
    """The real b9846 shape. Both numbers have to survive: they are the
    answer, and a better one than anything this layer could compute."""
    respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(
            400,
            json={
                "error": {
                    "code": 400,
                    "message": "the request exceeds the available context size, try increasing it",
                    "type": "exceed_context_size_error",
                    "n_prompt_tokens": 20597,
                    "n_ctx": 512,
                }
            },
        )
    )

    with pytest.raises(CliError) as excinfo:
        await _engine().generate(_request())

    assert excinfo.value.upstream_status == 400
    assert "20597" in str(excinfo.value)
    assert "512" in str(excinfo.value)


@pytest.mark.asyncio
@respx.mock
async def test_a_stream_refused_before_the_first_frame_also_carries_the_status() -> None:
    """Nothing has been sent yet, so this can still be a status code —
    and a streamed request must not be the one that loses the diagnosis."""
    respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(400, json={"error": {"type": "exceed_context_size_error"}})
    )

    with pytest.raises(CliError) as excinfo:
        async for _ in _engine().stream(_request()):
            pass

    assert excinfo.value.upstream_status == 400


def test_the_route_turns_a_backend_400_into_a_400_that_does_not_cascade(
    client: TestClient,
) -> None:
    """The gateway hard-fails a 4xx and cascades a 5xx. Flattening every
    backend failure into a 502 defeated that: an over-long prompt was
    retried against every replica and every tier before coming back
    retryable."""
    client.app.state.adapter = _Refusing(  # type: ignore[attr-defined]
        CliError(
            "openai_compat_http returned 400: n_prompt_tokens 20597, n_ctx 512",
            upstream_status=400,
        )
    )

    response = client.post("/v1/generate", json={"messages": [{"role": "user", "content": "x"}]})

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["status"] == 400
    assert detail["type"].endswith("#backend-rejected-request")
    assert "20597" in detail["detail"]
    assert "512" in detail["detail"]


@pytest.mark.parametrize("upstream", [408, 409, 425, 429])
def test_a_retryable_refusal_keeps_its_502_so_failover_still_fires(
    client: TestClient, upstream: int
) -> None:
    """A rate-limited cloud provider falling through to a local engine is
    the case the priority list was built for — the smoke test that
    motivated it was an OpenRouter 429. Passing every 4xx through would
    have made that hard-fail instead."""
    client.app.state.adapter = _Refusing(  # type: ignore[attr-defined]
        CliError(f"upstream said {upstream}", upstream_status=upstream)
    )

    response = client.post("/v1/generate", json={"messages": [{"role": "user", "content": "x"}]})

    assert response.status_code == 502
    assert response.json()["detail"]["type"].endswith("#backend-error")


def test_a_transport_failure_still_reads_as_a_backend_error(client: TestClient) -> None:
    """No status to carry, so 502 and a cascade are right."""
    client.app.state.adapter = _Refusing(CliError("connection refused"))  # type: ignore[attr-defined]

    response = client.post("/v1/generate", json={"messages": [{"role": "user", "content": "x"}]})

    assert response.status_code == 502


def test_info_reports_the_window_the_backend_admitted_to(client: TestClient) -> None:
    """`maxContextTokens` was contracted at M0 and populated by nothing —
    `capabilities.streaming`'s story one field over."""
    client.app.state.adapter = _WithWindow(8192)  # type: ignore[attr-defined]

    body = client.get("/v1/info").json()

    assert body["capabilities"]["maxContextTokens"] == 8192


def test_a_failing_probe_does_not_take_v1_info_down(client: TestClient) -> None:
    """A driver that 500s on `/v1/info` drops out of routing entirely. A
    window is the least important thing this endpoint reports, so it is
    the first thing to give up."""
    client.app.state.adapter = _WithWindow(RuntimeError("backend on fire"))  # type: ignore[attr-defined]

    response = client.get("/v1/info")

    assert response.status_code == 200
    assert response.json()["capabilities"]["maxContextTokens"] is None


class _Refusing:
    backend_kind = BackendKind.openai_compat_http
    supports_tool_calling = True

    def __init__(self, error: Exception) -> None:
        self._error = error

    async def generate(self, request: GenerateRequest) -> object:
        raise self._error


class _WithWindow:
    backend_kind = BackendKind.openai_compat_http
    supports_streaming = True
    supports_tool_calling = True
    runtime = None

    def __init__(self, value: object) -> None:
        self._value = value

    async def context_window(self) -> int | None:
        if isinstance(self._value, Exception):
            raise self._value
        return self._value  # type: ignore[return-value]
