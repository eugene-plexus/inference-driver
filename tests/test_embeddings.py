"""Call #2 — embeddings, driver half.

Settled 2026-09-12 as *serve what we already launch*: the library
detects dedicated embedding models and will launch one, so the gateway
had to be able to serve it.

Two things here are not obvious and are what these tests are mostly for:

* **The capability belongs to the running backend, not the model.**
  Nothing exposes it -- measured, `llama-server`'s `/props` carries no
  pooling field and Ollama's compatible surface says nothing -- so it is
  determined by trying once.
* **Order is the contract.** An embedding carries no identity, so the
  position of a vector is the only thing that matches it to its input.
  A backend is allowed to answer out of order.
"""

from __future__ import annotations

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from eugene_plexus_inference_driver._generated.models import BackendKind, EmbedResponse
from eugene_plexus_inference_driver.engines._subprocess import CliError
from eugene_plexus_inference_driver.engines.openai_compat_http import OpenAiCompatibleHttpEngine

BASE = "http://127.0.0.1:8081"


def _engine(model_id: str = "nomic-embed-text") -> OpenAiCompatibleHttpEngine:
    return OpenAiCompatibleHttpEngine(
        api_key=None,
        base_url=BASE,
        model_id=model_id,
        backend_kind=BackendKind.openai_compat_http,
        auth_required=False,
    )


def _envelope(vectors: list[list[float]], *, indices: list[int] | None = None) -> dict:
    idx = indices if indices is not None else list(range(len(vectors)))
    return {
        "object": "list",
        "model": "nomic-embed-text",
        "data": [
            {"object": "embedding", "index": i, "embedding": v}
            for i, v in zip(idx, vectors, strict=True)
        ],
        "usage": {"prompt_tokens": 8, "total_tokens": 8},
    }


# --------------------------------------------------------------------- #
# embedding
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
@respx.mock
async def test_a_batch_comes_back_as_one_vector_per_input() -> None:
    """The shape a real Ollama returned for two inputs: 768 floats each."""
    respx.post(f"{BASE}/v1/embeddings").mock(
        return_value=httpx.Response(200, json=_envelope([[0.1] * 768, [0.2] * 768]))
    )

    result = await _engine().embed(["hello world", "goodbye world"])

    assert isinstance(result, EmbedResponse)
    assert len(result.embeddings) == 2
    assert len(result.embeddings[0]) == 768
    assert result.usage is not None and result.usage.promptTokens == 8


@pytest.mark.asyncio
@respx.mock
async def test_vectors_are_ordered_by_index_not_by_arrival() -> None:
    """**The failure this prevents is invisible.** OpenAI documents that
    `data` may come back out of order, and a vector carries no identity
    -- so trusting arrival order pairs every vector with the wrong text
    and produces a system that works and searches badly."""
    respx.post(f"{BASE}/v1/embeddings").mock(
        return_value=httpx.Response(
            200,
            json=_envelope([[3.0], [1.0], [2.0]], indices=[2, 0, 1]),
        )
    )

    result = await _engine().embed(["first", "second", "third"])

    assert result.embeddings == [[1.0], [2.0], [3.0]]


@pytest.mark.asyncio
@respx.mock
async def test_a_short_count_is_refused_rather_than_returned() -> None:
    """Fewer vectors than inputs means the caller cannot match them up at
    all. Silently returning what arrived would hand back vectors
    attributed to the wrong text."""
    respx.post(f"{BASE}/v1/embeddings").mock(
        return_value=httpx.Response(200, json=_envelope([[1.0]]))
    )

    with pytest.raises(CliError) as excinfo:
        await _engine().embed(["one", "two"])

    assert "1 embeddings for 2 inputs" in str(excinfo.value)


@pytest.mark.asyncio
@respx.mock
async def test_a_base64_answer_to_a_float_request_is_refused() -> None:
    """We never ask for base64 -- the gateway does any encoding -- so
    getting one means something is rewriting the request. Coping quietly
    would hide that."""
    respx.post(f"{BASE}/v1/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "model": "m",
                "data": [{"object": "embedding", "index": 0, "embedding": "Cj2SPPUKwLs="}],
            },
        )
    )

    with pytest.raises(CliError) as excinfo:
        await _engine().embed(["x"])

    assert "base64" in str(excinfo.value)


# --------------------------------------------------------------------- #
# the capability, which nothing reports and everything depends on
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
@respx.mock
async def test_a_backend_that_embeds_is_detected_by_trying() -> None:
    route = respx.post(f"{BASE}/v1/embeddings").mock(
        return_value=httpx.Response(200, json=_envelope([[0.5] * 4]))
    )
    engine = _engine()

    assert await engine.probe_embeddings() is True
    assert await engine.probe_embeddings() is True
    assert route.call_count == 1, "a definite answer must be cached"


@pytest.mark.asyncio
@respx.mock
async def test_a_chat_only_backend_says_no() -> None:
    """Ollama's real refusal, verbatim: the runner was started for chat,
    so the very model it is serving cannot be embedded."""
    respx.post(f"{BASE}/v1/embeddings").mock(
        return_value=httpx.Response(
            400,
            json={
                "error": {
                    "message": "This server does not support embeddings. "
                    "Start it with `--embeddings`",
                    "type": "api_error",
                }
            },
        )
    )

    assert await _engine().probe_embeddings() is False


@pytest.mark.asyncio
@respx.mock
async def test_a_backend_that_is_down_is_not_recorded_as_incapable() -> None:
    """A transport failure is not a definite answer. Caching it would
    mark a backend that happened to be restarting as permanently unable
    to embed, for the life of the driver process."""
    route = respx.post(f"{BASE}/v1/embeddings").mock(side_effect=httpx.ConnectError("refused"))
    engine = _engine()

    assert await engine.probe_embeddings() is False
    route.mock(return_value=httpx.Response(200, json=_envelope([[0.5] * 4])))
    assert await engine.probe_embeddings() is True


@pytest.mark.asyncio
async def test_a_cli_subscription_cannot_embed() -> None:
    """There is no flag that makes a coding harness return a vector."""
    from eugene_plexus_inference_driver.engines.claude_code_cli import ClaudeCodeCliEngine

    engine = ClaudeCodeCliEngine(binary_path="claude")
    assert engine.supports_embeddings is False
    with pytest.raises(CliError):
        await engine.embed(["x"])


# --------------------------------------------------------------------- #
# the route
# --------------------------------------------------------------------- #


def test_the_route_refuses_a_backend_that_cannot_embed(client: TestClient) -> None:
    """400, naming itself. **Never substituted**: a caller cannot look at
    a vector and tell that it is wrong, and if it reaches a vector store
    the mistake outlives the request."""
    client.app.state.adapter = _Backend(capable=False)  # type: ignore[attr-defined]

    response = client.post("/v1/embed", json={"input": ["x"]})

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["type"].endswith("#embeddings-unsupported")
    assert "not of the model" in detail["detail"]


def test_the_route_returns_vectors_when_the_backend_can(client: TestClient) -> None:
    client.app.state.adapter = _Backend(capable=True)  # type: ignore[attr-defined]

    response = client.post("/v1/embed", json={"input": ["a", "b"]})

    assert response.status_code == 200
    assert len(response.json()["embeddings"]) == 2


def test_info_reports_the_embeddings_capability(client: TestClient) -> None:
    """Contracted at M0 on the agent and populated by nothing; this is
    the driver's own flag, and the third of three to be filled in after
    `streaming` (M10) and `maxContextTokens` (step 7)."""
    client.app.state.adapter = _Backend(capable=True)  # type: ignore[attr-defined]

    assert client.get("/v1/info").json()["capabilities"]["embeddings"] is True


class _Backend:
    backend_kind = BackendKind.openai_compat_http
    supports_streaming = True
    supports_tool_calling = True
    runtime = None

    def __init__(self, *, capable: bool) -> None:
        self._capable = capable
        self.supports_embeddings = capable

    async def probe_embeddings(self) -> bool:
        return self._capable

    async def context_window(self) -> int | None:
        return None

    async def embed(self, inputs: list[str]) -> EmbedResponse:
        return EmbedResponse(
            embeddings=[[0.1, 0.2] for _ in inputs],
            modelId="fake-embed",
            backend=self.backend_kind,
            latencyMs=1,
        )
