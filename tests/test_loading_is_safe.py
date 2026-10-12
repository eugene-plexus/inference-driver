"""A replica whose engine is loading is safe to try elsewhere (inference-driver#10, 2026-10-12).

Found by MR6's acceptance: two replicas of one model behind the gateway, one still loading.
llama-server answers the loading one with a 503, the driver made it a 502 whose
`retryDisposition` was `indeterminate` ("Outcome unknown: work may have occurred"), and the
gateway, which replays only `safe`, never tried the healthy replica. Nothing was computed: a
loading engine refuses before it looks at the request, as a full KV pool does
(`failures.capacity_refused`), which `disposition` already calls `safe`.

**The bytes were measured on 2026-10-12**, llama-server b10948 on Qwen3-0.6B-Q4_K_M, CPU only
(`-ngl 0`, CUDA hidden), polling every path from the moment the port was bound until the model
was in memory (0.73 s). For that whole window every path that needs the model -- chat (streamed
or not, with tools or without), embeddings and rerank -- and `/health`, `/props` and
`/v1/models` as well, answered HTTP 503 with these 75 bytes, `Server: llama.cpp`,
`Connection: close`, no `Retry-After`. **The key order is `message`, `type`, `code`**: this
repository's earlier fixtures had `code` first, which llama-server does not send.

Keyed on the 503 and the words together, as the pool-full case is keyed on its words: another
status with these words, or a 503 with other words, is not this.
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
    GenerateRequest,
    Message,
    RerankPassagesRequest,
    Role,
)
from eugene_plexus_inference_driver.app import create_app
from eugene_plexus_inference_driver.engines._subprocess import CliError
from eugene_plexus_inference_driver.engines.openai_compat_http import OpenAiCompatibleHttpEngine
from eugene_plexus_inference_driver.failures import disposition, loading_refused
from eugene_plexus_inference_driver.routes.generate import _backend_error
from eugene_plexus_inference_driver.settings import Settings

BASE = "http://llama.local"
LOADING_BODY = b'{"error":{"message":"Loading model","type":"unavailable_error","code":503}}'
LOADING_HEADERS = {
    "Server": "llama.cpp",
    "Content-Type": "application/json; charset=utf-8",
    "Connection": "close",
}
#: The other order of the same keys, which this repository's fixtures used before it was measured.
LOADING_BODY_CODE_FIRST = (
    b'{"error":{"code":503,"message":"Loading model","type":"unavailable_error"}}'
)
OK = {
    "model": "qwen",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
}


def _loading(body: bytes = LOADING_BODY) -> httpx.Response:
    return httpx.Response(503, content=body, headers=LOADING_HEADERS)


def _engine(model_id: str = "qwen") -> OpenAiCompatibleHttpEngine:
    return OpenAiCompatibleHttpEngine(
        base_url=BASE,
        model_id=model_id,
        backend_kind=BackendKind.openai_compat_http,
        auth_required=False,
    )


def _ask() -> GenerateRequest:
    return GenerateRequest(messages=[Message(role=Role.user, content="hi")])


async def _meet_a_loading_engine(engine: OpenAiCompatibleHttpEngine, operation: str) -> CliError:
    with pytest.raises(CliError) as caught:
        if operation == "generate":
            await engine.generate(_ask())
        elif operation == "stream":
            _ = [chunk async for chunk in engine.stream(_ask())]
        elif operation == "embed":
            await engine.embed(["a passage"])
        else:
            await engine.rerank(RerankPassagesRequest(query="q", documents=["a", "b"]))
    return caught.value


def test_the_fixture_is_the_measured_75_bytes() -> None:
    assert len(LOADING_BODY) == 75, "llama-server's Content-Length for it, measured"
    assert json.loads(LOADING_BODY) == json.loads(LOADING_BODY_CODE_FIRST)


# --------------------------------------------------------------------------- #
# the unit: what disposition says of it
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        # What the engine reports for each path that meets it (`openai_compat_http`).
        'openai_compat_http returned 503: {"error":{"message":"Loading model","type":"unavailable_error","code":503}}',
        'openai_compat_http returned 503 for embeddings: {"error":{"message":"Loading model","type":"unavailable_error","code":503}}',
        "openai_compat_http returned 503 for a rerank: Loading model",
        # The same words in the other key order, with a space, and as a plain-text body.
        'openai_compat_http returned 503: {"error":{"code":503,"message":"Loading model","type":"unavailable_error"}}',
        'openai_compat_http returned 503: {"error": {"message": "Loading model"}}',
        "openai_compat_http returned 503: Loading model",
        "openai_compat_http returned 503 for a rerank: Loading model.",
        # The other two shapes a backend's one message arrives in, read as `tools_refusal`
        # reads them: a bare `error` string and a bare `message`.
        'openai_compat_http returned 503: {"error":"Loading model"}',
        'openai_compat_http returned 503: {"message":"Loading model"}',
        # A frame in the stream's own words: the driver's words end at the first colon.
        "openai_compat_http stream error: Loading model",
    ],
    ids=[
        "chat",
        "embeddings",
        "rerank",
        "code-first",
        "spaced",
        "plain-text",
        "full-stop",
        "bare-error",
        "bare-message",
        "stream-frame",
    ],
)
def test_a_503_that_says_loading_model_is_safe_to_send_again(text: str) -> None:
    error = CliError(text, upstream_status=503)
    assert loading_refused(error)
    assert disposition(error) == "safe"


@pytest.mark.parametrize(
    ("status", "text"),
    [
        # A 503 of some other kind is a backend that may be broken: outcome unknown, as before.
        (503, 'openai_compat_http returned 503: {"error":{"message":"boom"}}'),
        (503, "openai_compat_http returned 503: <html>Service Unavailable</html>"),
        (503, "openai_compat_http returned 503: "),
        (
            503,
            'openai_compat_http returned 503: {"error":{"message":"Loading model weights failed"}}',
        ),
        (503, 'openai_compat_http returned 503: {"error":{"message":"Still Loading model"}}'),
        (503, 'openai_compat_http returned 503: {"detail":"Loading model"}'),
        (503, "openai_compat_http returned 503: <p>Loading model</p>"),
        (503, "fixture"),
        # The words are in the body, and the body is not llama-server's message. The match was
        # a search of the whole text, so each of these counted as loading and became safe:
        # echoed back inside a field of something else, behind a proxy's own words, and as the
        # message of an object nested in the error's own. Only the error's own message, exactly,
        # is the measured body (`_backend_message`, as `tools_refusal` reads one).
        (
            503,
            'openai_compat_http returned 503: {"echo":{"messages":[{"role":"user","message":"Loading model"}]}}',
        ),
        (503, "openai_compat_http returned 503: Service Unavailable: Loading model"),
        (
            503,
            'openai_compat_http returned 503: {"error":{"message":"upstream failed","request":{"message":"Loading model"}}}',
        ),
        (
            503,
            'openai_compat_http returned 503 for embeddings: {"error":{"message":"upstream failed","request":{"message":"Loading model"}}}',
        ),
        (503, "openai_compat_http returned 503 for a rerank: Service Unavailable: Loading model"),
        # The words under another status are not llama-server's.
        (500, 'openai_compat_http returned 500: {"error":{"message":"Loading model"}}'),
        (502, 'openai_compat_http returned 502: {"error":{"message":"Loading model"}}'),
        (504, "openai_compat_http returned 504 for a rerank: Loading model"),
        (None, "openai_compat_http request failed: Loading model"),
    ],
    ids=[
        "boom",
        "html",
        "empty",
        "weights",
        "still",
        "other-key",
        "in-markup",
        "fixture",
        "echoed-in-another-field",
        "behind-a-proxys-words",
        "nested-message",
        "nested-message-embeddings",
        "behind-a-proxys-words-rerank",
        "500",
        "502",
        "504",
        "no-status",
    ],
)
def test_nothing_else_is_a_loading_engine(status: int | None, text: str) -> None:
    error = CliError(text, upstream_status=status)
    assert not loading_refused(error)
    assert disposition(error) == "indeterminate"


@pytest.mark.parametrize(
    ("status", "expected"),
    [(400, "terminal"), (404, "terminal"), (429, "safe"), (408, "indeterminate")],
)
def test_the_words_under_a_4xx_keep_that_statuss_meaning(status: int, expected: str) -> None:
    """Status still decides first where it always did: the words do not turn a 400 into a retry."""
    error = CliError('returned: {"error":{"message":"Loading model"}}', upstream_status=status)
    assert not loading_refused(error)
    assert disposition(error) == expected


# --------------------------------------------------------------------------- #
# through the engine: every path that meets the engine while it loads
# --------------------------------------------------------------------------- #


@respx.mock
@pytest.mark.anyio
@pytest.mark.parametrize("body", [LOADING_BODY, LOADING_BODY_CODE_FIRST], ids=["measured", "other"])
@pytest.mark.parametrize("operation", ["generate", "stream", "embed", "rerank"])
async def test_a_loading_engine_is_safe_whichever_path_met_it(operation: str, body: bytes) -> None:
    respx.post(f"{BASE}/v1/chat/completions").mock(return_value=_loading(body))
    respx.post(f"{BASE}/v1/embeddings").mock(return_value=_loading(body))
    respx.post(f"{BASE}/v1/rerank").mock(return_value=_loading(body))
    engine = _engine()
    try:
        error = await _meet_a_loading_engine(engine, operation)
    finally:
        await engine.aclose()

    assert error.upstream_status == 503 and "Loading model" in str(error)
    assert loading_refused(error)
    assert disposition(error) == "safe"
    # What the route makes of it: a 502 (the contract's "backend refusal worth retrying
    # elsewhere", which cascades), in the backend's own words, and no claim that work happened.
    failure = _backend_error(error, "openai_compat_http")
    assert failure.status_code == 502
    detail = failure.detail
    assert detail["retryDisposition"] == "safe"
    assert detail["type"].endswith("#backend-error")
    assert "Loading model" in detail["detail"]
    assert "Outcome unknown" not in detail["detail"] and "work may have occurred" not in str(detail)


#: Bodies that merely mention the words: echoed inside another field, behind a proxy's own
#: words, and as the message of a nested object. Each was `safe` while the match searched the
#: text of the whole failure.
MENTIONS = {
    "echoed": (
        b'{"echo":{"messages":[{"role":"user","message":"Loading model"}]}}',
        "application/json",
    ),
    "proxy": (b"Service Unavailable: Loading model", "text/plain"),
    "nested": (
        b'{"error":{"message":"upstream failed","request":{"message":"Loading model"}}}',
        "application/json",
    ),
}


@respx.mock
@pytest.mark.anyio
@pytest.mark.parametrize("mention", list(MENTIONS))
@pytest.mark.parametrize("operation", ["generate", "stream", "embed", "rerank"])
async def test_a_503_that_only_mentions_loading_is_outcome_unknown_whichever_path_met_it(
    operation: str, mention: str
) -> None:
    body, content_type = MENTIONS[mention]
    refusal = httpx.Response(503, content=body, headers={"Content-Type": content_type})
    respx.post(f"{BASE}/v1/chat/completions").mock(return_value=refusal)
    respx.post(f"{BASE}/v1/embeddings").mock(return_value=refusal)
    respx.post(f"{BASE}/v1/rerank").mock(return_value=refusal)
    engine = _engine()
    try:
        error = await _meet_a_loading_engine(engine, operation)
    finally:
        await engine.aclose()

    assert error.upstream_status == 503
    assert not loading_refused(error)
    assert disposition(error) == "indeterminate"
    detail: Any = _backend_error(error, "openai_compat_http").detail
    assert detail["retryDisposition"] == "indeterminate"
    assert "Outcome unknown" in detail["detail"]


@respx.mock
@pytest.mark.anyio
async def test_the_same_503_from_a_broken_engine_is_still_outcome_unknown() -> None:
    respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(503, json={"error": {"message": "CUDA error", "code": 503}})
    )
    engine = _engine()
    try:
        error = await _meet_a_loading_engine(engine, "generate")
    finally:
        await engine.aclose()
    assert disposition(error) == "indeterminate"
    assert "Outcome unknown" in _backend_error(error, "openai_compat_http").detail["detail"]


# --------------------------------------------------------------------------- #
# through the real app: the answer the gateway reads
# --------------------------------------------------------------------------- #


def _app(tmp_path: Path, **config: Any) -> TestClient:
    path = tmp_path / "custom.yaml"
    path.write_text(
        json.dumps(
            {
                "provider": "openai_compat_custom",
                "baseUrl": BASE,
                "backendLocality": "local",
                "modelId": "qwen",
                **config,
            }
        ),
        "utf-8",
    )
    return TestClient(create_app(settings=Settings(config_file=path)))


def _llama_that_is_loading(state: dict[str, bool]) -> respx.Route:
    """A llama-server by its answers, loading until `state["loaded"]`; every path it does not
    have is a 404."""
    respx.get(f"{BASE}/props").mock(return_value=httpx.Response(404))
    respx.get(f"{BASE}/v1/models").mock(return_value=httpx.Response(404))
    chat = respx.post(f"{BASE}/v1/chat/completions").mock(
        side_effect=lambda request: httpx.Response(200, json=OK) if state["loaded"] else _loading()
    )
    respx.route(host="llama.local").mock(return_value=httpx.Response(404))
    return chat


@respx.mock
@pytest.mark.parametrize("path", ["/v1/generate", "/v1/generate/stream"])
def test_through_the_app_a_loading_engine_is_a_safe_502_and_answers_once_loaded(
    tmp_path: Path, path: str
) -> None:
    state = {"loaded": False}
    chat = _llama_that_is_loading(state)
    body = {"messages": [{"role": "user", "content": "hi"}]}
    with _app(tmp_path) as client:
        loading = client.post(path, json=body)
        assert loading.status_code == 502, loading.text
        problem = loading.json()["detail"]
        assert problem["retryDisposition"] == "safe"
        assert problem["type"].endswith("#backend-error")
        assert "Loading model" in problem["detail"]
        assert "Outcome unknown" not in problem["detail"]
        assert "Retry-After" not in loading.headers, "llama-server gave none, and none is invented"

        # Nothing is remembered: the next request after it has loaded is simply answered.
        state["loaded"] = True
        if path == "/v1/generate":
            answered = client.post(path, json=body)
            assert answered.status_code == 200, answered.text
    assert chat.called


@respx.mock
def test_through_the_app_an_embedding_replica_that_restarted_is_safe_too(tmp_path: Path) -> None:
    """The probe said yes once (and is kept), then the engine was restarted under this driver
    and is loading: the request that follows is the one the gateway must be able to replay."""
    state = {"loaded": True}
    respx.get(f"{BASE}/props").mock(return_value=httpx.Response(404))
    respx.get(f"{BASE}/v1/models").mock(return_value=httpx.Response(404))
    respx.post(f"{BASE}/v1/rerank").mock(return_value=httpx.Response(501, json={"error": {}}))
    vectors = {
        "object": "list",
        "model": "qwen",
        "data": [{"object": "embedding", "index": 0, "embedding": [0.5, 0.25]}],
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
    }
    respx.post(f"{BASE}/v1/embeddings").mock(
        side_effect=lambda request: (
            httpx.Response(200, json=vectors) if state["loaded"] else _loading()
        )
    )
    respx.route(host="llama.local").mock(return_value=httpx.Response(404))
    with _app(tmp_path) as client:
        first = client.post("/v1/embed", json={"input": ["a passage"]})
        assert first.status_code == 200, first.text
        state["loaded"] = False  # restarted
        loading = client.post("/v1/embed", json={"input": ["a passage"]})
    assert loading.status_code == 502, loading.text
    problem = loading.json()["detail"]
    assert problem["retryDisposition"] == "safe" and "Loading model" in problem["detail"]


@respx.mock
def test_through_the_app_a_loading_reranker_is_a_safe_502(tmp_path: Path) -> None:
    respx.post(f"{BASE}/v1/rerank").mock(return_value=_loading())
    with _app(tmp_path) as client:
        response = client.post("/v1/rerank", json={"query": "q", "documents": ["a", "b"]})
    assert response.status_code == 502, response.text
    problem = response.json()["detail"]
    assert problem["retryDisposition"] == "safe" and "Loading model" in problem["detail"]


# --------------------------------------------------------------------------- #
# what has not moved: the probes' "not now"
# --------------------------------------------------------------------------- #


@respx.mock
@pytest.mark.anyio
@pytest.mark.parametrize(
    "refusal",
    [_loading(), httpx.Response(503, json={"error": {"message": "boom"}})],
    ids=["loading", "any-other-503"],
)
async def test_the_probes_still_treat_a_503_as_not_now(refusal: httpx.Response) -> None:
    """The probes key on the status (`_NOT_NOW`) and not on `disposition`: a model that is still
    loading says nothing of what the backend can do, so it is asked again at the next poll and
    the answer, once it can give one, is the one that is kept. Unchanged by #10."""
    rerank = respx.post(f"{BASE}/v1/rerank").mock(return_value=refusal)
    embed = respx.post(f"{BASE}/v1/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "model": "qwen",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.5]}],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )
    )
    engine = _engine()
    try:
        assert await engine.probe_rerank() is None, "not an answer, and not kept"
        assert await engine.probe_embeddings() is False
        assert await engine.probe_embeddings() is False
        assert rerank.call_count == 3, "asked again each time, until it can say"
        assert embed.call_count == 0, "and nothing was sent to a backend that could not say"

        rerank.mock(
            return_value=httpx.Response(
                501, json={"error": {"message": "start it with --reranking"}}
            )
        )
        assert await engine.probe_embeddings() is True, "once loaded, the answer is believed"
    finally:
        await engine.aclose()
