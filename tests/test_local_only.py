"""A protected request must not reach even the first upstream operation."""

import pytest

from eugene_plexus_inference_driver._generated.models import (
    BackendKind,
    EmbedResponse,
    FinishReason,
    GenerateResponse,
)
from eugene_plexus_inference_driver.engines.base import Chunk


class CountingEngine:
    backend_kind = BackendKind.openai_compat_http
    supports_tool_calling = True
    supports_embeddings = True
    supports_streaming = True

    def __init__(self, locality):
        self.routing_locality = locality
        self.calls = []

    async def generate(self, body):
        self.calls.append(body)
        return GenerateResponse(content="local", finishReason=FinishReason.stop)

    async def stream(self, body):
        result = await self.generate(body)
        yield Chunk(text="local")
        yield Chunk(done=True, result=result)

    async def embed(self, inputs):
        self.calls.append(inputs)
        return EmbedResponse(embeddings=[[1.0]])


@pytest.mark.parametrize("locality", ["external", "unknown", None])
@pytest.mark.parametrize("path", ["/v1/generate", "/v1/generate/stream", "/v1/embed"])
def test_nonlocal_engine_never_receives_protected_content(client, locality, path):
    engine = CountingEngine(locality)
    client.app.state.adapter = engine
    body = (
        {"input": ["private"]}
        if path.endswith("embed")
        else {"messages": [{"role": "user", "content": "private"}]}
    )
    response = client.post(path, json={**body, "localOnly": True})
    assert response.status_code == 403, response.text
    assert engine.calls == []
    assert "local-only" in response.text
    assert "private" not in response.text


@pytest.mark.parametrize("path", ["/v1/generate", "/v1/generate/stream", "/v1/embed"])
def test_local_engine_serves_protected_work(client, path):
    engine = CountingEngine("local")
    client.app.state.adapter = engine
    body = (
        {"input": ["private"]}
        if path.endswith("embed")
        else {"messages": [{"role": "user", "content": "private"}]}
    )
    response = client.post(path, json={**body, "localOnly": True})
    assert response.status_code == 200, response.text
    assert len(engine.calls) == 1


def test_stale_local_info_cannot_authorize_replacement_engine(client):
    client.app.state.adapter = CountingEngine("local")
    info = client.get("/v1/info").json()
    assert info.get("locality") == "local" and info.get("localOnlyEnforced") is True
    replacement = CountingEngine("external")
    client.app.state.adapter = replacement
    response = client.post(
        "/v1/generate",
        json={"localOnly": True, "messages": [{"role": "user", "content": "private"}]},
    )
    assert response.status_code == 403
    assert not replacement.calls


def test_unrestricted_work_can_still_use_external_engine(client):
    engine = CountingEngine("external")
    client.app.state.adapter = engine
    response = client.post(
        "/v1/generate", json={"messages": [{"role": "user", "content": "ordinary"}]}
    )
    assert response.status_code == 200
    assert len(engine.calls) == 1


@pytest.mark.parametrize(
    "provider,declared,managed,expected",
    [
        ("openai_compat_custom", None, False, "unknown"),
        ("openai_compat_custom", "local", False, "local"),
        ("openai_compat_custom", "external", False, "external"),
        ("openai_compat_custom", None, True, "local"),
        ("ollama_local", None, False, "unknown"),
        ("lmstudio_local", "local", False, "local"),
        ("claude_subscription", "local", False, "external"),
        ("chatgpt_subscription", "local", False, "external"),
        ("openai", "local", False, "external"),
        ("openrouter", "local", True, "external"),
    ],
)
def test_engine_classification_is_captured_at_construction(provider, declared, managed, expected):
    from eugene_plexus_inference_driver.app import build_engine_with

    config = {
        "provider": provider,
        "apiKey": "not-a-secret-fixture",
        "backendLocality": declared,
        "baseUrl": "http://127.0.0.1:12345",
        "runtimeName": "managed" if managed else None,
    }
    engine = build_engine_with(config.get, resolve_runtime=lambda _: "http://managed:12345")
    assert engine.routing_locality == expected
    config["backendLocality"] = "external"
    assert engine.routing_locality == expected
