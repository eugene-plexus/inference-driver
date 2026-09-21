"""Tests for GET /v1/info."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient


def test_info_reports_configured_backend(client: TestClient) -> None:
    response = client.get("/v1/info")
    assert response.status_code == 200
    body = response.json()
    assert body["backend"] == "claude_code_cli"
    assert body["version"]
    # Driver does not self-assert identity; the gateway labels it.
    assert "hemisphere" not in body


@pytest.mark.parametrize("delay", [1.5, 30.0])
def test_optional_probes_cannot_block_local_policy_confirmation(
    client: TestClient, delay: float
) -> None:
    """The gateway has four seconds to reconfirm policy before it can wake a model."""
    cancelled = []

    async def probe(name, value):
        try:
            await asyncio.sleep(delay)
            return value
        except asyncio.CancelledError:
            cancelled.append(name)
            raise

    client.app.state.adapter = SimpleNamespace(
        backend_kind="openai_compat_http",
        routing_locality="local",
        runtime="cold-model",
        supported_settings=["maxTokens"],
        supports_streaming=True,
        supports_tool_calling=True,
        probe_image_input=lambda: probe("image", True),
        context_window=lambda: probe("context", 16384),
        probe_embeddings=lambda: probe("embeddings", True),
        aclose=AsyncMock(),
    )
    started = time.perf_counter()
    response = client.get("/v1/info")
    elapsed = time.perf_counter() - started
    assert response.status_code == 200
    assert elapsed < 3.5, "optional probes exhausted the gateway's policy-confirmation budget"
    body = response.json()
    assert body["locality"] == "local" and body["localOnlyEnforced"] is True
    assert body["runtime"] == "cold-model"
    assert body["capabilities"]["toolCalling"] is True
    assert body["capabilities"]["supportedSettings"] == ["maxTokens"]
    if delay < 3:
        assert body["capabilities"]["imageInput"] is True
        assert body["capabilities"]["maxContextTokens"] == 16384
        assert body["capabilities"]["embeddings"] is True
    else:
        assert body["capabilities"]["imageInput"] is False
        assert body["capabilities"]["maxContextTokens"] is None
        assert body["capabilities"]["embeddings"] is None
        assert sorted(cancelled) == ["context", "embeddings", "image"]
        delay = 0
        recovered = client.get("/v1/info").json()["capabilities"]
        assert recovered["imageInput"] is True
        assert recovered["maxContextTokens"] == 16384
        assert recovered["embeddings"] is True
