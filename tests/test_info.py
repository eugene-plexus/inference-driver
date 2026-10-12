"""Tests for GET /v1/info."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_inference_driver._generated.models import SettingNotHonoured


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
        model_id="cold-model-alias",
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
    (model,) = body["models"]
    assert model["id"] == "cold-model-alias"
    assert model["capabilities"]["toolCalling"] is True
    assert model["capabilities"]["supportedSettings"] == ["maxTokens"]
    if delay < 3:
        assert model["capabilities"]["imageInput"] is True
        assert model["capabilities"]["maxContextTokens"] == 16384
        assert model["surfaces"] == ["embeddings"]
    else:
        assert model["capabilities"]["imageInput"] is False
        assert model["capabilities"].get("maxContextTokens") is None
        # An unanswered embeddings probe is not a yes: the model is chat.
        assert model["surfaces"] == ["chat"]
        assert sorted(cancelled) == ["context", "embeddings", "image"]
        delay = 0
        (recovered,) = client.get("/v1/info").json()["models"]
        assert recovered["capabilities"]["imageInput"] is True
        assert recovered["capabilities"]["maxContextTokens"] == 16384
        assert recovered["surfaces"] == ["embeddings"]


def test_a_degraded_driver_says_why_it_serves_nothing(tmp_path) -> None:
    """2026-10-10: every Strata driver came up degraded and the gateway,
    reading only `/v1/info`, showed "no model reported" with no reason."""
    from eugene_plexus_inference_driver.app import create_app
    from eugene_plexus_inference_driver.settings import Settings

    config = tmp_path / "config.yaml"
    config.write_text("provider: openai_compat_custom\n", encoding="utf-8")
    with TestClient(create_app(settings=Settings(config_file=config))) as degraded:
        body = degraded.get("/v1/info").json()
        health = degraded.get("/healthz").json()
    assert body["models"] == []
    assert "has no backend" in body["degraded"]
    assert body["degraded"] == health["details"]["adapter_error"]


def test_a_working_driver_is_not_degraded(client: TestClient) -> None:
    assert "degraded" not in client.get("/v1/info").json()


# --------------------------------------------------------------------------- #
# settingsNotHonoured: reported on a chat model and on nothing else
# --------------------------------------------------------------------------- #

GAPS = [{"setting": "parallelToolCalls", "values": [False]}]


async def _yes() -> bool:
    return True


def _gappy(**extra: object) -> SimpleNamespace:
    """A single-model engine whose backend has a gap, in the shape `/v1/info` reads: a hook
    (`settings_not_honoured`, optional like `probe_embeddings`), and whatever else `extra` adds."""
    return SimpleNamespace(
        backend_kind="openai_compat_http",
        routing_locality="local",
        model_id="m",
        supported_settings=["tools"],
        supports_streaming=True,
        supports_tool_calling=True,
        settings_not_honoured=lambda: [
            SettingNotHonoured(setting="parallelToolCalls", values=[False])
        ],
        aclose=AsyncMock(),
        **extra,
    )


def test_a_chat_model_reports_what_its_backend_does_not_honour(client: TestClient) -> None:
    client.app.state.adapter = _gappy()  # type: ignore[attr-defined]
    (model,) = client.get("/v1/info").json()["models"]
    assert model["surfaces"] == ["chat"]
    assert model["capabilities"]["settingsNotHonoured"] == GAPS


@pytest.mark.parametrize(
    ("extra", "surfaces"),
    [
        # A backend that embeds, one that reranks, and one that decides never chat, so a gap in
        # what a chat request is sent says nothing of them.
        ({"probe_embeddings": _yes}, ["embeddings"]),
        ({"probe_rerank": _yes}, ["rerank"]),
        ({"decision_kinds": ["noul"]}, ["decisions"]),
        ({"chat_capable": False}, ["decisions"]),
    ],
    ids=["embeddings", "rerank", "decisions", "not-chat-capable"],
)
def test_a_model_that_does_not_chat_reports_no_gap(
    client: TestClient, extra: dict[str, object], surfaces: list[str]
) -> None:
    client.app.state.adapter = _gappy(**extra)  # type: ignore[attr-defined]
    (model,) = client.get("/v1/info").json()["models"]
    assert model["surfaces"] == surfaces
    assert "settingsNotHonoured" not in model["capabilities"]


def test_an_engine_whose_backend_has_no_gap_reports_none_not_an_empty_list(
    client: TestClient,
) -> None:
    adapter = _gappy()
    adapter.settings_not_honoured = lambda: []
    client.app.state.adapter = adapter  # type: ignore[attr-defined]
    (model,) = client.get("/v1/info").json()["models"]
    # Absent, not an empty list: absent says everything carried is honoured.
    assert "settingsNotHonoured" not in model["capabilities"]


def test_an_engine_with_no_hook_at_all_reports_none(client: TestClient) -> None:
    """An engine from before the mechanism, or one that never has a gap to name."""
    adapter = _gappy()
    del adapter.settings_not_honoured
    client.app.state.adapter = adapter  # type: ignore[attr-defined]
    (model,) = client.get("/v1/info").json()["models"]
    assert model["surfaces"] == ["chat"]
    assert "settingsNotHonoured" not in model["capabilities"]
