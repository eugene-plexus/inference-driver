"""Tests for degraded-mode startup.

The driver MUST come up even when adapter construction fails — otherwise a
broken config makes the driver unreachable, and the operator can't fix it
through the UI. (OpenClaw lesson.) These tests verify that contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_inference_driver.app import create_app
from eugene_plexus_inference_driver.settings import Settings


@pytest.fixture
def degraded_app_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """A config that picks the OpenAI provider without an API key.

    The OPENAI_API_KEY env var is also unset, so engine construction
    raises and the lifespan falls into degraded mode.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """\
provider: openai
modelId: gpt-4o
port: 8081
logLevel: INFO
requestTimeoutSeconds: 120
""",
        encoding="utf-8",
    )
    return Settings(config_file=config_path)


def test_driver_comes_up_with_broken_engine_config(
    degraded_app_settings: Settings,
) -> None:
    app = create_app(settings=degraded_app_settings)
    with TestClient(app) as client:
        # /healthz reachable, reports degraded
        health = client.get("/healthz")
        assert health.status_code == 200
        body = health.json()
        assert body["status"] == "degraded"
        assert "no API key" in (body.get("details") or {}).get("adapter_error", "")


def test_config_endpoints_work_in_degraded_mode(degraded_app_settings: Settings) -> None:
    """The whole point: config + schema endpoints must stay live so the operator
    can fix the broken config via PATCH."""
    app = create_app(settings=degraded_app_settings)
    with TestClient(app) as client:
        schema = client.get("/v1/config/schema")
        assert schema.status_code == 200
        assert schema.json()["component"] == "inference-driver"

        doc = client.get("/v1/config")
        assert doc.status_code == 200
        assert doc.json()["provider"] == "openai"

        patch = client.patch(
            "/v1/config",
            json={"apiKey": "sk-test-fix"},
        )
        assert patch.status_code == 200
        assert "apiKey" in patch.json()["applied"]


def test_generate_returns_503_in_degraded_mode(degraded_app_settings: Settings) -> None:
    app = create_app(settings=degraded_app_settings)
    with TestClient(app) as client:
        response = client.post(
            "/v1/generate",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 503
        detail = response.json()["detail"]
        assert detail["title"] == "Engine not configured"
        assert "Update the configuration" in detail["detail"]
