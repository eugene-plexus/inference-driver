"""Tests for the agent safe-mode contract.

Per specs/openapi/inference-driver.yaml: when started with
`EUGENE_PLEXUS_DRIVER_SAFE_MODE=1` the driver must

  - skip loading its persisted config file (defaults only)
  - still expose /v1/config endpoints (operator can repair via UI)
  - report /healthz as `degraded` with `safeMode: true`
  - return 503 from /v1/generate (no engine in safe mode)
  - allow PATCH /v1/config to write to the on-disk file as normal
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_inference_driver.app import create_app
from eugene_plexus_inference_driver.settings import Settings


@pytest.fixture
def safe_mode_settings(tmp_path: Path) -> Settings:
    # Pre-write a config file with values that would normally be loaded.
    # Safe mode must ignore this file and use defaults instead.
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "provider": "openai",
                "apiKey": "sk-pre-existing",
                "modelId": "gpt-4o",
                "logLevel": "DEBUG",
            }
        ),
        encoding="utf-8",
    )
    return Settings(config_file=config, safe_mode=True)


@pytest.fixture
def safe_mode_app(safe_mode_settings: Settings) -> FastAPI:
    return create_app(settings=safe_mode_settings)


@pytest.fixture
def safe_mode_client(safe_mode_app: FastAPI) -> Iterator[TestClient]:
    with TestClient(safe_mode_app) as c:
        yield c


def test_healthz_reports_safe_mode_and_degraded(safe_mode_client: TestClient) -> None:
    response = safe_mode_client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["safeMode"] is True


def test_config_get_returns_defaults_not_disk_values(safe_mode_client: TestClient) -> None:
    """The fixture wrote `logLevel: DEBUG` to disk; safe mode should ignore
    that and serve the built-in default `INFO`."""
    response = safe_mode_client.get("/v1/config")
    assert response.status_code == 200
    body = response.json()
    assert body["logLevel"] == "INFO", "safe mode should not have read disk config"
    # provider has no default in v0.1; safe mode leaves it as the built-in
    # default ("claude_subscription") which is the field's `default`.
    assert body["provider"] == "claude_subscription"


def test_generate_returns_503_in_safe_mode(safe_mode_client: TestClient) -> None:
    response = safe_mode_client.post(
        "/v1/generate",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 503


def test_patch_config_writes_to_disk_in_safe_mode(
    safe_mode_client: TestClient, safe_mode_settings: Settings
) -> None:
    """The operator's repair must persist so the next non-safe-mode boot
    picks it up — that's the whole point of safe mode as a recovery flow."""
    response = safe_mode_client.patch("/v1/config", json={"logLevel": "WARNING"})
    assert response.status_code == 200
    body = response.json()
    assert "logLevel" in body["applied"]

    # Confirm the on-disk file was rewritten.
    on_disk = yaml.safe_load(safe_mode_settings.config_file.read_text(encoding="utf-8"))
    assert on_disk["logLevel"] == "WARNING"
