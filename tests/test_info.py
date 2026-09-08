"""Tests for GET /v1/info."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_info_reports_configured_backend(client: TestClient) -> None:
    response = client.get("/v1/info")
    assert response.status_code == 200
    body = response.json()
    assert body["backend"] == "claude_code_cli"
    assert body["version"]
    # Driver does not self-assert identity; the gateway labels it.
    assert "hemisphere" not in body
