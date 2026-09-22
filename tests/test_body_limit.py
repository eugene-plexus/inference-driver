"""Every request-body inference path is bounded, not only generation.

`InferenceBodyLimit` reads at most 16 MiB before FastAPI parses a byte,
and until 2026-09-22 it was mounted on `/v1/generate` and
`/v1/generate/stream` alone. `/v1/embed` takes a batch of inputs and
`/v1/decide` a state plus questions; both were read to the end and
parsed whole, however large -- the cheapest memory exhaustion on the
driver, one request away.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_inference_driver import body_limit

BOUNDED = ["/v1/generate", "/v1/generate/stream", "/v1/embed", "/v1/decide"]


@pytest.mark.parametrize("path", BOUNDED)
def test_a_chunked_body_past_the_limit_is_refused(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    monkeypatch.setattr(body_limit, "MAX_BODY_BYTES", 128)
    response = client.post(path, content=iter([b" " * 80, b" " * 80]))

    assert response.status_code == 413, response.text
    assert response.json()["detail"]["title"] == "Request too large"
    assert "16 MiB" in response.text


@pytest.mark.parametrize("path", BOUNDED)
def test_a_declared_length_past_the_limit_is_refused_unread(client: TestClient, path: str) -> None:
    response = client.post(path, content=b"{}", headers={"content-length": "9" * 5000})
    assert response.status_code == 413, response.text
