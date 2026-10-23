from pathlib import Path

import httpx
import respx

from .test_gemini import BASE, _error, _sent, _start


@respx.mock
def test_dimensions_reach_gemini(tmp_path: Path) -> None:
    client = _start(tmp_path)
    route = respx.post(f"{BASE}/models/gemini-embedding-001:batchEmbedContents").mock(
        return_value=httpx.Response(
            200, json={"embeddings": [{"values": [1, 2]}, {"values": [3, 4]}]}
        )
    )
    with client:
        response = client.post(
            "/v1/embed",
            json={"model": "gemini-embedding-001", "input": ["a", "b"], "dimensions": 2},
        )
    assert response.status_code == 200, response.text
    assert [r["outputDimensionality"] for r in _sent(route)["requests"]] == [2, 2]
    assert response.json()["embeddings"] == [[1, 2], [3, 4]]
    assert "dimensions" not in response.headers.get("x-eugene-plexus-ignored-settings", "")


@respx.mock
def test_dimensions_refusal_names_the_field_and_cause(tmp_path: Path) -> None:
    client = _start(tmp_path)
    respx.post(f"{BASE}/models/gemini-embedding-001:batchEmbedContents").mock(
        return_value=_error(400, "INVALID_ARGUMENT", "outputDimensionality is out of range")
    )
    with client:
        response = client.post(
            "/v1/embed", json={"model": "gemini-embedding-001", "input": ["a"], "dimensions": 2}
        )
    assert response.status_code == 400
    assert "dimensions" in response.text and "outputDimensionality is out of range" in response.text
