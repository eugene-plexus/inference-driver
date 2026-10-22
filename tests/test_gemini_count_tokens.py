from pathlib import Path

import httpx
import respx

from .test_gemini import BASE, CHAT, _chat, _error, _sent, _start


@respx.mock
def test_count_tokens_counts_the_full_translated_prompt(tmp_path: Path) -> None:
    client = _start(tmp_path)
    route = respx.post(f"{BASE}/models/{CHAT}:countTokens").mock(
        return_value=httpx.Response(200, json={"totalTokens": 73})
    )
    with client:
        response = client.post(
            "/v1/generate/count",
            json=_chat(
                messages=[
                    {"role": "system", "content": "Be brief"},
                    {"role": "user", "content": "Hi"},
                ]
            ),
        )
    assert response.status_code == 200, response.text
    assert response.json() == {"promptTokens": 73}
    payload = _sent(route)["generateContentRequest"]
    assert payload["model"] == f"models/{CHAT}"
    assert payload["systemInstruction"]["parts"] == [{"text": "Be brief"}]
    assert payload["contents"][0]["parts"] == [{"text": "Hi"}]


@respx.mock
def test_count_tokens_names_the_observed_provider_refusal(tmp_path: Path) -> None:
    client = _start(tmp_path)
    respx.post(f"{BASE}/models/{CHAT}:countTokens").mock(
        return_value=_error(400, "INVALID_ARGUMENT", "countTokens: malformed content")
    )
    with client:
        response = client.post("/v1/generate/count", json=_chat())
    assert response.status_code == 400
    assert "countTokens: malformed content" in response.text


@respx.mock
def test_count_tokens_refuses_a_malformed_count(tmp_path: Path) -> None:
    client = _start(tmp_path)
    respx.post(f"{BASE}/models/{CHAT}:countTokens").mock(
        return_value=httpx.Response(200, json={"totalTokens": -1})
    )
    with client:
        response = client.post("/v1/generate/count", json=_chat())
    assert response.status_code == 502
    assert "totalTokens" in response.text
