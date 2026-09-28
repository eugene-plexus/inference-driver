"""POST /v1/transcribe through an account and a single llama-server (P3b).

Measured 2026-09-28 (`provider-accounts-measurement.md` sections 3 and 8):
OpenRouter takes OpenAI's multipart form and answers `json` or
`verbose_json`; `llama-server` b11235 answers only with a projector that
hears, only `json`, and leaves Qwen3-ASR's preamble in the text. Nine of the
ten fail against the driver before P3b, which had no `/v1/transcribe`; the
404 case passes either way, since a missing route is a 404 too.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import httpx
import respx
from fastapi.testclient import TestClient

from eugene_plexus_inference_driver.app import create_app
from eugene_plexus_inference_driver.settings import Settings

OPENROUTER = "https://openrouter.ai/api"
LLAMA = "http://llama.local"
FOX = b"ID3\x04" + bytes(range(256)) * 8
LISTING = {
    "data": [
        {
            "id": "openai/whisper-large-v3-turbo",
            "architecture": {"input_modalities": ["audio"], "output_modalities": ["transcription"]},
            "supported_parameters": [],
        },
        {
            "id": "mistralai/mistral-nemo",
            "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
            "supported_parameters": ["max_tokens"],
        },
    ]
}
WHISPER = "openai/whisper-large-v3-turbo"
PREAMBLE = "language English<asr_text>The quick brown fox jumps over the lazy dog."


def _config(tmp_path: Path, **values: Any) -> Path:
    config = tmp_path / f"{values['provider']}.yaml"
    config.write_text(json.dumps({"apiKey": "sk-test", **values}), "utf-8")
    return config


def _ready(client: TestClient) -> dict[str, Any]:
    for _ in range(250):
        info = client.get("/v1/info").json()
        catalogue = info.get("catalogue") or {}
        if catalogue.get("refreshedAt") or catalogue.get("error"):
            return info
    raise AssertionError("catalogue never read")


def ask(model: str | None = WHISPER, **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "audio": {
            "data": base64.b64encode(FOX).decode(),
            "filename": "fox.mp3",
            "mediaType": "audio/mpeg",
        },
        **extra,
    }
    if model is not None:
        body["model"] = model
    return body


def _set(values: dict[str, Any]) -> dict[str, Any]:
    """What was said: the route serialises every field, nulls included."""
    return {k: v for k, v in values.items() if v is not None}


def _fields(request: httpx.Request) -> bytes:
    return request.content if isinstance(request.content, bytes) else b"".join(request.stream)


def _openrouter(tmp_path: Path, answer: httpx.Response) -> tuple[TestClient, respx.Route]:
    respx.get(f"{OPENROUTER}/v1/models/user").mock(return_value=httpx.Response(200, json=LISTING))
    respx.get(f"{OPENROUTER}/v1/images/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    route = respx.post(f"{OPENROUTER}/v1/audio/transcriptions").mock(return_value=answer)
    app = create_app(settings=Settings(config_file=_config(tmp_path, provider="openrouter")))
    return TestClient(app), route


def _llama(
    tmp_path: Path, answer: httpx.Response, *, hears: bool = True
) -> tuple[TestClient, respx.Route]:
    respx.get(f"{LLAMA}/props").mock(
        return_value=httpx.Response(200, json={"modalities": {"vision": False, "audio": hears}})
    )
    respx.get(f"{LLAMA}/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "asr"}]})
    )
    route = respx.post(f"{LLAMA}/v1/audio/transcriptions").mock(return_value=answer)
    config = _config(tmp_path, provider="openai_compat_custom", baseUrl=LLAMA, modelId="asr")
    return TestClient(create_app(settings=Settings(config_file=config))), route


# --------------------------------------------------------------------------- #
# An account
# --------------------------------------------------------------------------- #


@respx.mock
def test_an_accounts_transcription_model_is_asked_in_openais_multipart_form(tmp_path: Path) -> None:
    client, upstream = _openrouter(
        tmp_path,
        httpx.Response(200, json={"text": " The quick brown fox.", "usage": {"seconds": 3.5}}),
    )
    with client:
        info = _ready(client)
        response = client.post("/v1/transcribe", json=ask(language="en", prompt="Foxes."))
    surfaces = {m["id"]: m["surfaces"] for m in info["models"]}
    assert surfaces[WHISPER] == ["transcription"]
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["text"] == " The quick brown fox." and _set(body["usage"]) == {"seconds": 3.5}
    sent = _fields(upstream.calls[0].request)
    assert b'name="model"\r\n\r\nopenai/whisper-large-v3-turbo\r\n' in sent
    assert b'name="language"\r\n\r\nen\r\n' in sent and b'name="prompt"\r\n\r\nFoxes.\r\n' in sent
    assert b'filename="fox.mp3"' in sent and FOX in sent
    # `json` is every backend's default, and llama-server refuses the rest.
    assert b'name="response_format"' not in sent


@respx.mock
def test_verbose_asks_for_verbose_json_with_each_granularity(tmp_path: Path) -> None:
    segments = [{"id": 0, "start": 0.0, "end": 1.5, "text": "The quick brown fox."}]
    client, upstream = _openrouter(
        tmp_path,
        httpx.Response(
            200,
            json={
                "text": "The quick brown fox.",
                "language": "english",
                "duration": 1.5,
                "segments": segments,
            },
        ),
    )
    with client:
        _ready(client)
        response = client.post(
            "/v1/transcribe", json=ask(verbose=True, timestampGranularities=["word", "segment"])
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["language"], body["duration"], body["segments"]) == ("english", 1.5, segments)
    sent = _fields(upstream.calls[0].request)
    assert b'name="response_format"\r\n\r\nverbose_json\r\n' in sent
    assert sent.count(b'name="timestamp_granularities[]"') == 2


@respx.mock
def test_a_chat_model_is_refused_before_any_audio_is_sent(tmp_path: Path) -> None:
    client, upstream = _openrouter(tmp_path, httpx.Response(200, json={"text": "x"}))
    with client:
        _ready(client)
        response = client.post("/v1/transcribe", json=ask("mistralai/mistral-nemo"))
    assert response.status_code == 400, response.text
    assert response.json()["detail"]["type"].endswith("#transcription-unsupported")
    assert not upstream.calls


@respx.mock
def test_a_model_the_account_does_not_serve_is_a_404(tmp_path: Path) -> None:
    client, upstream = _openrouter(tmp_path, httpx.Response(200, json={"text": "x"}))
    with client:
        _ready(client)
        response = client.post("/v1/transcribe", json=ask("openai/whisper-nope"))
    assert response.status_code == 404 and not upstream.calls


@respx.mock
def test_the_backends_refusal_is_relayed_with_its_words(tmp_path: Path) -> None:
    refusal = {"error": {"message": "Unsupported file format: fox.xyz", "code": 400}}
    client, _ = _openrouter(tmp_path, httpx.Response(400, json=refusal))
    with client:
        _ready(client)
        response = client.post("/v1/transcribe", json=ask())
    assert response.status_code == 400, response.text
    assert "Unsupported file format" in response.json()["detail"]["detail"]


@respx.mock
def test_bad_audio_and_timestamps_without_verbose_are_refused_first(tmp_path: Path) -> None:
    client, upstream = _openrouter(tmp_path, httpx.Response(200, json={"text": "x"}))
    with client:
        _ready(client)
        bad = ask()
        bad["audio"]["data"] = "not base64!"
        responses = [
            client.post("/v1/transcribe", json=bad),
            client.post("/v1/transcribe", json=ask(timestampGranularities=["word"])),
        ]
    assert [r.status_code for r in responses] == [400, 400], [r.text for r in responses]
    assert not upstream.calls


# --------------------------------------------------------------------------- #
# A single llama-server
# --------------------------------------------------------------------------- #


@respx.mock
def test_llama_server_with_a_projector_that_hears_transcribes_and_says_so(tmp_path: Path) -> None:
    answer = {
        "type": "transcript.text.done",
        "text": PREAMBLE,
        "usage": {"type": "tokens", "input_tokens": 61, "output_tokens": 14, "total_tokens": 75},
    }
    client, upstream = _llama(tmp_path, httpx.Response(200, json=answer))
    with client:
        info = client.get("/v1/info").json()
        response = client.post("/v1/transcribe", json=ask("asr"))
    assert info["models"][0]["surfaces"] == ["chat", "transcription"]
    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["text"], body["language"]) == (
        "The quick brown fox jumps over the lazy dog.",
        "english",
    )
    assert _set(body["usage"]) == {"inputTokens": 61, "outputTokens": 14, "totalTokens": 75}
    assert b'name="model"\r\n\r\nasr\r\n' in _fields(upstream.calls[0].request)


@respx.mock
def test_llama_server_without_one_does_not_transcribe(tmp_path: Path) -> None:
    client, upstream = _llama(tmp_path, httpx.Response(200, json={"text": "x"}), hears=False)
    with client:
        info = client.get("/v1/info").json()
        response = client.post("/v1/transcribe", json=ask("asr"))
    assert info["models"][0]["surfaces"] == ["chat"]
    assert response.status_code == 400 and not upstream.calls


@respx.mock
def test_verbose_on_llama_server_is_its_own_refusal_relayed(tmp_path: Path) -> None:
    refusal = {
        "error": {
            "code": 400,
            "message": "Only 'json' response_format is supported",
            "type": "invalid_request_error",
        }
    }
    client, _ = _llama(tmp_path, httpx.Response(400, json=refusal))
    with client:
        response = client.post("/v1/transcribe", json=ask("asr", verbose=True))
    assert response.status_code == 400, response.text
    assert "Only 'json' response_format is supported" in response.json()["detail"]["detail"]


# --------------------------------------------------------------------------- #
# The body limit: this path's own
# --------------------------------------------------------------------------- #


def test_this_path_takes_more_than_the_json_limit_and_less_than_forty(client: TestClient) -> None:
    over_json = client.post("/v1/transcribe", content=b" " * (17 * 1024 * 1024))
    assert over_json.status_code != 413, "a 25 MiB upload is 33 MiB of base64"
    too_big = client.post(
        "/v1/transcribe", content=b"{}", headers={"content-length": str(40 * 1024 * 1024)}
    )
    assert too_big.status_code == 413 and "36 MiB" in too_big.text
