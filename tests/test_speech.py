"""POST /v1/speak through an OpenAI-compatible account and ElevenLabs (P3a).

Measured 2026-09-28 (`provider-accounts-measurement.md` sections 3 and 8):
OpenRouter's speech route takes mp3 and pcm only and defaults to pcm;
ElevenLabs keys in `xi-api-key`, puts the voice in the path and the format
in the query, and a scoped key's refusal names the permission it lacks.
Every test here fails against the driver as it was before P3a, which had no
`/v1/speak`, no speech formats and no ElevenLabs engine.
"""

from __future__ import annotations

import json
import struct
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from eugene_plexus_inference_driver.app import create_app
from eugene_plexus_inference_driver.config import FIELDS
from eugene_plexus_inference_driver.engines._catalogue import from_openrouter
from eugene_plexus_inference_driver.settings import Settings

OPENROUTER = "https://openrouter.ai/api"
ELEVEN = "https://api.elevenlabs.io"
VOICE = "21m00Tcm4TlvDq8ikWAM"
MP3 = b"ID3\x04" + bytes(range(200))
PCM = bytes(range(256)) * 4
LISTING = {
    "data": [
        {
            "id": "hexgrad/kokoro-82m",
            "architecture": {"input_modalities": ["text"], "output_modalities": ["speech"]},
            "supported_voices": ["af_heart", "af_bella"],
            "supported_parameters": [],
        },
        {
            "id": "mistralai/mistral-nemo",
            "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
            "supported_parameters": ["max_tokens"],
        },
    ]
}
EL_MODELS = [
    {"model_id": "eleven_flash_v2_5", "name": "Flash v2.5", "can_do_text_to_speech": True},
    {"model_id": "eleven_english_sts_v2", "name": "STS", "can_do_text_to_speech": False},
]
EL_VOICES = {"voices": [{"voice_id": VOICE, "name": "Rachel"}]}
MISSING = {
    "detail": {
        "type": "authentication_error",
        "code": "unauthorized",
        "message": "The API key you used is missing the permission models_read to "
        "execute this operation.",
        "status": "missing_permissions",
    }
}


def _config(tmp_path: Path, **values: Any) -> Path:
    config = tmp_path / f"{values['provider']}.yaml"
    config.write_text(json.dumps({"apiKey": "sk-test", **values}), "utf-8")
    return config


def _ready(client: TestClient) -> dict[str, Any]:
    for _ in range(250):
        info = client.get("/v1/info").json()
        if (info.get("catalogue") or {}).get("refreshedAt") or (info.get("catalogue") or {}).get(
            "error"
        ):
            return info
        time.sleep(0.02)
    raise AssertionError("catalogue never read")


def _audio(body: bytes, media: str = "audio/mpeg") -> httpx.Response:
    return httpx.Response(200, content=body, headers={"content-type": media})


def speak(model: str, fmt: str | None = "mp3", **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "input": "Hello there.",
        "voice": extra.pop("voice", "af_heart"),
    }
    if fmt is not None:
        body["format"] = fmt
    return {**body, **extra}


# --------------------------------------------------------------------------- #
# OpenRouter, through the OpenAI-compatible engine
# --------------------------------------------------------------------------- #


def test_the_listing_says_a_model_speaks_in_which_voices_and_formats() -> None:
    kokoro, nemo = from_openrouter(LISTING)
    assert kokoro.surfaces == ["speech"] and kokoro.voices == ["af_heart", "af_bella"]
    assert kokoro.capabilities is not None
    assert [f.value for f in kokoro.capabilities.speechFormats] == ["mp3", "pcm", "wav"]
    assert nemo.capabilities is not None and not nemo.capabilities.speechFormats


def _openrouter(tmp_path: Path, upstream: httpx.Response):
    respx.get(f"{OPENROUTER}/v1/models/user").mock(return_value=httpx.Response(200, json=LISTING))
    route = respx.post(f"{OPENROUTER}/v1/audio/speech").mock(return_value=upstream)
    return TestClient(
        create_app(settings=Settings(config_file=_config(tmp_path, provider="openrouter")))
    ), route


@respx.mock
def test_speech_is_asked_for_in_openais_shape_and_streamed_back(tmp_path: Path) -> None:
    client, upstream = _openrouter(tmp_path, _audio(MP3))
    with client:
        _ready(client)
        response = client.post(
            "/v1/speak", json=speak("hexgrad/kokoro-82m", speed=1.1, instructions="cheerful")
        )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "audio/mpeg" and response.content == MP3
    sent = json.loads(upstream.calls[0].request.content)
    assert sent == {
        "model": "hexgrad/kokoro-82m",
        "input": "Hello there.",
        "voice": "af_heart",
        "response_format": "mp3",
        "speed": 1.1,
        "instructions": "cheerful",
    }


@respx.mock
def test_the_format_is_always_sent_mp3_by_default(tmp_path: Path) -> None:
    """OpenRouter's own default is pcm where OpenAI's is mp3 (measured)."""
    client, upstream = _openrouter(tmp_path, _audio(MP3))
    with client:
        _ready(client)
        response = client.post("/v1/speak", json=speak("hexgrad/kokoro-82m", fmt=None))
    assert response.status_code == 200, response.text
    assert json.loads(upstream.calls[0].request.content)["response_format"] == "mp3"
    assert response.headers["content-type"] == "audio/mpeg"


@respx.mock
def test_a_wav_is_made_here_from_openrouters_pcm(tmp_path: Path) -> None:
    client, upstream = _openrouter(tmp_path, _audio(PCM, "audio/pcm"))
    with client:
        _ready(client)
        response = client.post("/v1/speak", json=speak("hexgrad/kokoro-82m", fmt="wav"))
    assert response.status_code == 200, response.text
    assert json.loads(upstream.calls[0].request.content)["response_format"] == "pcm"
    assert response.headers["content-type"] == "audio/wav"
    raw = response.content
    assert raw[:4] == b"RIFF" and raw[8:12] == b"WAVE" and raw[36:40] == b"data"
    # Sizes unknown while streaming: the 0xFFFFFFFF convention.
    assert struct.unpack("<I", raw[4:8])[0] == 0xFFFFFFFF == struct.unpack("<I", raw[40:44])[0]
    assert struct.unpack("<I", raw[24:28])[0] == 24000
    assert raw[44:] == PCM


@pytest.mark.parametrize("fmt", ["opus", "aac", "flac"])
@respx.mock
def test_a_format_openrouter_cannot_make_is_refused_naming_what_it_can(
    tmp_path: Path, fmt: str
) -> None:
    client, upstream = _openrouter(tmp_path, _audio(MP3))
    with client:
        _ready(client)
        response = client.post("/v1/speak", json=speak("hexgrad/kokoro-82m", fmt=fmt))
    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert detail["type"].endswith("#speech-refused") and "mp3, pcm, wav" in detail["detail"]
    assert not upstream.called


@respx.mock
def test_a_model_that_does_not_speak_is_refused(tmp_path: Path) -> None:
    client, upstream = _openrouter(tmp_path, _audio(MP3))
    with client:
        _ready(client)
        response = client.post("/v1/speak", json=speak("mistralai/mistral-nemo"))
    assert response.status_code == 400, response.text
    assert "does not speak" in response.json()["detail"]["detail"]
    assert not upstream.called


@pytest.mark.parametrize(
    ("status", "code", "kind"),
    [
        (400, 400, "#backend-rejected-request"),
        (401, 502, "#backend-credential-refused"),
    ],
)
@respx.mock
def test_a_backend_refusal_before_the_first_byte_is_a_status(
    tmp_path: Path, status: int, code: int, kind: str
) -> None:
    refusal = httpx.Response(status, json={"error": {"message": "unknown voice 'alloy'"}})
    client, _ = _openrouter(tmp_path, refusal)
    with client:
        _ready(client)
        response = client.post("/v1/speak", json=speak("hexgrad/kokoro-82m", voice="alloy"))
    assert response.status_code == code, response.text
    assert response.json()["detail"]["type"].endswith(kind)
    assert "unknown voice" in response.json()["detail"]["detail"]


# --------------------------------------------------------------------------- #
# ElevenLabs
# --------------------------------------------------------------------------- #


def _eleven(
    tmp_path: Path,
    *,
    models: httpx.Response | None = None,
    voices: httpx.Response | None = None,
    upstream: httpx.Response | None = None,
):
    respx.get(f"{ELEVEN}/v1/models").mock(
        return_value=models or httpx.Response(200, json=EL_MODELS)
    )
    respx.get(f"{ELEVEN}/v1/voices").mock(
        return_value=voices or httpx.Response(200, json=EL_VOICES)
    )
    route = respx.post(url__regex=rf"{ELEVEN}/v1/text-to-speech/.*/stream").mock(
        return_value=upstream or _audio(MP3)
    )
    config = _config(tmp_path, provider="elevenlabs")
    return TestClient(create_app(settings=Settings(config_file=config))), route


@respx.mock
def test_an_elevenlabs_account_lists_its_speech_models_and_voices(tmp_path: Path) -> None:
    client, _ = _eleven(tmp_path)
    with client:
        info = _ready(client)
    assert info["backend"] == "elevenlabs_http"
    assert [m["id"] for m in info["models"]] == ["eleven_flash_v2_5"]
    model = info["models"][0]
    assert model["surfaces"] == ["speech"] and model["voices"] == [VOICE]
    assert model["capabilities"]["speechFormats"] == ["mp3", "opus", "pcm", "wav"]


@respx.mock
def test_elevenlabs_is_asked_in_its_own_shape(tmp_path: Path) -> None:
    client, upstream = _eleven(tmp_path)
    with client:
        _ready(client)
        response = client.post("/v1/speak", json=speak("eleven_flash_v2_5", voice=VOICE, speed=1.1))
    assert response.status_code == 200, response.text
    assert response.content == MP3 and response.headers["content-type"] == "audio/mpeg"
    call = upstream.calls[0].request
    assert call.url.path == f"/v1/text-to-speech/{VOICE}/stream"
    assert call.url.params["output_format"] == "mp3_44100_128"
    assert call.headers["xi-api-key"] == "sk-test" and "authorization" not in call.headers
    assert json.loads(call.content) == {
        "text": "Hello there.",
        "model_id": "eleven_flash_v2_5",
        "voice_settings": {"speed": 1.1},
    }


@respx.mock
def test_an_elevenlabs_wav_is_its_pcm_with_a_header(tmp_path: Path) -> None:
    client, upstream = _eleven(tmp_path, upstream=_audio(PCM, "audio/pcm"))
    with client:
        _ready(client)
        response = client.post("/v1/speak", json=speak("eleven_flash_v2_5", fmt="wav", voice=VOICE))
    assert upstream.calls[0].request.url.params["output_format"] == "pcm_24000"
    assert response.content[:4] == b"RIFF" and response.content[44:] == PCM


@pytest.mark.parametrize(
    ("extra", "said"),
    [
        ({"format": "aac"}, "mp3, opus, pcm, wav"),
        ({"format": "flac"}, "mp3, opus, pcm, wav"),
        ({"instructions": "whisper"}, "instructions"),
    ],
)
@respx.mock
def test_what_elevenlabs_cannot_do_is_refused_before_any_network(
    tmp_path: Path, extra: dict, said: str
) -> None:
    client, upstream = _eleven(tmp_path)
    with client:
        _ready(client)
        response = client.post(
            "/v1/speak", json={**speak("eleven_flash_v2_5", voice=VOICE), **extra}
        )
    assert response.status_code == 400, response.text
    assert said in response.json()["detail"]["detail"]
    assert not upstream.called


@respx.mock
def test_a_key_that_cannot_list_models_offers_none_and_says_which_permission(
    tmp_path: Path,
) -> None:
    """P3-2: the permission is required, and its name is the fix."""
    client, upstream = _eleven(tmp_path, models=httpx.Response(401, json=MISSING))
    with client:
        info = _ready(client)
        response = client.post("/v1/speak", json=speak("eleven_flash_v2_5", voice=VOICE))
    assert info["models"] == []
    assert "models_read" in info["catalogue"]["error"]
    assert response.status_code == 404, response.text
    assert not upstream.called


@respx.mock
def test_a_key_that_cannot_list_voices_still_speaks_any_voice(tmp_path: Path) -> None:
    denied = httpx.Response(
        401,
        json={
            "detail": {
                "message": "missing the permission voices_read",
                "status": "missing_permissions",
            }
        },
    )
    client, upstream = _eleven(tmp_path, voices=denied)
    with client:
        info = _ready(client)
        response = client.post("/v1/speak", json=speak("eleven_flash_v2_5", voice="any-voice-id"))
    assert "voices" not in info["models"][0]
    assert response.status_code == 200, response.text
    assert upstream.calls[0].request.url.path == "/v1/text-to-speech/any-voice-id/stream"


@respx.mock
def test_elevenlabs_refusing_the_drivers_key_is_the_credential_refusal(tmp_path: Path) -> None:
    denied = httpx.Response(
        401,
        json={
            "detail": {
                "message": "missing the permission text_to_speech",
                "status": "missing_permissions",
            }
        },
    )
    client, _ = _eleven(tmp_path, upstream=denied)
    with client:
        _ready(client)
        response = client.post("/v1/speak", json=speak("eleven_flash_v2_5", voice=VOICE))
    assert response.status_code == 502, response.text
    detail = response.json()["detail"]
    assert (
        detail["type"].endswith("#backend-credential-refused")
        and "text_to_speech" in detail["detail"]
    )


@respx.mock
def test_elevenlabs_does_not_chat(tmp_path: Path) -> None:
    client, _ = _eleven(tmp_path)
    with client:
        _ready(client)
        response = client.post(
            "/v1/generate",
            json={"model": "eleven_flash_v2_5", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 400, response.text


def test_the_api_key_is_one_field_shown_for_both_engines() -> None:
    fields = [f for f in FIELDS if f.key == "apiKey"]
    assert len(fields) == 1
    shown = set(fields[0].showWhen.equals)
    assert {"openrouter", "openai", "elevenlabs"} <= shown
