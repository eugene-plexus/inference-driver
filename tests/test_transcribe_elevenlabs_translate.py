"""ElevenLabs' speech-to-text (P3-1) and translation (P3-4) at /v1/transcribe.

Measured 2026-09-28 against the live APIs (`provider-accounts-measurement.md`
section 11): ElevenLabs' `/v1/models` lists text-to-speech only, and its
speech-to-text models are named by its refusal of an unknown model id, given
before any audio, at no cost and even to a wrong key; an empty file is then
refused as empty for a key that may transcribe and 401 for one that may not.
It ignores a `prompt` it does not know. Only OpenAI's whisper translates:
OpenAI answers `/v1/audio/translations` 404 for its gpt-4o transcribe models,
and so do OpenRouter and llama-server. Every test here fails against the
driver before P3-1/P3-4, which listed no scribe model and had no `translate`.
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

ELEVEN = "https://api.elevenlabs.io"
OPENAI = "https://api.openai.com"
OPENROUTER = "https://openrouter.ai/api"
LLAMA = "http://llama.local"
VOICE = "21m00Tcm4TlvDq8ikWAM"
FOX = b"ID3\x04" + bytes(range(256)) * 8
SAID = "The quick brown fox jumps over the lazy dog."
EL_MODELS = [{"model_id": "eleven_flash_v2_5", "name": "Flash v2.5", "can_do_text_to_speech": True}]
NAMED = {
    "detail": {
        "type": "validation_error",
        "code": "unsupported_model",
        "message": "'eugene-plexus-lists-models' is not a valid model_id. Available models: "
        "'scribe_v1', 'scribe_v2'",
        "status": "invalid_model_id",
        "param": "model_id",
    }
}
EMPTY = {
    "detail": {
        "type": "invalid_request",
        "code": "bad_request",
        "message": "The uploaded file is empty or corrupted.",
        "status": "empty_file",
        "param": "file",
    }
}
NO_STT = {
    "detail": {
        "message": "The API key you used is missing the permission speech_to_text to execute "
        "this operation.",
        "status": "missing_permissions",
    }
}
SCRIBED = {
    "language_code": "eng",
    "language_probability": 0.69,
    "text": SAID,
    "words": [
        {"text": "The", "start": 0.26, "end": 0.36, "type": "word", "logprob": -0.1},
        {"text": " ", "start": 0.36, "end": 0.42, "type": "spacing", "logprob": -0.1},
        {"text": "(laughter)", "start": 0.42, "end": 0.5, "type": "audio_event", "logprob": -0.1},
        {"text": "quick", "start": 0.5, "end": 0.6, "type": "word", "logprob": -0.1},
    ],
    "audio_duration_secs": 3.38,
    "transcription_id": "t1",
}


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


def ask(model: str, **extra: Any) -> dict[str, Any]:
    return {
        "model": model,
        "audio": {
            "data": base64.b64encode(FOX).decode(),
            "filename": "fox.mp3",
            "mediaType": "audio/mpeg",
        },
        **extra,
    }


def _fields(request: httpx.Request) -> bytes:
    return request.content if isinstance(request.content, bytes) else b"".join(request.stream)


def _form(request: httpx.Request) -> dict[str, str]:
    """The text fields of a form body, multipart or urlencoded, by name."""
    if request.headers.get("content-type", "").startswith("application/x-www-form-urlencoded"):
        return dict(httpx.QueryParams(_fields(request).decode()))
    fields: dict[str, str] = {}
    for part in _fields(request).split(b"--")[1:]:
        head, _, value = part.partition(b"\r\n\r\n")
        if b'name="' not in head or b"filename=" in head:
            continue
        name = head.split(b'name="', 1)[1].split(b'"', 1)[0].decode()
        fields[name] = value.rsplit(b"\r\n", 1)[0].decode()
    return fields


# --------------------------------------------------------------------------- #
# ElevenLabs
# --------------------------------------------------------------------------- #


class _Scribe:
    """ElevenLabs' `/v1/speech-to-text` as measured: the probes, then the
    real call."""

    def __init__(self, *, may: bool = True, names: dict | None = None, answer: Any = None) -> None:
        self.may = may
        self.names = NAMED if names is None else names
        self.answer = answer if answer is not None else httpx.Response(200, json=SCRIBED)
        self.probes: list[httpx.Request] = []
        self.calls: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        form = _form(request)
        body = _fields(request)
        if form.get("model_id") not in ("scribe_v1", "scribe_v2"):
            self.probes.append(request)
            return httpx.Response(400, json=self.names)
        if b'filename="empty.mp3"' in body:
            self.probes.append(request)
            return httpx.Response(400, json=EMPTY) if self.may else httpx.Response(401, json=NO_STT)
        self.calls.append(request)
        return self.answer


def _eleven(tmp_path: Path, scribe: _Scribe) -> TestClient:
    respx.get(f"{ELEVEN}/v1/models").mock(return_value=httpx.Response(200, json=EL_MODELS))
    respx.get(f"{ELEVEN}/v1/voices").mock(
        return_value=httpx.Response(200, json={"voices": [{"voice_id": VOICE}]})
    )
    respx.post(f"{ELEVEN}/v1/speech-to-text").mock(side_effect=scribe)
    respx.post(url__regex=rf"{ELEVEN}/v1/text-to-speech/.*/stream").mock(
        return_value=httpx.Response(200, content=b"ID3", headers={"content-type": "audio/mpeg"})
    )
    return TestClient(
        create_app(settings=Settings(config_file=_config(tmp_path, provider="elevenlabs")))
    )


@respx.mock
def test_a_key_that_may_transcribe_lists_the_models_elevenlabs_names(tmp_path: Path) -> None:
    scribe = _Scribe()
    with _eleven(tmp_path, scribe) as client:
        info = _ready(client)
    surfaces = {m["id"]: m["surfaces"] for m in info["models"]}
    assert surfaces == {
        "eleven_flash_v2_5": ["speech"],
        "scribe_v1": ["transcription"],
        "scribe_v2": ["transcription"],
    }
    listing, permission = scribe.probes
    # The listing probe carries no audio; the permission probe an empty file
    # for a model ElevenLabs named.
    assert _form(listing) == {"model_id": "eugene-plexus-lists-models"}
    assert b"filename=" not in _fields(listing)
    assert _form(permission)["model_id"] == "scribe_v1"
    assert permission.headers["xi-api-key"] == "sk-test"


@respx.mock
def test_a_key_that_may_not_transcribe_offers_no_scribe_and_still_speaks(tmp_path: Path) -> None:
    scribe = _Scribe(may=False)
    with _eleven(tmp_path, scribe) as client:
        info = _ready(client)
        transcribed = client.post("/v1/transcribe", json=ask("scribe_v2"))
        spoke = client.post(
            "/v1/speak", json={"model": "eleven_flash_v2_5", "input": "hi", "voice": VOICE}
        )
    assert [m["id"] for m in info["models"]] == ["eleven_flash_v2_5"]
    assert info["catalogue"].get("error") is None
    assert transcribed.status_code == 404 and not scribe.calls
    assert spoke.status_code == 200, spoke.text


@respx.mock
def test_an_unreadable_name_list_costs_transcription_and_not_speech(tmp_path: Path) -> None:
    scribe = _Scribe(names={"detail": "Service unavailable"})
    with _eleven(tmp_path, scribe) as client:
        info = _ready(client)
    assert [m["id"] for m in info["models"]] == ["eleven_flash_v2_5"]
    assert len(scribe.probes) == 1  # nothing named, so no permission probe


@respx.mock
def test_elevenlabs_is_asked_in_its_own_shape_and_answered_in_openais(tmp_path: Path) -> None:
    scribe = _Scribe()
    with _eleven(tmp_path, scribe) as client:
        _ready(client)
        response = client.post(
            "/v1/transcribe", json=ask("scribe_v2", language="en", temperature=0.2)
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["text"], body["language"], body["modelId"]) == (SAID, "eng", "scribe_v2")
    assert body["usage"]["seconds"] == 3.38
    # Not verbose: no duration, no words.
    assert body.get("duration") is None and body.get("words") is None
    [call] = scribe.calls
    assert _form(call) == {
        "model_id": "scribe_v2",
        # No (laughter) in an OpenAI-shaped transcript.
        "tag_audio_events": "false",
        "timestamps_granularity": "none",
        "language_code": "en",
        "temperature": "0.2",
    }
    assert b'filename="fox.mp3"' in _fields(call) and FOX in _fields(call)
    assert call.headers["xi-api-key"] == "sk-test" and "authorization" not in call.headers


@respx.mock
def test_verbose_word_timestamps_become_openais_words(tmp_path: Path) -> None:
    scribe = _Scribe()
    with _eleven(tmp_path, scribe) as client:
        _ready(client)
        response = client.post(
            "/v1/transcribe", json=ask("scribe_v2", verbose=True, timestampGranularities=["word"])
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["duration"] == 3.38
    # Spacing and audio events are ElevenLabs' entries, not words.
    assert body["words"] == [
        {"word": "The", "start": 0.26, "end": 0.36},
        {"word": "quick", "start": 0.5, "end": 0.6},
    ]
    assert _form(scribe.calls[0])["timestamps_granularity"] == "word"


@respx.mock
def test_what_elevenlabs_would_drop_is_refused_before_any_audio(tmp_path: Path) -> None:
    scribe = _Scribe()
    with _eleven(tmp_path, scribe) as client:
        _ready(client)
        prompted = client.post("/v1/transcribe", json=ask("scribe_v2", prompt="Foxes."))
        segments = client.post(
            "/v1/transcribe",
            json=ask("scribe_v2", verbose=True, timestampGranularities=["segment"]),
        )
    for response, said in ((prompted, "prompt"), (segments, "segments")):
        assert response.status_code == 400, response.text
        detail = response.json()["detail"]
        assert detail["type"].endswith("#transcription-refused") and said in detail["detail"]
    assert not scribe.calls


@respx.mock
def test_a_scribe_model_does_not_speak_or_translate(tmp_path: Path) -> None:
    scribe = _Scribe()
    with _eleven(tmp_path, scribe) as client:
        _ready(client)
        spoke = client.post("/v1/speak", json={"model": "scribe_v2", "input": "hi", "voice": VOICE})
        translated = client.post("/v1/transcribe", json=ask("scribe_v2", translate=True))
    assert spoke.status_code == 400 and "does not speak" in spoke.json()["detail"]["detail"]
    assert translated.status_code == 400, translated.text
    assert translated.json()["detail"]["type"].endswith("#translation-unsupported")
    assert not scribe.calls


@respx.mock
def test_elevenlabs_refusing_the_drivers_key_mid_life_is_the_credential_refusal(
    tmp_path: Path,
) -> None:
    scribe = _Scribe(answer=httpx.Response(401, json=NO_STT))
    with _eleven(tmp_path, scribe) as client:
        _ready(client)
        response = client.post("/v1/transcribe", json=ask("scribe_v2"))
    assert response.status_code == 502, response.text
    detail = response.json()["detail"]
    assert (
        detail["type"].endswith("#backend-credential-refused")
        and "speech_to_text" in detail["detail"]
    )


# --------------------------------------------------------------------------- #
# Translation
# --------------------------------------------------------------------------- #


def _openai(tmp_path: Path, answer: httpx.Response) -> tuple[TestClient, respx.Route, respx.Route]:
    listing = {"data": [{"id": "whisper-1"}, {"id": "gpt-4o-mini-transcribe"}, {"id": "gpt-4o"}]}
    respx.get(f"{OPENAI}/v1/models").mock(return_value=httpx.Response(200, json=listing))
    translations = respx.post(f"{OPENAI}/v1/audio/translations").mock(return_value=answer)
    transcriptions = respx.post(f"{OPENAI}/v1/audio/transcriptions").mock(
        return_value=httpx.Response(200, json={"text": "not this door"})
    )
    app = create_app(settings=Settings(config_file=_config(tmp_path, provider="openai")))
    return TestClient(app), translations, transcriptions


@respx.mock
def test_whisper_translates_at_openais_translation_door(tmp_path: Path) -> None:
    client, translations, transcriptions = _openai(
        tmp_path, httpx.Response(200, json={"text": "The fast brown fox."})
    )
    with client:
        info = _ready(client)
        response = client.post(
            "/v1/transcribe",
            json=ask("whisper-1", translate=True, prompt="Foxes.", temperature=0.1),
        )
    surfaces = {m["id"]: m["surfaces"] for m in info["models"]}
    assert surfaces["whisper-1"] == ["transcription", "translation"]
    assert surfaces["gpt-4o-mini-transcribe"] == ["transcription"]
    assert response.status_code == 200, response.text
    assert response.json()["text"] == "The fast brown fox."
    assert not transcriptions.calls
    assert _form(translations.calls[0].request) == {
        "model": "whisper-1",
        "prompt": "Foxes.",
        "temperature": "0.1",
    }


@respx.mock
def test_a_verbose_translation_asks_for_verbose_json(tmp_path: Path) -> None:
    answer = {
        "task": "translate",
        "language": "english",
        "duration": 3.38,
        "text": "Fox.",
        "segments": [{"id": 0, "start": 0.0, "end": 3.6, "text": " Fox."}],
    }
    client, translations, _ = _openai(tmp_path, httpx.Response(200, json=answer))
    with client:
        _ready(client)
        response = client.post(
            "/v1/transcribe", json=ask("whisper-1", translate=True, verbose=True)
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["language"], body["duration"], len(body["segments"])) == ("english", 3.38, 1)
    assert _form(translations.calls[0].request)["response_format"] == "verbose_json"


@respx.mock
def test_what_a_translation_does_not_take_is_refused_before_any_audio(tmp_path: Path) -> None:
    client, translations, _ = _openai(tmp_path, httpx.Response(200, json={"text": "x"}))
    with client:
        _ready(client)
        responses = {
            "language": client.post(
                "/v1/transcribe", json=ask("whisper-1", translate=True, language="fr")
            ),
            "timestampGranularities": client.post(
                "/v1/transcribe",
                json=ask(
                    "whisper-1", translate=True, verbose=True, timestampGranularities=["word"]
                ),
            ),
            "gpt-4o-mini-transcribe": client.post(
                "/v1/transcribe", json=ask("gpt-4o-mini-transcribe", translate=True)
            ),
        }
    for said, response in responses.items():
        assert response.status_code == 400, (said, response.text)
    assert "language" in responses["language"].json()["detail"]["detail"]
    assert (
        "timestampGranularities" in responses["timestampGranularities"].json()["detail"]["detail"]
    )
    assert (
        responses["gpt-4o-mini-transcribe"]
        .json()["detail"]["type"]
        .endswith("#translation-unsupported")
    )
    assert not translations.calls


@respx.mock
def test_openrouter_and_llama_server_do_not_translate(tmp_path: Path) -> None:
    listing = {
        "data": [
            {
                "id": "openai/whisper-large-v3-turbo",
                "architecture": {
                    "input_modalities": ["audio"],
                    "output_modalities": ["transcription"],
                },
                "supported_parameters": [],
            }
        ]
    }
    respx.get(f"{OPENROUTER}/v1/models/user").mock(return_value=httpx.Response(200, json=listing))
    respx.get(f"{OPENROUTER}/v1/images/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    respx.get(f"{OPENROUTER}/v1/videos/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    routed = respx.post(url__regex=rf"{OPENROUTER}/v1/audio/.*").mock(
        return_value=httpx.Response(200, json={"text": "x"})
    )
    respx.get(f"{LLAMA}/props").mock(
        return_value=httpx.Response(200, json={"modalities": {"vision": False, "audio": True}})
    )
    respx.get(f"{LLAMA}/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "asr"}]})
    )
    local = respx.post(url__regex=rf"{LLAMA}/v1/audio/.*").mock(
        return_value=httpx.Response(200, json={"text": "x"})
    )
    router = TestClient(
        create_app(settings=Settings(config_file=_config(tmp_path, provider="openrouter")))
    )
    llama_config = _config(tmp_path, provider="openai_compat_custom", baseUrl=LLAMA, modelId="asr")
    llama = TestClient(create_app(settings=Settings(config_file=llama_config)))
    with router, llama:
        _ready(router)
        answers = [
            router.post(
                "/v1/transcribe", json=ask("openai/whisper-large-v3-turbo", translate=True)
            ),
            llama.post("/v1/transcribe", json=ask("asr", translate=True)),
        ]
    for response in answers:
        assert response.status_code == 400, response.text
        assert response.json()["detail"]["type"].endswith("#translation-unsupported")
    assert not routed.calls and not local.calls
