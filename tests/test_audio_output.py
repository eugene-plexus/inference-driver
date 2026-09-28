"""A spoken answer: asked of the backend as a pcm16 stream, returned as asked (P2b).

Measured 2026-09-28 against OpenRouter (`provider-accounts-measurement.md`
section 4): a non-streamed audio answer is refused before any provider,
a streamed one is `pcm16` only, and Lyria sends one MP3 whatever it is
asked for. Every test here fails against the driver as it was before P2b,
which had no `audioOutput` on a request and read nothing but text,
reasoning and tool calls off a stream.
"""

from __future__ import annotations

import base64
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
from eugene_plexus_inference_driver.audio_out import detect_format, wav_from_pcm16
from eugene_plexus_inference_driver.engines._catalogue import from_openrouter
from eugene_plexus_inference_driver.settings import Settings

OPENROUTER = "https://openrouter.ai/api"
SPEAKS = "openai/gpt-audio-mini"
MUSIC = "google/lyria-3-clip-preview"
TEXT = "mistralai/mistral-nemo"

#: Two 0.4 s-shaped chunks of samples; the first sample is -1, which is
#: the bytes FF FF -- an MPEG frame sync by eleven bits, and not an MP3.
PCM_A = b"\xff\xff" + bytes(range(256)) * 4
PCM_B = bytes(reversed(range(256))) * 4
#: An MP3 as Lyria sends it: an ID3 tag, then frames.
MP3 = b"ID3\x03\x00\x00\x00\x00\x00\x00" + bytes(64)


def b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


def _or_model(model_id: str, output: list[str]) -> dict[str, Any]:
    return {
        "id": model_id,
        "context_length": 128000,
        "architecture": {"input_modalities": ["text"], "output_modalities": output},
        "supported_parameters": ["max_tokens", "temperature"],
    }


LISTING = {
    "data": [
        _or_model(SPEAKS, ["text", "audio"]),
        _or_model(MUSIC, ["text", "audio"]),
        _or_model(TEXT, ["text"]),
    ]
}


def _sse(*frames: dict[str, Any]) -> httpx.Response:
    body = "".join(f"data: {json.dumps(f)}\n\n" for f in frames) + "data: [DONE]\n\n"
    return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})


def _delta(**delta: Any) -> dict[str, Any]:
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]}


#: gpt-audio's stream, in the shape measured: id and transcript first, then
#: data with `expires_at`, `content` empty throughout, usage on its own.
SPEECH = (
    _delta(role="assistant", content=""),
    _delta(audio={"id": "audio_1", "transcript": "Hello"}),
    _delta(audio={"transcript": " there", "data": b64(PCM_A), "expires_at": 1790609721}),
    _delta(audio={"data": b64(PCM_B)}),
    {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 60, "total_tokens": 72}},
)
#: Lyria's: one delta carrying the whole MP3, lyrics in `content`.
SONG = (
    _delta(content="[0.0:2.0] LA LA"),
    _delta(audio={"data": b64(MP3)}),
    {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
)


def ask(model: str, fmt: str = "wav") -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "Say hello."}],
        "audioOutput": {"voice": "alloy", "format": fmt},
    }


@pytest.fixture
def account(tmp_path: Path) -> Path:
    config = tmp_path / "openrouter.yaml"
    config.write_text(json.dumps({"provider": "openrouter", "apiKey": "sk-or-test"}), "utf-8")
    return config


def _ready(client: TestClient) -> None:
    for _ in range(250):
        if (client.get("/v1/info").json().get("catalogue") or {}).get("refreshedAt"):
            return
        time.sleep(0.02)
    raise AssertionError("catalogue never read")


def _serve(account: Path, upstream: httpx.Response) -> tuple[TestClient, respx.Route]:
    respx.get(f"{OPENROUTER}/v1/models/user").mock(return_value=httpx.Response(200, json=LISTING))
    route = respx.post(f"{OPENROUTER}/v1/chat/completions").mock(return_value=upstream)
    return TestClient(create_app(settings=Settings(config_file=account))), route


# --------------------------------------------------------------------------- #
# What the bytes are
# --------------------------------------------------------------------------- #


def test_the_listing_confirms_audio_output_from_what_a_model_gives_back() -> None:
    speaks, music, text = from_openrouter(LISTING)
    assert speaks.capabilities is not None and speaks.capabilities.audioOutput is True
    assert music.capabilities is not None and music.capabilities.audioOutput is True
    assert text.capabilities is not None and text.capabilities.audioOutput is False


def test_the_format_is_read_from_the_bytes() -> None:
    assert detect_format(MP3).value == "mp3"
    assert detect_format(wav_from_pcm16(PCM_A)).value == "wav"
    assert detect_format(PCM_A).value == "pcm16"
    assert detect_format(bytes(64)).value == "pcm16"


def test_a_lone_frame_sync_in_samples_is_not_an_mp3() -> None:
    """`FF FB 90 00` is a valid MPEG-1 Layer III header, 417 bytes long at
    128 kbit/s and 44.1 kHz. Samples that happen to begin with it are
    still samples unless another header begins where it says the frame
    ends -- which is what a real MP3 has and speech does not."""
    header = b"\xff\xfb\x90\x00"
    assert detect_format(header + bytes(600)).value == "pcm16"
    assert detect_format(header + bytes(413) + header + bytes(200)).value == "mp3"


def test_the_wav_header_says_what_openai_documents_pcm16_is() -> None:
    wav = wav_from_pcm16(PCM_A)
    assert wav[:4] == b"RIFF" and wav[8:16] == b"WAVEfmt "
    fmt, channels, rate, per_second, block, bits = struct.unpack("<HHIIHH", wav[20:36])
    assert (fmt, channels, rate, per_second, block, bits) == (1, 1, 24000, 48000, 2, 16)
    assert struct.unpack("<I", wav[4:8])[0] == 36 + len(PCM_A)
    assert wav[36:40] == b"data" and struct.unpack("<I", wav[40:44])[0] == len(PCM_A)
    assert wav[44:] == PCM_A


# --------------------------------------------------------------------------- #
# A non-streamed answer is the stream assembled
# --------------------------------------------------------------------------- #


@respx.mock
def test_a_batch_answer_asks_for_a_pcm16_stream_and_returns_a_wav(account: Path) -> None:
    client, upstream = _serve(account, _sse(*SPEECH))
    with client:
        _ready(client)
        response = client.post("/v1/generate", json=ask(SPEAKS, "wav"))
    assert response.status_code == 200, response.text
    sent = json.loads(upstream.calls[0].request.content)
    assert sent["stream"] is True
    assert sent["modalities"] == ["text", "audio"]
    assert sent["audio"] == {"voice": "alloy", "format": "pcm16"}
    body = response.json()
    audio = body["audio"]
    assert audio["format"] == "wav"
    assert base64.b64decode(audio["data"]) == wav_from_pcm16(PCM_A + PCM_B)
    assert audio["transcript"] == "Hello there"
    assert audio["id"] == "audio_1" and audio["expiresAt"] == 1790609721
    # A spoken answer's text is its transcript, not an empty `content`.
    assert body.get("content") is None
    assert body["finishReason"] == "stop"


@respx.mock
def test_a_batch_answer_asked_for_pcm16_is_the_samples_alone(account: Path) -> None:
    client, _ = _serve(account, _sse(*SPEECH))
    with client:
        _ready(client)
        audio = client.post("/v1/generate", json=ask(SPEAKS, "pcm16")).json()["audio"]
    assert audio["format"] == "pcm16"
    assert base64.b64decode(audio["data"]) == PCM_A + PCM_B


@respx.mock
def test_lyrias_mp3_is_returned_as_sent_and_labelled_mp3(account: Path) -> None:
    """P2-2: asked for WAV, Lyria answers MP3; presenting it as WAV would
    be a lie the caller's player finds first."""
    client, _ = _serve(account, _sse(*SONG))
    with client:
        _ready(client)
        body = client.post("/v1/generate", json=ask(MUSIC, "wav")).json()
    assert body["audio"]["format"] == "mp3"
    assert base64.b64decode(body["audio"]["data"]) == MP3
    assert body["content"] == "[0.0:2.0] LA LA"


@respx.mock
def test_a_fragment_that_is_not_base64_fails_rather_than_being_passed_on(account: Path) -> None:
    client, _ = _serve(account, _sse(_delta(audio={"data": "not base64!"}), SPEECH[4]))
    with client:
        _ready(client)
        response = client.post("/v1/generate", json=ask(SPEAKS, "wav"))
    assert response.status_code == 502, response.text
    assert "not base64" in response.text


# --------------------------------------------------------------------------- #
# A streamed answer is the fragments as they come
# --------------------------------------------------------------------------- #


@respx.mock
def test_a_streamed_answer_forwards_each_fragment_and_says_its_format_once(account: Path) -> None:
    client, _ = _serve(account, _sse(*SPEECH))
    with client:
        _ready(client)
        response = client.post("/v1/generate/stream", json=ask(SPEAKS, "pcm16"))
    assert response.status_code == 200, response.text
    events = [block for block in response.text.split("\n\n") if block.strip()]
    tokens = [json.loads(e.split("data: ", 1)[1]) for e in events if e.startswith("event: token")]
    fragments = [t["audio"] for t in tokens if "audio" in t]
    assert fragments[0] == {"id": "audio_1", "transcript": "Hello"}
    assert fragments[1]["format"] == "pcm16" and fragments[1]["expiresAt"] == 1790609721
    assert "format" not in fragments[2]
    assert b"".join(base64.b64decode(f["data"]) for f in fragments if "data" in f) == PCM_A + PCM_B
    done = json.loads(next(e for e in events if e.startswith("event: done")).split("data: ", 1)[1])
    # The fragments were sent; the terminal frame does not send them twice.
    assert "audio" not in done


# --------------------------------------------------------------------------- #
# Refused before any network
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("path", "fmt"),
    [
        ("/v1/generate", "mp3"),
        ("/v1/generate", "flac"),
        ("/v1/generate", "opus"),
        ("/v1/generate", "aac"),
        ("/v1/generate/stream", "wav"),
        ("/v1/generate/stream", "mp3"),
    ],
)
@respx.mock
def test_a_format_the_pcm16_stream_cannot_become_is_refused(
    account: Path, path: str, fmt: str
) -> None:
    client, upstream = _serve(account, _sse(*SPEECH))
    with client:
        _ready(client)
        response = client.post(path, json=ask(SPEAKS, fmt))
    assert response.status_code == 400, response.text
    assert response.json()["detail"]["type"].endswith("#audio-format-unsupported")
    assert "pcm16" in response.json()["detail"]["detail"]
    assert not upstream.called


@respx.mock
def test_a_model_that_does_not_speak_is_never_asked_to(account: Path) -> None:
    client, upstream = _serve(account, _sse(*SPEECH))
    with client:
        _ready(client)
        response = client.post("/v1/generate", json=ask(TEXT, "wav"))
    assert response.status_code == 400, response.text
    assert response.json()["detail"]["type"].endswith("#audio-output-unsupported")
    assert not upstream.called


@respx.mock
def test_a_single_model_driver_does_not_speak(tmp_path: Path) -> None:
    config = tmp_path / "local.yaml"
    config.write_text(
        json.dumps({"provider": "openai_compat_custom", "baseUrl": "http://local", "modelId": "m"}),
        "utf-8",
    )
    upstream = respx.post("http://local/v1/chat/completions")
    with TestClient(create_app(settings=Settings(config_file=config))) as client:
        response = client.post("/v1/generate", json=ask("m", "wav"))
    assert response.status_code == 400, response.text
    assert response.json()["detail"]["type"].endswith("#audio-output-unsupported")
    assert not upstream.called


@respx.mock
def test_a_request_without_audio_asks_for_none(account: Path) -> None:
    completion = {
        "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}]
    }
    client, upstream = _serve(account, httpx.Response(200, json=completion))
    with client:
        _ready(client)
        response = client.post(
            "/v1/generate",
            json={"model": SPEAKS, "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 200, response.text
    sent = json.loads(upstream.calls[0].request.content)
    assert "modalities" not in sent and "audio" not in sent
    assert response.json().get("audio") is None
