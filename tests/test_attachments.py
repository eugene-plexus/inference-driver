"""Audio and PDF parts: validated, confirmed per model, carried unchanged (P2).

Every test here fails against the driver as it was before 2026-09-28,
when `MessageContentPart` held text and images only and a request
carrying `input_audio` or `file` was a 422 at the schema.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from eugene_plexus_inference_driver import images
from eugene_plexus_inference_driver._generated.models import GenerateRequest
from eugene_plexus_inference_driver.app import create_app
from eugene_plexus_inference_driver.engines._catalogue import from_openrouter
from eugene_plexus_inference_driver.engines.claude_code_cli import ClaudeCodeCliEngine
from eugene_plexus_inference_driver.engines.openai_compat_http import OpenAiCompatibleHttpEngine
from eugene_plexus_inference_driver.settings import Settings

OPENROUTER = "https://openrouter.ai/api"
LOCAL = "http://audio-test"

WAV = b"RIFF\x24\x00\x00\x00WAVEfmt " + bytes(28)
MP3 = b"ID3\x04\x00\x00\x00\x00\x00\x00" + bytes(32)
PDF = b"%PDF-1.4\n1 0 obj << >> endobj\ntrailer << >>\n%%EOF\n"


def b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


def audio_part(raw: bytes = MP3, fmt: str = "mp3") -> dict[str, Any]:
    return {"type": "input_audio", "input_audio": {"data": b64(raw), "format": fmt}}


def file_part(data: str | None = None, **extra: Any) -> dict[str, Any]:
    file = {"filename": "note.pdf", "file_data": data or f"data:application/pdf;base64,{b64(PDF)}"}
    file.update(extra)
    return {"type": "file", "file": file}


def ask(*parts: dict[str, Any], model: str | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "What is in this?"}, *parts]}
        ]
    }
    if model:
        body["model"] = model
    return body


def _or_model(model_id: str, inputs: list[str]) -> dict[str, Any]:
    return {
        "id": model_id,
        "context_length": 131072,
        "architecture": {"input_modalities": inputs, "output_modalities": ["text"]},
        "supported_parameters": ["max_tokens", "temperature"],
    }


LISTING = {
    "data": [
        _or_model("google/gemini-2.5-flash-lite", ["text", "image", "file", "audio", "video"]),
        _or_model("mistralai/mistral-nemo", ["text"]),
    ]
}


def _completion(text: str = "a fox") -> dict[str, Any]:
    return {
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
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
        import time

        time.sleep(0.02)
    raise AssertionError("catalogue never read")


# --------------------------------------------------------------------------- #
# The listing says what each model takes
# --------------------------------------------------------------------------- #


def test_a_listing_confirms_audio_and_files_per_model() -> None:
    hears, deaf = from_openrouter(LISTING)
    assert hears.capabilities is not None and deaf.capabilities is not None
    assert (hears.capabilities.audioInput, hears.capabilities.fileInput) == (True, True)
    assert (deaf.capabilities.audioInput, deaf.capabilities.fileInput) == (False, False)


# --------------------------------------------------------------------------- #
# An account carries the parts, unchanged, to a model that takes them
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("part", [audio_part(), audio_part(WAV, "wav"), file_part()])
@respx.mock
def test_the_part_reaches_the_backend_exactly_as_sent(account: Path, part: dict) -> None:
    respx.get(f"{OPENROUTER}/v1/models/user").mock(return_value=httpx.Response(200, json=LISTING))
    upstream = respx.post(f"{OPENROUTER}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_completion())
    )
    body = ask(part, model="google/gemini-2.5-flash-lite")
    with TestClient(create_app(settings=Settings(config_file=account))) as client:
        _ready(client)
        response = client.post("/v1/generate", json=body)
    assert response.status_code == 200, response.text
    assert json.loads(upstream.calls[0].request.content)["messages"] == body["messages"]


@pytest.mark.parametrize(("part", "word"), [(audio_part(), "audio"), (file_part(), "files")])
@respx.mock
def test_a_model_whose_listing_does_not_take_it_is_never_sent_it(
    account: Path, part: dict, word: str
) -> None:
    respx.get(f"{OPENROUTER}/v1/models/user").mock(return_value=httpx.Response(200, json=LISTING))
    upstream = respx.post(f"{OPENROUTER}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_completion())
    )
    with TestClient(create_app(settings=Settings(config_file=account))) as client:
        _ready(client)
        response = client.post("/v1/generate", json=ask(part, model="mistralai/mistral-nemo"))
    assert response.status_code == 400, response.text
    assert word in response.json()["detail"]["detail"]
    assert not upstream.called


@respx.mock
def test_bare_base64_is_carried_as_the_data_url_openrouter_requires(account: Path) -> None:
    """Measured 2026-09-28: OpenRouter answers bare base64 with *Invalid
    content*, while OpenAI's schema calls the field base64."""
    respx.get(f"{OPENROUTER}/v1/models/user").mock(return_value=httpx.Response(200, json=LISTING))
    upstream = respx.post(f"{OPENROUTER}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_completion())
    )
    body = ask(file_part(b64(PDF)), model="google/gemini-2.5-flash-lite")
    with TestClient(create_app(settings=Settings(config_file=account))) as client:
        _ready(client)
        assert client.post("/v1/generate", json=body).status_code == 200
    sent = json.loads(upstream.calls[0].request.content)["messages"][0]["content"][1]
    assert sent["file"]["file_data"] == f"data:application/pdf;base64,{b64(PDF)}"


# --------------------------------------------------------------------------- #
# Refused before any network, naming the field
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("part", "said"),
    [
        (audio_part(WAV, "mp3"), "does not match its declared mp3"),
        (audio_part(MP3, "wav"), "does not match its declared wav"),
        (
            {"type": "input_audio", "input_audio": {"data": "not base64!", "format": "mp3"}},
            "invalid base64",
        ),
        (file_part(file_id="file-abc"), "no file store"),
        (file_part("data:text/plain;base64," + b64(b"hi")), "PDF data URL"),
        (file_part("https://example.com/a.pdf"), "URLs are not fetched"),
        (file_part(b64(b"not a pdf at all")), "is not a PDF"),
    ],
)
@respx.mock
def test_a_bad_attachment_is_refused_before_any_network(
    account: Path, part: dict, said: str
) -> None:
    respx.get(f"{OPENROUTER}/v1/models/user").mock(return_value=httpx.Response(200, json=LISTING))
    upstream = respx.post(f"{OPENROUTER}/v1/chat/completions")
    with TestClient(create_app(settings=Settings(config_file=account))) as client:
        _ready(client)
        response = client.post("/v1/generate", json=ask(part, model="google/gemini-2.5-flash-lite"))
    assert response.status_code == 400, response.text
    assert said in response.text
    assert not upstream.called


def test_an_attachment_on_an_assistant_turn_is_refused() -> None:
    body = GenerateRequest.model_validate(
        {"messages": [{"role": "assistant", "content": [audio_part()]}]}
    )
    with pytest.raises(
        images.ImageRefusal, match="attachments are supported only on user messages"
    ):
        images.validate_messages(body.messages)


def test_the_request_total_counts_every_kind_of_attachment(monkeypatch) -> None:
    monkeypatch.setattr(images, "MAX_ATTACHMENTS_TOTAL", len(MP3) + len(PDF) - 1)
    body = GenerateRequest.model_validate(ask(audio_part(), file_part()))
    with pytest.raises(images.ImageRefusal, match="11 MiB"):
        images.validate_messages(body.messages)


# --------------------------------------------------------------------------- #
# A single-model backend confirms by asking it
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@respx.mock
async def test_llama_servers_props_decide_audio_input() -> None:
    adapter = OpenAiCompatibleHttpEngine(base_url=LOCAL, model_id="voxtral", auth_required=False)
    respx.get(LOCAL + "/v1/models").respond(200, json={"data": [{"id": "voxtral"}]})
    respx.get(LOCAL + "/props").respond(200, json={"modalities": {"vision": False, "audio": True}})
    assert await adapter.probe_audio_input() is True
    assert await adapter.probe_image_input() is False
    respx.get(LOCAL + "/props").respond(200, json={"modalities": {"vision": True, "audio": False}})
    assert await adapter.probe_audio_input() is False


@pytest.mark.parametrize("stream", [False, True])
def test_a_cli_backend_refuses_audio_rather_than_flattening_it(app, stream) -> None:
    with TestClient(app) as client:
        app.state.adapter = ClaudeCodeCliEngine(binary_path="p2-no-such-executable")
        response = client.post(
            "/v1/generate" + ("/stream" if stream else ""), json=ask(audio_part())
        )
    assert response.status_code == 400
    assert "audioInput" in response.text


@respx.mock
def test_a_backend_echo_of_the_audio_is_not_exposed(account: Path) -> None:
    respx.get(f"{OPENROUTER}/v1/models/user").mock(return_value=httpx.Response(200, json=LISTING))
    body = ask(audio_part(), model="google/gemini-2.5-flash-lite")
    respx.post(f"{OPENROUTER}/v1/chat/completions").mock(
        return_value=httpx.Response(400, json=body)
    )
    with TestClient(create_app(settings=Settings(config_file=account))) as client:
        _ready(client)
        response = client.post("/v1/generate", json=body)
    assert response.status_code == 400
    assert b64(MP3) not in response.text
    assert "Audio request refused" in response.text
