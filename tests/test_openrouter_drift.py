"""OpenRouter as documented on 2026-10-03, against what the driver sent.

Upstream drift audit, 2026-10-03 (OpenRouter's API reference, read that day):

* A listing's `expiration_date` was ignored while OpenAI's `shutdown_date`
  was honoured: 35 listed models carry one, and its Qwen3 family retires on
  2026-10-09, after which a slot naming one fails with no warning.
* Images are `POST /api/v1/images` now; we posted `/v1/images/generations`.
* Speech takes no `instructions` and transcription no `prompt` (neither is
  in its request schema), so both would be dropped; refused, as ElevenLabs
  already refuses them.
* Chat's `max_tokens` is "deprecated, use max_completion_tokens".

Each OpenRouter rule sits beside a backend whose behaviour must not move.
"""

from __future__ import annotations

import datetime as dt
import json

import httpx
import pytest
import respx

from eugene_plexus_inference_driver._generated.models import (
    BackendKind,
    GenerateRequest,
    ImageRequest,
    SpeakRequest,
    TranscribeRequest,
)
from eugene_plexus_inference_driver.engines._catalogue import from_openrouter
from eugene_plexus_inference_driver.engines.openai_compat_http import (
    OpenAiCompatibleHttpEngine,
    _Target,
)
from eugene_plexus_inference_driver.speech import SpeechRefusal
from eugene_plexus_inference_driver.transcription import TranscriptionRefusal

OPENROUTER = "https://openrouter.ai/api"
LOCAL = "http://127.0.0.1:8081"
OPENAI = "https://api.openai.com"
OK = {
    "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
}


def _engine(source: str, model: str = "acme/model") -> OpenAiCompatibleHttpEngine:
    base = {"openrouter": OPENROUTER, "openai": OPENAI}.get(source, LOCAL)
    return OpenAiCompatibleHttpEngine(
        api_key="sk-test",
        base_url=base,
        model_id=model,
        backend_kind=BackendKind.openai_compat_http,
        catalogue_source=source if source != "local" else "openai",
    )


# --------------------------------------------------------------------------- #
# A model past its expiration date is not routed
# --------------------------------------------------------------------------- #


def _entry(model_id: str, expires: str | None) -> dict:
    entry: dict = {
        "id": model_id,
        "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
        "supported_parameters": ["max_tokens"],
    }
    if expires is not None:
        entry["expiration_date"] = expires
    return entry


def test_a_model_past_its_expiration_date_is_not_listed() -> None:
    today = dt.datetime.now(dt.UTC).date()
    listing = {
        "data": [
            _entry("qwen/qwen3-old", (today - dt.timedelta(days=1)).isoformat()),
            _entry("qwen/qwen3-soon", (today + dt.timedelta(days=6)).isoformat()),
            _entry("acme/forever", None),
        ]
    }
    assert [m.id for m in from_openrouter(listing)] == ["qwen/qwen3-soon", "acme/forever"]


# --------------------------------------------------------------------------- #
# Images on OpenRouter's documented route
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("source", "path"), [("openrouter", "/v1/images"), ("openai", None)])
def test_images_go_to_the_route_each_provider_documents(source: str, path: str | None) -> None:
    engine = _engine(source)
    target = _Target(id="acme/image", upstream="acme/image", temperature_fixed=False)
    request = ImageRequest.model_validate({"model": "acme/image", "prompt": "A red square."})
    sent_to, sending = engine._image_request(target, request, [], None, stream=False)
    assert sent_to == (path or "/v1/images/generations")
    assert sending["json"]["prompt"] == "A red square."


# --------------------------------------------------------------------------- #
# Fields OpenRouter does not take are refused, not dropped
# --------------------------------------------------------------------------- #


@respx.mock
async def test_speech_instructions_are_refused_on_openrouter() -> None:
    route = respx.post(f"{OPENROUTER}/v1/audio/speech")
    request = SpeakRequest(
        model="acme/model", input="Hello.", voice="af_heart", instructions="cheerful"
    )
    with pytest.raises(SpeechRefusal) as refused:
        async for _ in _engine("openrouter").speak(request):
            pass
    assert "instructions" in str(refused.value)
    assert not route.called


@respx.mock
async def test_speech_instructions_still_reach_openai() -> None:
    route = respx.post(f"{OPENAI}/v1/audio/speech").mock(
        return_value=httpx.Response(200, content=b"ID3", headers={"content-type": "audio/mpeg"})
    )
    request = SpeakRequest(
        model="gpt-4o-mini-tts", input="Hello.", voice="alloy", instructions="cheerful"
    )
    async for _ in _engine("openai", "gpt-4o-mini-tts").speak(request):
        pass
    assert json.loads(route.calls[0].request.content)["instructions"] == "cheerful"


def _transcription(prompt: str | None) -> TranscribeRequest:
    return TranscribeRequest.model_validate(
        {
            "model": "acme/model",
            "audio": {"data": "SUQz", "filename": "fox.mp3", "mediaType": "audio/mpeg"},
            "prompt": prompt,
        }
    )


@respx.mock
async def test_a_transcription_prompt_is_refused_on_openrouter() -> None:
    route = respx.post(f"{OPENROUTER}/v1/audio/transcriptions")
    with pytest.raises(TranscriptionRefusal) as refused:
        await _engine("openrouter").transcribe(_transcription("Foxes."), b"ID3")
    assert "prompt" in str(refused.value)
    assert not route.called


@respx.mock
async def test_a_transcription_prompt_still_reaches_a_local_server() -> None:
    route = respx.post(f"{LOCAL}/v1/audio/transcriptions").mock(
        return_value=httpx.Response(200, json={"text": "The fox."})
    )
    await _engine("local").transcribe(_transcription("Foxes."), b"ID3")
    assert b'name="prompt"\r\n\r\nFoxes.' in route.calls[0].request.read()


# --------------------------------------------------------------------------- #
# The output cap under the name OpenRouter documents
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("source", "field", "other"),
    [
        ("openrouter", "max_completion_tokens", "max_tokens"),
        ("openai", "max_completion_tokens", "max_tokens"),
        ("local", "max_tokens", "max_completion_tokens"),
    ],
)
@respx.mock
async def test_the_output_cap_goes_under_the_name_each_backend_reads(
    source: str, field: str, other: str
) -> None:
    base = {"openrouter": OPENROUTER, "openai": OPENAI}.get(source, LOCAL)
    route = respx.post(f"{base}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=OK)
    )
    request = GenerateRequest.model_validate(
        {"messages": [{"role": "user", "content": "hi"}], "maxTokens": 512}
    )
    await _engine(source, "gpt-4.1" if source == "openai" else "acme/model").generate(request)
    sent = json.loads(route.calls[0].request.content)
    assert sent[field] == 512
    assert other not in sent
