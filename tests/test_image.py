"""POST /v1/image and /v1/image/stream through OpenRouter and OpenAI (P4).

Measured 2026-09-28 (`provider-accounts-measurement.md` section 9): OpenRouter
has no edit route, takes reference images as `input_references` OBJECTS on
`/images/generations`, lists image settings only on `/images/models`, ignores
`mask`, answers `created: 0` from some providers, streams bare `data:` frames
with `: ` keepalives, and answers plain JSON to a model that cannot stream.
OpenAI's SDK sends an edit as multipart with `image[]` for several images.
Every test here fails against the driver before P4, which had no image route
(the 404s pass either way, since a missing route is a 404 too).
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

from eugene_plexus_inference_driver.app import create_app
from eugene_plexus_inference_driver.engines._catalogue import with_openrouter_images
from eugene_plexus_inference_driver.images_out import sniff
from eugene_plexus_inference_driver.settings import Settings

OPENROUTER = "https://openrouter.ai/api"
OPENAI = "https://api.openai.com"
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF" + bytes(64)
WEBP = b"RIFF\x24\x00\x00\x00WEBPVP8 " + bytes(32)
B64 = base64.b64encode
FLUX = "black-forest-labs/flux.2-klein-4b"
MINI = "openai/gpt-image-1-mini"
GEMINI = "google/gemini-3.1-flash-lite-image"
AUTO = "openrouter/auto"

LISTING = {
    "data": [
        {
            "id": FLUX,
            "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["image"]},
            "supported_parameters": ["seed"],
        },
        {
            "id": MINI,
            "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["image"]},
            "supported_parameters": ["temperature"],
        },
        {
            "id": GEMINI,
            "architecture": {
                "input_modalities": ["image", "text"],
                "output_modalities": ["image", "text"],
            },
            "supported_parameters": ["max_tokens"],
        },
        {
            "id": AUTO,
            "architecture": {"input_modalities": ["text"], "output_modalities": ["image", "text"]},
            "supported_parameters": [],
        },
        {
            "id": "mistralai/mistral-nemo",
            "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
            "supported_parameters": ["max_tokens"],
        },
    ]
}
#: `GET /images/models`, as measured: typed descriptors, not a list of names.
IMAGES = {
    "data": [
        {
            "id": FLUX,
            "supported_parameters": {
                "aspect_ratio": {"type": "enum", "values": ["1:1", "16:9"]},
                "output_format": {"type": "enum", "values": ["png", "jpeg"]},
                "n": {"type": "range", "min": 1, "max": 1},
                "input_references": {"type": "range", "min": 0, "max": 4},
                "seed": {"type": "boolean"},
            },
            "supports_streaming": False,
        },
        {
            "id": MINI,
            "supported_parameters": {
                "quality": {"type": "enum", "values": ["auto", "low", "medium", "high"]},
                "background": {"type": "enum", "values": ["auto", "transparent", "opaque"]},
                "n": {"type": "range", "min": 1, "max": 10},
                "input_references": {"type": "range", "min": 0, "max": 16},
            },
            "supports_streaming": True,
        },
        {
            "id": GEMINI,
            "supported_parameters": {
                "n": {"type": "range", "min": 1, "max": 1},
                "input_references": {"type": "range", "min": 0, "max": 14},
            },
            "supports_streaming": False,
        },
    ]
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


def _openrouter(tmp_path: Path) -> TestClient:
    respx.get(f"{OPENROUTER}/v1/models/user").mock(return_value=httpx.Response(200, json=LISTING))
    respx.get(f"{OPENROUTER}/v1/images/models").mock(return_value=httpx.Response(200, json=IMAGES))
    app = create_app(settings=Settings(config_file=_config(tmp_path, provider="openrouter")))
    client = TestClient(app)
    return client


def _openai(tmp_path: Path) -> TestClient:
    listing = {"data": [{"id": "gpt-image-1"}, {"id": "dall-e-3"}, {"id": "gpt-4o"}]}
    respx.get(f"{OPENAI}/v1/models").mock(return_value=httpx.Response(200, json=listing))
    app = create_app(settings=Settings(config_file=_config(tmp_path, provider="openai")))
    return TestClient(app)


def _answer(*images: bytes, created: int = 0, **extra: Any) -> dict[str, Any]:
    """OpenRouter's measured shape: b64_json plus media_type, chat-style usage."""
    return {
        "created": created,
        "data": [{"b64_json": B64(i).decode(), "media_type": "image/jpeg"} for i in images],
        "usage": {
            "prompt_tokens": 6,
            "completion_tokens": 4096,
            "total_tokens": 4102,
            "cost": 0.014,
        },
        **extra,
    }


def _ask(model: str = FLUX, **extra: Any) -> dict[str, Any]:
    return {"model": model, "prompt": "a blue square", **extra}


def _ref(raw: bytes, media: str = "image/png") -> dict[str, str]:
    return {"data": B64(raw).decode(), "mediaType": media}


# --------------------------------------------------------------------------- #
# What a model takes
# --------------------------------------------------------------------------- #


@respx.mock
def test_the_images_listing_decides_the_image_surface_and_its_settings(tmp_path: Path) -> None:
    with _openrouter(tmp_path) as client:
        models = {m["id"]: m for m in _ready(client)["models"]}

    flux = models[FLUX]
    assert flux["surfaces"] == ["image"]
    caps = flux["capabilities"]["image"]
    assert caps["maxImages"] == 1 and caps["maxReferences"] == 4 and caps["minReferences"] == 0
    assert caps["outputFormats"] == ["png", "jpeg"]
    # Unlisted quality and background are NOT TAKEN: flux ignores them with
    # a 200 (measured), which is a silent drop.
    assert caps["qualities"] == [] and caps["backgrounds"] == []
    assert caps["streaming"] is False and caps["mask"] is False

    mini = models[MINI]["capabilities"]["image"]
    assert mini["streaming"] is True and mini["qualities"] == ["auto", "low", "medium", "high"]
    # An unlisted output_format is NOT SAID: mini honours it (measured).
    assert "outputFormats" not in mini
    # Gemini's image+text model answers the images route too (measured).
    assert models[GEMINI]["surfaces"] == ["chat", "image"]
    # Image output with no images entry is not an image model.
    assert "image" not in models[AUTO]["surfaces"]


def test_a_model_the_images_listing_drops_loses_its_capabilities() -> None:
    from eugene_plexus_inference_driver.engines._catalogue import from_openrouter

    models = with_openrouter_images(from_openrouter(LISTING), IMAGES)
    again = with_openrouter_images(models, {"data": []})
    assert all("image" not in m.surfaces for m in again)
    assert all(m.capabilities is None or m.capabilities.image is None for m in again)


@respx.mock
def test_an_openai_accounts_image_models_are_carried_as_its_api_checks(tmp_path: Path) -> None:
    with _openai(tmp_path) as client:
        models = {m["id"]: m for m in _ready(client)["models"]}
    image = models["gpt-image-1"]["capabilities"]["image"]
    assert models["gpt-image-1"]["surfaces"] == ["image"]
    assert image == {"streaming": True, "minReferences": 0, "mask": True}
    assert models["dall-e-3"]["capabilities"]["image"]["streaming"] is False


# --------------------------------------------------------------------------- #
# Generation and edits
# --------------------------------------------------------------------------- #


@respx.mock
def test_a_generation_through_openrouter_is_answered_in_our_shape(tmp_path: Path) -> None:
    route = respx.post(f"{OPENROUTER}/v1/images/generations").mock(
        return_value=httpx.Response(200, json=_answer(JPEG))
    )
    with _openrouter(tmp_path) as client:
        _ready(client)
        response = client.post("/v1/image", json=_ask(size="1536x1024", outputFormat="jpeg"))

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["images"][0]["mediaType"] == "image/jpeg"
    assert base64.b64decode(body["images"][0]["data"]) == JPEG
    # `created: 0` is not a time (flux and gemini say it, measured).
    assert "created" not in body
    assert body["usage"] == {
        "inputTokens": 6,
        "outputTokens": 4096,
        "totalTokens": 4102,
        "cost": 0.014,
    }
    sent = json.loads(route.calls.last.request.content)
    assert sent == {
        "model": FLUX,
        "prompt": "a blue square",
        "size": "1536x1024",
        "output_format": "jpeg",
    }


@respx.mock
def test_the_answer_is_labelled_by_its_bytes_not_by_what_was_asked(tmp_path: Path) -> None:
    respx.post(f"{OPENROUTER}/v1/images/generations").mock(
        return_value=httpx.Response(200, json=_answer(PNG))
    )
    with _openrouter(tmp_path) as client:
        _ready(client)
        body = client.post("/v1/image", json=_ask(MINI, outputFormat="webp")).json()
    assert body["images"][0]["mediaType"] == "image/png"


@respx.mock
def test_an_edit_through_openrouter_sends_input_references_as_objects(tmp_path: Path) -> None:
    route = respx.post(f"{OPENROUTER}/v1/images/generations").mock(
        return_value=httpx.Response(200, json=_answer(JPEG))
    )
    with _openrouter(tmp_path) as client:
        _ready(client)
        response = client.post(
            "/v1/image", json=_ask(references=[_ref(PNG), _ref(JPEG, "image/jpeg")])
        )
    assert response.status_code == 200, response.text
    sent = json.loads(route.calls.last.request.content)
    # Plain strings are OpenRouter's 400 (measured); objects are what it takes.
    assert sent["input_references"] == [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + B64(PNG).decode()}},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + B64(JPEG).decode()}},
    ]


@respx.mock
def test_a_mask_is_refused_for_a_backend_that_would_ignore_it(tmp_path: Path) -> None:
    route = respx.post(f"{OPENROUTER}/v1/images/generations").mock(
        return_value=httpx.Response(200, json=_answer(JPEG))
    )
    with _openrouter(tmp_path) as client:
        _ready(client)
        response = client.post("/v1/image", json=_ask(references=[_ref(PNG)], mask=_ref(PNG)))
    assert response.status_code == 400, response.text
    assert "mask" in response.json()["detail"]["detail"]
    assert not route.called


@respx.mock
def test_an_edit_on_openais_api_is_its_multipart_form(tmp_path: Path) -> None:
    route = respx.post(f"{OPENAI}/v1/images/edits").mock(
        return_value=httpx.Response(
            200, json={"created": 1790000000, "data": [{"b64_json": B64(PNG).decode()}]}
        )
    )
    with _openai(tmp_path) as client:
        _ready(client)
        response = client.post(
            "/v1/image",
            json=_ask(
                "gpt-image-1",
                references=[_ref(PNG), _ref(WEBP, "image/webp")],
                mask=_ref(PNG),
                n=2,
                inputFidelity="high",
            ),
        )
    assert response.status_code == 200, response.text
    assert response.json()["created"] == 1790000000
    request = route.calls.last.request
    assert request.headers["content-type"].startswith("multipart/form-data")
    content = request.content if isinstance(request.content, bytes) else b"".join(request.stream)
    assert content.count(b'name="image[]"') == 2
    assert b'name="mask"; filename="mask.png"' in content
    assert b'filename="image1.webp"' in content
    assert b'name="n"\r\n\r\n2' in content
    assert b'name="input_fidelity"\r\n\r\nhigh' in content
    # GPT image models take no response_format; they always answer base64.
    assert b"response_format" not in content


@respx.mock
def test_dall_e_on_openais_api_is_asked_for_base64(tmp_path: Path) -> None:
    route = respx.post(f"{OPENAI}/v1/images/generations").mock(
        return_value=httpx.Response(
            200,
            json={
                "created": 1,
                "data": [{"b64_json": B64(PNG).decode(), "revised_prompt": "a square, blue"}],
            },
        )
    )
    with _openai(tmp_path) as client:
        _ready(client)
        body = client.post("/v1/image", json=_ask("dall-e-3", style="vivid")).json()
    sent = json.loads(route.calls.last.request.content)
    assert sent["response_format"] == "b64_json" and sent["style"] == "vivid"
    assert body["images"][0]["revisedPrompt"] == "a square, blue"


# --------------------------------------------------------------------------- #
# Refused before anything leaves
# --------------------------------------------------------------------------- #


@respx.mock
def test_an_upload_that_is_not_an_image_is_refused_unsent(tmp_path: Path) -> None:
    route = respx.post(f"{OPENROUTER}/v1/images/generations").mock(
        return_value=httpx.Response(200, json=_answer(JPEG))
    )
    with _openrouter(tmp_path) as client:
        _ready(client)
        response = client.post("/v1/image", json=_ask(references=[_ref(b"%PDF-1.7 not an image")]))
    assert response.status_code == 400, response.text
    assert "references[0]" in response.json()["detail"]["detail"]
    assert not route.called


@respx.mock
def test_uploads_over_25_mib_in_all_are_413(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from eugene_plexus_inference_driver.routes import image as image_route

    monkeypatch.setattr(image_route, "MAX_UPLOAD_BYTES", len(PNG) * 2 - 1)
    route = respx.post(f"{OPENROUTER}/v1/images/generations").mock(
        return_value=httpx.Response(200, json=_answer(JPEG))
    )
    with _openrouter(tmp_path) as client:
        _ready(client)
        response = client.post("/v1/image", json=_ask(references=[_ref(PNG), _ref(PNG)]))
    assert response.status_code == 413, response.text
    assert not route.called


@respx.mock
def test_a_model_that_makes_no_images_is_refused(tmp_path: Path) -> None:
    with _openrouter(tmp_path) as client:
        _ready(client)
        response = client.post("/v1/image", json=_ask("mistralai/mistral-nemo"))
    assert response.status_code == 400, response.text
    assert response.json()["detail"]["type"].endswith("#image-unsupported")


@respx.mock
def test_a_providers_refusal_is_relayed_as_a_400_with_its_words(tmp_path: Path) -> None:
    words = "Black Forest Labs refused this prompt for graphic violence"
    respx.post(f"{OPENROUTER}/v1/images/generations").mock(
        return_value=httpx.Response(400, json={"error": {"message": words, "code": 400}})
    )
    with _openrouter(tmp_path) as client:
        _ready(client)
        response = client.post("/v1/image", json=_ask())
    assert response.status_code == 400, response.text
    assert words in response.text
    assert response.json()["detail"]["retryDisposition"] == "terminal"


@respx.mock
def test_an_answer_that_is_not_an_image_is_a_backend_error(tmp_path: Path) -> None:
    respx.post(f"{OPENROUTER}/v1/images/generations").mock(
        return_value=httpx.Response(200, json=_answer(b"<html>oops</html>"))
    )
    with _openrouter(tmp_path) as client:
        _ready(client)
        response = client.post("/v1/image", json=_ask())
    assert response.status_code == 502, response.text


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #


def _sse(*frames: str) -> bytes:
    return "".join(frames).encode()


def _events(text: str) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        out.append((lines["event"], json.loads(lines["data"])))
    return out


@respx.mock
def test_openrouters_bare_stream_becomes_partial_then_done(tmp_path: Path) -> None:
    partial = {
        "type": "image_generation.partial_image",
        "b64_json": B64(PNG).decode(),
        "partial_image_index": 0,
    }
    done = {
        "type": "image_generation.completed",
        "b64_json": B64(PNG).decode(),
        "created": 1790623843,
        "media_type": "image/png",
        "usage": {"prompt_tokens": 9, "completion_tokens": 372, "total_tokens": 381, "cost": 0.003},
    }
    route = respx.post(f"{OPENROUTER}/v1/images/generations").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse(
                ": \n\n",
                f"data: {json.dumps(partial)}\n\n",
                ": \n\n",
                f"data: {json.dumps(done)}\n\n",
                "data: [DONE]\n\n",
            ),
        )
    )
    with _openrouter(tmp_path) as client:
        _ready(client)
        response = client.post("/v1/image/stream", json=_ask(MINI, partialImages=1, quality="low"))
    assert response.status_code == 200, response.text
    events = _events(response.text)
    assert [e for e, _ in events] == ["partial", "done"]
    assert events[0][1]["index"] == 0 and events[0][1]["image"]["mediaType"] == "image/png"
    assert events[1][1]["created"] == 1790623843
    assert events[1][1]["usage"]["outputTokens"] == 372
    sent = json.loads(route.calls.last.request.content)
    assert sent["stream"] is True and sent["partial_images"] == 1


@respx.mock
def test_openais_named_event_stream_is_read_the_same_way(tmp_path: Path) -> None:
    common = {
        "created_at": 5,
        "size": "1024x1024",
        "quality": "low",
        "background": "opaque",
        "output_format": "png",
    }
    frames = [
        {
            "type": "image_edit.partial_image",
            "b64_json": B64(PNG).decode(),
            "partial_image_index": 0,
            **common,
        },
        {
            "type": "image_edit.completed",
            "b64_json": B64(PNG).decode(),
            **common,
            "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
        },
    ]
    respx.post(f"{OPENAI}/v1/images/edits").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse(*(f"event: {f['type']}\ndata: {json.dumps(f)}\n\n" for f in frames)),
        )
    )
    with _openai(tmp_path) as client:
        _ready(client)
        response = client.post("/v1/image/stream", json=_ask("gpt-image-1", references=[_ref(PNG)]))
    events = _events(response.text)
    assert [e for e, _ in events] == ["partial", "done"]
    assert events[1][1]["size"] == "1024x1024" and events[1][1]["usage"]["totalTokens"] == 3


@respx.mock
def test_a_model_that_does_not_stream_is_refused_before_anything_is_sent(tmp_path: Path) -> None:
    route = respx.post(f"{OPENROUTER}/v1/images/generations").mock(
        return_value=httpx.Response(200, json=_answer(JPEG))
    )
    with _openrouter(tmp_path) as client:
        _ready(client)
        response = client.post("/v1/image/stream", json=_ask(FLUX))
    assert response.status_code == 400, response.text
    assert "stream" in response.json()["detail"]["detail"]
    assert not route.called


@respx.mock
def test_json_where_a_stream_was_asked_for_is_refused_not_passed_off(tmp_path: Path) -> None:
    """OpenRouter answers plain JSON to a model that cannot stream (measured).
    Here the listing said it streams and the backend disagreed."""
    respx.post(f"{OPENROUTER}/v1/images/generations").mock(
        return_value=httpx.Response(200, json=_answer(PNG))
    )
    with _openrouter(tmp_path) as client:
        _ready(client)
        response = client.post("/v1/image/stream", json=_ask(MINI))
    assert response.status_code == 400, response.text
    assert "JSON document" in response.json()["detail"]["detail"]


@respx.mock
def test_an_error_event_before_any_image_is_a_status_code(tmp_path: Path) -> None:
    respx.post(f"{OPENROUTER}/v1/images/generations").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse('data: {"type": "error", "error": {"message": "provider exploded"}}\n\n'),
        )
    )
    with _openrouter(tmp_path) as client:
        _ready(client)
        response = client.post("/v1/image/stream", json=_ask(MINI))
    assert response.status_code == 502, response.text
    assert "provider exploded" in response.text


def test_the_sniffer_reads_every_format_a_backend_answers_in() -> None:
    assert sniff(PNG) == "image/png"
    assert sniff(JPEG) == "image/jpeg"
    assert sniff(WEBP) == "image/webp"
    assert sniff(b"GIF89a" + bytes(8)) == "image/gif"
    assert sniff(b'<?xml version="1.0"?><svg xmlns="x"/>') == "image/svg+xml"
    assert sniff(b"%PDF-1.7") is None
