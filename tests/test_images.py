"""The direct driver boundary validates images and confirms the loaded model."""

import base64
import io
import json
import logging

import pytest
import respx
from fastapi.testclient import TestClient
from PIL import Image

from eugene_plexus_inference_driver._generated.models import GenerateRequest
from eugene_plexus_inference_driver.engines.claude_code_cli import ClaudeCodeCliEngine
from eugene_plexus_inference_driver.engines.openai_compat_http import OpenAiCompatibleHttpEngine

BASE = "http://vision-test"


def request():
    data = io.BytesIO()
    Image.new("RGB", (8, 8), "purple").save(data, format="PNG")
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,"
                            + base64.b64encode(data.getvalue()).decode()
                        },
                    },
                ],
            }
        ]
    }


def engine():
    return OpenAiCompatibleHttpEngine(base_url=BASE, model_id="vision", auth_required=False)


def probes(vision=True, model="vision"):
    respx.get(BASE + "/props").respond(200, json={"modalities": {"vision": vision}})
    respx.get(BASE + "/v1/models").respond(200, json={"data": [{"id": model}]})


@pytest.mark.parametrize("stream", [False, True])
@respx.mock
def test_forwards_exact_parts_and_never_logs_payload(app, caplog, stream):
    probes()
    body = request()
    wire = {"choices": [{"message": {"content": "purple"}, "finish_reason": "stop"}]}
    if stream:
        wire = (
            "data: "
            + json.dumps({"choices": [{"delta": {"content": "purple"}, "finish_reason": "stop"}]})
            + "\n\ndata: [DONE]\n\n"
        )
        upstream = respx.post(BASE + "/v1/chat/completions").respond(200, text=wire)
    else:
        upstream = respx.post(BASE + "/v1/chat/completions").respond(200, json=wire)
    caplog.set_level(logging.DEBUG)
    with TestClient(app) as client:
        app.state.adapter = engine()
        response = client.post("/v1/generate" + ("/stream" if stream else ""), json=body)
    assert response.status_code == 200, response.text
    assert "purple" in response.text
    assert json.loads(upstream.calls[0].request.content)["messages"] == body["messages"]
    assert "data:image" not in caplog.text


@pytest.mark.parametrize("stream", [False, True])
@respx.mock
def test_backend_echo_is_not_exposed(app, caplog, stream):
    probes()
    body = request()
    respx.post(BASE + "/v1/chat/completions").respond(400, json=body)
    caplog.set_level(logging.DEBUG)
    with TestClient(app) as client:
        app.state.adapter = engine()
        response = client.post("/v1/generate" + ("/stream" if stream else ""), json=body)
    assert response.status_code == 400
    assert "projector" in response.text
    assert "data:image" not in caplog.text + response.text


@pytest.mark.asyncio
@respx.mock
async def test_capability_is_model_specific_and_rechecked():
    adapter = engine()
    probes()
    assert await adapter.probe_image_input() is True
    probes(vision=False)
    assert await adapter.probe_image_input() is False
    probes(model="different-model")
    assert await adapter.probe_image_input() is False


@pytest.mark.parametrize("stream", [False, True])
def test_cli_cannot_flatten_an_image(app, stream):
    with TestClient(app) as client:
        app.state.adapter = ClaudeCodeCliEngine(binary_path="a4-no-such-executable")
        response = client.post("/v1/generate" + ("/stream" if stream else ""), json=request())
    assert response.status_code == 400
    assert "vision model" in response.text


def test_invalid_schema_never_reflects_image_data(client):
    body = request()
    body["messages"][0]["content"][1]["image_url"]["detail"] = "PRIVATE-INVALID"
    response = client.post("/v1/generate", json=body)
    assert response.status_code == 422
    assert "PRIVATE" not in response.text
    assert "data:image" not in response.text


def test_thinking_directive_preserves_image():
    adapter = engine()
    adapter._thinking_mode = "off"
    body = request()
    payload = adapter._payload_for(GenerateRequest.model_validate(body))
    assert payload["messages"][-1] == body["messages"][0]


@pytest.mark.parametrize("stream", [False, True])
@respx.mock
def test_direct_driver_rejects_remote_images_before_any_network(app, stream):
    body = request()
    body["messages"][0]["content"][1]["image_url"]["url"] = "http://127.0.0.1/PRIVATE"
    with TestClient(app) as client:
        app.state.adapter = engine()
        response = client.post("/v1/generate" + ("/stream" if stream else ""), json=body)
    assert response.status_code == 400
    assert "URLs are not fetched" in response.text
    assert "PRIVATE" not in response.text
    assert not respx.calls


def test_direct_driver_chunked_body_is_bounded(client, monkeypatch):
    from eugene_plexus_inference_driver import body_limit

    monkeypatch.setattr(body_limit, "MAX_BODY_BYTES", 128)
    response = client.post("/v1/generate", content=iter([b" " * 80, b" " * 80]))
    assert response.status_code == 413
    assert "16 MiB" in response.text


def _with_images(count):
    body = request()
    picture = body["messages"][0]["content"][1]
    body["messages"][0]["content"] = [body["messages"][0]["content"][0]] + [picture] * count
    return body


def test_the_gateways_default_of_twelve_passes_the_drivers_ceiling():
    """Four until 2026-09-23. The gateway's `maxImagesPerRequest` is the
    policy now (12 by default); this is only the ceiling it cannot exceed."""
    from eugene_plexus_inference_driver import images

    parsed = GenerateRequest.model_validate(_with_images(12))
    images.validate_messages(parsed.messages)
    assert images.MAX_IMAGES == 64


@respx.mock
def test_more_than_the_ceiling_is_refused_before_any_network(app):
    with TestClient(app) as client:
        app.state.adapter = engine()
        response = client.post("/v1/generate", json=_with_images(65))
    assert response.status_code == 400
    assert "at most 64 images" in response.text
    assert not respx.calls
