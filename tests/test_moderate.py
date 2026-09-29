"""POST /v1/moderate through an OpenAI account (P6).

Measured 2026-09-28 (`provider-accounts-measurement.md` section 12): OpenAI
lists `omni-moderation-latest` and `omni-moderation-2024-09-26`, takes a
string, an array of strings (one result each) or an array of parts (one
result), and moderates at most one image; OpenRouter answers 404. Every test
here fails against the driver before P6, which had no `/v1/moderate`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import respx
from fastapi.testclient import TestClient

from eugene_plexus_inference_driver.app import create_app
from eugene_plexus_inference_driver.settings import Settings

OPENAI = "https://api.openai.com"
OPENROUTER = "https://openrouter.ai/api"
IMAGE = "data:image/png;base64,iVBORw0KGgo="
RESULT = {
    "flagged": True,
    "categories": {"violence": True, "harassment": False},
    "category_scores": {"violence": 0.87, "harassment": 0.36},
    "category_applied_input_types": {"violence": ["text"], "harassment": ["text"]},
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


def _openai(tmp_path: Path, answer: httpx.Response) -> tuple[TestClient, respx.Route]:
    listing = {"data": [{"id": "omni-moderation-latest"}, {"id": "gpt-4o"}]}
    respx.get(f"{OPENAI}/v1/models").mock(return_value=httpx.Response(200, json=listing))
    route = respx.post(f"{OPENAI}/v1/moderations").mock(return_value=answer)
    app = create_app(settings=Settings(config_file=_config(tmp_path, provider="openai")))
    return TestClient(app), route


def _answer(*results: dict) -> httpx.Response:
    return httpx.Response(
        200, json={"id": "modr-1", "model": "omni-moderation-latest", "results": list(results)}
    )


@respx.mock
def test_an_openai_moderation_model_is_listed_and_asked_in_openais_shape(tmp_path: Path) -> None:
    client, upstream = _openai(tmp_path, _answer(RESULT, {"flagged": False}))
    with client:
        info = _ready(client)
        response = client.post(
            "/v1/moderate", json={"model": "omni-moderation-latest", "texts": ["a", "b"]}
        )
    surfaces = {m["id"]: m["surfaces"] for m in info["models"]}
    assert surfaces["omni-moderation-latest"] == ["moderation"]
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["results"] == [RESULT, {"flagged": False}] and body["id"] == "modr-1"
    assert body["modelId"] == "omni-moderation-latest"
    sent = json.loads(upstream.calls[0].request.content)
    assert sent == {"model": "omni-moderation-latest", "input": ["a", "b"]}
    assert upstream.calls[0].request.headers["authorization"] == "Bearer sk-test"


@respx.mock
def test_parts_are_one_multimodal_input(tmp_path: Path) -> None:
    client, upstream = _openai(tmp_path, _answer(RESULT))
    with client:
        _ready(client)
        response = client.post(
            "/v1/moderate",
            json={
                "model": "omni-moderation-latest",
                "parts": [{"type": "text", "text": "look"}, {"type": "image", "image": IMAGE}],
            },
        )
    assert response.status_code == 200, response.text
    assert json.loads(upstream.calls[0].request.content)["input"] == [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": IMAGE}},
    ]


@respx.mock
def test_what_is_not_a_moderation_is_refused_before_anything_is_sent(tmp_path: Path) -> None:
    client, upstream = _openai(tmp_path, _answer(RESULT))
    with client:
        _ready(client)
        answers = {
            "chat model": client.post("/v1/moderate", json={"model": "gpt-4o", "texts": ["a"]}),
            "neither": client.post("/v1/moderate", json={"model": "omni-moderation-latest"}),
            "both": client.post(
                "/v1/moderate",
                json={
                    "model": "omni-moderation-latest",
                    "texts": ["a"],
                    "parts": [{"type": "text", "text": "b"}],
                },
            ),
            "remote image": client.post(
                "/v1/moderate",
                json={
                    "model": "omni-moderation-latest",
                    "parts": [{"type": "image", "image": "https://example.com/x.png"}],
                },
            ),
        }
    for label, response in answers.items():
        assert response.status_code == 400, (label, response.text)
    assert answers["chat model"].json()["detail"]["type"].endswith("#moderation-unsupported")
    assert "A4" in answers["remote image"].json()["detail"]["detail"]
    assert not upstream.calls


@respx.mock
def test_the_backends_refusal_is_relayed_with_its_words(tmp_path: Path) -> None:
    refusal = {"error": {"message": "Number of images (2) exceeds maximum of 1", "code": "x"}}
    client, _ = _openai(tmp_path, httpx.Response(400, json=refusal))
    with client:
        _ready(client)
        response = client.post(
            "/v1/moderate",
            json={
                "model": "omni-moderation-latest",
                "parts": [{"type": "image", "image": IMAGE}, {"type": "image", "image": IMAGE}],
            },
        )
    assert response.status_code == 400, response.text
    assert "exceeds maximum of 1" in response.json()["detail"]["detail"]


@respx.mock
def test_openrouter_offers_no_moderation(tmp_path: Path) -> None:
    listing = {
        "data": [
            {
                "id": "mistralai/mistral-nemo",
                "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
                "supported_parameters": ["max_tokens"],
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
    moderations = respx.post(f"{OPENROUTER}/v1/moderations").mock(
        return_value=httpx.Response(404, json={"error": {"message": "Not Found"}})
    )
    config = _config(tmp_path, provider="openrouter")
    with TestClient(create_app(settings=Settings(config_file=config))) as client:
        _ready(client)
        response = client.post(
            "/v1/moderate", json={"model": "mistralai/mistral-nemo", "texts": ["a"]}
        )
    assert response.status_code == 400 and not moderations.calls
