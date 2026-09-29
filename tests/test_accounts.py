"""One driver, many models: a provider account (P1, 2026-09-27).

An OpenAI-compatible driver with no `modelId` serves every model its
backend lists. Each test here fails against the driver as it was before:
one `modelId` on `/v1/info`, no `model` on a request, and a backend asked
for its one configured model whatever the caller wanted.

The upstream bodies are shaped after what OpenRouter and Ollama actually
returned on 2026-09-27 (`docs/acceptance/provider-accounts-measurement.md`
in specs), trimmed to the fields read.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from eugene_plexus_inference_driver.app import create_app
from eugene_plexus_inference_driver.engines._catalogue import (
    exposed_by,
    from_openrouter,
    matches,
)
from eugene_plexus_inference_driver.engines.openai_compat_http import classify_openai_model
from eugene_plexus_inference_driver.settings import Settings

OPENROUTER = "https://openrouter.ai/api"
OLLAMA = "http://127.0.0.1:11434"


def _or_model(model_id: str, **overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": model_id,
        "name": model_id.split("/")[-1],
        "context_length": 131072,
        "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
        "supported_parameters": ["max_tokens", "temperature", "tools", "tool_choice", "seed"],
        "supported_voices": None,
    }
    body.update(overrides)
    return body


OPENROUTER_LIST = {
    "data": [
        _or_model("mistralai/mistral-nemo"),
        _or_model("openai/gpt-oss-20b", supported_parameters=["max_tokens", "response_format"]),
        _or_model(
            "google/gemini-3.1-flash-lite-image",
            architecture={
                "input_modalities": ["text", "image"],
                "output_modalities": ["image", "text"],
            },
        ),
        _or_model(
            "hexgrad/kokoro-82m",
            context_length=0,
            architecture={"input_modalities": ["text"], "output_modalities": ["speech"]},
            supported_parameters=[],
            supported_voices=["af_alloy", "af_aoede"],
        ),
        _or_model("~z-ai/glm-flash-latest"),
        {"id": 7},  # one malformed row must not cost the others
    ]
}


def _completion(model: str, text: str = "pong") -> dict[str, Any]:
    return {
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }


def _write(path: Path, fields: dict[str, Any]) -> None:
    path.write_text(json.dumps(fields), encoding="utf-8")  # JSON is YAML


def _wait_for_catalogue(client: TestClient, timeout: float = 5.0) -> dict[str, Any]:
    deadline = time.perf_counter() + timeout
    while True:
        info = client.get("/v1/info").json()
        catalogue = info.get("catalogue") or {}
        if catalogue.get("refreshedAt") or catalogue.get("error"):
            return info
        if time.perf_counter() > deadline:
            raise AssertionError(f"catalogue never read: {info}")
        time.sleep(0.02)


@pytest.fixture
def openrouter_config(tmp_path: Path) -> Path:
    config = tmp_path / "openrouter.yaml"
    _write(config, {"provider": "openrouter", "apiKey": "sk-or-test"})
    return config


# --------------------------------------------------------------------------- #
# /v1/info
# --------------------------------------------------------------------------- #


@respx.mock
def test_an_openrouter_account_lists_its_models_not_one(openrouter_config: Path) -> None:
    listing = respx.get(f"{OPENROUTER}/v1/models/user").mock(
        return_value=httpx.Response(200, json=OPENROUTER_LIST)
    )
    respx.get(f"{OPENROUTER}/v1/images/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    respx.get(f"{OPENROUTER}/v1/videos/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    with TestClient(create_app(settings=Settings(config_file=openrouter_config))) as client:
        info = _wait_for_catalogue(client)

    # The account's own list, every modality: the measured difference
    # between 625 and the public 458.
    assert listing.calls.last.request.url.params["output_modalities"] == "all"
    assert "modelId" not in info
    by_id = {m["id"]: m for m in info["models"]}
    assert sorted(by_id) == [
        "google/gemini-3.1-flash-lite-image",
        "hexgrad/kokoro-82m",
        "mistralai/mistral-nemo",
        "openai/gpt-oss-20b",
        "~z-ai/glm-flash-latest",
    ]
    assert info["catalogue"]["source"] == "openrouter"
    assert info["catalogue"]["total"] == 5 and info["catalogue"]["exposed"] == 5
    assert info["catalogue"]["include"] == ["*"]

    nemo = by_id["mistralai/mistral-nemo"]
    assert nemo["surfaces"] == ["chat"]
    assert nemo["capabilities"]["toolCalling"] is True
    assert nemo["capabilities"]["maxContextTokens"] == 131072
    assert set(nemo["capabilities"]["supportedSettings"]) == {
        "maxTokens",
        "temperature",
        "tools",
        "toolChoice",
        "seed",
    }
    # Per model, not per driver: this one takes no tools.
    assert by_id["openai/gpt-oss-20b"]["capabilities"]["toolCalling"] is False
    image = by_id["google/gemini-3.1-flash-lite-image"]
    assert image["surfaces"] == ["chat"] and image["capabilities"]["imageInput"] is True
    speech = by_id["hexgrad/kokoro-82m"]
    assert speech["surfaces"] == ["speech"]
    assert speech["voices"] == ["af_alloy", "af_aoede"]


@respx.mock
def test_models_false_leaves_the_list_out(openrouter_config: Path) -> None:
    respx.get(f"{OPENROUTER}/v1/models/user").mock(
        return_value=httpx.Response(200, json=OPENROUTER_LIST)
    )
    respx.get(f"{OPENROUTER}/v1/images/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    respx.get(f"{OPENROUTER}/v1/videos/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    with TestClient(create_app(settings=Settings(config_file=openrouter_config))) as client:
        _wait_for_catalogue(client)
        light = client.get("/v1/info", params={"models": "false"}).json()
    assert "models" not in light
    assert light["catalogue"]["exposed"] == 5
    assert light["localOnlyEnforced"] is True


# --------------------------------------------------------------------------- #
# the request names its model
# --------------------------------------------------------------------------- #


@respx.mock
def test_each_request_reaches_the_backend_as_the_model_it_named(openrouter_config: Path) -> None:
    respx.get(f"{OPENROUTER}/v1/models/user").mock(
        return_value=httpx.Response(200, json=OPENROUTER_LIST)
    )
    respx.get(f"{OPENROUTER}/v1/images/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    respx.get(f"{OPENROUTER}/v1/videos/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    sent: list[str] = []

    def answer(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        sent.append(model)
        # An alias answers under its target's id, as OpenRouter's did.
        echoed = "z-ai/glm-5.3-flash" if model.startswith("~") else model
        return httpx.Response(200, json=_completion(echoed))

    respx.post(f"{OPENROUTER}/v1/chat/completions").mock(side_effect=answer)
    with TestClient(create_app(settings=Settings(config_file=openrouter_config))) as client:
        _wait_for_catalogue(client)
        replies = [
            client.post(
                "/v1/generate",
                json={"model": model, "messages": [{"role": "user", "content": "ping"}]},
            ).json()
            for model in ("mistralai/mistral-nemo", "openai/gpt-oss-20b", "~z-ai/glm-flash-latest")
        ]
    assert sent == ["mistralai/mistral-nemo", "openai/gpt-oss-20b", "~z-ai/glm-flash-latest"]
    # And each answer is reported as the model that was asked for -- the
    # alias included, which the backend answered as another id.
    assert [r["modelId"] for r in replies] == sent


@respx.mock
def test_a_model_the_account_does_not_list_is_404_and_reaches_nothing(
    openrouter_config: Path,
) -> None:
    respx.get(f"{OPENROUTER}/v1/models/user").mock(
        return_value=httpx.Response(200, json=OPENROUTER_LIST)
    )
    respx.get(f"{OPENROUTER}/v1/images/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    respx.get(f"{OPENROUTER}/v1/videos/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    chat = respx.post(f"{OPENROUTER}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_completion("x"))
    )
    with TestClient(create_app(settings=Settings(config_file=openrouter_config))) as client:
        _wait_for_catalogue(client)
        missing = client.post(
            "/v1/generate",
            json={"model": "anthropic/not-listed", "messages": [{"role": "user", "content": "hi"}]},
        )
        unnamed = client.post(
            "/v1/generate", json={"messages": [{"role": "user", "content": "hi"}]}
        )
        streamed = client.post(
            "/v1/generate/stream",
            json={"model": "anthropic/not-listed", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert missing.status_code == 404
    assert missing.json()["detail"]["type"].endswith("#model-not-served")
    assert missing.json()["detail"]["retryDisposition"] == "safe"
    assert unnamed.status_code == 400
    assert unnamed.json()["detail"]["type"].endswith("#model-required")
    assert streamed.status_code == 404
    assert not chat.called


@respx.mock
def test_a_speech_model_refuses_chat_with_what_it_is(openrouter_config: Path) -> None:
    respx.get(f"{OPENROUTER}/v1/models/user").mock(
        return_value=httpx.Response(200, json=OPENROUTER_LIST)
    )
    respx.get(f"{OPENROUTER}/v1/images/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    respx.get(f"{OPENROUTER}/v1/videos/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    chat = respx.post(f"{OPENROUTER}/v1/chat/completions")
    with TestClient(create_app(settings=Settings(config_file=openrouter_config))) as client:
        _wait_for_catalogue(client)
        refused = client.post(
            "/v1/generate",
            json={"model": "hexgrad/kokoro-82m", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert refused.status_code == 400
    assert "speech" in refused.json()["detail"]["detail"]
    assert not chat.called


@respx.mock
def test_tools_go_only_to_a_model_whose_listing_takes_them(openrouter_config: Path) -> None:
    respx.get(f"{OPENROUTER}/v1/models/user").mock(
        return_value=httpx.Response(200, json=OPENROUTER_LIST)
    )
    respx.get(f"{OPENROUTER}/v1/images/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    respx.get(f"{OPENROUTER}/v1/videos/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    chat = respx.post(f"{OPENROUTER}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_completion("mistralai/mistral-nemo"))
    )
    tool = {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}
    with TestClient(create_app(settings=Settings(config_file=openrouter_config))) as client:
        _wait_for_catalogue(client)
        refused = client.post(
            "/v1/generate",
            json={
                "model": "openai/gpt-oss-20b",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [tool],
            },
        )
        carried = client.post(
            "/v1/generate",
            json={
                "model": "mistralai/mistral-nemo",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [tool],
                "callerSettings": ["tools"],
            },
        )
    assert refused.status_code == 400
    assert carried.status_code == 200
    body = json.loads(chat.calls.last.request.content)
    assert body["tools"][0]["function"]["name"] == "f"
    # A setting the caller asked for is one OpenRouter must not drop.
    assert body["provider"] == {"require_parameters": True}


@respx.mock
def test_an_explicit_setting_the_model_does_not_list_is_refused_not_dropped(
    openrouter_config: Path,
) -> None:
    respx.get(f"{OPENROUTER}/v1/models/user").mock(
        return_value=httpx.Response(200, json=OPENROUTER_LIST)
    )
    respx.get(f"{OPENROUTER}/v1/images/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    respx.get(f"{OPENROUTER}/v1/videos/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    chat = respx.post(f"{OPENROUTER}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_completion("openai/gpt-oss-20b"))
    )
    with TestClient(create_app(settings=Settings(config_file=openrouter_config))) as client:
        _wait_for_catalogue(client)
        refused = client.post(
            "/v1/generate",
            json={
                "model": "openai/gpt-oss-20b",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.3,
                "callerSettings": ["temperature"],
            },
        )
        plain = client.post(
            "/v1/generate",
            json={"model": "openai/gpt-oss-20b", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert refused.status_code == 400, refused.json()
    assert plain.status_code == 200
    # Nothing explicit, so nothing for OpenRouter to be strict about.
    assert "provider" not in json.loads(chat.calls.last.request.content)


# --------------------------------------------------------------------------- #
# filters, failures, persistence
# --------------------------------------------------------------------------- #


@respx.mock
def test_patterns_apply_live_without_a_restart(openrouter_config: Path) -> None:
    respx.get(f"{OPENROUTER}/v1/models/user").mock(
        return_value=httpx.Response(200, json=OPENROUTER_LIST)
    )
    respx.get(f"{OPENROUTER}/v1/images/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    respx.get(f"{OPENROUTER}/v1/videos/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    respx.post(f"{OPENROUTER}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_completion("mistralai/mistral-nemo"))
    )
    with TestClient(create_app(settings=Settings(config_file=openrouter_config))) as client:
        _wait_for_catalogue(client)
        patched = client.patch(
            "/v1/config",
            json={"catalogueInclude": ["mistralai/*", "openai/*"], "catalogueExclude": ["*-20b"]},
        ).json()
        info = client.get("/v1/info").json()
        excluded = client.post(
            "/v1/generate",
            json={"model": "openai/gpt-oss-20b", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert patched["requiresRestart"] is False, patched
    assert [m["id"] for m in info["models"]] == ["mistralai/mistral-nemo"]
    assert info["catalogue"]["total"] == 5 and info["catalogue"]["exposed"] == 1
    assert excluded.status_code == 404


@respx.mock
def test_a_failed_read_keeps_the_last_good_list_and_says_why(
    openrouter_config: Path,
) -> None:
    listing = respx.get(f"{OPENROUTER}/v1/models/user").mock(
        return_value=httpx.Response(200, json=OPENROUTER_LIST)
    )
    respx.get(f"{OPENROUTER}/v1/images/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    respx.get(f"{OPENROUTER}/v1/videos/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    with TestClient(create_app(settings=Settings(config_file=openrouter_config))) as client:
        _wait_for_catalogue(client)
        engine = client.app.state.adapter
        listing.mock(
            return_value=httpx.Response(
                401, json={"error": {"message": "missing the permission models_read"}}
            )
        )
        client.portal.call(engine.catalogue.refresh)  # type: ignore[union-attr]
        info = client.get("/v1/info").json()
    assert len(info["models"]) == 5
    assert "missing the permission models_read" in info["catalogue"]["error"]


@respx.mock
def test_a_restart_with_the_upstream_down_still_serves_the_saved_list(
    openrouter_config: Path,
) -> None:
    listing = respx.get(f"{OPENROUTER}/v1/models/user").mock(
        return_value=httpx.Response(200, json=OPENROUTER_LIST)
    )
    respx.get(f"{OPENROUTER}/v1/images/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    respx.get(f"{OPENROUTER}/v1/videos/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    with TestClient(create_app(settings=Settings(config_file=openrouter_config))) as client:
        _wait_for_catalogue(client)
    saved = openrouter_config.with_name("openrouter.catalogue.json")
    assert saved.exists()

    listing.mock(side_effect=httpx.ConnectError("upstream down"))
    with TestClient(create_app(settings=Settings(config_file=openrouter_config))) as client:
        info = _wait_for_catalogue(client)
    assert len(info["models"]) == 5
    assert info["catalogue"]["refreshedAt"] is not None
    assert "could not reach" in info["catalogue"]["error"]


@respx.mock
def test_a_saved_list_from_another_backend_is_not_served(tmp_path: Path) -> None:
    config = tmp_path / "acct.yaml"
    _write(config, {"provider": "openrouter", "apiKey": "sk-or-test"})
    config.with_name("acct.catalogue.json").write_text(
        json.dumps(
            {
                "origin": "ollama_local|http://127.0.0.1:11434",
                "models": [{"id": "qwen3:8b", "surfaces": ["chat"]}],
            }
        ),
        encoding="utf-8",
    )
    respx.get(f"{OPENROUTER}/v1/models/user").mock(side_effect=httpx.ConnectError("down"))
    respx.get(f"{OPENROUTER}/v1/images/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    respx.get(f"{OPENROUTER}/v1/videos/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    with TestClient(create_app(settings=Settings(config_file=config))) as client:
        info = _wait_for_catalogue(client)
    assert info["models"] == []


# --------------------------------------------------------------------------- #
# other providers
# --------------------------------------------------------------------------- #


@respx.mock
def test_an_ollama_account_lists_every_pulled_model_with_its_capabilities(
    tmp_path: Path,
) -> None:
    respx.get(f"{OLLAMA}/api/tags").mock(
        return_value=httpx.Response(
            200,
            json={
                "models": [
                    {"name": "qwen3-coder:30b"},
                    {"name": "nomic-embed-text:latest"},
                    {"name": "gemma3:4b"},
                ]
            },
        )
    )
    shows = {
        "qwen3-coder:30b": ["completion", "tools"],
        "nomic-embed-text:latest": ["embedding"],
        "gemma3:4b": ["completion", "vision"],
    }
    respx.post(f"{OLLAMA}/api/show").mock(
        side_effect=lambda r: httpx.Response(
            200, json={"capabilities": shows[json.loads(r.content)["model"]]}
        )
    )
    config = tmp_path / "ollama.yaml"
    _write(config, {"provider": "ollama_local"})
    with TestClient(create_app(settings=Settings(config_file=config))) as client:
        info = _wait_for_catalogue(client)
    by_id = {m["id"]: m for m in info["models"]}
    assert info["catalogue"]["source"] == "ollama"
    # P6: a model Ollama lists as `completion` also continues raw text
    # (through /api/generate with `raw`).
    assert by_id["qwen3-coder:30b"]["surfaces"] == ["chat", "completion"]
    assert by_id["qwen3-coder:30b"]["capabilities"]["toolCalling"] is True
    assert by_id["nomic-embed-text:latest"]["surfaces"] == ["embeddings"]
    assert by_id["gemma3:4b"]["capabilities"]["imageInput"] is True
    assert by_id["gemma3:4b"]["capabilities"]["toolCalling"] is False


@respx.mock
def test_an_account_embeds_with_the_model_it_was_asked_for(tmp_path: Path) -> None:
    respx.get(f"{OLLAMA}/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": "nomic-embed-text:latest"}]})
    )
    respx.post(f"{OLLAMA}/api/show").mock(
        return_value=httpx.Response(200, json={"capabilities": ["embedding"]})
    )
    embed = respx.post(f"{OLLAMA}/v1/embeddings").mock(
        return_value=httpx.Response(
            200, json={"model": "nomic-embed-text:latest", "data": [{"embedding": [0.1, 0.2]}]}
        )
    )
    config = tmp_path / "ollama.yaml"
    _write(config, {"provider": "ollama_local"})
    with TestClient(create_app(settings=Settings(config_file=config))) as client:
        _wait_for_catalogue(client)
        vectors = client.post(
            "/v1/embed", json={"model": "nomic-embed-text:latest", "input": ["a"]}
        )
    assert vectors.status_code == 200, vectors.json()
    assert json.loads(embed.calls.last.request.content)["model"] == "nomic-embed-text:latest"


@respx.mock
def test_the_config_test_of_an_account_reads_its_list(openrouter_config: Path) -> None:
    respx.get(f"{OPENROUTER}/v1/models/user").mock(
        return_value=httpx.Response(200, json=OPENROUTER_LIST)
    )
    respx.get(f"{OPENROUTER}/v1/images/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    respx.get(f"{OPENROUTER}/v1/videos/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    chat = respx.post(f"{OPENROUTER}/v1/chat/completions")
    with TestClient(create_app(settings=Settings(config_file=openrouter_config))) as client:
        result = client.post("/v1/config/test").json()
    assert result["ok"] is True, result
    assert "5 models" in result["summary"]
    assert not chat.called  # an account's test generates nothing


# --------------------------------------------------------------------------- #
# a single-model driver is unchanged
# --------------------------------------------------------------------------- #


@respx.mock
def test_a_single_model_driver_serves_its_model_and_refuses_another(tmp_path: Path) -> None:
    base = "http://127.0.0.1:8090"
    chat = respx.post(f"{base}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_completion("qwen"))
    )
    config = tmp_path / "single.yaml"
    _write(config, {"provider": "openai_compat_custom", "baseUrl": base, "modelId": "qwen"})
    with TestClient(create_app(settings=Settings(config_file=config))) as client:
        info = client.get("/v1/info").json()
        unnamed = client.post("/v1/generate", json={"messages": [{"role": "user", "content": "a"}]})
        named = client.post(
            "/v1/generate", json={"model": "qwen", "messages": [{"role": "user", "content": "a"}]}
        )
        other = client.post(
            "/v1/generate", json={"model": "llama", "messages": [{"role": "user", "content": "a"}]}
        )
    assert [m["id"] for m in info["models"]] == ["qwen"]
    assert "catalogue" not in info
    assert unnamed.status_code == 200 and named.status_code == 200
    assert other.status_code == 404
    assert chat.call_count == 2


# --------------------------------------------------------------------------- #
# the pieces
# --------------------------------------------------------------------------- #


def test_star_is_the_only_wildcard_and_it_crosses_slashes() -> None:
    assert matches("openrouter/*", "openrouter/anthropic/claude-opus-5.5")
    assert matches("*:free", "respan/span-01-lite:free")
    assert not matches("anthropic/*", "openai/gpt-6")
    assert matches("a?b", "a?b") and not matches("a?b", "axb")
    assert matches("[x]", "[x]") and not matches("[x]", "x")
    assert exposed_by(["*"], ["*-20b"], "openai/gpt-oss-120b")
    assert not exposed_by(["*"], ["*-20b"], "openai/gpt-oss-20b")
    assert not exposed_by([], [], "anything")


def test_openais_own_list_is_sorted_by_surface_not_filtered() -> None:
    assert classify_openai_model("gpt-4o") == ["chat"]
    assert classify_openai_model("o3-mini") == ["chat"]
    assert classify_openai_model("text-embedding-3-large") == ["embeddings"]
    assert classify_openai_model("tts-1-hd") == ["speech"]
    # Only whisper translates (P3-4): OpenAI answers /audio/translations 404
    # for its gpt-4o transcribe models (measured).
    assert classify_openai_model("whisper-1") == ["transcription", "translation"]
    assert classify_openai_model("gpt-4o-mini-transcribe") == ["transcription"]
    assert classify_openai_model("gpt-image-1") == ["image"]
    assert classify_openai_model("chatgpt-image-latest") == ["image"]
    assert classify_openai_model("omni-moderation-latest") == ["moderation"]
    assert classify_openai_model("babbage-002") == []


def test_one_malformed_row_does_not_cost_the_rest() -> None:
    models = from_openrouter({"data": [{"id": None}, "junk", {"id": "a/b"}]})
    assert [m.id for m in models] == ["a/b"]


@respx.mock
def test_info_for_one_model_answers_that_entry_alone(openrouter_config: Path) -> None:
    respx.get(f"{OPENROUTER}/v1/models/user").mock(
        return_value=httpx.Response(200, json=OPENROUTER_LIST)
    )
    respx.get(f"{OPENROUTER}/v1/images/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    respx.get(f"{OPENROUTER}/v1/videos/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    with TestClient(create_app(settings=Settings(config_file=openrouter_config))) as client:
        _wait_for_catalogue(client)
        one = client.get("/v1/info", params={"model": "mistralai/mistral-nemo"}).json()
        none = client.get("/v1/info", params={"model": "not/listed"}).json()
    assert [m["id"] for m in one["models"]] == ["mistralai/mistral-nemo"]
    assert none["models"] == []
    assert one["catalogue"]["exposed"] == 5
