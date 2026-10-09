"""Google's Gemini API as a provider account (gemini-provider.md, G1-G7).

Fixtures are shaped from Google's documentation of 2026-10-09 (no key was used):
the listing's `models[]`, `generateContent` answers with `thought` parts and
`thoughtSignature`, `usageMetadata`, `{"error": {...}}` bodies, `batchEmbedContents`,
`predictLongRunning` operations. Every assertion on the wire reads what the driver
sent to the mocked Google, so a translation that drifted fails here.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from eugene_plexus_inference_driver._generated.models import ReasoningEffort
from eugene_plexus_inference_driver.app import create_app
from eugene_plexus_inference_driver.config import FIELDS
from eugene_plexus_inference_driver.engines import _gemini_wire as wire
from eugene_plexus_inference_driver.settings import Settings

BASE = "https://generativelanguage.googleapis.com/v1beta"
KEY = "AIzaSy-test-key-not-real-0123456789"
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
WAV = b"RIFF" + (36).to_bytes(4, "little") + b"WAVEfmt " + bytes(40)
PDF = b"%PDF-1.4\n" + bytes(range(64))
PCM = bytes(range(200)) * 3
CHAT = "gemini-3.5-flash"


def _model(name: str, *methods: str, **extra: Any) -> dict[str, Any]:
    return {
        "name": f"models/{name}",
        "baseModelId": name,
        "version": "001",
        "displayName": name,
        "inputTokenLimit": 1048576,
        "outputTokenLimit": 65536,
        "supportedGenerationMethods": list(methods),
        **extra,
    }


_GEN = ("generateContent", "streamGenerateContent", "countTokens")
PAGE_ONE = {
    "models": [
        _model(CHAT, *_GEN, thinking=True),
        _model("gemini-2.5-flash", *_GEN, thinking=True),
        _model("gemini-3.1-flash-lite", *_GEN, thinking=False),
        _model("gemini-3.1-flash-image", "generateContent"),
        _model("gemini-3.8-flash-tts", "generateContent"),
    ],
    "nextPageToken": "page-two",
}
PAGE_TWO = {
    "models": [
        _model("gemini-3.5-transcribe", "generateContent"),
        _model("gemini-embedding-001", "embedContent", "batchEmbedContents"),
        _model("veo-3.1-generate-preview", "predictLongRunning"),
        _model("gemma-3-27b-it", "generateContent"),
        _model("gemini-live-2.5-flash", "bidiGenerateContent"),
        _model("aqa", "generateAnswer"),
        {"displayName": "no name at all"},
    ]
}


def _config(tmp_path: Path, **values: Any) -> Path:
    config = tmp_path / "gemini.yaml"
    config.write_text(json.dumps({"provider": "gemini", "apiKey": KEY, **values}), "utf-8")
    return config


def _wait(client: TestClient) -> dict[str, Any]:
    deadline = time.perf_counter() + 5
    while True:
        info = client.get("/v1/info").json()
        catalogue = info.get("catalogue") or {}
        if catalogue.get("refreshedAt") or catalogue.get("error"):
            return info
        if time.perf_counter() > deadline:
            raise AssertionError(f"catalogue never read: {info}")
        time.sleep(0.02)


def _listing() -> respx.Route:
    def answer(request: httpx.Request) -> httpx.Response:
        page = PAGE_TWO if request.url.params.get("pageToken") == "page-two" else PAGE_ONE
        return httpx.Response(200, json=page)

    return respx.get(f"{BASE}/models").mock(side_effect=answer)


def _start(tmp_path: Path, **values: Any) -> TestClient:
    _listing()
    client = TestClient(create_app(settings=Settings(config_file=_config(tmp_path, **values))))
    client.__enter__()
    _wait(client)
    return client


def _answer(parts: list[dict[str, Any]], finish: str = "STOP", **extra: Any) -> dict[str, Any]:
    return {
        "candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": finish}],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 4,
            "thoughtsTokenCount": 6,
            "cachedContentTokenCount": 3,
            "totalTokenCount": 20,
        },
        **extra,
    }


def _chat(**fields: Any) -> dict[str, Any]:
    body: dict[str, Any] = {"model": CHAT, "messages": [{"role": "user", "content": "Hello"}]}
    return {**body, **fields}


def _sent(route: respx.Route, index: int = -1) -> dict[str, Any]:
    return json.loads(route.calls[index].request.content)


def _error(code: int, status: str, message: str, **extra: Any) -> httpx.Response:
    return httpx.Response(
        code, json={"error": {"code": code, "message": message, "status": status, **extra}}
    )


def _sse(*events: dict[str, Any]) -> httpx.Response:
    body = "".join(f"data: {json.dumps(e)}\r\n\r\n" for e in events)
    return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})


def _frames(text: str) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for block in text.strip().split("\n\n"):
        event, _, data = block.partition("\n")
        out.append((event.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return out


GENERATE = f"{BASE}/models/{CHAT}:generateContent"
STREAM = f"{BASE}/models/{CHAT}:streamGenerateContent"


# --------------------------------------------------------------------------- #
# The listing
# --------------------------------------------------------------------------- #


@respx.mock
def test_the_listing_is_paged_and_mapped_to_surfaces_and_capabilities(tmp_path: Path) -> None:
    route = _listing()
    with TestClient(create_app(settings=Settings(config_file=_config(tmp_path)))) as client:
        info = _wait(client)
    assert info["backend"] == "gemini_api" and info["provider"] == "gemini"
    assert [c.request.url.params.get("pageToken") for c in route.calls] == [None, "page-two"]
    assert route.calls[0].request.url.params["pageSize"] == "1000"
    assert all(c.request.headers["x-goog-api-key"] == KEY for c in route.calls)
    assert all(KEY not in str(c.request.url) for c in route.calls)
    models = {m["id"]: m for m in info["models"]}
    # Live-only, answer-only and nameless entries serve nothing Eugene routes.
    assert set(models) == {
        CHAT,
        "gemini-2.5-flash",
        "gemini-3.1-flash-lite",
        "gemini-3.1-flash-image",
        "gemini-3.8-flash-tts",
        "gemini-3.5-transcribe",
        "gemini-embedding-001",
        "veo-3.1-generate-preview",
        "gemma-3-27b-it",
    }
    chat = models[CHAT]
    assert chat["surfaces"] == ["chat", "transcription"]
    caps = chat["capabilities"]
    assert caps["maxContextTokens"] == 1048576 and caps["streaming"] is True
    assert caps["imageInput"] and caps["audioInput"] and caps["fileInput"] and caps["toolCalling"]
    assert "reasoningEffort" in caps["supportedSettings"]
    # Reasoning is offered only where Google says the model thinks.
    lite = models["gemini-3.1-flash-lite"]["capabilities"]["supportedSettings"]
    assert "reasoningEffort" not in lite and "tools" in lite
    gemma = models["gemma-3-27b-it"]["capabilities"]
    assert gemma["toolCalling"] is False and gemma["imageInput"] is False
    assert models["gemini-3.1-flash-image"]["surfaces"] == ["image"]
    speech = models["gemini-3.8-flash-tts"]
    assert speech["surfaces"] == ["speech"] and len(speech["voices"]) == 30
    assert [f for f in speech["capabilities"]["speechFormats"]] == ["wav", "pcm"]
    assert models["gemini-3.5-transcribe"]["surfaces"] == ["transcription"]
    assert models["gemini-embedding-001"]["surfaces"] == ["embeddings"]
    video = models["veo-3.1-generate-preview"]
    assert video["surfaces"] == ["video"]
    assert video["capabilities"]["video"]["durations"] == [4, 6, 8]


@respx.mock
def test_a_refused_listing_names_the_key_and_keeps_the_driver_up(tmp_path: Path) -> None:
    respx.get(f"{BASE}/models").mock(
        return_value=_error(
            400, "INVALID_ARGUMENT", "API key not valid. Please pass a valid API key."
        )
    )
    with TestClient(create_app(settings=Settings(config_file=_config(tmp_path)))) as client:
        info = _wait(client)
    assert info["catalogue"]["error"] and "API key" in info["catalogue"]["error"]
    assert KEY not in json.dumps(info)


@respx.mock
def test_model_id_narrows_the_account_to_one_model(tmp_path: Path) -> None:
    client = _start(tmp_path, modelId=CHAT)
    with client:
        ids = [m["id"] for m in client.get("/v1/info").json()["models"]]
    assert ids == [CHAT]


def test_the_key_and_stall_fields_are_shown_for_gemini() -> None:
    fields = {f.key: f for f in FIELDS}
    assert fields["apiKey"].showWhen is not None and "gemini" in fields["apiKey"].showWhen.equals
    assert "gemini" in fields["streamStallSeconds"].showWhen.equals  # type: ignore[union-attr]
    assert "gemini" in fields["baseUrl"].showWhen.equals  # type: ignore[union-attr]
    # Gemini has no upstream-name split: a field that reads nothing is not shown.
    assert "gemini" not in fields["upstreamModelId"].showWhen.equals  # type: ignore[union-attr]
    assert "gemini" not in fields["thinkingMode"].showWhen.equals  # type: ignore[union-attr]


def test_a_driver_without_a_key_comes_up_degraded_naming_the_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    config = tmp_path / "gemini.yaml"
    config.write_text(json.dumps({"provider": "gemini"}), "utf-8")
    with TestClient(create_app(settings=Settings(config_file=config))) as client:
        response = client.post("/v1/generate", json=_chat())
    assert response.status_code == 503 and "GEMINI_API_KEY" in response.text


@respx.mock
def test_the_environment_key_is_the_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "env-key-1234567")  # gitleaks:allow (a fake)
    config = tmp_path / "gemini.yaml"
    config.write_text(json.dumps({"provider": "gemini"}), "utf-8")
    route = _listing()
    with TestClient(create_app(settings=Settings(config_file=config))) as client:
        _wait(client)
    assert route.calls[0].request.headers["x-goog-api-key"] == "env-key-1234567"


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #


@respx.mock
def test_chat_is_translated_to_generate_content_and_back(tmp_path: Path) -> None:
    client = _start(tmp_path)
    route = respx.post(GENERATE).mock(
        return_value=httpx.Response(
            200,
            json=_answer(
                [{"text": "weighing it", "thought": True}, {"text": "Hi there."}],
            ),
        )
    )
    with client:
        response = client.post(
            "/v1/generate",
            json=_chat(
                messages=[
                    {"role": "system", "content": "Be brief."},
                    {"role": "system", "content": "Be kind."},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                    {"role": "user", "content": "And?"},
                ],
                temperature=0.4,
                topP=0.9,
                topK=20,
                maxTokens=77,
                stop=["END"],
                seed=0,
                frequencyPenalty=0.1,
                presencePenalty=0.2,
                reasoningEffort="high",
                callerSettings=["temperature", "seed", "reasoningEffort"],
            ),
        )
    assert response.status_code == 200, response.text
    sent = _sent(route)
    assert route.calls[0].request.headers["x-goog-api-key"] == KEY
    assert sent["systemInstruction"] == {"parts": [{"text": "Be brief.\n\nBe kind."}]}
    assert [c["role"] for c in sent["contents"]] == ["user", "model", "user"]
    assert sent["contents"][1]["parts"] == [{"text": "Hi"}]
    assert sent["generationConfig"] == {
        "temperature": 0.4,
        "topP": 0.9,
        "topK": 20,
        "maxOutputTokens": 77,
        "seed": 0,
        "presencePenalty": 0.2,
        "frequencyPenalty": 0.1,
        "stopSequences": ["END"],
        "thinkingConfig": {"includeThoughts": True, "thinkingLevel": "high"},
    }
    body = response.json()
    assert body["content"] == "Hi there." and body["reasoning"] == "weighing it"
    assert body["finishReason"] == "stop" and body["backend"] == "gemini_api"
    assert body["modelId"] == CHAT
    # Google counts thoughts apart; completion includes them, reasoning is carried.
    assert body["usage"] == {
        "promptTokens": 10,
        "completionTokens": 10,
        "totalTokens": 20,
        "cachedPromptTokens": 3,
        "reasoningTokens": 6,
    }


@respx.mock
def test_thinking_is_asked_for_by_budget_on_25_and_level_on_3(tmp_path: Path) -> None:
    client = _start(tmp_path)
    route = respx.post(f"{BASE}/models/gemini-2.5-flash:generateContent").mock(
        return_value=httpx.Response(200, json=_answer([{"text": "ok"}]))
    )
    with client:
        client.post("/v1/generate", json=_chat(model="gemini-2.5-flash", reasoningEffort="low"))
        client.post("/v1/generate", json=_chat(model="gemini-2.5-flash"))
    assert _sent(route, 0)["generationConfig"]["thinkingConfig"] == {
        "includeThoughts": True,
        "thinkingBudget": 1024,
    }
    # No effort asked: thoughts are still returned, the model's own depth.
    assert _sent(route, 1)["generationConfig"]["thinkingConfig"] == {"includeThoughts": True}
    assert wire.thinking_config("gemini-3.5-flash", ReasoningEffort.none) == {
        "includeThoughts": False,
        "thinkingLevel": "minimal",
    }
    assert wire.thinking_config("gemini-2.5-flash", ReasoningEffort.none)["thinkingBudget"] == 0
    assert wire.thinking_config("gemini-2.5-pro", ReasoningEffort.max)["thinkingBudget"] == 32768


@respx.mock
def test_structured_output_becomes_a_response_schema(tmp_path: Path) -> None:
    client = _start(tmp_path)
    route = respx.post(GENERATE).mock(
        return_value=httpx.Response(200, json=_answer([{"text": '{"a": 1}'}]))
    )
    schema = {
        "type": "object",
        "properties": {"a": {"type": "integer"}},
        "additionalProperties": False,
    }
    with client:
        client.post(
            "/v1/generate",
            json=_chat(
                responseFormat={
                    "type": "json_schema",
                    "json_schema": {"name": "r", "schema": schema, "strict": True},
                }
            ),
        )
        client.post("/v1/generate", json=_chat(responseFormat={"type": "json_object"}))
    first = _sent(route, 0)["generationConfig"]
    assert first["responseMimeType"] == "application/json"
    assert first["responseJsonSchema"] == schema
    second = _sent(route, 1)["generationConfig"]
    assert second["responseMimeType"] == "application/json" and "responseJsonSchema" not in second


@respx.mock
def test_attachments_go_inline(tmp_path: Path) -> None:
    client = _start(tmp_path)
    route = respx.post(GENERATE).mock(
        return_value=httpx.Response(200, json=_answer([{"text": "seen"}]))
    )
    b64 = lambda raw: base64.b64encode(raw).decode()  # noqa: E731
    with client:
        response = client.post(
            "/v1/generate",
            json=_chat(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "what is this"},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64(PNG)}"},
                            },
                            {
                                "type": "input_audio",
                                "input_audio": {"data": b64(WAV), "format": "wav"},
                            },
                            {
                                "type": "file",
                                "file": {
                                    "filename": "a.pdf",
                                    "file_data": f"data:application/pdf;base64,{b64(PDF)}",
                                },
                            },
                        ],
                    }
                ]
            ),
        )
    assert response.status_code == 200, response.text
    assert _sent(route)["contents"][0]["parts"] == [
        {"text": "what is this"},
        {"inlineData": {"mimeType": "image/png", "data": b64(PNG)}},
        {"inlineData": {"mimeType": "audio/wav", "data": b64(WAV)}},
        {"inlineData": {"mimeType": "application/pdf", "data": b64(PDF)}},
    ]


@respx.mock
def test_a_model_that_cannot_see_is_refused_before_anything_is_sent(tmp_path: Path) -> None:
    client = _start(tmp_path)
    route = respx.post(f"{BASE}/models/gemma-3-27b-it:generateContent").mock(
        return_value=httpx.Response(200, json=_answer([{"text": "x"}]))
    )
    with client:
        response = client.post(
            "/v1/generate",
            json=_chat(
                model="gemma-3-27b-it",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64.b64encode(PNG).decode()}"
                                },
                            }
                        ],
                    }
                ],
            ),
        )
        # Gemma takes no system instruction: it goes in front of the first user turn.
        client.post(
            "/v1/generate",
            json=_chat(
                model="gemma-3-27b-it",
                messages=[
                    {"role": "system", "content": "Be brief."},
                    {"role": "user", "content": "Hi"},
                ],
            ),
        )
    assert response.status_code == 400 and "Image input" in response.text
    assert route.call_count == 1
    sent = _sent(route)
    assert "systemInstruction" not in sent
    assert sent["contents"][0]["parts"][0]["text"].startswith("Be brief.")


@pytest.mark.parametrize(
    ("fields", "named"),
    [
        ({"logprobs": True}, "logprobs"),
        ({"minP": 0.1}, "min_p"),
        ({"logitBias": {"5": 1}}, "logit_bias"),
        ({"verbosity": "low"}, "verbosity"),
        ({"webSearchOptions": {}}, "web_search_options"),
        ({"parallelToolCalls": False}, "parallel_tool_calls"),
    ],
)
@respx.mock
def test_a_setting_gemini_cannot_honour_is_refused_not_dropped(
    tmp_path: Path, fields: dict[str, Any], named: str
) -> None:
    client = _start(tmp_path)
    route = respx.post(GENERATE).mock(
        return_value=httpx.Response(200, json=_answer([{"text": "ok"}]))
    )
    key = next(iter(fields))
    with client:
        response = client.post(
            "/v1/generate",
            json=_chat(
                **fields,
                tools=[{"type": "function", "function": {"name": "f"}}],
                callerSettings=[key],
            ),
        )
        # The same value inherited rather than asked for is omitted, not refused.
        inherited = client.post("/v1/generate", json=_chat(**fields))
    assert response.status_code == 400 and named in response.text
    assert inherited.status_code == 200 and route.call_count == 1
    sent = json.dumps(_sent(route))
    for word in ("minP", "min_p", "logprobs", "logitBias", "verbosity", "parallel"):
        assert word not in sent


@respx.mock
def test_reasoning_effort_is_refused_on_a_model_that_does_not_think(tmp_path: Path) -> None:
    client = _start(tmp_path)
    route = respx.post(f"{BASE}/models/gemini-3.1-flash-lite:generateContent").mock(
        return_value=httpx.Response(200, json=_answer([{"text": "x"}]))
    )
    with client:
        response = client.post(
            "/v1/generate",
            json=_chat(
                model="gemini-3.1-flash-lite",
                reasoningEffort="high",
                callerSettings=["reasoningEffort"],
            ),
        )
    assert response.status_code == 400 and "reasoning_effort" in response.text
    assert not route.called


@respx.mock
def test_a_model_that_does_not_chat_and_one_not_served_are_refused(tmp_path: Path) -> None:
    client = _start(tmp_path)
    with client:
        speech = client.post("/v1/generate", json=_chat(model="gemini-3.8-flash-tts"))
        missing = client.post("/v1/generate", json=_chat(model="gemini-9"))
    assert speech.status_code == 400 and "does not answer chat" in speech.text
    assert missing.status_code == 404


# --------------------------------------------------------------------------- #
# Tool calls and thought signatures (G2, G7)
# --------------------------------------------------------------------------- #

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "additionalProperties": False,
            },
        },
    }
]


def _call_turn(call_id: str, signature: str | None = "sig-abc") -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": "Weather in Oslo?"},
        {
            "role": "assistant",
            "content": None,
            "toolCalls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Oslo"}'},
                }
            ],
        },
        {"role": "tool", "toolCallId": call_id, "content": '{"temp": 4}'},
    ]


@respx.mock
def test_a_tool_call_keeps_its_signature_and_gets_it_back_on_the_next_turn(
    tmp_path: Path,
) -> None:
    client = _start(tmp_path)
    call_part = {
        "functionCall": {"name": "get_weather", "args": {"city": "Oslo"}},
        "thoughtSignature": "sig-abc",
    }
    route = respx.post(GENERATE).mock(
        side_effect=[
            httpx.Response(200, json=_answer([call_part])),
            httpx.Response(200, json=_answer([{"text": "4 degrees."}])),
        ]
    )
    with client:
        first = client.post(
            "/v1/generate",
            json=_chat(tools=TOOLS, toolChoice="required", callerSettings=["tools", "toolChoice"]),
        )
        call = first.json()["toolCalls"][0]
        second = client.post(
            "/v1/generate", json=_chat(tools=TOOLS, messages=_call_turn(call["id"]))
        )
    sent = _sent(route, 0)
    assert sent["tools"] == [
        {
            "functionDeclarations": [
                {
                    "name": "get_weather",
                    "description": "Weather",
                    "parametersJsonSchema": TOOLS[0]["function"]["parameters"],
                }
            ]
        }
    ]
    assert sent["toolConfig"] == {"functionCallingConfig": {"mode": "ANY"}}
    # Google gave no id: the driver made a stable one, and Google is never sent it.
    assert first.json()["finishReason"] == "tool_calls"
    assert call["id"].startswith("call_") and json.loads(call["function"]["arguments"]) == {
        "city": "Oslo"
    }
    assert second.status_code == 200, second.text
    turns = _sent(route, 1)["contents"]
    assert [t["role"] for t in turns] == ["user", "model", "user"]
    assert turns[1]["parts"] == [
        {
            "functionCall": {"name": "get_weather", "args": {"city": "Oslo"}},
            "thoughtSignature": "sig-abc",
        }
    ]
    assert turns[2]["parts"] == [
        {"functionResponse": {"name": "get_weather", "response": {"temp": 4}}}
    ]


@respx.mock
def test_an_id_google_gave_is_sent_back_and_parallel_results_share_a_turn(
    tmp_path: Path,
) -> None:
    client = _start(tmp_path)
    parts = [
        {
            "functionCall": {"id": "g1", "name": "get_weather", "args": {"city": "Oslo"}},
            "thoughtSignature": "s1",
        },
        {"functionCall": {"id": "g2", "name": "get_weather", "args": {"city": "Rome"}}},
    ]
    route = respx.post(GENERATE).mock(
        side_effect=[
            httpx.Response(200, json=_answer(parts)),
            httpx.Response(200, json=_answer([{"text": "done"}])),
        ]
    )
    with client:
        made = client.post("/v1/generate", json=_chat(tools=TOOLS)).json()["toolCalls"]
        assert [c["id"] for c in made] == ["g1", "g2"]
        client.post(
            "/v1/generate",
            json=_chat(
                tools=TOOLS,
                messages=[
                    {"role": "user", "content": "Weather?"},
                    {"role": "assistant", "toolCalls": made},
                    {"role": "tool", "toolCallId": "g1", "content": "cold"},
                    {"role": "tool", "toolCallId": "g2", "content": "warm"},
                ],
            ),
        )
    turns = _sent(route, 1)["contents"]
    assert turns[1]["parts"][0]["functionCall"]["id"] == "g1"
    assert turns[1]["parts"][0]["thoughtSignature"] == "s1"
    assert "thoughtSignature" not in turns[1]["parts"][1]
    assert turns[2]["parts"] == [
        {"functionResponse": {"id": "g1", "name": "get_weather", "response": {"result": "cold"}}},
        {"functionResponse": {"id": "g2", "name": "get_weather", "response": {"result": "warm"}}},
    ]


@respx.mock
def test_a_signature_lost_to_a_restart_is_named_as_the_cause(tmp_path: Path) -> None:
    client = _start(tmp_path)
    respx.post(GENERATE).mock(
        return_value=_error(
            400,
            "INVALID_ARGUMENT",
            "Function call is missing a thought_signature in functionCall parts.",
        )
    )
    with client:
        response = client.post(
            "/v1/generate", json=_chat(tools=TOOLS, messages=_call_turn("call_unknown"))
        )
    assert response.status_code == 400
    detail = response.json()["detail"]["detail"]
    assert "before the driver restarted" in detail and "thought signature" in detail


@respx.mock
def test_a_tool_result_with_no_call_before_it_is_refused(tmp_path: Path) -> None:
    client = _start(tmp_path)
    route = respx.post(GENERATE).mock(return_value=httpx.Response(200, json=_answer([])))
    with client:
        response = client.post(
            "/v1/generate",
            json=_chat(
                tools=TOOLS,
                messages=[
                    {"role": "user", "content": "x"},
                    {"role": "tool", "toolCallId": "nobody", "content": "r"},
                ],
            ),
        )
    assert response.status_code == 400 and "matches no earlier tool call" in response.text
    assert not route.called


def test_the_signature_cache_is_bounded_and_expires() -> None:
    cache = wire.SignatureCache(entries=3, seconds=60)
    for n in range(5):
        cache.remember(f"c{n}", wire.RememberedCall("f", f"s{n}", False))
    assert len(cache) == 3 and cache.recall("c0") is None and cache.recall("c1") is None
    found = cache.recall("c4")
    assert found is not None and found.signature == "s4"
    short = wire.SignatureCache(entries=3, seconds=0.0)
    short.remember("x", wire.RememberedCall("f", "s", False))
    assert short.recall("x") is None
    assert wire.SIGNATURE_ENTRIES == 4096 and wire.SIGNATURE_SECONDS == 86400


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #


@respx.mock
def test_a_stream_carries_text_reasoning_calls_and_usage(tmp_path: Path) -> None:
    client = _start(tmp_path)
    route = respx.post(STREAM).mock(
        return_value=_sse(
            {"candidates": [{"content": {"parts": [{"text": "thinking", "thought": True}]}}]},
            {"candidates": [{"content": {"parts": [{"text": "Hel"}]}}]},
            {"candidates": [{"content": {"parts": [{"text": "lo"}]}}]},
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": "get_weather",
                                        "args": {"city": "Oslo"},
                                    },
                                    "thoughtSignature": "sig-stream",
                                }
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 5,
                    "candidatesTokenCount": 2,
                    "totalTokenCount": 7,
                },
            },
        )
    )
    with client:
        response = client.post("/v1/generate/stream", json=_chat(tools=TOOLS, reportProgress=True))
        frames = _frames(response.text)
    assert route.calls[0].request.url.params["alt"] == "sse"
    kinds = [(event, next(iter(data))) for event, data in frames]
    assert kinds == [
        ("progress", "stage"),
        ("token", "reasoning"),
        ("token", "text"),
        ("token", "text"),
        ("token", "toolCalls"),
        ("done", "content"),
    ]
    done = frames[-1][1]
    assert done["content"] == "Hello" and done["reasoning"] == "thinking"
    assert done["finishReason"] == "tool_calls"
    assert done["usage"]["promptTokens"] == 5 and done["usage"]["completionTokens"] == 2
    call = done["toolCalls"][0]
    assert frames[4][1]["toolCalls"][0]["id"] == call["id"]
    # The streamed call's signature is remembered like a whole one's.
    assert client.app.state.adapter._signatures.recall(call["id"]).signature == "sig-stream"  # type: ignore[attr-defined]


@respx.mock
def test_a_stream_that_ends_without_a_finish_reason_is_an_error(tmp_path: Path) -> None:
    client = _start(tmp_path)
    respx.post(STREAM).mock(
        return_value=_sse({"candidates": [{"content": {"parts": [{"text": "half"}]}}]})
    )
    with client:
        response = client.post("/v1/generate/stream", json=_chat())
        frames = _frames(response.text)
    assert frames[-1][0] == "error" and "without a finishReason" in frames[-1][1]["detail"]


@respx.mock
def test_a_stream_refused_before_it_opens_is_a_status_code(tmp_path: Path) -> None:
    client = _start(tmp_path)
    respx.post(STREAM).mock(
        return_value=_error(403, "PERMISSION_DENIED", "The caller does not have permission")
    )
    with client:
        response = client.post("/v1/generate/stream", json=_chat())
    assert response.status_code == 502
    assert response.json()["detail"]["type"].endswith("#backend-credential-refused")


@respx.mock
def test_a_blocked_prompt_in_a_stream_is_named(tmp_path: Path) -> None:
    client = _start(tmp_path)
    respx.post(STREAM).mock(
        return_value=_sse({"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}})
    )
    with client:
        response = client.post("/v1/generate/stream", json=_chat())
    assert response.status_code == 400 and "PROHIBITED_CONTENT" in response.text


# --------------------------------------------------------------------------- #
# Finish reasons and errors
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("google", "ours"),
    [
        ("STOP", "stop"),
        ("MAX_TOKENS", "length"),
        ("SAFETY", "content_filter"),
        ("RECITATION", "content_filter"),
        ("BLOCKLIST", "content_filter"),
        ("PROHIBITED_CONTENT", "content_filter"),
        ("SPII", "content_filter"),
    ],
)
@respx.mock
def test_finish_reasons_map(tmp_path: Path, google: str, ours: str) -> None:
    client = _start(tmp_path)
    respx.post(GENERATE).mock(
        return_value=httpx.Response(
            200,
            json=_answer(
                [{"text": "partial"}] if ours != "content_filter" else [],
                google,
            ),
        )
    )
    with client:
        body = client.post("/v1/generate", json=_chat()).json()
    assert body["finishReason"] == ours
    if ours == "content_filter":
        # Google's own reason is in the answer, which is otherwise empty.
        assert f"finishReason={google}" in body["content"]


@respx.mock
def test_a_blocked_prompt_names_google_s_reason(tmp_path: Path) -> None:
    client = _start(tmp_path)
    respx.post(GENERATE).mock(
        return_value=httpx.Response(
            200, json={"promptFeedback": {"blockReason": "SAFETY", "blockReasonMessage": "no"}}
        )
    )
    with client:
        response = client.post("/v1/generate", json=_chat())
    assert response.status_code == 400
    assert "blockReason=SAFETY" in response.json()["detail"]["detail"]


@respx.mock
def test_a_malformed_function_call_is_an_error_not_an_answer(tmp_path: Path) -> None:
    client = _start(tmp_path)
    respx.post(GENERATE).mock(
        return_value=httpx.Response(200, json=_answer([], "MALFORMED_FUNCTION_CALL"))
    )
    with client:
        response = client.post("/v1/generate", json=_chat(tools=TOOLS))
    assert response.status_code == 502 and "MALFORMED_FUNCTION_CALL" in response.text


@pytest.mark.parametrize(
    ("answer", "code", "kind", "words"),
    [
        (
            _error(403, "PERMISSION_DENIED", "Method doesn't allow unregistered callers"),
            502,
            "backend-credential-refused",
            "API key",
        ),
        (
            _error(401, "UNAUTHENTICATED", "Request had invalid authentication credentials"),
            502,
            "backend-credential-refused",
            "API key",
        ),
        (
            _error(
                400,
                "INVALID_ARGUMENT",
                "API key not valid. Please pass a valid API key.",
                details=[
                    {
                        "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                        "reason": "API_KEY_INVALID",
                    }
                ],
            ),
            502,
            "backend-credential-refused",
            "API key",
        ),
        (
            _error(400, "INVALID_ARGUMENT", "Request contains an invalid argument."),
            400,
            "backend-rejected-request",
            "invalid argument",
        ),
        (
            # Refused before any work, for every request from here: the
            # account's, so another backend may take it (not "outcome unknown").
            _error(400, "FAILED_PRECONDITION", "User location is not supported for the API use."),
            502,
            "backend-credential-refused",
            "region",
        ),
    ],
)
@respx.mock
def test_errors_keep_their_cause(
    tmp_path: Path, answer: httpx.Response, code: int, kind: str, words: str
) -> None:
    client = _start(tmp_path)
    respx.post(GENERATE).mock(return_value=answer)
    with client:
        response = client.post("/v1/generate", json=_chat())
    assert response.status_code == code, response.text
    detail = response.json()["detail"]
    assert detail["type"].endswith(f"#{kind}") and words in detail["detail"]


@respx.mock
def test_a_rate_limit_carries_its_wait(tmp_path: Path) -> None:
    client = _start(tmp_path)
    route = respx.post(GENERATE).mock(
        side_effect=[
            httpx.Response(
                429,
                headers={"Retry-After": "7"},
                json={"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "quota"}},
            ),
            _error(
                429,
                "RESOURCE_EXHAUSTED",
                "quota",
                details=[
                    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "34s"}
                ],
            ),
        ]
    )
    with client:
        header = client.post("/v1/generate", json=_chat())
        body = client.post("/v1/generate", json=_chat())
    assert route.call_count == 2
    assert header.headers["retry-after"] == "7"
    assert body.headers["retry-after"] == "34"
    assert "RESOURCE_EXHAUSTED" in body.json()["detail"]["detail"]
    assert body.json()["detail"]["retryDisposition"] == "safe"


@respx.mock
def test_the_key_never_reaches_a_log_a_response_or_an_exception(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    client = _start(tmp_path)
    # A provider that echoes the credential back in its words.
    respx.post(GENERATE).mock(
        return_value=_error(400, "INVALID_ARGUMENT", f"bad request for key={KEY} and {KEY}")
    )
    respx.post(STREAM).mock(side_effect=httpx.ConnectError(f"cannot connect with {KEY}"))
    with client:
        plain = client.post("/v1/generate", json=_chat())
        streamed = client.post("/v1/generate/stream", json=_chat())
    assert plain.status_code == 400 and streamed.status_code == 502
    assert KEY not in plain.text + streamed.text
    assert KEY not in caplog.text
    assert "<redacted>" in plain.text


@respx.mock
def test_one_http_client_serves_every_call(tmp_path: Path) -> None:
    client = _start(tmp_path)
    respx.post(GENERATE).mock(return_value=httpx.Response(200, json=_answer([{"text": "x"}])))
    with client:
        engine = client.app.state.adapter  # type: ignore[attr-defined]
        first = engine._client()
        client.post("/v1/generate", json=_chat())
        client.post("/v1/generate", json=_chat())
        assert engine._client() is first


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #


@respx.mock
def test_embeddings_are_batched_in_order(tmp_path: Path) -> None:
    client = _start(tmp_path)

    def answer(request: httpx.Request) -> httpx.Response:
        asked = json.loads(request.content)["requests"]
        return httpx.Response(
            200,
            json={
                "embeddings": [
                    {"values": [float(r["content"]["parts"][0]["text"]), 0.5]} for r in asked
                ],
                "usageMetadata": {"promptTokenCount": len(asked)},
            },
        )

    route = respx.post(f"{BASE}/models/gemini-embedding-001:batchEmbedContents").mock(
        side_effect=answer
    )
    with client:
        response = client.post(
            "/v1/embed",
            json={"model": "gemini-embedding-001", "input": [str(n) for n in range(205)]},
        )
        refused = client.post("/v1/embed", json={"model": CHAT, "input": ["x"]})
    assert response.status_code == 200, response.text
    body = response.json()
    assert [v[0] for v in body["embeddings"]] == [float(n) for n in range(205)]
    assert route.call_count == 3 and body["usage"]["promptTokens"] == 205
    assert _sent(route, 0)["requests"][0] == {
        "model": "models/gemini-embedding-001",
        "content": {"parts": [{"text": "0"}]},
    }
    assert refused.status_code == 400


@respx.mock
def test_embeddings_with_a_wrong_count_are_refused(tmp_path: Path) -> None:
    client = _start(tmp_path)
    respx.post(f"{BASE}/models/gemini-embedding-001:batchEmbedContents").mock(
        return_value=httpx.Response(200, json={"embeddings": [{"values": [1.0]}]})
    )
    with client:
        response = client.post(
            "/v1/embed", json={"model": "gemini-embedding-001", "input": ["a", "b"]}
        )
    assert response.status_code == 502 and "1 embeddings for 2 inputs" in response.text


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #

IMAGE = "gemini-3.1-flash-image"


@respx.mock
def test_an_image_is_made_by_generate_content_with_an_image_answer(tmp_path: Path) -> None:
    client = _start(tmp_path)
    route = respx.post(f"{BASE}/models/{IMAGE}:generateContent").mock(
        return_value=httpx.Response(
            200,
            json=_answer(
                [
                    {"text": "Here."},
                    {
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": base64.b64encode(PNG).decode(),
                        }
                    },
                ]
            ),
        )
    )
    with client:
        response = client.post(
            "/v1/image",
            json={
                "model": IMAGE,
                "prompt": "a cat",
                "references": [{"data": base64.b64encode(PNG).decode(), "mediaType": "image/png"}],
            },
        )
        refused = client.post(
            "/v1/image", json={"model": IMAGE, "prompt": "a cat", "size": "1024x1024"}
        )
        many = client.post("/v1/image", json={"model": IMAGE, "prompt": "a cat", "n": 2})
        chat = client.post("/v1/image", json={"model": CHAT, "prompt": "a cat"})
    assert response.status_code == 200, response.text
    sent = _sent(route)
    assert sent["generationConfig"] == {"responseModalities": ["TEXT", "IMAGE"]}
    assert sent["contents"][0]["parts"][0] == {"text": "a cat"}
    assert sent["contents"][0]["parts"][1]["inlineData"]["mimeType"] == "image/png"
    body = response.json()
    assert body["images"][0]["mediaType"] == "image/png"
    assert base64.b64decode(body["images"][0]["data"]) == PNG
    assert body["usage"]["inputTokens"] == 10
    assert refused.status_code == 400 and "size" in refused.text
    assert many.status_code == 400 and "n" in many.text
    assert chat.status_code == 400 and route.call_count == 1


@respx.mock
def test_an_image_model_that_answers_only_words_says_so(tmp_path: Path) -> None:
    client = _start(tmp_path)
    respx.post(f"{BASE}/models/{IMAGE}:generateContent").mock(
        return_value=httpx.Response(
            200, json=_answer([{"text": "I cannot draw that."}], "IMAGE_SAFETY")
        )
    )
    with client:
        response = client.post("/v1/image", json={"model": IMAGE, "prompt": "x"})
    assert response.status_code == 400
    assert "IMAGE_SAFETY" in response.text and "cannot draw" in response.text


# --------------------------------------------------------------------------- #
# Video (Veo)
# --------------------------------------------------------------------------- #

VEO = "veo-3.1-generate-preview"
OPERATION = f"models/{VEO}/operations/abc123"
VIDEO_URI = f"{BASE}/files/xyz:download?alt=media"
MP4 = b"\x00\x00\x00 ftypisom" + bytes(range(256)) * 8


@respx.mock
def test_a_video_job_runs_through_its_states(tmp_path: Path) -> None:
    client = _start(tmp_path)
    submit = respx.post(f"{BASE}/models/{VEO}:predictLongRunning").mock(
        return_value=httpx.Response(200, json={"name": OPERATION, "done": False})
    )
    poll = respx.get(f"{BASE}/{OPERATION}").mock(
        side_effect=[
            httpx.Response(200, json={"name": OPERATION, "done": False}),
            httpx.Response(
                200,
                json={
                    "name": OPERATION,
                    "done": True,
                    "response": {
                        "generateVideoResponse": {
                            "generatedSamples": [{"video": {"uri": VIDEO_URI}}]
                        }
                    },
                },
            ),
            # content: polled again for the address
            httpx.Response(
                200,
                json={
                    "done": True,
                    "response": {
                        "generateVideoResponse": {
                            "generatedSamples": [{"video": {"uri": VIDEO_URI}}]
                        }
                    },
                },
            ),
        ]
    )
    respx.get(f"{BASE}/not-a-job").mock(
        return_value=_error(404, "NOT_FOUND", "Operation not-a-job not found")
    )
    download = respx.get(VIDEO_URI).mock(return_value=httpx.Response(200, content=MP4))
    with client:
        queued = client.post(
            "/v1/video",
            json={
                "model": VEO,
                "prompt": "a wave",
                "seconds": 8,
                "size": "1920x1080",
                "firstFrame": {"data": base64.b64encode(PNG).decode(), "mediaType": "image/png"},
            },
        )
        assert queued.status_code == 200, queued.text
        job_id = queued.json()["jobId"]
        assert "/" not in job_id and queued.json()["status"] == "queued"
        running = client.get(f"/v1/video/{job_id}").json()
        done = client.get(f"/v1/video/{job_id}").json()
        content = client.get(f"/v1/video/{job_id}/content")
        unknown = client.get("/v1/video/bm90LWEtam9i")  # an operation Google does not know
        garbage = client.get("/v1/video/!!!")  # not even an operation name
        sneaky = client.get(
            "/v1/video/" + base64.urlsafe_b64encode(b"models/../x").decode().rstrip("=")
        )
    sent = _sent(submit)
    assert sent["instances"][0]["prompt"] == "a wave"
    assert sent["instances"][0]["image"]["inlineData"]["mimeType"] == "image/png"
    assert sent["parameters"] == {
        "aspectRatio": "16:9",
        "resolution": "1080p",
        "durationSeconds": 8,
    }
    assert running["status"] == "in_progress" and done["status"] == "completed"
    assert content.status_code == 200 and content.content == MP4
    assert content.headers["content-type"] == "video/mp4"
    assert download.calls[0].request.headers["x-goog-api-key"] == KEY
    assert poll.call_count == 3
    assert unknown.status_code == garbage.status_code == sneaky.status_code == 404


@respx.mock
def test_a_failed_or_filtered_video_job_says_why(tmp_path: Path) -> None:
    client = _start(tmp_path)
    job = base64.urlsafe_b64encode(OPERATION.encode()).decode().rstrip("=")
    respx.get(f"{BASE}/{OPERATION}").mock(
        side_effect=[
            httpx.Response(200, json={"done": True, "error": {"code": 13, "message": "internal"}}),
            httpx.Response(
                200,
                json={
                    "done": True,
                    "response": {
                        "generateVideoResponse": {
                            "raiMediaFilteredCount": 1,
                            "raiMediaFilteredReasons": ["celebrity likeness"],
                        }
                    },
                },
            ),
            httpx.Response(200, json={"done": False}),
        ]
    )
    with client:
        failed = client.get(f"/v1/video/{job}").json()
        filtered = client.get(f"/v1/video/{job}").json()
        early = client.get(f"/v1/video/{job}/content")
    assert failed["status"] == "failed" and "internal" in failed["error"]
    assert filtered["status"] == "failed" and "celebrity likeness" in filtered["error"]
    assert early.status_code == 404


@pytest.mark.parametrize(
    ("fields", "named"),
    [
        ({"seconds": 5}, "seconds"),
        ({"size": "640x480"}, "size"),
        ({"seconds": 4, "size": "1920x1080"}, "8 seconds"),
    ],
)
@respx.mock
def test_a_video_setting_veo_cannot_make_is_refused(
    tmp_path: Path, fields: dict[str, Any], named: str
) -> None:
    client = _start(tmp_path)
    submit = respx.post(f"{BASE}/models/{VEO}:predictLongRunning").mock(
        return_value=httpx.Response(200, json={"name": OPERATION})
    )
    with client:
        response = client.post("/v1/video", json={"model": VEO, "prompt": "x", **fields})
    assert response.status_code == 400 and named in response.text
    assert not submit.called


@respx.mock
def test_the_key_is_not_sent_to_an_address_outside_google(tmp_path: Path) -> None:
    client = _start(tmp_path)
    job = base64.urlsafe_b64encode(OPERATION.encode()).decode().rstrip("=")
    respx.get(f"{BASE}/{OPERATION}").mock(
        return_value=httpx.Response(
            200,
            json={
                "done": True,
                "response": {
                    "generateVideoResponse": {
                        "generatedSamples": [{"video": {"uri": "https://evil.example/v.mp4"}}]
                    }
                },
            },
        )
    )
    elsewhere = respx.get("https://evil.example/v.mp4").mock(
        return_value=httpx.Response(200, content=MP4)
    )
    with client:
        response = client.get(f"/v1/video/{job}/content")
    assert response.status_code == 502 and not elsewhere.called


# --------------------------------------------------------------------------- #
# Speech (G5)
# --------------------------------------------------------------------------- #

TTS = "gemini-3.8-flash-tts"


def _speech_answer() -> httpx.Response:
    return httpx.Response(
        200,
        json=_answer(
            [
                {
                    "inlineData": {
                        "mimeType": "audio/L16;codec=pcm;rate=24000",
                        "data": base64.b64encode(PCM).decode(),
                    }
                }
            ]
        ),
    )


@respx.mock
def test_speech_is_a_complete_wav_made_from_geminis_pcm(tmp_path: Path) -> None:
    client = _start(tmp_path)
    route = respx.post(f"{BASE}/models/{TTS}:generateContent").mock(return_value=_speech_answer())
    with client:
        wav = client.post(
            "/v1/speak", json={"model": TTS, "input": "Hello.", "voice": "Kore", "format": "wav"}
        )
        default = client.post("/v1/speak", json={"model": TTS, "input": "Hello.", "voice": "Puck"})
        pcm = client.post(
            "/v1/speak", json={"model": TTS, "input": "Hello.", "voice": "Kore", "format": "pcm"}
        )
    assert wav.status_code == 200, wav.text
    assert wav.headers["content-type"] == "audio/wav"
    raw = wav.content
    assert raw[:4] == b"RIFF" and raw[8:12] == b"WAVE" and raw[44:] == PCM
    assert int.from_bytes(raw[40:44], "little") == len(PCM)  # a real length, not a stream's
    assert int.from_bytes(raw[24:28], "little") == 24000
    assert _sent(route, 0) == {
        "contents": [{"role": "user", "parts": [{"text": "Hello."}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}},
        },
    }
    assert default.headers["content-type"] == "audio/wav" and default.content[:4] == b"RIFF"
    assert pcm.headers["content-type"] == "audio/pcm" and pcm.content == PCM


@pytest.mark.parametrize(
    ("fields", "named"),
    [
        ({"format": "mp3"}, "wav, pcm"),
        ({"format": "opus"}, "wav, pcm"),
        ({"speed": 1.5}, "speed"),
        ({"instructions": "cheerful"}, "instructions"),
    ],
)
@respx.mock
def test_speech_gemini_cannot_make_is_refused(
    tmp_path: Path, fields: dict[str, Any], named: str
) -> None:
    client = _start(tmp_path)
    route = respx.post(f"{BASE}/models/{TTS}:generateContent").mock(return_value=_speech_answer())
    with client:
        response = client.post(
            "/v1/speak", json={"model": TTS, "input": "Hi.", "voice": "Kore", **fields}
        )
    assert response.status_code == 400 and named in response.text
    assert not route.called


# --------------------------------------------------------------------------- #
# Transcription
# --------------------------------------------------------------------------- #


@respx.mock
def test_transcription_sends_the_audio_inline_with_an_instruction(tmp_path: Path) -> None:
    client = _start(tmp_path)
    # The transcription model's own part, as Google sent it live (2026-10-09).
    route = respx.post(f"{BASE}/models/gemini-3.5-transcribe:generateContent").mock(
        return_value=httpx.Response(
            200, json=_answer([{"audioTranscription": {"text": " The quick brown fox. "}}])
        )
    )
    chat_route = respx.post(GENERATE).mock(
        return_value=httpx.Response(200, json=_answer([{"text": "hello"}]))
    )
    audio = {"data": base64.b64encode(WAV).decode(), "filename": "clip.wav"}
    with client:
        response = client.post(
            "/v1/transcribe",
            json={
                "model": "gemini-3.5-transcribe",
                "audio": audio,
                "language": "en",
                "prompt": "Kubernetes",
                "temperature": 0.0,
            },
        )
        via_chat = client.post("/v1/transcribe", json={"model": CHAT, "audio": audio})
        stamps = client.post(
            "/v1/transcribe",
            json={
                "model": CHAT,
                "audio": audio,
                "verbose": True,
                "timestampGranularities": ["word"],
            },
        )
        odd = client.post(
            "/v1/transcribe",
            json={
                "model": CHAT,
                "audio": {"data": base64.b64encode(WAV).decode(), "filename": "clip.xyz"},
            },
        )
    assert response.status_code == 200, response.text
    assert response.json()["text"] == "The quick brown fox."
    assert response.json()["usage"]["inputTokens"] == 10
    parts = _sent(route)["contents"][0]["parts"]
    assert parts[0] == {"inlineData": {"mimeType": "audio/wav", "data": audio["data"]}}
    assert "Transcribe" in parts[1]["text"] and "en" in parts[1]["text"]
    assert "Kubernetes" in parts[1]["text"]
    assert _sent(route)["generationConfig"] == {"temperature": 0.0}
    assert via_chat.status_code == 200 and via_chat.json()["text"] == "hello"
    assert chat_route.call_count == 1
    assert stamps.status_code == 400 and "timestamp" in stamps.text
    assert odd.status_code == 400 and "wav, mp3" in odd.text


@respx.mock
def test_a_redirect_to_storage_is_followed_without_the_key(tmp_path: Path) -> None:
    client = _start(tmp_path)
    job = base64.urlsafe_b64encode(OPERATION.encode()).decode().rstrip("=")
    respx.get(f"{BASE}/{OPERATION}").mock(
        return_value=httpx.Response(
            200,
            json={
                "done": True,
                "response": {
                    "generateVideoResponse": {"generatedSamples": [{"video": {"uri": VIDEO_URI}}]}
                },
            },
        )
    )
    respx.get(VIDEO_URI).mock(
        return_value=httpx.Response(302, headers={"location": "https://storage.example/v.mp4"})
    )
    stored = respx.get("https://storage.example/v.mp4").mock(
        return_value=httpx.Response(200, content=MP4)
    )
    with client:
        response = client.get(f"/v1/video/{job}/content")
    assert response.status_code == 200 and response.content == MP4
    assert "x-goog-api-key" not in stored.calls[0].request.headers


@pytest.mark.parametrize(
    "elsewhere",
    [
        "https://storage.example/files/xyz:download?alt=media",
        "http://generativelanguage.googleapis.com/v1beta/files/xyz:download",
        "https://generativelanguage.googleapis.com:8443/v1beta/files/xyz:download",
    ],
)
@respx.mock
def test_a_video_named_outside_googles_api_origin_is_not_fetched_with_the_key(
    tmp_path: Path, elsewhere: str
) -> None:
    """The key goes only where `baseUrl` sends it: same scheme, host and port."""
    client = _start(tmp_path)
    job = base64.urlsafe_b64encode(OPERATION.encode()).decode().rstrip("=")
    respx.get(f"{BASE}/{OPERATION}").mock(
        return_value=httpx.Response(
            200,
            json={
                "done": True,
                "response": {
                    "generateVideoResponse": {"generatedSamples": [{"video": {"uri": elsewhere}}]}
                },
            },
        )
    )
    fetched = respx.get(elsewhere).mock(return_value=httpx.Response(200, content=MP4))
    with client:
        response = client.get(f"/v1/video/{job}/content")
    assert response.status_code >= 400 and "outside its own API host" in response.text
    assert not fetched.called
