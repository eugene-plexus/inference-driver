"""Raw completion on /v1/generate(/stream) (P6).

Measured on `llama-server` b11235 (`provider-accounts-measurement.md`
section 12 and the P6 record): `/v1/completions` continues a prompt with its
special tokens parsed and **ignores `suffix`**, so a suffix goes to `/infill`,
which fills the middle and says `stop_type` `limit`, `eos` or `word`. Read in
their source: vLLM refuses a suffix for every model but DeepSeek V4 and
answers `/version`; Ollama's `/v1/completions` applies the chat template, so
raw continuation is `/api/generate` with `raw`. Every test here fails against
the driver before P6, which had no `completion` on a request.
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

LLAMA = "http://llama.local"
VLLM = "http://vllm.local"
OLLAMA = "http://127.0.0.1:11434"
PREFIX = "def add(a, b):\n    return"
SUFFIX = "\n\nprint(add(1, 2))\n"


def _config(tmp_path: Path, **values: Any) -> Path:
    config = tmp_path / f"{values['provider']}.yaml"
    config.write_text(json.dumps({"apiKey": "sk-test", **values}), "utf-8")
    return config


def _single(tmp_path: Path, base: str, *, locality: str = "local") -> TestClient:
    config = _config(
        tmp_path,
        provider="openai_compat_custom",
        baseUrl=base,
        modelId="coder",
        backendLocality=locality,
    )
    return TestClient(create_app(settings=Settings(config_file=config)))


def _llama(*, fills: bool = True) -> None:
    respx.get(f"{LLAMA}/props").mock(
        return_value=httpx.Response(200, json={"modalities": {"vision": False, "audio": False}})
    )
    respx.get(f"{LLAMA}/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "coder"}]})
    )
    probe = {"content": "", "stop": True} if fills else {"error": {"code": 501}}
    respx.post(f"{LLAMA}/infill").mock(
        side_effect=lambda r: httpx.Response(200 if fills else 501, json=_infill_answer(r, probe))
    )


def _infill_answer(request: httpx.Request, probe: dict[str, Any]) -> dict[str, Any]:
    body = json.loads(request.content)
    if body.get("n_predict") == 0:
        return probe
    return {
        "content": " a + b",
        "stop": True,
        "stop_type": "eos",
        "tokens_evaluated": 22,
        "tokens_predicted": 4,
    }


def ask(*, suffix: str | None = None, stream: bool = False, **extra: Any) -> dict[str, Any]:
    completion: dict[str, Any] = {"prompt": PREFIX}
    if suffix is not None:
        completion["suffix"] = suffix
    return {"model": "coder", "messages": [], "completion": completion, **extra}


def _sse(*frames: Any) -> str:
    return "".join(f"data: {f if isinstance(f, str) else json.dumps(f)}\n\n" for f in frames)


def _events(text: str) -> list[tuple[str, dict[str, Any]]]:
    events, kind = [], "message"
    for line in text.splitlines():
        if line.startswith("event:"):
            kind = line[6:].strip()
        elif line.startswith("data:"):
            events.append((kind, json.loads(line[5:])))
            kind = "message"
    return events


# --------------------------------------------------------------------------- #
# llama-server
# --------------------------------------------------------------------------- #


@respx.mock
def test_a_local_llama_server_continues_raw_text_and_fills_in_the_middle(tmp_path: Path) -> None:
    _llama()
    with _single(tmp_path, LLAMA) as client:
        model = client.get("/v1/info").json()["models"][0]
    assert model["surfaces"] == ["chat", "completion"]
    assert model["capabilities"]["fillInMiddle"] is True


@respx.mock
def test_a_prompt_goes_to_v1_completions_as_written(tmp_path: Path) -> None:
    _llama()
    upstream = respx.post(f"{LLAMA}/v1/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"text": " a + b", "finish_reason": "length", "index": 0}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 4},
            },
        )
    )
    with _single(tmp_path, LLAMA) as client:
        response = client.post(
            "/v1/generate",
            json=ask(maxTokens=4, temperature=0.01, stop=["\n\n"], callerSettings=["maxTokens"]),
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["content"], body["finishReason"]) == (" a + b", "length")
    assert (body["usage"]["promptTokens"], body["usage"]["completionTokens"]) == (8, 4)
    sent = json.loads(upstream.calls[0].request.content)
    assert sent == {
        "model": "coder",
        "prompt": PREFIX,
        "max_tokens": 4,
        "temperature": 0.01,
        "stop": ["\n\n"],
    }


@respx.mock
def test_a_suffix_goes_to_infill_since_v1_completions_would_drop_it(tmp_path: Path) -> None:
    _llama()
    completions = respx.post(f"{LLAMA}/v1/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"text": "x"}]})
    )
    with _single(tmp_path, LLAMA) as client:
        response = client.post("/v1/generate", json=ask(suffix=SUFFIX, maxTokens=12))
    assert response.status_code == 200, response.text
    assert response.json()["content"] == " a + b" and response.json()["finishReason"] == "stop"
    assert not completions.calls
    asked = [json.loads(c.request.content) for c in respx.calls if c.request.url.path == "/infill"]
    assert asked[-1] == {"input_prefix": PREFIX, "input_suffix": SUFFIX, "n_predict": 12}


@respx.mock
def test_a_suffix_for_a_model_that_cannot_fill_is_refused(tmp_path: Path) -> None:
    _llama(fills=False)
    with _single(tmp_path, LLAMA) as client:
        info = client.get("/v1/info").json()["models"][0]
        response = client.post("/v1/generate", json=ask(suffix=SUFFIX))
    assert info["capabilities"]["fillInMiddle"] is False
    assert response.status_code == 400, response.text
    assert response.json()["detail"]["type"].endswith("#completion-refused")
    assert "suffix" in response.json()["detail"]["detail"]


@respx.mock
def test_a_streamed_completion_is_relayed_and_must_say_it_finished(tmp_path: Path) -> None:
    _llama()
    frames = [
        {"choices": [{"text": " a", "finish_reason": None}]},
        {"choices": [{"text": " + b", "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": 4}},
        "[DONE]",
    ]
    respx.post(f"{LLAMA}/v1/completions").mock(
        side_effect=[
            httpx.Response(200, text=_sse(*frames)),
            httpx.Response(200, text=_sse(*frames[:1])),
        ]
    )
    with _single(tmp_path, LLAMA) as client:
        whole = client.post("/v1/generate/stream", json=ask())
        cut = client.post("/v1/generate/stream", json=ask())
    events = _events(whole.text)
    assert [e[1].get("text") for e in events if e[0] == "token"] == [" a", " + b"]
    done = next(e[1] for e in events if e[0] == "done")
    assert done["finishReason"] == "stop" and done["usage"]["completionTokens"] == 4
    assert any(e[0] == "error" for e in _events(cut.text)), cut.text


@respx.mock
def test_a_streamed_fill_comes_from_infill(tmp_path: Path) -> None:
    _llama()
    stream = _sse(
        {"content": " a", "stop": False},
        {"content": " +", "stop": False},
        {
            "content": "",
            "stop": True,
            "stop_type": "limit",
            "tokens_evaluated": 22,
            "tokens_predicted": 2,
        },
    )
    respx.post(f"{LLAMA}/infill").mock(
        side_effect=lambda r: (
            httpx.Response(200, json={"content": "", "stop": True})
            if json.loads(r.content).get("n_predict") == 0
            else httpx.Response(200, text=stream)
        )
    )
    with _single(tmp_path, LLAMA) as client:
        response = client.post("/v1/generate/stream", json=ask(suffix=SUFFIX, maxTokens=2))
    events = _events(response.text)
    assert [e[1].get("text") for e in events if e[0] == "token"] == [" a", " +"]
    assert next(e[1] for e in events if e[0] == "done")["finishReason"] == "length"


@respx.mock
def test_a_setting_a_raw_completion_would_drop_is_refused(tmp_path: Path) -> None:
    _llama()
    with _single(tmp_path, LLAMA) as client:
        response = client.post("/v1/generate", json=ask(topK=40, callerSettings=["topK"]))
    assert response.status_code == 400, response.text
    assert "topK" in response.json()["detail"]["detail"]


# --------------------------------------------------------------------------- #
# vLLM, an unknown server, a hosted one
# --------------------------------------------------------------------------- #


@respx.mock
def test_vllm_continues_raw_text_and_fills_nothing(tmp_path: Path) -> None:
    respx.get(f"{VLLM}/props").mock(return_value=httpx.Response(404))
    respx.get(f"{VLLM}/version").mock(return_value=httpx.Response(200, json={"version": "0.29.0"}))
    respx.get(f"{VLLM}/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "coder"}]})
    )
    with _single(tmp_path, VLLM) as client:
        model = client.get("/v1/info").json()["models"][0]
        refused = client.post("/v1/generate", json=ask(suffix=SUFFIX))
    assert "completion" in model["surfaces"] and model["capabilities"]["fillInMiddle"] is False
    assert refused.status_code == 400 and "suffix" in refused.json()["detail"]["detail"]


@respx.mock
def test_an_unknown_or_hosted_server_is_not_asked_to_continue_raw_text(tmp_path: Path) -> None:
    other = "http://other.local"
    respx.get(f"{other}/props").mock(return_value=httpx.Response(404))
    respx.get(f"{other}/version").mock(return_value=httpx.Response(404))
    respx.get(f"{other}/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "coder"}]})
    )
    completions = respx.post(f"{other}/v1/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"text": "x"}]})
    )
    with _single(tmp_path, other) as client:
        model = client.get("/v1/info").json()["models"][0]
        response = client.post("/v1/generate", json=ask())
    _llama()
    with _single(tmp_path, LLAMA, locality="external") as client:
        hosted = client.get("/v1/info").json()["models"][0]
    assert "completion" not in model["surfaces"] and "completion" not in hosted["surfaces"]
    assert response.status_code == 400, response.text
    assert response.json()["detail"]["type"].endswith("#completion-unsupported")
    assert not completions.calls


# --------------------------------------------------------------------------- #
# Ollama
# --------------------------------------------------------------------------- #


def _ollama(tmp_path: Path, generate: Any) -> tuple[TestClient, respx.Route]:
    respx.get(f"{OLLAMA}/api/tags").mock(
        return_value=httpx.Response(
            200, json={"models": [{"name": "qwen2.5-coder:1.5b"}, {"name": "llama3:8b"}]}
        )
    )
    shows = {
        "qwen2.5-coder:1.5b": ["completion", "insert"],
        "llama3:8b": ["completion", "tools"],
    }
    respx.post(f"{OLLAMA}/api/show").mock(
        side_effect=lambda r: httpx.Response(
            200, json={"capabilities": shows[json.loads(r.content)["model"]]}
        )
    )
    route = respx.post(f"{OLLAMA}/api/generate").mock(side_effect=generate)
    config = _config(tmp_path, provider="ollama_local")
    return TestClient(create_app(settings=Settings(config_file=config))), route


def _ready(client: TestClient) -> dict[str, Any]:
    for _ in range(250):
        info = client.get("/v1/info").json()
        catalogue = info.get("catalogue") or {}
        if catalogue.get("refreshedAt") or catalogue.get("error"):
            return info
    raise AssertionError("catalogue never read")


@respx.mock
def test_ollama_continues_raw_text_at_api_generate_and_fills_with_insert(tmp_path: Path) -> None:
    answer = {
        "response": " a + b",
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 9,
        "eval_count": 4,
    }
    client, generate = _ollama(tmp_path, lambda r: httpx.Response(200, json=answer))
    with client:
        info = _ready(client)
        plain = client.post(
            "/v1/generate",
            json={**ask(maxTokens=4), "model": "qwen2.5-coder:1.5b"},
        )
        filled = client.post(
            "/v1/generate", json={**ask(suffix=SUFFIX), "model": "qwen2.5-coder:1.5b"}
        )
        no_insert = client.post("/v1/generate", json={**ask(suffix=SUFFIX), "model": "llama3:8b"})
        biased = client.post(
            "/v1/generate",
            json={
                **ask(logitBias={"15": -100}, callerSettings=["logitBias"]),
                "model": "llama3:8b",
            },
        )
    caps = {m["id"]: m for m in info["models"]}
    assert caps["qwen2.5-coder:1.5b"]["surfaces"] == ["chat", "completion"]
    assert caps["qwen2.5-coder:1.5b"]["capabilities"]["fillInMiddle"] is True
    assert caps["llama3:8b"]["capabilities"]["fillInMiddle"] is False
    assert plain.status_code == 200 and plain.json()["content"] == " a + b", plain.text
    assert filled.status_code == 200, filled.text
    first, second = (json.loads(c.request.content) for c in generate.calls)
    assert first == {
        "model": "qwen2.5-coder:1.5b",
        "prompt": PREFIX,
        "stream": False,
        "raw": True,
        "options": {"num_predict": 4},
    }
    assert second["suffix"] == SUFFIX and "raw" not in second
    assert no_insert.status_code == 400 and "insert" in no_insert.json()["detail"]["detail"]
    assert biased.status_code == 400 and "logitBias" in biased.json()["detail"]["detail"]


@respx.mock
def test_ollama_streams_ndjson(tmp_path: Path) -> None:
    lines = "\n".join(
        json.dumps(f)
        for f in (
            {"response": " a", "done": False},
            {"response": " + b", "done": False},
            {
                "response": "",
                "done": True,
                "done_reason": "length",
                "prompt_eval_count": 9,
                "eval_count": 2,
            },
        )
    )
    client, _ = _ollama(tmp_path, lambda r: httpx.Response(200, text=lines))
    with client:
        _ready(client)
        response = client.post("/v1/generate/stream", json={**ask(), "model": "qwen2.5-coder:1.5b"})
    events = _events(response.text)
    assert [e[1].get("text") for e in events if e[0] == "token"] == [" a", " + b"]
    assert next(e[1] for e in events if e[0] == "done")["finishReason"] == "length"
