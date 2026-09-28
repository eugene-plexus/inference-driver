"""POST /v1/video, GET /v1/video/{jobId} and its content, through OpenRouter (P5).

Measured 2026-09-28 (`provider-accounts-measurement.md` section 10): OpenAI's
video API shut down on 2026-09-24; OpenRouter's is its own shape (`duration`
an integer, a listed `size`, the first frame as `frame_images`), lists each
model's settings only on `GET /videos/models`, accepts a job as `pending`,
ends it `completed` with `unsigned_urls` and `usage.cost` or `failed` with a
string `error`, answers an unknown job 404, and streams the MP4 chunked. Every
test here fails against the driver before P5, which had no video route.
"""

from __future__ import annotations

import base64
import datetime as dt
import json
from pathlib import Path
from typing import Any

import httpx
import respx
from fastapi.testclient import TestClient

from eugene_plexus_inference_driver.app import create_app
from eugene_plexus_inference_driver.engines._catalogue import shut_down
from eugene_plexus_inference_driver.settings import Settings

OPENROUTER = "https://openrouter.ai/api"
OPENAI = "https://api.openai.com"
GROK = "x-ai/grok-imagine-video"
FLUX_EDIT = "black-forest-labs/flux-video-edit"
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
MP4 = b"\x00\x00\x00 ftypisom" + bytes(range(256)) * 16
JOB = "gen-vid-1790630316-Z9CqQ0yFqoQmf5S98gYH"

LISTING = {
    "data": [
        {
            "id": GROK,
            "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["video"]},
            "supported_parameters": [],
        },
        {
            "id": FLUX_EDIT,
            "architecture": {"input_modalities": ["text", "video"], "output_modalities": ["video"]},
            "supported_parameters": [],
        },
        {
            "id": "mistralai/mistral-nemo",
            "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
            "supported_parameters": ["max_tokens"],
        },
    ]
}
#: `GET /videos/models` as measured for grok; flux-video-edit lists nothing.
VIDEOS = {
    "data": [
        {
            "id": GROK,
            "supported_durations": [1, 2, 3, 4, 5],
            "supported_sizes": ["854x480", "1280x720", "720x1280"],
            "supported_resolutions": ["480p", "720p"],
            "supported_frame_images": ["first_frame"],
        },
        {
            "id": FLUX_EDIT,
            "supported_durations": None,
            "supported_sizes": None,
            "supported_frame_images": None,
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


def _openrouter(tmp_path: Path, *, videos: httpx.Response | None = None) -> TestClient:
    respx.get(f"{OPENROUTER}/v1/models/user").mock(return_value=httpx.Response(200, json=LISTING))
    respx.get(f"{OPENROUTER}/v1/images/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    respx.get(f"{OPENROUTER}/v1/videos/models").mock(
        return_value=videos or httpx.Response(200, json=VIDEOS)
    )
    app = create_app(settings=Settings(config_file=_config(tmp_path, provider="openrouter")))
    return TestClient(app)


def _ask(model: str = GROK, **extra: Any) -> dict[str, Any]:
    return {"model": model, "prompt": "a red ball bouncing", **extra}


# --------------------------------------------------------------------------- #
# What each model is listed with
# --------------------------------------------------------------------------- #


@respx.mock
def test_the_video_listing_decides_the_surface_and_its_settings(tmp_path: Path) -> None:
    with _openrouter(tmp_path) as client:
        models = {m["id"]: m for m in _ready(client)["models"]}
    grok = models[GROK]
    assert grok["surfaces"] == ["video"]
    assert grok["capabilities"]["video"] == {
        "durations": [1, 2, 3, 4, 5],
        "sizes": ["854x480", "1280x720", "720x1280"],
        "firstFrame": True,
    }
    assert models[FLUX_EDIT]["capabilities"]["video"] == {"firstFrame": False}


@respx.mock
def test_an_unreadable_video_listing_leaves_the_account_serving(tmp_path: Path) -> None:
    with _openrouter(tmp_path, videos=httpx.Response(404, text="Not Found")) as client:
        info = _ready(client)
    models = {m["id"]: m for m in info["models"]}
    assert models["mistralai/mistral-nemo"]["surfaces"] == ["chat"]
    assert models[GROK]["surfaces"] == ["video"]
    assert "video" not in models[GROK]["capabilities"]


def test_a_model_past_its_shutdown_date_is_shut_down() -> None:
    today = dt.date(2026, 9, 28)
    assert shut_down({"id": "sora-2", "shutdown_date": "2026-09-24"}, today) is True
    assert shut_down({"id": "gpt-image-1-mini", "shutdown_date": "2026-12-01"}, today) is False
    assert shut_down({"id": "tts-1", "shutdown_date": None}, today) is False
    assert shut_down({"id": "odd", "shutdown_date": "soon"}, today) is False


@respx.mock
def test_an_openai_account_does_not_list_a_shut_down_model(tmp_path: Path) -> None:
    listing = {
        "data": [
            {"id": "sora-2", "shutdown_date": "2026-09-24"},
            {"id": "gpt-image-1-mini", "shutdown_date": "2099-12-01"},
            {"id": "gpt-4o"},
        ]
    }
    respx.get(f"{OPENAI}/v1/models").mock(return_value=httpx.Response(200, json=listing))
    app = create_app(settings=Settings(config_file=_config(tmp_path, provider="openai")))
    with TestClient(app) as client:
        ids = sorted(m["id"] for m in _ready(client)["models"])
    assert ids == ["gpt-4o", "gpt-image-1-mini"]


# --------------------------------------------------------------------------- #
# Submit, poll, download
# --------------------------------------------------------------------------- #


@respx.mock
def test_a_submit_asks_openrouter_in_its_own_shape(tmp_path: Path) -> None:
    route = respx.post(f"{OPENROUTER}/v1/videos").mock(
        return_value=httpx.Response(
            202,
            json={"id": JOB, "polling_url": f"{OPENROUTER}/v1/videos/{JOB}", "status": "pending"},
        )
    )
    frame = {"data": base64.b64encode(PNG).decode(), "mediaType": "image/png"}
    with _openrouter(tmp_path) as client:
        _ready(client)
        response = client.post("/v1/video", json=_ask(seconds=4, size="1280x720", firstFrame=frame))
    assert response.status_code == 200, response.text
    assert response.json() == {"jobId": JOB, "status": "queued", "modelId": GROK}
    sent = json.loads(route.calls.last.request.content)
    assert sent == {
        "model": GROK,
        "prompt": "a red ball bouncing",
        "duration": 4,
        "size": "1280x720",
        "frame_images": [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64," + frame["data"]},
                "frame_type": "first_frame",
            }
        ],
    }


@respx.mock
def test_a_first_frame_for_a_model_that_takes_none_is_refused_unsent(tmp_path: Path) -> None:
    route = respx.post(f"{OPENROUTER}/v1/videos").mock(return_value=httpx.Response(202, json={}))
    frame = {"data": base64.b64encode(PNG).decode(), "mediaType": "image/png"}
    with _openrouter(tmp_path) as client:
        _ready(client)
        response = client.post("/v1/video", json=_ask(FLUX_EDIT, firstFrame=frame))
    assert response.status_code == 400, response.text
    assert "first frame" in response.json()["detail"]["detail"]
    assert not route.called


@respx.mock
def test_a_first_frame_that_is_not_an_image_is_refused_unsent(tmp_path: Path) -> None:
    route = respx.post(f"{OPENROUTER}/v1/videos").mock(return_value=httpx.Response(202, json={}))
    frame = {"data": base64.b64encode(b"%PDF-1.7").decode(), "mediaType": "image/png"}
    with _openrouter(tmp_path) as client:
        _ready(client)
        response = client.post("/v1/video", json=_ask(firstFrame=frame))
    assert response.status_code == 400, response.text
    assert not route.called


@respx.mock
def test_a_size_openrouter_does_not_list_is_its_400_relayed(tmp_path: Path) -> None:
    words = 'Unsupported size "848x480". Use a standard resolution and aspect ratio combination'
    respx.post(f"{OPENROUTER}/v1/videos").mock(
        return_value=httpx.Response(400, json={"error": {"message": words, "code": 400}})
    )
    with _openrouter(tmp_path) as client:
        _ready(client)
        response = client.post("/v1/video", json=_ask(size="848x480"))
    assert response.status_code == 400, response.text
    assert "Unsupported size" in response.text


@respx.mock
def test_a_poll_reads_each_state_in_openais_words(tmp_path: Path) -> None:
    states = iter(
        [
            {"id": JOB, "generation_id": JOB, "status": "pending"},
            {
                "id": JOB,
                "status": "completed",
                "unsigned_urls": [f"{OPENROUTER}/v1/videos/{JOB}/content?index=0"],
                "usage": {"cost": 0.05, "is_byok": False},
            },
            {"id": JOB, "status": "failed", "error": "Image dimensions 1x1 are too small."},
        ]
    )
    respx.get(f"{OPENROUTER}/v1/videos/{JOB}").mock(
        side_effect=lambda request: httpx.Response(200, json=next(states))
    )
    with _openrouter(tmp_path) as client:
        _ready(client)
        answers = [client.get(f"/v1/video/{JOB}").json() for _ in range(3)]
    assert answers[0] == {"jobId": JOB, "status": "queued"}
    assert answers[1] == {"jobId": JOB, "status": "completed", "cost": 0.05}
    assert answers[2] == {
        "jobId": JOB,
        "status": "failed",
        "error": "Image dimensions 1x1 are too small.",
    }


@respx.mock
def test_an_unknown_job_is_a_404(tmp_path: Path) -> None:
    respx.get(f"{OPENROUTER}/v1/videos/nope").mock(
        return_value=httpx.Response(
            404, json={"error": {"message": "Job nope not found", "code": 404}}
        )
    )
    with _openrouter(tmp_path) as client:
        _ready(client)
        response = client.get("/v1/video/nope")
    assert response.status_code == 404, response.text


@respx.mock
def test_the_content_is_streamed_as_the_backend_sends_it(tmp_path: Path) -> None:
    route = respx.get(f"{OPENROUTER}/v1/videos/{JOB}/content").mock(
        return_value=httpx.Response(200, headers={"content-type": "video/mp4"}, content=MP4)
    )
    with _openrouter(tmp_path) as client:
        _ready(client)
        response = client.get(f"/v1/video/{JOB}/content")
    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"
    assert response.content == MP4
    assert route.calls.last.request.url.params["index"] == "0"


@respx.mock
def test_an_openai_account_makes_no_videos(tmp_path: Path) -> None:
    """OpenAI's video API shut down on 2026-09-24 (measured): a video model
    an OpenAI list still names without a shutdown date is refused, unsent."""
    respx.get(f"{OPENAI}/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "sora-3"}]})
    )
    route = respx.post(f"{OPENAI}/v1/videos").mock(return_value=httpx.Response(404))
    app = create_app(settings=Settings(config_file=_config(tmp_path, provider="openai")))
    with TestClient(app) as client:
        _ready(client)
        response = client.post("/v1/video", json=_ask("sora-3"))
    assert response.status_code == 400, response.text
    assert "OpenRouter" in response.json()["detail"]["detail"]
    assert not route.called
