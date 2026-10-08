"""Videos (P5, 2026-09-28): what a model takes, and what a job's answer means.

Measured first (`provider-accounts-measurement.md` section 10): OpenAI's video
API shut down on 2026-09-24, so OpenRouter is the one backend. Its shape is
its own: `duration` an integer, `size` one the model lists (another is its
400), the first frame as `frame_images`; a job is accepted as `pending`, ends
`completed` or `failed` with no `progress` and no `in_progress` seen, and a
failed job's `error` is a string.
"""

from __future__ import annotations

import re
from typing import Any

from ._generated.models import (
    VideoCapabilities,
    VideoJob,
    VideoJobStatus,
    VideoPrice,
    VideoPriceUnit,
)


class VideoRefusal(ValueError):
    """A video request this driver will not send. A 400, never a cascade."""


#: OpenRouter's words, and OpenAI's (which the gateway speaks).
_STATUSES = {
    "pending": VideoJobStatus.queued,
    "queued": VideoJobStatus.queued,
    "in_progress": VideoJobStatus.in_progress,
    "processing": VideoJobStatus.in_progress,
    "completed": VideoJobStatus.completed,
    "failed": VideoJobStatus.failed,
    "cancelled": VideoJobStatus.failed,
    "expired": VideoJobStatus.failed,
}


def openrouter_caps(entry: dict[str, Any]) -> VideoCapabilities:
    """One `GET /videos/models` entry."""
    durations = entry.get("supported_durations")
    sizes = entry.get("supported_sizes")
    frames = entry.get("supported_frame_images")
    listed_sizes = [s for s in sizes if isinstance(s, str)] if isinstance(sizes, list) else None
    return VideoCapabilities(
        durations=[d for d in durations if isinstance(d, int)]
        if isinstance(durations, list)
        else None,
        sizes=listed_sizes,
        firstFrame=isinstance(frames, list) and "first_frame" in frames,
        prices=openrouter_prices(entry.get("pricing_skus"), listed_sizes),
    )


# OpenRouter's `pricing_skus` (measured 2026-10-08, 30 models): a flat map of
# its own line names to decimal strings, in at least four units. Lines named
# `cents_...` are cents; the `duration_seconds` family is dollars a second
# (Veo 3.1 lists 0.40 with audio, its published price). Video tokens,
# megapixel-seconds, continuations and references price what no request
# here sends, so they are left out.
_CENTS_SECOND = re.compile(
    r"cents_per_(?:video_output_second|second_output)(?:_(?P<res>\d+p|[24]k))?"
)
_DOLLARS_SECOND = re.compile(
    r"(?:(?P<mode>text|image)_to_video_)?duration_seconds"
    r"(?:_(?P<audio>with|without)_audio)?(?:_(?P<res>\d+p|[24]k))?"
)
#: A resolution class's shorter side, in pixels.
_HEIGHTS = {"2K": 1440, "4K": 2160}


def _resolution(raw: str | None) -> str | None:
    return raw.upper() if raw and raw.endswith("k") else raw


def _sizes_of(resolution: str, sizes: list[str] | None) -> list[str] | None:
    """The listed sizes of one resolution class: those whose shorter side is
    its height (`854x480` is 480p). None when the model lists no sizes."""
    if sizes is None:
        return None
    height = _HEIGHTS.get(resolution) or int(resolution.removesuffix("p"))
    found = []
    for size in sizes:
        w, _, h = size.partition("x")
        if w.isdigit() and h.isdigit() and min(int(w), int(h)) == height:
            found.append(size)
    return found


def _usd(raw: Any, *, cents: bool) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value < 0 or value != value:  # negative, or NaN
        return None
    return value / 100 if cents else value


def openrouter_prices(skus: Any, sizes: list[str] | None) -> list[VideoPrice] | None:
    """The lines of a `pricing_skus` map a request can be priced by. None
    when no line prices a second of output: a list of only an input image's
    price would read as a price for the video."""
    if not isinstance(skus, dict):
        return None
    lines: list[VideoPrice] = []
    for sku, raw in skus.items():
        if not isinstance(sku, str):
            continue
        fields: dict[str, Any] = {}
        if match := _CENTS_SECOND.fullmatch(sku):
            per, usd = VideoPriceUnit.second, _usd(raw, cents=True)
        elif match := _DOLLARS_SECOND.fullmatch(sku):
            per, usd = VideoPriceUnit.second, _usd(raw, cents=False)
            if match["audio"]:
                fields["audio"] = match["audio"] == "with"
            if match["mode"]:
                fields["firstFrame"] = match["mode"] == "image"
        elif sku == "cents_per_image_input":
            match, per, usd = None, VideoPriceUnit.input_image, _usd(raw, cents=True)
        elif sku == "minimum_cents_per_generation":
            match, per, usd = None, VideoPriceUnit.minimum, _usd(raw, cents=True)
        else:
            continue
        if usd is None:
            continue
        resolution = _resolution(match["res"]) if match else None
        if resolution:
            fields["resolution"] = resolution
            fields["sizes"] = _sizes_of(resolution, sizes)
        lines.append(VideoPrice(sku=sku, per=per, usd=usd, **fields))
    if not any(line.per == VideoPriceUnit.second for line in lines):
        return None
    return lines


def job_from(body: Any, *, model_id: str | None) -> VideoJob:
    """OpenRouter's job, on submit or poll, in OpenAI's words."""
    if not isinstance(body, dict) or not isinstance(body.get("id"), str):
        raise ValueError("the video job had no `id`")
    raw = body.get("status")
    status = _STATUSES.get(raw) if isinstance(raw, str) else None
    if status is None:
        raise ValueError(f"the video job's status {raw!r} is not one this driver knows")
    usage = body.get("usage")
    cost = usage.get("cost") if isinstance(usage, dict) else None
    error = body.get("error")
    if isinstance(error, dict):
        error = error.get("message")
    progress = body.get("progress")
    return VideoJob(
        jobId=body["id"],
        status=status,
        progress=progress if isinstance(progress, int) and 0 <= progress <= 100 else None,
        error=error if isinstance(error, str) and error else None,
        cost=float(cost) if isinstance(cost, int | float) and not isinstance(cost, bool) else None,
        modelId=model_id,
    )
