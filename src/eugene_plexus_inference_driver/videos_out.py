"""Videos (P5, 2026-09-28): what a model takes, and what a job's answer means.

Measured first (`provider-accounts-measurement.md` section 10): OpenAI's video
API shut down on 2026-09-24, so OpenRouter is the one backend. Its shape is
its own: `duration` an integer, `size` one the model lists (another is its
400), the first frame as `frame_images`; a job is accepted as `pending`, ends
`completed` or `failed` with no `progress` and no `in_progress` seen, and a
failed job's `error` is a string.
"""

from __future__ import annotations

from typing import Any

from ._generated.models import VideoCapabilities, VideoJob, VideoJobStatus


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
    return VideoCapabilities(
        durations=[d for d in durations if isinstance(d, int)]
        if isinstance(durations, list)
        else None,
        sizes=[s for s in sizes if isinstance(s, str)] if isinstance(sizes, list) else None,
        firstFrame=isinstance(frames, list) and "first_frame" in frames,
    )


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
