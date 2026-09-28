"""Images out (P4, 2026-09-28): what a model takes, and what an answer means.

Every engine that makes images shares these. Measured against OpenRouter and
the OpenAI SDK first (`provider-accounts-measurement.md` section 9):

* **OpenRouter's image settings are only on `GET /images/models`.** The main
  listing's `supported_parameters` for an image model is chat-style and says
  nothing about images. A setting the images listing names is enforced by
  OpenRouter's own 400; one it does not name is silently ignored (flux given
  `background: transparent` answered an opaque JPEG with a 200) or silently
  honoured (gpt-image-1-mini given an unlisted `output_format`). So an
  unlisted `quality` or `background` is reported as *not taken* (`[]`), and
  an unlisted `output_format` as *not said* (`None`): carried, with the
  answer labelled by its bytes.
* **An answer's format is read from its bytes**, never from what was asked
  or what the backend claims (P2-2).
"""

from __future__ import annotations

import base64
import binascii
from typing import Any

from ._generated.models import (
    GeneratedImage,
    ImageCapabilities,
    ImagePartial,
    ImageResponse,
    ImageUsage,
)

#: OpenAI's decoded upload limit, shared across every reference and the mask.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


class ImageRefusal(ValueError):
    """A request this driver will not send: a setting the model does not take,
    or input that is not what the contract says. A 400, never a cascade."""


def sniff(raw: bytes) -> str | None:
    """The media type an image's bytes carry, or None when they are none of
    the ones any backend here reads or writes."""
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    if raw.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    head = raw[:512].lstrip()
    if head.startswith(b"<svg") or (head.startswith(b"<?xml") and b"<svg" in raw[:4096]):
        return "image/svg+xml"
    return None


def format_name(media_type: str) -> str:
    """`image/jpeg` -> `jpeg`, `image/svg+xml` -> `svg`: OpenAI's spelling."""
    sub = media_type.split("/", 1)[-1]
    return sub.split("+", 1)[0]


# ---------------------------------------------------------------------------
# What a model takes
# ---------------------------------------------------------------------------


def _enum(params: dict[str, Any], key: str) -> list[str] | None:
    """A listed enum's values; None when the listing does not name it."""
    descriptor = params.get(key)
    if not isinstance(descriptor, dict):
        return None
    values = descriptor.get("values")
    return [v for v in values if isinstance(v, str)] if isinstance(values, list) else None


def _range(params: dict[str, Any], key: str) -> tuple[int, int] | None:
    descriptor = params.get(key)
    if not isinstance(descriptor, dict):
        return None
    low, high = descriptor.get("min"), descriptor.get("max")
    if isinstance(low, int) and isinstance(high, int):
        return low, high
    return None


def openrouter_caps(entry: dict[str, Any]) -> ImageCapabilities:
    """One `GET /images/models` entry, read the way the module docstring
    says: listed settings enforced, unlisted `quality`/`background` not
    taken, unlisted `output_format` not said."""
    raw = entry.get("supported_parameters")
    params: dict[str, Any] = raw if isinstance(raw, dict) else {}
    images = _range(params, "n")
    refs = _range(params, "input_references")
    return ImageCapabilities(
        streaming=entry.get("supports_streaming") is True,
        # An unlisted `n` is not carried by OpenRouter either way: one image.
        maxImages=images[1] if images else 1,
        minReferences=refs[0] if refs else 0,
        maxReferences=refs[1] if refs else 0,
        mask=False,
        qualities=_enum(params, "quality") or [],
        backgrounds=_enum(params, "background") or [],
        outputFormats=_enum(params, "output_format"),
    )


def openai_caps(model_id: str) -> ImageCapabilities:
    """An image model on OpenAI's own API, which lists nothing per model and
    checks its own fields (P1-3): everything carried, its refusal relayed.
    Only streaming is known by id: OpenAI streams its GPT image models, not
    `dall-e-*`."""
    lowered = model_id.lower()
    return ImageCapabilities(
        streaming=not lowered.startswith("dall-e-"),
        mask=True,
    )


# ---------------------------------------------------------------------------
# What an answer means
# ---------------------------------------------------------------------------


def _image(b64: Any, revised: Any = None) -> GeneratedImage:
    if not isinstance(b64, str) or not b64:
        raise ValueError("an image in the answer had no b64_json")
    try:
        # The header is all the type needs; 4096 characters is a whole
        # number of base64 quanta, so the prefix decodes on its own.
        raw = base64.b64decode(b64[:4096], validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError("an image in the answer was not base64") from e
    media = sniff(raw)
    if media is None:
        raise ValueError("an image in the answer was not PNG, JPEG, WebP, GIF or SVG")
    return GeneratedImage(
        data=b64,
        mediaType=media,
        revisedPrompt=revised if isinstance(revised, str) and revised else None,
    )


def usage_from(body: Any) -> ImageUsage | None:
    """OpenAI's `input_tokens`/`output_tokens`, or OpenRouter's chat-style
    `prompt_tokens`/`completion_tokens` and `cost`: the same quantities."""
    usage = body.get("usage") if isinstance(body, dict) else None
    if not isinstance(usage, dict):
        return None

    def whole(*keys: str) -> int | None:
        for key in keys:
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        return None

    cost = usage.get("cost")
    found = ImageUsage(
        inputTokens=whole("input_tokens", "prompt_tokens"),
        outputTokens=whole("output_tokens", "completion_tokens"),
        totalTokens=whole("total_tokens"),
        cost=float(cost) if isinstance(cost, int | float) and not isinstance(cost, bool) else None,
    )
    return found if found.model_dump(exclude_none=True) else None


def _text(body: dict[str, Any], key: str) -> str | None:
    value = body.get(key)
    return value if isinstance(value, str) and value else None


def response_from(body: Any, *, model_id: str | None, latency_ms: int) -> ImageResponse:
    """A backend's JSON answer: OpenAI's `ImagesResponse` or OpenRouter's,
    which adds `media_type` and says `created: 0` from some providers."""
    if not isinstance(body, dict) or not isinstance(body.get("data"), list) or not body["data"]:
        raise ValueError("the image answer had no `data`")
    images = [
        _image(item.get("b64_json"), item.get("revised_prompt"))
        for item in body["data"]
        if isinstance(item, dict)
    ]
    if not images:
        raise ValueError("the image answer had no images")
    created = body.get("created")
    return ImageResponse(
        images=images,
        created=created if isinstance(created, int) and created > 0 else None,
        size=_text(body, "size"),
        quality=_text(body, "quality"),
        background=_text(body, "background"),
        usage=usage_from(body),
        modelId=model_id,
        latencyMs=latency_ms,
    )


def event_kind(event: Any) -> str | None:
    """`partial` or `completed` for an image stream event, from its `type`
    (`image_generation.partial_image`, `image_edit.completed`, ...)."""
    kind = event.get("type") if isinstance(event, dict) else None
    if not isinstance(kind, str):
        return None
    if kind.endswith(".partial_image"):
        return "partial"
    if kind.endswith(".completed"):
        return "completed"
    return None


def partial_from(event: dict[str, Any], fallback_index: int) -> ImagePartial:
    index = event.get("partial_image_index")
    return ImagePartial(
        image=_image(event.get("b64_json")),
        index=index if isinstance(index, int) and index >= 0 else fallback_index,
    )


def completed_from(
    events: list[dict[str, Any]], *, model_id: str | None, latency_ms: int
) -> ImageResponse:
    """The final answer of a stream: one image per `completed` event, and
    the usage the last one carries."""
    if not events:
        raise ValueError("the image stream ended with no completed image")
    last = events[-1]
    created = last.get("created_at", last.get("created"))
    return ImageResponse(
        images=[_image(e.get("b64_json"), e.get("revised_prompt")) for e in events],
        created=created if isinstance(created, int) and created > 0 else None,
        size=_text(last, "size"),
        quality=_text(last, "quality"),
        background=_text(last, "background"),
        usage=usage_from(last),
        modelId=model_id,
        latencyMs=latency_ms,
    )


def decode_upload(b64: str, what: str) -> bytes:
    """One reference image or mask from the request, with its type checked
    against its bytes."""
    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        raise ImageRefusal(f"{what}: must be base64 with no data: prefix.") from None
    if sniff(raw) not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
        raise ImageRefusal(f"{what}: is not a PNG, JPEG, WebP or GIF image.")
    return raw
