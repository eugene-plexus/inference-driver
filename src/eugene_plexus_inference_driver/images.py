"""Bounded inline attachment validation: images, audio and PDFs.

Never resolves a URL or reflects a value it was sent.
"""

from __future__ import annotations

import base64
import binascii
import io
import warnings
from typing import Any

from PIL import Image
from pydantic import RootModel

#: A ceiling, not the policy. The gateway's `maxImagesPerRequest` decides how
#: many images a request may carry (12 by default, counted across the whole
#: conversation) and cannot be set above this. It was a fixed four here and
#: at the gateway until 2026-09-23, which refused a Claude Code session on
#: every turn after its fifth screenshot.
MAX_IMAGES = 64
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_TOTAL_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 16_000_000
MAX_DIMENSION = 8192
#: One audio clip or file, decoded (P2, 2026-09-28).
MAX_ATTACHMENT_BYTES = 10 * 1024 * 1024
#: Every attachment in a request together, images included: what fits in the
#: 16 MiB JSON body once base64 has grown it by a third.
MAX_ATTACHMENTS_TOTAL = 11 * 1024 * 1024
#: A content part's `type` and the input kind it asks the model to take.
PART_KINDS = {"image_url": "image", "input_audio": "audio", "file": "file"}
_PDF_DATA_URL = "data:application/pdf;base64,"


class ImageRefusal(ValueError):
    def __init__(self, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(f"{field}: {reason}.")


def content_wire(content: Any) -> Any:
    return (
        content.model_dump(mode="json", exclude_none=True)
        if hasattr(content, "model_dump")
        else content
    )


def attachment_kinds(messages: Any) -> frozenset[str]:
    """Which kinds of input a request asks its model to take: `image`,
    `audio`, `file`. Empty for text alone."""
    return frozenset(
        PART_KINDS[part.get("type")]
        for message in messages or []
        if isinstance(content := content_wire(message.content), list)
        for part in content
        if part.get("type") in PART_KINDS
    )


def has_images(messages: Any) -> bool:
    return "image" in attachment_kinds(messages)


def _typed_parts(content: Any) -> list[Any] | None:
    """The content's own part objects, so a normalised value can be written back."""
    while isinstance(content, RootModel):
        content = content.root
    return content if isinstance(content, list) else None


def validate_messages(messages: Any) -> None:
    """Validate typed messages, then flatten only arrays consisting entirely of text.

    A PDF sent as bare base64 is rewritten in place as the data URL every
    backend accepts; nothing else is changed.
    """
    count = image_total = total = 0
    for index, message in enumerate(messages or []):
        content = content_wire(message.content)
        if not isinstance(content, list):
            continue
        typed = _typed_parts(message.content)
        contains_attachment = False
        for part_index, part in enumerate(content):
            kind = PART_KINDS.get(part["type"])
            if kind is None:
                continue
            field = f"messages[{index}].content[{part_index}].{part['type']}"
            if getattr(message.role, "value", message.role) != "user":
                raise ImageRefusal(
                    field,
                    "images are supported only on user messages"
                    if kind == "image"
                    else "attachments are supported only on user messages",
                )
            contains_attachment = True
            if kind == "image":
                count += 1
                if count > MAX_IMAGES:
                    raise ImageRefusal(
                        field, f"at most {MAX_IMAGES} images are allowed per request"
                    )
                size = _validate_image(part["image_url"]["url"], field)
                image_total += size
                if image_total > MAX_TOTAL_BYTES:
                    raise ImageRefusal(field, "images exceed the 10 MiB decoded request limit")
            elif kind == "audio":
                audio = part["input_audio"]
                size = validate_audio(audio["data"], audio["format"], field)
            else:
                url, size = validate_file(part["file"], field)
                if url != part["file"].get("file_data") and typed is not None:
                    _write_file_data(typed[part_index], url)
            total += size
            if total > MAX_ATTACHMENTS_TOTAL:
                raise ImageRefusal(field, "attachments exceed the 11 MiB decoded request limit")
        if not contains_attachment:
            message.content = "".join(part["text"] for part in content)


def _write_file_data(part: Any, url: str) -> None:
    if isinstance(part, dict):
        part["file"]["file_data"] = url
    else:
        part.file.file_data = url


def validate_audio(data: str, fmt: str, field: str) -> int:
    """An `input_audio` clip's decoded size, once its bytes match its format."""
    if len(data) > 4 * ((MAX_ATTACHMENT_BYTES + 2) // 3):
        raise ImageRefusal(field, "audio exceeds the 10 MiB decoded limit")
    try:
        raw = base64.b64decode(data, validate=True)
    except (ValueError, binascii.Error):
        raise ImageRefusal(
            field, "audio has invalid base64; send it without a data: prefix"
        ) from None
    if not raw or len(raw) > MAX_ATTACHMENT_BYTES:
        raise ImageRefusal(field, "audio is empty or exceeds the 10 MiB decoded limit")
    if fmt == "wav":
        matches = raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"
    else:
        # An ID3 tag, or straight into an MPEG frame: eleven set sync bits.
        matches = raw[:3] == b"ID3" or (len(raw) > 1 and raw[0] == 0xFF and raw[1] & 0xE0 == 0xE0)
    if not matches:
        raise ImageRefusal(field, f"audio does not match its declared {fmt} format")
    return len(raw)


def validate_file(file: dict[str, Any], field: str) -> tuple[str, int]:
    """A `file` part's PDF as a data URL, and its decoded size."""
    if file.get("file_id"):
        raise ImageRefusal(
            f"{field}.file_id",
            "names an uploaded file, and this install has no file store; send file_data",
        )
    data = file.get("file_data")
    if not isinstance(data, str) or not data:
        raise ImageRefusal(f"{field}.file_data", "is required: send the PDF inline")
    if data.startswith("data:"):
        header, separator, encoded = data.partition(",")
        if not separator or f"{header}," != _PDF_DATA_URL:
            raise ImageRefusal(
                f"{field}.file_data",
                "use a base64 PDF data URL (data:application/pdf;base64,...); "
                "other file types are not supported",
            )
    else:
        encoded = data
    if len(encoded) > 4 * ((MAX_ATTACHMENT_BYTES + 2) // 3):
        raise ImageRefusal(f"{field}.file_data", "file exceeds the 10 MiB decoded limit")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        raise ImageRefusal(
            f"{field}.file_data", "is neither a PDF data URL nor base64; URLs are not fetched"
        ) from None
    if not raw.startswith(b"%PDF-"):
        raise ImageRefusal(f"{field}.file_data", "is not a PDF")
    return _PDF_DATA_URL + encoded, len(raw)


def _validate_image(url: str, field: str) -> int:
    header, separator, encoded = url.partition(",")
    formats = {"data:image/png;base64": "PNG", "data:image/jpeg;base64": "JPEG"}
    if not separator or header not in formats:
        raise ImageRefusal(field, "use an inline base64 PNG or JPEG; URLs are not fetched")
    if len(encoded) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
        raise ImageRefusal(field, "image exceeds the 5 MiB decoded limit")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        raise ImageRefusal(field, "image has invalid base64") from None
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ImageRefusal(field, "image is empty or exceeds the 5 MiB decoded limit")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data), formats=[formats[header]]) as picture:
                width, height = picture.size
                if (
                    max(width, height) > MAX_DIMENSION
                    or width * height > MAX_PIXELS
                    or min(width, height) < 1
                ):
                    raise ImageRefusal(
                        field, "image exceeds 8192 pixels per side or 16 million pixels"
                    )
                if getattr(picture, "is_animated", False):
                    raise ImageRefusal(field, "animated images are not supported")
                picture.verify()
            # JPEG verify alone does not decode truncated pixel data.
            with Image.open(io.BytesIO(data), formats=[formats[header]]) as picture:
                picture.load()
    except ImageRefusal:
        raise
    except (
        OSError,
        ValueError,
        SyntaxError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ):
        raise ImageRefusal(field, "image is invalid or does not match its PNG/JPEG type") from None
    return len(data)
