"""Transcription (P3b, 2026-09-28): what a backend's answer means.

Every engine that transcribes shares these: the Qwen3-ASR preamble that
`llama-server` leaves in its text, and the reading of a backend's usage,
which is seconds of audio for Whisper and tokens for the rest (measured).
"""

from __future__ import annotations

import re
from typing import Any

from ._generated.models import TranscribeResponse, TranscriptionUsage

#: `llama-server` b11235 with Qwen3-ASR answers `"language English<asr_text>The
#: quick brown fox..."` (measured 2026-09-28). The same model on OpenRouter
#: answers the text alone, so the prefix is llama.cpp not parsing the model's
#: own output, and it is parsed here.
_ASR_PREAMBLE = re.compile(
    r"\A\s*language\s+([A-Za-z][A-Za-z -]{0,40}?)\s*<asr_text>(.*)\Z", re.DOTALL
)


def split_preamble(text: str) -> tuple[str | None, str]:
    """`(language, text)`: the language named by an ASR preamble, if the
    text carries one, and the text without it."""
    match = _ASR_PREAMBLE.match(text)
    if match is None:
        return None, text
    return match.group(1).strip().lower(), match.group(2).strip()


def usage_from(body: Any) -> TranscriptionUsage | None:
    """The backend's `usage`, in its own unit. OpenRouter's Whisper says
    `{"seconds": 3.5, "cost": ...}`; OpenAI's newer models and
    `llama-server` say tokens."""
    usage = body.get("usage") if isinstance(body, dict) else None
    if not isinstance(usage, dict):
        return None

    def number(key: str) -> Any:
        value = usage.get(key)
        return value if isinstance(value, int | float) and not isinstance(value, bool) else None

    def whole(key: str) -> int | None:
        value = number(key)
        return int(value) if value is not None else None

    found = TranscriptionUsage(
        seconds=number("seconds"),
        inputTokens=whole("input_tokens") if "input_tokens" in usage else whole("prompt_tokens"),
        outputTokens=whole("output_tokens")
        if "output_tokens" in usage
        else whole("completion_tokens"),
        totalTokens=whole("total_tokens"),
    )
    return found if found.model_dump(exclude_none=True) else None


def response_from(body: Any, *, model_id: str | None, latency_ms: int) -> TranscribeResponse:
    """A backend's JSON answer, read the one way every door needs it."""
    if not isinstance(body, dict) or not isinstance(body.get("text"), str):
        raise ValueError("the transcription answer had no `text`")
    language, text = split_preamble(body["text"])
    reported = body.get("language")
    duration = body.get("duration")
    segments = body.get("segments")
    words = body.get("words")
    return TranscribeResponse(
        text=text,
        language=reported if isinstance(reported, str) and reported else language,
        duration=duration
        if isinstance(duration, int | float) and not isinstance(duration, bool)
        else None,
        segments=segments if isinstance(segments, list) else None,
        words=words if isinstance(words, list) else None,
        usage=usage_from(body),
        modelId=model_id,
        latencyMs=latency_ms,
    )
