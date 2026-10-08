"""Speech (P3a, 2026-09-28): formats, media types, and a WAV made from pcm.

Every engine that speaks shares these: what a format is called on the wire,
which formats a kind of backend can make (measured 2026-09-28), and the one
format this driver makes itself -- `wav`, from a backend's `pcm`.
"""

from __future__ import annotations

import struct

from ._generated.models import SpeechFormat

#: The media type each format is served as.
MEDIA_TYPES: dict[SpeechFormat, str] = {
    SpeechFormat.mp3: "audio/mpeg",
    SpeechFormat.opus: "audio/ogg",
    SpeechFormat.aac: "audio/aac",
    SpeechFormat.flac: "audio/flac",
    SpeechFormat.wav: "audio/wav",
    SpeechFormat.pcm: "audio/pcm",
}

#: Everything OpenAI's own API makes, and what a local OpenAI-shaped
#: speech server is assumed to take until it refuses.
ALL_FORMATS: tuple[SpeechFormat, ...] = tuple(SpeechFormat)

#: The voices OpenAI's TTS models take, measured 2026-10-08 by sending each
#: documented voice: `tts-1` and its dated and `-hd` kin refuse ballad,
#: cedar, marin and verse (its 400 lists the nine); `gpt-4o-mini-tts` takes
#: all thirteen. OpenAI's `/v1/models` says nothing per model, so a TTS id
#: outside these families lists none, and its own 400 names what it takes.
OPENAI_TTS_1_VOICES: tuple[str, ...] = (
    "alloy", "ash", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer",
)  # fmt: skip
OPENAI_GPT_TTS_VOICES: tuple[str, ...] = (
    "alloy", "ash", "ballad", "cedar", "coral", "echo", "fable", "marin", "nova", "onyx",
    "sage", "shimmer", "verse",
)  # fmt: skip


def openai_voices(model_id: str) -> list[str] | None:
    """The voices an OpenAI TTS model takes, where measured; None otherwise."""
    lowered = model_id.lower()
    if lowered.startswith("tts-1"):
        return list(OPENAI_TTS_1_VOICES)
    if lowered.startswith("gpt-4o-mini-tts"):
        return list(OPENAI_GPT_TTS_VOICES)
    return None


#: OpenRouter's speech route takes `mp3` and `pcm` only (measured: a Zod 400
#: listing the two); `wav` is made here from `pcm`.
OPENROUTER_FORMATS: tuple[SpeechFormat, ...] = (
    SpeechFormat.mp3,
    SpeechFormat.pcm,
    SpeechFormat.wav,
)
#: ElevenLabs makes mp3, pcm and opus on its lower plans; its own WAV is
#: Pro-tier only (measured 403), so `wav` is made here from `pcm`.
ELEVENLABS_FORMATS: tuple[SpeechFormat, ...] = (
    SpeechFormat.mp3,
    SpeechFormat.opus,
    SpeechFormat.pcm,
    SpeechFormat.wav,
)

#: OpenAI's `pcm`: 24 kHz, mono, 16-bit little-endian, no header.
PCM_RATE = 24_000
_UNKNOWN = 0xFFFFFFFF


class SpeechRefusal(ValueError):
    """A speech request this driver refuses before any network: a format
    the model cannot make, a setting its backend cannot carry."""


def streaming_wav_header() -> bytes:
    """A WAV header for `pcm` samples still being generated.

    The sizes are unknown when the first byte goes out, and nothing is
    buffered to learn them, so both carry `0xFFFFFFFF`: the streaming
    convention, which players read as "until the end of the file".
    """
    block = 2  # mono, 16-bit
    return (
        b"RIFF"
        + struct.pack("<I", _UNKNOWN)
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, PCM_RATE, PCM_RATE * block, block, 16)
        + b"data"
        + struct.pack("<I", _UNKNOWN)
    )


def refuse_format(
    asked: SpeechFormat, formats: tuple[SpeechFormat, ...] | list[SpeechFormat]
) -> None:
    """Refuse a format the model cannot make, naming the ones it can."""
    if asked not in formats:
        named = ", ".join(f.value for f in formats)
        raise SpeechRefusal(
            f"format {asked.value!r} is not one this model can be given in; it can be "
            f"given in {named}"
        )
