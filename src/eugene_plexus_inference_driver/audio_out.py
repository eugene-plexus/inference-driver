"""Audio in the answer (P2b, 2026-09-28): what the bytes are, and one whole clip.

Every audio-output model behind an account answers audio only when streamed
and only as `pcm16` (measured 2026-09-28 against OpenRouter; OpenAI's own
400 says *"Supported values are: 'pcm16'"*). So this driver always asks for
a `pcm16` stream, and:

* a streamed caller is handed the fragments as they arrive;
* a non-streamed caller is handed the stream assembled, with a WAV header
  when `wav` was asked;
* either is told the format the bytes ARE, which is not always the format
  asked: Lyria sends one MP3 whatever it is asked for (P2-2).
"""

from __future__ import annotations

import base64
import binascii
import struct
from typing import Any

from ._generated.models import AudioDelta, AudioOutputFormat, GeneratedAudio

#: What `POST /v1/generate` can serve: the stream assembled, with or
#: without a header. The other formats would need a transcoder (P2-1).
BATCH_FORMATS = frozenset({AudioOutputFormat.wav, AudioOutputFormat.pcm16})
#: What `POST /v1/generate/stream` can carry: what the backend streams.
STREAM_FORMATS = frozenset({AudioOutputFormat.pcm16})

#: OpenAI's documented `pcm16`: 24 kHz, mono, 16-bit little-endian.
PCM16_RATE = 24_000
PCM16_CHANNELS = 1
PCM16_BITS = 16


#: Layer III bitrates in kbit/s by index, for MPEG-1 and for MPEG-2/2.5.
_MP3_KBPS = {
    3: (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320),
    2: (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160),
}
#: Sample rates by version (3 = MPEG-1, 2 = MPEG-2, 0 = MPEG-2.5) and index.
_MP3_RATES = {3: (44100, 48000, 32000), 2: (22050, 24000, 16000), 0: (11025, 12000, 8000)}


def _mp3_frame_length(raw: bytes, at: int) -> int | None:
    """The length of the Layer III frame whose header starts at `at`, or
    None when no valid one does."""
    if len(raw) < at + 4 or raw[at] != 0xFF or raw[at + 1] & 0xE0 != 0xE0:
        return None
    version = raw[at + 1] >> 3 & 3
    layer = raw[at + 1] >> 1 & 3
    rate_index = raw[at + 2] >> 2 & 3
    bitrate_index = raw[at + 2] >> 4
    if version == 1 or layer != 1 or rate_index == 3 or bitrate_index in (0, 15):
        return None
    kbps = _MP3_KBPS[3 if version == 3 else 2][bitrate_index]
    rate = _MP3_RATES[version][rate_index]
    padding = raw[at + 2] >> 1 & 1
    return (144 if version == 3 else 72) * kbps * 1000 // rate + padding


def detect_format(raw: bytes) -> AudioOutputFormat:
    """What the bytes are: an MP3, a WAV (RIFF/WAVE), or else headerless
    `pcm16` samples.

    **An MP3 without an ID3 tag must show two frames in a row.** A lone
    frame sync is eleven set bits, and `pcm16` meets that by accident: a
    first sample of -1 is the bytes `FF FF`. So the header must be a valid
    Layer III one AND another must begin where it says the frame ends --
    which samples of speech do not arrange.
    """
    if raw[:3] == b"ID3":
        return AudioOutputFormat.mp3
    length = _mp3_frame_length(raw, 0)
    if length is not None and _mp3_frame_length(raw, length) is not None:
        return AudioOutputFormat.mp3
    if raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        return AudioOutputFormat.wav
    return AudioOutputFormat.pcm16


def wav_from_pcm16(samples: bytes) -> bytes:
    """`pcm16` samples as a WAV file: a 44-byte RIFF header, then the data."""
    block = PCM16_CHANNELS * PCM16_BITS // 8
    return (
        b"RIFF"
        + struct.pack("<I", 36 + len(samples))
        + b"WAVEfmt "
        + struct.pack(
            "<IHHIIHH", 16, 1, PCM16_CHANNELS, PCM16_RATE, PCM16_RATE * block, block, PCM16_BITS
        )
        + b"data"
        + struct.pack("<I", len(samples))
        + samples
    )


class AudioAssembly:
    """One answer's audio, fragment by fragment.

    `feed` turns each upstream `delta.audio` into the fragment a streamed
    caller is sent; `result` is the whole clip for a non-streamed one.
    Only an assembling caller keeps the bytes: a streamed one has already
    been sent them, and a copy on the terminal frame would send them twice.
    """

    def __init__(self, *, keep: bool) -> None:
        self._keep = keep
        self._data: list[bytes] = []
        self._transcript: list[str] = []
        self.id: str | None = None
        self.expires_at: int | None = None
        self.format: AudioOutputFormat | None = None

    @property
    def heard(self) -> bool:
        """Whether any audio or transcript arrived at all."""
        return self.format is not None or bool(self._transcript)

    def feed(self, raw: Any) -> AudioDelta | None:
        """The fragment to forward for one upstream `delta.audio`, or None
        when it carries nothing."""
        if not isinstance(raw, dict):
            return None
        out = AudioDelta()
        if isinstance(raw.get("id"), str) and self.id is None:
            self.id = out.id = raw["id"]
        expires = raw.get("expires_at")
        if isinstance(expires, int) and not isinstance(expires, bool) and self.expires_at is None:
            self.expires_at = out.expiresAt = expires
        if isinstance(raw.get("transcript"), str) and raw["transcript"]:
            self._transcript.append(raw["transcript"])
            out.transcript = raw["transcript"]
        data = raw.get("data")
        if isinstance(data, str) and data:
            try:
                decoded = base64.b64decode(data, validate=True)
            except (ValueError, binascii.Error):
                # A fragment that is not base64 is not audio anyone can
                # play; forwarding it would hand the caller a clip that
                # fails in their player instead of here.
                raise ValueError("the backend sent an audio fragment that is not base64") from None
            if self.format is None:
                self.format = out.format = detect_format(decoded)
            if self._keep:
                self._data.append(decoded)
            out.data = data
        if out.model_dump(exclude_none=True):
            return out
        return None

    def result(self, asked: AudioOutputFormat) -> GeneratedAudio | None:
        """The whole clip, labelled with what it is. A `pcm16` stream asked
        for as `wav` gets the header; anything else is returned as sent."""
        if not self.heard:
            return None
        raw = b"".join(self._data)
        fmt = self.format or AudioOutputFormat.pcm16
        if fmt is AudioOutputFormat.pcm16 and asked is AudioOutputFormat.wav:
            raw, fmt = wav_from_pcm16(raw), AudioOutputFormat.wav
        return GeneratedAudio(
            data=base64.b64encode(raw).decode("ascii"),
            format=fmt,
            id=self.id,
            transcript="".join(self._transcript) or None,
            expiresAt=self.expires_at,
        )
