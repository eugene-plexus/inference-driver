"""Encode Gemini PCM with PyAV's bundled codecs, entirely in memory."""

from __future__ import annotations

import io

import av

from .engines._subprocess import CliError

_ENCODERS = {
    "mp3": ("mp3", "libmp3lame", "fltp", None),
    "opus": ("ogg", "libopus", "s16", 48000),
    "aac": ("adts", "aac", "fltp", None),
    "flac": ("flac", "flac", "s16", None),
}


def encode(wav: bytes, fmt: str) -> bytes:
    container, codec, sample_format, rate = _ENCODERS[fmt]
    result = io.BytesIO()
    try:
        with av.open(io.BytesIO(wav), mode="r") as source:
            original = source.streams.audio[0]
            with av.open(result, "w", format=container) as target:
                stream = target.add_stream(codec, rate=rate or original.rate)
                assert isinstance(stream, av.AudioStream)
                stream.layout = "mono"
                resampler = av.AudioResampler(
                    format=sample_format, layout="mono", rate=rate or original.rate
                )
                for frame in source.decode(audio=0):
                    for converted in resampler.resample(frame):
                        for packet in stream.encode(converted):
                            target.mux(packet)
                for converted in resampler.resample(None):
                    for packet in stream.encode(converted):
                        target.mux(packet)
                for packet in stream.encode(None):
                    target.mux(packet)
    except (av.FFmpegError, ValueError) as exc:
        raise CliError(f"format {fmt}: PyAV could not encode Gemini audio ({exc})") from exc
    return result.getvalue()
