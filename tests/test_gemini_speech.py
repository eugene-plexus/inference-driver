import io
import math
import struct
from pathlib import Path

import httpx
import pytest
import respx

from .test_gemini import BASE, TTS, _answer, _start


@respx.mock
@pytest.mark.parametrize(
    ("fmt", "mime", "codec"),
    [
        ("mp3", "audio/mpeg", "mp3"),
        ("opus", "audio/ogg", "opus"),
        ("aac", "audio/aac", "aac"),
        ("flac", "audio/flac", "flac"),
    ],
)
def test_mp3_and_opus_are_encoded(tmp_path: Path, fmt: str, mime: str, codec: str) -> None:
    import base64

    pcm = b"".join(
        struct.pack("<h", int(10000 * math.sin(i * 2 * math.pi * 440 / 24000)))
        for i in range(24000)
    )
    client = _start(tmp_path)
    respx.post(f"{BASE}/models/{TTS}:generateContent").mock(
        return_value=httpx.Response(
            200,
            json=_answer(
                [
                    {
                        "inlineData": {
                            "mimeType": "audio/L16;rate=24000",
                            "data": base64.b64encode(pcm).decode(),
                        }
                    }
                ]
            ),
        )
    )
    with client:
        response = client.post(
            "/v1/speak", json={"model": TTS, "input": "Hello", "voice": "Kore", "format": fmt}
        )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == mime
    import av

    with av.open(io.BytesIO(response.content)) as audio:
        assert codec in audio.streams.audio[0].codec_context.name
        frames = list(audio.decode(audio=0))
        duration = sum(frame.samples / frame.sample_rate for frame in frames)
    assert 0.9 < duration < 1.2
