"""The transcription helpers (P3b): the ASR preamble and a backend's usage.

The answers are the ones measured on 2026-09-28: `llama-server` b11235 with
Qwen3-ASR, and OpenRouter's Whisper and gpt-4o-mini-transcribe.
"""

from __future__ import annotations

import pytest

from eugene_plexus_inference_driver.transcription import response_from, split_preamble, usage_from

LLAMA = {
    "type": "transcript.text.done",
    "text": "language English<asr_text>The quick brown fox jumps over the lazy dog.",
    "usage": {"type": "tokens", "input_tokens": 61, "output_tokens": 14, "total_tokens": 75},
}
WHISPER = {
    "text": " The quick brown fox jumps over the lazy dog.",
    "usage": {"seconds": 3.5, "cost": 0.0001},
}


def test_llama_servers_preamble_is_parsed_into_the_language_and_the_text() -> None:
    answer = response_from(LLAMA, model_id="asr", latency_ms=300)
    assert answer.text == "The quick brown fox jumps over the lazy dog."
    assert answer.language == "english"
    assert answer.usage is not None
    assert (answer.usage.inputTokens, answer.usage.outputTokens, answer.usage.totalTokens) == (
        61,
        14,
        75,
    )


def test_a_text_without_the_preamble_is_left_as_it_is() -> None:
    answer = response_from(WHISPER, model_id="openai/whisper-large-v3-turbo", latency_ms=6200)
    assert answer.text == WHISPER["text"]
    assert answer.language is None
    assert answer.usage is not None and answer.usage.seconds == 3.5


@pytest.mark.parametrize(
    "text",
    [
        "The language English<asr_text> was said",
        "My language is English. <asr_text>",
        "language <asr_text>nothing named",
    ],
)
def test_only_a_leading_preamble_is_one(text: str) -> None:
    assert split_preamble(text) == (None, text)


def test_a_reported_language_wins_over_a_parsed_one() -> None:
    body = {**LLAMA, "language": "en"}
    assert response_from(body, model_id=None, latency_ms=1).language == "en"


def test_verbose_fields_are_carried_as_sent() -> None:
    body = {
        "text": "hi",
        "language": "english",
        "duration": 1.25,
        "segments": [{"id": 0, "start": 0.0, "end": 1.2, "text": "hi"}],
        "words": [{"word": "hi", "start": 0.1, "end": 0.4}],
    }
    answer = response_from(body, model_id=None, latency_ms=1)
    assert answer.duration == 1.25
    assert answer.segments == body["segments"] and answer.words == body["words"]


def test_openais_chat_style_token_names_are_read_too() -> None:
    usage = usage_from({"usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12}})
    assert usage is not None and (usage.inputTokens, usage.outputTokens) == (9, 3)


@pytest.mark.parametrize(
    "body", [{}, {"usage": None}, {"usage": {"type": "tokens"}}, {"usage": {"seconds": True}}]
)
def test_no_countable_usage_is_none(body: dict) -> None:
    assert usage_from(body) is None


@pytest.mark.parametrize("body", [{}, {"text": 3}, [], "text"])
def test_an_answer_without_text_is_refused(body: object) -> None:
    with pytest.raises(ValueError):
        response_from(body, model_id=None, latency_ms=1)
