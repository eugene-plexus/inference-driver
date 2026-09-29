"""Raw completion (P6, 2026-09-28): a prompt continued as written, no chat
template, and fill-in-the-middle where a model does it.

Each engine is asked the way it continues raw text, all measured or read in
its source:

* **`llama-server`**: `/v1/completions` continues the prompt with its special
  tokens parsed, so a client-rendered fill-in-the-middle prompt works
  (measured on b11235). Its `/v1/completions` **ignores `suffix`**, answering
  exactly as without it (measured), so a suffix goes to its native `/infill`,
  which fills the middle with the model's own tokens.
* **vLLM**: `/v1/completions` passes the prompt untemplated; it refuses a
  suffix for every model but DeepSeek V4, so none is sent (read).
* **Ollama**: its `/v1/completions` wraps a prompt without a suffix in the
  model's chat template (read), which would turn a rendered FIM prompt into a
  chat turn. `/api/generate` with `raw: true` continues it as written; with a
  suffix, `raw` is left off so the template's `.Suffix` fills the middle, for
  a model with the `insert` capability.
"""

from __future__ import annotations

import json
from typing import Any

from ._generated.models import FinishReason, GenerateRequest, Usage

#: The settings a raw completion carries. Anything else a caller set is
#: refused rather than dropped (A2).
CARRIED = frozenset(
    {
        "maxTokens",
        "temperature",
        "topP",
        "stop",
        "seed",
        "frequencyPenalty",
        "presencePenalty",
        "logitBias",
    }
)
#: Ollama's `/api/generate` has no logit bias.
OLLAMA_CARRIED = CARRIED - {"logitBias"}


class CompletionRefusal(ValueError):
    """A completion this driver refuses before anything is sent."""


def refuse_uncarried(request: GenerateRequest, carried: frozenset[str]) -> None:
    extra = sorted(set(request.callerSettings or []) - carried)
    if extra:
        raise CompletionRefusal(
            f"{', '.join(extra)}: not carried by a raw completion on this backend; remove "
            + ("it" if len(extra) == 1 else "them")
        )


def _stop(request: GenerateRequest) -> list[str] | None:
    stop = request.stop
    if stop is None:
        return None
    value = getattr(stop, "root", stop)
    return [value] if isinstance(value, str) else list(value)


def _logit_bias(request: GenerateRequest) -> dict[str, float] | None:
    bias = request.logitBias
    if bias is None:
        return None
    value = getattr(bias, "root", bias)
    return {str(k): float(v) for k, v in dict(value).items()}


def openai_payload(request: GenerateRequest, upstream: str, *, stream: bool) -> dict[str, Any]:
    """`/v1/completions` for `llama-server` and vLLM."""
    assert request.completion is not None
    payload: dict[str, Any] = {"model": upstream, "prompt": request.completion.prompt}
    optional = {
        "max_tokens": request.maxTokens,
        "temperature": request.temperature,
        "top_p": request.topP,
        "stop": _stop(request),
        "seed": request.seed,
        "frequency_penalty": request.frequencyPenalty,
        "presence_penalty": request.presencePenalty,
        "logit_bias": _logit_bias(request),
    }
    payload.update({k: v for k, v in optional.items() if v is not None})
    if stream:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
    return payload


def infill_payload(request: GenerateRequest, *, stream: bool) -> dict[str, Any]:
    """`llama-server`'s `/infill`: the prompt before the cursor, the suffix
    after it, filled with the model's own fill-in-the-middle tokens."""
    assert request.completion is not None
    payload: dict[str, Any] = {
        "input_prefix": request.completion.prompt,
        "input_suffix": request.completion.suffix or "",
    }
    optional = {
        "n_predict": request.maxTokens,
        "temperature": request.temperature,
        "top_p": request.topP,
        "stop": _stop(request),
        "seed": request.seed,
        "frequency_penalty": request.frequencyPenalty,
        "presence_penalty": request.presencePenalty,
        "logit_bias": _logit_bias(request),
    }
    payload.update({k: v for k, v in optional.items() if v is not None})
    if stream:
        payload["stream"] = True
    return payload


def ollama_payload(request: GenerateRequest, upstream: str, *, stream: bool) -> dict[str, Any]:
    """Ollama's `/api/generate`: `raw` for a continuation as written; with a
    suffix, templated, so `.Suffix` fills the middle."""
    assert request.completion is not None
    payload: dict[str, Any] = {
        "model": upstream,
        "prompt": request.completion.prompt,
        "stream": stream,
    }
    if request.completion.suffix is not None:
        payload["suffix"] = request.completion.suffix
    else:
        payload["raw"] = True
    options = {
        "num_predict": request.maxTokens,
        "temperature": request.temperature,
        "top_p": request.topP,
        "stop": _stop(request),
        "seed": request.seed,
        "frequency_penalty": request.frequencyPenalty,
        "presence_penalty": request.presencePenalty,
    }
    options = {k: v for k, v in options.items() if v is not None}
    if options:
        payload["options"] = options
    return payload


def finish(reason: Any) -> FinishReason:
    """`length` when the budget ran out, `stop` for every natural end:
    `/infill`'s `limit` is its word for the first, `eos` and `word` for the
    second (measured); Ollama's `done_reason` says `length` or `stop`."""
    return FinishReason.length if reason in ("length", "limit") else FinishReason.stop


def usage(prompt: Any, completion: Any) -> Usage | None:
    def whole(value: Any) -> int | None:
        return (
            value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
        )

    p, c = whole(prompt), whole(completion)
    if p is None and c is None:
        return None
    return Usage(promptTokens=p, completionTokens=c, totalTokens=(p or 0) + (c or 0))


def openai_answer(body: Any) -> tuple[str, FinishReason, Usage | None]:
    choice = body["choices"][0]
    text = choice.get("text")
    if not isinstance(text, str):
        raise ValueError("the completion answer had no text")
    got = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    return (
        text,
        finish(choice.get("finish_reason")),
        usage(got.get("prompt_tokens"), got.get("completion_tokens")),
    )


def infill_answer(body: Any) -> tuple[str, FinishReason, Usage | None]:
    text = body.get("content") if isinstance(body, dict) else None
    if not isinstance(text, str):
        raise ValueError("the infill answer had no content")
    return (
        text,
        finish(body.get("stop_type")),
        usage(body.get("tokens_evaluated"), body.get("tokens_predicted")),
    )


def ollama_answer(body: Any) -> tuple[str, FinishReason, Usage | None]:
    text = body.get("response") if isinstance(body, dict) else None
    if not isinstance(text, str):
        raise ValueError("the generate answer had no response")
    return (
        text,
        finish(body.get("done_reason")),
        usage(body.get("prompt_eval_count"), body.get("eval_count")),
    )


def sse_json(line: str) -> dict[str, Any] | None:
    """One SSE `data:` line's JSON, or None for anything else ([DONE],
    comments, blanks)."""
    if not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if not data or data == "[DONE]":
        return None
    try:
        value = json.loads(data)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None
