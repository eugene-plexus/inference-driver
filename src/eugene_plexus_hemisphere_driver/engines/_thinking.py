"""Translate operator `thinkingMode` config into a model-side directive.

The engines we ship in v0.2 don't have a uniform API for native
reasoning-budget control (Anthropic Messages API has
`thinking.budget_tokens`, OpenAI o-series has `reasoning_effort`, but
the CLI engines and OpenAI-compat HTTP route only see a chat-message
list). For v0.2.x the universal mechanism is prompt-side instruction:
we mutate the system message before dispatch and let the model honor
the directive as it can.

Models that emit `<think>...</think>` inline tags (DeepSeek R1, Qwen
QwQ, Kimi K2, MiniMax-M2 reasoning variants) respond to "do not
output thinking blocks". Models without thinking modes ignore the
extra sentence harmlessly. Native-thinking APIs (when we wire them
in v0.3+) will read the same `thinkingMode` field and translate to
their per-request budget knob instead of mutating the prompt.

The directive is appended to the first system message — if the
orchestrator didn't supply one, we prepend a system message carrying
just the directive.
"""

from __future__ import annotations

import re

from .._generated.models import Message, Role

# Strips <think>...</think> spans from a response content string.
# Used by `strip_thinking_blocks` to clean up reasoning-model output
# when the operator chose `thinkingMode=off` but the model emitted
# thinking tags anyway (MiniMax-M2, DeepSeek-R1, Qwen QwQ and other
# Chinese reasoning models bake `<think>` into their chat template at
# the server, so prompt-side suppression alone is unreliable).
#
# DOTALL so the pattern crosses newlines (thinking blocks are usually
# multi-line). IGNORECASE in case a model emits <Think> or <THINK>.
_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)

THINKING_MODE_INSTRUCTIONS: dict[str, str] = {
    # `auto` (or None / unrecognized): no directive — defer to the
    # model's natural behavior. Useful when the operator wants their
    # reasoning model to think the way it normally does.
    "off": (
        "Do not output any thinking blocks, scratchpad reasoning, or "
        "<think>...</think> tags. Respond with your final answer "
        "directly. Do not show your reasoning process."
    ),
    "low": (
        "Use brief internal reasoning when essential, but keep any "
        "<think>...</think> blocks short — a sentence or two at most."
    ),
    "medium": (
        "Use moderate internal reasoning before answering when helpful. "
        "A few sentences of <think>...</think> are fine; keep it focused."
    ),
    "high": (
        "Use thorough internal reasoning before answering. Explore "
        "multiple angles in <think>...</think> blocks if it helps "
        "produce a well-reasoned response."
    ),
}


def apply_thinking_mode(messages: list[Message], thinking_mode: str | None) -> list[Message]:
    """Return a copy of `messages` with the thinking directive merged in.

    `thinking_mode` of `"auto"`, `None`, or an unknown value returns
    the input unchanged so engines don't have to special-case the
    operator-default path.
    """
    if not thinking_mode or thinking_mode == "auto":
        return list(messages)
    directive = THINKING_MODE_INSTRUCTIONS.get(thinking_mode)
    if not directive:
        return list(messages)

    out = list(messages)
    # Append the directive to the FIRST system message (the orchestrator's
    # bicameral preamble lives there). If there's no system message, push
    # one onto the front carrying just the directive.
    for i, msg in enumerate(out):
        if msg.role == Role.system:
            combined = f"{msg.content.rstrip()}\n\n{directive}"
            out[i] = msg.model_copy(update={"content": combined})
            return out
    out.insert(0, Message(role=Role.system, content=directive))
    return out


def strip_thinking_blocks(text: str) -> str:
    """Remove `<think>...</think>` spans from a model response.

    Called by every engine when `thinkingMode == "off"`. Defensive
    cleanup: the system-prompt directive in `apply_thinking_mode`
    tries to stop emission upstream, but Chinese reasoning models
    (MiniMax-M2, DeepSeek-R1, Qwen QwQ, Kimi K2-Thinking) bake
    thinking tags into the chat template at the server and ignore
    polite English requests. The regex strip works regardless.

    Leaves any thinking text intact when the operator chose `auto`,
    `low`, `medium`, or `high` — those modes deliberately want
    reasoning visible.
    """
    cleaned = _THINK_BLOCK_RE.sub("", text)
    # Collapse the blank-line gap a stripped block leaves behind so
    # the visible response doesn't start with leading whitespace.
    return cleaned.strip()


__all__ = ["THINKING_MODE_INSTRUCTIONS", "apply_thinking_mode", "strip_thinking_blocks"]
