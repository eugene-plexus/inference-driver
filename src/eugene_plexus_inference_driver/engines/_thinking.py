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
gateway didn't supply one, we prepend a system message carrying
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
    # Append the directive to the FIRST system message (the gateway's
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


class ThinkingFilter:
    """`strip_thinking_blocks`, but for text arriving a piece at a time.

    **Why this has to exist.** The batch stripper runs the regex over a
    finished response, which streaming cannot do: a `<think>` block
    forwarded to the client cannot be un-sent. M10 would otherwise have
    made `thinkingMode: off` silently weaker for exactly the clients
    that stream -- the same shape of defect as M8's, where the
    non-streaming path carried the routing envelope and the streaming
    path did not.

    So this is a small state machine that holds back any text which
    might yet turn out to be a tag. Two things it must get right, both
    of which a naive "strip each chunk" gets wrong:

      * **Tags split across chunks.** Upstream may deliver `<th` and
        `ink>` in separate SSE events, so the filter withholds any tail
        that is still a viable prefix of an opening tag.
      * **Attributes.** The batch pattern is `<think[^>]*>`, so an
        opening tag can be arbitrarily long; once `<think` is seen the
        filter waits for `>` rather than assuming a fixed width.

    `feed` returns the text safe to emit now, which may be empty.
    `flush` returns whatever was still withheld at end of stream --
    non-empty only when the model stopped mid-tag, in which case the
    withheld text was never a tag and is real output the client is owed.

    The invariant worth testing, and tested: for any chunking of any
    input, the concatenation of every `feed` plus `flush` equals
    `strip_thinking_blocks` of the whole -- modulo the outer `.strip()`,
    which a streamer cannot apply to text it has already sent.
    """

    _OPEN = "<think"
    _CLOSE = "</think>"

    def __init__(self) -> None:
        self._buf = ""
        #: None = outside a block; "opening" = seen `<think`, waiting for
        #: `>`; "inside" = within a block, waiting for `</think>`.
        self._state: str | None = None

    def feed(self, text: str) -> str:
        self._buf += text
        out: list[str] = []
        while self._buf:
            if self._state is None:
                index = self._find_open(self._buf)
                if index is None:
                    # No tag can begin in what is left; emit all but a
                    # tail that is still a viable prefix of one.
                    keep = self._viable_prefix_len(self._buf)
                    if keep:
                        out.append(self._buf[:-keep])
                        self._buf = self._buf[-keep:]
                    else:
                        out.append(self._buf)
                        self._buf = ""
                    break
                out.append(self._buf[:index])
                self._buf = self._buf[index:]
                self._state = "opening"
            elif self._state == "opening":
                end = self._buf.find(">")
                if end == -1:
                    # Still inside the opening tag; withhold everything.
                    break
                self._buf = self._buf[end + 1 :]
                self._state = "inside"
            else:
                end = self._buf.lower().find(self._CLOSE)
                if end == -1:
                    # Inside the block. Drop all but a tail that might be
                    # a partial closing tag -- dropping that too would
                    # lose the boundary and leak the rest of the answer.
                    keep = min(len(self._buf), len(self._CLOSE) - 1)
                    self._buf = self._buf[len(self._buf) - keep :] if keep else ""
                    break
                self._buf = self._buf[end + len(self._CLOSE) :]
                self._state = None
        return "".join(out)

    def flush(self) -> str:
        """Whatever is still withheld. Only non-empty on a truncated tag."""
        if self._state is not None:
            # An unterminated block is thinking text, not an answer.
            self._buf = ""
            self._state = None
            return ""
        out, self._buf = self._buf, ""
        return out

    @classmethod
    def _find_open(cls, text: str) -> int | None:
        """Index of a complete-enough `<think` marker, or None."""
        lowered = text.lower()
        index = lowered.find(cls._OPEN)
        return index if index != -1 else None

    @classmethod
    def _viable_prefix_len(cls, text: str) -> int:
        """How many trailing chars could still become an opening tag.

        `<thin` must be withheld because the next chunk may carry `k>`.
        Checked case-insensitively, since the batch pattern is
        IGNORECASE and a model emitting `<THINK>` must not slip past a
        filter that only knows the lowercase spelling.
        """
        lowered = text.lower()
        for size in range(min(len(text), len(cls._OPEN) - 1), 0, -1):
            if cls._OPEN.startswith(lowered[-size:]):
                return size
        return 0


__all__ = [
    "THINKING_MODE_INSTRUCTIONS",
    "ThinkingFilter",
    "apply_thinking_mode",
    "strip_thinking_blocks",
]
