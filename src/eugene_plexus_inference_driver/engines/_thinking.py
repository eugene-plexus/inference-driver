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
from ._prompt import text_content

# **Two spellings, and R3 is where that was learned.** `<think>` is what
# the DeepSeek/Qwen/MiniMax family emits; `<thinking>` is what
# Claude-distill fine-tunes emit, and before R3 the two halves of this
# module disagreed about whether the second one was a tag at all. The
# batch pattern was `<think\b…`, so `<thinking>` was ordinary text and
# survived whole; the streaming filter matched a bare `<think` with no
# boundary, opened a block on it, then waited for a literal `</think>`
# that `</thinking>` never provides -- so it swallowed the rest of the
# answer and `flush` dropped it. `thinkingMode: off` streamed an empty
# 200. Review §6.2 #21; `tests/test_thinking_filter.py`.
#
# `\b` after the name is what keeps `<thinker>` out, and the `\1`
# backreference is what stops `</think>` closing a `<thinking>` block --
# closing on the wrong tag would end the span early and leak the
# reasoning this exists to hide.
#
# DOTALL so the pattern crosses newlines (thinking blocks are usually
# multi-line). IGNORECASE in case a model emits <Think> or <THINK>.
_THINK_BLOCK_RE = re.compile(r"<(think(?:ing)?)\b[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)

# An opening tag with no close, which is what a model that stopped
# mid-thought leaves behind. **Stripped rather than kept**, and that is a
# decision rather than a detail: what follows an unterminated `<think>`
# is reasoning, and `thinkingMode: off` says not to show reasoning. The
# alternative -- show it, tag and all -- hands the operator the exact
# output they turned the switch off to avoid.
#
# It also has to be stripped here for the streaming filter to be able to
# agree with this function at all, since a streamer that has not sent
# the text yet can still withhold it. Two paths that disagree is the
# defect this whole module was in.
_THINK_TAIL_RE = re.compile(r"<think(?:ing)?\b.*$", re.DOTALL | re.IGNORECASE)

#: Tag names this module treats as thinking. Longest first: a scanner
#: that tried `think` before `thinking` would decide the name ended at
#: the `i`.
_THINK_NAMES = ("thinking", "think")

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
    # `content` is nullable since tool calling landed: an assistant turn
    # that only calls a tool carries no text. A system message with none
    # is not a thing anyone sends, but the type now says it could be.
    for i, msg in enumerate(out):
        if msg.role == Role.system:
            combined = f"{text_content(msg.content).rstrip()}\n\n{directive}"
            out[i] = msg.model_copy(update={"content": combined})
            return out
    out.insert(0, Message(role=Role.system, content=directive))
    return out


def strip_thinking_blocks(text: str) -> str:
    """Remove `<think>`/`<thinking>` spans from a model response.

    Called by every engine when `thinkingMode == "off"`. Defensive
    cleanup: the system-prompt directive in `apply_thinking_mode`
    tries to stop emission upstream, but reasoning models
    (MiniMax-M2, DeepSeek-R1, Qwen QwQ, Kimi K2-Thinking, and the
    Claude-distill fine-tunes that spell it `<thinking>`) bake
    thinking tags into the chat template at the server and ignore
    polite English requests. The regexes work regardless.

    Complete blocks go first, then whatever unterminated opening tag is
    left -- in that order, because the tail pattern is greedy to the end
    of the string and would otherwise eat a later, perfectly good
    answer that happened to follow a closed block.

    Leaves any thinking text intact when the operator chose `auto`,
    `low`, `medium`, or `high` — those modes deliberately want
    reasoning visible.
    """
    cleaned = _THINK_TAIL_RE.sub("", _THINK_BLOCK_RE.sub("", text))
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
    might yet turn out to be a tag. Four things it must get right, and
    a naive "strip each chunk" gets all four wrong:

      * **Tags split across chunks.** Upstream may deliver `<th` and
        `ink>` in separate SSE events, so the filter withholds any tail
        that is still a viable prefix of an opening tag.
      * **Attributes.** The batch pattern is `[^>]*`, so an opening tag
        can be arbitrarily long; once the name is read the filter waits
        for `>` rather than assuming a fixed width.
      * **Which name it is.** `<think` may still turn out to be
        `<thinking`, or `<thinker`, which is not a thinking tag at all.
        The name is read to its end before anything is decided.
      * **Closing on the matching name.** `</think>` does not close
        `<thinking>`. Before R3 it was assumed to, and the block
        therefore never closed: every later chunk was swallowed as
        reasoning and `flush` dropped the answer on the floor.

    `feed` returns the text safe to emit now, which may be empty.
    `flush` returns whatever was still withheld at end of stream.

    The invariant, and it is asserted now rather than merely claimed:
    for any chunking of any input, the concatenation of every `feed`
    plus `flush` equals `strip_thinking_blocks` of the whole -- modulo
    the outer `.strip()`, which a streamer cannot apply to text it has
    already sent. `tests/test_thinking_filter.py` drives it over seven
    chunkings, of which `1` is the one that matters: it splits every
    tag across as many events as it has characters.
    """

    _OPEN = "<think"

    def __init__(self) -> None:
        self._buf = ""
        #: None  = outside a block
        #: "name"    = seen `<think`, reading the rest of the tag name
        #: "opening" = it IS a thinking tag, consuming to `>`
        #: "inside"  = within the block, waiting for the matching close
        self._state: str | None = None
        #: Which spelling opened the current block, so the close tag can
        #: be required to match it.
        self._name = ""

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
                self._state = "name"
            elif self._state == "name":
                name, complete = self._read_name(self._buf)
                if not complete:
                    # The name runs to the end of what we have; it may
                    # yet become `thinking`. Withhold and wait.
                    break
                if name.lower() not in _THINK_NAMES:
                    # `<thinker>` and friends. Emit the `<` and rescan
                    # from the next character, so a real tag starting
                    # later in the same buffer is still found.
                    out.append(self._buf[0])
                    self._buf = self._buf[1:]
                    self._state = None
                    continue
                self._name = name.lower()
                self._state = "opening"
            elif self._state == "opening":
                end = self._buf.find(">")
                if end == -1:
                    # Still inside the opening tag; withhold everything.
                    break
                self._buf = self._buf[end + 1 :]
                self._state = "inside"
            else:
                close = f"</{self._name}>"
                end = self._buf.lower().find(close)
                if end == -1:
                    # Inside the block. Drop all but a tail that might be
                    # a partial closing tag -- dropping that too would
                    # lose the boundary and leak the rest of the answer.
                    keep = min(len(self._buf), len(close) - 1)
                    self._buf = self._buf[len(self._buf) - keep :] if keep else ""
                    break
                self._buf = self._buf[end + len(close) :]
                self._state = None
                self._name = ""
        return "".join(out)

    def flush(self) -> str:
        """Whatever is still withheld, decided the way batch decides it.

        `opening` and `inside` are an unterminated block: reasoning the
        operator asked not to see, dropped. `name` is undecided, and is
        dropped only when what was read is exactly a thinking name --
        `<think` at the very end is a tag start, `<thinke` is not and is
        real output the client is owed. That split is not a nicety; it
        is what makes this function agree with `_THINK_TAIL_RE`.
        """
        buf, state = self._buf, self._state
        self._buf, self._state, self._name = "", None, ""
        if state in {"opening", "inside"}:
            return ""
        if state == "name":
            read, _ = self._read_name(buf)
            return "" if read.lower() in _THINK_NAMES else buf
        return buf

    @classmethod
    def _find_open(cls, text: str) -> int | None:
        """Index of a complete-enough `<think` marker, or None."""
        lowered = text.lower()
        index = lowered.find(cls._OPEN)
        return index if index != -1 else None

    @staticmethod
    def _read_name(text: str) -> tuple[str, bool]:
        """The tag name after a leading `<`, and whether it ended.

        `complete` is False when the name runs to the end of the buffer,
        which is the case that must wait rather than decide: `<think` is
        not yet distinguishable from `<thinking` or `<thinker`.
        """
        i = 1
        while i < len(text) and (text[i].isalnum() or text[i] == "_"):
            i += 1
        return text[1:i], i < len(text)

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
