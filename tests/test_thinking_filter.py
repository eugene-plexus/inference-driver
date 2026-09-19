"""The thinking filter, batch and streaming, and the invariant that binds them.

R3 item 1, review §6.2 #21. **The only finding on the release list that
returns a wrong answer rather than an error**, and the reverse of this
project's usual failure: `ThinkingFilter`'s own docstring says *"The
invariant worth testing, and tested"* and there was no test file at all.
A prose claim of coverage standing in for the coverage.

What it got wrong. The batch stripper matches `<think\\b[^>]*>`, so
`<thinking>` is not a think tag to it and the whole span survives. The
streaming filter matched a bare `<think` with no word boundary, so
`<thinking>` *opened* a block -- and then waited for a literal
`</think>`, which `</thinking>` never provides. The block never closed,
every later chunk was swallowed as thinking, and `flush` dropped the
remainder on the floor. A Claude-distill fine-tune with
`thinkingMode: off` streamed an empty 200.

So the two paths disagreed about what a thinking tag even is, and the
streaming one lost the answer. Both now know both spellings, and the
invariant below is what keeps them from drifting apart again.
"""

from __future__ import annotations

import pytest

from eugene_plexus_inference_driver.engines._thinking import (
    ThinkingFilter,
    strip_thinking_blocks,
)

# Chunkings to run every invariant case under. `1` is the cruel one: it
# splits every tag across as many events as it has characters, which is
# what a real token stream does to `</think>`.
CHUNK_SIZES = [1, 2, 3, 5, 7, 11, 1000]


def stream(text: str, size: int) -> str:
    """The filter's whole output for `text` delivered `size` at a time."""
    filt = ThinkingFilter()
    out = [filt.feed(text[i : i + size]) for i in range(0, len(text), size)]
    out.append(filt.flush())
    return "".join(out)


# --------------------------------------------------------------------------- #
# The reproduction
# --------------------------------------------------------------------------- #

CLAUDE_DISTILL = "<thinking>secret plan</thinking>The answer is 4."


@pytest.mark.parametrize("size", CHUNK_SIZES)
def test_a_thinking_tag_does_not_swallow_the_answer(size: int) -> None:
    """**The finding.** Streamed `''` at every chunking, for everybody
    whose model spells the tag `<thinking>`."""
    assert stream(CLAUDE_DISTILL, size) == "The answer is 4."


def test_the_batch_stripper_agrees_about_thinking(size: int = 0) -> None:
    """The other half: batch kept the whole span, tag and all, so the
    two paths disagreed about what a thinking tag is."""
    assert strip_thinking_blocks(CLAUDE_DISTILL) == "The answer is 4."


# --------------------------------------------------------------------------- #
# The invariant the docstring claimed
# --------------------------------------------------------------------------- #

CASES = [
    "",
    "plain text with no tags at all",
    "<think>hidden</think>visible",
    "<thinking>hidden</thinking>visible",
    "before <think>hidden</think> after",
    "before <thinking>hidden</thinking> after",
    '<think foo="bar">hidden</think>visible',
    '<thinking signature="abc">hidden</thinking>visible',
    "<THINK>hidden</THINK>visible",
    "<Thinking>hidden</Thinking>visible",
    "a<think>one</think>b<think>two</think>c",
    "a<thinking>one</thinking>b<thinking>two</thinking>c",
    # A `<` that never becomes a tag, and a near-miss name.
    "5 < 6 and 7 > 6",
    "<thinker>not a thinking tag</thinker>kept",
    "<thin",
    # Mixed spellings in one answer, which a router across two models
    # can genuinely produce in one conversation.
    "<think>a</think>mid<thinking>b</thinking>end",
    # **The four below were added because the sabotage pass escaped on
    # them**, and each named a missing case rather than a weak fix.
    # An unterminated block: nothing compared the two paths here, so
    # batch could have gone back to leaking it while streaming dropped
    # it and every test stayed green.
    "visible<think>cut off right here",
    "visible<thinking>cut off right here",
    # A buffer that ends mid-name. `<think` IS a tag start and is
    # dropped; `<thinke` is not one and is real output. Nothing reached
    # the `name` state at end of stream at all, so `flush` could have
    # dropped both.
    "<think",
    "<thinke",
    # A rejected name followed by a real tag in the same buffer. A
    # scanner that gave up on the whole buffer after `<thinker>` rather
    # than advancing one character would pass every other case here.
    "<thinker>x</thinker><think>drop</think>keep",
]


@pytest.mark.parametrize("text", CASES)
@pytest.mark.parametrize("size", CHUNK_SIZES)
def test_streaming_equals_batch_for_every_chunking(text: str, size: int) -> None:
    """The invariant `ThinkingFilter`'s docstring claims, finally asserted.

    Modulo the outer `.strip()`, which a streamer cannot apply to text
    it has already sent -- so the comparison strips both sides.
    """
    assert stream(text, size).strip() == strip_thinking_blocks(text).strip()


# --------------------------------------------------------------------------- #
# The properties each path has to hold on its own
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("size", CHUNK_SIZES)
def test_a_tag_split_across_chunks_is_still_a_tag(size: int) -> None:
    assert stream("keep<think>drop</think>keep2", size) == "keepkeep2"


@pytest.mark.parametrize("size", CHUNK_SIZES)
def test_a_close_tag_must_match_the_open_tag(size: int) -> None:
    """`</think>` does not close `<thinking>`.

    This is the exact mechanism of the finding: the old filter opened on
    `<thinking>` and then waited for a literal `</think>` that never
    came. Asserting it from the other side -- a genuinely unmatched
    pair -- keeps the fix from becoming "close on anything that looks
    roughly right", which would end a block early and leak reasoning.
    """
    # Unterminated as far as the filter is concerned, so everything from
    # the open tag on is thinking and is dropped.
    assert stream("a<thinking>b</think>c", size) == "a"


@pytest.mark.parametrize("size", CHUNK_SIZES)
def test_a_near_miss_tag_name_is_left_alone(size: int) -> None:
    """`<thinker>` is not a thinking tag, and swallowing it would be the
    same class of bug one name over."""
    assert "not a thinking tag" in stream("<thinker>not a thinking tag</thinker>", size)


@pytest.mark.parametrize("size", CHUNK_SIZES)
def test_an_unterminated_block_is_dropped_rather_than_leaked(size: int) -> None:
    """A model that stops mid-thought owes the client nothing; what it
    was writing was reasoning, and `thinkingMode: off` says not to show
    it."""
    assert stream("visible<think>cut off right here", size) == "visible"


@pytest.mark.parametrize("size", CHUNK_SIZES)
def test_text_that_merely_starts_like_a_tag_is_emitted(size: int) -> None:
    """`flush` exists for this: a tail withheld in case it became a tag,
    which the end of the stream proves it never was."""
    assert stream("the answer is <thin", size) == "the answer is <thin"


@pytest.mark.parametrize("size", CHUNK_SIZES)
def test_nothing_is_withheld_forever_on_ordinary_text(size: int) -> None:
    text = "a perfectly ordinary answer with < and > in it"
    assert stream(text, size) == text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("visible<think>cut off right here", "visible"),
        ("visible<thinking>cut off right here", "visible"),
        ("<think", ""),
        ("<thinke", "<thinke"),
    ],
)
def test_the_batch_path_decides_an_unterminated_tag_the_same_way(text: str, expected: str) -> None:
    """**A sabotage escaped here**: nothing asserted what the BATCH path
    does with an unterminated opening tag, so it could have gone back to
    leaking one while streaming dropped it, with every test green.

    Stripped, on both paths, and that is a decision: what follows an
    unterminated `<think>` is reasoning, and `thinkingMode: off` says
    not to show reasoning. `<thinke` is not a tag start at all, so it is
    real output either way -- which is the line the escape was hiding.
    """
    assert strip_thinking_blocks(text) == expected


@pytest.mark.parametrize("size", CHUNK_SIZES)
def test_a_real_tag_after_a_rejected_name_is_still_found(size: int) -> None:
    """The third escape. A scanner that abandoned the whole buffer after
    `<thinker>` instead of advancing one character passed every other
    case in this file."""
    assert stream("<thinker>x</thinker><think>drop</think>keep", size) == (
        "<thinker>x</thinker>keep"
    )
