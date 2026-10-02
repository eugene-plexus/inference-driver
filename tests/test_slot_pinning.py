"""Each conversation in one llama-server slot, never a busy one (CB4).

**The safe rule is the point of this file.** A request pinned to a slot still
serving another wedged llama-server b11211 for up to 30 minutes; waiting in
front of the engine instead stalled nothing in 1,224 turns
(`specs/docs/acceptance/cache-aware-balancing-measurement.md` §4).
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
import respx

from eugene_plexus_inference_driver._generated.models import (
    BackendKind,
    GenerateRequest,
    Message,
    Role,
)
from eugene_plexus_inference_driver.engines.openai_compat_http import OpenAiCompatibleHttpEngine
from eugene_plexus_inference_driver.slots import SlotPins

BASE = "http://127.0.0.1:9"


# --- the map ---------------------------------------------------------------------


async def test_a_conversation_keeps_its_slot():
    pins = SlotPins(4)
    async with pins.hold("a") as first:
        pass
    async with pins.hold("b"):
        pass
    async with pins.hold("a") as again:
        pass
    assert again == first


async def test_a_turn_whose_slot_is_busy_waits_for_it():
    """**The safe rule.** Neither the busy slot nor another one: it waits."""
    pins = SlotPins(4)
    release = asyncio.Event()
    got: list[int] = []

    async def first() -> None:
        async with pins.hold("a") as slot:
            got.append(slot)
            await release.wait()

    async def second() -> None:
        async with pins.hold("a") as slot:
            got.append(slot)

    one = asyncio.ensure_future(first())
    await asyncio.sleep(0.01)
    two = asyncio.ensure_future(second())
    await asyncio.sleep(0.05)
    assert len(got) == 1, "the second turn was given a slot while its own was busy"
    release.set()
    await asyncio.gather(one, two)
    assert got[0] == got[1], "it must come back to its own slot, not another"


async def test_a_new_conversation_waits_when_no_slot_is_idle():
    pins = SlotPins(2)
    release = asyncio.Event()

    async def busy(key: str) -> None:
        async with pins.hold(key):
            await release.wait()

    holders = [asyncio.ensure_future(busy(k)) for k in ("a", "b")]
    await asyncio.sleep(0.01)
    third = asyncio.ensure_future(_take(pins, "c"))
    await asyncio.sleep(0.05)
    assert not third.done(), "a new conversation was given a busy slot"
    release.set()
    await asyncio.gather(*holders)
    assert await third in (0, 1)


async def _take(pins: SlotPins, key: str | None) -> int:
    async with pins.hold(key) as slot:
        return slot


async def test_the_least_recently_used_conversation_gives_up_its_slot():
    pins = SlotPins(2)
    a = await _take(pins, "a")
    await _take(pins, "b")
    await _take(pins, "a")  # a is now the most recent
    c = await _take(pins, "c")
    assert pins.owner("b") is None, "b was the least recently used"
    assert c != a and pins.owner("a") == a


async def test_an_idle_slot_nobody_owns_is_taken_first():
    pins = SlotPins(3)
    a = await _take(pins, "a")
    b = await _take(pins, "b")
    assert a != b and pins.owner("a") == a


async def test_a_request_with_no_conversation_takes_a_slot_and_keeps_none():
    pins = SlotPins(2)
    a = await _take(pins, "a")
    slot = await _take(pins, None)
    assert slot != a, "an unowned slot first"
    assert pins.owner("a") == a


async def test_a_cancelled_request_frees_its_slot():
    pins = SlotPins(1)
    started = asyncio.Event()

    async def held() -> None:
        async with pins.hold("a"):
            started.set()
            await asyncio.sleep(30)

    task = asyncio.ensure_future(held())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert pins.busy() == frozenset()
    assert await asyncio.wait_for(_take(pins, "b"), 1) == 0


# --- through the engine --------------------------------------------------------------


def _engine(pinning: bool) -> OpenAiCompatibleHttpEngine:
    return OpenAiCompatibleHttpEngine(
        base_url=BASE,
        api_key=None,
        model_id="m",
        backend_kind=BackendKind.openai_compat_http,
        auth_required=False,
        slot_pinning=pinning,
    )


def _request(key: str | None) -> GenerateRequest:
    return GenerateRequest(messages=[Message(role=Role.user, content="hi")], conversationKey=key)


_ANSWER = {
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
    ],
    "model": "m",
}


@respx.mock
async def test_off_by_default_nothing_is_pinned_and_props_is_not_read():
    props = respx.get(f"{BASE}/props").mock(
        return_value=httpx.Response(200, json={"total_slots": 4})
    )
    chat = respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_ANSWER)
    )
    await _engine(False).generate(_request("k:a"))
    sent = json.loads(chat.calls[0].request.content)
    assert "id_slot" not in sent and not props.called


@respx.mock
async def test_on_each_conversation_names_its_slot_and_the_key_stays_here():
    respx.get(f"{BASE}/props").mock(return_value=httpx.Response(200, json={"total_slots": 4}))
    chat = respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_ANSWER)
    )
    engine = _engine(True)
    for key in ("k:a", "k:b", "k:a"):
        await engine.generate(_request(key))
    sent = [json.loads(c.request.content) for c in chat.calls]
    slots = [s["id_slot"] for s in sent]
    assert slots[0] == slots[2] != slots[1]
    assert all("conversationKey" not in s and "k:a" not in json.dumps(s) for s in sent)


@respx.mock
async def test_through_the_engine_a_busy_slot_is_never_named():
    """Two turns of one conversation at once: the engine sees one request,
    then the other, never two pinned to one slot."""
    respx.get(f"{BASE}/props").mock(return_value=httpx.Response(200, json={"total_slots": 4}))
    gate = asyncio.Event()
    in_flight: list[int] = []
    peak: list[int] = []

    async def answer(request: httpx.Request) -> httpx.Response:
        in_flight.append(json.loads(request.content)["id_slot"])
        peak.append(len(in_flight))
        await gate.wait()
        in_flight.pop()
        return httpx.Response(200, json=_ANSWER)

    respx.post(f"{BASE}/v1/chat/completions").mock(side_effect=answer)
    engine = _engine(True)
    turns = [asyncio.ensure_future(engine.generate(_request("k:a"))) for _ in range(2)]
    await asyncio.sleep(0.05)
    assert peak == [1], "the second turn was sent while its slot was busy"
    gate.set()
    await asyncio.gather(*turns)
    assert max(peak) == 1


@respx.mock
async def test_a_stream_holds_its_slot_until_it_ends():
    respx.get(f"{BASE}/props").mock(return_value=httpx.Response(200, json={"total_slots": 1}))
    frames = (
        b'data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
    )

    def answer(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content).get("stream"):
            return httpx.Response(
                200, content=frames, headers={"content-type": "text/event-stream"}
            )
        return httpx.Response(200, json=_ANSWER)

    chat = respx.post(f"{BASE}/v1/chat/completions").mock(side_effect=answer)
    engine = _engine(True)
    stream = engine.stream(_request("k:a"))
    await anext(stream)
    other = asyncio.ensure_future(engine.generate(_request("k:b")))
    await asyncio.sleep(0.05)
    assert not other.done() and chat.call_count == 1, "the one slot was named twice"
    async for _ in stream:
        pass
    await asyncio.wait_for(other, 1)
    assert json.loads(chat.calls[1].request.content)["id_slot"] == 0


@respx.mock
async def test_an_engine_that_will_not_say_its_slots_is_not_pinned():
    respx.get(f"{BASE}/props").mock(return_value=httpx.Response(500))
    chat = respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_ANSWER)
    )
    await _engine(True).generate(_request("k:a"))
    assert "id_slot" not in json.loads(chat.calls[0].request.content)
