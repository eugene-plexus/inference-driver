"""Each conversation in one llama-server slot, and never a busy one (CB4).

**llama-server's own slot choice keeps one conversation per prompt family**
(b11211, measured 2026-10-02; upstream #22083): it sends a new conversation
to the idle slot most similar to it, so every Claude Code session of one
project shares one slot and each rereads its history. Pinning by `id_slot`
keeps one conversation a slot, and is worth +3.4 points of prompt reuse where
the pool holds a conversation per slot (and loses 1-10 where it does not,
which is why the profile setting is off by default).

**The rule that makes it safe: never name a slot that is busy.** A request
pinned to a slot still serving another wedged llama-server b11211 for up to
30 minutes, its main loop processing nothing (cache-aware-balancing-
measurement.md §4; not upstream). Waiting in front of the engine until the
slot is idle removed it: no stall in 1,224 turns. So a turn whose slot is
busy waits here, and so does a new conversation when no slot is idle; the
engine would have queued both anyway.

Every request that takes a slot goes through this map while pinning is on,
a conversation or not, because one the engine placed itself could be in a
slot the map believes idle.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import OrderedDict
from collections.abc import AsyncIterator


class SlotPins:
    """Conversation -> slot, least recently used out, over `total` slots."""

    def __init__(self, total: int) -> None:
        if total < 1:
            raise ValueError("a pinned engine needs at least one slot")
        self.total = total
        self._owners: OrderedDict[str, int] = OrderedDict()
        self._busy: set[int] = set()
        #: Set, and replaced, whenever a slot is released. The bookkeeping
        #: around it is synchronous on the event loop, so a check and the
        #: claim that follows cannot be split by another request, and a
        #: release needs no await -- a cancelled request (a closed tab)
        #: still frees its slot.
        self._freed: asyncio.Event | None = None

    def _choose(self, key: str | None) -> int | None:
        """The slot this request may have now, or None: it waits."""
        if key is not None and key in self._owners:
            slot = self._owners[key]
            if slot in self._busy:
                # Its own slot is serving something: wait for it. Naming it
                # now is the request that wedged the engine.
                return None
            self._owners.move_to_end(key)
            return slot
        idle = [slot for slot in range(self.total) if slot not in self._busy]
        if not idle:
            return None
        owned = {slot: owner for owner, slot in self._owners.items()}
        unowned = [slot for slot in idle if slot not in owned]
        if unowned:
            slot = unowned[0]
        else:
            # Every idle slot holds a conversation: take the one used least
            # recently, and that conversation starts again elsewhere.
            order = list(self._owners)
            slot = min(idle, key=lambda s: order.index(owned[s]))
            del self._owners[owned[slot]]
        if key is not None:
            self._owners[key] = slot
        return slot

    @contextlib.asynccontextmanager
    async def hold(self, key: str | None) -> AsyncIterator[int]:
        """The slot to name, held busy until the request is done."""
        while (slot := self._choose(key)) is None:
            if self._freed is None:
                self._freed = asyncio.Event()
            await self._freed.wait()
        self._busy.add(slot)
        try:
            yield slot
        finally:
            self._busy.discard(slot)
            if self._freed is not None:
                self._freed.set()
                self._freed = None

    def busy(self) -> frozenset[int]:
        return frozenset(self._busy)

    def owner(self, key: str) -> int | None:
        return self._owners.get(key)
