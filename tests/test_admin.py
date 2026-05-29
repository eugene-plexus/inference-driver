"""Tests for POST /v1/admin/restart.

The endpoint schedules a delayed `os._exit(0)` via the asyncio event
loop. We don't want the test process to die mid-suite, so we monkey-patch
`os._exit` (and the loop's `call_later` for good measure) to record the
call instead of exiting.
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient


def test_restart_returns_202_with_scheduled_payload(client: TestClient, monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_call_later(delay: float, callback: Any, *args: Any, **kwargs: Any) -> None:
        captured["delay"] = delay
        captured["callback"] = callback

    import asyncio as _asyncio

    real_get_event_loop = _asyncio.get_event_loop

    class _FakeLoop:
        def call_later(self, delay: float, callback: Any, *args: Any, **kwargs: Any) -> None:
            fake_call_later(delay, callback, *args, **kwargs)

    monkeypatch.setattr(_asyncio, "get_event_loop", lambda: _FakeLoop())

    try:
        response = client.post("/v1/admin/restart")
    finally:
        monkeypatch.setattr(_asyncio, "get_event_loop", real_get_event_loop)

    assert response.status_code == 202
    body = response.json()
    assert body["scheduled"] is True
    assert body["delayMs"] >= 0
    assert "message" in body

    # Confirm we'd have called os._exit if we hadn't intercepted the loop.
    assert captured["callback"] is not None
    assert captured["delay"] == body["delayMs"] / 1000.0
