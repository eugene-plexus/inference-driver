"""A learned "no tools" does not outlive its cause (Troy, 2026-10-12, round two).

Round one remembered a backend's refusal of tools "for the engine's life" and said a restart
of the driver would undo it. **The driver does not restart when its runtime does**: the agent
restarts an engine under a driver that goes on. So a llama-server started again with `--jinja`
would have stayed unable to take tools until someone rebuilt the driver, and the report said
otherwise. The contract (specs 6178255, `Capabilities.toolCalling`) now says what the driver
does instead, and these tests are that:

* while `toolCalling` is false for a measured refusal, the backend is asked again every ten
  minutes (an attribute here, `_tools_reask_seconds`, so a test need not wait ten minutes) with
  the smallest request that carries one tool, one token of output;
* an answer clears it, and a refusal in the measured words keeps it;
* anything else (still loading, unreachable, some other error) neither clears nor confirms,
  and is asked again at the next interval;
* a request that carries no tools is not refused for `tool_choice` or `parallel_tool_calls`
  after a flip: they say nothing without tools, so they are left out, once per field in the log.

The backend here is `respx`, answering in the bytes llama-server b10948 sent for `--no-jinja`
(`test_tools_evidence.py`). A real one, restarted in place, is in the commit message.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from eugene_plexus_inference_driver import _http
from eugene_plexus_inference_driver._generated.models import Capabilities, DriverModel
from eugene_plexus_inference_driver.engines import openai_compat_http
from eugene_plexus_inference_driver.engines._subprocess import ToolsRefused
from eugene_plexus_inference_driver.engines.openai_compat_http import (
    OpenAiCompatibleHttpEngine,
    _Target,
)
from eugene_plexus_inference_driver.failures import every
from tests.test_accounts import _wait_for_catalogue
from tests.test_http_client_reuse import _count_constructions
from tests.test_tools_evidence import (
    BAD_TOOLS,
    BASE,
    CHAT,
    LOADING,
    NO_JINJA,
    OK,
    UNPARSEABLE_SCHEMA,
    WORDS,
    _account,
    _ask,
    _body,
    _custom_app,
    _engine,
    _llama,
)

ENGINE_LOG = "eugene_plexus_inference_driver.engines.openai_compat_http"
FAST = 0.02
#: How long a test waits for what it is waiting FOR. Generous by default, so a slow machine does
#: not fail a good build; the sabotage pass shortens it (a broken driver never gets there).
PATIENCE = float(os.environ.get("TOOLS_REASK_TEST_PATIENCE", "5"))
SSE_OK = (
    'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":null}]}\n\n'
    'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
    "data: [DONE]\n\n"
)


def _ok(body: dict[str, Any]) -> httpx.Response:
    """A good answer, in the shape the request asked for."""
    if body.get("stream"):
        return httpx.Response(200, text=SSE_OK, headers={"content-type": "text/event-stream"})
    return httpx.Response(200, json=OK)


class Backend:
    """A llama-server that refuses tools (`--no-jinja`) until it is `fixed` (started again with
    `--jinja`), and records every chat request it was sent."""

    def __init__(self) -> None:
        self.fixed = False
        #: When set, what the backend says to a request that carries tools instead.
        self.answer: Callable[[httpx.Request], httpx.Response] | None = None
        self.requests: list[dict[str, Any]] = []
        self.headers: list[httpx.Headers] = []
        #: Index-aligned with `requests`: when each arrived (`perf_counter`), and the time budget
        #: it carried (httpx's `timeout` extension: connect, read, write and pool seconds).
        self.stamps: list[float] = []
        self.budgets: list[dict[str, float | None]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        self.headers.append(request.headers)
        self.stamps.append(time.perf_counter())
        self.budgets.append(request.extensions["timeout"])
        if not body.get("tools"):
            return _ok(body)
        if self.answer is not None:
            return self.answer(request)
        return _ok(body) if self.fixed else httpx.Response(500, json=NO_JINJA)

    @staticmethod
    def _is_a_question(body: dict[str, Any]) -> bool:
        return bool(body.get("tools")) and body.get("max_tokens") == 1

    @property
    def probes(self) -> list[dict[str, Any]]:
        """The questions asked again: a tool and one token of output."""
        return [b for b in self.requests if self._is_a_question(b)]

    @property
    def probe_stamps(self) -> list[float]:
        """When each question arrived."""
        return [
            at for b, at in zip(self.requests, self.stamps, strict=True) if self._is_a_question(b)
        ]

    @property
    def probe_budgets(self) -> list[dict[str, float | None]]:
        """The time each question was allowed."""
        return [
            t for b, t in zip(self.requests, self.budgets, strict=True) if self._is_a_question(b)
        ]


async def _until(condition: Callable[[], bool], what: str, patience: float = PATIENCE) -> None:
    """Wait for the thing itself, never for a time: fails saying what never happened."""
    deadline = time.perf_counter() + patience
    while not condition():
        if time.perf_counter() > deadline:
            raise AssertionError(f"never happened: {what}")
        await asyncio.sleep(0.005)


def _until_sync(condition: Callable[[], bool], what: str, patience: float = PATIENCE) -> None:
    deadline = time.perf_counter() + patience
    while not condition():
        if time.perf_counter() > deadline:
            raise AssertionError(f"never happened: {what}")
        time.sleep(0.005)


def _fast(engine: OpenAiCompatibleHttpEngine, seconds: float = FAST) -> OpenAiCompatibleHttpEngine:
    engine._tools_reask_seconds = seconds
    return engine


# --------------------------------------------------------------------------- #
# the interval, in a person's words
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("seconds", "words"),
    [
        (600.0, "10 minutes"),
        (60.0, "1 minute"),
        (90.0, "1.5 minutes"),
        (30.0, "30 seconds"),
        (1.0, "1 second"),
        (0.02, "0.02 seconds"),
    ],
)
def test_the_interval_is_said_in_words(seconds: float, words: str) -> None:
    assert every(seconds) == words


def test_ten_minutes_is_what_a_new_engine_asks_at() -> None:
    """Specs 6178255: "asks the backend again every 10 minutes"."""
    assert _engine().tools_reask_seconds == 600.0


# --------------------------------------------------------------------------- #
# an answer clears it, a refusal keeps it
# --------------------------------------------------------------------------- #


@respx.mock
@pytest.mark.anyio
async def test_an_answer_turns_tools_back_on_without_the_driver_restarting(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger=ENGINE_LOG)
    backend = Backend()
    respx.post(CHAT).mock(side_effect=backend)
    engine = _fast(_engine())
    try:
        with pytest.raises(ToolsRefused):
            await engine.generate(_ask())
        assert engine.supports_tool_calling is False
        assert "tools" not in engine.supported_settings

        # A refusal in the measured words keeps it off, and the backend is asked again.
        await _until(lambda: len(backend.probes) >= 3, "three questions to a backend that refuses")
        assert engine.supports_tool_calling is False
        assert engine.tools_refused_words() == WORDS

        # Fixed in place: llama-server started again with `--jinja`. Nothing restarts here.
        backend.fixed = True
        await _until(lambda: engine.supports_tool_calling, "tools back on after an answer")
        assert engine.tools_refused_words() is None
        assert {"tools", "toolChoice", "parallelToolCalls"} <= set(engine.supported_settings)

        # And they are carried again, on the same engine.
        reply = await engine.generate(_ask())
        assert reply.content == "ok"
        assert backend.requests[-1]["tools"][0]["function"]["name"] == "get_weather"
    finally:
        await engine.aclose()

    said = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert sum("Reporting toolCalling false" in m for m in said) == 1
    assert sum("Reporting toolCalling true again" in m for m in said) == 1


@respx.mock
@pytest.mark.anyio
async def test_the_question_is_the_smallest_request_that_carries_one_tool() -> None:
    backend = Backend()
    respx.post(CHAT).mock(side_effect=backend)
    engine = _fast(_engine(api_key="sk-local"))
    try:
        with pytest.raises(ToolsRefused):
            await engine.generate(_ask(maxTokens=512, temperature=0.7))
        await _until(lambda: bool(backend.probes), "a first question")
        probe = backend.probes[0]
        index = backend.requests.index(probe)
        headers = backend.headers[index]
    finally:
        await engine.aclose()

    assert set(probe) == {"model", "messages", "max_tokens", "tools"}
    assert probe["model"] == "qwen" and probe["max_tokens"] == 1
    assert probe["messages"] == [{"role": "user", "content": "Hi"}]
    [tool] = probe["tools"]
    assert tool["type"] == "function" and tool["function"]["name"]
    # Not the caller's request: none of its settings, no stream, no choice among tools.
    assert "stream" not in probe and "tool_choice" not in probe and "temperature" not in probe
    # The operator's key goes with it, like every other call to that backend.
    assert headers["authorization"] == "Bearer sk-local"


@respx.mock
@pytest.mark.anyio
@pytest.mark.parametrize(
    "answer",
    [
        lambda request: httpx.Response(503, json=LOADING),
        lambda request: httpx.Response(500, json=BAD_TOOLS),
        lambda request: httpx.Response(400, json=UNPARSEABLE_SCHEMA),
        lambda request: httpx.Response(429, json={"error": {"message": "slow down"}}),
        lambda request: httpx.Response(502, text="<html>bad gateway</html>"),
        lambda request: httpx.Response(500, text="[" * 4000),
        lambda request: (_ for _ in ()).throw(httpx.ConnectError("connection refused")),
        lambda request: (_ for _ in ()).throw(httpx.ReadTimeout("no answer")),
    ],
    ids=[
        "still-loading",
        "another-500",
        "another-400",
        "rate-limited",
        "a-proxys-html",
        "a-body-nested-too-deep",
        "unreachable",
        "timed-out",
    ],
)
async def test_an_answer_that_says_nothing_of_tools_neither_clears_nor_confirms_and_is_asked_again(
    answer: Callable[[httpx.Request], httpx.Response],
) -> None:
    """Only a refusal in the measured words is a "no", exactly as when it was learned; only an
    answer is a "yes". The backend being down, loading, or unhappy about something else is
    neither, so tools stay as they were and the asking goes on."""
    backend = Backend()
    respx.post(CHAT).mock(side_effect=backend)
    engine = _fast(_engine())
    try:
        with pytest.raises(ToolsRefused):
            await engine.generate(_ask())
        backend.answer = answer
        mark = len(backend.requests)
        await _until(lambda: len(backend.requests) >= mark + 3, "three more questions")
        assert engine.supports_tool_calling is False, "silence is not an answer"
        assert engine.tools_refused_words() == WORDS
        assert engine._tools_watch is not None and not engine._tools_watch.done()

        backend.answer = None
        backend.fixed = True
        await _until(lambda: engine.supports_tool_calling, "tools back on once it answers")
    finally:
        await engine.aclose()


@respx.mock
@pytest.mark.anyio
async def test_the_asking_stops_when_nothing_is_off_and_asks_nothing_more() -> None:
    backend = Backend()
    respx.post(CHAT).mock(side_effect=backend)
    engine = _fast(_engine())
    try:
        with pytest.raises(ToolsRefused):
            await engine.generate(_ask())
        backend.fixed = True
        await _until(lambda: engine.supports_tool_calling, "tools back on")
        watch = engine._tools_watch
        assert watch is not None
        await _until(watch.done, "the asking to end")
        asked = len(backend.requests)
        await asyncio.sleep(FAST * 10)
        assert len(backend.requests) == asked, "nothing is off, so nothing is asked"
    finally:
        await engine.aclose()


@respx.mock
@pytest.mark.anyio
async def test_a_backend_that_refuses_again_after_an_answer_is_learned_and_asked_about_again() -> (
    None
):
    backend = Backend()
    respx.post(CHAT).mock(side_effect=backend)
    engine = _fast(_engine())
    try:
        with pytest.raises(ToolsRefused):
            await engine.generate(_ask())
        backend.fixed = True
        await _until(lambda: engine.supports_tool_calling, "tools back on")
        first = engine._tools_watch
        assert first is not None
        await _until(first.done, "the first asking to end")

        backend.fixed = False  # restarted again with --no-jinja
        with pytest.raises(ToolsRefused):
            await engine.generate(_ask())
        assert engine.supports_tool_calling is False
        mark = len(backend.probes)
        await _until(lambda: len(backend.probes) >= mark + 2, "the second asking")
        assert engine._tools_watch is not first
    finally:
        await engine.aclose()


@respx.mock
@pytest.mark.anyio
async def test_closing_the_engine_stops_the_asking() -> None:
    backend = Backend()
    respx.post(CHAT).mock(side_effect=backend)
    engine = _fast(_engine())
    with pytest.raises(ToolsRefused):
        await engine.generate(_ask())
    await _until(lambda: bool(backend.probes), "a first question")
    await engine.aclose()

    assert engine._tools_watch is None
    assert [t for t in asyncio.all_tasks() if t.get_name() == "tools-again"] == []
    asked = len(backend.requests)
    await asyncio.sleep(FAST * 10)
    assert len(backend.requests) == asked, "a closed engine asks no more"
    await engine.aclose()  # idempotent


@respx.mock
@pytest.mark.anyio
async def test_a_surprise_in_one_round_does_not_end_the_asking(
    caplog: pytest.LogCaptureFixture,
) -> None:
    backend = Backend()
    respx.post(CHAT).mock(side_effect=backend)
    engine = _fast(_engine())
    rounds = [0]
    real = engine._tools_answered

    async def surprising(upstream: str) -> bool | None:
        rounds[0] += 1
        if rounds[0] == 1:
            raise RuntimeError("something nobody planned for")
        return await real(upstream)

    engine._tools_answered = surprising  # type: ignore[method-assign]
    caplog.set_level(logging.WARNING, logger=ENGINE_LOG)
    try:
        with pytest.raises(ToolsRefused):
            await engine.generate(_ask())
        await _until(lambda: rounds[0] >= 3, "asking again after a surprise")
        assert engine.supports_tool_calling is False
        backend.fixed = True
        await _until(lambda: engine.supports_tool_calling, "tools back on")
    finally:
        await engine.aclose()
    assert any("failed unexpectedly" in r.getMessage() for r in caplog.records)


def test_a_refusal_with_no_event_loop_to_ask_on_starts_nothing() -> None:
    """Synchronous callers (nothing in this driver is one) have no one to ask on; they must not
    crash, and must not leave a task behind."""
    engine = _engine()
    target = _Target(id="qwen", upstream="qwen", temperature_fixed=False)
    engine._learn_tools_refused(target, WORDS, 500)
    assert engine.supports_tool_calling is False
    assert engine._tools_watch is None


@respx.mock
@pytest.mark.anyio
async def test_the_question_uses_the_engines_one_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """One client per instance, never per call: the first request and every later question share it."""
    _http.reset_shared()
    backend = Backend()
    respx.post(CHAT).mock(side_effect=backend)
    built = _count_constructions(monkeypatch)
    engine = _fast(_engine())
    try:
        with pytest.raises(ToolsRefused):
            await engine.generate(_ask())
        await _until(lambda: len(backend.probes) >= 3, "three questions")
        assert built[0] == 1, f"{built[0]} clients; one engine owns one client"
    finally:
        await engine.aclose()
        _http.reset_shared()


# --------------------------------------------------------------------------- #
# an account: each model on its own
# --------------------------------------------------------------------------- #


@respx.mock
@pytest.mark.anyio
async def test_an_account_asks_about_each_model_it_turned_off_and_clears_them_one_by_one(
    tmp_path: Path,
) -> None:
    respx.get(f"{BASE}/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "a"}, {"id": "b"}, {"id": "c"}]})
    )
    refusing = {"a", "b"}
    asked: list[dict[str, Any]] = []

    def chat(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        asked.append(body)
        if body.get("tools") and body["model"] in refusing:
            return httpx.Response(500, json=NO_JINJA)
        return httpx.Response(200, json=OK)

    respx.post(CHAT).mock(side_effect=chat)
    engine = _fast(_account(tmp_path / "models.json"))
    assert engine.catalogue is not None
    try:
        await engine.catalogue.refresh()
        for model in ("a", "b"):
            with pytest.raises(ToolsRefused):
                await engine.generate(_ask(model=model))

        def tools() -> dict[str, bool | None]:
            assert engine.catalogue is not None
            return {m.id: m.capabilities.toolCalling for m in engine.catalogue.exposed()}  # type: ignore[union-attr]

        assert tools() == {"a": False, "b": False, "c": True}

        def questions(model: str) -> int:
            return sum(1 for b in asked if b["model"] == model and b.get("max_tokens") == 1)

        await _until(lambda: questions("a") >= 2 and questions("b") >= 2, "both asked about")
        assert questions("c") == 0, "only a model that was turned off is asked about"
        assert tools() == {"a": False, "b": False, "c": True}

        refusing.discard("a")  # one of them is fixed (a router loaded it differently)
        await _until(lambda: tools()["a"] is True, "a back on")
        assert tools() == {"a": True, "b": False, "c": True}
        found = engine.catalogue.find("a")
        assert found is not None and found.capabilities is not None
        assert "tools" in (found.capabilities.supportedSettings or [])
        assert engine.tools_refused_words("a") is None and engine.tools_refused_words("b") == WORDS

        # b is still being asked about; a is not.
        mark_a = questions("a")
        mark_b = questions("b")
        await _until(lambda: questions("b") >= mark_b + 2, "b asked about again")
        assert questions("a") == mark_a

        refusing.clear()
        await _until(lambda: tools() == {"a": True, "b": True, "c": True}, "all back on")
        watch = engine._tools_watch
        assert watch is not None
        await _until(watch.done, "the asking to end")
    finally:
        await engine.aclose()


@respx.mock
@pytest.mark.anyio
async def test_a_model_the_account_stopped_serving_is_forgotten_not_asked_about(
    tmp_path: Path,
) -> None:
    listing: dict[str, Any] = {"data": [{"id": "a"}, {"id": "b"}]}
    respx.get(f"{BASE}/v1/models").mock(side_effect=lambda r: httpx.Response(200, json=listing))
    asked: list[dict[str, Any]] = []

    def chat(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        asked.append(body)
        return (
            httpx.Response(500, json=NO_JINJA)
            if body.get("tools")
            else httpx.Response(200, json=OK)
        )

    respx.post(CHAT).mock(side_effect=chat)
    engine = _fast(_account(tmp_path / "models.json"), seconds=0.05)
    assert engine.catalogue is not None
    try:
        await engine.catalogue.refresh()
        with pytest.raises(ToolsRefused):
            await engine.generate(_ask(model="a"))
        with pytest.raises(ToolsRefused):
            await engine.generate(_ask(model="b"))

        listing["data"] = [{"id": "b"}]  # the provider no longer lists a
        await engine.catalogue.refresh()
        mark = len(asked)
        await _until(lambda: engine.tools_refused_words("a") is None, "a forgotten")
        await _until(lambda: len(asked) >= mark + 2, "b still asked about")
        assert [b["model"] for b in asked[mark:] if b.get("max_tokens") == 1].count("a") == 0
        assert engine.tools_refused_words("b") == WORDS
    finally:
        await engine.aclose()


# --------------------------------------------------------------------------- #
# through the real app: the whole wire
# --------------------------------------------------------------------------- #


@respx.mock
def test_through_the_app_tools_come_back_when_the_backend_is_fixed_in_place(
    tmp_path: Path,
) -> None:
    backend = Backend()
    _llama(backend)
    with _custom_app(tmp_path, modelId="qwen") as client:
        engine = client.app.state.adapter  # type: ignore[attr-defined]
        _fast(engine)

        def caps() -> dict[str, Any]:
            return client.get("/v1/info").json()["models"][0]["capabilities"]

        assert caps()["toolCalling"] is True
        first = client.post("/v1/generate", json=_body())
        assert first.status_code == 400 and WORDS in first.json()["detail"]["detail"]
        assert caps()["toolCalling"] is False

        # While it stays refused: the gateway-facing answer does not change, and the door holds.
        _until_sync(lambda: len(backend.probes) >= 2, "two questions")
        assert caps()["toolCalling"] is False
        assert client.post("/v1/generate", json=_body()).status_code == 400

        # llama-server started again with `--jinja`, under the same driver.
        backend.fixed = True
        _until_sync(lambda: caps()["toolCalling"] is True, "toolCalling true again in /v1/info")
        assert "tools" in caps()["supportedSettings"]
        again = client.post("/v1/generate", json=_body())
        assert again.status_code == 200, again.text
        assert backend.requests[-1]["tools"], "and the tools reached the backend"


@respx.mock
@pytest.mark.parametrize("path", ["/v1/generate", "/v1/generate/stream"])
def test_after_a_flip_a_request_with_no_tools_is_served_without_what_rides_with_tools(
    tmp_path: Path, path: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Codex sends `tool_choice` and `parallel_tool_calls` on every request, tools or not, as
    explicit settings. Round one took both away from the offer with `tools`, so after a flip
    a plain chat from Codex was a 400 about `tool_choice`: for a setting that means nothing
    without tools. Now they are left out of that request, said once per field, and a request
    that DOES carry tools is still refused at the door."""
    caplog.set_level(logging.WARNING, logger=ENGINE_LOG)
    backend = Backend()
    _llama(backend)
    with _custom_app(tmp_path, modelId="qwen") as client:
        assert client.post("/v1/generate", json=_body()).status_code == 400  # the flip
        settings = client.get("/v1/info").json()["models"][0]["capabilities"]["supportedSettings"]
        assert "tools" not in settings
        assert {"toolChoice", "parallelToolCalls"} <= set(settings)

        plain = _body(
            tools=False,
            toolChoice="auto",
            parallelToolCalls=False,
            callerSettings=["toolChoice", "parallelToolCalls"],
        )
        for _ in range(3):
            served = client.post(path, json=plain)
            assert served.status_code == 200, served.text
        sent = backend.requests[-1]
        assert not {"tools", "tool_choice", "parallel_tool_calls"} & sent.keys()

        # With tools it is refused at the door, in the backend's words, and nothing is sent.
        mark = len(backend.requests)
        carried = _body(toolChoice="auto", callerSettings=["tools", "toolChoice"])
        refused = client.post(path, json=carried)
        assert refused.status_code == 400
        assert refused.json()["detail"]["type"].endswith("#tools-unsupported")
        assert WORDS in refused.json()["detail"]["detail"]
        assert len(backend.requests) == mark

    # Said once per field however many requests, and with its cause in words.
    left_out = [r.getMessage() for r in caplog.records if "omitting `" in r.getMessage()]
    assert sum("`tool_choice`" in m for m in left_out) == 1, left_out
    assert sum("`parallel_tool_calls`" in m for m in left_out) == 1, left_out
    assert all(WORDS in m and "carries no tools" in m for m in left_out)


@respx.mock
@pytest.mark.anyio
async def test_after_a_flip_one_models_companions_are_left_out_and_anothers_are_sent(
    tmp_path: Path,
) -> None:
    respx.get(f"{BASE}/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "a"}, {"id": "b"}]})
    )
    seen: list[dict[str, Any]] = []

    def chat(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if body["model"] == "a" and body.get("tools"):
            return httpx.Response(500, json=NO_JINJA)
        return httpx.Response(200, json=OK)

    respx.post(CHAT).mock(side_effect=chat)
    engine = _account(tmp_path / "models.json")
    assert engine.catalogue is not None
    try:
        await engine.catalogue.refresh()
        with pytest.raises(ToolsRefused):
            await engine.generate(_ask(model="a"))
        explicit = {"callerSettings": ["toolChoice", "parallelToolCalls"]}
        await engine.generate(
            _ask(tools=False, model="a", toolChoice="auto", parallelToolCalls=True, **explicit)
        )
        a = seen[-1]
        assert a["model"] == "a" and not {"tool_choice", "parallel_tool_calls"} & a.keys()
        # b has refused nothing, so what rides with its tools goes with them.
        await engine.generate(
            _ask(model="b", toolChoice="auto", parallelToolCalls=True, **explicit)
        )
        b = seen[-1]
        assert b["model"] == "b" and b["tool_choice"] == "auto"
        assert b["parallel_tool_calls"] is True and b["tools"]
    finally:
        await engine.aclose()


@respx.mock
def test_a_custom_endpoint_that_never_refuses_is_never_asked_anything_extra(
    tmp_path: Path,
) -> None:
    """The asking is for a backend that refused. A healthy one sees only what the caller sent."""
    backend = Backend()
    backend.fixed = True
    _llama(backend)
    with _custom_app(tmp_path, modelId="qwen") as client:
        for _ in range(3):
            assert client.post("/v1/generate", json=_body()).status_code == 200
        engine = client.app.state.adapter  # type: ignore[attr-defined]
        assert engine._tools_watch is None
    assert backend.probes == []


@respx.mock
def test_an_account_through_the_app_serves_a_companion_only_request_after_a_flip(
    tmp_path: Path,
) -> None:
    respx.get(f"{BASE}/props").mock(return_value=httpx.Response(404))
    respx.get(f"{BASE}/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "a"}, {"id": "b"}]})
    )
    seen: list[dict[str, Any]] = []

    def chat(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if body["model"] == "a" and body.get("tools"):
            return httpx.Response(500, json=NO_JINJA)
        return httpx.Response(200, json=OK)

    respx.post(CHAT).mock(side_effect=chat)
    respx.route(host="llama.local").mock(return_value=httpx.Response(404))
    with _custom_app(tmp_path) as client:
        _wait_for_catalogue(client)
        assert client.post("/v1/generate", json=_body(model="a")).status_code == 400
        plain = _body(model="a", tools=False, toolChoice="auto", callerSettings=["toolChoice"])
        served = client.post("/v1/generate", json=plain)
        assert served.status_code == 200, served.text
        assert "tool_choice" not in seen[-1]


# --------------------------------------------------------------------------- #
# the asking, held to what the report and the contract say of it
# --------------------------------------------------------------------------- #
#
# Every test above shortens the interval on the engine (`_fast`), and the contract (specs
# 6178255) says "every 10 minutes", a question "with one tool and one token", and a bounded wait.
# These are the checks that a number, or a cap, written once is the one the loop uses.


@respx.mock
@pytest.mark.anyio
async def test_the_spacing_is_the_constant_and_not_a_number_the_loop_remembers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The constant is patched and the engine is built the way the driver builds it, never
    shortened by hand: the first question waits one whole interval after the refusal, and each
    next one waits one whole interval after the last. (A hard-coded 600 anywhere between the
    constant and the sleep never asks at all within this test's patience.)"""
    interval = 0.3
    monkeypatch.setattr(openai_compat_http, "TOOLS_REASK_SECONDS", interval)
    backend = Backend()
    respx.post(CHAT).mock(side_effect=backend)
    engine = _engine()
    assert engine.tools_reask_seconds == interval
    try:
        with pytest.raises(ToolsRefused):
            await engine.generate(_ask())
        refused = time.perf_counter()
        await _until(lambda: len(backend.probe_stamps) >= 3, "three questions")
        stamps = backend.probe_stamps[:3]
    finally:
        await engine.aclose()

    # Not a tick early: `asyncio.sleep` runs on a clock that can be 15.6 ms coarse on Windows.
    least = interval * 0.8
    assert stamps[0] - refused >= least, "asked at once, not after an interval"
    assert stamps[1] - stamps[0] >= least and stamps[2] - stamps[1] >= least


@respx.mock
@pytest.mark.anyio
async def test_the_question_is_allowed_60_seconds_and_5_to_connect_whatever_a_request_is() -> None:
    """Longer than a probe's, because it is a real one-token generation that may queue behind a
    busy slot; shorter than a request's, because nobody is waiting on it. Carried on the ask
    itself (the engine's one client carries the request budget): 42 s here, so an ask that
    inherited it would show."""
    backend = Backend()
    respx.post(CHAT).mock(side_effect=backend)
    engine = _fast(_engine(timeout_seconds=42.0))
    try:
        with pytest.raises(ToolsRefused):
            await engine.generate(_ask())
        await _until(lambda: bool(backend.probe_budgets), "a first question")
        asked = backend.probe_budgets[0]
        requested = backend.budgets[0]
    finally:
        await engine.aclose()

    assert requested == {"connect": 10.0, "read": 42.0, "write": 42.0, "pool": 42.0}
    assert asked == {"connect": 5.0, "read": 60.0, "write": 60.0, "pool": 60.0}


@respx.mock
@pytest.mark.anyio
async def test_a_second_refusal_does_not_start_a_second_asker(tmp_path: Path) -> None:
    """One task for the engine however many of its models are off. Two would ask about each
    model twice a round, and `aclose()` would cancel only the last one made."""
    respx.get(f"{BASE}/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "a"}, {"id": "b"}]})
    )
    backend = Backend()
    respx.post(CHAT).mock(side_effect=backend)
    engine = _fast(_account(tmp_path / "models.json"), seconds=0.05)
    assert engine.catalogue is not None
    try:
        await engine.catalogue.refresh()
        with pytest.raises(ToolsRefused):
            await engine.generate(_ask(model="a"))
        asker = engine._tools_watch
        assert asker is not None and not asker.done()

        with pytest.raises(ToolsRefused):
            await engine.generate(_ask(model="b"))
        assert engine._tools_watch is asker, "the asker already running is the one for both"
        askers = [t for t in asyncio.all_tasks() if t.get_name() == "tools-again"]
        assert askers == [asker]
        assert engine.tools_refused_words("a") == WORDS and engine.tools_refused_words("b") == WORDS
    finally:
        await engine.aclose()
    assert [t for t in asyncio.all_tasks() if t.get_name() == "tools-again"] == []


@respx.mock
@pytest.mark.anyio
async def test_the_question_names_the_upstream_model_and_never_the_public_one() -> None:
    """A driver may serve a model under one name and ask its backend for another
    (`upstreamModelId`). The request does, and so must the question: asked by the public name
    it would be answered by nothing (a 404 or a different model) and a backend that had been
    fixed would never be heard from."""
    backend = Backend()
    respx.post(CHAT).mock(side_effect=backend)
    engine = _fast(_engine(upstream_model_id="vendor/qwen-real"))
    try:
        with pytest.raises(ToolsRefused):
            await engine.generate(_ask())
        await _until(lambda: len(backend.probes) >= 2, "two questions")
        assert backend.requests[0]["model"] == "vendor/qwen-real"
        assert {b["model"] for b in backend.probes} == {"vendor/qwen-real"}
        # And it is the same model that is believed when the backend answers.
        backend.fixed = True
        await _until(lambda: engine.supports_tool_calling, "tools back on")
    finally:
        await engine.aclose()
    assert "qwen" not in {b["model"] for b in backend.requests}


@respx.mock
@pytest.mark.anyio
async def test_an_accounts_question_names_the_entrys_upstream_id(tmp_path: Path) -> None:
    respx.get(f"{BASE}/v1/models").mock(return_value=httpx.Response(200, json={"data": []}))
    backend = Backend()
    respx.post(CHAT).mock(side_effect=backend)
    engine = _fast(_account(tmp_path / "models.json"))
    assert engine.catalogue is not None
    engine.catalogue._set(
        [
            DriverModel(
                id="public/alias",
                upstreamId="vendor/real",
                surfaces=["chat"],
                capabilities=Capabilities(supportedSettings=["tools"], toolCalling=True),
            )
        ],
        None,
    )
    try:
        with pytest.raises(ToolsRefused):
            await engine.generate(_ask(model="public/alias"))
        await _until(lambda: len(backend.probes) >= 2, "two questions")
        assert engine.tools_refused_words("public/alias") == WORDS
    finally:
        await engine.aclose()
    assert {b["model"] for b in backend.requests} == {"vendor/real"}
