"""One client, one context, one clock — the checks that reproduce it.

Each test here fails against the code as it stood on 2026-09-17, and
each names the number it is defending:

  - constructing `httpx.AsyncClient()` without `verify=` parses
    certifi's PEM bundle. Measured in this venv on Windows/CPython
    3.12, the Python both installers provision: **104-136 ms** of
    *synchronous* CPU, against **0.03 ms** with a shared context. The
    engine built one per completion, per stream and per embed.
  - `response.elapsed` starts after the client is built, so the
    driver's own largest cost sat outside the driver's own
    measurement. That is the ~116 ms this project carried as
    unexplained from M8.
  - `trust_env` sends a loopback probe through a corporate
    `HTTP_PROXY` that cannot reach 127.0.0.1 — and on Windows the
    logon task inherits that variable from the user environment.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from eugene_plexus_inference_driver import _http
from eugene_plexus_inference_driver._generated.models import GenerateRequest, Message, Role
from eugene_plexus_inference_driver.engines.openai_compat_http import OpenAiCompatibleHttpEngine

BASE = "http://127.0.0.1:9931"


def _engine(base_url: str = BASE) -> OpenAiCompatibleHttpEngine:
    return OpenAiCompatibleHttpEngine(
        api_key="k", base_url=base_url, model_id="m", auth_required=False
    )


def _request() -> GenerateRequest:
    return GenerateRequest(messages=[Message(role=Role.user, content="hi")])


def _completion() -> dict[str, Any]:
    return {
        "model": "m",
        "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


@pytest.fixture(autouse=True)
def _reset_shared() -> Any:
    _http.reset_shared()
    yield
    _http.reset_shared()


def _count_constructions(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Counts `httpx.AsyncClient()` constructions. Wraps rather than
    replaces, so the client still works."""
    calls = [0]
    real = httpx.AsyncClient.__init__

    def counting(self: httpx.AsyncClient, *args: Any, **kwargs: Any) -> None:
        calls[0] += 1
        real(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", counting)
    return calls


# --------------------------------------------------------------------------- #
# one client
# --------------------------------------------------------------------------- #


@respx.mock
async def test_three_completions_build_one_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """**The headline.** Three requests used to mean three certifi
    parses — ~312 ms of synchronous CPU on the event loop, for three
    trivial completions."""
    respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_completion())
    )
    engine = _engine()
    built = _count_constructions(monkeypatch)
    for _ in range(3):
        await engine.generate(_request())
    assert built[0] == 1, f"{built[0]} clients for 3 completions; one engine owns one client"
    await engine.aclose()


@respx.mock
async def test_generate_stream_and_embed_share_the_one_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All three request paths, plus the context probe, are one client.
    They had four construction sites between them."""
    respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            text='data: {"choices":[{"delta":{"content":"x"},"finish_reason":null}]}\n\ndata: [DONE]\n\n',
        )
    )
    respx.post(f"{BASE}/v1/embeddings").mock(
        return_value=httpx.Response(
            200, json={"data": [{"embedding": [0.1, 0.2]}], "usage": {"prompt_tokens": 1}}
        )
    )
    engine = _engine()
    built = _count_constructions(monkeypatch)
    async for _ in engine.stream(_request()):
        pass
    await engine.embed(["a"])
    assert built[0] == 1, f"{built[0]} clients across stream+embed"
    await engine.aclose()


@respx.mock
async def test_aclose_releases_the_pool_and_a_later_call_rebuilds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Closing is real, and the engine is usable again afterwards —
    which is what makes `/v1/config/test`'s throwaway engine safe to
    close in a `finally`."""
    respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_completion())
    )
    engine = _engine()
    await engine.generate(_request())
    client = engine._client()
    await engine.aclose()
    assert client.is_closed
    built = _count_constructions(monkeypatch)
    await engine.generate(_request())
    assert built[0] == 1
    await engine.aclose()


# --------------------------------------------------------------------------- #
# one clock
# --------------------------------------------------------------------------- #


@respx.mock
async def test_latency_brackets_the_whole_call_not_just_the_socket() -> None:
    """`latencyMs` covers everything `stream()`'s does.

    Against `response.elapsed` this fails: httpx starts that clock after
    the client exists and after the request is built, so work the driver
    does on either side is invisible to the driver's own number — which
    is precisely how a 104 ms per-call cost hid for a week behind a
    gateway that measured the same request at 116 ms more.
    """
    respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_completion())
    )
    engine = _engine()
    real_payload = engine._payload_for

    def slow_payload(request: GenerateRequest) -> dict[str, Any]:
        time.sleep(0.05)  # stands in for the client build this used to hide
        return real_payload(request)

    engine._payload_for = slow_payload  # type: ignore[method-assign]
    response = await engine.generate(_request())
    assert response.latencyMs is not None
    assert response.latencyMs >= 45, (
        f"latencyMs={response.latencyMs} excludes 50 ms spent inside generate(); "
        "it is measuring the socket, not the call"
    )
    await engine.aclose()


def test_the_engine_measures_durations_with_perf_counter() -> None:
    """`time.monotonic()` is `GetTickCount64` on Windows/CPython 3.12 —
    a **15.6 ms grid**, 20 distinct values in 300 ms — and 3.12 is what
    both installers provision. The developer box runs 3.14, where
    CPython fixed it, so this is invisible exactly where it is written
    and present on every shipped install.

    Read as text rather than by calling, because the defect is which
    function the source names.
    """
    root = Path(__file__).resolve().parents[1] / "src" / "eugene_plexus_inference_driver"
    offenders = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if "_generated" not in path.parts and "time.monotonic()" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, f"these measure with monotonic(): {offenders}"


def test_perf_counter_is_finer_than_the_thing_it_measures() -> None:
    """The grid is real on this interpreter, or the rule above is
    cargo-cult. Asserts the *instrument*, not the platform: perf_counter
    must resolve far finer than one HTTP hop."""
    assert time.get_clock_info("perf_counter").resolution <= 1e-6


# --------------------------------------------------------------------------- #
# no ambient proxy on a loopback client
# --------------------------------------------------------------------------- #


def test_a_loopback_engine_declines_an_ambient_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """A user's `HTTP_PROXY` must not be applied to 127.0.0.1.

    Asserted through the mounts httpx actually builds, not through the
    flag we passed — the flag is the fix, the mounts are the behaviour.
    """
    monkeypatch.setenv("HTTP_PROXY", "http://corp.proxy:3128")
    monkeypatch.setenv("HTTPS_PROXY", "http://corp.proxy:3128")
    engine = _engine("http://127.0.0.1:8081")
    assert engine._client()._mounts == {}, "a loopback engine is routed through the user's proxy"


def test_a_lan_node_also_declines_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """Another node of this install is a private address, and a
    corporate proxy cannot reach it either."""
    monkeypatch.setenv("HTTPS_PROXY", "http://corp.proxy:3128")
    engine = _engine("http://192.168.16.75:8081")
    assert engine._client()._mounts == {}


def test_a_cloud_backend_still_honours_the_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """**The half a blanket `trust_env=False` would have broken.** This
    engine also fronts OpenAI, OpenRouter and xAI; for a user behind a
    corporate proxy that IS how they reach the internet."""
    monkeypatch.setenv("HTTPS_PROXY", "http://corp.proxy:3128")
    engine = _engine("https://api.openai.com")
    assert engine._client()._mounts, "a cloud backend lost the user's proxy"


def test_is_internal_classifies_the_cases_this_product_meets() -> None:
    for url in (
        "http://127.0.0.1:8081",
        "http://localhost:8080",
        "http://192.168.16.75:8079",
        "http://10.0.0.4:8080",
        "http://[::1]:8080",
        "http://amish-station:8079",
    ):
        assert _http.is_internal(url), url
    for url in (
        "https://api.openai.com",
        "https://huggingface.co",
        "https://openrouter.ai/api",
        "http://140.82.121.4:8080",
    ):
        assert not _http.is_internal(url), url
    # Documentation space (TEST-NET) reads as private to `ipaddress`, so
    # it lands on the internal side. Recorded rather than special-cased:
    # nobody's backend is there, and the cost of being wrong that way is
    # one proxy hop not taken.
    assert _http.is_internal("http://203.0.113.9:8080")


# --------------------------------------------------------------------------- #
# the context itself
# --------------------------------------------------------------------------- #


def test_the_ssl_context_is_built_once_per_process() -> None:
    assert _http.ssl_context() is _http.ssl_context()


async def test_building_a_client_does_not_reparse_the_bundle() -> None:
    """The measurement, as a check. Not a benchmark: the assertion is
    two orders of magnitude of headroom, so it cannot fail on a slow
    machine while the defect is back."""
    _http.ssl_context()  # pay it once, deliberately, outside the timing
    started = time.perf_counter()
    clients = [_http.internal_client() for _ in range(5)]
    elapsed_ms = (time.perf_counter() - started) * 1000
    await asyncio.gather(*(c.aclose() for c in clients))
    assert elapsed_ms < 50, (
        f"five clients cost {elapsed_ms:.0f} ms; a certifi parse is ~104 ms each, "
        "so the shared context is not being used"
    )
