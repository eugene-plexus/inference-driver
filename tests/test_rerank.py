"""MR6 -- the rerank door, driver half (2026-10-12).

`POST /v1/rerank` scores passages against a question on a backend that sorts
results (a `llama-server` started with `--reranking`), best first, **every
score on one scale**. The two scales are measured, not assumed (llama-server
b10948, the same four passages): Qwen3-Reranker-0.6B answers probabilities
(0.996 down to 0.00001) and bge-reranker-v2-m3 answers logits (8.19 down to
-11.04). A threshold means nothing across the two, so the driver learns each
backend's scale (`probe_rerank`, with two passages) and puts logits through
`1 / (1 + e^-x)`.

The upstream bodies here are the shapes llama-server answered, trimmed to the
fields read. The engine and the app are the real ones; the backend is a fake.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from eugene_plexus_inference_driver._generated.models import (
    BackendKind,
    DriverModel,
    RerankPassagesRequest,
    RerankScale,
)
from eugene_plexus_inference_driver.app import create_app
from eugene_plexus_inference_driver.engines._catalogue import Catalogue
from eugene_plexus_inference_driver.engines._subprocess import BackendTimeout, CliError
from eugene_plexus_inference_driver.engines.base import RerankUnsupported
from eugene_plexus_inference_driver.engines.openai_compat_http import OpenAiCompatibleHttpEngine
from eugene_plexus_inference_driver.engines.strata_http import StrataHttpEngine
from eugene_plexus_inference_driver.failures import disposition
from eugene_plexus_inference_driver.routes.generate import _backend_error
from eugene_plexus_inference_driver.settings import Settings

from .conftest import FakeInstall

BASE = "http://127.0.0.1:8081"
ENGINE_LOG = "eugene_plexus_inference_driver.engines.openai_compat_http"
QUESTION = "What is the capital of France?"

#: What a probe gets from Qwen3-Reranker (probabilities) and from
#: bge-reranker-v2-m3 (logits), the measured numbers.
PROBABILITIES = [(0, 0.9977), (1, 0.00002)]
LOGITS = [(0, 8.6184), (1, -10.9946)]


def _engine(model_id: str = "the-reranker", **kwargs: Any) -> OpenAiCompatibleHttpEngine:
    return OpenAiCompatibleHttpEngine(
        api_key=None,
        base_url=BASE,
        model_id=model_id,
        backend_kind=BackendKind.openai_compat_http,
        auth_required=False,
        **kwargs,
    )


def _request(passages: int = 3, **fields: Any) -> RerankPassagesRequest:
    return RerankPassagesRequest(
        query=QUESTION, documents=[f"passage {i}" for i in range(passages)], **fields
    )


def _answer(
    *scored: tuple[int, float], model: str = "the-reranker", usage: Any = "default"
) -> dict[str, Any]:
    """What `llama-server --reranking` answers, in the order given."""
    return {
        "model": model,
        "object": "list",
        "usage": {"prompt_tokens": 170, "total_tokens": 170} if usage == "default" else usage,
        "results": [{"index": i, "relevance_score": s} for i, s in scored],
    }


def _ok(*scored: tuple[int, float], **fields: Any) -> httpx.Response:
    return httpx.Response(200, json=_answer(*scored, **fields))


def _rerank_route(*scored: tuple[int, float]) -> respx.Route:
    return respx.post(f"{BASE}/v1/rerank").mock(return_value=_ok(*scored))


def _logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _sent(route: respx.Route, call: int = -1) -> dict[str, Any]:
    return json.loads(route.calls[call].request.content)


def _said_logits(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if "outside 0 to 1" in r.getMessage()]


# --------------------------------------------------------------------- #
# the scores: one scale, whatever the backend answers
# --------------------------------------------------------------------- #


@respx.mock
async def test_a_probability_backends_scores_pass_through_unchanged() -> None:
    _rerank_route((0, 0.9977), (2, 0.0009), (1, 0.00002))

    answer = await _engine().rerank(_request(3))

    assert answer.scale is RerankScale.probability
    assert [(r.index, r.score, r.rawScore) for r in answer.results] == [
        (0, 0.9977, 0.9977),
        (2, 0.0009, 0.0009),
        (1, 0.00002, 0.00002),
    ]


@respx.mock
async def test_a_logit_backends_scores_go_through_the_logistic_and_keep_their_order() -> None:
    """bge-reranker-v2-m3's four numbers from the measurement, arriving in the
    passages' own order: out of 0 to 1, so logits, so mapped; `rawScore` is the
    backend's own number, and the order is the logits' order."""
    _rerank_route((0, 8.19), (1, -11.04), (2, -5.45), (3, 7.80))

    answer = await _engine().rerank(_request(4))

    assert answer.scale is RerankScale.logit
    assert [r.index for r in answer.results] == [0, 3, 2, 1]
    assert [r.rawScore for r in answer.results] == [8.19, 7.80, -5.45, -11.04]
    for row in answer.results:
        assert row.score == pytest.approx(_logistic(row.rawScore))
        assert 0.0 <= row.score <= 1.0
    scores = [r.score for r in answer.results]
    assert scores == sorted(scores, reverse=True) and len(set(scores)) == 4


@respx.mock
async def test_the_logistic_cannot_overflow_on_a_large_negative_logit() -> None:
    """`math.exp(1000)` is an OverflowError, which is what `1 / (1 + e^-x)` is
    for x = -1000. Both ends, and the order between them, must survive."""
    _rerank_route((0, -1000.0), (1, 1000.0), (2, 0.0), (3, -700.0))

    answer = await _engine().rerank(_request(4))

    assert [(r.index, r.rawScore) for r in answer.results] == [
        (1, 1000.0),
        (2, 0.0),
        (3, -700.0),
        (0, -1000.0),
    ]
    assert [r.score for r in answer.results][:2] == [1.0, 0.5]
    assert 0.0 < answer.results[2].score < 1e-300
    assert answer.results[3].score == 0.0


@respx.mock
async def test_equal_scores_are_ordered_by_index() -> None:
    """The contract: best first, and by `index` where two are equal. The
    backend sorted them its own way (here, last first)."""
    _rerank_route((2, 0.5), (1, 0.5), (0, 0.5), (3, 0.9))

    answer = await _engine().rerank(_request(4))

    assert [r.index for r in answer.results] == [3, 0, 1, 2]


@respx.mock
async def test_equal_logits_are_ordered_by_index_too() -> None:
    _rerank_route((1, 3.0), (0, 3.0), (2, 8.0))

    answer = await _engine().rerank(_request(3))

    assert answer.scale is RerankScale.logit
    assert [r.index for r in answer.results] == [2, 0, 1]


# --------------------------------------------------------------------- #
# topN, usage, identity
# --------------------------------------------------------------------- #


@respx.mock
async def test_top_n_is_sent_to_the_backend_and_applied_to_the_answer() -> None:
    route = _rerank_route((3, 0.99), (1, 0.7))

    answer = await _engine().rerank(_request(4, topN=2))

    assert _sent(route)["top_n"] == 2
    assert [r.index for r in answer.results] == [3, 1]


@respx.mock
async def test_top_n_is_applied_even_when_the_backend_ignores_it() -> None:
    _rerank_route((0, 0.2), (1, 0.9), (2, 0.5), (3, 0.7))

    answer = await _engine().rerank(_request(4, topN=2))

    assert [r.index for r in answer.results] == [1, 3]


@respx.mock
async def test_top_n_beyond_the_passages_is_all_of_them() -> None:
    _rerank_route((0, 0.2), (1, 0.9), (2, 0.5))

    answer = await _engine().rerank(_request(3, topN=10))

    assert [r.index for r in answer.results] == [1, 2, 0]


@respx.mock
async def test_no_top_n_means_none_is_sent() -> None:
    route = _rerank_route((0, 0.2), (1, 0.9))

    answer = await _engine().rerank(_request(2))

    assert "top_n" not in _sent(route)
    assert len(answer.results) == 2


@respx.mock
async def test_the_body_sent_is_jinas_shape_with_the_upstream_model() -> None:
    route = _rerank_route((0, 0.9))

    await _engine("public-name", upstream_model_id="default_model").rerank(_request(1))

    assert _sent(route) == {
        "model": "default_model",
        "query": QUESTION,
        "documents": ["passage 0"],
    }


@respx.mock
async def test_usage_carries_the_two_counts_a_reranker_reports() -> None:
    _rerank_route((0, 0.9))
    answer = await _engine().rerank(_request(1))
    assert answer.usage is not None
    assert (answer.usage.promptTokens, answer.usage.totalTokens) == (170, 170)
    assert answer.usage.completionTokens is None, "a reranker generates nothing; none is invented"


@respx.mock
async def test_usage_with_only_a_total_is_a_total() -> None:
    """vLLM's rerank reports `total_tokens` alone."""
    respx.post(f"{BASE}/v1/rerank").mock(return_value=_ok((0, 0.9), usage={"total_tokens": 42}))
    answer = await _engine().rerank(_request(1))
    assert answer.usage is not None
    assert (answer.usage.promptTokens, answer.usage.totalTokens) == (None, 42)


@respx.mock
@pytest.mark.parametrize("usage", [None, {}, "nope", {"prompt_tokens": True}])
async def test_usage_is_absent_when_the_backend_says_nothing_usable(usage: Any) -> None:
    respx.post(f"{BASE}/v1/rerank").mock(return_value=_ok((0, 0.9), usage=usage))
    assert (await _engine().rerank(_request(1))).usage is None


@respx.mock
async def test_the_answer_names_the_public_model_the_backend_and_the_request() -> None:
    """Public id, as `EmbedResponse.modelId` is: the backend's own name for a
    model it was asked for as `default_model` must not leak out."""
    route = respx.post(f"{BASE}/v1/rerank").mock(return_value=_ok((0, 0.9), model="default_model"))

    answer = await _engine("public-name", upstream_model_id="default_model").rerank(
        _request(1, requestId="req-7")
    )

    assert answer.modelId == "public-name"
    assert answer.backend is BackendKind.openai_compat_http
    assert answer.requestId == "req-7"
    assert route.calls[0].request.headers["x-request-id"] == "req-7"
    assert answer.latencyMs is not None and answer.latencyMs >= 0


@respx.mock
async def test_a_vllm_shaped_answer_reads_the_same() -> None:
    """vLLM's rerank adds `id` and a `document` object to every result and
    reports only `total_tokens`; the two fields read are the same."""
    respx.post(f"{BASE}/v1/rerank").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "rerank-1",
                "model": "the-reranker",
                "usage": {"total_tokens": 30},
                "results": [
                    {"index": 1, "document": {"text": "passage 1"}, "relevance_score": 0.91},
                    {"index": 0, "document": {"text": "passage 0"}, "relevance_score": 0.12},
                ],
            },
        )
    )
    answer = await _engine().rerank(_request(2))
    assert [(r.index, r.score) for r in answer.results] == [(1, 0.91), (0, 0.12)]


# --------------------------------------------------------------------- #
# the scale is learned with the capability, and corrected by an answer
# --------------------------------------------------------------------- #


@respx.mock
async def test_the_probe_asks_a_question_with_two_passages_one_that_answers_it() -> None:
    route = _rerank_route(*PROBABILITIES)
    engine = _engine("public-name", upstream_model_id="default_model")

    assert await engine.probe_rerank() is True

    sent = _sent(route)
    assert set(sent) == {"model", "query", "documents"}, "tiny, and no top_n"
    assert sent["model"] == "default_model"
    assert len(sent["documents"]) == 2
    answers, does_not = sent["documents"]
    assert "Paris" in answers and "capital of France" in sent["query"]
    assert "Paris" not in does_not and "France" not in does_not


@respx.mock
@pytest.mark.parametrize(
    ("probed", "scale"),
    [
        (PROBABILITIES, RerankScale.probability),
        (LOGITS, RerankScale.logit),
        ([(0, 0.4), (1, 3.2)], RerankScale.logit),  # one above 1 is enough
        ([(0, -0.2), (1, 0.5)], RerankScale.logit),  # one below 0 is enough
        ([(0, 1.0), (1, 0.0)], RerankScale.probability),  # the ends are inside
    ],
)
async def test_the_probes_answer_says_the_scale_and_a_later_rerank_uses_it(
    probed: list[tuple[int, float]], scale: RerankScale
) -> None:
    """A real answer entirely inside 0 to 1 cannot say what scale it is on --
    a logit of 0.3 is a logit -- so which one a backend is has to come from the
    probe: here the same two scores at the real request come back as 0.3 and 0.8."""
    route = _rerank_route(*probed)
    engine = _engine()
    assert await engine.probe_rerank() is True

    route.mock(return_value=_ok((0, 0.3), (1, 0.8)))
    answer = await engine.rerank(_request(2))

    assert answer.scale is scale
    expected = {
        RerankScale.probability: {0: 0.3, 1: 0.8},
        RerankScale.logit: {0: _logistic(0.3), 1: _logistic(0.8)},
    }[scale]
    assert {r.index: r.score for r in answer.results} == pytest.approx(expected)
    assert {r.index: r.rawScore for r in answer.results} == {0: 0.3, 1: 0.8}


@respx.mock
async def test_the_scale_is_kept_with_the_answer_and_not_asked_for_again() -> None:
    route = _rerank_route(*LOGITS)
    engine = _engine()

    assert await engine.probe_rerank() is True
    assert await engine.probe_rerank() is True
    route.mock(return_value=_ok((0, 0.3), (1, 0.8)))
    answer = await engine.rerank(_request(2))

    assert route.call_count == 2, "one probe, then the rerank: never a second probe"
    assert answer.scale is RerankScale.logit


@respx.mock
async def test_two_probes_out_at_once_are_said_once(caplog: pytest.LogCaptureFixture) -> None:
    """`/v1/info` asks twice at a cold start, once for itself and once inside
    the embeddings probe, before either has an answer. Both are out at once, and
    the first answer to arrive is the one that stands and the one that is said."""
    caplog.set_level(logging.INFO, logger=ENGINE_LOG)

    async def slowly(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.05)
        return _ok(*PROBABILITIES)

    route = respx.post(f"{BASE}/v1/rerank").mock(side_effect=slowly)
    engine = _engine()

    assert await asyncio.gather(engine.probe_rerank(), engine.probe_rerank()) == [True, True]

    assert route.call_count == 2, "both were out before either had an answer"
    assert len([r for r in caplog.records if "reranks, scale" in r.getMessage()]) == 1


@respx.mock
async def test_a_backend_taken_for_probabilities_that_answers_outside_0_to_1_is_logits_after(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The probe's two passages are a first reading and not the last. From the
    first answer outside 0 to 1 the backend is `logit` for the engine's life,
    that answer is mapped as logits (the 0.3 in it too), and the log says so
    once."""
    caplog.set_level(logging.INFO, logger=ENGINE_LOG)
    route = _rerank_route(*PROBABILITIES)
    engine = _engine()
    assert await engine.probe_rerank() is True

    route.mock(return_value=_ok((0, 7.5), (1, 0.3), (2, -4.0)))
    first = await engine.rerank(_request(3))
    assert first.scale is RerankScale.logit
    assert {r.index: r.score for r in first.results} == pytest.approx(
        {0: _logistic(7.5), 1: _logistic(0.3), 2: _logistic(-4.0)}
    )
    assert {r.index: r.rawScore for r in first.results} == {0: 7.5, 1: 0.3, 2: -4.0}

    # Inside 0 to 1 again, and still logits: it is not a backend that is
    # sometimes one and sometimes the other, and it does not go back.
    route.mock(return_value=_ok((0, 0.2), (1, 0.9)))
    second = await engine.rerank(_request(2))
    assert second.scale is RerankScale.logit
    assert {r.index: r.score for r in second.results} == pytest.approx(
        {0: _logistic(0.2), 1: _logistic(0.9)}
    )

    route.mock(return_value=_ok((0, 12.0), (1, -3.0)))
    await engine.rerank(_request(2))

    said = _said_logits(caplog)
    assert len(said) == 1, [r.getMessage() for r in said]
    assert said[0].levelno == logging.INFO
    assert "logits" in said[0].getMessage() and "taken for probabilities" in said[0].getMessage()
    assert await engine.probe_rerank() is True
    assert route.call_count == 4, "the probe once, the three reranks; the switch is not re-asked"


@respx.mock
async def test_a_rerank_before_the_probe_has_an_answer_reads_its_scale_off_its_own(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger=ENGINE_LOG)
    route = _rerank_route((0, 0.2), (1, 0.9))
    engine = _engine()

    inside = await engine.rerank(_request(2))
    assert inside.scale is RerankScale.probability
    assert [(r.index, r.score) for r in inside.results] == [(1, 0.9), (0, 0.2)]
    assert not _said_logits(caplog), "inside 0 to 1 proves nothing, and nothing is said"

    route.mock(return_value=_ok((0, 6.0), (1, -2.0)))
    outside = await engine.rerank(_request(2))
    assert outside.scale is RerankScale.logit
    assert [r.score for r in outside.results] == pytest.approx([_logistic(6.0), _logistic(-2.0)])

    route.mock(return_value=_ok((0, 0.2), (1, 0.9)))
    after = await engine.rerank(_request(2))
    assert after.scale is RerankScale.logit, "and it stays what the answer proved"
    said = _said_logits(caplog)
    assert len(said) == 1
    # The one answer inside 0 to 1 was a guess for that answer: nothing took
    # this backend for probabilities, so the log does not say it was.
    assert "taken for probabilities" not in said[0].getMessage()


@respx.mock
async def test_the_probe_never_takes_a_backend_back_from_logit() -> None:
    """A real answer outside 0 to 1 arrived while the probe was still out; the
    probe's own two passages, inside 0 to 1, must not outweigh it."""
    route = _rerank_route((0, 6.0), (1, -2.0))
    engine = _engine()
    assert (await engine.rerank(_request(2))).scale is RerankScale.logit

    route.mock(return_value=_ok(*PROBABILITIES))
    assert await engine.probe_rerank() is True
    route.mock(return_value=_ok((0, 0.3), (1, 0.8)))
    assert (await engine.rerank(_request(2))).scale is RerankScale.logit


@respx.mock
@pytest.mark.parametrize("answer", [{"detail": "ok"}, {"results": []}, ["not", "an", "object"]])
async def test_a_rerank_answered_without_any_score_is_not_a_reranker(answer: object) -> None:
    respx.post(f"{BASE}/v1/rerank").mock(return_value=httpx.Response(200, json=answer))
    assert await _engine().probe_rerank() is False


@respx.mock
@pytest.mark.parametrize(
    "results",
    [
        [{"index": 0}],
        [{"index": 0, "relevance_score": "0.9"}],
        [{"index": 0, "relevance_score": True}],
        [{"index": 0, "relevance_score": None}],
        ["0.9"],
    ],
)
async def test_results_with_no_number_in_them_are_not_a_rerank_either(results: list) -> None:
    """The scores are what say a server reranks, and the scale cannot be read
    off a score that is not there."""
    respx.post(f"{BASE}/v1/rerank").mock(
        return_value=httpx.Response(200, json={"results": results})
    )
    assert await _engine().probe_rerank() is False


# --------------------------------------------------------------------- #
# a backend that does not sort results says so, and is believed
# --------------------------------------------------------------------- #


@respx.mock
async def test_a_backend_started_without_reranking_is_refused_in_its_own_words() -> None:
    respx.post(f"{BASE}/v1/rerank").mock(
        return_value=httpx.Response(
            501,
            json={
                "error": {
                    "code": 501,
                    "message": "This server does not support reranking. Start it with `--reranking`",
                    "type": "not_supported_error",
                }
            },
        )
    )

    with pytest.raises(RerankUnsupported) as caught:
        await _engine().rerank(_request(2))

    assert "501" in str(caught.value) and "--reranking" in str(caught.value)


@respx.mock
@pytest.mark.parametrize("status", [404, 405, 410, 501])
async def test_a_backend_with_no_such_endpoint_is_not_a_reranker_from_then_on(
    status: int,
) -> None:
    """Probed as one earlier, and then restarted without `--reranking` (the
    agent restarts an engine under a driver that goes on): the backend's own
    answer to a real request is the newer fact, and `/v1/info` must stop
    listing a surface it will refuse."""
    route = _rerank_route(*PROBABILITIES)
    engine = _engine()
    assert await engine.probe_rerank() is True

    route.mock(return_value=httpx.Response(status, json={"error": {"message": "no"}}))
    with pytest.raises(RerankUnsupported):
        await engine.rerank(_request(2))

    assert await engine.probe_rerank() is False


@respx.mock
async def test_an_account_model_is_refused_by_the_engine_too() -> None:
    """The route refuses it first; this is the engine not being the one that
    would send it, whatever calls it."""
    route = respx.post(f"{BASE}/v1/rerank").mock(return_value=_ok((0, 0.9)))
    engine = OpenAiCompatibleHttpEngine(
        base_url=BASE,
        model_id=None,
        auth_required=False,
        backend_kind=BackendKind.openai_compat_http,
    )
    assert engine.catalogue is not None
    engine.catalogue._set([DriverModel(id="acme/sorter", surfaces=["chat"])], None)

    with pytest.raises(RerankUnsupported) as caught:
        await engine.rerank(_request(1, model="acme/sorter"))

    assert "account" in str(caught.value)
    assert not route.called


# --------------------------------------------------------------------- #
# over-long input: the request's fault, said with the limit, never retried
# --------------------------------------------------------------------- #

#: What llama-server b10948 answered a reranker (Qwen3-Reranker-0.6B and
#: bge-reranker-v2-m3 alike, `-c 4096 --parallel 4 -b 2048 -ub 2048`), verbatim:
#: input over a slot's context is the 400 every over-long input gets, and input
#: over the physical batch is a **500**.
OVER_THE_CONTEXT = (
    400,
    {
        "error": {
            "code": 400,
            "message": "input (1828 tokens) is larger than the max context size "
            "(1024 tokens). skipping",
            "type": "exceed_context_size_error",
            "n_prompt_tokens": 1828,
            "n_ctx": 1024,
        }
    },
    "1024",
)
OVER_THE_BATCH = (
    500,
    {
        "error": {
            "code": 500,
            "message": "input (2994 tokens) is too large to process. increase the physical "
            "batch size (current batch size: 2048)",
            "type": "server_error",
        }
    },
    "2048",
)


@respx.mock
@pytest.mark.parametrize(("status", "body", "limit"), [OVER_THE_CONTEXT, OVER_THE_BATCH])
async def test_an_over_long_passage_is_a_400_that_names_the_limit(
    status: int, body: dict, limit: str
) -> None:
    respx.post(f"{BASE}/v1/rerank").mock(return_value=httpx.Response(status, json=body))

    with pytest.raises(CliError) as caught:
        await _engine().rerank(_request(2))

    assert caught.value.upstream_status == status
    problem = _backend_error(caught.value, "openai_compat_http")
    assert problem.status_code == 400, "the request's fault, which another replica repeats"
    assert problem.detail["retryDisposition"] == "terminal"
    detail = problem.detail["detail"]
    assert limit in detail and "tokens" in detail, detail
    assert "longer than the backend reads at once" in detail
    assert "shorter passages" in detail


def test_input_over_the_physical_batch_is_the_requests_not_the_backends() -> None:
    """The only 500 llama-server gives that is a verdict on the request."""
    error = CliError(OVER_THE_BATCH[1]["error"]["message"], upstream_status=500)
    assert disposition(error) == "terminal"
    assert disposition(CliError("Internal Server Error", upstream_status=500)) == "indeterminate"


# --------------------------------------------------------------------- #
# every other failure is embed's
# --------------------------------------------------------------------- #


@respx.mock
@pytest.mark.parametrize("status", [503, 429, 502, 504, 408])
async def test_a_backend_that_is_loading_or_asks_for_a_retry_keeps_its_502(status: int) -> None:
    """`llama-server` answers 503 until its model is in memory. Neither that nor
    a rate limit says anything about the request, so neither is a 400."""
    respx.post(f"{BASE}/v1/rerank").mock(
        return_value=httpx.Response(
            status,
            headers={"Retry-After": "5"},
            json={"error": {"code": status, "message": "Loading model", "type": "unavailable"}},
        )
    )

    with pytest.raises(CliError) as caught:
        await _engine().rerank(_request(2))

    problem = _backend_error(caught.value, "openai_compat_http")
    assert problem.status_code == 502
    assert problem.detail["retryDisposition"] != "terminal"
    assert "Loading model" in problem.detail["detail"]
    assert problem.headers == {"Retry-After": "5"}


@respx.mock
async def test_a_backend_refusing_this_drivers_key_is_not_the_callers_400() -> None:
    respx.post(f"{BASE}/v1/rerank").mock(
        return_value=httpx.Response(401, json={"error": {"message": "bad key"}})
    )
    with pytest.raises(CliError) as caught:
        await _engine().rerank(_request(1))
    problem = _backend_error(caught.value, "openai_compat_http")
    assert problem.status_code == 502
    assert problem.detail["type"].endswith("#backend-credential-refused")


@respx.mock
async def test_a_backends_own_400_is_relayed_in_its_words_and_is_terminal() -> None:
    respx.post(f"{BASE}/v1/rerank").mock(
        return_value=httpx.Response(
            400, json={"error": {"message": '"query" must be provided', "type": "invalid"}}
        )
    )
    with pytest.raises(CliError) as caught:
        await _engine().rerank(_request(1))
    problem = _backend_error(caught.value, "openai_compat_http")
    assert problem.status_code == 400
    assert '"query" must be provided' in problem.detail["detail"]


@respx.mock
async def test_a_read_timeout_is_identified_and_names_its_deadline() -> None:
    respx.post(f"{BASE}/v1/rerank").mock(side_effect=httpx.ReadTimeout(""))
    with pytest.raises(BackendTimeout) as caught:
        await _engine().rerank(_request(1))
    assert "requestTimeoutSeconds" in str(caught.value)
    assert _backend_error(caught.value, "x").status_code == 504


@respx.mock
@pytest.mark.parametrize(
    "failure", [httpx.ConnectTimeout("no route"), httpx.ConnectError("refused")]
)
async def test_a_dead_host_keeps_its_502_so_the_next_replica_is_tried(failure: Exception) -> None:
    respx.post(f"{BASE}/v1/rerank").mock(side_effect=failure)
    with pytest.raises(CliError) as caught:
        await _engine().rerank(_request(1))
    assert not isinstance(caught.value, BackendTimeout)
    assert _backend_error(caught.value, "x").status_code == 502


@respx.mock
@pytest.mark.parametrize(
    ("answer", "why"),
    [
        (httpx.Response(200, text="<html>nope</html>"), "non-JSON"),
        (httpx.Response(200, json={}), "no rerank results"),
        (httpx.Response(200, json={"results": []}), "no rerank results"),
        (httpx.Response(200, json=["a", "b", "c"]), "no rerank results"),
        (_ok((0, 0.9), (1, 0.5)), "2 rerank scores for 3 passages"),
        (_ok((0, 0.9), (1, 0.5), (3, 0.1)), "malformed"),  # no passage 3
        (_ok((0, 0.9), (1, 0.5), (-1, 0.1)), "malformed"),
        (_ok((0, 0.9), (1, 0.5), (1, 0.1)), "malformed"),  # one passage twice
        (
            httpx.Response(200, json={"results": [{"index": 0, "relevance_score": "high"}]}),
            "malformed",
        ),
        (
            httpx.Response(200, json={"results": [{"index": True, "relevance_score": 0.5}]}),
            "malformed",
        ),
        (httpx.Response(200, json={"results": [{"relevance_score": 0.5}]}), "malformed"),
        (
            httpx.Response(200, json={"results": [{"index": 0, "relevance_score": True}]}),
            "malformed",
        ),
        (httpx.Response(200, json={"results": ["x"]}), "malformed"),
        (
            httpx.Response(200, content=b'{"results": [{"index": 0, "relevance_score": NaN}]}'),
            "malformed",
        ),
        (
            httpx.Response(
                200, content=b'{"results": [{"index": 0, "relevance_score": Infinity}]}'
            ),
            "malformed",
        ),
    ],
)
async def test_an_answer_that_cannot_be_matched_to_the_passages_is_refused(
    answer: httpx.Response, why: str
) -> None:
    """A passage is known by its index and nothing else, so a score without a
    usable index, a passage missing, one scored twice, or a number that is not
    one is a backend error and not a ranking."""
    respx.post(f"{BASE}/v1/rerank").mock(return_value=answer)

    with pytest.raises(CliError) as caught:
        await _engine().rerank(_request(3))

    assert why in str(caught.value)
    assert caught.value.upstream_status is None
    assert _backend_error(caught.value, "x").status_code == 502


@respx.mock
async def test_fewer_scores_than_passages_is_fine_when_top_n_asked_for_fewer() -> None:
    _rerank_route((2, 0.9), (0, 0.5))
    answer = await _engine().rerank(_request(3, topN=2))
    assert [r.index for r in answer.results] == [2, 0]


# --------------------------------------------------------------------- #
# engines that do not sort results
# --------------------------------------------------------------------- #


@respx.mock
async def test_strata_is_not_asked_and_does_not_rerank() -> None:
    """A text-chat subset: nothing is sent to it (an unmocked request would
    raise), and it is a definite no."""
    engine = StrataHttpEngine(base_url=BASE, model_id="qwen", auth_required=False)
    assert await engine.probe_rerank() is False
    with pytest.raises(RerankUnsupported):
        await engine.rerank(_request(1))


# --------------------------------------------------------------------- #
# the door, on the real app with the real engine and a fake backend
# --------------------------------------------------------------------- #


def _app(tmp_path: Path, **config: Any) -> TestClient:
    file = tmp_path / "config.yaml"
    file.write_text(
        json.dumps(
            {
                "provider": "openai_compat_custom",
                "baseUrl": BASE,
                "modelId": "the-reranker",
                "backendLocality": "local",
                **config,
            }
        ),
        encoding="utf-8",
    )
    return TestClient(create_app(settings=Settings(config_file=file)))


def _problem(response: httpx.Response) -> dict[str, Any]:
    assert response.headers["content-type"].startswith("application/json")
    return response.json()["detail"]


@respx.mock
def test_the_door_answers_scored_passages_best_first(tmp_path: Path) -> None:
    route = _rerank_route((0, 8.19), (1, -11.04), (2, -5.45), (3, 7.80))
    with _app(tmp_path) as client:
        response = client.post(
            "/v1/rerank",
            json={
                "query": QUESTION,
                "documents": ["a", "b", "c", "d"],
                "topN": 3,
                "requestId": "req-9",
            },
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert [r["index"] for r in body["results"]] == [0, 3, 2]
    assert [r["rawScore"] for r in body["results"]] == [8.19, 7.80, -5.45]
    assert body["scale"] == "logit"
    assert body["modelId"] == "the-reranker" and body["backend"] == "openai_compat_http"
    assert body["usage"] == {"promptTokens": 170, "totalTokens": 170}
    assert body["requestId"] == "req-9" and body["latencyMs"] >= 0
    assert None not in body.values(), "absent is absent, not null"
    assert _sent(route) == {
        "model": "the-reranker",
        "query": QUESTION,
        "documents": ["a", "b", "c", "d"],
        "top_n": 3,
    }
    assert route.calls[0].request.headers["x-request-id"] == "req-9"


@respx.mock
def test_a_model_this_driver_does_not_serve_is_a_404_and_nothing_is_sent(tmp_path: Path) -> None:
    route = _rerank_route((0, 0.9))
    with _app(tmp_path) as client:
        response = client.post(
            "/v1/rerank", json={"model": "someone-else", "query": "q", "documents": ["a"]}
        )
    assert response.status_code == 404
    assert _problem(response)["type"].endswith("#model-not-served")
    assert not route.called


@respx.mock
def test_the_named_model_is_served(tmp_path: Path) -> None:
    _rerank_route((0, 0.9))
    with _app(tmp_path) as client:
        response = client.post(
            "/v1/rerank", json={"model": "the-reranker", "query": "q", "documents": ["a"]}
        )
    assert response.status_code == 200, response.text


@respx.mock
@pytest.mark.parametrize(
    "body",
    [
        {"query": "q", "documents": []},
        {"query": "", "documents": ["a"]},
        {"documents": ["a"]},
        {"query": "q"},
        {"query": "q", "documents": ["a"], "topN": 0},
        {"query": "q", "documents": ["a"] * 1001},
    ],
    ids=["no passages", "empty question", "no question", "no documents", "topN 0", "1001 passages"],
)
def test_a_request_the_contract_does_not_allow_is_refused_before_anything_is_sent(
    tmp_path: Path, body: dict
) -> None:
    route = _rerank_route((0, 0.9))
    with _app(tmp_path) as client:
        response = client.post("/v1/rerank", json=body)
    assert response.status_code == 422
    assert not route.called


@respx.mock
def test_a_backend_that_does_not_sort_results_is_a_400_in_its_own_words(tmp_path: Path) -> None:
    respx.post(f"{BASE}/v1/rerank").mock(
        return_value=httpx.Response(
            501,
            json={
                "error": {
                    "code": 501,
                    "message": "This server does not support reranking. Start it with `--reranking`",
                    "type": "not_supported_error",
                }
            },
        )
    )
    with _app(tmp_path) as client:
        response = client.post("/v1/rerank", json={"query": "q", "documents": ["a"]})

    assert response.status_code == 400
    problem = _problem(response)
    assert problem["type"].endswith("#rerank-unsupported")
    assert problem["title"] == "This backend does not sort results"
    assert "does not sort results" in problem["detail"]
    assert "--reranking" in problem["detail"], "what the backend itself answered"
    assert problem["component"] == "inference-driver:openai_compat_http"


@respx.mock
@pytest.mark.parametrize(("status", "body", "limit"), [OVER_THE_CONTEXT, OVER_THE_BATCH])
def test_an_over_long_passage_through_the_door_is_a_400_naming_the_limit(
    tmp_path: Path, status: int, body: dict, limit: str
) -> None:
    respx.post(f"{BASE}/v1/rerank").mock(return_value=httpx.Response(status, json=body))
    with _app(tmp_path) as client:
        response = client.post("/v1/rerank", json={"query": "q", "documents": ["a", "b"]})

    assert response.status_code == 400
    problem = _problem(response)
    assert limit in problem["detail"] and problem["retryDisposition"] == "terminal"


@respx.mock
@pytest.mark.parametrize("status", [503, 429])
def test_a_loading_backend_through_the_door_is_a_502_with_its_words(
    tmp_path: Path, status: int
) -> None:
    respx.post(f"{BASE}/v1/rerank").mock(
        return_value=httpx.Response(
            status, json={"error": {"code": status, "message": "Loading model"}}
        )
    )
    with _app(tmp_path) as client:
        response = client.post("/v1/rerank", json={"query": "q", "documents": ["a"]})
    assert response.status_code == 502
    assert "Loading model" in _problem(response)["detail"]


def test_a_cli_subscription_does_not_sort_results(client: TestClient) -> None:
    """There is no flag that makes a coding harness score passages."""
    from eugene_plexus_inference_driver.engines.claude_code_cli import ClaudeCodeCliEngine

    client.app.state.adapter = ClaudeCodeCliEngine(binary_path="claude")  # type: ignore[attr-defined]

    response = client.post("/v1/rerank", json={"query": "q", "documents": ["a"]})

    assert response.status_code == 400
    problem = _problem(response)
    assert problem["type"].endswith("#rerank-unsupported")
    assert "does not sort results" in problem["detail"]


def test_a_driver_with_no_engine_says_why(client: TestClient) -> None:
    client.app.state.adapter = None  # type: ignore[attr-defined]
    client.app.state.adapter_error = "no backend set"  # type: ignore[attr-defined]
    response = client.post("/v1/rerank", json={"query": "q", "documents": ["a"]})
    assert response.status_code == 503
    assert "no backend set" in _problem(response)["detail"]


@respx.mock
def test_the_door_is_behind_the_same_auth_as_the_others(
    tmp_path: Path, install: FakeInstall
) -> None:
    """Registered with the others' `require_authorized`: no token is a 401, and
    the gateway's service token is let through (the 400 is this fixture's
    default engine, a CLI subscription, saying it does not sort results, which
    is the route and so past the door)."""
    app = create_app(settings=Settings(config_file=tmp_path / "config.yaml"))
    app.state.auth_state = install.auth_state()
    body = {"query": "q", "documents": ["a"]}
    with TestClient(app) as client:
        refused = client.post("/v1/rerank", json=body)
        token = install.service("gateway")
        passed = client.post("/v1/rerank", json=body, headers={"Authorization": f"Bearer {token}"})
    assert refused.status_code == 401
    assert passed.status_code == 400
    assert _problem(passed)["type"].endswith("#rerank-unsupported")


# --------------------------------------------------------------------- #
# /v1/info: a reranker is listed as one, and never asked to embed
# --------------------------------------------------------------------- #


def _llama_server_probes() -> dict[str, respx.Route]:
    """What the other `/v1/info` probes find on a local `llama-server`
    (measured on b10948 with `--reranking`)."""
    return {
        "props": respx.get(f"{BASE}/props").mock(
            return_value=httpx.Response(200, json={"default_generation_settings": {"n_ctx": 1024}})
        ),
        "infill": respx.post(f"{BASE}/infill").mock(
            return_value=httpx.Response(501, json={"error": {"message": "not supported"}})
        ),
        "models": respx.get(f"{BASE}/v1/models").mock(return_value=httpx.Response(404)),
        "ps": respx.get(f"{BASE}/api/ps").mock(return_value=httpx.Response(404)),
    }


@respx.mock
def test_info_lists_a_reranking_backend_as_rerank_and_never_asks_it_to_embed(
    tmp_path: Path,
) -> None:
    """Measured on b10948, Qwen3-Reranker: listed as `["chat", "completion"]`
    before MR6. It has to be a reranker, and the embeddings probe, which a
    reranker has been seen to die of, must not reach it."""
    rerank = _rerank_route(*PROBABILITIES)
    embed = respx.post(f"{BASE}/v1/embeddings").mock(
        return_value=httpx.Response(200, json={"data": [{"embedding": [0.0]}]})
    )
    _llama_server_probes()
    with _app(tmp_path) as client:
        first = client.get("/v1/info").json()
        asked = rerank.call_count
        second = client.get("/v1/info").json()

    (model,) = first["models"]
    assert model["surfaces"] == ["rerank"]
    assert second["models"][0]["surfaces"] == ["rerank"]
    assert embed.call_count == 0, "a reranker is never asked to embed"
    assert asked >= 1 and rerank.call_count == asked, "asked until it can say, and then not again"


@respx.mock
def test_info_keeps_listing_chat_until_a_loading_backend_can_say_it_reranks(
    tmp_path: Path,
) -> None:
    """`llama-server` answers 503 while its model loads, and the driver's first
    look lands in that window. Not an answer, so not kept."""
    rerank = respx.post(f"{BASE}/v1/rerank").mock(
        return_value=httpx.Response(503, json={"error": {"code": 503, "message": "Loading model"}})
    )
    embed = respx.post(f"{BASE}/v1/embeddings").mock(
        return_value=httpx.Response(200, json={"data": [{"embedding": [0.0]}]})
    )
    _llama_server_probes()
    with _app(tmp_path) as client:
        loading = client.get("/v1/info").json()["models"][0]["surfaces"]
        rerank.mock(return_value=_ok(*LOGITS))
        loaded = client.get("/v1/info").json()["models"][0]["surfaces"]

    assert "rerank" not in loading
    assert loaded == ["rerank"]
    assert embed.call_count == 0


@respx.mock
def test_info_still_lists_a_chat_backend_as_chat(tmp_path: Path) -> None:
    """The refusal a server not started with `--reranking` gives at once is the
    probe's definite no, and the reading of everything else is as it was."""
    respx.post(f"{BASE}/v1/rerank").mock(
        return_value=httpx.Response(501, json={"error": {"message": "start it with --reranking"}})
    )
    respx.post(f"{BASE}/v1/embeddings").mock(
        return_value=httpx.Response(501, json={"error": {"message": "start it with --embeddings"}})
    )
    _llama_server_probes()
    with _app(tmp_path) as client:
        surfaces = client.get("/v1/info").json()["models"][0]["surfaces"]
    assert surfaces == ["chat", "completion"]


# --------------------------------------------------------------------- #
# an account never reports `rerank`
# --------------------------------------------------------------------- #

OPENROUTER = "https://openrouter.ai/api"


def _listing(*models: dict[str, Any]) -> dict[str, Any]:
    return {"data": list(models)}


def _or_model(model_id: str, output: list[str]) -> dict[str, Any]:
    return {
        "id": model_id,
        "context_length": 8192,
        "architecture": {"input_modalities": ["text"], "output_modalities": output},
        "supported_parameters": ["max_tokens"],
    }


@respx.mock
def test_an_account_model_listed_only_for_reranking_serves_nothing(tmp_path: Path) -> None:
    """OpenRouter's own listing can say a model's output is `rerank`. No account
    engine sorts results, so the driver would list a surface it refuses: it
    reports none, and the model serves nothing here, as it did before the door."""
    respx.get(f"{OPENROUTER}/v1/models/user").mock(
        return_value=httpx.Response(
            200,
            json=_listing(
                _or_model("cohere/rerank-v3.5", ["rerank"]),
                _or_model("acme/chat-and-rerank", ["text", "rerank"]),
                _or_model("acme/chat", ["text"]),
            ),
        )
    )
    for listing in ("images", "videos"):
        respx.get(f"{OPENROUTER}/v1/{listing}/models").mock(
            return_value=httpx.Response(200, json={"data": []})
        )
    upstream = respx.post(f"{OPENROUTER}/v1/rerank").mock(return_value=_ok((0, 0.9)))
    config = tmp_path / "openrouter.yaml"
    config.write_text(json.dumps({"provider": "openrouter", "apiKey": "sk-or-test"}), "utf-8")

    with TestClient(create_app(settings=Settings(config_file=config))) as client:
        for _ in range(250):
            info = client.get("/v1/info").json()
            if (info.get("catalogue") or {}).get("refreshedAt"):
                break
        by_id = {m["id"]: m for m in info["models"]}
        refused = {
            model: client.post(
                "/v1/rerank", json={"model": model, "query": "q", "documents": ["a"]}
            )
            for model in ("cohere/rerank-v3.5", "acme/chat-and-rerank", "acme/chat")
        }
        unnamed = client.post("/v1/rerank", json={"query": "q", "documents": ["a"]})
        unknown = client.post(
            "/v1/rerank", json={"model": "no/such", "query": "q", "documents": ["a"]}
        )

    assert by_id["cohere/rerank-v3.5"]["surfaces"] == []
    assert by_id["cohere/rerank-v3.5"]["outputModalities"] == ["rerank"], "what it says stays said"
    assert by_id["acme/chat-and-rerank"]["surfaces"] == ["chat"]
    assert by_id["acme/chat"]["surfaces"] == ["chat"]
    for model, response in refused.items():
        assert response.status_code == 400, (model, response.text)
        assert _problem(response)["type"].endswith("#rerank-unsupported")
        assert "no account sorts results" in _problem(response)["detail"]
    assert _problem(unnamed)["type"].endswith("#model-required")
    assert unknown.status_code == 404
    assert not upstream.called


def test_a_model_list_kept_on_disk_by_an_older_driver_loses_rerank_on_the_way_in(
    tmp_path: Path,
) -> None:
    """The last good list is read back when the upstream is down at start, and
    an older driver kept `rerank` in it. It must not come back as a surface."""
    path = tmp_path / "openrouter.catalogue.json"
    path.write_text(
        json.dumps(
            {
                "origin": "openrouter|x",
                "source": "openrouter",
                "refreshedAt": "2026-10-11T10:00:00+00:00",
                "models": [
                    {"id": "cohere/rerank-v3.5", "surfaces": ["rerank"]},
                    {"id": "acme/both", "surfaces": ["chat", "rerank"]},
                ],
            }
        ),
        encoding="utf-8",
    )

    catalogue = Catalogue(
        source="openrouter", origin="openrouter|x", fetch=_never, get=lambda _k: None, path=path
    )

    assert {m.id: m.surfaces for m in catalogue.exposed()} == {
        "cohere/rerank-v3.5": [],
        "acme/both": ["chat"],
    }


async def _never() -> list:
    raise AssertionError("not read in this test")
