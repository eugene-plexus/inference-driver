import sys
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import httpx
import pytest

from eugene_plexus_inference_driver._generated.models import GenerateRequest
from eugene_plexus_inference_driver.engines._subprocess import CliError
from eugene_plexus_inference_driver.engines.codex_cli import CodexCliEngine
from eugene_plexus_inference_driver.failures import disposition, retry_after
from eugene_plexus_inference_driver.routes.generate import _backend_error


@pytest.mark.parametrize(
    "status,expected",
    [
        (429, "safe"),
        (400, "terminal"),
        (401, "terminal"),
        (408, "indeterminate"),
        (409, "indeterminate"),
        (425, "indeterminate"),
        (500, "indeterminate"),
        (503, "indeterminate"),
    ],
)
def test_status_does_not_flatten_execution_outcomes(status, expected):
    failure = _backend_error(
        CliError("fixture", upstream_status=status, retry_after_seconds=13), "fixture"
    )
    assert failure.detail["retryDisposition"] == expected
    assert failure.detail["retryAfterSeconds"] == 13
    assert failure.headers["Retry-After"] == "13"


@pytest.mark.parametrize(
    "cause,expected",
    [
        (httpx.ConnectError("no route"), "safe"),
        (httpx.ConnectTimeout("no connection"), "safe"),
        (httpx.ReadError("accepted then lost"), "indeterminate"),
        (httpx.WriteError("partial"), "indeterminate"),
    ],
)
def test_transport_phase_is_preserved(cause, expected):
    try:
        raise CliError("wrapped transport") from cause
    except CliError as error:
        assert disposition(error) == expected


async def test_cli_side_effect_followed_by_failure_has_unknown_outcome(tmp_path):
    effect = tmp_path / "effect.txt"
    code = (
        "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('acted'); sys.exit(2)"
    )
    engine = CodexCliEngine()
    engine._build_argv = lambda prompt: [sys.executable, "-c", code, str(effect)]
    with pytest.raises(CliError) as caught:
        await engine.generate(GenerateRequest(messages=[]))
    assert effect.read_text() == "acted"
    assert disposition(caught.value) == "indeterminate"


def test_delta_and_http_date_retry_hints():
    assert retry_after("17") == 17
    assert 18 <= retry_after(format_datetime(datetime.now(UTC) + timedelta(seconds=20))) <= 20
    assert retry_after("-2") == 0
    for value in (None, "", "tomorrow", "nan", "inf"):
        assert retry_after(value) is None
