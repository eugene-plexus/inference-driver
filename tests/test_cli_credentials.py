"""A backend CLI is outside the control-plane credential boundary."""

import json
import sys

import pytest

from eugene_plexus_inference_driver.engines._subprocess import (
    _utf8_subprocess_env,
    run_cli,
    stream_cli_lines,
)


@pytest.mark.parametrize(
    "name",
    [
        "EUGENE_PLEXUS_DRIVER_MASTER_KEY",
        "EUGENE_PLEXUS_DRIVER_AUTH_SIGNING_KEY",
        "EUGENE_PLEXUS_DRIVER_SERVICE_TOKEN",
        "eugene_plexus_agent_master_key",
        "EUGENE_PLEXUS_CONTROL_PASSPHRASE_FILE",
    ],
)
def test_cli_never_inherits_control_plane_credentials(name, monkeypatch):
    monkeypatch.setenv(name, "private-to-plexus")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "operator-backend-key")
    monkeypatch.setenv("HTTP_PROXY", "http://operator-proxy:8000")
    env = _utf8_subprocess_env()
    assert name.upper() not in {key.upper() for key in env}
    assert env["ANTHROPIC_API_KEY"] == "operator-backend-key"
    assert env["HTTP_PROXY"] == "http://operator-proxy:8000"


@pytest.mark.parametrize("streaming", [False, True])
async def test_real_cli_child_receives_only_backend_credentials(streaming, monkeypatch):
    monkeypatch.setenv("EUGENE_PLEXUS_DRIVER_MASTER_KEY", "dummy-master")
    monkeypatch.setenv("EUGENE_PLEXUS_DRIVER_AUTH_SIGNING_KEY", "dummy-signing")
    monkeypatch.setenv("EUGENE_PLEXUS_DRIVER_SERVICE_TOKEN", "dummy-service")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-backend")
    # Report names only; never print actual credentials or use a real backend.
    argv = [sys.executable, "-c", "import json, os; print(json.dumps(sorted(os.environ)))"]
    if streaming:
        lines = [line async for line in stream_cli_lines(argv, timeout_seconds=10)]
        names = json.loads(lines[0])
    else:
        result = await run_cli(argv, timeout_seconds=10)
        assert result.returncode == 0
        names = json.loads(result.stdout)
    assert not any(key.upper().startswith("EUGENE_PLEXUS_") for key in names)
    assert "ANTHROPIC_API_KEY" in names
