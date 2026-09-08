"""POST /v1/admin/restart — schedule a process exit so a supervisor relaunches us.

The driver only re-reads `requiresRestart: true` config keys at startup;
this endpoint is the UI's mechanism for completing a config-change flow
that touches one of those keys. We respond first, then exit shortly
after so the HTTP response has time to flush.
"""

from __future__ import annotations

import asyncio
import logging
import os

from fastapi import APIRouter

from .._generated.models import RestartResult

router = APIRouter(tags=["admin"])

# Long enough for the 202 response body to flush back to the client over
# a slow LAN, short enough that the operator doesn't sit waiting. Tune
# only if real users report seeing a hung "restarting…" dialog.
_EXIT_DELAY_MS = 500


@router.post("/v1/admin/restart", response_model=RestartResult, status_code=202)
async def restart() -> RestartResult:
    log = logging.getLogger(__name__)
    log.warning("restart requested via /v1/admin/restart; exiting in %dms", _EXIT_DELAY_MS)

    loop = asyncio.get_event_loop()
    loop.call_later(_EXIT_DELAY_MS / 1000.0, lambda: os._exit(0))

    return RestartResult(
        scheduled=True,
        delayMs=_EXIT_DELAY_MS,
        message=(
            f"Process exiting in {_EXIT_DELAY_MS}ms. A supervisor (systemd, "
            "docker, deploy launcher, …) is expected to relaunch it; in v0.1 "
            "personal-use installs without one, relaunch manually."
        ),
    )
