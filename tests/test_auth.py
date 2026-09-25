"""Bearer auth on the inference-driver, against this machine's trust bundle.

Verify-only role: the driver holds no key (per-node token keys,
2026-09-25). `FakeInstall` stands in for the agent and the root with a
real bundle on disk, and these assert the dependencies accept and refuse
the right shapes.

Auth posture is selected by whether `app.state.auth_state` is
pre-populated before the lifespan runs. Default fixtures leave it unset
-> lifespan reads env vars (empty) -> `auth_disabled=True`.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_inference_driver import tokens
from eugene_plexus_inference_driver.app import create_app
from eugene_plexus_inference_driver.auth_state import load_auth_state
from eugene_plexus_inference_driver.settings import Settings

from .conftest import FakeInstall


@pytest.fixture
def authed_app(tmp_path: Path, install: FakeInstall) -> FastAPI:
    settings = Settings(config_file=tmp_path / "config.yaml")
    app = create_app(settings=settings)
    # Pre-populate so the lifespan leaves it alone (hasattr is True).
    app.state.auth_state = install.auth_state()
    return app


@pytest.fixture
def authed_client(authed_app: FastAPI) -> Iterator[TestClient]:
    with TestClient(authed_app) as c:
        yield c


@pytest.fixture
def operator_token(install: FakeInstall) -> str:
    return install.session()


@pytest.fixture
def orchestrator_service_token(install: FakeInstall) -> str:
    """A typical inbound: the gateway beside this driver calling /v1/generate."""
    return install.service("gateway")


# --------------------------------------------------------------------------- #
# Auth-disabled path (default client fixture)
# --------------------------------------------------------------------------- #


def test_auth_disabled_lets_everything_through(client: TestClient) -> None:
    """No signing key wired in → every route answers normally without
    a bearer header. The dev / standalone posture; production via the
    agent supplies the env vars."""
    assert client.get("/healthz").status_code == 200
    assert client.get("/v1/config").status_code == 200


# --------------------------------------------------------------------------- #
# Health is always open
# --------------------------------------------------------------------------- #


def test_healthz_is_always_open(authed_client: TestClient) -> None:
    response = authed_client.get("/healthz")
    assert response.status_code == 200


# --------------------------------------------------------------------------- #
# Missing / malformed / wrong-key tokens reject with 401 + Problem JSON
# --------------------------------------------------------------------------- #


def test_missing_bearer_rejects_with_401(authed_client: TestClient) -> None:
    response = authed_client.get("/v1/config")
    assert response.status_code == 401
    assert response.json()["detail"]["component"] == "inference-driver"


def test_a_token_signed_by_a_key_the_bundle_does_not_list_rejects(
    authed_client: TestClient, install: FakeInstall
) -> None:
    stranger = tokens.Signer(key=tokens.generate_private_key(), issuer="control")
    token, _ = stranger.mint(
        typ=tokens.TYP_SESSION, sub="operator", aud=[install.recipient], ttl_seconds=60
    )
    response = authed_client.get("/v1/config", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401


def test_garbage_bearer_rejects(authed_client: TestClient) -> None:
    response = authed_client.get("/v1/config", headers={"Authorization": "Bearer not.a.jwt"})
    assert response.status_code == 401


def test_expired_token_rejects(authed_client: TestClient, install: FakeInstall) -> None:
    # Past the 300 s clock-skew leeway, not merely past exp.
    expired = install.session(ttl=60, now=int(time.time()) - 1000)
    response = authed_client.get("/v1/config", headers={"Authorization": f"Bearer {expired}"})
    assert response.status_code == 401


# --------------------------------------------------------------------------- #
# Operator-audience tokens accepted on operator-only routes
# --------------------------------------------------------------------------- #


def test_operator_token_accepted_on_config_get(
    authed_client: TestClient, operator_token: str
) -> None:
    response = authed_client.get(
        "/v1/config", headers={"Authorization": f"Bearer {operator_token}"}
    )
    assert response.status_code == 200


def test_operator_token_accepted_on_config_patch(
    authed_client: TestClient, operator_token: str
) -> None:
    response = authed_client.patch(
        "/v1/config",
        json={"logLevel": "DEBUG"},
        headers={"Authorization": f"Bearer {operator_token}"},
    )
    assert response.status_code == 200


# --------------------------------------------------------------------------- #
# Service-audience tokens accepted on mixed routes only
# --------------------------------------------------------------------------- #


def test_service_token_rejected_on_config_patch(
    authed_client: TestClient, orchestrator_service_token: str
) -> None:
    """A compromised gateway must not be able to rewrite the
    driver's config — operator audience only."""
    response = authed_client.patch(
        "/v1/config",
        json={"logLevel": "DEBUG"},
        headers={"Authorization": f"Bearer {orchestrator_service_token}"},
    )
    assert response.status_code == 401


def test_service_token_rejected_on_admin_restart(
    authed_client: TestClient, orchestrator_service_token: str
) -> None:
    response = authed_client.post(
        "/v1/admin/restart",
        headers={"Authorization": f"Bearer {orchestrator_service_token}"},
    )
    assert response.status_code == 401


def test_service_token_accepted_on_info(
    authed_client: TestClient, orchestrator_service_token: str
) -> None:
    """The gateway's drivers-list probe authenticates with a
    service token to read /v1/info. Must work."""
    response = authed_client.get(
        "/v1/info",
        headers={"Authorization": f"Bearer {orchestrator_service_token}"},
    )
    # No engine configured in tests, but the auth layer must let the
    # request through; the route then handles "no engine" on its own
    # terms. Asserting *not* 401 is what we actually care about.
    assert response.status_code != 401


def test_operator_token_accepted_on_info(authed_client: TestClient, operator_token: str) -> None:
    """UI dropdown population also hits /v1/info, with an operator
    token. Must work."""
    response = authed_client.get("/v1/info", headers={"Authorization": f"Bearer {operator_token}"})
    assert response.status_code != 401


# --------------------------------------------------------------------------- #
# load_auth_state contract
# --------------------------------------------------------------------------- #


def test_a_gateway_on_another_machine_may_call_with_the_grant(
    authed_client: TestClient, install: FakeInstall
) -> None:
    """The gateway on the control host calling a companion driver here."""
    install.far_grants = ("gateway",)
    install.publish()
    token = install.foreign_service("gateway")
    response = authed_client.get("/v1/info", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code != 401


def test_another_machines_components_call_nothing_here(
    authed_client: TestClient, install: FakeInstall
) -> None:
    """No `service:*` wildcard: another machine's agent, and a gateway
    token from a machine the operator never granted one, are refused."""
    for token in (install.foreign_service("agent"), install.foreign_service("gateway")):
        response = authed_client.get("/v1/info", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 401


def test_load_auth_state_disabled_when_nothing_is_supplied() -> None:
    state = load_auth_state(
        trust_bundle_file=None,
        trust_authority=None,
        auth_recipient=None,
        service_token=None,
        master_key_b64=None,
    )
    assert state.auth_disabled is True


@pytest.mark.parametrize(
    "missing", ["trust_bundle_file", "trust_authority", "auth_recipient", "service_token"]
)
def test_load_auth_state_refuses_a_partial_environment(install: FakeInstall, missing: str) -> None:
    env: dict[str, str | None] = {
        "trust_bundle_file": str(install.bundle_path),
        "trust_authority": install.authority,
        "auth_recipient": install.recipient,
        "service_token": install.service("inference-driver"),
        "master_key_b64": None,
    }
    env[missing] = None
    with pytest.raises(ValueError, match="missing"):
        load_auth_state(**env)  # type: ignore[arg-type]


def test_load_auth_state_rejects_a_misshaped_master_key(install: FakeInstall) -> None:
    import base64

    with pytest.raises(ValueError, match="32 bytes"):
        install.auth_state(master_key_b64=base64.b64encode(b"\x00" * 16).decode("ascii"))
