"""Tests for v0.2 bearer auth on the inference-driver.

Verify-only role — the agent issues tokens; this component just
validates them. Tests construct JWTs directly via PyJWT against a
known signing key (standing in for the agent) and assert the
dependencies accept / reject the right shapes.

Auth posture is selected by whether `app.state.auth_state` is
pre-populated before the lifespan runs. Default fixtures leave it
unset → lifespan reads env vars (empty) → `auth_disabled=True`, so
existing tests keep working unchanged.
"""

from __future__ import annotations

import secrets
import time
from collections.abc import Iterator
from pathlib import Path

import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_inference_driver.app import create_app
from eugene_plexus_inference_driver.auth_state import AuthState
from eugene_plexus_inference_driver.settings import Settings

_JWT_ALG = "HS256"


def _issue(
    *,
    signing_key: bytes,
    sub: str,
    aud: str,
    ttl_seconds: int = 60,
    iat: int | None = None,
) -> str:
    """Mint a JWT exactly the way the agent would."""
    issued_at = iat if iat is not None else int(time.time())
    claims = {
        "sub": sub,
        "aud": aud,
        "iat": issued_at,
        "exp": issued_at + ttl_seconds,
    }
    return jwt.encode(claims, signing_key, algorithm=_JWT_ALG)


@pytest.fixture
def signing_key() -> bytes:
    return secrets.token_bytes(32)


@pytest.fixture
def authed_app(tmp_path: Path, signing_key: bytes) -> FastAPI:
    settings = Settings(config_file=tmp_path / "config.yaml")
    app = create_app(settings=settings)
    # Pre-populate so the lifespan leaves it alone (hasattr is True).
    app.state.auth_state = AuthState(
        signing_key=signing_key,
        service_token=_issue(
            signing_key=signing_key,
            sub="inference-driver",
            aud="service:inference-driver",
            ttl_seconds=365 * 24 * 3600,
        ),
        master_key=None,
    )
    return app


@pytest.fixture
def authed_client(authed_app: FastAPI) -> Iterator[TestClient]:
    with TestClient(authed_app) as c:
        yield c


@pytest.fixture
def operator_token(signing_key: bytes) -> str:
    return _issue(signing_key=signing_key, sub="operator", aud="operator")


@pytest.fixture
def orchestrator_service_token(signing_key: bytes) -> str:
    """A typical inbound: gateway calling /v1/generate."""
    return _issue(signing_key=signing_key, sub="gateway", aud="service:gateway")


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


def test_wrong_signing_key_rejects(authed_client: TestClient) -> None:
    other = secrets.token_bytes(32)
    token = _issue(signing_key=other, sub="operator", aud="operator")
    response = authed_client.get("/v1/config", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401


def test_garbage_bearer_rejects(authed_client: TestClient) -> None:
    response = authed_client.get("/v1/config", headers={"Authorization": "Bearer not.a.real.jwt"})
    assert response.status_code == 401


def test_expired_token_rejects(authed_client: TestClient, signing_key: bytes) -> None:
    expired = _issue(
        signing_key=signing_key,
        sub="operator",
        aud="operator",
        ttl_seconds=-60,
        iat=int(time.time()) - 120,
    )
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


def test_load_auth_state_disabled_when_no_signing_key() -> None:
    from eugene_plexus_inference_driver.auth_state import load_auth_state

    state = load_auth_state(signing_key_b64=None, service_token=None, master_key_b64=None)
    assert state.auth_disabled is True


def test_load_auth_state_rejects_partial_auth() -> None:
    """SERVICE_TOKEN without AUTH_SIGNING_KEY is a configuration bug."""
    from eugene_plexus_inference_driver.auth_state import load_auth_state

    with pytest.raises(ValueError, match="inconsistent"):
        load_auth_state(
            signing_key_b64=None,
            service_token="dummy",
            master_key_b64=None,
        )


def test_load_auth_state_allows_signing_key_without_service_token(
    signing_key: bytes,
) -> None:
    """Unlike the gateway, the inference-driver doesn't need a
    service token for outbound calls. AUTH_SIGNING_KEY alone is a valid
    posture — the driver can validate inbound traffic without ever
    needing to authenticate outbound."""
    import base64

    from eugene_plexus_inference_driver.auth_state import load_auth_state

    state = load_auth_state(
        signing_key_b64=base64.b64encode(signing_key).decode("ascii"),
        service_token=None,
        master_key_b64=None,
    )
    assert state.signing_key == signing_key
    assert state.service_token is None
    assert state.auth_disabled is False


def test_load_auth_state_rejects_wrong_length_signing_key() -> None:
    import base64

    from eugene_plexus_inference_driver.auth_state import load_auth_state

    short = base64.b64encode(b"\x00" * 16).decode("ascii")
    with pytest.raises(ValueError, match="32 bytes"):
        load_auth_state(
            signing_key_b64=short,
            service_token=None,
            master_key_b64=None,
        )
