"""Pytest fixtures shared across the test suite."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_inference_driver import tokens
from eugene_plexus_inference_driver.app import create_app
from eugene_plexus_inference_driver.auth_state import AuthState, load_auth_state
from eugene_plexus_inference_driver.settings import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(config_file=tmp_path / "config.yaml")


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    return create_app(settings=settings)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


# --------------------------------------------------------------------------- #
# Trust (per-node token keys, 2026-09-25)
# --------------------------------------------------------------------------- #


@dataclass
class FakeInstall:
    """A trust bundle as this component's agent would keep it on disk.

    The component runs on node `gw`. The root's token key signs sessions
    and client keys; `gw`'s own key signs the tokens its agent hands its
    children; `far` is another machine of the install.
    """

    directory: Path
    name: str = "gw"
    grants: tuple[str, ...] = ()
    far_grants: tuple[str, ...] = ()
    identity: Ed25519PrivateKey = field(default_factory=tokens.generate_private_key)
    root: tokens.Signer = field(
        default_factory=lambda: tokens.Signer(key=tokens.generate_private_key(), issuer="control")
    )
    node: tokens.Signer | None = None
    far: tokens.Signer = field(
        default_factory=lambda: tokens.Signer(key=tokens.generate_private_key(), issuer="node:far")
    )
    version: int = 0

    def __post_init__(self) -> None:
        if self.node is None:
            self.node = tokens.Signer(key=tokens.generate_private_key(), issuer=self.recipient)
        self.publish()

    @property
    def recipient(self) -> str:
        return tokens.node_recipient(self.name)

    @property
    def authority(self) -> str:
        return tokens.public_b64(self.identity)

    @property
    def bundle_path(self) -> Path:
        return self.directory / "trust_bundle.json"

    def publish(self, *, revoked: tuple[tuple[str, int], ...] = ()) -> tokens.TrustBundle:
        assert self.node is not None
        self.version += 1
        bundle = tokens.build_bundle(
            authority=self.identity,
            version=self.version,
            epoch=1,
            keys=[
                self.root.trust_key(["authority"]),
                self.node.trust_key(["node", *self.grants]),
                self.far.trust_key(["node", *self.far_grants]),
            ],
            revoked_sessions=revoked,
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        tokens.write_bundle_file(self.bundle_path, bundle)
        return bundle

    def auth_state(self, *, master_key_b64: str | None = None) -> AuthState:
        return load_auth_state(
            trust_bundle_file=str(self.bundle_path),
            trust_authority=self.authority,
            auth_recipient=self.recipient,
            service_token=self.service("gateway", ttl=365 * 24 * 3600),
            master_key_b64=master_key_b64,
        )

    def session(
        self,
        *,
        sub: str = "operator",
        ttl: int = 3600,
        aud: list[str] | None = None,
        now: int | None = None,
    ) -> str:
        token, _ = self.root.mint(
            typ=tokens.TYP_SESSION,
            sub=sub,
            aud=aud or [self.recipient, "control"],
            ttl_seconds=ttl,
            now=now,
        )
        return token

    def client_key(
        self, *, name: str = "app", jti: str = "key-1", ttl: int = 3600, now: int | None = None
    ) -> str:
        token, _ = self.root.mint(
            typ=tokens.TYP_CLIENT, sub=name, aud=["gateway"], ttl_seconds=ttl, now=now, jti=jti
        )
        return token

    def service(self, sub: str = "gateway", *, ttl: int = 3600) -> str:
        """A token this machine's agent minted for one of its children."""
        assert self.node is not None
        token, _ = self.node.mint(
            typ=tokens.TYP_SERVICE, sub=sub, aud=[self.recipient], ttl_seconds=ttl
        )
        return token

    def foreign_service(self, sub: str = "agent", *, ttl: int = 600) -> str:
        """Another machine's token, addressed here and correctly signed."""
        token, _ = self.far.mint(
            typ=tokens.TYP_SERVICE, sub=sub, aud=[self.recipient], ttl_seconds=ttl
        )
        return token

    def raw(self, signer: tokens.Signer, typ: str, **claims: Any) -> str:
        """A token with exactly these claims, for the shapes `mint` will not make."""
        now = int(time.time())
        body: dict[str, Any] = {"iss": signer.issuer, "iat": now, "exp": now + 60, **claims}
        body = {k: v for k, v in body.items() if v is not None}
        return jwt.encode(
            body, signer.key, algorithm="EdDSA", headers={"typ": typ, "kid": signer.kid}
        )


@pytest.fixture
def install(tmp_path: Path) -> FakeInstall:
    return FakeInstall(tmp_path / "node")
