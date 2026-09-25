"""FastAPI dependencies for bearer auth, against this machine's trust bundle.

`require_authorized` (`/v1/info`, `/v1/generate`, ...) takes an operator
session addressed to this machine, a service token from this machine's
own components, or a `gateway` token from any machine the operator
granted one (per-node token keys, D5): the gateway on the control host
calling a companion driver on a GPU node. **No `service:*` wildcard**:
another machine's agent, library or driver calls nothing here.
`require_operator` takes a session only. Both pass through when
`AuthState.auth_disabled` is true (the dev path). All rejection paths
produce 7807-style Problem JSON so the UI can render one error template.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from . import tokens
from ._generated.models import Problem
from .auth_state import AuthState

_bearer_scheme = HTTPBearer(auto_error=False)


def _problem(status_code: int, title: str, detail: str) -> HTTPException:
    slug = title.replace(" ", "-").lower()
    return HTTPException(
        status_code=status_code,
        detail=Problem(
            type=f"https://github.com/eugene-plexus/inference-driver#{slug}",
            title=title,
            status=status_code,
            detail=detail,
            component="inference-driver",
        ).model_dump(exclude_none=True),
    )


_REMOTE_CALLERS = frozenset({tokens.SUB_GATEWAY})


def _validate(
    request: Request,
    creds: HTTPAuthorizationCredentials | None,
    *,
    reads: bool,
) -> tokens.Claims | None:
    auth: AuthState = request.app.state.auth_state
    if auth.auth_disabled:
        return None
    if creds is None or not creds.credentials:
        raise _problem(
            status.HTTP_401_UNAUTHORIZED,
            "Missing token",
            "Provide a bearer token via the Authorization: Bearer header.",
        )
    classes = (tokens.TYP_SESSION, tokens.TYP_SERVICE) if reads else (tokens.TYP_SESSION,)
    try:
        claims = auth.verify(creds.credentials, classes=classes)
    except tokens.TokenError as e:
        raise _problem(
            status.HTTP_401_UNAUTHORIZED,
            "Invalid token",
            f"Bearer token rejected: {e}",
        ) from e
    if claims.is_service and not (
        claims.is_local_service(str(auth.recipient)) or claims.sub in _REMOTE_CALLERS
    ):
        raise _problem(
            status.HTTP_401_UNAUTHORIZED,
            "Wrong audience",
            f"A {claims.sub!r} service token from {claims.iss!r} may not call this driver; "
            "only this machine's components and the install's gateway may.",
        )
    return claims


def require_authorized(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> tokens.Claims | None:
    """A session, this machine's services, or the gateway from any machine."""
    return _validate(request, creds, reads=True)


def require_operator(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> tokens.Claims | None:
    """An operator session only -- for config edits and admin/restart."""
    return _validate(request, creds, reads=False)
