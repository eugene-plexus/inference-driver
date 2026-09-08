"""FastAPI dependencies for v0.2 bearer auth.

Same shape as the gateway: `require_authorized` accepts operator
OR any `service:*` audience; `require_operator` accepts operator only.
Both pass-through when `AuthState.auth_disabled` is true (the dev
path). All auth-rejection paths produce 7807-style Problem JSON so
the UI can render one error template across components.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from . import security
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


def _validate(
    request: Request,
    creds: HTTPAuthorizationCredentials | None,
    *,
    accept_operator: bool,
    accept_any_service: bool,
) -> security.TokenPayload | None:
    auth: AuthState = request.app.state.auth_state
    if auth.auth_disabled:
        return None
    if creds is None or not creds.credentials:
        raise _problem(
            status.HTTP_401_UNAUTHORIZED,
            "Missing token",
            "Provide a bearer token via the Authorization: Bearer header.",
        )
    assert auth.signing_key is not None
    try:
        return security.decode_token(
            token=creds.credentials,
            signing_key=auth.signing_key,
            accept_operator=accept_operator,
            accept_any_service=accept_any_service,
        )
    except Exception as e:
        raise _problem(
            status.HTTP_401_UNAUTHORIZED,
            "Invalid token",
            f"Bearer token rejected: {e}",
        ) from e


def require_authorized(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> security.TokenPayload | None:
    """Operator OR any service-audience token accepted."""
    return _validate(request, creds, accept_operator=True, accept_any_service=True)


def require_operator(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> security.TokenPayload | None:
    """Operator-audience tokens only — for config edits and admin/restart."""
    return _validate(request, creds, accept_operator=True, accept_any_service=False)
