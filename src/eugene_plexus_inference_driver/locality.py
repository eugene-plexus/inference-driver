"""Routing trust belongs to the configured engine, never its URL spelling."""

from typing import Any

from fastapi import HTTPException

from ._generated.models import Locality, Problem

DECLARABLE_PROVIDERS = frozenset(
    {"openai_compat_custom", "ollama_local", "lmstudio_local", "systemone_custom"}
)


def classify(provider: str, declared: Any, *, managed: bool) -> str:
    if provider not in DECLARABLE_PROVIDERS:
        return "external"
    if managed:
        return "local"
    return declared if declared in ("local", "external") else "unknown"


def engine_locality(engine: Any) -> Locality:
    value = getattr(engine, "routing_locality", None)
    return Locality(value) if value in ("local", "external") else Locality.unknown


def enforce(engine: Any, local_only: bool | None) -> None:
    if local_only and engine_locality(engine) != Locality.local:
        raise HTTPException(
            status_code=403,
            detail=Problem(
                type="https://github.com/eugene-plexus/inference-driver#local-only",
                title="Local-only policy refused this backend",
                status=403,
                detail="This local-only request requires a confirmed local active engine. "
                "No prompt, image or embedding input was forwarded.",
                component="inference-driver",
            ).model_dump(exclude_none=True),
        )
