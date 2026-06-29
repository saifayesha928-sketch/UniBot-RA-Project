from __future__ import annotations

import hmac
from typing import Any

from fastapi import Depends, HTTPException, Request
from fastapi.security import APIKeyHeader
from qdrant_client import QdrantClient
from sqlalchemy.orm import Session

from unibot.api.serialization import serialize_datetime
from unibot.db.models import ServingGeneration
from unibot.db.session import get_runtime_session_factory


def get_session(request: Request) -> tuple[Session, bool]:
    session_factory = getattr(request.app.state, "session_factory", None)
    if session_factory is None:
        session_factory = get_runtime_session_factory()
    close_session = bool(getattr(request.app.state, "close_sessions", True))
    return session_factory(), close_session


def get_qdrant_client(request: Request) -> QdrantClient:
    client: QdrantClient | None = getattr(request.app.state, "qdrant_client", None)
    if client is not None:
        return client

    from unibot.settings import get_settings

    settings = get_settings()
    return QdrantClient(url=str(settings.qdrant_url), api_key=settings.qdrant_api_key)


def serialize_generation(generation: ServingGeneration) -> dict[str, Any]:
    return {
        "generation_id": generation.generation_id,
        "generation_label": generation.generation_label,
        "status": generation.status,
        "qdrant_collection": generation.qdrant_collection,
        "generation_metadata": generation.generation_metadata,
        "activated_at": serialize_datetime(generation.activated_at),
        "created_at": serialize_datetime(generation.created_at),
    }


def serialize_generation_slim(generation: ServingGeneration) -> dict[str, Any]:
    """Lightweight generation info for query responses (no metadata blob)."""
    return {
        "generation_id": generation.generation_id,
        "generation_label": generation.generation_label,
        "status": generation.status,
        "activated_at": serialize_datetime(generation.activated_at),
    }


admin_key_header = APIKeyHeader(name="X-Admin-Key", auto_error=False)


def require_admin_key(
    request: Request,
    provided_key: str | None = Depends(admin_key_header),
) -> None:
    if not bool(getattr(request.app.state, "enable_admin_auth", True)):
        return

    expected_key = getattr(request.app.state, "admin_api_key", None)
    if not expected_key:
        raise HTTPException(status_code=503, detail="Admin API key is not configured.")
    if not hmac.compare_digest(provided_key or "", expected_key):
        raise HTTPException(status_code=401, detail="Invalid or missing admin API key")
