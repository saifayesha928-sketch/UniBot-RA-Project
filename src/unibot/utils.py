from __future__ import annotations

from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_url_identity(url: str) -> str:
    parts = urlsplit(url.strip())
    path = parts.path or "/"
    return urlunsplit((
        parts.scheme.lower(),
        (parts.hostname or "").lower(),
        path.rstrip("/") + "/",
        "",
        "",
    ))
