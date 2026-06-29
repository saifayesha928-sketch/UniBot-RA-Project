from __future__ import annotations

from datetime import datetime


def serialize_datetime(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()
