from __future__ import annotations

import hashlib
import re


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def safe_dedupe_slug(value: str, *, max_length: int = 200) -> str:
    """Slugify *value* and truncate to *max_length*, appending a hash when truncated."""
    slug = slugify(value)
    return clamp_str(slug, max_length=max_length)


def clamp_str(value: str, *, max_length: int = 255) -> str:
    """Return *value* truncated to *max_length* with a hash suffix to avoid collisions."""
    if len(value) <= max_length:
        return value
    digest = hashlib.sha256(value.encode()).hexdigest()[:12]
    return f"{value[: max_length - 13]}-{digest}"


def stable_slug_with_hash(value: str, *, max_length: int = 200) -> str:
    """Slugify *value* and always append a short digest of the original input.

    This guarantees that inputs which slugify to the same string (e.g.
    ``"Serving New"`` vs ``"serving/new"``) still produce distinct names.
    """
    slug = slugify(value) or "generation"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    combined = f"{slug}-{digest}"
    if len(combined) <= max_length:
        return combined
    return f"{slug[: max_length - 11]}-{digest}"
