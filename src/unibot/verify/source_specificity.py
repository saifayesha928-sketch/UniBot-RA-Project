from __future__ import annotations

from urllib.parse import urlparse

_SPECIFICITY_KEYWORDS = ("norms", "policy", "criteria", "eligibility", "schedule")


def compute_source_specificity(source_url: str, record_type: str) -> int:
    """Higher value = more specific source. Used as within-tier tiebreaker."""
    path = urlparse(source_url).path.rstrip("/")
    depth = len([seg for seg in path.split("/") if seg])
    keyword_bonus = 2 if any(kw in path.lower() for kw in _SPECIFICITY_KEYWORDS) else 0
    return depth + keyword_bonus
