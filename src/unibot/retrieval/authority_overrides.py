"""Query-time authority tier adjustments.

Promotes sources that are contextually authoritative for the current query
intent without changing the stored base tier.
"""

from __future__ import annotations

import re

# (source_class_hint, url_pattern, tier_adjustment)
# Positive adjustment = demote, negative = promote.
_AUTHORITY_ADJUSTMENTS: tuple[tuple[str, str, int], ...] = (
    ("scholarship", r"financial-assistance|scholarship", -2),
    ("policy", r"policy|policies|regulation|regulations", -1),
)


def adjusted_authority_tier(
    base_tier: int,
    source_url: str,
    source_class_hint: str | None,
) -> int:
    """Return an adjusted authority tier for query-time scoring.

    When *source_class_hint* matches a known pattern in *source_url*, the
    tier is adjusted (lower = more authoritative).  The result is clamped
    to a minimum of 1.
    """
    if source_class_hint is None:
        return base_tier
    for hint, pattern, adjustment in _AUTHORITY_ADJUSTMENTS:
        if hint == source_class_hint and re.search(pattern, source_url, re.IGNORECASE):
            return max(1, base_tier + adjustment)
    return base_tier
