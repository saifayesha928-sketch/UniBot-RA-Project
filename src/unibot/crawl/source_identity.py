"""Exact source identity resolution for WordPress API fetches.

Resolves slug-collision ambiguity by comparing the requested URL against
the ``link`` field returned by the WordPress REST API, rather than blindly
picking ``pages[0]``.
"""

from __future__ import annotations

from unibot.utils import normalize_url_identity


class SourceIdentityAmbiguousError(Exception):
    """Raised when a WordPress slug matches multiple pages and none match the requested URL."""


def resolve_wordpress_page_match(
    *,
    requested_url: str,
    pages: list[dict],
) -> dict | None:
    """Return the single page whose ``link`` matches *requested_url*, or None.

    When the WordPress API returns multiple pages for the same slug (e.g.
    ``/about/`` vs ``/unesco/about/``), this function picks the one whose
    ``link`` field matches the requested URL exactly (after normalization).

    Returns ``None`` when no page matches or when the match is ambiguous.
    """
    requested_identity = normalize_url_identity(requested_url)
    exact_matches = [
        page
        for page in pages
        if normalize_url_identity(page.get("link", "")) == requested_identity
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]
    return None
