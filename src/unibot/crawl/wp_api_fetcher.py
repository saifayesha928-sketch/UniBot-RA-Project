"""WordPress REST API fetcher.

Fetches page content via the WP REST API (/wp-json/wp/v2/pages) instead of
HTML crawling.  Returns a FetchedArtifact identical in shape to the crawl-based
fetchers so the extraction layer needs zero changes.
"""

from __future__ import annotations

from urllib.parse import urlencode, urlsplit

import httpx
import structlog

from unibot.crawl.fetchers import FetchedArtifact
from unibot.crawl.source_identity import (
    SourceIdentityAmbiguousError,
    resolve_wordpress_page_match,
)

logger = structlog.get_logger(__name__)

_WP_API_FIELDS = "id,slug,title,content,modified,link"
_WP_API_PATH = "/wp-json/wp/v2/pages"


def _slug_from_url(url: str) -> str:
    """Extract the last non-empty path segment as the WordPress page slug."""
    parts = urlsplit(url)
    segments = [s for s in parts.path.split("/") if s]
    if not segments:
        raise ValueError(f"cannot extract slug from URL: {url}")
    return segments[-1]


def _wp_api_url(source_url: str) -> str:
    """Build the WP REST API query URL for a given source page URL."""
    parts = urlsplit(source_url)
    slug = _slug_from_url(source_url)
    query = urlencode({"slug": slug, "_fields": _WP_API_FIELDS})
    return f"{parts.scheme}://{parts.netloc}{_WP_API_PATH}?{query}"


class WordPressAPIFetcher:
    """Fetch page content from the WordPress REST API."""

    def __init__(self, *, timeout: float = 30.0) -> None:
        self._timeout = timeout

    def fetch(self, url: str) -> FetchedArtifact:
        api_url = _wp_api_url(url)

        response = httpx.get(
            api_url,
            timeout=self._timeout,
            follow_redirects=True,
        )
        response.raise_for_status()

        pages = response.json()
        if not pages:
            raise ValueError(
                f"WordPress API returned no results for slug "
                f"'{_slug_from_url(url)}' — page not found"
            )

        page = resolve_wordpress_page_match(requested_url=url, pages=pages)
        if page is None:
            slug = _slug_from_url(url)
            page_links = [p.get("link", "?") for p in pages]
            raise SourceIdentityAmbiguousError(
                f"Slug '{slug}' matched {len(pages)} pages but none match "
                f"requested URL '{url}'. Candidates: {page_links}"
            )

        identity_match_type = "exact_link"
        content_html = page.get("content", {}).get("rendered", "")
        content_html = f'<main id="main">{content_html}</main>'

        return FetchedArtifact(
            source_url=url,
            content=content_html.encode("utf-8"),
            content_type="text/html",
            fetch_method="wordpress_api",
            http_status=response.status_code,
            metadata={
                "wp_page_id": page.get("id"),
                "wp_modified": page.get("modified"),
                "requested_url": url,
                "resolved_wp_link": page.get("link"),
                "identity_match_type": identity_match_type,
            },
        )
