from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import httpx
import structlog

from unibot.crawl.crawl4ai_fetcher import (
    Crawl4AIBatchFetcher,
    Crawl4AIBrowserFetcher,
    Crawl4AIStaticFetcher,
)


@dataclass(frozen=True, slots=True)
class FetchedArtifact:
    source_url: str
    content: bytes
    content_type: str
    fetch_method: str
    http_status: int
    etag: str | None = None
    last_modified: str | None = None
    requires_browser: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


def _normalize_to_main(artifact: FetchedArtifact) -> FetchedArtifact:
    """Extract <main> content from full HTML for consistency with WP API path."""
    content_type = artifact.content_type or ""
    if "html" not in content_type.casefold():
        return artifact
    try:
        html_text = artifact.content.decode("utf-8", errors="replace")
    except Exception:
        return artifact
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_text, "lxml")
    main_el = soup.find("main")
    if main_el is None:
        return artifact
    main_html = f'<main id="main">{main_el.decode_contents()}</main>'
    return FetchedArtifact(
        source_url=artifact.source_url,
        content=main_html.encode("utf-8"),
        content_type=artifact.content_type,
        fetch_method=artifact.fetch_method,
        http_status=artifact.http_status,
        etag=artifact.etag,
        last_modified=artifact.last_modified,
        requires_browser=artifact.requires_browser,
        metadata=artifact.metadata,
    )


class RawArtifactFetcher:
    def __init__(
        self,
        *,
        html_fetcher: Callable[[str], FetchedArtifact] | None = None,
        browser_fetcher: Callable[[str], FetchedArtifact] | None = None,
        wp_api_fetcher: Callable[[str], FetchedArtifact] | None = None,
        html_batch_fetcher: Callable[[tuple[str, ...] | list[str]], tuple[FetchedArtifact, ...]]
        | None = None,
        browser_batch_fetcher: Callable[[tuple[str, ...] | list[str]], tuple[FetchedArtifact, ...]]
        | None = None,
    ) -> None:
        self._html_fetcher = html_fetcher or Crawl4AIStaticFetcher().fetch
        self._browser_fetcher = browser_fetcher or Crawl4AIBrowserFetcher().fetch
        self._wp_api_fetcher = wp_api_fetcher
        self._html_batch_fetcher = html_batch_fetcher or Crawl4AIBatchFetcher().fetch_many
        self._browser_batch_fetcher = (
            browser_batch_fetcher
            or Crawl4AIBatchFetcher(requires_browser=True).fetch_many
        )

    def fetch(self, url: str, *, requires_browser: bool = False) -> FetchedArtifact:
        if not requires_browser:
            return self._html_fetcher(url)

        return self._browser_fetcher(url)

    def fetch_wp_api(self, url: str) -> FetchedArtifact:
        if self._wp_api_fetcher is None:
            from unibot.crawl.wp_api_fetcher import WordPressAPIFetcher

            self._wp_api_fetcher = WordPressAPIFetcher().fetch
        try:
            return self._wp_api_fetcher(url)
        except Exception as exc:
            from unibot.crawl.source_identity import SourceIdentityAmbiguousError

            if isinstance(exc, SourceIdentityAmbiguousError):
                raise
            structlog.get_logger(__name__).warning(
                "crawl.wp_api_failed_falling_back_to_html",
                url=url,
                exc_info=True,
            )
            artifact = self._html_fetcher(url)
            # Normalize to <main> content for consistency with WP API path
            artifact = _normalize_to_main(artifact)
            return artifact

    def fetch_binary(self, url: str, *, timeout: float = 60.0) -> FetchedArtifact:
        with httpx.Client(follow_redirects=True, timeout=timeout) as client:
            response = client.get(url)
            response.raise_for_status()

        return FetchedArtifact(
            source_url=str(response.url),
            content=response.content,
            content_type=response.headers.get("content-type", "application/octet-stream"),
            fetch_method="http_direct",
            http_status=response.status_code,
            etag=response.headers.get("etag"),
            last_modified=response.headers.get("last-modified"),
        )

    def fetch_many(
        self,
        urls: tuple[str, ...] | list[str],
        *,
        requires_browser: bool = False,
    ) -> tuple[FetchedArtifact, ...]:
        if not urls:
            return ()
        if not requires_browser:
            return tuple(self._html_batch_fetcher(urls))
        return tuple(self._browser_batch_fetcher(urls))
