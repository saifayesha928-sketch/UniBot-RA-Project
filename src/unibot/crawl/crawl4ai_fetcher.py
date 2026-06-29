from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import structlog

from unibot.crawl.async_runner import run_sync
from crawl4ai import (  # type: ignore[import-untyped]
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    HTTPCrawlerConfig,
    SemaphoreDispatcher,
)
from crawl4ai.async_crawler_strategy import AsyncHTTPCrawlerStrategy  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from unibot.crawl.fetchers import FetchedArtifact


logger = structlog.get_logger(__name__)


def _header(headers: dict[str, Any], key: str) -> str | None:
    normalized_key = key.casefold()
    for header_name, value in headers.items():
        if header_name.casefold() == normalized_key:
            return str(value)
    return None


def _artifact_from_crawl_result(
    result: Any,
    *,
    fetch_method: str,
    requires_browser: bool,
) -> FetchedArtifact:
    from unibot.crawl.fetchers import FetchedArtifact

    if not getattr(result, "success", False) or not getattr(result, "html", None):
        raise ValueError(getattr(result, "error_message", None) or "crawl failed")

    headers = getattr(result, "response_headers", None) or {}
    content_type = str(_header(headers, "content-type") or "text/html; charset=utf-8")
    if "html" not in content_type.casefold():
        raise ValueError(f"non-html crawl result content type: {content_type}")
    markdown = getattr(result, "markdown", None)

    return FetchedArtifact(
        source_url=getattr(result, "redirected_url", None) or getattr(result, "url"),
        content=str(result.html).encode("utf-8"),
        content_type=content_type,
        fetch_method=fetch_method,
        http_status=int(getattr(result, "status_code", None) or 200),
        etag=_header(headers, "etag"),
        last_modified=_header(headers, "last-modified"),
        requires_browser=requires_browser,
        metadata={
            "markdown": getattr(markdown, "raw_markdown", None)
            if markdown is not None
            else None,
            "links": getattr(result, "links", None),
            "tables": getattr(result, "tables", None),
        },
    )


class Crawl4AIStaticFetcher:
    def fetch(self, url: str) -> FetchedArtifact:
        async def _run(url: str = url) -> Any:
            return await self._run_fetch(url)

        result = run_sync(_run)
        return _artifact_from_crawl_result(
            result,
            fetch_method="html_static",
            requires_browser=False,
        )

    async def _run_fetch(self, url: str) -> Any:
        strategy = AsyncHTTPCrawlerStrategy(
            browser_config=HTTPCrawlerConfig(follow_redirects=True)
        )
        async with AsyncWebCrawler(crawler_strategy=strategy) as crawler:
            return await crawler.arun(url=url, config=CrawlerRunConfig())


class Crawl4AIBrowserFetcher:
    def fetch(self, url: str) -> FetchedArtifact:
        async def _run(url: str = url) -> Any:
            return await self._run_fetch(url)

        result = run_sync(_run)
        return _artifact_from_crawl_result(
            result,
            fetch_method="browser",
            requires_browser=True,
        )

    async def _run_fetch(self, url: str) -> Any:
        async with AsyncWebCrawler(config=BrowserConfig(headless=True)) as crawler:
            return await crawler.arun(
                url=url,
                config=CrawlerRunConfig(
                    wait_until="networkidle",
                    process_iframes=True,
                    remove_overlay_elements=True,
                ),
            )


class Crawl4AIBatchFetcher:
    def __init__(self, *, requires_browser: bool = False, semaphore_count: int = 5) -> None:
        self._requires_browser = requires_browser
        self._semaphore_count = semaphore_count

    def fetch_many(
        self,
        urls: tuple[str, ...] | list[str],
    ) -> tuple[FetchedArtifact, ...]:
        if not urls:
            return ()
        async def _run(urls: tuple[str, ...] | list[str] = urls) -> Any:
            return await self._run_batch_fetch(list(urls))

        results = run_sync(_run)
        fetch_method = "browser" if self._requires_browser else "html_static"
        artifacts: list[FetchedArtifact] = []
        for result in results:
            try:
                artifacts.append(
                    _artifact_from_crawl_result(
                        result,
                        fetch_method=fetch_method,
                        requires_browser=self._requires_browser,
                    )
                )
            except ValueError:
                logger.warning(
                    "crawl.batch_result_invalid",
                    url=getattr(result, "url", None),
                    redirected_url=getattr(result, "redirected_url", None),
                    exc_info=True,
                )
        return tuple(artifacts)

    async def _run_batch_fetch(self, urls: list[str]) -> list[Any]:
        dispatcher = SemaphoreDispatcher(semaphore_count=self._semaphore_count)
        run_config = (
            CrawlerRunConfig(
                wait_until="networkidle",
                process_iframes=True,
                remove_overlay_elements=True,
            )
            if self._requires_browser
            else CrawlerRunConfig()
        )
        async with self._build_crawler() as crawler:
            results = await crawler.arun_many(
                urls=urls,
                config=run_config,
                dispatcher=dispatcher,
            )
        return list(_iter_results(results))

    def _build_crawler(self) -> AsyncWebCrawler:
        if self._requires_browser:
            return AsyncWebCrawler(config=BrowserConfig(headless=True))
        return AsyncWebCrawler(
            crawler_strategy=AsyncHTTPCrawlerStrategy(
                browser_config=HTTPCrawlerConfig(follow_redirects=True)
            )
        )


def _iter_results(results: Any) -> Iterable[Any]:
    if isinstance(results, list):
        return results
    if isinstance(results, tuple):
        return results
    return list(results)
