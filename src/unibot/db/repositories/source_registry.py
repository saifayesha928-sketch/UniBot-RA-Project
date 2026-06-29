from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePosixPath
from typing import Literal, cast

from sqlalchemy import select
from sqlalchemy.orm import Session

from unibot.db.models import CanonicalRecord, RawSnapshot, SourceRegistry, SourceSection
from unibot.domain.source_authority import assign_authority_tier
from unibot.domain.source_policies import get_source_policy, legacy_source_url_aliases
from unibot.domain.types import ContentKind, PageKind, SourceClass, SourceContext

CrawlMethod = Literal["html_static", "browser", "wordpress_api"]
LegalStatus = Literal["allowed", "restricted", "blocked"]
CrawlStatus = Literal["unverified", "verified_live", "verified_placeholder", "blocked"]
ParserTarget = Literal["html", "document"]


@dataclass(frozen=True, slots=True)
class SourceRegistryEntry:
    source_url: str
    canonical_url: str
    source_class: SourceClass
    crawl_method: CrawlMethod
    legal_status: LegalStatus
    crawl_status: CrawlStatus | None
    default_authority_tier: int
    refresh_policy: str
    parser_target: ParserTarget = "html"
    parent_source_url: str | None = None
    link_text: str | None = None
    is_active: bool = True
    last_crawled_at: datetime | None = None
    last_successful_crawl_at: datetime | None = None


def _build_entry(
    url: str,
    *,
    source_class: SourceClass | None = None,
    page_kind: PageKind,
    content_kind: ContentKind,
    crawl_method: CrawlMethod = "html_static",
    refresh_policy: str,
    crawl_status: CrawlStatus | None = None,
    parser_target: ParserTarget | None = None,
    parent_source_url: str | None = None,
    link_text: str | None = None,
) -> SourceRegistryEntry:
    policy = get_source_policy(url)
    authority_tier = assign_authority_tier(
        SourceContext(
            source_url=url,
            page_kind=page_kind,
            content_kind=content_kind,
        )
    )

    return SourceRegistryEntry(
        source_url=url,
        canonical_url=policy.canonical_url,
        source_class=source_class or policy.source_class,
        crawl_method=crawl_method,
        legal_status=policy.access_level,
        crawl_status=crawl_status,
        default_authority_tier=authority_tier,
        refresh_policy=refresh_policy,
        parser_target=parser_target or _infer_parser_target(url, source_class or policy.source_class),
        parent_source_url=parent_source_url,
        link_text=link_text,
        is_active=True,
    )


def _infer_parser_target(url: str, source_class: str) -> ParserTarget:
    if source_class == "document_asset":
        suffix = PurePosixPath(urlsplit_path(url)).suffix.lower()
        if suffix:
            return "document"
    return "html"


def urlsplit_path(url: str) -> str:
    from urllib.parse import urlsplit

    return urlsplit(url).path or "/"


def build_research_subdomain_entry(
    url: str,
    *,
    crawl_status: CrawlStatus,
    parent_source_url: str | None = None,
    link_text: str | None = None,
) -> SourceRegistryEntry:
    refresh_policy = "weekly_or_monthly_based_on_change_rate"
    legal_status: LegalStatus = "allowed"
    if crawl_status == "blocked":
        legal_status = "blocked"

    entry = _build_entry(
        url,
        source_class="research_subdomain",
        page_kind="dedicated_child",
        content_kind="page_body",
        refresh_policy=refresh_policy,
        crawl_status=crawl_status,
        parent_source_url=parent_source_url,
        link_text=link_text,
    )

    return SourceRegistryEntry(
        source_url=entry.source_url,
        canonical_url=entry.canonical_url,
        source_class=entry.source_class,
        crawl_method=entry.crawl_method,
        legal_status=legal_status,
        crawl_status=entry.crawl_status,
        default_authority_tier=entry.default_authority_tier,
        refresh_policy=entry.refresh_policy,
        parser_target=entry.parser_target,
        parent_source_url=entry.parent_source_url,
        link_text=entry.link_text,
        is_active=entry.is_active,
    )


def build_seed_source_registry_entries() -> tuple[SourceRegistryEntry, ...]:
    """Return the seed source-registry entries used to bootstrap crawling.

    Empty by default: partner deployments ingest already-extracted records
    (``data/records.jsonl``) and load their own sources from
    ``data/sources.json`` instead of crawling a hard-coded seed list. If you
    operate the optional crawl path, populate this with your own URLs via
    ``_build_entry(...)`` / ``build_research_subdomain_entry(...)``.
    """
    return ()


class SourceRegistryRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get_active_entries(
        self,
        *,
        exclude_research_subdomains: bool = False,
        exclude_news: bool = False,
    ) -> tuple[SourceRegistryEntry, ...]:
        rows = self._session.execute(
            select(SourceRegistry).where(
                SourceRegistry.is_active.is_(True),
                SourceRegistry.disappeared_at.is_(None),
            )
        ).scalars().all()
        if exclude_research_subdomains:
            rows = [
                row for row in rows if row.source_class != "research_subdomain"
            ]
        if exclude_news:
            rows = [
                row for row in rows if row.source_class != "news_event"
            ]
        return tuple(
            SourceRegistryEntry(
                source_url=row.source_url,
                canonical_url=row.canonical_url,
                source_class=cast(SourceClass, row.source_class),
                crawl_method=cast(CrawlMethod, row.crawl_method),
                legal_status=cast(LegalStatus, row.legal_status),
                crawl_status=cast(CrawlStatus | None, row.crawl_status),
                default_authority_tier=row.default_authority_tier,
                refresh_policy=row.refresh_policy,
                parser_target=cast(ParserTarget, row.parser_target or "html"),
                parent_source_url=row.parent_source_url,
                link_text=row.link_text,
                is_active=row.is_active,
                last_crawled_at=row.last_crawled_at,
                last_successful_crawl_at=row.last_successful_crawl_at,
            )
            for row in rows
        )

    def upsert_entries(self, entries: tuple[SourceRegistryEntry, ...]) -> int:
        if not entries:
            return 0

        for entry in entries:
            for legacy_url in legacy_source_url_aliases(entry.source_url):
                self.rebind_source_url(old_url=legacy_url, new_url=entry.source_url)

        existing_by_url = {
            row.source_url: row
            for row in self._session.execute(
                select(SourceRegistry).where(
                    SourceRegistry.source_url.in_(tuple(entry.source_url for entry in entries))
                )
            ).scalars()
        }

        for entry in entries:
            existing = existing_by_url.get(entry.source_url)

            if existing is None:
                existing = SourceRegistry(
                    source_url=entry.source_url,
                    canonical_url=entry.canonical_url,
                    source_class=entry.source_class,
                    crawl_method=entry.crawl_method,
                    legal_status=entry.legal_status,
                    crawl_status=entry.crawl_status,
                    default_authority_tier=entry.default_authority_tier,
                    refresh_policy=entry.refresh_policy,
                    parser_target=entry.parser_target,
                    parent_source_url=entry.parent_source_url,
                    link_text=entry.link_text,
                    is_active=entry.is_active,
                    last_crawled_at=entry.last_crawled_at,
                    last_successful_crawl_at=entry.last_successful_crawl_at,
                )
                self._session.add(existing)
                existing_by_url[entry.source_url] = existing
                continue

            existing.canonical_url = entry.canonical_url
            existing.source_class = entry.source_class
            existing.crawl_method = entry.crawl_method
            existing.legal_status = entry.legal_status
            if entry.crawl_status is not None:
                existing.crawl_status = entry.crawl_status
            existing.default_authority_tier = entry.default_authority_tier
            existing.refresh_policy = entry.refresh_policy
            existing.parser_target = entry.parser_target
            existing.parent_source_url = entry.parent_source_url
            existing.link_text = entry.link_text
            existing.is_active = entry.is_active
            if entry.last_crawled_at is not None:
                existing.last_crawled_at = entry.last_crawled_at
            if entry.last_successful_crawl_at is not None:
                existing.last_successful_crawl_at = entry.last_successful_crawl_at

        self._session.flush()
        return len(entries)

    def rebind_source_url(self, *, old_url: str, new_url: str) -> None:
        if old_url == new_url:
            return

        old_row = self._session.execute(
            select(SourceRegistry).where(SourceRegistry.source_url == old_url)
        ).scalar_one_or_none()
        if old_row is None:
            return

        new_row = self._session.execute(
            select(SourceRegistry).where(SourceRegistry.source_url == new_url)
        ).scalar_one_or_none()
        new_policy = get_source_policy(new_url)

        for child in self._session.execute(
            select(SourceRegistry).where(SourceRegistry.parent_source_url == old_url)
        ).scalars():
            child.parent_source_url = new_url

        if new_row is None:
            old_row.source_url = new_url
            old_row.canonical_url = new_policy.canonical_url
            old_row.source_class = new_policy.source_class
            for record in self._session.execute(
                select(CanonicalRecord).where(CanonicalRecord.source_id == old_row.source_id)
            ).scalars():
                record.source_url = new_url
                record.canonical_url = new_policy.canonical_url
            for snapshot in self._session.execute(
                select(RawSnapshot).where(RawSnapshot.source_id == old_row.source_id)
            ).scalars():
                snapshot.source_url = new_url
            self._session.flush()
            return

        for record in self._session.execute(
            select(CanonicalRecord).where(CanonicalRecord.source_id == old_row.source_id)
        ).scalars():
            record.source_id = new_row.source_id
            record.source_url = new_url
            record.canonical_url = new_policy.canonical_url
        for snapshot in self._session.execute(
            select(RawSnapshot).where(RawSnapshot.source_id == old_row.source_id)
        ).scalars():
            snapshot.source_id = new_row.source_id
            snapshot.source_url = new_url
        for section in self._session.execute(
            select(SourceSection).where(SourceSection.source_id == old_row.source_id)
        ).scalars():
            section.source_id = new_row.source_id

        new_row.canonical_url = new_policy.canonical_url
        new_row.source_class = new_policy.source_class
        new_row.is_active = new_row.is_active or old_row.is_active
        if new_row.last_crawled_at is None or (
            old_row.last_crawled_at is not None and old_row.last_crawled_at > new_row.last_crawled_at
        ):
            new_row.last_crawled_at = old_row.last_crawled_at
        if new_row.last_seen_at is None or (
            old_row.last_seen_at is not None and old_row.last_seen_at > new_row.last_seen_at
        ):
            new_row.last_seen_at = old_row.last_seen_at
        if new_row.last_successful_crawl_at is None or (
            old_row.last_successful_crawl_at is not None
            and old_row.last_successful_crawl_at > new_row.last_successful_crawl_at
        ):
            new_row.last_successful_crawl_at = old_row.last_successful_crawl_at
        if old_row.link_text and not new_row.link_text:
            new_row.link_text = old_row.link_text

        self._session.delete(old_row)
        self._session.flush()


def build_discovered_source_registry_entry(
    url: str,
    *,
    source_class: SourceClass | None = None,
    page_kind: PageKind,
    content_kind: ContentKind,
    refresh_policy: str,
    crawl_method: CrawlMethod = "html_static",
    parser_target: ParserTarget | None = None,
    parent_source_url: str | None = None,
    link_text: str | None = None,
) -> SourceRegistryEntry:
    return _build_entry(
        url,
        source_class=source_class,
        page_kind=page_kind,
        content_kind=content_kind,
        refresh_policy=refresh_policy,
        crawl_method=crawl_method,
        parser_target=parser_target,
        parent_source_url=parent_source_url,
        link_text=link_text,
    )
