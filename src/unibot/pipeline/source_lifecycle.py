from __future__ import annotations

from dataclasses import replace
from typing import cast

import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from unibot.crawl.jobs import CrawlJob, is_source_within_grace_window
from unibot.db.models import CanonicalRecord, SourceRegistry, VerificationEvent
from unibot.db.repositories.source_registry import (
    CrawlMethod,
    CrawlStatus,
    LegalStatus,
    ParserTarget,
    SourceRegistryEntry,
)
from unibot.domain.types import SourceClass
from unibot.pipeline.contracts import _REMOVABLE_RECORD_TYPES, _SERVING_ELIGIBLE_STATUSES
from unibot.utils import utc_now as _utc_now
from unibot.verify.rules import VerificationCandidate

logger = structlog.get_logger()


def normalized_crawl_status(
    job: CrawlJob,
    candidates: tuple[VerificationCandidate, ...],
) -> str | None:
    if job.source_class == "research_subdomain":
        return "verified_live" if candidates else "verified_placeholder"
    if candidates:
        return "verified_live"
    return None


def reload_crawl_timestamps(
    session: Session,
    entries: tuple[SourceRegistryEntry, ...],
) -> tuple[SourceRegistryEntry, ...]:
    """Reload last_crawled_at from the DB so refresh-interval checks are accurate."""
    urls = [e.source_url for e in entries]
    rows = session.execute(
        select(
            SourceRegistry.source_url,
            SourceRegistry.last_crawled_at,
            SourceRegistry.last_successful_crawl_at,
        ).where(SourceRegistry.source_url.in_(urls))
    ).all()
    ts_by_url = {
        row.source_url: (row.last_crawled_at, row.last_successful_crawl_at)
        for row in rows
    }
    return tuple(
        replace(
            entry,
            last_crawled_at=ts_by_url.get(entry.source_url, (None, None))[0],
            last_successful_crawl_at=ts_by_url.get(entry.source_url, (None, None))[1],
        )
        for entry in entries
    )


def record_source_error(
    session: Session,
    job: CrawlJob,
    error_type: str,
    source_id: str | None,
) -> None:
    now = _utc_now()
    event = VerificationEvent(
        event_type=error_type,
        verification_status="pending",
        notes=f"{error_type} for {job.source_url} (source_class={job.source_class})",
        event_payload={
            "source_url": job.source_url,
            "source_class": job.source_class,
            "error_type": error_type,
        },
    )
    session.add(event)
    session.flush()
    if source_id is not None:
        source_row = session.execute(
            select(SourceRegistry).where(SourceRegistry.source_id == source_id)
        ).scalar_one_or_none()
        if source_row is not None:
            source_row.last_crawled_at = now
            source_row.crawl_status = f"failed:{error_type}"
    session.flush()


def record_source_success(
    session: Session,
    job: CrawlJob,
    *,
    crawl_status: str | None = None,
) -> None:
    now = _utc_now()
    source_row = session.execute(
        select(SourceRegistry).where(SourceRegistry.source_url == job.source_url)
    ).scalar_one_or_none()
    if source_row is None:
        return
    source_row.last_crawled_at = now
    source_row.last_successful_crawl_at = now
    source_row.last_seen_at = now
    source_row.disappeared_at = None
    source_row.is_active = True
    if crawl_status is not None:
        source_row.crawl_status = crawl_status
    elif source_row.crawl_status is None or source_row.crawl_status.startswith("failed:"):
        source_row.crawl_status = "verified_live"
    session.flush()


def reconcile_disappeared_sources(
    session: Session,
    crawled_parent_urls: set[str],
    discovered_child_urls: dict[str, set[str]],
) -> None:
    now = _utc_now()
    for parent_url in crawled_parent_urls:
        current_children = discovered_child_urls.get(parent_url, set())
        previous_children = session.execute(
            select(SourceRegistry).where(
                SourceRegistry.parent_source_url == parent_url,
                SourceRegistry.is_active.is_(True),
                SourceRegistry.disappeared_at.is_(None),
            )
        ).scalars().all()

        for child_row in previous_children:
            if child_row.source_url in current_children:
                child_row.last_seen_at = now
                continue

            child_row.disappeared_at = now
            child_row.is_active = False
            logger.info(
                "update_cycle.source_disappeared",
                source_url=child_row.source_url,
                parent_url=parent_url,
            )

            affected_records = session.execute(
                select(CanonicalRecord).where(
                    CanonicalRecord.source_url == child_row.source_url,
                    CanonicalRecord.freshness_status.in_(("current", "unknown")),
                )
            ).scalars().all()
            for record in affected_records:
                if record.record_type in _REMOVABLE_RECORD_TYPES:
                    was_indexed = record.serving_status == "indexed_active"
                    record.freshness_status = "removed"
                    record.serving_status = (
                        "pending_deindex" if was_indexed else "ineligible"
                    )
                    record.is_current_authoritative = False
                    record.is_current_candidate = False

    session.flush()


def downgrade_expired_sources(session: Session) -> tuple[str, ...]:
    now = _utc_now()
    pending_deindex_record_version_ids: list[str] = []
    rows = session.execute(
        select(SourceRegistry).where(
            SourceRegistry.is_active.is_(True),
            SourceRegistry.disappeared_at.is_(None),
        )
    ).scalars().all()

    for row in rows:
        if not (row.crawl_status or "").startswith("failed:"):
            continue
        entry = SourceRegistryEntry(
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
        if is_source_within_grace_window(
            entry=entry,
            last_successful_crawl_at=row.last_successful_crawl_at,
            now=now,
        ):
            continue

        affected_records = session.execute(
            select(CanonicalRecord).where(
                CanonicalRecord.source_id == row.source_id,
                CanonicalRecord.freshness_status == "current",
            )
        ).scalars().all()
        for record in affected_records:
            was_serving = record.serving_status in _SERVING_ELIGIBLE_STATUSES or record.serving_status in {
                "indexed_active",
                "pending_deindex",
            }
            if was_serving:
                pending_deindex_record_version_ids.append(record.record_version_id)
            record.freshness_status = "unknown"
            record.serving_status = "pending_deindex" if was_serving else "ineligible"
            record.is_current_authoritative = False
            record.is_current_candidate = False

    session.flush()
    return tuple(sorted(set(pending_deindex_record_version_ids)))


def source_id_for_url(
    session: Session,
    source_url: str,
    cache: dict[str, str | None],
) -> str | None:
    if source_url in cache:
        return cache[source_url]
    result = session.execute(
        select(SourceRegistry.source_id).where(SourceRegistry.source_url == source_url)
    ).scalar_one_or_none()
    cache[source_url] = result
    return result
