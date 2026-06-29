from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from unibot.db.repositories.source_registry import SourceRegistryEntry
from unibot.domain.source_policies import HIGH_RISK_SOURCE_CLASSES
from unibot.domain.types import SourceClass
# Optional per-URL crawl priority overrides (lower number = higher priority).
# Partners may populate this with their own high-churn URLs (e.g. admissions,
# fee structure, merit lists) if they run the optional crawl path. Empty by
# default — partners deliver already-extracted records and do not crawl.
HIGH_RISK_WATCHLIST_PRIORITY: dict[str, int] = {}

# Minimum interval before re-crawling a source.  If a policy is absent from
# this mapping the source is never auto-selected (manual_only behaviour).
_REFRESH_INTERVALS: dict[str, timedelta] = {
    "every_6_hours_while_active_daily_otherwise": timedelta(hours=6),
    "daily_during_season_weekly_otherwise": timedelta(days=1),
    "weekly": timedelta(weeks=1),
    "weekly_or_monthly_based_on_change_rate": timedelta(weeks=1),
}
_GRACE_WINDOWS_BY_SOURCE_CLASS: dict[str, timedelta] = {
    "admissions_cycle": timedelta(hours=24),
    "program_fee_schedule": timedelta(hours=24),
    "merit_list": timedelta(hours=12),
    "scholarship": timedelta(days=7),
    "program": timedelta(days=7),
    "faculty": timedelta(days=7),
    "policy": timedelta(days=14),
    "research_main": timedelta(days=14),
    "research_subdomain": timedelta(days=14),
    "student_service": timedelta(days=14),
    "university_info": timedelta(days=14),
    "org_unit": timedelta(days=14),
    "news_event": timedelta(hours=48),
}


@dataclass(frozen=True, slots=True)
class CrawlJob:
    source_url: str
    source_class: SourceClass
    crawl_method: str
    legal_status: str
    default_authority_tier: int
    refresh_policy: str
    parser_target: str
    parent_source_url: str | None = None
    link_text: str | None = None


def _is_admissions_season(now: datetime) -> bool:
    # Admissions, fee, and merit-list pages change most during the intake
    # cycle, which we treat as April through September for deterministic
    # scheduling and tests.
    return 4 <= now.month <= 9


def _is_high_risk_active_window(entry: SourceRegistryEntry, *, now: datetime) -> bool:
    if entry.source_class not in HIGH_RISK_SOURCE_CLASSES:
        return False
    return _is_admissions_season(now)


def resolve_refresh_interval(
    entry: SourceRegistryEntry,
    *,
    now: datetime,
) -> timedelta | None:
    if entry.refresh_policy == "manual_only":
        return None
    if entry.refresh_policy == "daily_during_season_weekly_otherwise":
        if _is_high_risk_active_window(entry, now=now):
            return timedelta(days=1)
        return timedelta(weeks=1)
    if entry.refresh_policy == "every_6_hours_while_active_daily_otherwise":
        if _is_high_risk_active_window(entry, now=now):
            return timedelta(hours=6)
        return timedelta(days=1)
    return _REFRESH_INTERVALS.get(entry.refresh_policy)


def resolve_grace_window(entry: SourceRegistryEntry) -> timedelta:
    grace_window = _GRACE_WINDOWS_BY_SOURCE_CLASS.get(entry.source_class)
    if grace_window is not None:
        return grace_window
    refresh_interval = resolve_refresh_interval(entry, now=datetime.now(timezone.utc))
    if refresh_interval is not None:
        return refresh_interval * 2
    return timedelta(days=14)


def is_source_within_grace_window(
    *,
    entry: SourceRegistryEntry,
    last_successful_crawl_at: datetime | None,
    now: datetime,
) -> bool:
    if last_successful_crawl_at is None:
        return False
    normalized_now = _normalize_timestamp(now)
    normalized_last_successful = _normalize_timestamp(last_successful_crawl_at)
    return (normalized_now - normalized_last_successful) <= resolve_grace_window(entry)


def _normalize_timestamp(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _priority(entry: SourceRegistryEntry) -> tuple[int, int, str]:
    if entry.source_class in HIGH_RISK_SOURCE_CLASSES:
        return (
            0,
            HIGH_RISK_WATCHLIST_PRIORITY.get(
                entry.source_url,
                len(HIGH_RISK_WATCHLIST_PRIORITY),
            ),
            entry.source_url,
        )
    if entry.source_class == "research_subdomain":
        return (1, 0, entry.source_url)
    return (2, 0, entry.source_url)


def _is_crawlable(entry: SourceRegistryEntry) -> bool:
    if not entry.is_active or entry.legal_status != "allowed":
        return False
    if entry.source_class == "research_subdomain":
        return entry.crawl_status in {"unverified", "verified_live"}
    return True


def _is_refresh_due(entry: SourceRegistryEntry, *, now: datetime) -> bool:
    """Return True when the source is due for a re-crawl based on its refresh policy."""
    if entry.refresh_policy == "manual_only":
        return False
    if entry.last_crawled_at is None:
        # Never crawled — always eligible
        return True
    interval = resolve_refresh_interval(entry, now=now)
    if interval is None:
        # Unknown policy — treat as eligible to avoid missing crawls
        return True
    normalized_now = _normalize_timestamp(now)
    normalized_last = _normalize_timestamp(entry.last_crawled_at)
    return (normalized_now - normalized_last) >= interval


def select_sources_for_crawl(
    entries: tuple[SourceRegistryEntry, ...],
    *,
    limit: int | None = None,
    force: bool = False,
) -> list[CrawlJob]:
    now = datetime.now(timezone.utc)

    crawlable_entries = sorted(
        (
            entry
            for entry in entries
            if _is_crawlable(entry) and (force or _is_refresh_due(entry, now=now))
        ),
        key=_priority,
    )

    if limit is not None:
        crawlable_entries = crawlable_entries[:limit]

    return [
        CrawlJob(
            source_url=entry.source_url,
            source_class=entry.source_class,
            crawl_method=entry.crawl_method,
            legal_status=entry.legal_status,
            default_authority_tier=entry.default_authority_tier,
            refresh_policy=entry.refresh_policy,
            parser_target=entry.parser_target,
            parent_source_url=entry.parent_source_url,
            link_text=entry.link_text,
        )
        for entry in crawlable_entries
    ]
