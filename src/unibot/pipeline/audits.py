from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from importlib import import_module

import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from unibot.db.models import CanonicalRecord, RawSnapshot, SourceSection as SourceSectionModel, SourceRegistry
from unibot.pipeline.contracts import (
    UpdateCycleAudit,
    _ZERO_CANONICAL_AUDIT_EXEMPT_SOURCE_CLASSES,
)
from unibot.pipeline.supporting_evidence import load_current_authoritative_decisions
from unibot.verify.deduplication import resolve_duplicates

logger = structlog.get_logger()


def _facade_logger():
    """Resolve logger through the scheduler.jobs facade so monkeypatching it still intercepts."""
    return import_module("unibot.scheduler.jobs").logger


def run_audits(
    session: Session,
    *,
    fetched_source_urls: tuple[str, ...] = (),
    auto_activate: bool = True,
    qdrant_writer: Any = None,
    alias_name: str = "",
) -> UpdateCycleAudit:
    contradictory_scope_ids = tuple(sorted(find_contradictory_scopes(session)))
    sources_with_sections_but_no_canonical_records = tuple(
        sorted(
            find_sources_with_sections_but_no_canonical_records(
                session,
                fetched_source_urls=fetched_source_urls,
            )
        )
    )
    canonical_eligible_count = len(
        resolve_duplicates(load_current_authoritative_decisions(session)).primary_records
    )
    if not auto_activate:
        return UpdateCycleAudit(
            active_index_count=0,
            canonical_eligible_count=canonical_eligible_count,
            blocked_or_inactive_record_version_ids=(),
            contradictory_scope_ids=contradictory_scope_ids,
            duplicate_rule_violations=(),
            sources_with_sections_but_no_canonical_records=(
                sources_with_sections_but_no_canonical_records
            ),
        )

    active_record_version_ids = qdrant_writer.list_record_version_ids(alias_name)
    active_rows = session.execute(
        select(
            CanonicalRecord.record_version_id,
            CanonicalRecord.conflict_scope_id,
            CanonicalRecord.dedupe_key,
            CanonicalRecord.value_hash if hasattr(CanonicalRecord, "value_hash") else CanonicalRecord.source_text_hash,
            SourceRegistry.legal_status,
            SourceRegistry.is_active,
        )
        .join(
            SourceRegistry,
            CanonicalRecord.source_id == SourceRegistry.source_id,
            isouter=True,
        )
        .where(CanonicalRecord.record_version_id.in_(tuple(active_record_version_ids) or ("",)))
    ).all()

    blocked_or_inactive = tuple(
        sorted(
            row.record_version_id
            for row in active_rows
            if (row.legal_status not in {None, "allowed"}) or row.is_active is False
        )
    )
    duplicate_rule_violations = tuple(sorted(find_active_duplicate_violations(active_rows)))

    return UpdateCycleAudit(
        active_index_count=len(active_record_version_ids),
        canonical_eligible_count=canonical_eligible_count,
        blocked_or_inactive_record_version_ids=blocked_or_inactive,
        contradictory_scope_ids=contradictory_scope_ids,
        duplicate_rule_violations=duplicate_rule_violations,
        sources_with_sections_but_no_canonical_records=(
            sources_with_sections_but_no_canonical_records
        ),
    )


def assert_audit_invariants(
    audit: UpdateCycleAudit,
    *,
    require_active_alias_checks: bool = True,
) -> None:
    if (
        require_active_alias_checks
        and audit.active_index_count != audit.canonical_eligible_count
    ):
        raise RuntimeError(
            "update cycle audit failed: active index count does not match canonical eligible count"
        )
    if require_active_alias_checks and audit.blocked_or_inactive_record_version_ids:
        raise RuntimeError(
            "update cycle audit failed: blocked or inactive records remain in the active alias"
        )
    _log = _facade_logger()
    if audit.contradictory_scope_ids:
        _log.warning(
            "update_cycle.audit_warning",
            issue="contradictory_scopes",
            scope_ids=audit.contradictory_scope_ids,
        )
    if audit.sources_with_sections_but_no_canonical_records:
        _log.warning(
            "update_cycle.audit_warning",
            issue="sections_without_canonical_records",
            source_urls=audit.sources_with_sections_but_no_canonical_records,
        )
    if require_active_alias_checks and audit.duplicate_rule_violations:
        raise RuntimeError(
            "update cycle audit failed: duplicate rule violations are present"
        )


def find_sources_with_sections_but_no_canonical_records(
    session: Session,
    *,
    fetched_source_urls: tuple[str, ...],
) -> tuple[str, ...]:
    if not fetched_source_urls:
        return ()

    rows = session.execute(
        select(
            SourceRegistry.source_id,
            SourceRegistry.source_url,
            SourceRegistry.source_class,
            SourceRegistry.legal_status,
            SourceRegistry.parser_target,
        ).where(
            SourceRegistry.is_active.is_(True),
            SourceRegistry.disappeared_at.is_(None),
            SourceRegistry.source_url.in_(fetched_source_urls),
        )
    ).all()

    latest_snapshot_by_source_id: dict[str, RawSnapshot] = {}
    for snapshot in session.execute(
        select(RawSnapshot)
        .join(SourceRegistry, RawSnapshot.source_id == SourceRegistry.source_id)
        .where(SourceRegistry.source_url.in_(fetched_source_urls))
        .order_by(RawSnapshot.source_id, RawSnapshot.fetched_at.desc())
    ).scalars():
        latest_snapshot_by_source_id.setdefault(snapshot.source_id, snapshot)

    section_snapshot_ids = set(
        session.execute(select(SourceSectionModel.snapshot_id).distinct()).scalars()
    )
    canonical_hash_pairs = set(
        session.execute(
            select(CanonicalRecord.source_id, CanonicalRecord.page_content_hash).distinct()
        ).all()
    )

    flagged_urls: list[str] = []
    for row in rows:
        latest_snap = latest_snapshot_by_source_id.get(row.source_id)
        if (
            latest_snap is not None
            and row.legal_status == "allowed"
            and row.parser_target == "html"
            and row.source_class not in _ZERO_CANONICAL_AUDIT_EXEMPT_SOURCE_CLASSES
            and latest_snap.snapshot_id in section_snapshot_ids
            and (row.source_id, latest_snap.page_content_hash) not in canonical_hash_pairs
        ):
            flagged_urls.append(row.source_url)
    return tuple(flagged_urls)


def find_contradictory_scopes(session: Session) -> set[str]:
    rows = session.execute(
        select(
            CanonicalRecord.conflict_scope_id,
            CanonicalRecord.source_authority_tier,
            CanonicalRecord.source_text_hash,
        )
        .where(CanonicalRecord.freshness_status == "contradictory")
    ).all()
    grouped: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.conflict_scope_id].append(
            (row.source_authority_tier, row.source_text_hash)
        )

    contradictory_scopes: set[str] = set()
    for scope_id, values in grouped.items():
        winning_tier = min(tier for tier, _ in values)
        winning_values = {
            value_hash for tier, value_hash in values if tier == winning_tier
        }
        if len(winning_values) > 1:
            contradictory_scopes.add(scope_id)
    return contradictory_scopes


def find_active_duplicate_violations(active_rows: Sequence[Any]) -> set[str]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in active_rows:
        grouped[row.dedupe_key].add(row.record_version_id)
    return {dedupe_key for dedupe_key, ids in grouped.items() if len(ids) > 1}
