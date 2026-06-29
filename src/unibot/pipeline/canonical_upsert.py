from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import replace
from datetime import date

from importlib import import_module

from sqlalchemy import select
from sqlalchemy.orm import Session

from unibot.db.models import (
    CanonicalRecord,
    VALID_CANONICAL_RECORD_TYPES,
    VerificationEvent,
)
from unibot.domain.source_policies import get_source_policy
from unibot.pipeline.contracts import (
    _ScopeProcessingResult,
    _SERVING_ELIGIBLE_STATUSES,
)
from unibot.pipeline.decision_engine import (
    build_parent_state_lookup,
    candidate_from_row,
    classify_candidates_for_parent_resolution,
)
from unibot.utils import utc_now as _utc_now
from unibot.verify.currentness import classify_currentness
from unibot.verify.rules import VerificationCandidate, VerificationDecision


def _facade_logger():
    """Resolve logger through the scheduler.jobs facade so monkeypatching it still intercepts."""
    return import_module("unibot.scheduler.jobs").logger


def process_scope_updates(
    session: Session,
    incoming_candidates: tuple[VerificationCandidate, ...],
    *,
    source_id_resolver: Callable[[str], str | None],
    create_verification_event: Callable[[Session, VerificationDecision], VerificationEvent],
) -> _ScopeProcessingResult:
    crawl_time = _utc_now()
    pending_deindex_record_version_ids: list[str] = []
    verification_event_ids: list[str] = []
    candidates_by_scope: dict[str, list[VerificationCandidate]] = defaultdict(list)
    prepared_candidates = tuple(
        replace(
            candidate,
            fetched_at=candidate.fetched_at or crawl_time,
            parent_record_id=(
                candidate.parent_record_id
                or candidate.record_payload.get("parent_record_id")
            ),
            parent_source_url=(
                candidate.parent_source_url
                or candidate.record_payload.get("parent_page_url")
                or candidate.record_payload.get("parent_source_url")
            ),
        )
        for candidate in incoming_candidates
    )
    for candidate in prepared_candidates:
        candidates_by_scope[candidate.conflict_scope_id].append(candidate)

    existing_rows_by_scope: dict[str, list[CanonicalRecord]] = {}
    existing_by_version_id_by_scope: dict[str, dict[str, CanonicalRecord]] = {}
    combined_candidates_by_scope: dict[str, list[VerificationCandidate]] = {}

    all_scope_ids = tuple(candidates_by_scope.keys())
    all_existing_rows = session.execute(
        select(CanonicalRecord).where(
            CanonicalRecord.conflict_scope_id.in_(all_scope_ids)
        )
    ).scalars().all()
    rows_by_scope: dict[str, list[CanonicalRecord]] = defaultdict(list)
    for existing_row in all_existing_rows:
        rows_by_scope[existing_row.conflict_scope_id].append(existing_row)

    for scope_id, scope_candidates in candidates_by_scope.items():
        existing_rows = rows_by_scope.get(scope_id, [])
        existing_rows_by_scope[scope_id] = existing_rows
        existing_by_version_id_by_scope[scope_id] = {
            r.record_version_id: r for r in existing_rows
        }
        combined_candidates_by_scope[scope_id] = [
            *(candidate_from_row(r) for r in existing_rows),
            *scope_candidates,
        ]

    base_decisions = classify_candidates_for_parent_resolution(
        session=session,
        candidates_by_scope=combined_candidates_by_scope,
    )
    parent_lookup = build_parent_state_lookup(
        session=session,
        combined_candidates_by_scope=combined_candidates_by_scope,
        base_decisions=base_decisions,
    )

    for scope_id, scope_candidates in candidates_by_scope.items():
        existing_rows = existing_rows_by_scope[scope_id]
        existing_by_version_id = existing_by_version_id_by_scope[scope_id]
        combined_candidates = combined_candidates_by_scope[scope_id]
        decisions = {
            candidate.record_version_id: classify_currentness(
                candidate,
                [
                    sibling
                    for sibling in combined_candidates
                    if sibling.record_version_id != candidate.record_version_id
                ],
                get_source_policy(candidate.source_url),
                parent_state=parent_lookup.resolve(candidate),
            )
            for candidate in combined_candidates
        }
        incoming_ids = {
            candidate.record_version_id for candidate in scope_candidates
        }
        fail_closed = any(
            decisions[record_version_id].requires_manual_review
            for record_version_id in incoming_ids
        )

        for candidate in combined_candidates:
            decision = decisions[candidate.record_version_id]
            row = existing_by_version_id.get(candidate.record_version_id)
            if row is None:
                row = create_record(
                    session, candidate, decision, source_id_resolver,
                )
                if row is None:
                    continue
            else:
                if row.record_version_id not in incoming_ids:
                    apply_decision(row, decision)
                else:
                    update_record(session, row, candidate, decision, source_id_resolver)

            if fail_closed and row.record_version_id not in incoming_ids:
                if row.serving_status == "indexed_active":
                    row.serving_status = "pending_deindex"
                    pending_deindex_record_version_ids.append(row.record_version_id)
                elif row.serving_status in _SERVING_ELIGIBLE_STATUSES:
                    row.serving_status = "ineligible"
                row.is_current_candidate = False
                row.is_current_authoritative = False

            if decision.requires_manual_review:
             event = create_verification_event(session, decision)
             if event is not None:
              verification_event_ids.append(event.event_id)

        if not fail_closed:
            pending_deindex_record_version_ids.extend(
                link_superseded_records(session, existing_rows, scope_candidates, decisions)
            )
        else:
            for row in existing_rows:
                if row.serving_status == "indexed_active":
                    row.serving_status = "pending_deindex"
                    pending_deindex_record_version_ids.append(row.record_version_id)

    session.flush()
    return _ScopeProcessingResult(
        pending_deindex_record_version_ids=tuple(
            sorted(set(pending_deindex_record_version_ids))
        ),
        verification_event_ids=tuple(sorted(verification_event_ids)),
    )


def create_record(
    session: Session,
    candidate: VerificationCandidate,
    decision: VerificationDecision,
    source_id_resolver: Callable[[str], str | None],
) -> CanonicalRecord | None:
    if candidate.record_type not in VALID_CANONICAL_RECORD_TYPES:
        _facade_logger().warning(
            "update_cycle.invalid_record_type_skipped",
            record_type=candidate.record_type,
            record_version_id=candidate.record_version_id,
            source_url=candidate.source_url,
        )
        return None
    row = CanonicalRecord(
        record_version_id=candidate.record_version_id,
        record_id=candidate.record_id,
        record_type=candidate.record_type,
        source_id=source_id_resolver(candidate.source_url),
        source_section_id=None,
        source_url=candidate.source_url,
        canonical_url=get_source_policy(candidate.source_url).canonical_url,
        source_title=candidate.record_id,
        source_section_label=candidate.source_section_label or candidate.record_type,
        source_locator=candidate.source_locator,
        source_text_hash=candidate.value_hash,
        page_content_hash=(
            candidate.page_content_hash or f"{candidate.record_version_id}:page"
        ),
        source_last_modified_text=candidate.record_payload.get(
            "source_last_modified_text"
        ),
        source_authority_tier=candidate.source_authority_tier,
        conflict_scope_id=candidate.conflict_scope_id,
        dedupe_key=candidate.dedupe_key,
        record_payload=candidate.record_payload,
        fetched_at=candidate.fetched_at or _utc_now(),
        verified_at=_utc_now() if decision.verification_status == "verified" else None,
        effective_from=candidate.effective_from,
        effective_to=candidate.effective_to,
        cycle_label=candidate.cycle_label,
        year_confidence=candidate.year_confidence,
        extraction_confidence=str(
            candidate.record_payload.get("extraction_confidence", "rule_based")
        ),
        freshness_status=decision.freshness_status,
        verification_status=decision.verification_status,
        serving_status=decision.serving_status,
        is_current_candidate=decision.is_current_candidate,
        is_current_authoritative=decision.is_current_authoritative,
    )
    session.add(row)
    session.flush()
    return row


def update_record(
    session: Session,
    row: CanonicalRecord,
    candidate: VerificationCandidate,
    decision: VerificationDecision,
    source_id_resolver: Callable[[str], str | None],
) -> None:
    row.source_id = source_id_resolver(candidate.source_url)
    row.source_section_id = None
    row.source_url = candidate.source_url
    row.canonical_url = get_source_policy(candidate.source_url).canonical_url
    row.source_title = candidate.record_id
    row.source_section_label = candidate.source_section_label or candidate.record_type
    row.source_locator = candidate.source_locator
    row.source_text_hash = candidate.value_hash
    row.page_content_hash = (
        candidate.page_content_hash or f"{candidate.record_version_id}:page"
    )
    row.source_last_modified_text = candidate.record_payload.get(
        "source_last_modified_text"
    )
    row.source_authority_tier = candidate.source_authority_tier
    row.conflict_scope_id = candidate.conflict_scope_id
    row.dedupe_key = candidate.dedupe_key
    row.record_payload = candidate.record_payload
    row.fetched_at = candidate.fetched_at or _utc_now()
    row.verified_at = _utc_now() if decision.verification_status == "verified" else None
    row.effective_from = candidate.effective_from
    row.effective_to = candidate.effective_to
    row.cycle_label = candidate.cycle_label
    row.year_confidence = candidate.year_confidence
    row.extraction_confidence = str(
        candidate.record_payload.get("extraction_confidence", "rule_based")
    )
    apply_decision(row, decision)


def apply_decision(
    row: CanonicalRecord,
    decision: VerificationDecision,
) -> None:
    was_indexed = row.serving_status == "indexed_active"
    row.freshness_status = decision.freshness_status
    row.verification_status = decision.verification_status
    row.serving_status = decision.serving_status
    row.is_current_candidate = decision.is_current_candidate
    row.is_current_authoritative = decision.is_current_authoritative
    row.verified_at = _utc_now() if decision.verification_status == "verified" else None
    if was_indexed and not decision.is_current_authoritative:
        row.serving_status = "pending_deindex"


def link_superseded_records(
    session: Session,
    existing_rows: list[CanonicalRecord],
    incoming_candidates: list[VerificationCandidate],
    decisions: dict[str, VerificationDecision],
) -> tuple[str, ...]:
    pending_deindex_record_version_ids: list[str] = []
    incoming_current = [
        candidate
        for candidate in incoming_candidates
        if decisions[candidate.record_version_id].is_current_authoritative
    ]
    if not incoming_current:
        return ()

    latest_incoming = max(
        incoming_current,
        key=lambda candidate: (
            candidate.effective_from or date.min,
            -candidate.source_authority_tier,
            candidate.record_version_id,
        ),
    )
    latest_row = session.get(CanonicalRecord, latest_incoming.record_version_id)
    if latest_row is None:
        return ()

    for row in existing_rows:
        if row.record_version_id == latest_row.record_version_id:
            continue
        if row.is_current_authoritative or row.serving_status in {
            "indexed_active",
            "pending_deindex",
        }:
            row.superseded_by_version_id = latest_row.record_version_id
            latest_row.supersedes_version_id = row.record_version_id
            if row.serving_status in {"indexed_active", "pending_deindex"}:
                row.serving_status = "pending_deindex"
                pending_deindex_record_version_ids.append(row.record_version_id)
            else:
                row.serving_status = "ineligible"
            row.is_current_authoritative = False

    return tuple(sorted(set(pending_deindex_record_version_ids)))


def create_verification_event(
    session: Session,
    decision: VerificationDecision,
) -> VerificationEvent:
    existing = session.execute(
        select(VerificationEvent).where(
            VerificationEvent.record_version_id == decision.candidate.record_version_id,
            VerificationEvent.event_type == "manual_review_required",
            VerificationEvent.verification_status == "pending",
        )
    ).scalar_one_or_none()
    if existing is not None:
        return existing

    event = VerificationEvent(
        record_version_id=decision.candidate.record_version_id,
        source_section_id=decision.candidate.source_section_id,
        event_type="manual_review_required",
        verification_status="pending",
        notes=decision.notes,
        event_payload={
            "reason": decision.manual_review_reason,
            "record_id": decision.candidate.record_id,
            "record_type": decision.candidate.record_type,
            "source_url": decision.candidate.source_url,
            "conflicting_record_ids": list(decision.conflicting_record_ids),
        },
    )
    session.add(event)
    session.flush()
    return event
