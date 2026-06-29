from __future__ import annotations

from collections import defaultdict

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from unibot.db.models import CanonicalRecord, IndexJob
from unibot.domain.source_policies import get_source_policy
from unibot.pipeline.decision_engine import (
    build_parent_state_lookup,
    candidate_from_row,
    classify_candidates_for_parent_resolution,
)
from unibot.utils import utc_now as _utc_now
from unibot.verify.currentness import can_enter_serving, classify_currentness
from unibot.verify.rules import VerificationCandidate, VerificationDecision


def recompute_scope_state(session: Session, record_version_id: str) -> None:
    anchor = session.get(CanonicalRecord, record_version_id)
    if anchor is None:
        return

    affected_rows = load_recompute_rows(session, anchor)
    if not affected_rows:
        return

    rows_by_version_id = {
        row.record_version_id: row for row in affected_rows
    }
    candidates_by_scope: dict[str, list[VerificationCandidate]] = defaultdict(list)
    for row in affected_rows:
        candidates_by_scope[row.conflict_scope_id].append(candidate_from_row(row))

    active_candidates_by_scope = {
        scope_id: [
            candidate
            for candidate in candidates
            if rows_by_version_id[candidate.record_version_id].verification_status != "rejected"
        ]
        for scope_id, candidates in candidates_by_scope.items()
    }

    base_decisions = classify_candidates_for_parent_resolution(
        session=session,
        candidates_by_scope=active_candidates_by_scope,
    )
    parent_lookup = build_parent_state_lookup(
        session=session,
        combined_candidates_by_scope=candidates_by_scope,
        base_decisions=base_decisions,
    )

    decisions: dict[str, VerificationDecision] = {}
    for scope_id, scope_candidates in candidates_by_scope.items():
        active_scope_candidates = active_candidates_by_scope[scope_id]
        for candidate in scope_candidates:
            row = rows_by_version_id[candidate.record_version_id]
            base_decision = classify_currentness(
                candidate,
                [
                    sibling
                    for sibling in active_scope_candidates
                    if sibling.record_version_id != candidate.record_version_id
                ],
                get_source_policy(candidate.source_url),
                parent_state=parent_lookup.resolve(candidate),
            )
            if row.verification_status == "rejected":
                decisions[candidate.record_version_id] = VerificationDecision(
                    candidate=base_decision.candidate,
                    verification_status="rejected",
                    freshness_status=base_decision.freshness_status,
                    serving_status="ineligible",
                    is_current_candidate=False,
                    is_current_authoritative=False,
                    supporting_record_ids=base_decision.supporting_record_ids,
                    conflicting_record_ids=base_decision.conflicting_record_ids,
                    notes=base_decision.notes,
                )
                continue
            decisions[candidate.record_version_id] = base_decision

    for row in affected_rows:
        _apply_recomputed_decision(session, row, decisions[row.record_version_id])

    session.flush()


def load_recompute_rows(
    session: Session,
    anchor: CanonicalRecord,
) -> tuple[CanonicalRecord, ...]:
    affected_by_version_id: dict[str, CanonicalRecord] = {}
    tracked_scope_ids = {anchor.conflict_scope_id}
    tracked_record_ids = {anchor.record_id}
    tracked_source_urls = {anchor.source_url}

    changed = True
    while changed:
        changed = False
        rows = session.execute(
            select(CanonicalRecord).where(
                or_(
                    CanonicalRecord.conflict_scope_id.in_(tuple(tracked_scope_ids)),
                    CanonicalRecord.record_id.in_(tuple(tracked_record_ids)),
                    CanonicalRecord.source_url.in_(tuple(tracked_source_urls)),
                    CanonicalRecord.record_payload["parent_record_id"]
                    .as_string()
                    .in_(tuple(tracked_record_ids)),
                    CanonicalRecord.record_payload["parent_page_url"]
                    .as_string()
                    .in_(tuple(tracked_source_urls)),
                    CanonicalRecord.record_payload["parent_source_url"]
                    .as_string()
                    .in_(tuple(tracked_source_urls)),
                )
            )
        ).scalars().all()
        for row in rows:
            if row.record_version_id not in affected_by_version_id:
                affected_by_version_id[row.record_version_id] = row
                changed = True
            if row.conflict_scope_id not in tracked_scope_ids:
                tracked_scope_ids.add(row.conflict_scope_id)
                changed = True
            if row.record_id not in tracked_record_ids:
                tracked_record_ids.add(row.record_id)
                changed = True
            if row.source_url not in tracked_source_urls:
                tracked_source_urls.add(row.source_url)
                changed = True

    return tuple(affected_by_version_id.values())


def _apply_recomputed_decision(
    session: Session,
    row: CanonicalRecord,
    decision: VerificationDecision,
) -> None:
    was_indexed = row.serving_status == "indexed_active"
    was_pending_index = row.serving_status == "pending_index"
    was_pending_deindex = row.serving_status == "pending_deindex"

    row.freshness_status = decision.freshness_status
    row.verification_status = decision.verification_status
    row.is_current_candidate = decision.is_current_candidate
    row.is_current_authoritative = decision.is_current_authoritative
    row.verified_at = _utc_now() if decision.verification_status == "verified" else None

    if can_enter_serving(decision):
        row.serving_status = "indexed_active" if was_indexed else "pending_index"
        if not was_indexed or was_pending_index:
            _ensure_index_job(session, row, operation="index")
        return

    if was_indexed or was_pending_deindex:
        row.serving_status = "pending_deindex"
        _ensure_index_job(session, row, operation="deindex")
        return

    row.serving_status = "ineligible"


def _ensure_index_job(
    session: Session,
    row: CanonicalRecord,
    *,
    operation: str,
) -> IndexJob:
    existing = session.execute(
        select(IndexJob).where(
            IndexJob.record_version_id == row.record_version_id,
            IndexJob.operation == operation,
            IndexJob.status.in_(("pending", "running")),
        )
    ).scalar_one_or_none()
    if existing is not None:
        return existing

    job = IndexJob(
        record_version_id=row.record_version_id,
        operation=operation,
        status="pending",
        job_scope={
            "record_id": row.record_id,
            "conflict_scope_id": row.conflict_scope_id,
            "reason": "manual_review_resolution",
        },
    )
    session.add(job)
    session.flush()
    return job
