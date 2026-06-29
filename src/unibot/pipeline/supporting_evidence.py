from __future__ import annotations

from collections.abc import Callable

from sqlalchemy import delete, or_, select
from sqlalchemy.orm import Session

from unibot.db.models import CanonicalRecord, SourceRegistry, SupportingEvidenceLink, VerificationEvent
from unibot.pipeline.contracts import _SERVING_ELIGIBLE_STATUSES
from unibot.pipeline.decision_engine import candidate_from_row
from unibot.verify.deduplication import resolve_duplicates
from unibot.verify.rules import VerificationDecision


def load_current_authoritative_decisions(
    session: Session,
) -> tuple[VerificationDecision, ...]:
    rows = session.execute(
        select(CanonicalRecord, SourceRegistry.legal_status)
        .join(
            SourceRegistry,
            CanonicalRecord.source_id == SourceRegistry.source_id,
            isouter=True,
        )
        .where(CanonicalRecord.is_current_authoritative.is_(True))
        .where(CanonicalRecord.freshness_status == "current")
        .where(CanonicalRecord.verification_status == "verified")
        .where(CanonicalRecord.serving_status.in_(tuple(_SERVING_ELIGIBLE_STATUSES)))
        .where(
            or_(
                SourceRegistry.legal_status.is_(None),
                SourceRegistry.legal_status == "allowed",
            )
        )
    ).all()

    decisions: list[VerificationDecision] = []
    for row, _legal_status in rows:
        candidate = candidate_from_row(row)
        decisions.append(
            VerificationDecision(
                candidate=candidate,
                verification_status=row.verification_status,
                freshness_status=row.freshness_status,
                serving_status="eligible",
                is_current_candidate=row.is_current_candidate,
                is_current_authoritative=row.is_current_authoritative,
            )
        )
    return tuple(decisions)


def persist_supporting_links(
    session: Session,
    create_verification_event: Callable[[VerificationDecision], VerificationEvent],
) -> None:
    session.execute(delete(SupportingEvidenceLink))

    dedupe_result = resolve_duplicates(load_current_authoritative_decisions(session))
    for link in dedupe_result.supporting_links:
        session.add(
            SupportingEvidenceLink(
                primary_record_version_id=link.primary_record_version_id,
                supporting_record_version_id=link.supporting_record_version_id,
                relation_type=link.relation_type,
            )
        )

    for conflict in dedupe_result.conflicts:
        conflict_rows = []
        for record_version_id in conflict.record_ids:
            row = session.get(CanonicalRecord, record_version_id)
            if row is not None:
                conflict_rows.append(row)

        for row in conflict_rows:
            was_indexed = row.serving_status == "indexed_active"
            row.freshness_status = "contradictory"
            row.serving_status = "pending_deindex" if was_indexed else "ineligible"
            row.is_current_authoritative = False
            row.is_current_candidate = False

            other_record_ids = tuple(sorted(
                r.record_id
                for r in conflict_rows
                if r.record_version_id != row.record_version_id
            ))
            candidate = candidate_from_row(row)
            decision = VerificationDecision(
                candidate=candidate,
                verification_status="pending",
                freshness_status="contradictory",
                serving_status=row.serving_status,
                is_current_candidate=False,
                is_current_authoritative=False,
                conflicting_record_ids=other_record_ids,
                requires_manual_review=True,
                manual_review_reason="same_tier_conflict",
                notes=f"duplicate conflict in dedupe_key={conflict.dedupe_key}",
            )
            create_verification_event(decision)

    session.flush()
