from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from unibot.db.models import CanonicalRecord
from unibot.domain.source_policies import get_source_policy
from unibot.pipeline.contracts import _EXPLICIT_PARENT_STATE_TYPES, _SERVING_ELIGIBLE_STATUSES
from unibot.verify.currentness import classify_currentness
from unibot.verify.rules import VerificationCandidate, VerificationDecision
from unibot.verify.source_class_currentness import ParentState
from unibot.verify.value_identity import value_hash_for_stored_record


@dataclass(frozen=True, slots=True)
class ParentStateLookup:
    by_record_id: dict[str, ParentState]
    by_source_url: dict[str, ParentState]

    def resolve(self, candidate: VerificationCandidate) -> ParentState | None:
        if candidate.parent_record_id is not None:
            return self.by_record_id.get(candidate.parent_record_id, ParentState.missing())
        if candidate.parent_source_url is not None:
            return self.by_source_url.get(candidate.parent_source_url, ParentState.missing())
        return None


def candidate_from_row(row: CanonicalRecord) -> VerificationCandidate:
    payload = row.record_payload or {}
    return VerificationCandidate(
        record_id=row.record_id,
        record_version_id=row.record_version_id,
        record_type=row.record_type,
        conflict_scope_id=row.conflict_scope_id,
        dedupe_key=row.dedupe_key,
        value_hash=value_hash_for_stored_record(
            row.record_type,
            payload,
            row.source_text_hash,
        ),
        source_authority_tier=row.source_authority_tier,
        source_url=row.source_url,
        source_locator=row.source_locator,
        source_section_id=row.source_section_id,
        source_section_label=row.source_section_label,
        cycle_label=row.cycle_label,
        effective_from=row.effective_from,
        effective_to=row.effective_to,
        fetched_at=row.fetched_at,
        parent_record_id=payload.get("parent_record_id"),
        parent_source_url=payload.get("parent_page_url") or payload.get("parent_source_url"),
        year_confidence=row.year_confidence,
        page_content_hash=row.page_content_hash,
        record_payload=payload,
    )


def build_parent_state_lookup(
    *,
    session: Session,
    combined_candidates_by_scope: dict[str, list[VerificationCandidate]],
    base_decisions: dict[str, VerificationDecision],
) -> ParentStateLookup:
    record_id_states: dict[str, tuple[tuple[object, ...], ParentState]] = {}
    source_url_states: dict[str, tuple[tuple[object, ...], ParentState]] = {}

    for decision in base_decisions.values():
        _remember_parent_state(
            record_id_states=record_id_states,
            source_url_states=source_url_states,
            record_id=decision.candidate.record_id,
            source_url=decision.candidate.source_url,
            state=ParentState.from_decision(decision),
            fetched_at=decision.candidate.fetched_at,
            record_version_id=decision.candidate.record_version_id,
        )

    for row in load_parent_rows(
        session=session,
        combined_candidates_by_scope=combined_candidates_by_scope,
    ):
        _remember_parent_state(
            record_id_states=record_id_states,
            source_url_states=source_url_states,
            record_id=row.record_id,
            source_url=row.source_url,
            state=ParentState(
                resolved=True,
                verification_status=row.verification_status,
                freshness_status=row.freshness_status,
                serving_status=row.serving_status,
                is_current_authoritative=row.is_current_authoritative,
            ),
            fetched_at=row.fetched_at,
            record_version_id=row.record_version_id,
        )

    return ParentStateLookup(
        by_record_id={key: state for key, (_rank, state) in record_id_states.items()},
        by_source_url={key: state for key, (_rank, state) in source_url_states.items()},
    )


def classify_candidates_for_parent_resolution(
    *,
    session: Session,
    candidates_by_scope: dict[str, list[VerificationCandidate]],
) -> dict[str, VerificationDecision]:
    record_id_states: dict[str, tuple[tuple[object, ...], ParentState]] = {}
    source_url_states: dict[str, tuple[tuple[object, ...], ParentState]] = {}

    for row in load_parent_rows(
        session=session,
        combined_candidates_by_scope=candidates_by_scope,
    ):
        _remember_parent_state(
            record_id_states=record_id_states,
            source_url_states=source_url_states,
            record_id=row.record_id,
            source_url=row.source_url,
            state=ParentState(
                resolved=True,
                verification_status=row.verification_status,
                freshness_status=row.freshness_status,
                serving_status=row.serving_status,
                is_current_authoritative=row.is_current_authoritative,
            ),
            fetched_at=row.fetched_at,
            record_version_id=row.record_version_id,
        )

    decisions: dict[str, VerificationDecision] = {}
    ordered_candidates = sorted(
        (
            candidate
            for candidates in candidates_by_scope.values()
            for candidate in candidates
        ),
        key=_candidate_parent_resolution_rank,
    )

    for candidate in ordered_candidates:
        siblings = [
            sibling
            for sibling in candidates_by_scope[candidate.conflict_scope_id]
            if sibling.record_version_id != candidate.record_version_id
        ]
        decision = classify_currentness(
            candidate,
            siblings,
            get_source_policy(candidate.source_url),
            parent_state=_resolve_parent_state(
                candidate,
                record_id_states=record_id_states,
                source_url_states=source_url_states,
            ),
        )
        decisions[candidate.record_version_id] = decision
        _remember_parent_state(
            record_id_states=record_id_states,
            source_url_states=source_url_states,
            record_id=decision.candidate.record_id,
            source_url=decision.candidate.source_url,
            state=ParentState.from_decision(decision),
            fetched_at=decision.candidate.fetched_at,
            record_version_id=decision.candidate.record_version_id,
        )

    return decisions


def _candidate_parent_resolution_rank(candidate: VerificationCandidate) -> tuple[int, str, str]:
    return (
        1 if candidate.record_type in _EXPLICIT_PARENT_STATE_TYPES else 0,
        candidate.conflict_scope_id,
        candidate.record_version_id,
    )


def _resolve_parent_state(
    candidate: VerificationCandidate,
    *,
    record_id_states: dict[str, tuple[tuple[object, ...], ParentState]],
    source_url_states: dict[str, tuple[tuple[object, ...], ParentState]],
) -> ParentState | None:
    if candidate.record_type not in _EXPLICIT_PARENT_STATE_TYPES:
        return None

    parent_record_id = candidate.parent_record_id or candidate.record_payload.get(
        "parent_record_id"
    )
    parent_source_url = candidate.parent_source_url or candidate.record_payload.get(
        "parent_page_url"
    ) or candidate.record_payload.get("parent_source_url")

    if parent_record_id is not None:
        return record_id_states.get(parent_record_id, ((None,), ParentState.missing()))[1]
    if parent_source_url is not None:
        return source_url_states.get(
            parent_source_url,
            ((None,), ParentState.missing()),
        )[1]
    return None


def load_parent_rows(
    *,
    session: Session,
    combined_candidates_by_scope: dict[str, list[VerificationCandidate]],
) -> tuple[CanonicalRecord, ...]:
    parent_record_ids = {
        candidate.parent_record_id
        for combined_candidates in combined_candidates_by_scope.values()
        for candidate in combined_candidates
        if candidate.parent_record_id is not None
    }
    parent_source_urls = {
        candidate.parent_source_url
        for combined_candidates in combined_candidates_by_scope.values()
        for candidate in combined_candidates
        if candidate.parent_source_url is not None
    }

    filters = []
    if parent_record_ids:
        filters.append(CanonicalRecord.record_id.in_(tuple(parent_record_ids)))
    if parent_source_urls:
        filters.append(CanonicalRecord.source_url.in_(tuple(parent_source_urls)))
    if not filters:
        return ()

    return tuple(session.execute(select(CanonicalRecord).where(or_(*filters))).scalars().all())


def _remember_parent_state(
    *,
    record_id_states: dict[str, tuple[tuple[object, ...], ParentState]],
    source_url_states: dict[str, tuple[tuple[object, ...], ParentState]],
    record_id: str,
    source_url: str,
    state: ParentState,
    fetched_at: datetime | None,
    record_version_id: str,
) -> None:
    rank = _parent_state_rank(
        state=state,
        fetched_at=fetched_at,
        record_version_id=record_version_id,
    )
    _set_best_parent_state(record_id_states, record_id, rank, state)
    _set_best_parent_state(source_url_states, source_url, rank, state)


def _set_best_parent_state(
    states: dict[str, tuple[tuple[object, ...], ParentState]],
    key: str,
    rank: tuple[object, ...],
    state: ParentState,
) -> None:
    existing = states.get(key)
    if existing is None or rank > existing[0]:
        states[key] = (rank, state)


def _parent_state_rank(
    *,
    state: ParentState,
    fetched_at: datetime | None,
    record_version_id: str,
) -> tuple[object, ...]:
    return (
        state.resolved,
        state.is_current_authoritative,
        state.verification_status == "verified",
        state.freshness_status == "current",
        state.serving_status in _SERVING_ELIGIBLE_STATUSES,
        normalize_fetched_at(fetched_at),
        record_version_id,
    )


def normalize_fetched_at(fetched_at: datetime | None) -> datetime:
    if fetched_at is None:
        return datetime.min.replace(tzinfo=timezone.utc)
    if fetched_at.tzinfo is None:
        return fetched_at.replace(tzinfo=timezone.utc)
    return fetched_at.astimezone(timezone.utc)
