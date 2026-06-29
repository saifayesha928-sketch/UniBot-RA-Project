from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone

from unibot.domain.types import SourcePolicy
from unibot.verify.rules import VerificationCandidate, VerificationDecision
from unibot.verify.source_class_currentness import (
    CYCLE_AWARE_TYPES,
    DEFAULT_LATEST_VERSION_TYPES,
    ParentState,
)

_AMBIGUOUS_YEAR_CONFIDENCE = {"low", "unknown"}
_NON_SERVING_FRESHNESS = {"stale", "unknown", "contradictory", "restricted", "removed"}
_NON_CONTRADICTORY_SAME_TIER_TYPES = DEFAULT_LATEST_VERSION_TYPES | frozenset(
    {"faculty_profile"}
)


def classify_currentness(
    candidate: VerificationCandidate,
    siblings: Iterable[VerificationCandidate],
    source_policy: SourcePolicy,
    *,
    parent_state: ParentState | None = None,
) -> VerificationDecision:
    sibling_list = list(siblings)

    # Shared guard: source policy check
    if (
        source_policy.access_level != "allowed"
        or not source_policy.allows_automated_crawling
    ):
        return VerificationDecision(
            candidate=candidate,
            verification_status="rejected",
            freshness_status="restricted",
            serving_status="ineligible",
            is_current_candidate=False,
            is_current_authoritative=False,
            notes="source policy blocks serving eligibility",
        )

    today = datetime.now(timezone.utc).date()
    if candidate.effective_to is not None and candidate.effective_to < today:
        return VerificationDecision(
            candidate=candidate,
            verification_status="verified",
            freshness_status="stale",
            serving_status="ineligible",
            is_current_candidate=True,
            is_current_authoritative=False,
            notes="candidate expired because effective_to has passed",
        )

    # Shared guard: same-tier conflict detection
    conflict_ids = _same_tier_conflicts(candidate, sibling_list)
    if conflict_ids:
        return VerificationDecision(
            candidate=candidate,
            verification_status="pending",
            freshness_status="contradictory",
            serving_status="ineligible",
            is_current_candidate=False,
            is_current_authoritative=False,
            conflicting_record_ids=conflict_ids,
            requires_manual_review=True,
            manual_review_reason="same_tier_conflict",
            notes="same-tier contradiction detected",
        )

    # Shared guard: ambiguous year check (only for cycle-aware types)
    if (
        candidate.record_type in CYCLE_AWARE_TYPES
        and candidate.year_confidence in _AMBIGUOUS_YEAR_CONFIDENCE
    ):
        return VerificationDecision(
            candidate=candidate,
            verification_status="pending",
            freshness_status="unknown",
            serving_status="ineligible",
            is_current_candidate=False,
            is_current_authoritative=False,
            requires_manual_review=True,
            manual_review_reason="ambiguous_date",
            notes="candidate cannot be made current with an ambiguous year",
        )

    # Delegate to source-class-specific rules
    from unibot.verify.source_class_currentness import classify

    return classify(candidate, sibling_list, source_policy, parent_state=parent_state)


def can_enter_serving(decision: VerificationDecision) -> bool:
    return (
        decision.verification_status == "verified"
        and decision.freshness_status == "current"
        and decision.is_current_authoritative
        and decision.serving_status == "eligible"
    )


def _same_tier_conflicts(
    candidate: VerificationCandidate,
    siblings: list[VerificationCandidate],
) -> tuple[str, ...]:
    relevant_siblings = []
    for sibling in siblings:
        if sibling.conflict_scope_id != candidate.conflict_scope_id:
            continue
        if sibling.source_authority_tier != candidate.source_authority_tier:
            continue
        if sibling.record_version_id == candidate.record_version_id:
            continue
        # Different record types in the same scope are not conflicts
        if sibling.record_type != candidate.record_type:
            continue
        # Latest-version record types supersede older same-tier values instead of contradicting.
        if candidate.record_type in _NON_CONTRADICTORY_SAME_TIER_TYPES:
            continue
        # Cycle-aware types with different cycle labels are supersessions, not conflicts
        if (
            candidate.record_type in CYCLE_AWARE_TYPES
            and candidate.cycle_label is not None
            and sibling.cycle_label is not None
            and sibling.cycle_label != candidate.cycle_label
        ):
            continue
        relevant_siblings.append(sibling)

    from unibot.verify.value_identity import effective_value_hash

    candidate_hash = effective_value_hash(
        candidate.record_type, candidate.record_payload, candidate.value_hash
    )
    conflicting_siblings = [
        sibling
        for sibling in relevant_siblings
        if effective_value_hash(
            sibling.record_type, sibling.record_payload, sibling.value_hash
        )
        != candidate_hash
    ]
    if not conflicting_siblings:
        return ()

    # For admissions_cycle: use source specificity to break same-tier ties
    if candidate.record_type == "admissions_cycle" and conflicting_siblings:
        from unibot.verify.source_specificity import compute_source_specificity
        from unibot.verify.source_class_currentness import _rank_latest

        candidate_specificity = compute_source_specificity(
            candidate.source_url, candidate.record_type
        )
        sibling_specificities = {
            sibling.record_version_id: compute_source_specificity(
                sibling.source_url, sibling.record_type
            )
            for sibling in conflicting_siblings
        }
        max_specificity = max(
            sibling_specificities.values(),
            default=candidate_specificity,
        )
        if candidate_specificity > max_specificity:
            return ()
        if candidate_specificity < max_specificity:
            conflicting_ids = {s.record_id for s in conflicting_siblings}
            conflicting_ids.add(candidate.record_id)
            return tuple(sorted(conflicting_ids))

        equal_specificity_siblings = [
            sibling
            for sibling in conflicting_siblings
            if sibling_specificities[sibling.record_version_id] == candidate_specificity
        ]
        candidate_rank = _rank_latest(candidate)[:-1]
        if all(
            candidate_rank > _rank_latest(sibling)[:-1]
            for sibling in equal_specificity_siblings
        ):
            return ()

    conflicting_ids = {s.record_id for s in conflicting_siblings}
    conflicting_ids.add(candidate.record_id)
    return tuple(sorted(conflicting_ids))
