"""Source-class-aware currentness classification.

Registry of rule functions keyed by record type. Each rule receives the
candidate, siblings in the same scope, and source policy, and returns a
VerificationDecision. Unknown record types fail closed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timezone

from unibot.domain.types import SourcePolicy
from unibot.verify.value_identity import effective_value_hash
from unibot.verify.rules import (
    FreshnessStatus,
    ServingStatus,
    VerificationCandidate,
    VerificationDecision,
    VerificationStatus,
)

_AMBIGUOUS_YEAR_CONFIDENCE = {"low", "unknown"}

# Record types that use cycle_label to distinguish non-conflicting records
CYCLE_AWARE_TYPES = frozenset({"admissions_cycle", "program_fee_schedule", "merit_list"})

# Record types whose children inherit parent serving eligibility
PARENT_DEPENDENT_TYPES = frozenset({"faculty_publication", "faculty_award", "faculty_affiliation"})

# Record types that use default latest-version rule
DEFAULT_LATEST_VERSION_TYPES = frozenset({
    "general",
    "program",
    "program_curriculum",
    "university_info",
    "student_service",
    "org_unit",
    "research_entity",
    "news_event",
    "scholarship",
    "evidence",
    "document_landing",
})

# Record types that use version-aware supersession
VERSION_AWARE_TYPES = frozenset({"faculty_profile", "policy_rule"})

# Context-dependent types
CONTEXT_DEPENDENT_TYPES = frozenset({"document_asset"})

RuleFunction = Callable[
    [VerificationCandidate, list[VerificationCandidate], SourcePolicy, "ParentState | None"],
    VerificationDecision,
]

_MIN_FETCHED_AT = datetime.min.replace(tzinfo=timezone.utc)


@dataclass(frozen=True, slots=True)
class ParentState:
    resolved: bool
    verification_status: VerificationStatus
    freshness_status: FreshnessStatus
    serving_status: ServingStatus
    is_current_authoritative: bool

    @classmethod
    def missing(cls) -> "ParentState":
        return cls(
            resolved=False,
            verification_status="pending",
            freshness_status="unknown",
            serving_status="ineligible",
            is_current_authoritative=False,
        )

    @classmethod
    def from_decision(cls, decision: VerificationDecision) -> "ParentState":
        return cls(
            resolved=True,
            verification_status=decision.verification_status,
            freshness_status=decision.freshness_status,
            serving_status=decision.serving_status,
            is_current_authoritative=decision.is_current_authoritative,
        )


def _cycle_aware_supersession(
    candidate: VerificationCandidate,
    siblings: list[VerificationCandidate],
    source_policy: SourcePolicy,
    parent_state: ParentState | None = None,
) -> VerificationDecision:
    """Newest by effective_from is current, older are stale."""
    comparable = [
        record
        for record in [candidate, *siblings]
        if record.conflict_scope_id == candidate.conflict_scope_id
        and record.record_type == candidate.record_type
        and record.year_confidence not in _AMBIGUOUS_YEAR_CONFIDENCE
    ]
    if not comparable:
        return VerificationDecision(
            candidate=candidate,
            verification_status="pending",
            freshness_status="unknown",
            serving_status="ineligible",
            is_current_candidate=False,
            is_current_authoritative=False,
            requires_manual_review=True,
            manual_review_reason="ambiguous_date",
            notes=f"{candidate.record_type} needs a resolved effective date",
        )

    latest = max(
        comparable,
        key=lambda r: (
            r.effective_from or date.min,
            _normalized_fetched_at(r.fetched_at),
            -r.source_authority_tier,
            r.record_id,
        ),
    )

    if latest.record_version_id != candidate.record_version_id:
        return VerificationDecision(
            candidate=candidate,
            verification_status="verified",
            freshness_status="stale",
            serving_status="ineligible",
            is_current_candidate=True,
            is_current_authoritative=False,
            notes=f"a newer {candidate.record_type} exists in the same scope",
        )

    # Gate: weekday-inferred admissions dates require corroboration
    if (
        candidate.record_type == "admissions_cycle"
        and (candidate.record_payload or {}).get("date_resolution") == "weekday_inferred"
        and not _has_admissions_corroboration(candidate, comparable)
    ):
        return VerificationDecision(
            candidate=candidate,
            verification_status="verified",
            freshness_status="current",
            serving_status="ineligible",
            is_current_candidate=True,
            is_current_authoritative=False,
            requires_manual_review=True,
            manual_review_reason="requires_corroboration",
            notes="weekday-inferred admissions date requires corroboration",
        )

    return VerificationDecision(
        candidate=candidate,
        verification_status="verified",
        freshness_status="current",
        serving_status="eligible",
        is_current_candidate=True,
        is_current_authoritative=True,
    )


def _has_admissions_corroboration(
    candidate: VerificationCandidate,
    comparable: list[VerificationCandidate],
) -> bool:
    candidate_hash = effective_value_hash(
        candidate.record_type,
        candidate.record_payload or {},
        candidate.value_hash,
    )
    for sibling in comparable:
        if sibling.record_version_id == candidate.record_version_id:
            continue
        sibling_hash = effective_value_hash(
            sibling.record_type,
            sibling.record_payload or {},
            sibling.value_hash,
        )
        if sibling_hash != candidate_hash:
            continue

        sibling_resolution = (sibling.record_payload or {}).get("date_resolution")
        if sibling_resolution == "explicit":
            return True
        if sibling.source_url != candidate.source_url:
            return True
    return False


def _version_aware_supersession(
    candidate: VerificationCandidate,
    siblings: list[VerificationCandidate],
    source_policy: SourcePolicy,
    parent_state: ParentState | None = None,
) -> VerificationDecision:
    """Latest version by record_version_id is current."""
    return _default_latest_version(candidate, siblings, source_policy)


def _parent_dependent(
    candidate: VerificationCandidate,
    siblings: list[VerificationCandidate],
    source_policy: SourcePolicy,
    parent_state: ParentState | None = None,
) -> VerificationDecision:
    """Inherits serving eligibility from parent record."""
    resolved_parent = parent_state or _fallback_parent_state(candidate, siblings)
    if resolved_parent is None or not resolved_parent.resolved:
        return _pending_unknown_parent(candidate)
    if (
        not resolved_parent.is_current_authoritative
        or resolved_parent.verification_status != "verified"
    ):
        return _ineligible_child(candidate, resolved_parent, note_prefix="parent")

    return VerificationDecision(
        candidate=candidate,
        verification_status="verified",
        freshness_status="current",
        serving_status="eligible",
        is_current_candidate=True,
        is_current_authoritative=True,
    )


def _context_dependent(
    candidate: VerificationCandidate,
    siblings: list[VerificationCandidate],
    source_policy: SourcePolicy,
    parent_state: ParentState | None = None,
) -> VerificationDecision:
    """Document assets need authoritative parent signal."""
    resolved_parent = parent_state or _fallback_parent_state(candidate, siblings)
    if resolved_parent is None or not resolved_parent.resolved:
        return _pending_unknown_parent(candidate)
    if (
        not resolved_parent.is_current_authoritative
        or resolved_parent.verification_status != "verified"
    ):
        return _ineligible_child(candidate, resolved_parent, note_prefix="parent page")
    return _default_latest_version(candidate, siblings, source_policy)


def _default_latest_version(
    candidate: VerificationCandidate,
    siblings: list[VerificationCandidate],
    source_policy: SourcePolicy,
    parent_state: ParentState | None = None,
) -> VerificationDecision:
    """Latest verified version is current, previous are stale."""
    comparable = [
        record
        for record in [candidate, *siblings]
        if record.conflict_scope_id == candidate.conflict_scope_id
        and record.record_type == candidate.record_type
    ]
    latest = max(comparable, key=_rank_latest)
    if latest.record_version_id != candidate.record_version_id:
        if latest.value_hash == candidate.value_hash:
            return _stale_duplicate_decision(candidate)
        return _stale_superseded_decision(candidate)

    return VerificationDecision(
        candidate=candidate,
        verification_status="verified",
        freshness_status="current",
        serving_status="eligible",
        is_current_candidate=True,
        is_current_authoritative=True,
    )


def _rank_latest(record: VerificationCandidate) -> tuple[object, ...]:
    return (
        record.effective_from or date.min,
        _normalized_fetched_at(record.fetched_at),
        -record.source_authority_tier,
        record.record_version_id,
    )


def _normalized_fetched_at(fetched_at: datetime | None) -> datetime:
    if fetched_at is None:
        return _MIN_FETCHED_AT
    if fetched_at.tzinfo is None:
        return fetched_at.replace(tzinfo=timezone.utc)
    return fetched_at.astimezone(timezone.utc)


def _stale_duplicate_decision(candidate: VerificationCandidate) -> VerificationDecision:
    return VerificationDecision(
        candidate=candidate,
        verification_status="verified",
        freshness_status="stale",
        serving_status="ineligible",
        is_current_candidate=True,
        is_current_authoritative=False,
        notes=f"a newer duplicate {candidate.record_type} exists in the same scope",
    )


def _stale_superseded_decision(candidate: VerificationCandidate) -> VerificationDecision:
    return VerificationDecision(
        candidate=candidate,
        verification_status="verified",
        freshness_status="stale",
        serving_status="ineligible",
        is_current_candidate=True,
        is_current_authoritative=False,
        notes=f"a newer {candidate.record_type} exists in the same scope",
    )


def _unknown_type_fail_closed(
    candidate: VerificationCandidate,
    siblings: list[VerificationCandidate],
    source_policy: SourcePolicy,
    parent_state: ParentState | None = None,
) -> VerificationDecision:
    """Unknown record types fail closed."""
    return VerificationDecision(
        candidate=candidate,
        verification_status="pending",
        freshness_status="unknown",
        serving_status="ineligible",
        is_current_candidate=False,
        is_current_authoritative=False,
        notes=f"unknown record type: {candidate.record_type}",
    )


# Build the registry
_RULE_REGISTRY: dict[str, RuleFunction] = {}

for _t in CYCLE_AWARE_TYPES:
    _RULE_REGISTRY[_t] = _cycle_aware_supersession

for _t in VERSION_AWARE_TYPES:
    _RULE_REGISTRY[_t] = _version_aware_supersession

for _t in PARENT_DEPENDENT_TYPES:
    _RULE_REGISTRY[_t] = _parent_dependent

for _t in CONTEXT_DEPENDENT_TYPES:
    _RULE_REGISTRY[_t] = _context_dependent

for _t in DEFAULT_LATEST_VERSION_TYPES:
    _RULE_REGISTRY[_t] = _default_latest_version


def classify(
    candidate: VerificationCandidate,
    siblings: list[VerificationCandidate],
    source_policy: SourcePolicy,
    parent_state: ParentState | None = None,
) -> VerificationDecision:
    """Dispatch to the registered rule function for this record type."""
    rule = _RULE_REGISTRY.get(candidate.record_type, _unknown_type_fail_closed)
    return rule(candidate, siblings, source_policy, parent_state)


def _pending_unknown_parent(candidate: VerificationCandidate) -> VerificationDecision:
    return VerificationDecision(
        candidate=candidate,
        verification_status="pending",
        freshness_status="unknown",
        serving_status="ineligible",
        is_current_candidate=False,
        is_current_authoritative=False,
        notes="parent state could not be resolved",
    )


def _ineligible_child(
    candidate: VerificationCandidate,
    parent_state: ParentState,
    *,
    note_prefix: str,
) -> VerificationDecision:
    freshness_status = (
        "stale"
        if parent_state.freshness_status == "current"
        else parent_state.freshness_status
    )
    verification_status: VerificationStatus = (
        "verified" if parent_state.verification_status == "verified" else "pending"
    )
    return VerificationDecision(
        candidate=candidate,
        verification_status=verification_status,
        freshness_status=freshness_status,
        serving_status="ineligible",
        is_current_candidate=False,
        is_current_authoritative=False,
        notes=f"{note_prefix} is not current authoritative",
    )


def _fallback_parent_state(
    candidate: VerificationCandidate,
    siblings: list[VerificationCandidate],
) -> ParentState | None:
    parent_record_id = candidate.parent_record_id or candidate.record_payload.get(
        "parent_record_id"
    )
    parent_source_url = candidate.parent_source_url or candidate.record_payload.get(
        "parent_page_url"
    ) or candidate.record_payload.get("parent_source_url")
    if parent_record_id is None and parent_source_url is None:
        return None

    for sibling in siblings:
        if parent_record_id is not None and sibling.record_id == parent_record_id:
            return _parent_state_from_candidate(sibling)
        if parent_source_url is not None and sibling.source_url == parent_source_url:
            return _parent_state_from_candidate(sibling)
    return None


def _parent_state_from_candidate(candidate: VerificationCandidate) -> ParentState | None:
    if (candidate.record_payload or {}).get("_freshness_override") != "removed":
        return None
    return ParentState(
        resolved=True,
        verification_status="verified",
        freshness_status="removed",
        serving_status="ineligible",
        is_current_authoritative=False,
    )
