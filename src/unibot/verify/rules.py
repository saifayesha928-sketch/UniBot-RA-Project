from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Literal

YearConfidence = Literal["high", "medium", "low", "unknown"]
VerificationStatus = Literal["pending", "verified", "rejected"]
FreshnessStatus = Literal[
    "current",
    "stale",
    "unknown",
    "contradictory",
    "restricted",
    "removed",
]
ServingStatus = Literal[
    "eligible",
    "ineligible",
    "pending_index",
    "pending_deindex",
    "indexed_active",
    "deindexed",
    "failed",
]
ManualReviewReason = Literal["ambiguous_date", "same_tier_conflict", "requires_corroboration"]


@dataclass(frozen=True, slots=True)
class VerificationCandidate:
    record_id: str
    record_version_id: str
    record_type: str
    conflict_scope_id: str
    dedupe_key: str
    value_hash: str
    source_authority_tier: int
    source_url: str
    source_locator: str = "body"
    source_section_id: str | None = None
    source_section_label: str | None = None
    cycle_label: str | None = None
    effective_from: date | None = None
    effective_to: date | None = None
    fetched_at: datetime | None = None
    parent_record_id: str | None = None
    parent_source_url: str | None = None
    year_confidence: YearConfidence = "unknown"
    page_content_hash: str | None = None
    record_payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VerificationDecision:
    candidate: VerificationCandidate
    verification_status: VerificationStatus
    freshness_status: FreshnessStatus
    serving_status: ServingStatus
    is_current_candidate: bool
    is_current_authoritative: bool
    supporting_record_ids: tuple[str, ...] = ()
    conflicting_record_ids: tuple[str, ...] = ()
    requires_manual_review: bool = False
    manual_review_reason: ManualReviewReason | None = None
    notes: str | None = None


@dataclass(frozen=True, slots=True)
class SupportingEvidenceLink:
    primary_record_version_id: str
    supporting_record_version_id: str
    relation_type: str = "duplicate_support"


@dataclass(frozen=True, slots=True)
class DedupeConflict:
    dedupe_key: str
    record_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DedupeResult:
    primary_records: tuple[VerificationDecision, ...]
    supporting_links: tuple[SupportingEvidenceLink, ...]
    conflicts: tuple[DedupeConflict, ...]


@dataclass(frozen=True, slots=True)
class ManualReviewEvent:
    record_version_id: str
    event_type: str
    verification_status: VerificationStatus
    event_payload: dict[str, Any]
    source_section_id: str | None = None
    reviewer: str | None = None
    notes: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
