from __future__ import annotations

from datetime import date, datetime
from uuid import uuid4

from sqlalchemy import Boolean, CheckConstraint, Date, DateTime, ForeignKey, Integer
from sqlalchemy import JSON, SmallInteger, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, validates

from unibot.db.base import Base
from unibot.utils import utc_now as _utc_now
from unibot.verify.rules import FreshnessStatus, ServingStatus, VerificationStatus, YearConfidence


VALID_CANONICAL_RECORD_TYPES = (
    "general",
    "admissions_cycle",
    "program",
    "program_curriculum",
    "program_fee_schedule",
    "merit_list",
    "faculty_profile",
    "faculty_publication",
    "faculty_award",
    "faculty_affiliation",
    "research_entity",
    "scholarship",
    "news_event",
    "policy_rule",
    "document_landing",
    "document_asset",
    "evidence",
    "student_service",
    "university_info",
    "org_unit",
)
VALID_EXTRACTION_CONFIDENCE_VALUES = ("high", "medium", "low", "unknown", "rule_based")


def _quoted_constraint_values(values: tuple[str, ...]) -> str:
    return ", ".join(f"'{value}'" for value in values)


class SourceRegistry(Base):
    __tablename__ = "source_registry"

    source_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    source_url: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    canonical_url: Mapped[str] = mapped_column(Text, nullable=False)
    source_class: Mapped[str] = mapped_column(String(128), nullable=False)
    crawl_method: Mapped[str] = mapped_column(String(64), nullable=False)
    legal_status: Mapped[str] = mapped_column(String(32), nullable=False)
    crawl_status: Mapped[str | None] = mapped_column(String(32))
    default_authority_tier: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    refresh_policy: Mapped[str] = mapped_column(String(64), nullable=False)
    parser_target: Mapped[str | None] = mapped_column(String(32))
    parent_source_url: Mapped[str | None] = mapped_column(Text)
    link_text: Mapped[str | None] = mapped_column(Text)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_crawled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_successful_crawl_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    disappeared_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default="true")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utc_now,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utc_now,
        onupdate=_utc_now,
    )


class RawSnapshot(Base):
    __tablename__ = "raw_snapshots"

    snapshot_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    source_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("source_registry.source_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    content_type: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_uri: Mapped[str] = mapped_column(Text, nullable=False)
    storage_backend: Mapped[str] = mapped_column(String(32), nullable=False)
    page_content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    http_status: Mapped[int | None] = mapped_column(Integer)
    etag: Mapped[str | None] = mapped_column(String(255))
    last_modified: Mapped[str | None] = mapped_column(String(255))
    fetch_metadata: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utc_now,
    )


class SourceSection(Base):
    __tablename__ = "source_sections"
    __table_args__ = (
        UniqueConstraint("snapshot_id", "section_order"),
    )

    source_section_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    snapshot_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("raw_snapshots.snapshot_id", ondelete="CASCADE"),
        nullable=False,
    )
    source_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("source_registry.source_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    section_label: Mapped[str] = mapped_column(String(255), nullable=False)
    section_type: Mapped[str] = mapped_column(String(64), nullable=False)
    source_locator: Mapped[str] = mapped_column(Text, nullable=False)
    section_order: Mapped[int] = mapped_column(Integer, nullable=False)
    text_content: Mapped[str] = mapped_column(Text, nullable=False)
    source_text_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    parser_backend: Mapped[str | None] = mapped_column(String(64))
    page_number: Mapped[int | None] = mapped_column(Integer)
    grounding_data: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    extracted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utc_now,
    )

    @validates("text_content", "section_label", "source_locator")
    def _strip_nul_bytes(self, _key: str, value: str) -> str:
        if value and "\x00" in value:
            return value.replace("\x00", "")
        return value


class CanonicalRecord(Base):
    __tablename__ = "canonical_records"
    __table_args__ = (
        CheckConstraint(
            f"record_type IN ({_quoted_constraint_values(VALID_CANONICAL_RECORD_TYPES)})",
            name="record_type_valid",
        ),
        CheckConstraint(
            "source_authority_tier BETWEEN 1 AND 5",
            name="source_authority_tier_valid",
        ),
        CheckConstraint(
            "year_confidence IN ('high', 'medium', 'low', 'unknown')",
            name="year_confidence_valid",
        ),
        CheckConstraint(
            "extraction_confidence IN "
            f"({_quoted_constraint_values(VALID_EXTRACTION_CONFIDENCE_VALUES)})",
            name="extraction_confidence_valid",
        ),
        CheckConstraint(
            "freshness_status IN ('current', 'stale', 'unknown', 'contradictory', 'restricted', 'removed')",
            name="freshness_status_valid",
        ),
        CheckConstraint(
            "verification_status IN ('pending', 'verified', 'rejected')",
            name="verification_status_valid",
        ),
        CheckConstraint(
            "serving_status IN ('eligible', 'ineligible', 'pending_index', 'pending_deindex', 'indexed_active', 'deindexed', 'failed')",
            name="serving_status_valid",
        ),
        CheckConstraint(
            "(is_current_authoritative = false) OR "
            "(freshness_status = 'current' AND verification_status = 'verified')",
            name="current_authoritative_requires_verified_current",
        ),
    )

    record_version_id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    record_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    record_type: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    source_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("source_registry.source_id", ondelete="SET NULL"),
        index=True,
    )
    source_section_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("source_sections.source_section_id", ondelete="SET NULL"),
    )
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    canonical_url: Mapped[str] = mapped_column(Text, nullable=False)
    source_title: Mapped[str | None] = mapped_column(String(512))
    source_section_label: Mapped[str | None] = mapped_column(String(255))
    source_locator: Mapped[str] = mapped_column(Text, nullable=False)
    source_text_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    page_content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    source_last_modified_text: Mapped[str | None] = mapped_column(Text)
    source_authority_tier: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    conflict_scope_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    dedupe_key: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    record_payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    effective_from: Mapped[date | None] = mapped_column(Date)
    effective_to: Mapped[date | None] = mapped_column(Date)
    cycle_label: Mapped[str | None] = mapped_column(String(255))
    year_confidence: Mapped[YearConfidence] = mapped_column(String(16), nullable=False, default="unknown")
    extraction_confidence: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="rule_based",
        server_default="rule_based",
    )
    freshness_status: Mapped[FreshnessStatus] = mapped_column(String(32), nullable=False, default="unknown", index=True)
    verification_status: Mapped[VerificationStatus] = mapped_column(String(32), nullable=False, default="pending", index=True)
    serving_status: Mapped[ServingStatus] = mapped_column(String(32), nullable=False, default="ineligible", index=True)
    supersedes_version_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("canonical_records.record_version_id", ondelete="SET NULL"),
    )
    superseded_by_version_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("canonical_records.record_version_id", ondelete="SET NULL"),
    )
    is_current_candidate: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
    )
    is_current_authoritative: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utc_now,
    )


class SupportingEvidenceLink(Base):
    __tablename__ = "supporting_evidence_links"
    __table_args__ = (
        UniqueConstraint("primary_record_version_id", "supporting_record_version_id"),
    )

    evidence_link_id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    primary_record_version_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("canonical_records.record_version_id", ondelete="CASCADE"),
        nullable=False,
    )
    supporting_record_version_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("canonical_records.record_version_id", ondelete="CASCADE"),
        nullable=False,
    )
    relation_type: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utc_now,
    )


class ServingGeneration(Base):
    __tablename__ = "serving_generations"
    __table_args__ = (
        CheckConstraint(
            "status IN ('staged', 'active', 'retired', 'failed')",
            name="status_valid",
        ),
    )

    generation_id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    generation_label: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="staged")
    qdrant_collection: Mapped[str] = mapped_column(String(255), nullable=False)
    generation_metadata: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utc_now,
    )


class IndexJob(Base):
    __tablename__ = "index_jobs"
    __table_args__ = (
        CheckConstraint(
            "operation IN ('index', 'deindex')",
            name="operation_valid",
        ),
        CheckConstraint(
            "status IN ('pending', 'running', 'succeeded', 'failed', 'cancelled')",
            name="status_valid",
        ),
    )

    job_id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    generation_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("serving_generations.generation_id", ondelete="SET NULL"),
    )
    record_version_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("canonical_records.record_version_id", ondelete="SET NULL"),
    )
    operation: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    job_scope: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    error_message: Mapped[str | None] = mapped_column(Text)
    scheduled_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utc_now,
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ContextualChunkCache(Base):
    __tablename__ = "contextual_chunk_cache"

    cache_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    record_version_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    chunk_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    chunk_text_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    prompt_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    context_text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utc_now)


class VerificationEvent(Base):
    __tablename__ = "verification_events"
    __table_args__ = (
        CheckConstraint(
            "verification_status IN ('pending', 'verified', 'rejected')",
            name="verification_status_valid",
        ),
    )

    event_id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    record_version_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("canonical_records.record_version_id", ondelete="SET NULL"),
    )
    source_section_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("source_sections.source_section_id", ondelete="SET NULL"),
    )
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    verification_status: Mapped[VerificationStatus] = mapped_column(String(16), nullable=False)
    reviewer: Mapped[str | None] = mapped_column(String(255))
    notes: Mapped[str | None] = mapped_column(Text)
    event_payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utc_now,
    )
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
