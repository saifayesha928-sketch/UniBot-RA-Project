"""initial canonical schema

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-03-11 18:30:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "source_registry",
        sa.Column("source_id", sa.String(length=36), primary_key=True),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("canonical_url", sa.Text(), nullable=False),
        sa.Column("source_class", sa.String(length=128), nullable=False),
        sa.Column("crawl_method", sa.String(length=64), nullable=False),
        sa.Column("legal_status", sa.String(length=32), nullable=False),
        sa.Column("default_authority_tier", sa.SmallInteger(), nullable=False),
        sa.Column("refresh_policy", sa.String(length=64), nullable=False),
        sa.Column(
            "is_active",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.UniqueConstraint("source_url", name=op.f("uq_source_registry_source_url")),
    )

    op.create_table(
        "raw_snapshots",
        sa.Column("snapshot_id", sa.String(length=36), primary_key=True),
        sa.Column("source_id", sa.String(length=36), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("content_type", sa.String(length=255), nullable=False),
        sa.Column("storage_uri", sa.Text(), nullable=False),
        sa.Column("storage_backend", sa.String(length=32), nullable=False),
        sa.Column("page_content_hash", sa.String(length=128), nullable=False),
        sa.Column("http_status", sa.Integer()),
        sa.Column("etag", sa.String(length=255)),
        sa.Column("last_modified", sa.String(length=255)),
        sa.Column("fetch_metadata", sa.JSON(), nullable=False, server_default=sa.text("'{}'::json")),
        sa.Column(
            "fetched_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.ForeignKeyConstraint(
            ["source_id"],
            ["source_registry.source_id"],
            ondelete="CASCADE",
            name=op.f("fk_raw_snapshots_source_id_source_registry"),
        ),
    )

    op.create_table(
        "source_sections",
        sa.Column("source_section_id", sa.String(length=36), primary_key=True),
        sa.Column("snapshot_id", sa.String(length=36), nullable=False),
        sa.Column("source_id", sa.String(length=36), nullable=False),
        sa.Column("section_label", sa.String(length=255), nullable=False),
        sa.Column("section_type", sa.String(length=64), nullable=False),
        sa.Column("source_locator", sa.Text(), nullable=False),
        sa.Column("section_order", sa.Integer(), nullable=False),
        sa.Column("text_content", sa.Text(), nullable=False),
        sa.Column("source_text_hash", sa.String(length=128), nullable=False),
        sa.Column("parser_backend", sa.String(length=64)),
        sa.Column("page_number", sa.Integer()),
        sa.Column("grounding_data", sa.JSON(), nullable=False, server_default=sa.text("'{}'::json")),
        sa.Column(
            "extracted_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.ForeignKeyConstraint(
            ["snapshot_id"],
            ["raw_snapshots.snapshot_id"],
            ondelete="CASCADE",
            name=op.f("fk_source_sections_snapshot_id_raw_snapshots"),
        ),
        sa.ForeignKeyConstraint(
            ["source_id"],
            ["source_registry.source_id"],
            ondelete="CASCADE",
            name=op.f("fk_source_sections_source_id_source_registry"),
        ),
        sa.UniqueConstraint(
            "snapshot_id",
            "section_order",
            name=op.f("uq_source_sections_snapshot_id"),
        ),
    )

    op.create_table(
        "canonical_records",
        sa.Column("record_version_id", sa.String(length=36), primary_key=True),
        sa.Column("record_id", sa.String(length=255), nullable=False),
        sa.Column("record_type", sa.String(length=128), nullable=False),
        sa.Column("source_id", sa.String(length=36)),
        sa.Column("source_section_id", sa.String(length=36)),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("canonical_url", sa.Text(), nullable=False),
        sa.Column("source_title", sa.String(length=512)),
        sa.Column("source_section_label", sa.String(length=255)),
        sa.Column("source_locator", sa.Text(), nullable=False),
        sa.Column("source_text_hash", sa.String(length=128), nullable=False),
        sa.Column("page_content_hash", sa.String(length=128), nullable=False),
        sa.Column("source_authority_tier", sa.SmallInteger(), nullable=False),
        sa.Column("conflict_scope_id", sa.String(length=255), nullable=False),
        sa.Column("dedupe_key", sa.String(length=255), nullable=False),
        sa.Column("record_payload", sa.JSON(), nullable=False, server_default=sa.text("'{}'::json")),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("verified_at", sa.DateTime(timezone=True)),
        sa.Column("effective_from", sa.Date()),
        sa.Column("effective_to", sa.Date()),
        sa.Column("cycle_label", sa.String(length=255)),
        sa.Column("year_confidence", sa.String(length=16), nullable=False, server_default="unknown"),
        sa.Column("freshness_status", sa.String(length=32), nullable=False, server_default="unknown"),
        sa.Column("verification_status", sa.String(length=32), nullable=False, server_default="pending"),
        sa.Column("serving_status", sa.String(length=32), nullable=False, server_default="ineligible"),
        sa.Column("supersedes_version_id", sa.String(length=36)),
        sa.Column("superseded_by_version_id", sa.String(length=36)),
        sa.Column(
            "is_current_candidate",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "is_current_authoritative",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.CheckConstraint(
            "year_confidence IN ('high', 'medium', 'low', 'unknown')",
            name=op.f("ck_canonical_records_year_confidence_valid"),
        ),
        sa.CheckConstraint(
            "freshness_status IN ('current', 'stale', 'unknown', 'contradictory', 'restricted', 'removed')",
            name=op.f("ck_canonical_records_freshness_status_valid"),
        ),
        sa.CheckConstraint(
            "verification_status IN ('pending', 'verified', 'rejected')",
            name=op.f("ck_canonical_records_verification_status_valid"),
        ),
        sa.CheckConstraint(
            "serving_status IN ('eligible', 'ineligible', 'pending_index', 'pending_deindex', 'indexed_active', 'deindexed', 'failed')",
            name=op.f("ck_canonical_records_serving_status_valid"),
        ),
        sa.CheckConstraint(
            "(is_current_authoritative = false) OR "
            "(freshness_status = 'current' AND verification_status = 'verified')",
            name=op.f("ck_canonical_records_current_authoritative_requires_verified_current"),
        ),
        sa.ForeignKeyConstraint(
            ["source_id"],
            ["source_registry.source_id"],
            ondelete="SET NULL",
            name=op.f("fk_canonical_records_source_id_source_registry"),
        ),
        sa.ForeignKeyConstraint(
            ["source_section_id"],
            ["source_sections.source_section_id"],
            ondelete="SET NULL",
            name=op.f("fk_canonical_records_source_section_id_source_sections"),
        ),
        sa.ForeignKeyConstraint(
            ["supersedes_version_id"],
            ["canonical_records.record_version_id"],
            ondelete="SET NULL",
            name=op.f("fk_canonical_records_supersedes_version_id_canonical_records"),
        ),
        sa.ForeignKeyConstraint(
            ["superseded_by_version_id"],
            ["canonical_records.record_version_id"],
            ondelete="SET NULL",
            name=op.f("fk_canonical_records_superseded_by_version_id_canonical_records"),
        ),
    )
    op.create_index(op.f("ix_canonical_records_record_id"), "canonical_records", ["record_id"])
    op.create_index(op.f("ix_canonical_records_record_type"), "canonical_records", ["record_type"])
    op.create_index(op.f("ix_canonical_records_conflict_scope_id"), "canonical_records", ["conflict_scope_id"])
    op.create_index(op.f("ix_canonical_records_dedupe_key"), "canonical_records", ["dedupe_key"])
    op.create_index(op.f("ix_canonical_records_freshness_status"), "canonical_records", ["freshness_status"])
    op.create_index(op.f("ix_canonical_records_verification_status"), "canonical_records", ["verification_status"])
    op.create_index(op.f("ix_canonical_records_serving_status"), "canonical_records", ["serving_status"])
    op.create_index(
        op.f("ix_canonical_records_is_current_authoritative"),
        "canonical_records",
        ["is_current_authoritative"],
    )

    op.create_table(
        "supporting_evidence_links",
        sa.Column("evidence_link_id", sa.String(length=36), primary_key=True),
        sa.Column("primary_record_version_id", sa.String(length=36), nullable=False),
        sa.Column("supporting_record_version_id", sa.String(length=36), nullable=False),
        sa.Column("relation_type", sa.String(length=32), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.ForeignKeyConstraint(
            ["primary_record_version_id"],
            ["canonical_records.record_version_id"],
            ondelete="CASCADE",
            name=op.f("fk_supporting_evidence_links_primary_record_version_id_canonical_records"),
        ),
        sa.ForeignKeyConstraint(
            ["supporting_record_version_id"],
            ["canonical_records.record_version_id"],
            ondelete="CASCADE",
            name=op.f("fk_supporting_evidence_links_supporting_record_version_id_canonical_records"),
        ),
        sa.UniqueConstraint(
            "primary_record_version_id",
            "supporting_record_version_id",
            name=op.f("uq_supporting_evidence_links_primary_record_version_id"),
        ),
    )

    op.create_table(
        "serving_generations",
        sa.Column("generation_id", sa.String(length=36), primary_key=True),
        sa.Column("generation_label", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="staged"),
        sa.Column("qdrant_collection", sa.String(length=255), nullable=False),
        sa.Column("generation_metadata", sa.JSON(), nullable=False, server_default=sa.text("'{}'::json")),
        sa.Column("activated_at", sa.DateTime(timezone=True)),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.CheckConstraint(
            "status IN ('staged', 'active', 'retired', 'failed')",
            name=op.f("ck_serving_generations_status_valid"),
        ),
        sa.UniqueConstraint(
            "generation_label",
            name=op.f("uq_serving_generations_generation_label"),
        ),
    )

    op.create_table(
        "index_jobs",
        sa.Column("job_id", sa.String(length=36), primary_key=True),
        sa.Column("generation_id", sa.String(length=36)),
        sa.Column("record_version_id", sa.String(length=36)),
        sa.Column("operation", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="pending"),
        sa.Column("job_scope", sa.JSON(), nullable=False, server_default=sa.text("'{}'::json")),
        sa.Column("error_message", sa.Text()),
        sa.Column(
            "scheduled_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint(
            "operation IN ('index', 'deindex')",
            name=op.f("ck_index_jobs_operation_valid"),
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'succeeded', 'failed', 'cancelled')",
            name=op.f("ck_index_jobs_status_valid"),
        ),
        sa.ForeignKeyConstraint(
            ["generation_id"],
            ["serving_generations.generation_id"],
            ondelete="SET NULL",
            name=op.f("fk_index_jobs_generation_id_serving_generations"),
        ),
        sa.ForeignKeyConstraint(
            ["record_version_id"],
            ["canonical_records.record_version_id"],
            ondelete="SET NULL",
            name=op.f("fk_index_jobs_record_version_id_canonical_records"),
        ),
    )

    op.create_table(
        "verification_events",
        sa.Column("event_id", sa.String(length=36), primary_key=True),
        sa.Column("record_version_id", sa.String(length=36)),
        sa.Column("source_section_id", sa.String(length=36)),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("verification_status", sa.String(length=16), nullable=False),
        sa.Column("reviewer", sa.String(length=255)),
        sa.Column("notes", sa.Text()),
        sa.Column("event_payload", sa.JSON(), nullable=False, server_default=sa.text("'{}'::json")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.CheckConstraint(
            "verification_status IN ('pending', 'verified', 'rejected')",
            name=op.f("ck_verification_events_verification_status_valid"),
        ),
        sa.ForeignKeyConstraint(
            ["record_version_id"],
            ["canonical_records.record_version_id"],
            ondelete="SET NULL",
            name=op.f("fk_verification_events_record_version_id_canonical_records"),
        ),
        sa.ForeignKeyConstraint(
            ["source_section_id"],
            ["source_sections.source_section_id"],
            ondelete="SET NULL",
            name=op.f("fk_verification_events_source_section_id_source_sections"),
        ),
    )


def downgrade() -> None:
    op.drop_table("verification_events")
    op.drop_table("index_jobs")
    op.drop_table("serving_generations")
    op.drop_table("supporting_evidence_links")
    op.drop_index(op.f("ix_canonical_records_is_current_authoritative"), table_name="canonical_records")
    op.drop_index(op.f("ix_canonical_records_serving_status"), table_name="canonical_records")
    op.drop_index(op.f("ix_canonical_records_verification_status"), table_name="canonical_records")
    op.drop_index(op.f("ix_canonical_records_freshness_status"), table_name="canonical_records")
    op.drop_index(op.f("ix_canonical_records_dedupe_key"), table_name="canonical_records")
    op.drop_index(op.f("ix_canonical_records_conflict_scope_id"), table_name="canonical_records")
    op.drop_index(op.f("ix_canonical_records_record_type"), table_name="canonical_records")
    op.drop_index(op.f("ix_canonical_records_record_id"), table_name="canonical_records")
    op.drop_table("canonical_records")
    op.drop_table("source_sections")
    op.drop_table("raw_snapshots")
    op.drop_table("source_registry")
