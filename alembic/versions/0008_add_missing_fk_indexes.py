"""add missing foreign-key indexes

Revision ID: 0008_fk_indexes
Revises: 0007_review_event_resolved
Create Date: 2026-03-12 15:10:00.000000
"""

from __future__ import annotations

from alembic import op


revision = "0008_fk_indexes"
down_revision = "0007_review_event_resolved"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(op.f("ix_raw_snapshots_source_id"), "raw_snapshots", ["source_id"])
    op.create_index(op.f("ix_source_sections_source_id"), "source_sections", ["source_id"])
    op.create_index(op.f("ix_canonical_records_source_id"), "canonical_records", ["source_id"])


def downgrade() -> None:
    op.drop_index(op.f("ix_canonical_records_source_id"), table_name="canonical_records")
    op.drop_index(op.f("ix_source_sections_source_id"), table_name="source_sections")
    op.drop_index(op.f("ix_raw_snapshots_source_id"), table_name="raw_snapshots")
