"""add canonical record freshness fields

Revision ID: 0003_canonical_record_freshness
Revises: 0002_source_registry_status
Create Date: 2026-03-11 21:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0003_canonical_record_freshness"
down_revision = "0002_source_registry_status"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "canonical_records",
        sa.Column("source_last_modified_text", sa.Text(), nullable=True),
    )
    op.add_column(
        "canonical_records",
        sa.Column(
            "extraction_confidence",
            sa.String(length=16),
            nullable=False,
            server_default="rule_based",
        ),
    )


def downgrade() -> None:
    op.drop_column("canonical_records", "extraction_confidence")
    op.drop_column("canonical_records", "source_last_modified_text")
