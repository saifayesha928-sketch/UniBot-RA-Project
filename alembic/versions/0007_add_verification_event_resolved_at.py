"""add resolved_at to verification events

Revision ID: 0007_review_event_resolved
Revises: 0006_last_successful_crawl
Create Date: 2026-03-12 13:30:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0007_review_event_resolved"
down_revision = "0006_last_successful_crawl"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "verification_events",
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("verification_events", "resolved_at")
