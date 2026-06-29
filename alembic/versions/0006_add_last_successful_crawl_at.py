"""add last successful crawl timestamp to source registry

Revision ID: 0006_last_successful_crawl
Revises: 0005_pipeline_reliability
Create Date: 2026-03-12 12:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0006_last_successful_crawl"
down_revision = "0005_pipeline_reliability"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "source_registry",
        sa.Column("last_successful_crawl_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("source_registry", "last_successful_crawl_at")
