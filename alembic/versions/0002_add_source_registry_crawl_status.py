"""add crawl status to source registry

Revision ID: 0002_source_registry_status
Revises: 0001_initial_schema
Create Date: 2026-03-11 20:30:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0002_source_registry_status"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "source_registry",
        sa.Column("crawl_status", sa.String(length=32), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("source_registry", "crawl_status")
