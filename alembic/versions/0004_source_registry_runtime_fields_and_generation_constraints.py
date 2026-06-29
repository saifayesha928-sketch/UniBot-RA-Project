"""add source registry runtime/discovery fields and generation constraints

Revision ID: 0004_source_registry_runtime
Revises: 0003_canonical_record_freshness
Create Date: 2026-03-12 04:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0004_source_registry_runtime"
down_revision = "0003_canonical_record_freshness"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "source_registry",
        sa.Column("parser_target", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "source_registry",
        sa.Column("parent_source_url", sa.Text(), nullable=True),
    )
    op.add_column(
        "source_registry",
        sa.Column("link_text", sa.Text(), nullable=True),
    )
    op.add_column(
        "source_registry",
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "source_registry",
        sa.Column("last_crawled_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "source_registry",
        sa.Column("disappeared_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_index(
        "ix_serving_generations_single_active",
        "serving_generations",
        ["status"],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
    )


def downgrade() -> None:
    op.drop_index("ix_serving_generations_single_active", table_name="serving_generations")
    op.drop_column("source_registry", "disappeared_at")
    op.drop_column("source_registry", "last_crawled_at")
    op.drop_column("source_registry", "last_seen_at")
    op.drop_column("source_registry", "link_text")
    op.drop_column("source_registry", "parent_source_url")
    op.drop_column("source_registry", "parser_target")
