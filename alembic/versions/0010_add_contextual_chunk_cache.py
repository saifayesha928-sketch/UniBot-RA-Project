"""add contextual chunk cache table

Revision ID: 0010_contextual_chunk_cache
Revises: 0009_document_landing_type
Create Date: 2026-03-19 12:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "0010_contextual_chunk_cache"
down_revision = "0009_document_landing_type"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "contextual_chunk_cache",
        sa.Column("cache_key", sa.String(64), primary_key=True),
        sa.Column("record_version_id", sa.String(36), nullable=False),
        sa.Column("chunk_id", sa.String(255), nullable=False),
        sa.Column("chunk_text_hash", sa.String(64), nullable=False),
        sa.Column("model_name", sa.String(255), nullable=False),
        sa.Column("prompt_hash", sa.String(64), nullable=False),
        sa.Column("context_text", sa.Text, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        op.f("ix_contextual_chunk_cache_record_version_id"),
        "contextual_chunk_cache",
        ["record_version_id"],
    )
    op.create_index(
        op.f("ix_contextual_chunk_cache_chunk_id"),
        "contextual_chunk_cache",
        ["chunk_id"],
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_contextual_chunk_cache_chunk_id"),
        table_name="contextual_chunk_cache",
    )
    op.drop_index(
        op.f("ix_contextual_chunk_cache_record_version_id"),
        table_name="contextual_chunk_cache",
    )
    op.drop_table("contextual_chunk_cache")
