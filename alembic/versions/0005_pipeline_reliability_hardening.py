"""add canonical record authority and state guardrails

Revision ID: 0005_pipeline_reliability
Revises: 0004_source_registry_runtime
Create Date: 2026-03-12 06:30:00.000000
"""

from __future__ import annotations

from alembic import op


revision = "0005_pipeline_reliability"
down_revision = "0004_source_registry_runtime"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_check_constraint(
        op.f("ck_canonical_records_record_type_valid"),
        "canonical_records",
        (
            "record_type IN ("
            "'general', 'admissions_cycle', 'program', 'program_fee_schedule', "
            "'merit_list', 'faculty_profile', 'faculty_publication', "
            "'faculty_award', 'faculty_affiliation', 'research_entity', "
            "'scholarship', 'news_event', 'policy_rule', 'document_asset', "
            "'evidence', 'student_service', 'university_info', 'org_unit'"
            ")"
        ),
    )
    op.create_check_constraint(
        op.f("ck_canonical_records_source_authority_tier_valid"),
        "canonical_records",
        "source_authority_tier BETWEEN 1 AND 5",
    )
    op.create_check_constraint(
        op.f("ck_canonical_records_extraction_confidence_valid"),
        "canonical_records",
        (
            "extraction_confidence IN "
            "('high', 'medium', 'low', 'unknown', 'rule_based')"
        ),
    )


def downgrade() -> None:
    op.drop_constraint(
        op.f("ck_canonical_records_extraction_confidence_valid"),
        "canonical_records",
        type_="check",
    )
    op.drop_constraint(
        op.f("ck_canonical_records_source_authority_tier_valid"),
        "canonical_records",
        type_="check",
    )
    op.drop_constraint(
        op.f("ck_canonical_records_record_type_valid"),
        "canonical_records",
        type_="check",
    )
