"""add document_landing to record_type check constraint

Revision ID: 0009_document_landing_type
Revises: 0008_fk_indexes
Create Date: 2026-03-14 18:30:00.000000
"""

from __future__ import annotations

from alembic import op


revision = "0009_document_landing_type"
down_revision = "0008_fk_indexes"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint(
        op.f("ck_canonical_records_record_type_valid"),
        "canonical_records",
        type_="check",
    )
    op.create_check_constraint(
        op.f("ck_canonical_records_record_type_valid"),
        "canonical_records",
        (
            "record_type IN ("
            "'general', 'admissions_cycle', 'program', 'program_fee_schedule', "
            "'merit_list', 'faculty_profile', 'faculty_publication', "
            "'faculty_award', 'faculty_affiliation', 'research_entity', "
            "'scholarship', 'news_event', 'policy_rule', 'document_landing', "
            "'document_asset', 'evidence', 'student_service', 'university_info', "
            "'org_unit'"
            ")"
        ),
    )


def downgrade() -> None:
    op.drop_constraint(
        op.f("ck_canonical_records_record_type_valid"),
        "canonical_records",
        type_="check",
    )
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
