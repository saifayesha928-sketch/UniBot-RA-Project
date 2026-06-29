from __future__ import annotations

from unibot.domain.types import AuthorityDecision, AuthorityRecord, SourceContext


def assign_authority_tier(source_context: SourceContext) -> int:
    if source_context.page_kind == "dedicated_child":
        if source_context.content_kind == "page_body":
            return 1

    if source_context.page_kind in {"dedicated_child", "dedicated_section"}:
        if source_context.content_kind in {"structured_section", "table"}:
            return 2

    if (
        source_context.page_kind == "official_document"
        or source_context.content_kind == "official_document"
    ):
        return 3

    if source_context.page_kind == "overview":
        return 4

    if source_context.page_kind == "navigation":
        return 5

    raise ValueError(
        "Unsupported source context for authority tier assignment: "
        f"{source_context.page_kind}/{source_context.content_kind}"
    )


def resolve_authority_conflict(
    records: list[AuthorityRecord] | tuple[AuthorityRecord, ...],
) -> AuthorityDecision:
    if not records:
        raise ValueError("records must not be empty")

    eligible_records = [record for record in records if record.is_current and record.is_verified]
    if not eligible_records:
        return AuthorityDecision(
            status="insufficient",
            primary_record=None,
            supporting_records=(),
            conflicting_records=(),
        )

    winning_tier = min(record.source_authority_tier for record in eligible_records)
    winning_records = [
        record
        for record in eligible_records
        if record.source_authority_tier == winning_tier
    ]

    if len({record.value_hash for record in winning_records}) > 1:
        return AuthorityDecision(
            status="contradictory",
            primary_record=None,
            supporting_records=(),
            conflicting_records=tuple(
                sorted(winning_records, key=lambda record: record.record_id)
            ),
        )

    primary_record = sorted(winning_records, key=lambda record: record.record_id)[0]
    supporting_records = tuple(
        record
        for record in eligible_records
        if record.record_id != primary_record.record_id
        and record.dedupe_key == primary_record.dedupe_key
    )

    return AuthorityDecision(
        status="authoritative",
        primary_record=primary_record,
        supporting_records=supporting_records,
        conflicting_records=(),
    )
