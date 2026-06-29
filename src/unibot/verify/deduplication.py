from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from unibot.verify.currentness import can_enter_serving
from unibot.verify.rules import (
    DedupeConflict,
    DedupeResult,
    SupportingEvidenceLink,
    VerificationDecision,
)
from unibot.verify.value_identity import effective_value_hash


def resolve_duplicates(records: Iterable[VerificationDecision]) -> DedupeResult:
    grouped_records: dict[str, list[VerificationDecision]] = defaultdict(list)
    for record in records:
        grouped_records[record.candidate.dedupe_key].append(record)

    primary_records: list[VerificationDecision] = []
    supporting_links: list[SupportingEvidenceLink] = []
    conflicts: list[DedupeConflict] = []

    for dedupe_key, grouped in grouped_records.items():
        serving_candidates = [record for record in grouped if can_enter_serving(record)]
        if not serving_candidates:
            continue

        winning_tier = min(
            record.candidate.source_authority_tier for record in serving_candidates
        )
        winning_records = [
            record
            for record in serving_candidates
            if record.candidate.source_authority_tier == winning_tier
        ]

        if len({
            effective_value_hash(
                record.candidate.record_type,
                record.candidate.record_payload,
                record.candidate.value_hash,
            )
            for record in winning_records
        }) > 1:
            conflicts.append(
                DedupeConflict(
                    dedupe_key=dedupe_key,
                    record_ids=tuple(
                        sorted(
                            record.candidate.record_version_id for record in winning_records
                        )
                    ),
                )
            )
            continue

        primary_record = min(
            winning_records,
            key=lambda record: (
                record.candidate.source_authority_tier,
                record.candidate.record_version_id,
            ),
        )
        primary_records.append(primary_record)

        primary_effective_hash = effective_value_hash(
            primary_record.candidate.record_type,
            primary_record.candidate.record_payload,
            primary_record.candidate.value_hash,
        )
        for record in serving_candidates:
            if record.candidate.record_version_id == primary_record.candidate.record_version_id:
                continue
            record_effective_hash = effective_value_hash(
                record.candidate.record_type,
                record.candidate.record_payload,
                record.candidate.value_hash,
            )
            if record_effective_hash != primary_effective_hash:
                continue
            supporting_links.append(
                SupportingEvidenceLink(
                    primary_record_version_id=primary_record.candidate.record_version_id,
                    supporting_record_version_id=record.candidate.record_version_id,
                )
            )

    return DedupeResult(
        primary_records=tuple(
            sorted(primary_records, key=lambda record: record.candidate.record_version_id)
        ),
        supporting_links=tuple(
            sorted(
                supporting_links,
                key=lambda link: (
                    link.primary_record_version_id,
                    link.supporting_record_version_id,
                ),
            )
        ),
        conflicts=tuple(sorted(conflicts, key=lambda conflict: conflict.dedupe_key)),
    )
