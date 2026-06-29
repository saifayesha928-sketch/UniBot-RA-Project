from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping

from sqlalchemy import select

from unibot.db.models import CanonicalRecord
from unibot.db.session import direct_session_scope
from unibot.domain.record_renderers import render_search_text

HIGH_VALUE_FIELDS: dict[str, tuple[str, ...]] = {
    "admissions_cycle": ("milestone_name", "date_text"),
    "program": ("program_name", "overview_text"),
    "document_landing": ("title", "linked_document_labels"),
    "research_entity": ("name", "subdomain_url"),
    "faculty_profile": ("name", "designation_text", "biography_text"),
}


def audit_rendering_rows(
    rows: Iterable[CanonicalRecord],
) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for row in rows:
        payload = row.record_payload or {}
        fields = HIGH_VALUE_FIELDS.get(row.record_type)
        if not fields:
            continue
        rendered = render_search_text(row.record_type, payload)
        missing_fields = [
            field for field in fields if not _field_is_rendered(payload, field, rendered)
        ]
        if missing_fields:
            findings.append(
                {
                    "record_version_id": row.record_version_id,
                    "record_type": row.record_type,
                    "missing_fields": missing_fields,
                    "rendered_text": rendered,
                }
            )
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="audit_index_rendering")
    parser.add_argument("--limit", type=int, default=25)
    args = parser.parse_args(argv)

    with direct_session_scope() as session:
        rows: list[CanonicalRecord] = []
        for record_type in sorted(HIGH_VALUE_FIELDS):
            rows.extend(
                session.execute(
                    select(CanonicalRecord)
                    .where(CanonicalRecord.record_type == record_type)
                    .order_by(CanonicalRecord.record_version_id.asc())
                    .limit(args.limit)
                ).scalars()
            )
        findings = audit_rendering_rows(rows)

    print(json.dumps({"finding_count": len(findings), "findings": findings}, indent=2))
    return 1 if findings else 0


def _field_is_rendered(
    payload: Mapping[str, object],
    field: str,
    rendered_text: str,
) -> bool:
    value = payload.get(field)
    fragments = _value_fragments(value)
    if not fragments:
        return True
    haystack = rendered_text.casefold()
    return all(fragment.casefold() in haystack for fragment in fragments)


def _value_fragments(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        stripped = value.strip()
        return (stripped,) if stripped else ()
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return tuple(parts)
    return ()


if __name__ == "__main__":
    raise SystemExit(main())
