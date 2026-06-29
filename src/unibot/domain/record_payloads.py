from __future__ import annotations

from collections.abc import Mapping

from unibot.domain.record_renderers import (
    infer_record_type,
    render_search_sections,
    render_search_text,
)


def extract_primary_content(
    record_payload: Mapping[str, object],
    *,
    record_type: str | None = None,
) -> str:
    resolved_record_type = record_type or infer_record_type(record_payload) or "generic"
    return render_search_text(resolved_record_type, record_payload)


def extract_sections(
    record_payload: Mapping[str, object],
    *,
    record_type: str | None = None,
) -> list[tuple[str, str]]:
    resolved_record_type = record_type or infer_record_type(record_payload) or "generic"
    return render_search_sections(resolved_record_type, record_payload)
