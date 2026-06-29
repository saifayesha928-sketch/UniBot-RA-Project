from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from urllib.parse import urlsplit


def render_search_text(record_type: str, record_payload: Mapping[str, object]) -> str:
    renderer = _SEARCH_RENDERERS.get(record_type, _render_generic_search_text)
    return renderer(record_payload).strip()


def render_identity_text(record_type: str, record_payload: Mapping[str, object]) -> str:
    renderer = _SEARCH_RENDERERS.get(record_type)
    if renderer is not None:
        return renderer(record_payload).strip()
    return render_search_text(record_type, record_payload)


def render_search_sections(
    record_type: str, record_payload: Mapping[str, object]
) -> list[tuple[str, str]]:
    """Return named sections for field-level chunking.

    Each element is ``(section_label, text)``.  Types without a dedicated
    section renderer fall back to a single section whose text matches
    ``render_search_text()``.
    """
    renderer = _SECTION_RENDERERS.get(record_type)
    if renderer is None:
        return [("", render_search_text(record_type, record_payload))]
    sections = renderer(record_payload)
    return [(label, text.strip()) for label, text in sections if text.strip()]


def render_chunk_prefix(
    record_type: str, record_payload: Mapping[str, object]
) -> str:
    """Return an entity-specific prefix string for a chunk.

    Examples:
        ``"Program: BS Computer Science — BS — Faculty of Computing"``
        ``"Faculty: Dr. Ahmed Khan — Associate Professor — CS"``
    """
    renderer = _PREFIX_RENDERERS.get(record_type)
    if renderer is not None:
        result = renderer(record_payload).strip()
        if result:
            return result
    # Fallback preserves source_locator parity with current production prefixes.
    return f"Source: {record_type}"


def infer_record_type(record_payload: Mapping[str, object]) -> str | None:
    if "milestone_name" in record_payload and "date_text" in record_payload:
        return "admissions_cycle"
    if "curriculum_table_text" in record_payload:
        return "program_curriculum"
    if "program_name" in record_payload and (
        "overview_text" in record_payload or "degree_level" in record_payload
    ):
        return "program"
    if "linked_document_labels" in record_payload or "linked_document_urls" in record_payload:
        return "document_landing"
    if "subdomain_url" in record_payload or "main_site_url" in record_payload:
        return "research_entity"
    if "rule_text" in record_payload or "rule_title" in record_payload:
        return "policy_rule"
    if "scholarship_name" in record_payload or "benefit_text" in record_payload:
        return "scholarship"
    if "raw_table_text" in record_payload or "fee_table_markdown" in record_payload:
        return "program_fee_schedule"
    if "designation_text" in record_payload or "biography_text" in record_payload:
        return "faculty_profile"
    if "raw_citation" in record_payload:
        return "faculty_publication"
    if "document_url" in record_payload or "filename" in record_payload:
        return "document_asset"
    return None


def _render_admissions_cycle(payload: Mapping[str, object]) -> str:
    rendered = _join_sections(
        (
            "Admissions Milestone",
            _join_inline(
                _scalar(payload.get("milestone_name")),
                _scalar(payload.get("date_text")),
                _scalar(payload.get("calendar_scope")),
            ),
            _list_section("Normalized Dates", payload.get("normalized_dates")),
            _scalar(payload.get("source_last_modified_text")),
        )
    )
    return rendered if rendered.strip() != "Admissions Milestone" else _render_generic_search_text(payload)


def _render_program(payload: Mapping[str, object]) -> str:
    rendered = _join_sections(
        (
            _join_inline(
                _scalar(payload.get("program_name")),
                _scalar(payload.get("degree_level")),
            ),
            _scalar(payload.get("overview_text")),
            _scalar(payload.get("eligibility_summary")),
            _list_section("Test Options", payload.get("test_options")),
            _scalar(payload.get("faculty_label")),
            _scalar(payload.get("department_label")),
            _scalar(payload.get("credit_hours")),
            _scalar(payload.get("program_structure_text")),
            _list_section("Curriculum Documents", payload.get("curriculum_document_urls")),
        )
    )
    return rendered or _render_generic_search_text(payload)


def _render_program_fee_schedule(payload: Mapping[str, object]) -> str:
    rendered = _join_sections(
        (
            _join_inline(
                _scalar(payload.get("program_name")),
                _scalar(payload.get("table_kind")),
                _scalar(payload.get("audience")),
                _scalar(payload.get("currency")),
                _scalar(payload.get("cycle_label")),
            ),
            _scalar(payload.get("total_degree_fee")),
            _scalar(payload.get("repeat_fee_per_credit_hour")),
            _scalar(payload.get("annual_cost_per_student")),
            _scalar(payload.get("raw_table_text")),
            _scalar(payload.get("fee_table_markdown")),
            _list_section_lines("Rows", payload.get("rows")),
            _list_section_lines("Warnings", payload.get("parse_warnings")),
        )
    )
    return rendered or _render_generic_search_text(payload)


def _render_faculty_profile(payload: Mapping[str, object]) -> str:
    rendered = _join_sections(
        (
            _join_inline(
                _scalar(payload.get("name")),
                _scalar(payload.get("designation_text")),
                _scalar(payload.get("department_label")),
                _scalar(payload.get("faculty_label")),
            ),
            _scalar(payload.get("qualification_text")),
            _scalar(payload.get("role_text")),
            _list_section("Research Interests", payload.get("research_interests")),
            _scalar(payload.get("biography_text")),
            _scalar(payload.get("email")),
        )
    )
    return rendered or _render_generic_search_text(payload)


def _render_faculty_publication(payload: Mapping[str, object]) -> str:
    rendered = _join_sections(
        (
            _join_inline(
                _scalar(payload.get("parent_name")),
                _scalar(payload.get("publication_type")),
                _scalar(payload.get("year")),
            ),
            _scalar(payload.get("title")),
            _list_section("Authors", payload.get("authors")),
            _scalar(payload.get("venue")),
            _scalar(payload.get("raw_citation")),
            _scalar(payload.get("value_text")),
        )
    )
    return rendered or _render_generic_search_text(payload)


def _render_faculty_child(payload: Mapping[str, object]) -> str:
    rendered = _join_sections(
        (
            _scalar(payload.get("parent_name")),
            _scalar(payload.get("title")),
            _scalar(payload.get("organization")),
            _scalar(payload.get("role")),
            _scalar(payload.get("content")),
            _scalar(payload.get("value_text")),
        )
    )
    return rendered or _render_generic_search_text(payload)


def _render_document_landing(payload: Mapping[str, object]) -> str:
    rendered = _join_sections(
        (
            _scalar(payload.get("title")),
            _scalar(payload.get("summary_text")),
            _list_section("Document Labels", payload.get("linked_document_labels")),
            _list_section("Document URLs", payload.get("linked_document_urls")),
            _list_section("Sections", payload.get("section_labels")),
        )
    )
    return rendered or _render_generic_search_text(payload)


def _render_document_asset(payload: Mapping[str, object]) -> str:
    rendered = _join_sections(
        (
            _scalar(payload.get("document_title")),
            _scalar(payload.get("filename")),
            _scalar(payload.get("document_kind")),
            _scalar(payload.get("media_type")),
            _scalar(payload.get("document_url")),
            _scalar(payload.get("parent_page_url")),
        )
    )
    return rendered or _render_generic_search_text(payload)


def _render_research_entity(payload: Mapping[str, object]) -> str:
    subdomain = _scalar(payload.get("subdomain_url"))
    host = urlsplit(subdomain).netloc if subdomain else ""
    rendered = _join_sections(
        (
            _scalar(payload.get("name")),
            subdomain,
            host,
            _scalar(payload.get("main_site_url")),
            _scalar(payload.get("crawl_status")),
        )
    )
    return rendered or _render_generic_search_text(payload)


def _render_merit_list(payload: Mapping[str, object]) -> str:
    rendered = _join_sections(
        (
            _scalar(payload.get("title")),
            _scalar(payload.get("program_name")),
            _scalar(payload.get("merit_stage")),
            _scalar(payload.get("cycle_label")),
            _scalar(payload.get("detail_url")),
            _scalar(payload.get("content")),
        )
    )
    return rendered or _render_generic_search_text(payload)


def _render_policy_rule(payload: Mapping[str, object]) -> str:
    rendered = _join_sections(
        (
            _join_inline(
                _scalar(payload.get("policy_area")),
                _scalar(payload.get("rule_title")),
            ),
            _scalar(payload.get("rule_text")),
        )
    )
    return rendered or _render_generic_search_text(payload)


def _render_scholarship(payload: Mapping[str, object]) -> str:
    rendered = _join_sections(
        (
            _join_inline(
                _scalar(payload.get("scholarship_name")),
                _scalar(payload.get("category_label")),
            ),
            _scalar(payload.get("content")),
            _scalar(payload.get("benefit_text")),
            _scalar(payload.get("eligibility_text")),
            _scalar(payload.get("duration_text")),
            _list_section("Columns", payload.get("columns")),
        )
    )
    return rendered or _render_generic_search_text(payload)


def _render_title_content(payload: Mapping[str, object]) -> str:
    rendered = _join_sections(
        (
            _scalar(payload.get("title")),
            _scalar(payload.get("faq_question")),
            _scalar(payload.get("faq_answer")),
            _scalar(payload.get("content")),
        )
    )
    return rendered or _render_generic_search_text(payload)


def _render_org_unit(payload: Mapping[str, object]) -> str:
    variant = _scalar(payload.get("section_variant"))
    identity_parts = [
        _scalar(payload.get("name")),
        _scalar(payload.get("unit_type")),
    ]
    if variant and variant != "overview":
        identity_parts.append(f"[{variant}]")
    rendered = _join_sections(
        (
            _join_inline(*identity_parts),
            _scalar(payload.get("content")),
            _list_section("Staff Contacts", payload.get("staff_contacts")),
        )
    )
    return rendered or _render_generic_search_text(payload)


def _render_generic_search_text(payload: Mapping[str, object]) -> str:
    for key in (
        "content",
        "raw_row_text",
        "raw_citation",
        "raw_table_text",
        "fee_table_markdown",
        "overview_text",
        "value_text",
        "biography_text",
        "rule_text",
        "benefit_text",
        "summary_text",
        "body_text",
        "eligibility_text",
        "qualification_text",
        "milestone_text",
    ):
        value = _scalar(payload.get(key))
        if value:
            return value
    header = _join_inline(
        _scalar(payload.get("name")),
        _scalar(payload.get("designation_text")),
        _scalar(payload.get("department_label")),
        _scalar(payload.get("faculty_label")),
    )
    return header


def _render_program_curriculum(payload: Mapping[str, object]) -> str:
    return _join_sections([
        _join_inline(
            _scalar(payload.get("program_name")),
            _scalar(payload.get("degree_level")),
            "Curriculum",
        ),
        _scalar(payload.get("section_label")),
        _scalar(payload.get("curriculum_table_text")),
    ])


# ---------------------------------------------------------------------------
# Section renderers — return list[(label, text)] for field-level chunking
# ---------------------------------------------------------------------------

def _sections_program(payload: Mapping[str, object]) -> list[tuple[str, str]]:
    identity = _join_inline(
        _scalar(payload.get("program_name")),
        _scalar(payload.get("degree_level")),
    )
    overview_parts = [
        identity,
        _scalar(payload.get("overview_text")),
        _scalar(payload.get("faculty_label")),
        _scalar(payload.get("department_label")),
        _scalar(payload.get("credit_hours")),
    ]
    sections: list[tuple[str, str]] = []
    overview = _join_sections(p for p in overview_parts if p)
    if overview:
        sections.append(("Overview", overview))
    eligibility = _scalar(payload.get("eligibility_summary"))
    test_options = _list_section("Test Options", payload.get("test_options"))
    elig_parts = [p for p in (eligibility, test_options) if p]
    if elig_parts:
        sections.append(("Eligibility", _join_sections(elig_parts)))
    structure_text = _scalar(payload.get("program_structure_text"))
    if structure_text:
        sections.append(("Structure", structure_text))
    curriculum_docs = _list_section(
        "Curriculum Documents", payload.get("curriculum_document_urls")
    )
    if curriculum_docs:
        sections.append(("Curriculum", curriculum_docs))
    return sections if sections else [("", _render_generic_search_text(payload))]


def _sections_faculty_profile(payload: Mapping[str, object]) -> list[tuple[str, str]]:
    identity = _join_inline(
        _scalar(payload.get("name")),
        _scalar(payload.get("designation_text")),
        _scalar(payload.get("department_label")),
        _scalar(payload.get("faculty_label")),
    )
    profile_parts = [
        identity,
        _scalar(payload.get("qualification_text")),
        _scalar(payload.get("role_text")),
        _scalar(payload.get("email")),
    ]
    sections: list[tuple[str, str]] = []
    profile = _join_sections(p for p in profile_parts if p)
    if profile:
        sections.append(("Profile", profile))
    biography = _scalar(payload.get("biography_text"))
    if biography:
        sections.append(("Biography", biography))
    research = _list_section("Research Interests", payload.get("research_interests"))
    if research:
        sections.append(("Research", research))
    return sections if sections else [("", _render_generic_search_text(payload))]


def _sections_org_unit(payload: Mapping[str, object]) -> list[tuple[str, str]]:
    variant = _scalar(payload.get("section_variant"))
    identity = _join_inline(
        _scalar(payload.get("name")),
        _scalar(payload.get("unit_type")),
    )
    about_parts = [identity, _scalar(payload.get("content"))]
    sections: list[tuple[str, str]] = []
    # Use variant-aware section labels for better retrieval context
    section_label = "Directory" if variant == "directory" else "About"
    about = _join_sections(p for p in about_parts if p)
    if about:
        sections.append((section_label, about))
    staff = _list_section("Staff Contacts", payload.get("staff_contacts"))
    if staff:
        sections.append(("Staff", staff))
    return sections if sections else [("", _render_generic_search_text(payload))]


def _sections_title_content_faq(payload: Mapping[str, object]) -> list[tuple[str, str]]:
    """Section renderer for university_info and student_service."""
    header_parts = [
        _scalar(payload.get("title")),
        _scalar(payload.get("faq_question")),
    ]
    header = _join_sections(p for p in header_parts if p)
    detail_parts = [
        _scalar(payload.get("faq_answer")),
        _scalar(payload.get("content")),
    ]
    detail = _join_sections(p for p in detail_parts if p)
    sections: list[tuple[str, str]] = []
    if header:
        sections.append(("", header))
    if detail:
        sections.append(("Details", detail))
    return sections if sections else [("", _render_generic_search_text(payload))]


def _sections_program_curriculum(payload: Mapping[str, object]) -> list[tuple[str, str]]:
    """Section renderer for program_curriculum records.

    Curriculum tables are the highest-value retrieval gap (PD01/PD02/PD04/CO05).
    Keep identity context (program name + degree + section label) attached to the
    table so embedding captures both the program identity and the course data.
    """
    identity = _join_inline(
        _scalar(payload.get("program_name")),
        _scalar(payload.get("degree_level")),
    )
    section_label = _scalar(payload.get("section_label"))
    table_text = _scalar(payload.get("curriculum_table_text"))

    header_parts = [p for p in (identity, section_label) if p]
    header = _join_sections(header_parts) if header_parts else ""

    sections: list[tuple[str, str]] = []
    if table_text:
        curriculum_content = f"{header}\n\n{table_text}" if header else table_text
        sections.append(("Curriculum", curriculum_content))
    elif header:
        sections.append(("Curriculum", header))

    return sections if sections else [("", _render_generic_search_text(payload))]


_SECTION_RENDERERS: dict[
    str,
    Callable[[Mapping[str, object]], list[tuple[str, str]]],
] = {
    "program": _sections_program,
    "program_curriculum": _sections_program_curriculum,
    "faculty_profile": _sections_faculty_profile,
    "org_unit": _sections_org_unit,
    "university_info": _sections_title_content_faq,
    "student_service": _sections_title_content_faq,
    "news_event": _sections_title_content_faq,
}


# ---------------------------------------------------------------------------
# Prefix renderers — return entity-specific one-line identity string
# ---------------------------------------------------------------------------

def _prefix_program(payload: Mapping[str, object]) -> str:
    identity = _join_inline(
        _scalar(payload.get("program_name")),
        _scalar(payload.get("degree_level")),
        _scalar(payload.get("faculty_label")),
    )
    return f"Program: {identity}" if identity else ""


def _prefix_faculty_profile(payload: Mapping[str, object]) -> str:
    identity = _join_inline(
        _scalar(payload.get("name")),
        _scalar(payload.get("designation_text")),
        _scalar(payload.get("department_label")),
    )
    return f"Faculty: {identity}" if identity else ""


def _prefix_faculty_publication(payload: Mapping[str, object]) -> str:
    identity = _join_inline(
        _scalar(payload.get("parent_name")),
        _scalar(payload.get("publication_type")),
        _scalar(payload.get("year")),
    )
    return f"Publication: {identity}" if identity else ""


def _prefix_admissions_cycle(payload: Mapping[str, object]) -> str:
    identity = _join_inline(
        _scalar(payload.get("milestone_name")),
        _scalar(payload.get("calendar_scope")),
    )
    return f"Admissions: {identity}" if identity else ""


def _prefix_program_curriculum(payload: Mapping[str, object]) -> str:
    identity = _join_inline(
        _scalar(payload.get("program_name")),
        _scalar(payload.get("degree_level")),
        _scalar(payload.get("section_label")),
    )
    return f"Curriculum: {identity}" if identity else ""


def _prefix_fee_schedule(payload: Mapping[str, object]) -> str:
    identity = _join_inline(
        _scalar(payload.get("program_name")),
        _scalar(payload.get("table_kind")),
        _scalar(payload.get("cycle_label")),
        _scalar(payload.get("total_degree_fee")),
        _scalar(payload.get("repeat_fee_per_credit_hour")),
    )
    return f"Fee Schedule: {identity}" if identity else ""


def _prefix_policy_rule(payload: Mapping[str, object]) -> str:
    identity = _join_inline(
        _scalar(payload.get("policy_area")),
        _scalar(payload.get("rule_title")),
    )
    return f"Policy: {identity}" if identity else ""


def _prefix_scholarship(payload: Mapping[str, object]) -> str:
    identity = _join_inline(
        _scalar(payload.get("scholarship_name")),
        _scalar(payload.get("category_label")),
    )
    return f"Scholarship: {identity}" if identity else ""


def _prefix_org_unit(payload: Mapping[str, object]) -> str:
    variant = _scalar(payload.get("section_variant"))
    identity = _join_inline(
        _scalar(payload.get("name")),
        _scalar(payload.get("unit_type")),
    )
    if not identity:
        return ""
    if variant and variant != "overview":
        return f"Unit: {identity} ({variant})"
    return f"Unit: {identity}"


def _prefix_merit_list(payload: Mapping[str, object]) -> str:
    identity = _join_inline(
        _scalar(payload.get("program_name")),
        _scalar(payload.get("merit_stage")),
        _scalar(payload.get("cycle_label")),
    )
    return f"Merit List: {identity}" if identity else ""


def _prefix_university_info(payload: Mapping[str, object]) -> str:
    identity = _scalar(payload.get("title"))
    return f"Info: {identity}" if identity else ""


def _prefix_student_service(payload: Mapping[str, object]) -> str:
    identity = _scalar(payload.get("title"))
    return f"Service: {identity}" if identity else ""


def _prefix_faculty_award(payload: Mapping[str, object]) -> str:
    identity = _join_inline(
        _scalar(payload.get("parent_name")),
        _scalar(payload.get("title")),
    )
    return f"Award: {identity}" if identity else ""


def _prefix_faculty_affiliation(payload: Mapping[str, object]) -> str:
    identity = _join_inline(
        _scalar(payload.get("parent_name")),
        _scalar(payload.get("organization")),
    )
    return f"Affiliation: {identity}" if identity else ""


def _prefix_news_event(payload: Mapping[str, object]) -> str:
    identity = _scalar(payload.get("title"))
    return f"News: {identity}" if identity else ""


_PREFIX_RENDERERS: dict[str, Callable[[Mapping[str, object]], str]] = {
    "program": _prefix_program,
    "program_curriculum": _prefix_program_curriculum,
    "faculty_profile": _prefix_faculty_profile,
    "faculty_publication": _prefix_faculty_publication,
    "faculty_award": _prefix_faculty_award,
    "faculty_affiliation": _prefix_faculty_affiliation,
    "admissions_cycle": _prefix_admissions_cycle,
    "program_fee_schedule": _prefix_fee_schedule,
    "policy_rule": _prefix_policy_rule,
    "scholarship": _prefix_scholarship,
    "org_unit": _prefix_org_unit,
    "merit_list": _prefix_merit_list,
    "university_info": _prefix_university_info,
    "student_service": _prefix_student_service,
    "news_event": _prefix_news_event,
}


_SEARCH_RENDERERS = {
    "admissions_cycle": _render_admissions_cycle,
    "program": _render_program,
    "program_curriculum": _render_program_curriculum,
    "program_fee_schedule": _render_program_fee_schedule,
    "faculty_profile": _render_faculty_profile,
    "faculty_publication": _render_faculty_publication,
    "faculty_award": _render_faculty_child,
    "faculty_affiliation": _render_faculty_child,
    "document_landing": _render_document_landing,
    "document_asset": _render_document_asset,
    "research_entity": _render_research_entity,
    "merit_list": _render_merit_list,
    "policy_rule": _render_policy_rule,
    "scholarship": _render_scholarship,
    "university_info": _render_title_content,
    "student_service": _render_title_content,
    "news_event": _render_title_content,
    "org_unit": _render_org_unit,
    "generic": _render_generic_search_text,
}

def _scalar(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    return ""


def _join_inline(*parts: str) -> str:
    return " — ".join(part for part in parts if part)


def _join_sections(parts: Iterable[str]) -> str:
    return "\n\n".join(part for part in parts if part)


def _list_section(label: str, value: object) -> str:
    items = _flatten_items(value)
    if not items:
        return ""
    return f"{label}: " + "; ".join(items)


def _list_section_lines(label: str, value: object) -> str:
    """Render a list as individual lines for structure-preserving chunking."""
    items = _flatten_items(value)
    if not items:
        return ""
    return label + ":\n" + "\n".join(f"- {item}" for item in items)


def _flatten_items(value: object) -> list[str]:
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, Mapping):
        parts = [_join_inline(*(_scalar(v) for v in value.values()))]
        return [part for part in parts if part]
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        items: list[str] = []
        for item in value:
            if isinstance(item, Mapping):
                joined = _join_inline(*(_scalar(v) for v in item.values()))
                if joined:
                    items.append(joined)
                continue
            scalar = _scalar(item)
            if scalar:
                items.append(scalar)
        return items
    scalar = _scalar(value)
    return [scalar] if scalar else []
