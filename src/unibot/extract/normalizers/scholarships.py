from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from unibot.extract.text import slugify

_DASH_VARIANTS = re.compile(r"[\u2013\u2014\u2015\u2212]")
_TRAILING_PUNCTUATION = re.compile(r"[\s*.,;:!?]+$")
_WHITESPACE = re.compile(r"\s+")

# Patterns for extracting structured info from single-column scholarship rows
_BENEFIT_KEYWORDS = re.compile(
    r"(?:tuition fee waiver|fee waiver|waiver|discount|stipend of|"
    r"\d+%\s*(?:tuition|fee|scholarship|waiver|discount)|"
    r"rs\.?\s*[\d,]+)",
    re.IGNORECASE,
)
_DURATION_KEYWORDS = re.compile(
    r"(?:for\s+(?:one|two|three|four|five|six|\d+)\s+years?|"
    r"for\s+(?:first|following)\s+semester|"
    r"for\s+standard\s+duration\s+of\s+the\s+program|"
    r"for\s+(?:the\s+)?program\s+duration)",
    re.IGNORECASE,
)
_ELIGIBILITY_KEYWORDS = re.compile(
    r"(?:top\s+\d+\s+of\s+the\s+admitted|"
    r"cgpa\s+(?:of\s+)?\d|"
    r"full[- ]time\s+student|"
    r"government\s+servants?|armed\s+forces|"
    r"open\s+to\s+faculty|"
    r"all\s+enrolled|"
    r"for\s+each\s+discipline)",
    re.IGNORECASE,
)


def build_scholarship_identity(name: str) -> str:
    normalized = _DASH_VARIANTS.sub("-", name)
    normalized = _TRAILING_PUNCTUATION.sub("", normalized)
    normalized = _WHITESPACE.sub(" ", normalized).strip()
    return slugify(normalized)


def normalize_scholarship_payload(
    *,
    category_label: str,
    columns: Mapping[str, str],
    raw_row_text: str,
    row_values: Sequence[str] = (),
) -> dict[str, object]:
    scholarship_name = _pick_column(
        columns,
        "Scholarship",
        "Scholarship Name",
        "Name",
    ) or _fallback_scholarship_name(columns, row_values)

    benefit_text = _pick_column(columns, "Support", "Benefit", "Benefits", "Details", "Detail")
    eligibility_text = _pick_column(columns, "Eligibility", "Criteria", "Condition")
    duration_text = _pick_column(columns, "Duration", "Tenure")

    # Fallback: extract from raw_row_text when columns don't have structured fields
    if benefit_text is None and eligibility_text is None and duration_text is None:
        benefit_text, eligibility_text, duration_text = _extract_from_raw_text(raw_row_text)
        # Also try to extract scholarship name from raw text if not found via columns
        if scholarship_name is None:
            scholarship_name = _extract_name_from_raw_text(raw_row_text)

    # When benefit came from a catch-all "Details" column, sub-classify its segments
    if (
        eligibility_text is None
        and duration_text is None
        and benefit_text is not None
        and _pick_column(columns, "Support", "Benefit", "Benefits") is None
    ):
        sub_benefit, sub_elig, sub_duration = _extract_from_raw_text(benefit_text)
        if sub_elig:
            eligibility_text = sub_elig
        if sub_duration:
            duration_text = sub_duration
        if sub_benefit:
            benefit_text = sub_benefit

    payload: dict[str, object] = {
        "category_label": category_label,
        "content": raw_row_text,
        "columns": dict(columns),
        "scholarship_name": scholarship_name,
        "benefit_text": benefit_text,
        "eligibility_text": eligibility_text,
        "duration_text": duration_text,
    }

    # Promote scholarship type from columns to top-level field
    scholarship_type = _pick_column(columns, "Type")
    if scholarship_type:
        payload["scholarship_type"] = scholarship_type

    return payload


def _extract_name_from_raw_text(raw: str) -> str | None:
    """Extract scholarship name from the first segment (before dash-separated details)."""
    # Pattern: "1. Name of Scholarship – Type ..."
    # Split on the first dash-prefixed segment
    segments = re.split(r"\s+-(?=[A-Z\d])", raw, maxsplit=1)
    if not segments:
        return None
    first = segments[0].strip()
    # Strip leading number prefix like "1." or "4."
    first = re.sub(r"^\d+\.\s*", "", first).strip()
    if first:
        return first
    return None


def _extract_from_raw_text(raw: str) -> tuple[str | None, str | None, str | None]:
    """Parse benefit, eligibility, and duration from dash-separated single-column text."""
    # Split on leading bullet dashes: handles both "-Tuition..." and " -For..."
    segments = re.split(r"(?:^|\s)-(?=\s*[A-Z\d])", raw)
    if len(segments) < 2:
        # Try splitting on " -" (space + dash)
        segments = re.split(r"\s+-", raw)

    benefit_parts: list[str] = []
    eligibility_parts: list[str] = []
    duration_text: str | None = None

    for segment in segments[1:]:  # Skip first segment (name + type)
        segment = segment.strip().lstrip("-").strip()
        if not segment:
            continue

        # Handle mixed segments with "Slabs:" separator
        if "slabs:" in segment.casefold():
            _split_slab_segment(segment, benefit_parts, eligibility_parts)
            continue

        has_benefit = _BENEFIT_KEYWORDS.search(segment) or any(
            kw in segment.casefold() for kw in ("stipend", "waiver", "discount", "fee")
        )
        has_duration = _DURATION_KEYWORDS.search(segment)
        has_eligibility = _ELIGIBILITY_KEYWORDS.search(segment)

        if has_benefit and has_eligibility:
            benefit_part, eligibility_part = _split_mixed_segment(segment)
            if benefit_part:
                benefit_parts.append(benefit_part)
            if eligibility_part:
                eligibility_parts.append(eligibility_part)
            if not benefit_part and not eligibility_part:
                eligibility_parts.append(segment)
        elif has_benefit:
            benefit_parts.append(segment)
        elif has_duration:
            duration_text = segment
        elif has_eligibility:
            eligibility_parts.append(segment)
        else:
            # Default: likely an eligibility/condition clause
            eligibility_parts.append(segment)

    return (
        "; ".join(benefit_parts) if benefit_parts else None,
        "; ".join(eligibility_parts) if eligibility_parts else None,
        duration_text,
    )


def _split_slab_segment(
    segment: str,
    benefit_parts: list[str],
    eligibility_parts: list[str],
) -> None:
    """Split a segment containing 'Slabs:' into eligibility prefix and benefit slabs."""
    idx = segment.casefold().index("slabs:")
    prefix = segment[:idx].strip()
    slab_content = segment[idx + len("Slabs:") :].strip()
    if prefix:
        eligibility_parts.append(prefix)
    if slab_content:
        benefit_parts.append(slab_content)


def _split_mixed_segment(segment: str) -> tuple[str | None, str | None]:
    """Split a mixed benefit/eligibility segment at the first eligibility clause."""
    match = _ELIGIBILITY_KEYWORDS.search(segment)
    if match is None:
        return None, None
    if match.start() == 0:
        return None, segment.strip().lstrip("-").strip()
    benefit = segment[: match.start()].strip().lstrip("-").strip(" ;,")
    eligibility = segment[match.start() :].strip().lstrip("-").strip(" ;,")
    return benefit or None, eligibility or None


def _pick_column(columns: Mapping[str, str], *names: str) -> str | None:
    normalized = {key.casefold(): value for key, value in columns.items()}
    for name in names:
        value = normalized.get(name.casefold())
        if value:
            return value
    return None


def _fallback_scholarship_name(
    columns: Mapping[str, str],
    row_values: Sequence[str],
) -> str | None:
    for key, value in columns.items():
        lowered = key.casefold()
        if lowered in {"sr.", "sr", "s.#", "s#", "#", "type", "eligibility", "support", "benefit"}:
            continue
        if value and not _looks_like_ordinal(value):
            return value
    for value in row_values:
        if value and not _looks_like_ordinal(value):
            return value
    return None


def _looks_like_ordinal(value: str) -> bool:
    normalized = value.strip().casefold()
    if not normalized:
        return True
    if re.fullmatch(r"\d+[.)]?", normalized):
        return True
    return normalized in {"merit", "stipend", "support", "benefit", "details", "detail"}
