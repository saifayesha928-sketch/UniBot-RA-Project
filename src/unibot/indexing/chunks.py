from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from unibot.domain.record_payloads import extract_primary_content, extract_sections
from unibot.domain.record_renderers import render_chunk_prefix
from unibot.verify.currentness import can_enter_serving
from unibot.verify.rules import VerificationDecision

MAX_CHUNK_CHARS = 2000
CHUNK_OVERLAP_CHARS = 200
_SPLIT_SEPARATORS = ("\n\n", "\n", ". ", "? ", "! ", " ")


@dataclass(frozen=True, slots=True)
class IndexChunk:
    chunk_id: str
    record_version_id: str
    text: str
    metadata: dict[str, Any]
    is_active: bool = True
    original_text: str = ""  # Clean text for citations; empty = text is original


def build_chunks(
    records: list[VerificationDecision] | tuple[VerificationDecision, ...],
    *,
    serving_generation_id: str,
) -> tuple[IndexChunk, ...]:
    chunks: list[IndexChunk] = []

    for decision in records:
        #if not can_enter_serving(decision):
            #continue

        metadata = {
            "record_version_id": decision.candidate.record_version_id,
            "record_type": decision.candidate.record_type,
            "dedupe_key": decision.candidate.dedupe_key,
            "conflict_scope_id": decision.candidate.conflict_scope_id,
            "source_url": decision.candidate.source_url,
            "source_locator": decision.candidate.source_locator,
            "freshness_status": decision.freshness_status,
            "cycle_label": decision.candidate.cycle_label,
            "source_authority_tier": decision.candidate.source_authority_tier,
            "source_class": decision.candidate.record_payload.get("source_class"),
            "serving_generation_id": serving_generation_id,
        }

        strategy = _CHUNK_STRATEGIES.get(
            decision.candidate.record_type, _chunk_prose_fallback
        )
        text_parts = strategy(decision)

        chunk_count = len(text_parts)
        for chunk_index, text in enumerate(text_parts):
            chunks.append(
                IndexChunk(
                    chunk_id=f"{decision.candidate.record_version_id}:chunk:{chunk_index}",
                    record_version_id=decision.candidate.record_version_id,
                    text=text,
                    metadata={
                        **metadata,
                        "chunk_index": chunk_index,
                        "chunk_count": chunk_count,
                    },
                    is_active=True,
                )
            )

    return tuple(chunks)


# ---------------------------------------------------------------------------
# Prefix helpers
# ---------------------------------------------------------------------------

def _entity_prefix(decision: VerificationDecision) -> str:
    """Entity-specific prefix derived from the record payload.

    Always includes source_locator to preserve retrieval identity parity
    with the current production prefix format.
    """
    prefix = render_chunk_prefix(
        decision.candidate.record_type,
        decision.candidate.record_payload,
    )
    locator = decision.candidate.source_locator
    if locator:
        prefix = f"{prefix}\nSource Locator: {locator}"
    if decision.candidate.cycle_label:
        prefix = f"{prefix}\nCycle: {decision.candidate.cycle_label}"
    return prefix


def _section_prefix(decision: VerificationDecision, section_label: str) -> str:
    """Full chunk prefix: entity identity + optional section label."""
    prefix = _entity_prefix(decision)
    if section_label:
        prefix = f"{prefix}\nSection: {section_label}"
    return prefix


# ---------------------------------------------------------------------------
# Chunk strategies
# ---------------------------------------------------------------------------

def _chunk_atomic(decision: VerificationDecision) -> tuple[str, ...]:
    """Keep the record as a single chunk. If it exceeds MAX_CHUNK_CHARS,
    fall back to prose splitting rather than silently truncating.

    Used for record types where splitting destroys semantic integrity
    (e.g. faculty_publication, admissions_cycle, document_asset).
    """
    content = extract_primary_content(
        decision.candidate.record_payload,
        record_type=decision.candidate.record_type,
    )
    prefix = _entity_prefix(decision)
    text = f"{prefix}\n\n{content}" if content else prefix

    if len(text) <= MAX_CHUNK_CHARS:
        return (text,)

    # Record is too large for a single chunk — fall back to prose splitting
    return _chunk_prose_fallback(decision)


def _chunk_by_sections(decision: VerificationDecision) -> tuple[str, ...]:
    """Split by payload field boundaries. Each section becomes one or more chunks."""
    sections = extract_sections(
        decision.candidate.record_payload,
        record_type=decision.candidate.record_type,
    )
    if not sections:
        return _chunk_prose_fallback(decision)

    parts: list[str] = []
    for section_label, section_text in sections:
        prefix = _section_prefix(decision, section_label)
        full_text = f"{prefix}\n\n{section_text}" if section_text else prefix

        if len(full_text) <= MAX_CHUNK_CHARS:
            parts.append(full_text)
        else:
            # Section too large — sub-split its content, re-attach prefix to each
            available = MAX_CHUNK_CHARS - len(prefix) - 2  # account for \n\n
            if available <= 0:
                parts.extend(_split_text(full_text, MAX_CHUNK_CHARS))
            else:
                for sub_part in _split_text(section_text, available):
                    parts.append(f"{prefix}\n\n{sub_part}")

    return tuple(parts) if parts else _chunk_prose_fallback(decision)


def _chunk_prose_fallback(decision: VerificationDecision) -> tuple[str, ...]:
    """Original chunking logic: render to flat text, split by separators."""
    content = extract_primary_content(
        decision.candidate.record_payload,
        record_type=decision.candidate.record_type,
    )
    prefix = _entity_prefix(decision)
    text = f"{prefix}\n\n{content}" if content else prefix

    if len(text) <= MAX_CHUNK_CHARS:
        return (text,)

    available = MAX_CHUNK_CHARS - len(prefix) - 2
    if not content or available <= 0:
        return tuple(_split_text(text, MAX_CHUNK_CHARS))

    return tuple(
        f"{prefix}\n\n{part}" for part in _split_text(content, available)
    )


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

_CHUNK_STRATEGIES: dict[
    str,
    Callable[[VerificationDecision], tuple[str, ...]],
] = {
    # Section-based: split by payload field boundaries
    "program": _chunk_by_sections,
    "program_curriculum": _chunk_by_sections,
    "faculty_profile": _chunk_by_sections,
    "org_unit": _chunk_by_sections,
    "university_info": _chunk_by_sections,
    "student_service": _chunk_by_sections,
    "news_event": _chunk_by_sections,
    # Atomic: keep as single chunk (splitting destroys semantic integrity)
    "faculty_publication": _chunk_atomic,
    "faculty_award": _chunk_atomic,
    "faculty_affiliation": _chunk_atomic,
    "admissions_cycle": _chunk_atomic,
    "document_asset": _chunk_atomic,
    "research_entity": _chunk_atomic,
    "merit_list": _chunk_atomic,
    "program_fee_schedule": _chunk_atomic,
    # All other types fall through to _chunk_prose_fallback
}


# ---------------------------------------------------------------------------
# Text splitting (unchanged from original)
# ---------------------------------------------------------------------------

def _split_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    parts = _split_by_separators(text, max_chars, separators=_SPLIT_SEPARATORS)
    base_parts = parts if parts else _split_fixed_width(text, max_chars)
    return _apply_chunk_overlap(base_parts, max_chars)


def _split_by_separators(
    text: str,
    max_chars: int,
    *,
    separators: tuple[str, ...],
) -> list[str]:
    stripped_text = text.strip()
    if len(stripped_text) <= max_chars:
        return [stripped_text]
    if not separators:
        return _split_fixed_width(stripped_text, max_chars)

    separator = separators[0]
    if separator not in stripped_text:
        return _split_by_separators(
            stripped_text, max_chars, separators=separators[1:]
        )

    segments = _split_segments(stripped_text, separator)
    parts: list[str] = []
    current = ""

    for segment in segments:
        candidate = f"{current}{segment}" if current else segment
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            parts.append(current.strip())
            current = ""
        if len(segment) <= max_chars:
            current = segment
            continue
        parts.extend(
            _split_by_separators(segment, max_chars, separators=separators[1:])
        )

    if current:
        parts.append(current.strip())

    return parts


def _split_segments(text: str, separator: str) -> list[str]:
    raw_segments = text.split(separator)
    segments = [f"{part}{separator}" for part in raw_segments[:-1] if part]
    last = raw_segments[-1].strip()
    if last:
        segments.append(last)
    return segments


def _split_fixed_width(text: str, max_chars: int) -> list[str]:
    return [text[index : index + max_chars].strip() for index in range(0, len(text), max_chars)]


def _apply_chunk_overlap(parts: list[str], max_chars: int) -> list[str]:
    if len(parts) < 2 or CHUNK_OVERLAP_CHARS <= 0:
        return parts

    overlapped = [parts[0].strip()]
    for part in parts[1:]:
        current = part.strip()
        available_overlap = max(max_chars - len(current) - 2, 0)
        overlap_size = min(CHUNK_OVERLAP_CHARS, available_overlap)
        overlap = overlapped[-1][-overlap_size:].strip() if overlap_size else ""
        if overlap:
            overlapped.append(f"{overlap}\n\n{current}".strip())
        else:
            overlapped.append(current)
    return overlapped
