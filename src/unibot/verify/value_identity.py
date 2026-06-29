"""Single source of truth for canonical value hashing.

Rules:
1. canonical_value_hash normalizes with .strip() then SHA-256
2. VerificationCandidate.value_hash must ALWAYS hold a SHA-256 digest, never raw text
3. value_hash_for_stored_record in the fallback case returns source_text_hash directly
   (it is already a digest from the write path)
4. Search and identity rendering go through typed record renderers
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from datetime import date

from unibot.domain.record_renderers import infer_record_type, render_identity_text
from unibot.utils import normalize_url_identity as _normalize_url_identity
from unibot.extract.normalizers.scholarships import build_scholarship_identity

_DASH_VARIANTS = re.compile(r"[\u2013\u2014\u2015\u2212]")
_INLINE_NOISE = re.compile(r"\s*\*\s*")
_TRAILING_MARKS = re.compile(r"[\s.,;:!?]+$")
_WHITESPACE_NORM = re.compile(r"\s+")


def canonical_value_hash(text: str) -> str:
    """Normalize whitespace, then SHA-256. Used everywhere."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def value_hash_for_incoming_record(
    record_type_or_payload: str | Mapping[str, object] | None,
    record_payload: Mapping[str, object] | None = None,
    fallback_text: str | None = None,
) -> str:
    """For freshly extracted records.

    Extracts primary content via CONTENT_KEYS, falls back to fallback_text,
    then hashes. Always returns a SHA-256 digest.
    """
    if isinstance(record_type_or_payload, Mapping):
        record_type = None
        payload = record_type_or_payload
        fallback = fallback_text or ""
    else:
        record_type = record_type_or_payload
        payload = record_payload or {}
        fallback = fallback_text or ""
    resolved_record_type = record_type or infer_record_type(payload) or "generic"
    content = render_identity_text(resolved_record_type, payload)
    return canonical_value_hash(content if content else fallback)


def _normalize_text(text: str) -> str:
    """Normalize text for semantic comparison: dashes, trailing marks, whitespace."""
    normalized = _DASH_VARIANTS.sub("-", text)
    normalized = _INLINE_NOISE.sub(" ", normalized)
    normalized = _TRAILING_MARKS.sub("", normalized)
    normalized = _WHITESPACE_NORM.sub(" ", normalized).strip().lower()
    return normalized



def semantic_identity_payload(
    record_type: str, record_payload: Mapping[str, object]
) -> str:
    """Record-type-aware semantic identity that normalizes minor variants."""
    if record_type == "scholarship":
        scholarship_name = str(record_payload.get("scholarship_name", "")).strip()
        if scholarship_name:
            return build_scholarship_identity(scholarship_name)
        content = str(record_payload.get("content", ""))
        return _normalize_text(content)
    if record_type == "admissions_cycle":
        name = str(record_payload.get("milestone_name", ""))
        date_parts = _normalized_admissions_dates(record_payload)
        date_text = "|".join(date_parts) if date_parts else str(record_payload.get("date_text", ""))
        calendar_scope = str(record_payload.get("calendar_scope", ""))
        return _normalize_text(f"{calendar_scope}:{name}:{date_text}")
    if record_type == "program_fee_schedule":
        program_name = str(record_payload.get("program_name", ""))
        audience = str(record_payload.get("audience", ""))
        table_kind = str(record_payload.get("table_kind", ""))
        cycle_label = str(record_payload.get("cycle_label", ""))
        fee = str(record_payload.get("raw_table_text") or record_payload.get("fee_table_markdown", ""))
        return _normalize_text(f"{program_name}:{audience}:{table_kind}:{cycle_label}:{fee}")
    if record_type == "faculty_publication":
        val = str(record_payload.get("raw_citation") or record_payload.get("value_text", ""))
        return _normalize_text(val)
    if record_type == "research_entity":
        center_url = str(
            record_payload.get("subdomain_url")
            or record_payload.get("canonical_center_url")
            or ""
        ).strip()
        if center_url:
            return _normalize_url_identity(center_url)
        return _normalize_text(str(record_payload.get("name", "")))
    if record_type in ("faculty_award", "faculty_affiliation"):
        val = str(record_payload.get("value_text", ""))
        return _normalize_text(val)
    resolved_record_type = record_type or infer_record_type(record_payload) or "generic"
    content = render_identity_text(resolved_record_type, record_payload)
    return _normalize_text(content) if content else ""


def semantic_value_hash(
    record_type: str, record_payload: Mapping[str, object]
) -> str:
    """Semantic-aware value hash for comparison.

    Returns a hash based on normalized content so minor punctuation
    variants (dash types, trailing asterisks) produce the same hash.
    """
    identity = semantic_identity_payload(record_type, record_payload)
    return canonical_value_hash(identity) if identity else ""


def effective_value_hash(record_type: str, record_payload: Mapping[str, object], raw_hash: str) -> str:
    """Return semantic hash if available, otherwise raw hash."""
    sem = semantic_value_hash(record_type, record_payload)
    return sem if sem else raw_hash


def value_hash_for_stored_record(
    record_type_or_payload: str | Mapping[str, object] | None,
    record_payload: Mapping[str, object] | str,
    source_text_hash: str | None = None,
) -> str:
    """For DB-reconstructed records.

    Re-extracts content if possible; falls back to source_text_hash AS-IS
    (never re-hashes a hash).
    """
    if isinstance(record_type_or_payload, Mapping):
        record_type = None
        payload = record_type_or_payload
        stored_hash = str(record_payload)
    else:
        record_type = record_type_or_payload
        payload = record_payload if isinstance(record_payload, Mapping) else {}
        stored_hash = source_text_hash or ""
    resolved_record_type = record_type or infer_record_type(payload) or "generic"
    content = render_identity_text(resolved_record_type, payload)
    if content:
        return canonical_value_hash(content)
    return stored_hash


def _normalized_admissions_dates(record_payload: Mapping[str, object]) -> tuple[str, ...]:
    values = record_payload.get("normalized_dates")
    if not isinstance(values, list):
        return ()

    normalized: list[str] = []
    for value in values:
        if isinstance(value, date):
            normalized.append(value.isoformat())
        elif isinstance(value, str) and value.strip():
            normalized.append(value.strip())
    return tuple(normalized)
