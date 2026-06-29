from __future__ import annotations

import re
from collections.abc import Mapping
from difflib import SequenceMatcher

_MIN_PROGRAM_NAME_LEN = 6  # reject very short names that risk false-positive fuzzy matches


def _normalize_for_comparison(name: str) -> str:
    """Normalize a program name for fuzzy comparison."""
    s = name.upper()
    s = re.sub(r"\bAND\b", " ", s)
    s = s.replace("&", " ")
    return " ".join(s.split())


def fuzzy_program_lookup[V](
    program_name: str,
    program_map: dict[str, V],
) -> V | None:
    """Try progressively relaxed matching for program name -> map value.

    Strategies (in order of strictness):
    1. Exact upper-case match
    2. Strip degree prefix: "MS Executive MBA" -> "EXECUTIVE MBA"
    3. Ampersand-stripped match: "MANAGEMENT TECHNOLOGY" ~ "MANAGEMENT & TECHNOLOGY"
    4. Prefix match (longest key first): "MS PUBLIC POLICY" ~ "MS PUBLIC POLICY & SOCIETY"
    5. Edit-distance match (SequenceMatcher >= 0.85): "DEVELOPEMENT" ~ "DEVELOPMENT"
    """
    upper = program_name.upper()

    if len(upper) < _MIN_PROGRAM_NAME_LEN:
        return None

    # 1. Exact match
    if upper in program_map:
        return program_map[upper]

    # 2. Try without degree prefix (for "MS Executive MBA" -> "EXECUTIVE MBA")
    for prefix in ("BS ", "MS ", "PHD "):
        if upper.startswith(prefix):
            without = upper[len(prefix):]
            if without in program_map:
                return program_map[without]

    # 3. Ampersand-stripped match
    norm_upper = _normalize_for_comparison(upper)
    for key, value in program_map.items():
        if _normalize_for_comparison(key) == norm_upper:
            return value

    # 4. Prefix matching (longest key first to prefer most specific match)
    for key, value in sorted(program_map.items(), key=lambda x: len(x[0]), reverse=True):
        if len(key) < _MIN_PROGRAM_NAME_LEN:
            continue  # reject short keys that match too broadly (e.g., "BS")
        if key.startswith(upper) or upper.startswith(key):
            return value

    # 5. Edit-distance fallback for typos
    best_ratio = 0.0
    best_value: V | None = None
    for key, value in program_map.items():
        ratio = SequenceMatcher(None, upper, key).ratio()
        if ratio >= 0.85 and ratio > best_ratio:
            best_ratio = ratio
            best_value = value
    if best_value is not None:
        return best_value

    return None


def enrich_faculty_labels(
    records: list[dict],
    school_map: dict[str, str],
) -> list[dict]:
    """Set faculty_label on faculty_profile records from the school map.

    Only fills in faculty_label when it is currently None/null.
    Does not overwrite existing values.
    """
    for record in records:
        if record.get("record_type") != "faculty_profile":
            continue
        payload = record.get("record_payload", {})
        if payload.get("faculty_label") not in (None, "null", ""):
            continue
        name = payload.get("name", "")
        label = school_map.get(name)
        if label:
            payload["faculty_label"] = label
    return records


def enrich_faculty_department_labels(
    records: list[dict],
    faculty_dept_map: dict[str, str],
) -> list[dict]:
    """Set department_label on faculty_profile records from the dept map.

    Only fills in department_label when it is currently None/null/empty.
    """
    for record in records:
        if record.get("record_type") != "faculty_profile":
            continue
        payload = record.get("record_payload", {})
        if payload.get("department_label") not in (None, "null", ""):
            continue
        name = payload.get("name", "")
        label = faculty_dept_map.get(name)
        if label:
            payload["department_label"] = label
    return records


def enrich_program_departments(
    records: list[dict],
    program_dept_map: Mapping[str, str | tuple[str, str]],
) -> list[dict]:
    """Set department_label on program records from the cross-source map."""
    dept_dict: dict[str, str | tuple[str, str]] = dict(program_dept_map)
    for record in records:
        if record.get("record_type") != "program":
            continue
        payload = record.get("record_payload", {})
        if payload.get("department_label"):
            continue
        program_name = payload.get("program_name", "")
        entry = fuzzy_program_lookup(program_name, dept_dict)
        if entry:
            label = entry[0] if isinstance(entry, tuple) else entry
            payload["department_label"] = label
    return records


def enrich_program_faculty_labels(
    records: list[dict],
    program_faculty_map: dict[str, str],
) -> list[dict]:
    """Set faculty_label on program records from the program-faculty map.

    Only fills in faculty_label when it is currently None/null/empty.
    Uses fuzzy matching to handle typos, dropped ampersands, and prefix mismatches.
    """
    for record in records:
        if record.get("record_type") != "program":
            continue
        payload = record.get("record_payload", {})
        if payload.get("faculty_label") not in (None, "null", ""):
            continue
        program_name = payload.get("program_name", "")
        label = fuzzy_program_lookup(program_name, program_faculty_map)
        if label:
            payload["faculty_label"] = label
    return records
