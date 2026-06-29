from __future__ import annotations

import re

_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")


def _strip_md_links(text: str) -> str:
    """Replace [text](url) with just text."""
    return _MD_LINK_RE.sub(r"\1", text)


_NOISE_LINES = frozenset({"\uffeb", "\uffe9", "x", "\u00d7", "Previous", "Next", ""})

_MD_HEADING_PREFIX = re.compile(r"^#{1,6}\s+")


def _strip_md_heading(text: str) -> str:
    """Remove leading ``# `` markdown heading markers."""
    return _MD_HEADING_PREFIX.sub("", text)

_PERSON_NAME_RE = re.compile(
    r"^(?:(?:Tenured\s+)?Prof\.?\s+)?(?:Dr\.?\s+)?[A-Z][a-z]+"
    r"(?:\s+[A-Za-z][A-Za-z.'\-]*)+$"
)

_DESIGNATION_BLOCKLIST = frozenset({
    "Assistant Professor", "Associate Professor", "Professor",
    "Teaching Fellow", "Tenured Associate Professor",
    "Tenured Professor", "Adjunct Faculty", "IPFP Fellow",
    "Vice Chancellor", "Lecturer", "Senior Lecturer",
})

_DESIGNATION_PREFIXES = (
    "Professor ", "Assistant Professor ", "Associate Professor ",
    "Tenured ", "Teaching ", "Adjunct ", "Lecturer ",
)

_SKIP_PREFIXES = (
    "BS ", "MS ", "PHD ", "EXECUTIVE ", "Department of",
    "Faculty of", "School of",
)

_DEPARTMENT_RE = re.compile(
    r"^(?:#{1,6}\s+)?(?:Department of|School of)\s+.+$",
)
_PROGRAM_RE = re.compile(
    # IGNORECASE: real pages mix "BS COMPUTER SCIENCE" and "BS Financial Technology".
    # All matched results are stored via .upper() so the case-insensitive match is intentional.
    r"^(?:BS|MS|PHD|EXECUTIVE)\s+[A-Za-z][A-Za-z &]+$",
    re.IGNORECASE,
)

# Normalization patterns for program lines with parenthesised suffixes or colons
_PAREN_SUFFIX_RE = re.compile(r"\s*\(.*?\)\s*$")
_COLON_SUFFIX_RE = re.compile(r"\s*:.*$")

LISTING_SLUG_TO_LABEL: dict[str, str] = {
    "faculty-of-engineering": "Faculty of Engineering",
    "faculty-of-science": "Faculty of Science",
    "faculty-of-business-management-science": "Faculty of Business & Management Sciences",
    "faculty-of-humanities-social-sciences": "Faculty of Humanities & Social Sciences",
}

def parse_faculty_names_from_listing(content: str) -> list[str]:
    """Extract person names from a faculty listing page's text content."""
    names: list[str] = []
    for line in content.split("\n"):
        line = line.strip()
        if line in _NOISE_LINES:
            continue
        if any(line.startswith(prefix) for prefix in _SKIP_PREFIXES):
            continue
        if len(line) > 80 or len(line) < 5:
            continue
        if _PERSON_NAME_RE.match(line):
            if line in _DESIGNATION_BLOCKLIST:
                continue
            if any(line.startswith(p) for p in _DESIGNATION_PREFIXES):
                continue
            names.append(line)
    return names


def build_faculty_school_map(
    listings: dict[str, str],
) -> dict[str, str]:
    """Build {faculty_name -> faculty_label} from listing page contents.

    Args:
        listings: mapping of faculty_label -> page text content
    """
    result: dict[str, str] = {}
    for faculty_label, content in listings.items():
        for name in parse_faculty_names_from_listing(content):
            result[name] = faculty_label
    return result


def build_faculty_department_map(
    listings: dict[str, str],
) -> dict[str, str]:
    """Build {faculty_name -> department_label} from listing page contents.

    Scans each listing for 'Department of ...' headings followed by person
    names. Each name is mapped to the most recent department heading.
    """
    result: dict[str, str] = {}
    for _faculty_label, content in listings.items():
        lines = [_strip_md_links(line).strip() for line in content.split("\n") if line.strip()]
        current_dept: str | None = None
        consecutive_non_name = 0
        for line in lines:
            if _DEPARTMENT_RE.match(line):
                current_dept = _strip_md_heading(line)
                consecutive_non_name = 0
            elif current_dept:
                if line in _NOISE_LINES or len(line) > 80 or len(line) < 5:
                    consecutive_non_name += 1
                elif _PERSON_NAME_RE.match(line) and line not in _DESIGNATION_BLOCKLIST:
                    if not any(line.startswith(p) for p in _DESIGNATION_PREFIXES):
                        result[line] = current_dept
                        consecutive_non_name = 0
                    else:
                        consecutive_non_name += 1
                elif _PROGRAM_RE.match(line):
                    # Programs listed under department — skip but don't break
                    consecutive_non_name = 0
                    continue
                else:
                    consecutive_non_name += 1
                if consecutive_non_name >= 3:
                    current_dept = None
    return result


_DEPT_PROGRAM_SEPARATOR = re.compile(r"[/,;]")


def _extract_dept_subject_words(dept_label: str) -> set[str]:
    """Extract subject keywords from a department label.

    ``"Department of Computer Science"`` → ``{"computer", "science"}``
    ``"School of Business & Management"`` → ``{"business", "management"}``
    """
    cleaned = re.sub(
        r"^(?:Department of|School of)\s+",
        "",
        dept_label,
        flags=re.IGNORECASE,
    )
    return {w.lower() for w in re.findall(r"[A-Za-z]+", cleaned) if len(w) > 1}


def _keyword_match_department(
    program_name: str,
    departments: list[tuple[str, str]],
) -> tuple[str, str] | None:
    """Match a program to a department by keyword overlap.

    Args:
        program_name: e.g. ``"BS COMPUTER SCIENCE"``
        departments: list of ``(dept_label, faculty_label)`` tuples

    Returns:
        Best-matching ``(dept_label, faculty_label)`` or None if no match.
    """
    # Strip degree prefix from program name for matching
    prog_words = {
        w.lower()
        for w in re.findall(r"[A-Za-z]+", program_name)
        if w.upper() not in {"BS", "MS", "PHD", "EXECUTIVE", "MBA"}
        and len(w) > 1
    }
    if not prog_words:
        return None

    best: tuple[str, str] | None = None
    best_score = 0
    for dept_label, faculty_label in departments:
        dept_words = _extract_dept_subject_words(dept_label)
        if not dept_words:
            continue
        overlap = len(prog_words & dept_words)
        if overlap > best_score:
            best_score = overlap
            best = (dept_label, faculty_label)
    # Require at least 2 keyword overlaps to be confident.  A single
    # shared word (e.g. "science" in both "Data Science" and "Computer
    # Science") is too ambiguous to override positional evidence.
    return best if best_score >= 2 else None


def parse_program_department_map(
    listings: dict[str, str],
) -> dict[str, tuple[str, str]]:
    """Parse program->department mapping from faculty listing pages.

    Uses a two-pass approach:
    1. Positional pass: maps programs to the most recent department heading
       (works when programs are grouped under their department).
    2. Keyword correction pass: for each faculty that has multiple
       departments, re-scores every program against all department names
       using keyword overlap.  A keyword match overrides the positional
       assignment only when it finds a better-scoring department.

    This fixes the Science faculty case where both department headings
    appear before all programs in a flat list — ``BS Computer Science``
    is correctly assigned to ``Department of Computer Science`` by
    keyword match even though the positional pass maps it to
    ``Department of Artificial Intelligence`` (the last heading).
    """
    # --- Pass 1: positional (unchanged logic) ---
    result: dict[str, tuple[str, str]] = {}
    # Track all departments found per faculty for keyword matching
    faculty_departments: dict[str, list[tuple[str, str]]] = {}

    for faculty_label, content in listings.items():
        lines = [_strip_md_links(line).strip() for line in content.split("\n") if line.strip()]
        current_dept: str | None = None
        consecutive_non_program = 0
        for line in lines:
            if _DEPARTMENT_RE.match(line):
                current_dept = _strip_md_heading(line)
                faculty_departments.setdefault(faculty_label, []).append(
                    (current_dept, faculty_label),
                )
                consecutive_non_program = 0
            elif current_dept:
                parts = _DEPT_PROGRAM_SEPARATOR.split(line)
                matched_any = False
                for part in parts:
                    normalized = _PAREN_SUFFIX_RE.sub("", part.strip())
                    normalized = _COLON_SUFFIX_RE.sub("", normalized)
                    if _PROGRAM_RE.match(normalized):
                        result[normalized.upper()] = (current_dept, faculty_label)
                        matched_any = True
                if not matched_any:
                    consecutive_non_program += 1
                    if consecutive_non_program >= 2:
                        current_dept = None

    # --- Pass 2: keyword correction for faculties with 2+ departments ---
    for faculty_label, depts in faculty_departments.items():
        if len(depts) < 2:
            continue
        for prog, (dept, fac) in list(result.items()):
            if fac != faculty_label:
                continue
            better = _keyword_match_department(prog, depts)
            if better is not None and better[0] != dept:
                result[prog] = better

    return result


_CONTENT_ZONE_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T")
_FACULTY_LABEL_LINE_RE = re.compile(r"^Faculty of\s", re.IGNORECASE)

# Programs not listed on any faculty org-unit page but with known affiliations.
# Kept minimal — add entries only when source pages genuinely lack the information.
_PROGRAM_FACULTY_OVERRIDES: dict[str, str] = {
    "MS CYBER SECURITY": "Faculty of Science",
}


def parse_program_faculty_map(
    listings: dict[str, str],
    *,
    dept_map: dict[str, tuple[str, str]] | None = None,
) -> dict[str, str]:
    """Map program names to their faculty label using content-zone detection.

    Each faculty listing page has a consistent structure:
      nav (all programs) → date → description → CONTENT PROGRAMS → Faculty bio section → footer

    This function scans only the content zone (between the ISO date line
    and the faculty bio section) to avoid nav/footer contamination.

    Programs found in the department map are included first (highest priority).
    Then each page's content zone is scanned for additional programs (handles
    faculties like Business and Humanities that have no department headings).

    First-write-wins: if a program appears in multiple content zones (shouldn't
    happen but defensive), the first faculty page processed claims it.

    Args:
        listings: mapping of faculty_label -> page text content
        dept_map: pre-built department map to avoid recomputation; if None,
            parse_program_department_map is called internally
    """
    # Phase 1: seed from dept_map (most reliable for faculties with departments)
    if dept_map is None:
        dept_map = parse_program_department_map(listings)
    result: dict[str, str] = {}
    for prog, (_dept, faculty) in dept_map.items():
        result[prog] = faculty

    # Phase 2: content-zone scan for each faculty page
    for faculty_label, content in listings.items():
        lines = [_strip_md_links(line).strip() for line in content.split("\n") if line.strip()]

        # If no date line exists (already-clean text), scan all lines.
        has_date = any(_CONTENT_ZONE_DATE_RE.match(ln) for ln in lines)
        past_date = not has_date  # start scanning immediately if no date
        found_programs = False
        consecutive_non_program = 0

        for line in lines:
            if _CONTENT_ZONE_DATE_RE.match(line):
                past_date = True
                continue
            if not past_date:
                continue

            # Skip description paragraphs (long prose lines)
            if len(line) > 80:
                continue

            # Skip noise lines (very short)
            if len(line) < 3:
                continue

            # Skip department headings (already handled by dept_map)
            if _DEPARTMENT_RE.match(line):
                consecutive_non_program = 0
                continue

            # "Faculty of X" after programs signals bio section start — exit zone
            if found_programs and _FACULTY_LABEL_LINE_RE.match(line):
                break

            # "Faculty of X" before programs — skip (label line on Science page)
            if not found_programs and _FACULTY_LABEL_LINE_RE.match(line):
                continue

            normalized = _PAREN_SUFFIX_RE.sub("", line)
            normalized = _COLON_SUFFIX_RE.sub("", normalized)

            if _PROGRAM_RE.match(normalized):
                key = normalized.upper()
                if key not in result:
                    result[key] = faculty_label
                found_programs = True
                consecutive_non_program = 0
            elif found_programs:
                consecutive_non_program += 1
                if consecutive_non_program >= 2:
                    break

    # Phase 3: hardcoded overrides for programs missing from all listing pages
    for prog, faculty in _PROGRAM_FACULTY_OVERRIDES.items():
        if prog not in result:
            result[prog] = faculty

    return result
