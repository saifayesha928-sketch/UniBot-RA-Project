from __future__ import annotations

import re
from collections.abc import Iterable

_QUALIFICATION_PATTERN = re.compile(
    r"(?:ph\.?d|dphil|m\.?s\.?|m\.?sc|mba|b\.?s\.?|b\.?sc|m\.?phil|masters?|post\s*doc)",
    re.IGNORECASE,
)
_BIOGRAPHY_NOISE_PATTERNS = (
    re.compile(r"^\s*(?:previous|next)\s*$", re.IGNORECASE),
    re.compile(r"^\s*image\s+\d+\s+of\s+\d+\s*$", re.IGNORECASE),
    re.compile(r"^\s*gallery\s*$", re.IGNORECASE),
)
_INLINE_BIOGRAPHY_NOISE = re.compile(r"\b(?:previous\s+next|next\s+previous)\b", re.IGNORECASE)
# Modal/lightbox navigation artifacts: ￩ (U+FFE9), ￫ (U+FFEB), × (U+00D7), standalone 'x'
_TRAILING_MODAL_ARTIFACTS = re.compile(
    r"[\s\u00d7\uffe9\uffeb]*(?:\bx\b[\s\u00d7\uffe9\uffeb]*)*[\s\u00d7\uffe9\uffeb]+$"
)
_INLINE_MODAL_ARTIFACTS = re.compile(r"[\uffe9\uffeb]+\s*(?:x\s*)?[\u00d7]?")


_FACULTY_OF_PATTERN = re.compile(
    r"Faculty\s+of\s+[\w&\s]+",
    re.IGNORECASE,
)


_CANONICAL_FACULTY_LABELS: dict[str, str] = {
    "faculty of sciences": "Faculty of Science",
    "faculty of science": "Faculty of Science",
    "faculty of engineering": "Faculty of Engineering",
    "faculty of business and management sciences": "Faculty of Business & Management Sciences",
    "faculty of business & management sciences": "Faculty of Business & Management Sciences",
    "faculty of business management sciences": "Faculty of Business & Management Sciences",
    "faculty of humanities and social sciences": "Faculty of Humanities & Social Sciences",
    "faculty of humanities & social sciences": "Faculty of Humanities & Social Sciences",
    "faculty of electrical engineering": "Faculty of Engineering",
}

_CANONICAL_DEPARTMENT_LABELS: dict[str, str] = {
    "department of computer and software engineering": "Department of Computer & Software Engineering",
    "department of electrical engineering": "Department of Electrical Engineering",
    "department of computer science": "Department of Computer Science",
    "department of artificial intelligence": "Department of Artificial Intelligence",
    "department of economics": "Department of Economics",
    "department of governance and global studies": "Department of Governance & Global Studies",
    "school of business & management": "School of Business & Management",
    "school of business and management": "School of Business & Management",
}


def extract_faculty_label(*texts: str | None) -> str | None:
    """Extract 'Faculty of X' from designation, role, or biography text."""
    for text in texts:
        if text is None:
            continue
        match = _FACULTY_OF_PATTERN.search(text)
        if match:
            raw = match.group(0).strip()
            return _CANONICAL_FACULTY_LABELS.get(raw.casefold(), raw)
    return None


def is_qualification_line(value: str | None) -> bool:
    if value is None:
        return False
    return bool(_QUALIFICATION_PATTERN.search(value))


def normalize_profile_header_lines(
    lines: Iterable[str],
) -> tuple[str | None, str | None, str | None]:
    designation_text: str | None = None
    qualification_text: str | None = None
    role_text: str | None = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        if qualification_text is None and is_qualification_line(line):
            qualification_text = line
            continue

        if designation_text is None:
            designation_text = line
            continue

        if role_text is None:
            role_text = line

    return designation_text, qualification_text, role_text


_ROLE_KEYWORDS = (
    "chairperson", "chair", "dean", "director", "head",
    "vice chancellor", "chancellor",
)

_DEPARTMENT_PATTERN = re.compile(
    r"(?:Department of|School of)\s+[\w& ]+",
    re.IGNORECASE,
)


def parse_designation_text(
    text: str | None,
) -> dict[str, str | None]:
    """Parse raw designation text into structured designation, role, department."""
    if text is None:
        return {
            "designation": None,
            "role": None,
            "department_label": None,
        }

    working = text.strip()
    department: str | None = None
    role: str | None = None
    designation: str | None = None

    # Extract department (segment containing Department/School/Faculty of)
    dept_match = _DEPARTMENT_PATTERN.search(working)
    if dept_match:
        raw_dept = dept_match.group(0).strip()
        department = _CANONICAL_DEPARTMENT_LABELS.get(raw_dept.casefold(), raw_dept)

    # Split by comma to get main segments
    comma_parts = [p.strip() for p in working.split(",", 1)]
    title_part = comma_parts[0]

    # Check for "X and Role" pattern in title_part
    and_pattern = re.compile(
        r"^(.+?)\s+and\s+((?:"
        + "|".join(re.escape(k) for k in _ROLE_KEYWORDS)
        + r")(?:\s+.+)?)",
        re.IGNORECASE,
    )
    and_match = and_pattern.match(title_part)
    if and_match:
        designation = and_match.group(1).strip()
        role = and_match.group(2).strip()
    else:
        # Check for "Chairperson-Department" pattern in suffix
        if len(comma_parts) > 1:
            suffix = comma_parts[1].strip()
            for keyword in _ROLE_KEYWORDS:
                if suffix.casefold().startswith(keyword):
                    role = keyword.title()
                    rest = suffix[len(keyword):].lstrip(" -")
                    if rest:
                        dept_in_rest = _DEPARTMENT_PATTERN.search(rest)
                        if dept_in_rest:
                            raw_dept = dept_in_rest.group(0).strip()
                            department = _CANONICAL_DEPARTMENT_LABELS.get(
                                raw_dept.casefold(), raw_dept,
                            )
                    break
        designation = title_part.strip()

    # Handle "Vice Chancellor" — the whole thing is the designation
    if designation and designation.casefold() in ("vice chancellor",):
        role = None
        department = None

    return {
        "designation": designation or None,
        "role": role or None,
        "department_label": department or None,
    }


def clean_biography_text(raw_text: str | None) -> str | None:
    if raw_text is None:
        return None

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    cleaned_lines = [
        line
        for line in lines
        if not any(pattern.match(line) for pattern in _BIOGRAPHY_NOISE_PATTERNS)
    ]
    if not cleaned_lines:
        return None
    cleaned_text = re.sub(r"\s+", " ", " ".join(cleaned_lines)).strip()
    cleaned_text = _INLINE_BIOGRAPHY_NOISE.sub("", cleaned_text)
    cleaned_text = _INLINE_MODAL_ARTIFACTS.sub("", cleaned_text)
    cleaned_text = _TRAILING_MODAL_ARTIFACTS.sub("", cleaned_text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip(" ,;:-")
    return cleaned_text or None
