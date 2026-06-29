from __future__ import annotations

import calendar
import re
from datetime import date, datetime
from typing import Any

from unibot.extract.records import YearConfidence
from unibot.extract.text import slugify

DEFAULT_CALENDAR_SCOPE = "undergraduate_or_general_admissions"

_YEAR_PATTERN = re.compile(r"\b(?P<year>20\d{2})\b")
_ORDINAL_SUFFIX_PATTERN = re.compile(r"(?P<day>\d{1,2})(?:st|nd|rd|th)\b", re.IGNORECASE)
_WEEKDAY_PREFIX_PATTERN = re.compile(
    r"^(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday),?\s+",
    re.IGNORECASE,
)

_WEEKDAY_TO_INT: dict[str, int] = {
    name.lower(): i for i, name in enumerate(calendar.day_name)
}


def normalize_milestone_name(label: str) -> str:
    return re.sub(r"\s+", " ", label.strip(" :-")).strip()


def milestone_scope_id(
    milestone_name: str,
    *,
    calendar_scope: str = DEFAULT_CALENDAR_SCOPE,
) -> str:
    return (
        f"admissions_cycle:{slugify(calendar_scope)}:"
        f"{slugify(normalize_milestone_name(milestone_name))}"
    )


def split_date_range(date_text: str) -> tuple[str, ...]:
    parts = [part.strip() for part in re.split(r"\s*&\s*", date_text) if part.strip()]
    return tuple(parts) or (date_text.strip(),)


def normalize_date_text(date_text: str) -> tuple[date | None, str]:
    normalized_text = _WEEKDAY_PREFIX_PATTERN.sub("", date_text.strip())
    normalized_text = _ORDINAL_SUFFIX_PATTERN.sub(r"\g<day>", normalized_text)
    normalized_text = re.sub(r"\s+", " ", normalized_text).strip()
    if not _YEAR_PATTERN.search(normalized_text):
        inferred = _infer_year_from_weekday(date_text, normalized_text)
        if inferred is None:
            return None, "unknown"
        normalized_text, confidence = inferred
        for fmt in ("%d %B %Y", "%d %b %Y", "%B %d %Y", "%b %d %Y"):
            try:
                return datetime.strptime(normalized_text, fmt).date(), confidence
            except ValueError:
                continue
        return None, "unknown"

    for fmt in (
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d %B %Y",
        "%d %b %Y",
        "%B %d %Y",
        "%b %d %Y",
    ):
        try:
            return datetime.strptime(normalized_text, fmt).date(), "high"
        except ValueError:
            continue

    return None, "unknown"


def normalize_milestone_payload(
    milestone_name: str,
    date_text: str,
    *,
    calendar_scope: str = DEFAULT_CALENDAR_SCOPE,
    source_last_modified_text: str | None = None,
) -> tuple[dict[str, Any], YearConfidence]:
    normalized_dates: list[date] = []
    confidences: list[str] = []
    for part in split_date_range(date_text):
        normalized_date, confidence = normalize_date_text(part)
        if normalized_date is not None:
            normalized_dates.append(normalized_date)
        confidences.append(confidence)

    year_confidence: YearConfidence
    if not confidences or "unknown" in confidences:
        year_confidence = "unknown"
    elif "low" in confidences:
        year_confidence = "low"
    elif "medium" in confidences:
        year_confidence = "medium"
    else:
        year_confidence = "high"

    if "medium" in confidences or "low" in confidences:
        date_resolution = "weekday_inferred"
    elif year_confidence == "high":
        date_resolution = "explicit"
    else:
        date_resolution = "unknown"

    return (
        {
            "milestone_name": normalize_milestone_name(milestone_name),
            "date_text": date_text.strip(),
            "normalized_dates": normalized_dates,
            "calendar_scope": calendar_scope,
            "source_last_modified_text": source_last_modified_text,
            "date_resolution": date_resolution,
        },
        year_confidence,
    )


def _infer_year_from_weekday(
    original_text: str,
    normalized_no_year: str,
) -> tuple[str, str] | None:
    """Infer the year for a date string that has a weekday prefix but no year.

    Scores candidate years (current, +1, −1) on two axes:
    1. Does the weekday match the stated day-name?
    2. Is the resulting date in the future or recent past (within 180 days)?

    Ranking (highest wins):
    - weekday match + future/recent  →  confidence "medium"
    - future/recent only             →  confidence "low"
    - weekday match + distant past   →  confidence "medium"
    - neither                        →  skip
    """
    from datetime import timedelta

    weekday_match = _WEEKDAY_PREFIX_PATTERN.match(original_text.strip())
    if weekday_match is None:
        return None

    weekday_str = weekday_match.group(0).strip().rstrip(",").strip().lower()
    target_weekday = _WEEKDAY_TO_INT.get(weekday_str)
    if target_weekday is None:
        return None

    today = date.today()
    recency_cutoff = today - timedelta(days=180)
    candidates: list[tuple[date, str, bool, bool, int]] = []

    for candidate_year in (today.year, today.year + 1, today.year - 1):
        test_text = f"{normalized_no_year} {candidate_year}"
        for fmt in ("%d %B %Y", "%d %b %Y", "%B %d %Y", "%b %d %Y"):
            try:
                parsed = datetime.strptime(test_text, fmt).date()
                weekday_ok = parsed.weekday() == target_weekday
                future_or_recent = parsed >= recency_cutoff
                candidates.append(
                    (parsed, test_text, weekday_ok, future_or_recent, candidate_year)
                )
                break  # found a valid parse for this year
            except ValueError:
                continue

    if not candidates:
        return None

    year_priority = {today.year: 0, today.year + 1: 1, today.year - 1: 2}

    def _sort_key(
        c: tuple[date, str, bool, bool, int],
    ) -> tuple[int, int]:
        _parsed, _text, weekday_ok, future_ok, year = c
        if weekday_ok and future_ok:
            score = 3
        elif future_ok:
            score = 2
        elif weekday_ok:
            score = 1
        else:
            score = 0
        return (-score, year_priority.get(year, 3))

    candidates.sort(key=_sort_key)
    best = candidates[0]
    confidence = "medium" if best[2] else "low"
    return best[1], confidence
