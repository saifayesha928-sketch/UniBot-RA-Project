from __future__ import annotations

import functools
import re

# Domain-specific synonym groups for university queries.
# Each group maps related terms to each other.
_SYNONYM_GROUPS: tuple[tuple[str, ...], ...] = (
    ("fee", "tuition", "charges", "cost"),
    ("admission", "enrollment", "entry", "intake"),
    ("faculty", "professor", "teacher", "instructor"),
    ("complete", "completion", "graduate", "graduation", "degree completion"),
    ("grading", "grade", "cgpa", "gpa"),
    ("attendance", "class attendance", "attendance policy"),
    ("required", "requirement", "minimum", "needed"),
    ("program", "programme", "degree", "course of study"),
    ("scholarship", "financial aid", "merit scholarship", "need-based aid"),
    ("deadline", "last date", "due date", "closing date"),
    ("requirement", "eligibility", "criteria", "prerequisite"),
    ("semester", "term", "session"),
    ("campus", "university", "institute"),
    ("department", "school", "division"),
)

# Pre-build lookup: term -> group
_TERM_TO_GROUP: dict[str, tuple[str, ...]] = {}
for _group in _SYNONYM_GROUPS:
    for _term in _group:
        _TERM_TO_GROUP.setdefault(_term, _group)


@functools.lru_cache(maxsize=256)
def expand_query_with_synonyms(query_text: str, *, max_expansions: int = 3) -> tuple[str, ...]:
    """Generate query variants by substituting domain synonyms."""
    queries = [query_text]
    query_lower = query_text.lower()
    seen_lower = {query_lower}

    for term, group in _TERM_TO_GROUP.items():
        pattern = _term_pattern(term)
        if not pattern.search(query_lower):
            continue

        for synonym in group:
            if synonym == term:
                continue
            # Skip if the synonym already appears in the query to avoid
            # duplication (e.g., "attendance policy" → "attendance policy policy").
            if _term_pattern(synonym).search(query_lower):
                continue
            variant = pattern.sub(synonym, query_lower, count=1)
            if variant in seen_lower:
                continue
            queries.append(variant)
            seen_lower.add(variant)
            if len(queries) >= max_expansions:
                return tuple(queries)

    return tuple(queries)


def _term_pattern(term: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])")
