from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from enum import StrEnum


class QueryClass(StrEnum):
    ENTITY_LOOKUP = "entity_lookup"
    POLICY_OR_THRESHOLD = "policy_or_threshold"
    FEE_OR_ADMISSIONS_CYCLE = "fee_or_admissions_cycle"
    FACULTY_EXPERTISE_OR_PUBLICATION = "faculty_expertise_or_publication"
    NEWS_OR_EVENT = "news_or_event"
    OUT_OF_SCOPE = "out_of_scope"


@dataclass(frozen=True, slots=True)
class QueryClassification:
    query_class: QueryClass
    source_class_hint: str | None
    record_type_hint: str | None
    abstain_immediately: bool
    reason: str


_SOURCE_CLASS_ALIASES = {
    "faculty_profile": "faculty",
    "faculty_publication": "faculty",
    "policy_rule": "policy",
    "research_entity": "research_main",
}
_OUT_OF_SCOPE_TERMS = (
    "weather",
    "temperature",
    "forecast",
    "rain",
    "humidity",
    "stock",
    "share price",
    "bitcoin",
    "crypto",
    "recipe",
    "restaurant",
    "movie",
    "football",
    "cricket",
    "nba",
    "nfl",
)
_NEWS_EVENT_TERMS = (
    "news",
    "event",
    "events",
    "seminar",
    "workshop",
    "conference",
    "convocation",
)
_FACULTY_TERMS = (
    "faculty",
    "professor",
    "dr ",
    "dr.",
    "publication",
    "publications",
    "research interest",
    "research interests",
    "expertise",
    "works on",
)
_POLICY_TERMS = (
    "policy",
    "policies",
    "attendance",
    "probation",
    "threshold",
    "cgpa",
    "grade",
    "grading",
    "rule",
    "rules",
    "regulation",
    "regulations",
    "requirement",
)
_FEE_TERMS = ("fee", "fees", "tuition")
_SCHOLARSHIP_TERMS = ("scholarship", "scholarships", "financial aid", "financial assistance")
_ELIGIBILITY_TERMS = ("eligibility", "eligible", "eligibility criteria", "who can apply")
_MERIT_TERMS = ("merit", "merit list", "merit lists")
_ADMISSIONS_TERMS = (
    "admission",
    "admissions",
    "apply",
    "application",
    "deadline",
    "intake",
)
_IN_SCOPE_TERMS = (
    _NEWS_EVENT_TERMS
    + _FACULTY_TERMS
    + _POLICY_TERMS
    + _FEE_TERMS
    + _SCHOLARSHIP_TERMS
    + _ELIGIBILITY_TERMS
    + _MERIT_TERMS
    + _ADMISSIONS_TERMS
    + (
        "university",
        "department",
        "program",
        "office",
        "student service",
        "hostel",
        "placement",
        "campus",
    )
)


_VALID_SOURCE_CLASS_HINTS: frozenset[str] = frozenset({
    "faculty", "policy", "news_event", "scholarship",
    "merit_list", "admissions_cycle", "research_main", "general",
    "program", "program_fee_schedule", "library", "university_info",
    "student_service", "org_unit", "document_asset",
})


def normalize_source_class_hint(source_class_hint: str | None) -> str | None:
    if source_class_hint is None:
        return None
    # Strip control characters to prevent log injection.
    cleaned = "".join(ch for ch in source_class_hint if ch >= " ")
    normalized = cleaned.strip().lower()
    if not normalized:
        return None
    resolved = _SOURCE_CLASS_ALIASES.get(normalized, normalized)
    return resolved if resolved in _VALID_SOURCE_CLASS_HINTS else None


def classify_query(query_text: str) -> QueryClassification:
    normalized = _normalize_query(query_text)

    if _contains_any(normalized, _OUT_OF_SCOPE_TERMS) and not _contains_any(
        normalized, _IN_SCOPE_TERMS
    ):
        return QueryClassification(
            query_class=QueryClass.OUT_OF_SCOPE,
            source_class_hint=None,
            record_type_hint=None,
            abstain_immediately=True,
            reason="The request is outside the retrieval scope.",
        )

    if _contains_any(normalized, _NEWS_EVENT_TERMS):
        return QueryClassification(
            query_class=QueryClass.NEWS_OR_EVENT,
            source_class_hint="news_event",
            record_type_hint="news_event",
            abstain_immediately=False,
            reason="Route through news and events sources.",
        )

    if _contains_any(normalized, _FACULTY_TERMS):
        return QueryClassification(
            query_class=QueryClass.FACULTY_EXPERTISE_OR_PUBLICATION,
            source_class_hint="faculty",
            record_type_hint=(
                "faculty_publication"
                if _contains_any(normalized, ("publication", "publications"))
                else None
            ),
            abstain_immediately=False,
            reason="Route through faculty profile sources.",
        )

    if _contains_any(normalized, _POLICY_TERMS):
        return QueryClassification(
            query_class=QueryClass.POLICY_OR_THRESHOLD,
            source_class_hint="policy",
            record_type_hint=None,  # Policy content spans multiple record types.
            abstain_immediately=False,
            reason="Route through policy sources.",
        )

    if _contains_any(normalized, _FEE_TERMS):
        return QueryClassification(
            query_class=QueryClass.FEE_OR_ADMISSIONS_CYCLE,
            source_class_hint=None,
            record_type_hint="program_fee_schedule",
            abstain_immediately=False,
            reason="Route through current fee schedule sources.",
        )

    if _contains_any(normalized, _SCHOLARSHIP_TERMS):
        return QueryClassification(
            query_class=QueryClass.FEE_OR_ADMISSIONS_CYCLE,
            source_class_hint="scholarship",
            record_type_hint="scholarship",
            abstain_immediately=False,
            reason="Route through scholarship sources.",
        )

    if _contains_any(normalized, _MERIT_TERMS):
        return QueryClassification(
            query_class=QueryClass.FEE_OR_ADMISSIONS_CYCLE,
            source_class_hint="merit_list",
            record_type_hint="merit_list",
            abstain_immediately=False,
            reason="Route through merit list sources.",
        )

    if _contains_any(normalized, _ELIGIBILITY_TERMS):
        return QueryClassification(
            query_class=QueryClass.FEE_OR_ADMISSIONS_CYCLE,
            source_class_hint=None,
            record_type_hint=None,
            abstain_immediately=False,
            reason="Route through eligibility sources (broad).",
        )

    if _contains_any(normalized, _ADMISSIONS_TERMS):
        return QueryClassification(
            query_class=QueryClass.FEE_OR_ADMISSIONS_CYCLE,
            source_class_hint="admissions_cycle",
            record_type_hint="admissions_cycle",
            abstain_immediately=False,
            reason="Route through admissions-cycle sources.",
        )

    return QueryClassification(
        query_class=QueryClass.ENTITY_LOOKUP,
        source_class_hint=None,
        record_type_hint=None,
        abstain_immediately=False,
        reason="Use broad entity lookup routing.",
    )


def extract_record_type_hint(query_text: str, query_class: QueryClass) -> str | None:
    """Extract fine-grained record_type_hint via keyword matching for a given class.

    Used by the semantic classifier to recover the sub-type precision that
    keyword classification provides natively.
    """
    normalized = _normalize_query(query_text)
    if query_class == QueryClass.FEE_OR_ADMISSIONS_CYCLE:
        if _contains_any(normalized, _FEE_TERMS):
            return "program_fee_schedule"
        if _contains_any(normalized, _SCHOLARSHIP_TERMS):
            return "scholarship"
        if _contains_any(normalized, _MERIT_TERMS):
            return "merit_list"
        if _contains_any(normalized, _ELIGIBILITY_TERMS):
            return None
        if _contains_any(normalized, _ADMISSIONS_TERMS):
            return "admissions_cycle"
        return None
    if query_class == QueryClass.FACULTY_EXPERTISE_OR_PUBLICATION:
        if _contains_any(normalized, ("publication", "publications")):
            return "faculty_publication"
        return None
    return None


def resolve_effective_source_class_hint(
    *,
    classified_hint: str | None,
    requested_hint: str | None,
) -> str | None:
    normalized_classified = normalize_source_class_hint(classified_hint)
    normalized_requested = normalize_source_class_hint(requested_hint)

    if normalized_classified is None:
        return normalized_requested
    if normalized_requested is None:
        return normalized_classified
    if normalized_requested == normalized_classified:
        return normalized_requested
    return normalized_classified


def _normalize_query(query_text: str) -> str:
    return re.sub(r"\s+", " ", query_text.strip().lower())


@functools.lru_cache(maxsize=256)
def _word_boundary_pattern(term: str) -> re.Pattern[str]:
    """Build a regex that matches term at word boundaries.

    Terms ending with a space or punctuation (like "dr ") already carry
    an implicit boundary, so they are matched literally.
    """
    if term.endswith((" ", ".")):
        return re.compile(re.escape(term))
    return re.compile(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])")


def _contains_any(query_text: str, terms: tuple[str, ...]) -> bool:
    return any(_word_boundary_pattern(term).search(query_text) for term in terms)


_SECONDARY_INTENT_MAP: tuple[
    tuple[tuple[str, ...], str | None, str | None], ...
] = (
    (_FEE_TERMS, None, "program_fee_schedule"),
    (_SCHOLARSHIP_TERMS, "scholarship", "scholarship"),
    (_ADMISSIONS_TERMS, "admissions_cycle", "admissions_cycle"),
    (_MERIT_TERMS, "merit_list", "merit_list"),
    (_FACULTY_TERMS, "faculty", None),
    (_POLICY_TERMS, "policy", None),
    (_NEWS_EVENT_TERMS, "news_event", "news_event"),
    (_ELIGIBILITY_TERMS, None, None),
)


def extract_secondary_hints(
    query_text: str,
    primary: QueryClassification | None = None,
) -> tuple[tuple[str | None, str | None], ...]:
    """Return (source_class_hint, record_type_hint) pairs for secondary intents.

    The primary intent is handled by classify_query(). This function detects
    additional intents that the first-match chain missed.

    Accepts an optional pre-computed primary classification to avoid redundant
    classify_query() calls in the keyword path.
    """
    normalized = _normalize_query(query_text)
    if primary is None:
        primary = classify_query(query_text)
    primary_pair = (primary.source_class_hint, primary.record_type_hint)

    secondary: list[tuple[str | None, str | None]] = []
    seen: set[tuple[str | None, str | None]] = {primary_pair}
    for terms, src_hint, rec_hint in _SECONDARY_INTENT_MAP:
        if not _contains_any(normalized, terms):
            continue
        pair = (src_hint, rec_hint)
        if pair in seen:
            continue
        seen.add(pair)
        secondary.append(pair)

    return tuple(secondary)


# High-precision keyword hint terms that semantic classification must not override.
_GUARDRAIL_SOURCE_CLASSES = frozenset({"policy", "fee", "scholarship", "admissions"})
_GUARDRAIL_RECORD_TYPES = frozenset({
    "program_fee_schedule", "scholarship", "admissions_cycle",
})


def reconcile_classification(
    keyword: QueryClassification,
    semantic: QueryClassification,
) -> QueryClassification:
    """Merge keyword and semantic classifications with keyword guardrails.

    When the keyword classifier returns a high-precision hint (policy, fee,
    scholarship, admissions), it wins over the semantic result.  Otherwise
    the semantic classification is trusted.
    """
    keyword_has_strong_hint = (
        keyword.source_class_hint in _GUARDRAIL_SOURCE_CLASSES
        or keyword.record_type_hint in _GUARDRAIL_RECORD_TYPES
    )
    if keyword_has_strong_hint:
        return keyword
    return semantic
